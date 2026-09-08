"""
ClawKeep Pro Proxy
==================
Validates per-user JWTs (minted from App Store receipts), forwards
/v1/chat/completions and /v1/responses to the real AI provider.
Provider API keys live here only — never shipped in the app.

Also exposes a job-based research API (/research, /research/{id}) so iOS
can fire a long-running request and poll for the result instead of
holding an open connection for 1-3 minutes (which iOS network stack
struggles with). The job worker runs the actual provider call in the
background; iOS just polls a tiny status endpoint.

Config via environment variables (Railway injects these) or local config.env.
JWT_SECRET is required; the server refuses to start without it.
"""

import os
import time
import json
import uuid
import asyncio
import logging
import hashlib
import hmac
import socket
from urllib.parse import urlsplit
from security import subscription_deadline, token_deadline, validate_upstream, validate_resolved_addresses, sanitize_upstream_headers
from job_store import JobStore, JobConflict, JobCapacityError
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import jwt
import uvicorn
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier, VerificationException
from appstoreserverlibrary.models.Environment import Environment

# ─── Config ──────────────────────────────────────────────────────────────────

# Load local config.env if present (for NAS / local dev)
CONFIG_PATH = Path(__file__).parent / "config.env"
if CONFIG_PATH.exists():
    for line in CONFIG_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

AI_PROVIDER    = os.environ.get("AI_PROVIDER", "anthropic")   # anthropic | openai | gemini
AI_API_KEY     = os.environ.get("AI_API_KEY", "")
DEFAULT_MODEL  = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")
PORT           = int(os.environ.get("PORT", "8080"))

# ── Pro subscription auth (StoreKit 2 JWS → JWT) ─────────────────────────────
# APPLE_SHARED_SECRET is retained for legacy reference only — the current
# validation path uses StoreKit 2 JWS + offline signature verification and no
# longer talks to Apple's deprecated verifyReceipt endpoint.
JWT_SECRET          = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM       = "HS256"

if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET environment variable is required. "
        "Generate one with: openssl rand -hex 32"
    )

VALID_PRODUCT_IDS = {
    "com.clawkeep.pro.monthly",
    "com.clawkeep.pro.yearly",
}
EXPECTED_BUNDLE_ID = "Personal.ClawKeep"

# App Apple ID: numeric ID assigned in App Store Connect once the app is listed.
# Required by Apple's library when verifying Production JWS. Leave unset for
# TestFlight-only pre-launch testing; the Sandbox verifier doesn't need it.
APP_APPLE_ID = os.environ.get("APP_APPLE_ID", "")
APP_APPLE_ID_INT = int(APP_APPLE_ID) if APP_APPLE_ID.isdigit() else None

# Apple's Root CA certificate (G3). Shipped in the repo so verification is
# fully offline — no dependency on Apple's verifyReceipt endpoint, which
# deprecated our previous flow and returned 21002 for modern StoreKit 2
# receipts. Cert fetched from https://www.apple.com/certificateauthority/
_APPLE_ROOT_CERT_PATH = Path(__file__).parent / "AppleRootCA-G3.cer"
try:
    _APPLE_ROOT_CERTS = [_APPLE_ROOT_CERT_PATH.read_bytes()]
except FileNotFoundError:
    raise RuntimeError(
        f"Missing Apple root certificate at {_APPLE_ROOT_CERT_PATH}. "
        "Run: curl -sSLo AppleRootCA-G3.cer https://www.apple.com/certificateauthority/AppleRootCA-G3.cer"
    )

# Production and TestFlight credentials are deliberately separate.
APPLE_ENVIRONMENT = os.environ.get("APPLE_ENVIRONMENT", "production").lower()
if APPLE_ENVIRONMENT not in ("production", "sandbox"):
    raise RuntimeError("APPLE_ENVIRONMENT must be production or sandbox")
if APPLE_ENVIRONMENT == "production" and APP_APPLE_ID_INT is None:
    raise RuntimeError("APP_APPLE_ID is required for production subscription verification")
_verification_environment = Environment.PRODUCTION if APPLE_ENVIRONMENT == "production" else Environment.SANDBOX
_VERIFIERS = [(_verification_environment, SignedDataVerifier(
    root_certificates=_APPLE_ROOT_CERTS, enable_online_checks=False,
    environment=_verification_environment, bundle_id=EXPECTED_BUNDLE_ID,
    app_apple_id=APP_APPLE_ID_INT if APPLE_ENVIRONMENT == "production" else None,
))]

# Provider upstream URLs
PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai":    "https://api.openai.com/v1/chat/completions",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}

# OpenAI Responses API — used for web_search_preview tool (deep research with web grounding)
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clawproxy")

app = FastAPI(title="ClawKeep Pro Proxy", version="1.0.0")

# ─── Rate limiting ────────────────────────────────────────────────────────────

def _jwt_sub_or_ip(request: Request) -> str:
    """Rate-limit key: JWT `sub` when the token is valid, else client IP."""
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            sub = payload.get("sub")
            if sub:
                return f"sub:{sub}"
        except jwt.PyJWTError:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    log.warning(
        "rate_limited path=%s key=%s limit=%s",
        request.url.path,
        _jwt_sub_or_ip(request) if request.url.path != "/validate-receipt" else f"ip:{get_remote_address(request)}",
        getattr(exc, "detail", ""),
    )
    return JSONResponse(
        {"error": "rate_limited"},
        status_code=429,
        headers={"Retry-After": "60"},
    )


# ─── Auth ─────────────────────────────────────────────────────────────────────

def check_auth(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        payload = jwt.decode(auth[7:].strip(), JWT_SECRET, algorithms=[JWT_ALGORITHM],
                             options={"require": ["exp", "sub", "jti"]})
        if not payload["sub"] or job_store.is_revoked(payload["jti"]):
            raise HTTPException(status_code=401, detail="Subscription access revoked")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Session expired. Refresh your subscription access.")
    request.state.sub = payload["sub"]


# ─── Apple StoreKit 2 JWS verification + JWT minting ─────────────────────────

def _verify_transaction_jws(jws: str):
    """Verify a StoreKit 2 signed transaction JWS and return the decoded payload.

    Tries every configured verifier (Production first if available, then
    Sandbox). iOS hands us the JWS straight from
    `Transaction.currentEntitlements` — Apple signs it with a cert chain
    rooted at AppleRootCA-G3, which the verifier validates offline.
    """
    last_error: VerificationException | None = None
    for env, verifier in _VERIFIERS:
        try:
            payload = verifier.verify_and_decode_signed_transaction(jws)
            log.info(
                "validate-receipt: JWS verified env=%s productId=%s txId=%s",
                env.value,
                getattr(payload, "productId", "?"),
                getattr(payload, "transactionId", "?"),
            )
            return payload
        except VerificationException as e:
            last_error = e
            log.debug("validate-receipt: env=%s rejected: %s", env.value, e)

    log.error("validate-receipt: JWS verification failed against all envs: %s", last_error)
    raise HTTPException(status_code=400, detail="Invalid subscription token")


def _check_active_subscription(payload) -> None:
    try:
        subscription_deadline(payload, int(time.time()))
    except ValueError as exc:
        raise HTTPException(status_code=402, detail=str(exc))
    transaction_id = str(getattr(payload, "transactionId", "") or "")
    if not transaction_id or job_store.is_revoked(transaction_id):
        raise HTTPException(status_code=402, detail="Subscription transaction is missing or revoked")


def _mint_jwt(transaction_id: str, product_id: str, sub_expires_ms: int, signed_transaction_id: str) -> tuple[str, int]:
    now = int(time.time())
    exp = token_deadline(sub_expires_ms, now)
    payload = {"sub": transaction_id, "jti": signed_transaction_id,
               "tier": "yearly" if "yearly" in product_id else "monthly", "exp": exp, "iat": now}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM), exp


@app.post("/validate-receipt")
@limiter.limit("10/minute", key_func=get_remote_address)
async def validate_receipt(request: Request):
    """iOS POSTs a StoreKit 2 signed transaction JWS, gets a per-user JWT back.

    The JWS is produced by iOS via `Transaction.jwsRepresentation` — the
    transaction payload signed by Apple with a cert chain rooted at
    AppleRootCA-G3. We verify the signature offline and mint our JWT.
    """
    body = await read_json_object(request)
    jws = body.get("transaction_jws")
    if not jws or not isinstance(jws, str):
        raise HTTPException(status_code=400, detail="Missing transaction_jws")

    payload = _verify_transaction_jws(jws)
    _check_active_subscription(payload)

    # originalTransactionId stays stable across renewals; transactionId changes
    # every billing period. We scope the JWT's sub to originalTransactionId so
    # rate-limits and usage tracking don't reset on renewal.
    transaction_id = str(getattr(payload, "originalTransactionId", "") or "")
    product_id = str(getattr(payload, "productId", "") or "")
    expires_ms = int(getattr(payload, "expiresDate", 0) or 0)

    if not transaction_id:
        raise HTTPException(status_code=400, detail="Missing transaction id in signed payload")

    token, exp = _mint_jwt(transaction_id, product_id, expires_ms, str(payload.transactionId))
    tier = "yearly" if "yearly" in product_id else "monthly"
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    return JSONResponse({
        "token": token,
        "expires_at": expires_at,
        "tier": tier,
    })

# Mount data/ on a persistent volume in production. One serving process owns
# in-flight network workers; durable metadata also survives process restarts.
job_store = JobStore(os.environ.get("JOB_DATABASE_PATH", "data/jobs.sqlite3"))

async def _periodic_job_cleanup():
    """Background task — sweeps stale jobs every 5 minutes."""
    while True:
        try:
            await asyncio.sleep(300)
            removed = await job_store.cleanup_old()
            if removed:
                log.info("job_cleanup: removed %d stale job(s)", removed)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("job_cleanup error: %s", e)


@app.on_event("startup")
async def _startup_jobs():
    await job_store.recover_interrupted()
    app.state.job_cleanup_task = asyncio.create_task(_periodic_job_cleanup())


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Research Job API ─────────────────────────────────────────────────────────
#
# iOS POSTs /research to start a long-running research call, gets back a
# job_id immediately. The actual upstream provider call (gpt-5.4 +
# web_search_preview, typically 30-180s) runs in a background asyncio task.
# iOS polls /research/{id} every few seconds — each poll is tiny and fast,
# which works perfectly with iOS background URLSession (the long-held-
# connection pattern doesn't, hence this redesign). Cancellation via DELETE
# stops the upstream call and stops the bill (best-effort).

@app.post("/research")
@limiter.limit("30/minute", key_func=_jwt_sub_or_ip)
async def start_research(request: Request):
    check_auth(request)
    if AI_PROVIDER != "openai":
        raise HTTPException(status_code=400, detail="Research is not configured for this provider")
    body = validate_pro_body(await read_json_object(request), responses_api=True)
    job_id, created = await admit_job(request, body, mode="pro")
    if created:
        task = asyncio.create_task(_run_research_job_generic(
            job_id=job_id, body=body, upstream_url=OPENAI_RESPONSES_URL,
            upstream_headers={"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"},
            provider="openai"))
        job_store.register_worker(job_id, task)
    return await start_response(job_id)


@app.post("/research/passthrough")
@limiter.limit("30/minute", key_func=get_remote_address)
async def start_research_passthrough(request: Request):
    envelope = await read_json_object(request)
    upstream_url = envelope.get("upstream_url")
    body = envelope.get("request_body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request_body must be an object")
    try:
        host = validate_upstream(upstream_url, os.environ.get("CUSTOM_UPSTREAM_HOSTS", "").split(","))
        addresses = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        validate_resolved_addresses({entry[4][0] for entry in addresses})
        headers = sanitize_upstream_headers(envelope.get("upstream_headers") or {})
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Provider destination is unavailable or not permitted") from exc
    # No provider credentials or full request bodies are stored in JobStore.
    # Idempotency is scoped to a client-generated installation ID, hashed here.
    client_id = request.headers.get("X-ClawKeep-Client-ID", "")
    if not client_id or len(client_id) > 128:
        raise HTTPException(status_code=400, detail="Update ClawKeep to use hosted research")
    request.state.sub = "byok:" + hashlib.sha256(client_id.encode()).hexdigest()
    job_id, created = await admit_job(request, {"url": upstream_url, "body": body}, mode="passthrough")
    if created:
        task = asyncio.create_task(_run_research_job_generic(
            job_id=job_id, body=body, upstream_url=upstream_url,
            upstream_headers=headers, provider=_detect_provider(upstream_url)))
        job_store.register_worker(job_id, task)
    return await start_response(job_id, include_capability=True)


def _detect_provider(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host == "api.openai.com":
        return "openai"
    if host == "api.anthropic.com":
        return "anthropic"
    if host == "generativelanguage.googleapis.com":
        return "gemini"
    return "custom"


@app.get("/research/{job_id}")
@limiter.limit("120/minute", key_func=get_remote_address)
async def get_research(request: Request, job_id: str):
    await authorize_job(request, job_id)
    job = await job_store.get_by_id_only(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


@app.get("/research")
@limiter.limit("60/minute", key_func=_jwt_sub_or_ip)
async def list_research(request: Request):
    """Return all active jobs for the user (pending/running) plus any
    that completed within the last minute. Lets iOS poll N jobs in a
    single request instead of N separate requests."""
    check_auth(request)
    sub = request.state.sub
    jobs = await job_store.list_active(sub)
    return JSONResponse({"jobs": jobs})


@app.delete("/research/{job_id}")
@limiter.limit("30/minute", key_func=get_remote_address)
async def cancel_research(request: Request, job_id: str):
    await authorize_job(request, job_id)
    await job_store.cancel_by_id_only(job_id)
    # Idempotent cancellation also succeeds for already-finished work.
    return JSONResponse({"id": job_id, "status": "cancelled"})


async def _run_research_job_generic(
    job_id: str,
    body: dict,
    upstream_url: str,
    upstream_headers: dict,
    provider: str,
):
    """Provider-aware research worker. OpenAI gets streaming + per-event
    progress capture so iOS can show "Searching: <query>" in the UI.
    Anthropic/Gemini/custom get a plain non-streaming POST — same job-queue
    polling reliability, just no live activity readout (each provider
    streams differently; we wire those parsers in a follow-up).

    `upstream_headers` is held only in this function's local scope. When
    the task exits (success/cancel/error), Python's GC drops the reference
    and the user's API key (if it was in there) is gone.

    Runs as a background asyncio task. Cancellation propagates from
    job_store.cancel() via task.cancel()."""
    try:
        await job_store.update(
            job_id,
            status="running",
            progress={"phase": "starting"},
        )

        if provider == "openai" and urlsplit(upstream_url).path.endswith("/responses"):
            await _run_openai_streaming(job_id, body, upstream_url, upstream_headers)
        else:
            await _run_simple_call(job_id, body, upstream_url, upstream_headers, provider)

    except asyncio.CancelledError:
        await job_store.update(job_id, status="cancelled", error="Cancelled by client")
        log.info("research_job_cancelled job=%s", job_id)
        raise
    except Exception as e:
        log.error("research_job_error job=%s provider=%s type=%s", job_id, provider, type(e).__name__)
        await job_store.update(job_id, status="failed", error="The provider connection failed. Please retry.")
    finally:
        job_store.unregister_worker(job_id)


def _extract_upstream_error(status: int, body_text: str) -> str:
    """Pull a friendly inner message out of an upstream error body. Falls
    back to a generic HTTP status string when the body isn't JSON or
    doesn't have the expected shape."""
    msg = f"Upstream error: HTTP {status}"
    try:
        err_json = json.loads(body_text)
        if isinstance(err_json, dict):
            err_obj = err_json.get("error")
            if isinstance(err_obj, dict):
                inner = err_obj.get("message")
                if inner:
                    return f"{msg} — {inner[:150]}"
            elif isinstance(err_obj, str):
                return f"{msg} — {err_obj[:150]}"
    except Exception:
        pass
    return msg


async def _run_openai_streaming(job_id: str, body: dict, url: str, headers: dict):
    """Streaming variant — only OpenAI Responses API today. Captures
    web_search and text-generation events, surfaces progress through
    job_store.update so the iOS skeleton can narrate the call."""
    body["stream"] = True

    full_response: dict | None = None
    searches_total = 0
    searches_done = 0
    current_query: str | None = None
    text_started = False

    async with httpx.AsyncClient(follow_redirects=False, timeout=300) as client:
        async with client.stream("POST", url, headers=headers, json=body) as r:
            if r.status_code != 200:
                err_body = (await r.aread()).decode("utf-8", errors="replace")[:500]
                log.error("research_job_upstream_error job=%s status=%s", job_id, r.status_code)
                await job_store.update(
                    job_id,
                    status="failed",
                    error=_extract_upstream_error(r.status_code, err_body),
                )
                return

            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "response.output_item.added":
                    item = event.get("item") or {}
                    if item.get("type") == "web_search_call":
                        searches_total += 1
                        action = item.get("action") or {}
                        q = action.get("query")
                        if not q and isinstance(action.get("queries"), list):
                            q = action["queries"][0] if action["queries"] else None
                        current_query = q
                        await job_store.update(job_id, progress={
                            "phase": "searching",
                            "current_query": current_query,
                            "searches_done": searches_done,
                            "searches_total": searches_total,
                        })

                elif event_type == "response.web_search_call.completed":
                    searches_done += 1
                    current_query = None
                    await job_store.update(job_id, progress={
                        "phase": "searching",
                        "current_query": None,
                        "searches_done": searches_done,
                        "searches_total": searches_total,
                    })

                elif event_type == "response.output_text.delta":
                    # First text delta means search phase is done and the
                    # model is now writing the report. Don't update on every
                    # delta — those fire dozens of times per second.
                    if not text_started:
                        text_started = True
                        await job_store.update(job_id, progress={
                            "phase": "writing",
                            "searches_done": searches_done,
                            "searches_total": searches_total,
                        })

                elif event_type == "response.completed":
                    full_response = event.get("response")
                    # Don't break — let the stream end naturally so any
                    # final events are processed. Loop will exit on its own.

                elif event_type == "response.incomplete":
                    # Output truncated (typically max_output_tokens, sometimes
                    # content_filter or stop). The response object IS in the
                    # event payload — just with status="incomplete" and an
                    # incomplete_details field. Treat as success: a partial
                    # report is way better than failing the whole job, which
                    # was happening for any research call that did 20+ web
                    # searches because the accumulating context overran the
                    # output budget.
                    full_response = event.get("response")
                    incomplete_reason = (
                        (full_response or {}).get("incomplete_details") or {}
                    ).get("reason", "unknown")
                    log.warning(
                        "research_job_incomplete job=%s reason=%s searches=%d/%d",
                        job_id, incomplete_reason, searches_done, searches_total,
                    )
                    # Don't break — let the stream end naturally.

                elif event_type == "response.failed":
                    err_obj = (event.get("response") or {}).get("error") or {}
                    err_msg = err_obj.get("message") or "Stream reported failure"
                    await job_store.update(job_id, status="failed", error=err_msg[:200])
                    return

                elif event_type == "response.cancelled":
                    # Server-side cancellation (rare — usually when a user
                    # cancels via the OpenAI dashboard or the request got
                    # aborted upstream). Treat as cancelled rather than
                    # failed so iOS doesn't show a "Tap retry" pill.
                    await job_store.update(
                        job_id, status="cancelled",
                        error="Cancelled upstream",
                    )
                    return

                elif event_type == "error":
                    # Top-level stream error (not wrapped in a response object).
                    # Surfaces things like rate-limit errors that fire mid-stream.
                    err_msg = event.get("message") or "Stream error"
                    log.error("research_job_stream_error job=%s msg=%s", job_id, err_msg)
                    await job_store.update(job_id, status="failed", error=str(err_msg)[:200])
                    return

    if full_response is None:
        await job_store.update(
            job_id,
            status="failed",
            error="Stream ended without response.completed event",
        )
        log.error(
            "research_job_no_completion job=%s searches_done=%s searches_total=%s",
            job_id, searches_done, searches_total,
        )
        return

    await job_store.update(
        job_id,
        status="completed",
        response=full_response,
        progress={
            "phase": "done",
            "searches_done": searches_done,
            "searches_total": searches_total,
        },
    )
    log.info(
        "research_job_complete job=%s searches=%d/%d",
        job_id, searches_done, searches_total,
    )


async def _run_simple_call(job_id: str, body: dict, url: str, headers: dict, provider: str):
    """Non-streaming variant — Anthropic/Gemini/custom. The caller still
    polls and gets the same reliability properties, just no per-event
    progress events because each provider's streaming format is different
    enough to need its own parser. iOS surfaces a static "running" state
    until the call returns."""
    # Strip stream:true if the caller set it — we want the full response in one shot.
    body.pop("stream", None)

    async with httpx.AsyncClient(follow_redirects=False, timeout=300) as client:
        r = await client.post(url, headers=headers, json=body)

    if r.status_code != 200:
        err_body = r.text[:500]
        log.error("research_job_upstream_error job=%s provider=%s status=%s", job_id, provider, r.status_code)
        await job_store.update(
            job_id,
            status="failed",
            error=_extract_upstream_error(r.status_code, err_body),
        )
        return

    try:
        result = r.json()
    except Exception:
        await job_store.update(job_id, status="failed", error="Upstream returned non-JSON response")
        return

    await job_store.update(
        job_id,
        status="completed",
        response=result,
        progress={"phase": "done"},
    )
    log.info("research_job_complete job=%s provider=%s", job_id, provider)


@app.post("/v1/chat/completions")
@limiter.limit("60/minute", key_func=_jwt_sub_or_ip)
async def chat_completions(request: Request):
    check_auth(request)

    body = validate_pro_body(await read_json_object(request), responses_api=request.url.path.endswith("/responses"))
    stream = body.get("stream", False)
    model = body.get("model", DEFAULT_MODEL)

    if AI_PROVIDER == "anthropic":
        return await _forward_anthropic(body, model, stream, request)
    else:
        return await _forward_openai_compat(body, model, stream, request)


@app.post("/v1/responses")
@limiter.limit("20/minute", key_func=_jwt_sub_or_ip)
async def responses(request: Request):
    """
    Passthrough for OpenAI Responses API (required for web_search_preview tool).
    Only works with AI_PROVIDER=openai. Supports deep research with live web grounding.
    """
    check_auth(request)

    if AI_PROVIDER != "openai":
        raise HTTPException(
            status_code=400,
            detail=f"/v1/responses requires AI_PROVIDER=openai (current: {AI_PROVIDER})"
        )

    body = validate_pro_body(await read_json_object(request), responses_api=request.url.path.endswith("/responses"))
    stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }

    if stream:
        return await _stream_response(OPENAI_RESPONSES_URL, headers, body, "openai", request)

    # Responses API calls can be long (web searches take time)
    async with httpx.AsyncClient(follow_redirects=False, timeout=180) as client:
        r = await client.post(OPENAI_RESPONSES_URL, headers=headers, json=body)
        if r.status_code != 200:
            _log_upstream_error(request, "responses", r.status_code, r.text)
            return JSONResponse({"error": "upstream_error"}, status_code=502)
        return JSONResponse(r.json())


# ─── Upstream error logging ──────────────────────────────────────────────────

def _log_upstream_error(request: Request, provider: str, status: int, text: str):
    sub = getattr(request.state, "sub", "-") if request is not None else "-"
    log.error("upstream_error provider=%s status=%s sub=%s", provider, status, sub)


# ─── Anthropic forwarding ─────────────────────────────────────────────────────

async def _forward_anthropic(body: dict, model: str, stream: bool, request: Request):
    """Convert OpenAI-style request → Anthropic Messages API."""
    messages = body.get("messages", [])
    system_msgs = [m["content"] for m in messages if m["role"] == "system"]
    user_msgs   = [m for m in messages if m["role"] != "system"]

    anthropic_body = {
        "model": model,
        "max_tokens": body.get("max_tokens", 4096),
        "messages": user_msgs,
        "stream": stream,
    }
    if system_msgs:
        anthropic_body["system"] = "\n\n".join(system_msgs)

    headers = {
        "x-api-key": AI_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    if stream:
        return await _stream_response(PROVIDER_URLS["anthropic"], headers, anthropic_body, "anthropic", request)

    async with httpx.AsyncClient(follow_redirects=False, timeout=90) as client:
        r = await client.post(PROVIDER_URLS["anthropic"], headers=headers, json=anthropic_body)
        if r.status_code != 200:
            _log_upstream_error(request, "anthropic", r.status_code, r.text)
            return JSONResponse({"error": "upstream_error"}, status_code=502)

        data = r.json()
        # Convert Anthropic response → OpenAI-compatible shape
        content = data.get("content", [{}])[0].get("text", "")
        return JSONResponse({
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": data.get("model", model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": data.get("stop_reason", "stop"),
            }],
            "usage": data.get("usage", {}),
        })


# ─── OpenAI-compatible forwarding (OpenAI / Gemini) ──────────────────────────

async def _forward_openai_compat(body: dict, model: str, stream: bool, request: Request):
    url = PROVIDER_URLS.get(AI_PROVIDER, PROVIDER_URLS["openai"])
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }

    if stream:
        return await _stream_response(url, headers, body, "openai", request)

    async with httpx.AsyncClient(follow_redirects=False, timeout=90) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            _log_upstream_error(request, AI_PROVIDER, r.status_code, r.text)
            return JSONResponse({"error": "upstream_error"}, status_code=502)
        return JSONResponse(r.json())


# ─── Streaming ────────────────────────────────────────────────────────────────

async def _stream_response(url: str, headers: dict, body: dict, provider: str, request: Request):
    async def generate():
        async with httpx.AsyncClient(follow_redirects=False, timeout=90) as client:
            async with client.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code >= 400:
                    err_body = await r.aread()
                    _log_upstream_error(request, provider, r.status_code, err_body.decode("utf-8", errors="replace"))
                    yield b'data: {"error":"upstream_error"}\n\n'
                    return
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─── Validated boundaries ──────────────────────────────────────────────────

async def read_json_object(request: Request, maximum_bytes: int = 2 * 1024 * 1024):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > maximum_bytes:
            raise HTTPException(status_code=413, detail="Request is too large")
        body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Request must be an object")
    return value


def validate_pro_body(body: dict, *, responses_api: bool):
    allowed = {DEFAULT_MODEL, "gpt-5.4-mini", "gpt-5.4-nano"}
    configured = os.environ.get("ALLOWED_PRO_MODELS")
    if configured:
        allowed = {m.strip() for m in configured.split(",") if m.strip()}
    model = body.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or model not in allowed:
        raise HTTPException(status_code=400, detail="This model is not available with ClawKeep Pro")
    body["model"] = model
    token_key = "max_output_tokens" if responses_api else ("max_completion_tokens" if model.startswith("gpt-5") else "max_tokens")
    limits = []
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = body.pop(key, None)
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise HTTPException(status_code=400, detail="Invalid output limit")
            limits.append(value)
    body[token_key] = min([16384] + limits) if limits else (8192 if responses_api else 4096)
    if responses_api:
        calls = body.get("max_tool_calls", 8)
        if not isinstance(calls, int) or isinstance(calls, bool) or calls < 1:
            raise HTTPException(status_code=400, detail="Invalid tool limit")
        body["max_tool_calls"] = min(calls, 12)
        requested_tools = body.get("tools", [])
        if not isinstance(requested_tools, list) or len(requested_tools) > 4:
            raise HTTPException(status_code=400, detail="Invalid research tools")
        for tool in requested_tools:
            if not isinstance(tool, dict) or tool.get("type") not in ("web_search", "web_search_preview"):
                raise HTTPException(status_code=400, detail="Unsupported research tool")
    return body


async def admit_job(request, body, mode):
    key = request.headers.get("Idempotency-Key")
    if key is not None and (not key.strip() or len(key) > 128):
        raise HTTPException(status_code=400, detail="Invalid request key")
    try:
        return await job_store.create(str(uuid.uuid4()), request.state.sub, body,
                                      mode=mode, idempotency_key=key)
    except JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except JobCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "15"})


def job_capability(job_id):
    return hmac.new(JWT_SECRET.encode(), ("research:" + job_id).encode(), hashlib.sha256).hexdigest()


async def start_response(job_id, include_capability=False):
    job = await job_store.get_by_id_only(job_id)
    response = {"job_id": job_id, "status": job["status"]}
    if include_capability:
        response["job_token"] = job_capability(job_id)
    return JSONResponse(response)


async def authorize_job(request, job_id):
    identity = await job_store.identity(job_id)
    if identity is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if identity["mode"] == "pro":
        check_auth(request)
        authorized = request.state.sub == identity["sub"]
    else:
        authorized = hmac.compare_digest(request.headers.get("X-Research-Token", ""), job_capability(job_id))
    if not authorized:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/app-store/notifications")
@limiter.limit("60/minute", key_func=get_remote_address)
async def app_store_notification(request: Request):
    body = await read_json_object(request)
    signed = body.get("signedPayload")
    if not isinstance(signed, str):
        raise HTTPException(status_code=400, detail="Missing signed notification")
    try:
        verifier = _VERIFIERS[0][1]
        notification = verifier.verify_and_decode_notification(signed)
        transaction = verifier.verify_and_decode_signed_transaction(notification.data.signedTransactionInfo)
    except (VerificationException, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid signed notification")
    transaction_id = str(transaction.transactionId or "")
    if not transaction_id:
        raise HTTPException(status_code=400, detail="Missing transaction")
    kind = getattr(notification.notificationType, "value", notification.notificationType)
    if kind in ("REFUND", "REVOKE", "REFUND_REVERSED"):
        expiry = max(time.time() + 900, (transaction.expiresDate or 0) / 1000)
        job_store.set_revoked(transaction_id, expiry, revoked=kind != "REFUND_REVERSED")
    return JSONResponse({"status": "ok"})


@app.on_event("shutdown")
async def shutdown_jobs():
    task = getattr(app.state, "job_cleanup_task", None)
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    await job_store.close()


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not AI_API_KEY:
        log.warning("⚠️  AI_API_KEY not set in config.env — requests will fail")
    log.info("Starting ClawKeep Pro Proxy on port %s (provider: %s)", PORT, AI_PROVIDER)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
