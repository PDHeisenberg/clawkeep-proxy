"""
ClawKeep Pro Proxy
==================
Validates per-user JWTs (minted from App Store receipts), forwards
/v1/chat/completions and /v1/responses to the real AI provider.
Provider API keys live here only — never shipped in the app.

Config via environment variables (Railway injects these) or local config.env.
JWT_SECRET is required; the server refuses to start without it.
"""

import os
import time
import logging
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
JWT_EXPIRY_DAYS     = 30

# Shared TestFlight bearer — beta testers don't have purchased subscriptions,
# so they can't mint a per-user JWT via /validate-receipt. The app falls back
# to this token when isTestFlight && JWT empty. Leave unset to disable the
# bypass entirely (production hardening). Rate-limited per-IP, not per-sub,
# since every TestFlight tester sends the same sub identifier.
TESTFLIGHT_BEARER   = os.environ.get("TESTFLIGHT_BEARER", "")

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

# One verifier per environment. Each attempt is tried in order until one
# accepts the JWS, so a single endpoint works for TestFlight (Sandbox) and
# App Store (Production) without the caller having to know which it is.
# The Production verifier requires the numeric App Apple ID (assigned by
# App Store Connect once the app is listed). Until that ID is configured,
# only Sandbox is wired up — which covers TestFlight + sandbox purchases.
_VERIFIERS: list[tuple[Environment, SignedDataVerifier]] = []
if APP_APPLE_ID_INT is not None:
    _VERIFIERS.append((
        Environment.PRODUCTION,
        SignedDataVerifier(
            root_certificates=_APPLE_ROOT_CERTS,
            enable_online_checks=False,
            environment=Environment.PRODUCTION,
            bundle_id=EXPECTED_BUNDLE_ID,
            app_apple_id=APP_APPLE_ID_INT,
        ),
    ))
_VERIFIERS.append((
    Environment.SANDBOX,
    SignedDataVerifier(
        root_certificates=_APPLE_ROOT_CERTS,
        enable_online_checks=False,
        environment=Environment.SANDBOX,
        bundle_id=EXPECTED_BUNDLE_ID,
        app_apple_id=None,
    ),
))

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
    """Rate-limit key: JWT `sub` when the token is valid, else client IP.
    TestFlight shared-token requests always rate-limit by IP since every tester
    sends the same token (no per-user identity)."""
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token:
        if TESTFLIGHT_BEARER and token == TESTFLIGHT_BEARER:
            return f"ip:{get_remote_address(request)}"
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
    """Require a valid per-user JWT signed with JWT_SECRET.

    Also accepts the shared TESTFLIGHT_BEARER token (when configured) so beta
    testers without a purchased subscription can still use Pro features.
    The TestFlight path doesn't carry per-user identity — it's identified in
    logs and rate-limit keys as `testflight-shared`.
    """
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if TESTFLIGHT_BEARER and token == TESTFLIGHT_BEARER:
        request.state.sub = "testflight-shared"
        return

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Unauthorized: token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Unauthorized")

    request.state.sub = payload.get("sub", "-")


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
    """Enforce that the verified transaction is for one of our products and
    hasn't expired past the grace window."""
    product_id = getattr(payload, "productId", "") or ""
    if product_id not in VALID_PRODUCT_IDS:
        log.error("validate-receipt: unknown productId=%s", product_id)
        raise HTTPException(status_code=402, detail="Unsupported product")

    expires_ms = getattr(payload, "expiresDate", 0) or 0
    now_ms = int(time.time() * 1000)
    grace_ms = 7 * 24 * 60 * 60 * 1000
    if expires_ms and (expires_ms + grace_ms) < now_ms:
        log.error(
            "validate-receipt: expired expires_ms=%s now_ms=%s productId=%s",
            expires_ms, now_ms, product_id,
        )
        raise HTTPException(status_code=402, detail="Subscription expired")


def _mint_jwt(transaction_id: str, product_id: str, sub_expires_ms: int) -> tuple[str, int]:
    now = int(time.time())
    sub_expires_sec = int(sub_expires_ms / 1000)
    default_exp = now + JWT_EXPIRY_DAYS * 24 * 60 * 60
    exp = min(sub_expires_sec, default_exp) if sub_expires_sec > now else default_exp

    tier = "yearly" if "yearly" in product_id else "monthly"
    payload = {
        "sub": transaction_id,
        "tier": tier,
        "exp": exp,
        "iat": now,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, exp


@app.post("/validate-receipt")
@limiter.limit("10/minute", key_func=get_remote_address)
async def validate_receipt(request: Request):
    """iOS POSTs a StoreKit 2 signed transaction JWS, gets a per-user JWT back.

    The JWS is produced by iOS via `Transaction.jwsRepresentation` — the
    transaction payload signed by Apple with a cert chain rooted at
    AppleRootCA-G3. We verify the signature offline and mint our JWT.
    """
    body = await request.json()
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

    token, exp = _mint_jwt(transaction_id, product_id, expires_ms)
    tier = "yearly" if "yearly" in product_id else "monthly"
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    return JSONResponse({
        "token": token,
        "expires_at": expires_at,
        "tier": tier,
    })

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
@limiter.limit("60/minute", key_func=_jwt_sub_or_ip)
async def chat_completions(request: Request):
    check_auth(request)

    body = await request.json()
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

    body = await request.json()
    stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }

    if stream:
        return await _stream_response(OPENAI_RESPONSES_URL, headers, body, "openai", request)

    # Responses API calls can be long (web searches take time)
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(OPENAI_RESPONSES_URL, headers=headers, json=body)
        if r.status_code != 200:
            _log_upstream_error(request, "responses", r.status_code, r.text)
            return JSONResponse({"error": "upstream_error"}, status_code=502)
        return JSONResponse(r.json())


# ─── Upstream error logging ──────────────────────────────────────────────────

def _log_upstream_error(request: Request, provider: str, status: int, text: str):
    sub = getattr(request.state, "sub", "-") if request is not None else "-"
    log.error("upstream_error provider=%s status=%s sub=%s body=%s", provider, status, sub, text[:500])


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

    async with httpx.AsyncClient(timeout=90) as client:
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

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            _log_upstream_error(request, AI_PROVIDER, r.status_code, r.text)
            return JSONResponse({"error": "upstream_error"}, status_code=502)
        return JSONResponse(r.json())


# ─── Streaming ────────────────────────────────────────────────────────────────

async def _stream_response(url: str, headers: dict, body: dict, provider: str, request: Request):
    async def generate():
        async with httpx.AsyncClient(timeout=90) as client:
            async with client.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code >= 400:
                    err_body = await r.aread()
                    _log_upstream_error(request, provider, r.status_code, err_body.decode("utf-8", errors="replace"))
                    yield b'data: {"error":"upstream_error"}\n\n'
                    return
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not AI_API_KEY:
        log.warning("⚠️  AI_API_KEY not set in config.env — requests will fail")
    log.info("Starting ClawKeep Pro Proxy on port %s (provider: %s)", PORT, AI_PROVIDER)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
