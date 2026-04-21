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

# ─── Job Store (in-memory, ephemeral) ────────────────────────────────────────
#
# Backs the polling-based research API. Jobs live for the lifetime of the
# container — Railway redeploys lose in-flight research, which is acceptable:
# a research job runs for 1-3 minutes, redeploys are minutes apart, and a
# lost job surfaces as "failed" on the iOS side and the user retries.
# Persistence (SQLite + Railway Volume) is a future upgrade if we see this
# matter in practice. Cleaned up after 1 hour to keep memory bounded.

JOB_TTL_SECONDS = 3600         # Drop completed/failed jobs after 1 hour
JOB_MAX_AGE_SECONDS = 7200     # Hard TTL even for stuck jobs

class JobStore:
    """Thread-safe in-memory store for research jobs. All methods async-safe."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_id: str, sub: str, request_body: dict) -> None:
        now = time.time()
        async with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "sub": sub,
                "status": "pending",
                "request_body": request_body,
                "response": None,
                "error": None,
                "progress": None,    # {phase, current_query, searches_done, searches_total}
                "created_at": now,
                "updated_at": now,
            }

    async def update(
        self,
        job_id: str,
        status: str | None = None,
        response: dict | None = None,
        error: str | None = None,
        progress: dict | None = None,
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job["status"] = status
            if response is not None:
                job["response"] = response
            if error is not None:
                job["error"] = error
            if progress is not None:
                job["progress"] = progress
            job["updated_at"] = time.time()

    async def get(self, job_id: str, sub: str) -> dict | None:
        """Fetch a job, scoped to the requesting sub. Returns None for
        unknown jobs OR jobs belonging to a different sub (404, not 403,
        on purpose — don't leak existence)."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["sub"] != sub:
                return None
            return _public_job_view(job)

    async def get_by_id_only(self, job_id: str) -> dict | None:
        """Fetch a job by UUID alone — no sub check. Used by the public
        polling endpoint where the UUID acts as a capability token. The
        UUID is unguessable (122 bits of randomness); its possession is
        sufficient proof that the requester is the original creator (or
        someone the creator legitimately shared it with)."""
        async with self._lock:
            job = self._jobs.get(job_id)
            return _public_job_view(job) if job is not None else None

    async def cancel_by_id_only(self, job_id: str) -> bool:
        """Cancel by UUID alone — same capability-token model as
        get_by_id_only. Returns True if cancelled, False if not found
        or already terminal."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] not in ("pending", "running"):
                return False
            job["status"] = "cancelled"
            job["updated_at"] = time.time()

        worker = self._workers.pop(job_id, None)
        if worker is not None and not worker.done():
            worker.cancel()
        return True

    async def list_active(self, sub: str) -> list[dict]:
        """All jobs for this sub that are pending/running, plus completed/
        failed ones updated within the last 60s (so iOS catches the
        completion on its next poll even if the job finished between polls)."""
        recent_cutoff = time.time() - 60
        async with self._lock:
            return [
                _public_job_view(j)
                for j in self._jobs.values()
                if j["sub"] == sub
                and (j["status"] in ("pending", "running") or j["updated_at"] > recent_cutoff)
            ]

    async def cancel(self, job_id: str, sub: str) -> bool:
        """Cancel a running job. Returns True if cancelled, False if not
        found / not yours / already completed."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["sub"] != sub:
                return False
            if job["status"] not in ("pending", "running"):
                return False
            job["status"] = "cancelled"
            job["updated_at"] = time.time()

        worker = self._workers.pop(job_id, None)
        if worker is not None and not worker.done():
            worker.cancel()
        return True

    def register_worker(self, job_id: str, task: asyncio.Task) -> None:
        self._workers[job_id] = task

    def unregister_worker(self, job_id: str) -> None:
        self._workers.pop(job_id, None)

    async def cleanup_old(self) -> int:
        """Drop jobs older than the TTL. Called periodically by the
        background sweeper task. Returns count removed."""
        now = time.time()
        removed = 0
        async with self._lock:
            to_remove: list[str] = []
            for jid, job in self._jobs.items():
                if job["status"] in ("completed", "failed", "cancelled"):
                    if now - job["updated_at"] > JOB_TTL_SECONDS:
                        to_remove.append(jid)
                elif now - job["created_at"] > JOB_MAX_AGE_SECONDS:
                    to_remove.append(jid)
            for jid in to_remove:
                self._jobs.pop(jid, None)
                removed += 1
        return removed


def _public_job_view(job: dict) -> dict:
    """Strip the request_body from external responses — it's only stored for
    debugging and could be sizable. Clients already have the body locally."""
    return {
        "id": job["id"],
        "status": job["status"],
        "response": job["response"],
        "error": job["error"],
        "progress": job.get("progress"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


job_store = JobStore()


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
    asyncio.create_task(_periodic_job_cleanup())


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
    """Pro-mode research. Proxy uses its own AI_API_KEY for the upstream call.
    Internally just a thin wrapper around the generic worker, hardcoded to
    OpenAI Responses API."""
    check_auth(request)
    sub = request.state.sub

    if AI_PROVIDER != "openai":
        raise HTTPException(
            status_code=400,
            detail=f"/research requires AI_PROVIDER=openai (current: {AI_PROVIDER})"
        )

    body = await request.json()
    job_id = str(uuid.uuid4())
    await job_store.create(job_id, sub, body)

    upstream_headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    task = asyncio.create_task(_run_research_job_generic(
        job_id=job_id,
        body=body,
        upstream_url=OPENAI_RESPONSES_URL,
        upstream_headers=upstream_headers,
        provider="openai",
    ))
    job_store.register_worker(job_id, task)

    log.info("research_start sub=%s job=%s provider=openai", sub, job_id)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.post("/research/passthrough")
@limiter.limit("30/minute", key_func=get_remote_address)
async def start_research_passthrough(request: Request):
    """Direct-API-mode research. Caller supplies their OWN provider URL +
    auth headers; proxy uses those for the upstream call instead of our
    AI_API_KEY. The user's key passes through proxy memory only — never
    written to disk, never logged, scrubbed when the worker exits.

    NO AUTH REQUIRED. This is safe by construction: the endpoint takes
    `upstream_url` and `upstream_headers` from the request body and uses
    THOSE for the upstream call. It never reads `AI_API_KEY`, so it
    cannot be exploited to grant Pro access — the path that uses our
    key is in the separate `/research` handler which still requires JWT.
    Anyone hitting this endpoint pays their own provider for their own
    tokens; our cost is just compute for relaying. Per-IP rate limit
    bounds compute exposure.

    Body shape:
      {
        "upstream_url": "https://api.openai.com/v1/responses",
        "upstream_headers": {"Authorization": "Bearer sk-...", "Content-Type": "application/json"},
        "request_body": { ...the original provider request body... }
      }

    Provider auto-detected from URL — OpenAI gets streaming + progress,
    Anthropic/Gemini/custom get reliability via the polling shape but
    no live progress events (different streaming formats; deferred)."""
    # Try to identify the caller for nicer log lines, but don't reject
    # them if they're anonymous. Pro JWTs and TestFlight bearers still
    # show up in logs as their proper sub identifier; everyone else is
    # logged as "anonymous-{ip-prefix}" so we can grep.
    sub = "anonymous"
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token:
        if TESTFLIGHT_BEARER and token == TESTFLIGHT_BEARER:
            sub = "testflight-shared"
        else:
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                sub = payload.get("sub", "anonymous")
            except jwt.PyJWTError:
                pass  # Stays "anonymous" — invalid token is fine here
    request.state.sub = sub

    body = await request.json()
    upstream_url = body.get("upstream_url")
    upstream_headers = body.get("upstream_headers") or {}
    request_body = body.get("request_body")

    if not upstream_url or not isinstance(upstream_url, str):
        raise HTTPException(status_code=400, detail="Missing upstream_url")
    if request_body is None:
        raise HTTPException(status_code=400, detail="Missing request_body")
    if not isinstance(upstream_headers, dict):
        raise HTTPException(status_code=400, detail="upstream_headers must be a dict")

    provider = _detect_provider(upstream_url)
    job_id = str(uuid.uuid4())
    # We persist request_body for diagnostics (it's the prompt payload — not
    # secret). upstream_headers stays out of JobStore entirely; it lives only
    # in the worker task's local scope and is GC'd when the task exits.
    await job_store.create(job_id, sub, request_body)

    task = asyncio.create_task(_run_research_job_generic(
        job_id=job_id,
        body=request_body,
        upstream_url=upstream_url,
        upstream_headers=upstream_headers,
        provider=provider,
    ))
    job_store.register_worker(job_id, task)

    log.info("research_passthrough_start sub=%s job=%s provider=%s", sub, job_id, provider)
    return JSONResponse({"job_id": job_id, "status": "pending"})


def _detect_provider(url: str) -> str:
    """Identify provider from upstream URL so the worker picks the right
    request handling strategy (streaming for OpenAI, plain POST otherwise)."""
    lower = url.lower()
    if "openai.com" in lower or "/v1/responses" in lower or "/v1/chat/completions" in lower:
        return "openai"
    if "anthropic.com" in lower:
        return "anthropic"
    if "googleapis.com" in lower or "generativelanguage" in lower:
        return "gemini"
    return "custom"


@app.get("/research/{job_id}")
@limiter.limit("120/minute", key_func=get_remote_address)
async def get_research(request: Request, job_id: str):
    """Poll for status. Tiny response when running, full response payload
    when completed.

    NO AUTH. The job_id is a UUID v4 — 122 bits of randomness, effectively
    unguessable. Anyone with the UUID can read the job; you only get the
    UUID by being the caller who started the job (it's in the response to
    POST /research[/passthrough]). This sidesteps the question of "who
    can poll a passthrough job created with no auth" — the answer is
    "whoever has the UUID", and the UUID acts as a one-shot capability
    token. If a UUID leaks (logs, etc.), worst case someone reads someone
    else's research result, which is not a meaningful privacy leak — these
    are saved-link summaries, not user secrets."""
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
    """Cancel a job. Same UUID-as-capability auth model as GET — no
    JWT required, the UUID is the access control. The biggest harm a
    leaked UUID could do is cancelling someone else's research; cheap
    annoyance, not a security incident."""
    cancelled = await job_store.cancel_by_id_only(job_id)
    if not cancelled:
        # 404 covers both "doesn't exist" and "already completed"; iOS
        # treats both the same way (no further polling needed).
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    log.info("research_cancel job=%s", job_id)
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

        if provider == "openai":
            await _run_openai_streaming(job_id, body, upstream_url, upstream_headers)
        else:
            await _run_simple_call(job_id, body, upstream_url, upstream_headers, provider)

    except asyncio.CancelledError:
        await job_store.update(job_id, status="cancelled", error="Cancelled by client")
        log.info("research_job_cancelled job=%s", job_id)
        raise
    except Exception as e:
        log.exception("research_job_error job=%s provider=%s", job_id, provider)
        await job_store.update(job_id, status="failed", error=str(e)[:200])
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

    async with httpx.AsyncClient(timeout=300) as client:
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

    async with httpx.AsyncClient(timeout=300) as client:
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
