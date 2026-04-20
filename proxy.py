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

# ── Pro subscription auth (receipt → JWT) ────────────────────────────────────
APPLE_SHARED_SECRET = os.environ.get("APPLE_SHARED_SECRET", "")
JWT_SECRET          = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM       = "HS256"
JWT_EXPIRY_DAYS     = 30

if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET environment variable is required. "
        "Generate one with: openssl rand -hex 32"
    )

APPLE_PROD_URL    = "https://buy.itunes.apple.com/verifyReceipt"
APPLE_SANDBOX_URL = "https://sandbox.itunes.apple.com/verifyReceipt"

VALID_PRODUCT_IDS = {
    "com.clawkeep.pro.monthly",
    "com.clawkeep.pro.yearly",
}
EXPECTED_BUNDLE_ID = "Personal.ClawKeep"

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
    """Require a valid per-user JWT signed with JWT_SECRET."""
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Unauthorized: token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Unauthorized")

    request.state.sub = payload.get("sub", "-")


# ─── Apple receipt verification + JWT minting ────────────────────────────────

async def _verify_apple_receipt(receipt_b64: str) -> dict:
    if not APPLE_SHARED_SECRET:
        raise HTTPException(status_code=500, detail="APPLE_SHARED_SECRET not configured")

    payload = {
        "receipt-data": receipt_b64,
        "password": APPLE_SHARED_SECRET,
        "exclude-old-transactions": True,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(APPLE_PROD_URL, json=payload)
        data = r.json()
        # 21007 → this is a sandbox receipt, retry against sandbox URL
        if data.get("status") == 21007:
            r = await client.post(APPLE_SANDBOX_URL, json=payload)
            data = r.json()

    status = data.get("status")
    if status != 0:
        raise HTTPException(status_code=400, detail=f"Apple receipt status {status}")

    bundle_id = data.get("receipt", {}).get("bundle_id", "")
    if bundle_id != EXPECTED_BUNDLE_ID:
        raise HTTPException(status_code=400, detail="Bundle ID mismatch")

    return data


def _latest_active_subscription(receipt_data: dict) -> dict:
    entries = receipt_data.get("latest_receipt_info") or []
    if not entries:
        entries = receipt_data.get("receipt", {}).get("in_app", [])

    subs = [e for e in entries if e.get("product_id") in VALID_PRODUCT_IDS]
    if not subs:
        raise HTTPException(status_code=402, detail="No ClawKeep Pro subscription found")

    subs.sort(key=lambda e: int(e.get("expires_date_ms", 0)), reverse=True)
    latest = subs[0]

    expires_ms = int(latest.get("expires_date_ms", 0))
    now_ms = int(time.time() * 1000)
    grace_ms = 7 * 24 * 60 * 60 * 1000
    if expires_ms + grace_ms < now_ms:
        raise HTTPException(status_code=402, detail="Subscription expired")

    return latest


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
    """iOS POSTs App Store receipt, gets a per-user JWT in return."""
    body = await request.json()
    receipt_b64 = body.get("receipt_data")
    if not receipt_b64:
        raise HTTPException(status_code=400, detail="Missing receipt_data")

    apple_data = await _verify_apple_receipt(receipt_b64)
    sub = _latest_active_subscription(apple_data)

    transaction_id = sub.get("original_transaction_id", "")
    product_id = sub.get("product_id", "")
    expires_ms = int(sub.get("expires_date_ms", 0))

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
