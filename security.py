"""Pure validation rules shared by request handlers and offline tests."""
import ipaddress
from urllib.parse import urlsplit

VALID_PRODUCT_IDS = {"com.clawkeep.pro.monthly", "com.clawkeep.pro.yearly"}
PROVIDER_HOSTS = {"api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com"}


def subscription_deadline(payload, now: int) -> int:
    if getattr(payload, "productId", None) not in VALID_PRODUCT_IDS:
        raise ValueError("Unsupported subscription product")
    if getattr(payload, "revocationDate", None) is not None:
        raise ValueError("Subscription revoked")
    expiry = getattr(payload, "expiresDate", None)
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or expiry <= 0:
        raise ValueError("Missing subscription expiry")
    deadline = int(expiry / 1000)
    if deadline <= now:
        # A signed transaction alone does not prove billing-grace eligibility.
        raise ValueError("Subscription expired")
    return deadline


def token_deadline(subscription_expiry_ms: int, now: int, lifetime: int = 900) -> int:
    expiry = int(subscription_expiry_ms / 1000)
    if expiry <= now:
        raise ValueError("Subscription expired")
    return min(expiry, now + lifetime)


def validate_upstream(url: str, extra_hosts=()) -> str:
    if not isinstance(url, str) or len(url) > 4096:
        raise ValueError("Invalid provider URL")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid provider URL") from exc
    host = (parts.hostname or "").lower()
    if (parts.scheme != "https" or port not in (None, 443)
            or parts.username is not None or parts.password is not None or parts.fragment):
        raise ValueError("Provider URL must use HTTPS on port 443 without user information")
    allowed = PROVIDER_HOSTS | {h.strip().lower() for h in extra_hosts if h.strip()}
    if host not in allowed:
        raise ValueError("This provider host is not enabled for hosted research")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None or host == "localhost" or host.endswith((".local", ".internal")):
        raise ValueError("Provider must use an approved public hostname")
    return host


def validate_resolved_addresses(addresses) -> None:
    if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):
        raise ValueError("Provider must resolve only to public addresses")


def sanitize_upstream_headers(headers) -> dict:
    if not isinstance(headers, dict):
        raise ValueError("Provider headers must be an object")
    # Never allow caller-controlled Host, forwarding, connection or cookie headers.
    allowed = {"authorization", "x-api-key", "x-goog-api-key", "anthropic-version", "anthropic-beta"}
    result = {"Content-Type": "application/json"}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str) or len(value) > 8192:
            raise ValueError("Invalid provider header")
        if "\r" in value or "\n" in value:
            raise ValueError("Invalid provider header")
        if name.lower() in allowed:
            result[name] = value
    return result
