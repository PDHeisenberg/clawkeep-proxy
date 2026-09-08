# ClawKeep proxy

Python 3.12 service for verified Pro subscriptions and provider-backed research.

## Configuration

Required in production: `JWT_SECRET` (a persistent random secret), `APP_APPLE_ID`
(the numeric App Store Connect ID), `AI_PROVIDER=openai`, `AI_API_KEY`, and
`DEFAULT_MODEL=gpt-5.4-mini`. `APPLE_ENVIRONMENT` defaults to `production`.
Use a separate service and secret with `APPLE_ENVIRONMENT=sandbox` for TestFlight.
Production never accepts sandbox receipts. `ALLOWED_PRO_MODELS` can narrow the
comma-separated model allowlist; requests also have output and tool-call limits.

Set `JOB_DATABASE_PATH=/data/jobs.sqlite3` and mount a persistent volume at `/data`.
Use **one service replica and one Uvicorn process**. SQLite persists job results,
idempotency records and transaction revocations. Network workers are in-process:
a restart marks unfinished jobs failed with a retry message rather than pretending
to resume a provider request. Completed results remain available for seven days.
Back up the volume with SQLite's online backup API. To scale beyond one replica,
replace JobStore with a shared database and move execution to a leased worker
queue; do not increase web replicas against this implementation.

Configure App Store Server Notifications V2 to POST to `/app-store/notifications`
for each environment. Only Apple-verified notifications are accepted. Refund and
revocation records invalidate existing tokens. Tokens expire within 15 minutes
and never outlive the verified subscription. A billing grace period is not inferred
from an expired receipt; supporting grace requires verified renewal-state evidence.

## Research API

`POST /research` requires a Pro Bearer token. `POST /research/passthrough` uses
the caller's provider credentials and requires `X-ClawKeep-Client-ID`, a stable
installation UUID. This ID scopes admission; it is not an authentication credential.
Both accept `Idempotency-Key` (use the durable client work-ticket ID) and return
`job_id` and `status`. Reusing a key with a changed body returns 409. Each owner
may have four pending/running jobs; the whole service admits at most 100.

Passthrough also returns `job_token`. Persist it privately alongside the job ID and
send it as `X-Research-Token` when polling or cancelling that job. Pro jobs require
the current owner's Bearer token. Knowing a job ID alone does not grant access.
Treat tokens and provider headers as secrets and never put them in URLs or logs.

Relay destinations are limited to the three built-in provider hosts over HTTPS.
`CUSTOM_UPSTREAM_HOSTS` may explicitly add trusted custom-provider hostnames.
These must resolve only to public addresses; redirects, private destinations,
arbitrary ports and caller-supplied transport headers are rejected. Provider API
keys and request bodies are never stored in SQLite. Treat custom hosts as trusted
administrative configuration, not user input. Research results do contain user
content and belong on an access-restricted volume.

## Validation and rollout

Run `pip install -r requirements.txt` and `python -m unittest discover -s tests -v`.
The tests use temporary databases and mock provider boundaries, not paid requests.
CI runs the complete HTTP and unit suites. On machines without dependencies,
the standard-library unit tests run and the HTTP suite explicitly skips.

The installation/capability headers and short token lifetime require the matching
iOS update. Validate both branches against a sandbox deployment before release.
Before promoting: attach the volume, configure the correct Apple environment,
configure notifications, test purchase/restore/refund, and exercise a server restart
during research. Rolling back code must preserve the volume and JWT secret.
