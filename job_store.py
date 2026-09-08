"""Durable job metadata/results. Provider credentials are never persisted.

SQLite serializes admission/idempotency decisions across processes. Network
workers stay in the serving process; interrupted jobs become explicit failures
on restart, while completed results remain available for seven days.
"""
import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class JobConflict(ValueError):
    pass


class JobCapacityError(ValueError):
    pass


class JobStore:
    def __init__(self, path="data/jobs.sqlite3", retention_seconds=7 * 86400,
                 per_owner_limit=4, global_limit=100):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, sub TEXT NOT NULL, mode TEXT NOT NULL,
                idempotency_key TEXT, request_hash TEXT NOT NULL, status TEXT NOT NULL,
                response TEXT, error TEXT, progress TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                UNIQUE(sub, mode, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS jobs_status_owner ON jobs(status, sub);
            CREATE TABLE IF NOT EXISTS revoked_transactions (
                transaction_id TEXT PRIMARY KEY, expires_at REAL NOT NULL
            );
        """)
        self.db.commit()
        self.lock = threading.RLock()
        self._workers = {}
        self.retention_seconds = retention_seconds
        self.per_owner_limit = per_owner_limit
        self.global_limit = global_limit

    async def _run(self, operation):
        def locked():
            with self.lock:
                return operation()
        return await asyncio.to_thread(locked)

    async def create(self, job_id, sub, request_body, *, mode="pro", idempotency_key=None):
        fingerprint = hashlib.sha256(json.dumps(request_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        def operation():
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                if idempotency_key:
                    old = self.db.execute("SELECT * FROM jobs WHERE sub=? AND mode=? AND idempotency_key=?",
                                          (sub, mode, idempotency_key)).fetchone()
                    if old:
                        if old["request_hash"] != fingerprint:
                            raise JobConflict("Request key was already used for different research")
                        return old["id"], False
                rows = self.db.execute("SELECT sub, COUNT(*) AS n FROM jobs WHERE status IN ('pending','running') GROUP BY sub").fetchall()
                if sum(r["n"] for r in rows) >= self.global_limit or any(r["sub"] == sub and r["n"] >= self.per_owner_limit for r in rows):
                    raise JobCapacityError("Research queue is full. Wait for an existing job to finish.")
                now = time.time()
                self.db.execute("INSERT INTO jobs(id,sub,mode,idempotency_key,request_hash,status,created_at,updated_at) VALUES(?,?,?,?,?,'pending',?,?)",
                                (job_id, sub, mode, idempotency_key, fingerprint, now, now))
                return job_id, True
        return await self._run(operation)

    @staticmethod
    def _view(row):
        if row is None:
            return None
        return {key: json.loads(row[key]) if row[key] is not None and key in ("response", "progress") else row[key]
                for key in ("id", "status", "response", "error", "progress", "created_at", "updated_at")}

    async def update(self, job_id, status=None, response=None, error=None, progress=None):
        def operation():
            with self.db:
                row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row is None or row["status"] in ("completed", "failed", "cancelled"):
                    return
                self.db.execute("UPDATE jobs SET status=?, response=?, error=?, progress=?, updated_at=? WHERE id=?",
                                (status or row["status"], json.dumps(response) if response is not None else row["response"],
                                 error if error is not None else row["error"],
                                 json.dumps(progress) if progress is not None else row["progress"], time.time(), job_id))
        await self._run(operation)

    async def get(self, job_id, sub):
        return await self._run(lambda: self._view(self.db.execute("SELECT * FROM jobs WHERE id=? AND sub=?", (job_id, sub)).fetchone()))

    async def get_by_id_only(self, job_id):
        # Internal primitive. Public handlers must authorize before calling.
        return await self._run(lambda: self._view(self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()))

    async def identity(self, job_id):
        return await self._run(lambda: self.db.execute("SELECT sub,mode FROM jobs WHERE id=?", (job_id,)).fetchone())

    async def list_active(self, sub):
        def operation():
            rows = self.db.execute("SELECT * FROM jobs WHERE sub=? AND (status IN ('pending','running') OR updated_at>?) ORDER BY created_at DESC LIMIT 100",
                                   (sub, time.time() - 60)).fetchall()
            return [self._view(r) for r in rows]
        return await self._run(operation)

    async def cancel_by_id_only(self, job_id):
        def operation():
            with self.db:
                cur = self.db.execute("UPDATE jobs SET status='cancelled',updated_at=? WHERE id=? AND status IN ('pending','running')", (time.time(), job_id))
                return cur.rowcount > 0
        changed = await self._run(operation)
        worker = self._workers.pop(job_id, None)
        if worker and not worker.done():
            worker.cancel()
        return changed

    async def cancel(self, job_id, sub):
        if await self.get(job_id, sub) is None:
            return False
        return await self.cancel_by_id_only(job_id)

    def register_worker(self, job_id, task):
        self._workers[job_id] = task

    def unregister_worker(self, job_id):
        self._workers.pop(job_id, None)

    async def recover_interrupted(self):
        def operation():
            with self.db:
                self.db.execute("UPDATE jobs SET status='failed',error=?,updated_at=? WHERE status IN ('pending','running')",
                                ("Research was interrupted by a server restart. Retry to start a new attempt.", time.time()))
        await self._run(operation)

    async def cleanup_old(self):
        def operation():
            with self.db:
                cur = self.db.execute("DELETE FROM jobs WHERE status IN ('completed','failed','cancelled') AND updated_at<?", (time.time() - self.retention_seconds,))
                self.db.execute("DELETE FROM revoked_transactions WHERE expires_at<?", (time.time(),))
                return cur.rowcount
        return await self._run(operation)

    def is_revoked(self, transaction_id):
        with self.lock:
            return self.db.execute("SELECT 1 FROM revoked_transactions WHERE transaction_id=? AND expires_at>?", (transaction_id, time.time())).fetchone() is not None

    def set_revoked(self, transaction_id, expiry, revoked=True):
        with self.lock, self.db:
            if revoked:
                self.db.execute("INSERT OR REPLACE INTO revoked_transactions VALUES(?,?)", (transaction_id, expiry))
            else:
                self.db.execute("DELETE FROM revoked_transactions WHERE transaction_id=?", (transaction_id,))

    async def close(self):
        workers = list(self._workers.values())
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        await self._run(self.db.close)
