import asyncio
import tempfile
import unittest
from pathlib import Path
from job_store import JobStore, JobConflict, JobCapacityError


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "jobs.sqlite3"
        self.store = JobStore(self.path)

    async def asyncTearDown(self):
        await self.store.close()
        self.temp.cleanup()

    async def test_completed_results_survive_restart(self):
        await self.store.create("one", "alice", {})
        await self.store.update("one", status="completed", response={"content": "saved result"})
        other = JobStore(self.path)
        try:
            await other.recover_interrupted()
            self.assertEqual((await other.get("one", "alice"))["response"], {"content": "saved result"})
            self.assertIsNone(await other.get("one", "bob"))
        finally:
            await other.close()

    async def test_interrupted_work_has_an_explicit_terminal_outcome(self):
        await self.store.create("one", "alice", {})
        await self.store.update("one", status="running")
        await self.store.recover_interrupted()
        job = await self.store.get("one", "alice")
        self.assertEqual(job["status"], "failed")
        self.assertIn("server restart", job["error"])

    async def test_concurrent_duplicate_starts_create_only_one_job(self):
        results = await asyncio.gather(*(self.store.create(str(i), "alice", {"prompt": "same"}, idempotency_key="request-one") for i in range(8)))
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertEqual(len({job_id for job_id, _ in results}), 1)

    async def test_idempotency_rejects_changed_body_and_is_scoped_to_owner(self):
        await self.store.create("one", "alice", {"a": 1}, idempotency_key="same-key")
        with self.assertRaises(JobConflict):
            await self.store.create("two", "alice", {"a": 2}, idempotency_key="same-key")
        self.assertEqual(await self.store.create("three", "bob", {"a": 2}, idempotency_key="same-key"), ("three", True))

    async def test_admission_is_bounded_and_terminal_jobs_release_capacity(self):
        self.store.per_owner_limit = 1
        await self.store.create("one", "alice", {})
        with self.assertRaises(JobCapacityError):
            await self.store.create("two", "alice", {})
        await self.store.update("one", status="completed", response={})
        self.assertEqual(await self.store.create("two", "alice", {}), ("two", True))

    async def test_cancelled_work_cannot_be_overwritten_by_a_late_result(self):
        await self.store.create("one", "alice", {})
        self.assertFalse(await self.store.cancel("one", "bob"))
        self.assertTrue(await self.store.cancel("one", "alice"))
        await self.store.update("one", status="completed", response={"late": True})
        self.assertEqual((await self.store.get("one", "alice"))["status"], "cancelled")

    async def test_retention_does_not_evict_recent_results(self):
        await self.store.create("one", "alice", {})
        await self.store.update("one", status="completed", response={})
        self.assertEqual(await self.store.cleanup_old(), 0)
        self.store.db.execute("UPDATE jobs SET updated_at=0")
        self.store.db.commit()
        self.assertEqual(await self.store.cleanup_old(), 1)

    async def test_revocations_survive_restart(self):
        self.store.set_revoked("txn-one", 4_000_000_000)
        other = JobStore(self.path)
        try:
            self.assertTrue(other.is_revoked("txn-one"))
            other.set_revoked("txn-one", 4_000_000_000, revoked=False)
            self.assertFalse(self.store.is_revoked("txn-one"))
        finally:
            await other.close()
