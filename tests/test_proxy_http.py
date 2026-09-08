"""HTTP contract tests run in CI with the pinned application dependencies."""
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("HTTP tests require requirements.txt (installed in CI)")

temp = tempfile.TemporaryDirectory()
os.environ.update(JWT_SECRET="test-only-secret-at-least-32-characters", APPLE_ENVIRONMENT="sandbox",
                  JOB_DATABASE_PATH=temp.name + "/jobs.sqlite3", AI_PROVIDER="openai")
from fastapi.testclient import TestClient
import jwt
import time
import proxy


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(proxy.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        temp.cleanup()

    def auth(self, sub="alice"):
        token = jwt.encode({"sub": sub, "jti": "txn-" + sub, "exp": int(time.time()) + 300}, proxy.JWT_SECRET, algorithm="HS256")
        return {"Authorization": "Bearer " + token}

    def test_pro_endpoints_require_authentication(self):
        for path in ("/research", "/v1/chat/completions", "/v1/responses"):
            self.assertEqual(self.client.post(path, json={}).status_code, 401)

    def test_internal_relay_destination_rejected_without_worker_dispatch(self):
        with patch.object(proxy, "_run_research_job_generic") as worker:
            response = self.client.post("/research/passthrough", json={"upstream_url": "http://127.0.0.1:8080/internal", "request_body": {}})
            self.assertEqual(response.status_code, 400)
            worker.assert_not_called()

    def test_unbounded_model_and_body_rejected_before_provider_call(self):
        self.assertEqual(self.client.post("/research", json={"model": "unapproved"}, headers=self.auth()).status_code, 400)
        self.assertEqual(self.client.post("/research", content="x" * (2 * 1024 * 1024 + 1), headers=self.auth()).status_code, 413)

    def test_malformed_json_returns_client_error(self):
        self.assertEqual(self.client.post("/research", json=[], headers=self.auth()).status_code, 400)
        self.assertEqual(self.client.post("/research", content="{", headers=self.auth()).status_code, 400)

    def test_unsigned_store_notification_rejected(self):
        self.assertEqual(self.client.post("/app-store/notifications", json={"signedPayload": "invalid"}).status_code, 400)

    def test_job_results_and_cancellation_require_the_owner(self):
        import uuid
        job_id = str(uuid.uuid4())
        self.client.portal.call(proxy.job_store.create, job_id, "alice", {})
        for method in (self.client.get, self.client.delete):
            self.assertEqual(method("/research/" + job_id).status_code, 401)
            self.assertEqual(method("/research/" + job_id, headers=self.auth("bob")).status_code, 404)
        self.assertEqual(self.client.get("/research/" + job_id, headers=self.auth()).status_code, 200)
        self.assertEqual(self.client.delete("/research/" + job_id, headers=self.auth()).status_code, 200)

    def test_passthrough_job_id_and_pro_token_do_not_replace_private_capability(self):
        import uuid
        from functools import partial
        job_id = str(uuid.uuid4())
        self.client.portal.call(partial(proxy.job_store.create, job_id, "byok:installation", {}, mode="passthrough"))
        for headers in ({}, self.auth(), {"X-Research-Token": "wrong"}):
            self.assertEqual(self.client.get("/research/" + job_id, headers=headers).status_code, 404)
        headers = {"X-Research-Token": proxy.job_capability(job_id)}
        self.assertEqual(self.client.get("/research/" + job_id, headers=headers).status_code, 200)
        self.assertEqual(self.client.delete("/research/" + job_id, headers=headers).status_code, 200)

    def test_replayed_start_returns_same_job_without_second_worker(self):
        from unittest.mock import AsyncMock
        import uuid
        headers = {**self.auth("idempotency"), "Idempotency-Key": str(uuid.uuid4())}
        with patch.object(proxy, "_run_research_job_generic", new_callable=AsyncMock) as worker:
            first = self.client.post("/research", json={"input": "Question"}, headers=headers)
            second = self.client.post("/research", json={"input": "Question"}, headers=headers)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.json()["job_id"], second.json()["job_id"])
            self.assertEqual(worker.call_count, 1)
            changed = self.client.post("/research", json={"input": "Changed question"}, headers=headers)
            self.assertEqual(changed.status_code, 409)

    def test_refunded_transaction_rejects_an_otherwise_unexpired_token(self):
        proxy.job_store.set_revoked("txn-refunded", time.time() + 600)
        response = self.client.post("/research", json={}, headers=self.auth("refunded"))
        self.assertEqual(response.status_code, 401)
