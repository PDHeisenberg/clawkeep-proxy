import unittest
from types import SimpleNamespace
from security import subscription_deadline, token_deadline, validate_upstream, validate_resolved_addresses, sanitize_upstream_headers


class SubscriptionTests(unittest.TestCase):
    now = 1_800_000_000

    def payload(self, **overrides):
        values = dict(productId="com.clawkeep.pro.monthly", expiresDate=(self.now + 3600) * 1000)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_active_token_never_outlives_subscription(self):
        expiry = subscription_deadline(self.payload(), self.now)
        self.assertEqual(expiry, self.now + 3600)
        self.assertEqual(token_deadline(expiry * 1000, self.now), self.now + 900)
        self.assertEqual(token_deadline((self.now + 30) * 1000, self.now), self.now + 30)

    def test_expired_receipt_cannot_gain_thirty_days(self):
        for seconds in (0, -1, -86400, -6 * 86400):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                subscription_deadline(self.payload(expiresDate=(self.now + seconds) * 1000), self.now)
            with self.assertRaises(ValueError):
                token_deadline((self.now + seconds) * 1000, self.now)

    def test_revoked_and_incomplete_receipts_rejected(self):
        for changes in ({"revocationDate": self.now * 1000}, {"expiresDate": None},
                        {"expiresDate": 0}, {"expiresDate": True}, {"productId": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                subscription_deadline(self.payload(**changes), self.now)


class DestinationTests(unittest.TestCase):
    def test_only_explicit_provider_hosts(self):
        self.assertEqual(validate_upstream("https://api.openai.com/v1/responses"), "api.openai.com")
        for url in ("http://127.0.0.1:8080/internal", "https://api.openai.com.evil.test/v1/responses",
                    "https://api.openai.com@evil.test", "https://api.openai.com:444/v1/responses",
                    "https://evil.test/?provider=api.openai.com", "file:///etc/passwd",
                    "https://api.openai.com/#secret", "https://api.openai.com:bad/", None):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_upstream(url)

    def test_custom_hosts_need_explicit_configuration(self):
        self.assertEqual(validate_upstream("https://models.example.org/v1/responses", ["models.example.org"]), "models.example.org")
        for host in ("127.0.0.1", "localhost", "metadata.internal"):
            with self.assertRaises(ValueError):
                validate_upstream("https://" + host, [host])

    def test_private_or_mixed_dns_answers_rejected(self):
        validate_resolved_addresses(["8.8.8.8", "2606:4700:4700::1111"])
        for addresses in ([], ["127.0.0.1"], ["::1"], ["169.254.169.254"],
                          ["10.0.0.1"], ["8.8.8.8", "192.168.1.1"]):
            with self.assertRaises(ValueError):
                validate_resolved_addresses(addresses)

    def test_transport_headers_cannot_be_injected(self):
        headers = sanitize_upstream_headers({"Authorization": "Bearer test", "Host": "internal",
                                             "Cookie": "private", "X-Forwarded-For": "127.0.0.1"})
        self.assertEqual(headers, {"Content-Type": "application/json", "Authorization": "Bearer test"})
        with self.assertRaises(ValueError):
            sanitize_upstream_headers({"Authorization": "key\r\nHost: internal"})
