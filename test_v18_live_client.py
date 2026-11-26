"""v18.1 live client tests: retries, per-item isolation, ordering. No network access."""
import unittest

import requests

from hn_live import BASE, HackerNewsClient


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Scripted responses: url -> list of outcomes consumed in order (Exception or FakeResponse)."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        key = url[len(BASE) + 1:-len(".json")]
        outcomes = self.script.get(key)
        if not outcomes:
            return FakeResponse(None, 404)
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def story(i):
    return {"id": i, "type": "story", "title": f"s{i}", "score": i, "time": 1700000000 + i, "by": "u"}


class RetryTests(unittest.TestCase):
    def test_transient_timeout_is_retried(self):
        sleeps = []
        session = FakeSession({"topstories": [FakeResponse([1])],
                               "item/1": [requests.Timeout("slow"), FakeResponse(story(1))]})
        client = HackerNewsClient(session=session, sleep=sleeps.append, backoff=0.5)
        df = client.fetch_feed("top", limit=1, pause=0)
        self.assertEqual(list(df["id"]), [1])
        self.assertEqual(sleeps, [0.5])  # one backoff before the successful retry

    def test_http_429_is_retried_then_succeeds(self):
        session = FakeSession({"topstories": [FakeResponse([2])],
                               "item/2": [FakeResponse(None, 429), FakeResponse(story(2))]})
        client = HackerNewsClient(session=session, sleep=lambda _: None)
        self.assertEqual(list(client.fetch_feed("top", limit=1, pause=0)["id"]), [2])

    def test_non_retryable_404_is_not_retried(self):
        session = FakeSession({"topstories": [FakeResponse([3])], "item/3": [FakeResponse(None, 404)]})
        client = HackerNewsClient(session=session, sleep=lambda _: None)
        df = client.fetch_feed("top", limit=1, pause=0)
        self.assertTrue(df.empty)
        self.assertEqual(session.calls.count(f"{BASE}/item/3.json"), 1)
        self.assertEqual(len(client.failures), 1)

    def test_retries_are_bounded(self):
        session = FakeSession({"topstories": [FakeResponse([4])],
                               "item/4": [requests.ConnectionError("down")]})
        client = HackerNewsClient(session=session, sleep=lambda _: None, max_retries=2)
        df = client.fetch_feed("top", limit=1, pause=0)
        self.assertTrue(df.empty)
        self.assertEqual(session.calls.count(f"{BASE}/item/4.json"), 3)  # 1 try + 2 retries
        self.assertEqual(client.failures[0][0], "4")


class IsolationAndOrderingTests(unittest.TestCase):
    def test_one_failed_item_does_not_abort_feed(self):
        session = FakeSession({
            "topstories": [FakeResponse([10, 11, 12])],
            "item/10": [FakeResponse(story(10))],
            "item/11": [requests.ConnectionError("gone")],
            "item/12": [FakeResponse(story(12))],
        })
        client = HackerNewsClient(session=session, sleep=lambda _: None, max_retries=0)
        df = client.fetch_feed("top", limit=3, pause=0, max_workers=3)
        self.assertEqual(list(df["id"]), [10, 12])
        self.assertEqual([f[0] for f in client.failures], ["11"])

    def test_output_order_matches_feed_order(self):
        ids = list(range(20, 40))
        session = FakeSession({"topstories": [FakeResponse(ids)],
                               **{f"item/{i}": [FakeResponse(story(i))] for i in ids}})
        client = HackerNewsClient(session=session, sleep=lambda _: None)
        df = client.fetch_feed("top", limit=20, pause=0, max_workers=8)
        self.assertEqual(list(df["id"]), ids)

    def test_invalid_worker_count_rejected(self):
        client = HackerNewsClient(session=FakeSession({}), sleep=lambda _: None)
        with self.assertRaises(ValueError):
            client.fetch_feed("top", limit=1, max_workers=0)


if __name__ == "__main__":
    unittest.main()
