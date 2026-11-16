import email.utils

import pytest

from src.data.wikimedia_client import WikimediaClient, retry_after_seconds


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def make_client(tmp_path, responses):
    waits = []
    session = FakeSession(responses)
    client = WikimediaClient(tmp_path, "test-agent", sleep=0.5, max_retries=3, max_wait=600,
                             session=session, sleeper=waits.append)
    return client, session, waits


def test_retry_after_seconds_integer_and_cap():
    assert retry_after_seconds("30", fallback=5, cap=600) == 30
    assert retry_after_seconds("9999", fallback=5, cap=600) == 600
    assert retry_after_seconds(None, fallback=5, cap=600) == 5
    assert retry_after_seconds("garbage", fallback=7, cap=600) == 7


def test_retry_after_http_date():
    now = 1_800_000_000.0
    header = email.utils.formatdate(now + 120, usegmt=True)
    assert retry_after_seconds(header, fallback=5, cap=600, now=now) == 120


def test_429_waits_then_succeeds_and_sets_user_agent(tmp_path):
    client, session, waits = make_client(tmp_path, [
        FakeResponse(429, headers={"Retry-After": "12"}),
        FakeResponse(200, {"ok": 1}),
    ])
    assert client.get_json("https://x/api", {"a": 1}) == {"ok": 1}
    assert waits == [12, 0.5]                       # backoff, then the polite delay after the 200
    assert session.headers["User-Agent"] == "test-agent"


def test_json_is_cached_on_disk(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(200, {"v": "Đền Hùng"})])
    first = client.get_json("https://x/api", {"q": "Đền"})
    second = client.get_json("https://x/api", {"q": "Đền"})
    assert first == second == {"v": "Đền Hùng"}
    assert len(session.calls) == 1


def test_api_error_body_is_retried_not_cached(tmp_path):
    client, session, waits = make_client(tmp_path, [
        FakeResponse(200, {"error": {"code": "maxlag", "info": "lagged"}}, headers={"Retry-After": "5"}),
        FakeResponse(200, {"query": {}}),
    ])
    assert client.get_json("https://x/api", {"a": 2}) == {"query": {}}
    assert 5 in waits
    assert len(list(tmp_path.glob("*.json"))) == 1   # only the good answer is cached


def test_other_api_error_raises(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(200, {"error": {"code": "badvalue", "info": "x"}})])
    with pytest.raises(RuntimeError, match="badvalue"):
        client.get_json("https://x/api", {"a": 3})


def test_gives_up_after_max_retries(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(503)] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.get_json("https://x/api", {"a": 4})


def test_download_returns_bytes_without_caching(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(200, content=b"img"), FakeResponse(200, content=b"img")])
    assert client.download("https://upload/x.jpg") == b"img"
    assert client.download("https://upload/x.jpg") == b"img"
    assert len(session.calls) == 2


class FlakySession(FakeSession):
    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_network_error_is_retried_with_backoff(tmp_path):
    import requests

    waits = []
    session = FlakySession([requests.exceptions.ReadTimeout("read timed out"), FakeResponse(200, content=b"img")])
    client = WikimediaClient(tmp_path, "ua", sleep=0.5, max_retries=3, session=session, sleeper=waits.append)
    assert client.download("https://thumb/x.jpg") == b"img"
    assert waits == [15, 0.5]


def test_network_error_gives_up_after_max_retries(tmp_path):
    import requests

    session = FlakySession([requests.exceptions.ConnectionError("reset")] * 3)
    client = WikimediaClient(tmp_path, "ua", max_retries=3, session=session, sleeper=lambda s: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.download("https://thumb/x.jpg")
