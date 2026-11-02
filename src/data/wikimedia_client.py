"""HTTP access to Wikidata, Vietnamese Wikipedia and Wikimedia Commons for the Vietnamese MMKG.

Every JSON answer is cached on disk (one file per request, keyed by a hash of method, URL
and parameters), so the fetch scripts can be stopped and rerun at any time and the dataset
can be rebuilt without calling the APIs again. Rate limiting (HTTP 429/5xx, or a MediaWiki
`maxlag`-style error body) is handled by waiting `Retry-After` (seconds or HTTP date,
capped) or an increasing fallback, then retrying. Error answers are never cached. Network errors
(connection reset, read timeout) are retried the same way. A short polite delay follows every
successful request.
"""
import email.utils
import hashlib
import json
import time
from pathlib import Path

import requests

SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
VIWIKI_API = "https://vi.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_API_ERRORS = {"maxlag", "ratelimited", "readonly"}


def retry_after_seconds(value: str | None, fallback: int, cap: int, now: float | None = None) -> int:
    """Seconds to wait for a Retry-After header given as seconds or as an HTTP date."""
    if value is not None:
        value = value.strip()
        if value.isdigit():
            return min(int(value), cap)
        try:
            when = email.utils.parsedate_to_datetime(value).timestamp()
            return min(max(int(round(when - (time.time() if now is None else now))), 0), cap)
        except (TypeError, ValueError):
            pass
    return min(fallback, cap)


class WikimediaClient:
    def __init__(self, cache_dir: Path, user_agent: str, sleep: float = 1.0, max_retries: int = 6,
                 max_wait: int = 600, session=None, sleeper=time.sleep):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep, self.max_retries, self.max_wait = sleep, max_retries, max_wait
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.sleeper = sleeper

    def get_json(self, url: str, params: dict) -> dict:
        return self._cached_json("GET", url, params=dict(params, format="json"))

    def post_json(self, url: str, data: dict) -> dict:
        return self._cached_json("POST", url, data=dict(data, format="json"))

    def download(self, url: str) -> bytes:
        return self._request("GET", url).content

    def _cache_path(self, method: str, url: str, payload: dict) -> Path:
        key = json.dumps([method, url, sorted(payload.items())], ensure_ascii=False, default=str)
        return self.cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"

    def _cached_json(self, method: str, url: str, **kwargs) -> dict:
        path = self._cache_path(method, url, kwargs.get("params") or kwargs.get("data"))
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(self.max_retries):
            resp = self._request(method, url, **kwargs)
            data = resp.json()
            error = data.get("error") if isinstance(data, dict) else None
            if error is None:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
                return data
            if error.get("code") not in RETRY_API_ERRORS:
                raise RuntimeError(f"API error {error.get('code')}: {error.get('info')}")
            self.sleeper(retry_after_seconds(resp.headers.get("Retry-After"), 15 * (attempt + 1), self.max_wait))
        raise RuntimeError(f"API still erroring after {self.max_retries} attempts: {url}")

    def _request(self, method: str, url: str, **kwargs):
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(method, url, timeout=120, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
                wait = min(15 * (attempt + 1), self.max_wait)
                print(f"{type(err).__name__} from {url}, waiting {wait}s ({attempt + 1}/{self.max_retries})")
                self.sleeper(wait)
                continue
            if resp.status_code in RETRY_STATUS:
                wait = retry_after_seconds(resp.headers.get("Retry-After"), 15 * (attempt + 1), self.max_wait)
                print(f"HTTP {resp.status_code} from {url}, waiting {wait}s ({attempt + 1}/{self.max_retries})")
                self.sleeper(wait)
                continue
            resp.raise_for_status()
            self.sleeper(self.sleep)
            return resp
        raise RuntimeError(f"still rate-limited after {self.max_retries} attempts: {url}")


def client_from_config(cfg: dict) -> WikimediaClient:
    from src.utils.config import REPO_ROOT

    v = cfg["vnmmkg"]
    return WikimediaClient(REPO_ROOT / v["raw_dir"] / "http_cache", v["user_agent"], sleep=v["sleep_seconds"],
                           max_retries=v["max_retries"], max_wait=v["max_retry_wait"])
