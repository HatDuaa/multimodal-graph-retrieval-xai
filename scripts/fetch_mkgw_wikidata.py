"""Fetch labels, descriptions, aliases and the main image of every MKG-W entity.

MKG-W (NativE release) only ships Wikidata URIs plus pre-extracted feature vectors;
the raw text and images are not distributed. This script rebuilds them from the
Wikidata SPARQL endpoint: for each entity in benchmarks/MKG-W/entity2id.txt it
stores one JSON line with the English and Vietnamese label/description/aliases
and the P18 image file name (if any). One query covers a whole batch of
entities, which is far faster than wbgetentities with claims. The cache is
resumable: already-fetched entities are skipped.

Usage:  python scripts/fetch_mkgw_wikidata.py [--limit N] [--batch N]
Output: data/raw/mkgw/wikidata_entities.jsonl (path from configs/default.yaml)
"""
import argparse
import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

SPARQL = "https://query.wikidata.org/sparql"
SLEEP = 3.0           # polite delay between queries
RETRIES = 6           # on 429/5xx: wait Retry-After (or an increasing fallback) and try again
HEADERS = {"User-Agent": "mkgw-retrieval-course-project/0.1 (academic use)"}
QUERY = """
SELECT ?item (SAMPLE(?len) AS ?label_en) (SAMPLE(?lvi) AS ?label_vi)
       (SAMPLE(?den) AS ?desc_en) (SAMPLE(?dvi) AS ?desc_vi)
       (GROUP_CONCAT(DISTINCT ?aen; separator="\\t") AS ?aliases_en)
       (GROUP_CONCAT(DISTINCT ?avi; separator="\\t") AS ?aliases_vi)
       (SAMPLE(?img) AS ?image) WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item rdfs:label ?len FILTER(LANG(?len) = "en") }
  OPTIONAL { ?item rdfs:label ?lvi FILTER(LANG(?lvi) = "vi") }
  OPTIONAL { ?item schema:description ?den FILTER(LANG(?den) = "en") }
  OPTIONAL { ?item schema:description ?dvi FILTER(LANG(?dvi) = "vi") }
  OPTIONAL { ?item skos:altLabel ?aen FILTER(LANG(?aen) = "en") }
  OPTIONAL { ?item skos:altLabel ?avi FILTER(LANG(?avi) = "vi") }
  OPTIONAL { ?item wdt:P18 ?img }
} GROUP BY ?item
"""


def read_entity2id(native_repo: Path) -> list[tuple[str, int]]:
    lines = (native_repo / "benchmarks" / "MKG-W" / "entity2id.txt").read_text().splitlines()
    n = int(lines[0])
    pairs = []
    for line in lines[1 : n + 1]:
        uri, idx = line.rsplit(" ", 1)
        pairs.append((uri.rsplit("/", 1)[-1], int(idx)))  # http://www.wikidata.org/entity/Q42 -> Q42
    assert len(pairs) == n
    return pairs


def image_file_name(binding: dict | None) -> str | None:
    """SPARQL returns a Special:FilePath URL; keep only the Commons file name."""
    if not binding:
        return None
    return urllib.parse.unquote(binding["value"].rsplit("/", 1)[-1]).replace("_", " ")


def parse_binding(b: dict) -> dict:
    def value(key):
        return b.get(key, {}).get("value") or None

    def aliases(key):
        raw = value(key)
        return raw.split("\t") if raw else []

    return {
        "qid": b["item"]["value"].rsplit("/", 1)[-1],
        "label_en": value("label_en"), "desc_en": value("desc_en"), "aliases_en": aliases("aliases_en"),
        "label_vi": value("label_vi"), "desc_vi": value("desc_vi"), "aliases_vi": aliases("aliases_vi"),
        "image": image_file_name(b.get("image")),
    }


def run_query(session: requests.Session, query: str) -> dict:
    for attempt in range(RETRIES):
        resp = session.post(SPARQL, data={"query": query, "format": "json"}, timeout=120)
        if resp.status_code in (429, 500, 502, 503):
            wait = int(resp.headers.get("Retry-After", 30 * (attempt + 1)))
            print(f"\nHTTP {resp.status_code}, waiting {wait}s (attempt {attempt + 1}/{RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"still rate-limited after {RETRIES} attempts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N entities (0 = all)")
    ap.add_argument("--batch", type=int, default=250)
    args = ap.parse_args()

    cfg = load_config()
    native_repo = Path(cfg["mkgw"]["native_repo"]).expanduser()
    out = REPO_ROOT / cfg["mkgw"]["wikidata_cache"]
    out.parent.mkdir(parents=True, exist_ok=True)

    pairs = read_entity2id(native_repo)
    id_of = dict(pairs)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["qid"])
    todo = [qid for qid, _ in pairs if qid not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(pairs)} entities, {len(done)} cached, {len(todo)} to fetch")

    session = requests.Session()
    session.headers.update(HEADERS)
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            query = QUERY % " ".join(f"wd:{qid}" for qid in batch)
            data = run_query(session, query)
            rows = {r["qid"]: r for r in map(parse_binding, data["results"]["bindings"])}
            for qid in batch:   # deleted/redirected entities get an empty row, so the run resumes past them
                row = rows.get(qid) or {"qid": qid, "label_en": None, "desc_en": None, "aliases_en": [],
                                        "label_vi": None, "desc_vi": None, "aliases_vi": [], "image": None}
                row["id"] = id_of[qid]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"fetched {min(start + args.batch, len(todo))}/{len(todo)}", end="\r")
            time.sleep(SLEEP)
    print("\ndone ->", out)


if __name__ == "__main__":
    main()
