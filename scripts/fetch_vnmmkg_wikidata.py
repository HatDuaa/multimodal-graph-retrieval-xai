"""Step 1 of the Vietnamese MMKG pipeline: landmark candidates and their Wikidata closure.

1. One SPARQL query lists every item in Viet Nam (P17) that has a heritage designation
   (P1435) or whose P31 is one of the configured landmark classes.
2. Every candidate is fetched with wbgetentities (50 per call) and stored in compact form.
3. The closure needed by the build is fetched the same way: targets of the configured
   landmark PIDs, and the P131 ancestors of every landmark up to max_p131_depth (for the
   province rule), plus targets of context entities (for one-hop context triples).
All answers are cached (src/data/wikimedia_client.py); rerunning only fetches what is missing.

Usage:  python scripts/fetch_vnmmkg_wikidata.py
Output: data/raw/vnmmkg/candidates.json, data/raw/vnmmkg/wikidata/<qid>.json
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.vnmmkg_cache import load_entities  # noqa: E402
from src.data.vnmmkg_wikidata import KEEP_PIDS_EXTRA, qid_key, rank_candidates, trim_entity, truthy  # noqa: E402
from src.data.wikimedia_client import SPARQL_URL, WIKIDATA_API, client_from_config  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

CANDIDATE_QUERY = """
SELECT DISTINCT ?item WHERE {
  ?item wdt:P17 wd:%(country)s .
  { ?item wdt:P1435 ?designation . } UNION { ?item wdt:P31 ?cls . VALUES ?cls { %(classes)s } }
}"""


def keep_pids_from_config(cfg_v: dict) -> set[str]:
    return set(cfg_v["landmark_pids"]) | set(cfg_v["context_extra_pids"]) | set(cfg_v["exclude_if_has"]) | KEEP_PIDS_EXTRA


def fetch_entities(client, qids: list[str], out_dir: Path, keep_pids: set[str]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    redirects_path = out_dir / "_redirects.json"
    redirects = json.loads(redirects_path.read_text(encoding="utf-8")) if redirects_path.exists() else {}
    todo = sorted({q for q in qids if q not in redirects and not (out_dir / f"{q}.json").exists()}, key=qid_key)
    for start in range(0, len(todo), 50):
        batch = todo[start:start + 50]
        data = client.get_json(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(batch),
                                              "props": "labels|aliases|descriptions|claims|sitelinks|info",
                                              "languages": "vi|en", "maxlag": 5})
        for asked, raw in data["entities"].items():
            entity = trim_entity(raw, keep_pids)
            if entity is None:
                continue
            if entity["qid"] != asked:
                redirects[asked] = entity["qid"]
            (out_dir / f"{entity['qid']}.json").write_text(json.dumps(entity, ensure_ascii=False, indent=1),
                                                           encoding="utf-8", newline="\n")
        print(f"entities {min(start + 50, len(todo))}/{len(todo)}", end="\r")
    redirects_path.write_text(json.dumps(dict(sorted(redirects.items())), indent=1), encoding="utf-8", newline="\n")
    return redirects


def main() -> None:
    cfg = load_config()
    v = cfg["vnmmkg"]
    raw_dir = REPO_ROOT / v["raw_dir"]
    ent_dir = raw_dir / "wikidata"
    client = client_from_config(cfg)
    keep = keep_pids_from_config(v)

    cand_path = raw_dir / "candidates.json"
    if cand_path.exists():
        cands = json.loads(cand_path.read_text(encoding="utf-8"))
    else:
        query = CANDIDATE_QUERY % {"country": v["country"],
                                   "classes": " ".join(f"wd:{c}" for c in v["landmark_classes"])}
        rows = client.post_json(SPARQL_URL, {"query": query})["results"]["bindings"]
        qids = sorted({b["item"]["value"].rsplit("/", 1)[-1] for b in rows}, key=qid_key)
        cands = {"fetched": datetime.date.today().isoformat(), "sparql_qids": qids}
        raw_dir.mkdir(parents=True, exist_ok=True)
        cand_path.write_text(json.dumps(cands, indent=1), encoding="utf-8", newline="\n")
    print(f"SPARQL candidates: {len(cands['sparql_qids'])} (fetched {cands['fetched']})")

    fetch_entities(client, cands["sparql_qids"], ent_dir, keep)
    entities = load_entities(raw_dir)
    ranked, rejected = rank_candidates(entities, cands["sparql_qids"], {**v, "max_landmark_candidates": 10**6})
    print(f"\nvalid candidates: {len(ranked)}; rejected: {rejected}")
    landmarks = ranked[: v["max_landmark_candidates"]]

    # targets of landmark relations (context entities)
    targets = {t for q in ranked for pid in v["landmark_pids"] for t in truthy(entities[q], pid) if t.startswith("Q")}
    targets |= {p["qid"] for p in v["periods"] if p["qid"]}      # tails of the derived period rule
    fetch_entities(client, sorted(targets, key=qid_key), ent_dir, keep)
    # P131 ancestors of every valid candidate, for the province rule
    frontier = {t for q in ranked for t in truthy(entities[q], "P131")}
    visited: set[str] = set()
    for depth in range(v["max_p131_depth"]):
        fetch_entities(client, sorted(frontier, key=qid_key), ent_dir, keep)
        entities = load_entities(raw_dir)
        visited |= frontier
        frontier = {t for q in frontier if q in entities for t in truthy(entities[q], "P131")} - visited
        if not frontier:
            break
    entities = load_entities(raw_dir)
    print(f"cached entities: {len(entities)}; landmark candidates kept for image review: {len(landmarks)}")


if __name__ == "__main__":
    main()
