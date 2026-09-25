"""Fetch English labels for the 169 MKG-W relations (one SPARQL query).

Output: data/raw/mkgw/wikidata_relations.json  ({pid: {"id": int, "label": str|null}}).
Deleted Wikidata properties keep label null; consumers fall back to the pid.
"""
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "mkgw-retrieval-course-project/0.1 (academic use)"}


def main() -> None:
    cfg = load_config()
    native_repo = Path(cfg["mkgw"]["native_repo"]).expanduser()
    lines = (native_repo / "benchmarks" / "MKG-W" / "relation2id.txt").read_text().splitlines()
    n = int(lines[0])
    pairs = [(l.rsplit(" ", 1)[0].rsplit("/", 1)[-1], int(l.rsplit(" ", 1)[1])) for l in lines[1 : n + 1]]

    values = " ".join(f"wd:{pid}" for pid, _ in pairs)
    query = ('SELECT ?item ?label WHERE { VALUES ?item { %s } '
             '?item rdfs:label ?label FILTER(LANG(?label)="en") }') % values
    resp = requests.post(SPARQL, data={"query": query, "format": "json"}, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    labels = {b["item"]["value"].rsplit("/", 1)[-1]: b["label"]["value"]
              for b in resp.json()["results"]["bindings"]}

    out_path = REPO_ROOT / "data" / "raw" / "mkgw" / "wikidata_relations.json"
    out = {pid: {"id": idx, "label": labels.get(pid)} for pid, idx in pairs}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = [p for p, v in out.items() if not v["label"]]
    print(f"{len(out)} relations, {len(out) - len(missing)} labelled, missing: {missing}")


if __name__ == "__main__":
    main()
