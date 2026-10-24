"""Read the Vietnamese MMKG raw cache (data/raw/vnmmkg/) written by the fetch scripts. No network."""
import json
from pathlib import Path


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def load_entities(raw_dir: Path) -> dict[str, dict]:
    wd = Path(raw_dir) / "wikidata"
    redirects = _read_json(wd / "_redirects.json", {})
    entities = {}
    for path in sorted(wd.glob("Q*.json")):
        entity = json.loads(path.read_text(encoding="utf-8"))
        for statements in entity["claims"].values():
            for s in statements:
                if isinstance(s["value"], str) and s["value"] in redirects:
                    s["value"] = redirects[s["value"]]
        entities[entity["qid"]] = entity
    return entities
