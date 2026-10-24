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


def load_viwiki(raw_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((Path(raw_dir) / "viwiki").glob("Q*.json"))}


def load_link_qids(raw_dir: Path) -> dict[str, str]:
    return {title: qid for title, qid in _read_json(Path(raw_dir) / "viwiki_links.json", {}).items() if qid}


def infobox_rows(viwiki: dict[str, dict], cfg_v: dict) -> dict[str, list[dict]]:
    from src.data.vnmmkg_triples import parse_infobox

    fields = [*cfg_v["infobox_fields"], *cfg_v["founded_fields"]]
    return {qid: parse_infobox(page["wikitext"], cfg_v["infobox_templates"], fields) for qid, page in viwiki.items()}


def landmark_candidates(raw_dir: Path, cfg_v: dict) -> tuple[list[str], dict[str, dict]]:
    from src.data.vnmmkg_wikidata import rank_candidates

    entities = load_entities(raw_dir)
    sparql_qids = _read_json(Path(raw_dir) / "candidates.json", {"sparql_qids": []})["sparql_qids"]
    ranked, _ = rank_candidates(entities, sparql_qids, cfg_v)
    return ranked, entities
