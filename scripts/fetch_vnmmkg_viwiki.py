"""Step 2 of the Vietnamese MMKG pipeline: lead text and infobox of each candidate's viwiki article.

For every valid landmark candidate (step 1), one request fetches the plain-text lead
(`extracts`, exintro) and the wikitext of the same article, pinned by its revid. Then the
wikilinks found in the configured infobox fields are resolved to Wikidata QIDs through
viwiki `pageprops` (following redirects), and the entities behind them are fetched like in
step 1. `--inspect` prints which infobox templates and fields occur, to fill the mapping
in configs/default.yaml before resolving links.

Usage:  python scripts/fetch_vnmmkg_viwiki.py [--inspect]
Output: data/raw/vnmmkg/viwiki/<qid>.json, data/raw/vnmmkg/viwiki_links.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import mwparserfromhell

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_vnmmkg_wikidata import fetch_entities, keep_pids_from_config  # noqa: E402
from src.data.vnmmkg_cache import infobox_rows, landmark_candidates, load_viwiki  # noqa: E402
from src.data.vnmmkg_triples import normalize_title  # noqa: E402
from src.data.wikimedia_client import VIWIKI_API, client_from_config  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402


def fetch_page(client, title: str) -> dict | None:
    data = client.get_json(VIWIKI_API, {"action": "query", "prop": "extracts|revisions", "exintro": 1,
                                        "explaintext": 1, "rvprop": "ids|content", "rvslots": "main",
                                        "redirects": 1, "titles": title, "formatversion": 2, "maxlag": 5})
    page = data["query"]["pages"][0]
    if page.get("missing") or "revisions" not in page:
        return None
    rev = page["revisions"][0]
    return {"page_title": page["title"], "pageid": page["pageid"], "revid": rev["revid"],
            "extract": page.get("extract", ""), "wikitext": rev["slots"]["main"]["content"]}


def resolve_titles(client, titles: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        data = client.get_json(VIWIKI_API, {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                                            "redirects": 1, "titles": "|".join(batch), "formatversion": 2,
                                            "maxlag": 5})
        q = data["query"]
        normalized = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in q.get("redirects", [])}
        item_of = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q["pages"]}
        for title in batch:
            final = normalized.get(title, title)
            final = redirects.get(final, final)
            out[title] = item_of.get(final)
    return out


def inspect(viwiki: dict[str, dict]) -> None:
    templates, fields = Counter(), {}
    for page in viwiki.values():
        for tpl in mwparserfromhell.parse(page["wikitext"]).filter_templates(recursive=False):
            name = normalize_title(tpl.name)
            if name.startswith(("Thông tin", "Infobox", "Hộp thông tin")):
                templates[name] += 1
                fields.setdefault(name, Counter()).update(str(p.name).strip() for p in tpl.params)
    for name, n in templates.most_common():
        print(f"{n:4d}  {name}")
        print("      ", ", ".join(f"{f}({c})" for f, c in fields[name].most_common(40)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    v = cfg["vnmmkg"]
    raw_dir = REPO_ROOT / v["raw_dir"]
    client = client_from_config(cfg)

    ranked, entities = landmark_candidates(raw_dir, {**v, "max_landmark_candidates": 10**6})
    out_dir = raw_dir / "viwiki"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for i, qid in enumerate(ranked, 1):
        path = out_dir / f"{qid}.json"
        if path.exists():
            continue
        title = entities[qid]["viwiki_title"]
        page = fetch_page(client, title)
        if page is None:
            missing += 1
            continue
        path.write_text(json.dumps({"qid": qid, "title": title, **page}, ensure_ascii=False, indent=1),
                        encoding="utf-8", newline="\n")
        print(f"pages {i}/{len(ranked)}", end="\r")
    viwiki = load_viwiki(raw_dir)
    print(f"\nviwiki pages cached: {len(viwiki)}; missing articles: {missing}")
    if args.inspect:
        inspect(viwiki)
        return

    rows = infobox_rows(viwiki, v)
    titles = sorted({t for page_rows in rows.values() for r in page_rows for t in r["links"]})
    links_path = raw_dir / "viwiki_links.json"
    links = json.loads(links_path.read_text(encoding="utf-8")) if links_path.exists() else {}
    links.update(resolve_titles(client, [t for t in titles if t not in links]))
    links_path.write_text(json.dumps(dict(sorted(links.items())), ensure_ascii=False, indent=1),
                          encoding="utf-8", newline="\n")
    new_qids = sorted({q for q in links.values() if q})
    fetch_entities(client, new_qids, raw_dir / "wikidata", keep_pids_from_config(v))
    n_rows = sum(len(r) for r in rows.values())
    print(f"infobox rows: {n_rows}; link titles: {len(titles)}; resolved: {sum(1 for t in titles if links.get(t))}")


if __name__ == "__main__":
    main()
