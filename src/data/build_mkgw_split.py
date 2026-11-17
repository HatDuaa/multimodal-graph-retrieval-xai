"""Build the MKG-W retrieval split files (format: docs/interfaces.md, section 1).

The retrieval task on MKG-W: the query is an entity description with the entity
name masked out, the answer is the entity's image. Images come from the dataset
authors' own archive (MMRNS MKG-W_img.zip, extracted by
scripts/extract_mkgw_images.py) — the same images the official visual
embeddings were computed from. Only entities with an English description AND an
author image are kept. Entities are split train/val/test by a seeded shuffle
(fractions in configs/default.yaml); this is an entity-level split, so no query
in val/test describes a train candidate. NativE's own KGC triple split is
copied out separately to data/processed/mkgw_triples.jsonl so that
graph-derived signals can be restricted to KGC-train triples later
(anti-leakage, assignment section 6).

Prerequisites: scripts/fetch_mkgw_wikidata.py (Wikidata cache) and
scripts/extract_mkgw_images.py (author image index) have both run.
Usage:  python -m src.data.build_mkgw_split
Output: data/splits/mkgw_{train,val,test}.json, data/processed/mkgw_triples.jsonl
        and a statistics block appended to experiments/data_stats.md.
"""
import json
import random
import re
import urllib.parse
from pathlib import Path

from src.utils.config import REPO_ROOT, load_config, resolve

DATASET = "mkgw"
KGC_FILES = {"train": "train2id.txt", "val": "valid2id.txt", "test": "test2id.txt"}


def mask_name(description: str, names: list[str]) -> str:
    """Remove every occurrence of the entity's label/aliases from its description.

    Word-boundary, case-insensitive; leftover double spaces and dangling
    punctuation are collapsed. Returns "" when nothing survives.
    """
    masked = description
    for name in sorted(filter(None, names), key=len, reverse=True):
        masked = re.sub(rf"\b{re.escape(name)}\b", "", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\s{2,}", " ", masked)
    masked = re.sub(r"\s+([,;:.)])", r"\1", masked)
    masked = re.sub(r"^[\s,;:.'\"-]+|[\s,;:'\"-]+$", "", masked)
    return masked


def commons_url(file_name: str, width: int) -> str:
    quoted = urllib.parse.quote(file_name.replace(" ", "_"))
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width={width}"


def load_cache(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len({r["qid"] for r in rows}) == len(rows), "duplicate qid in cache"
    return rows


def split_entities(rows: list[dict], fractions: dict[str, float], seed: int) -> dict[str, list[dict]]:
    rows = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = round(n * fractions["train"])
    n_val = round(n * fractions["val"])
    return {"train": rows[:n_train],
            "val": rows[n_train : n_train + n_val],
            "test": rows[n_train + n_val :]}


def to_image_entry(row: dict, lang: str, width: int) -> dict:
    query = mask_name(row[f"desc_{lang}"], [row.get(f"label_{lang}")] + row.get(f"aliases_{lang}", []))
    return {
        "image_id": row["id"],
        "qid": row["qid"],
        "label": row.get(f"label_{lang}"),
        "file_name": f"{row['qid']}.jpg",
        # local file comes from the author archive; url is a Commons fallback (P18), when one exists
        "url": commons_url(row["image"], width) if row.get("image") else None,
        "captions": [{"caption_id": f"{row['qid']}_0", "text": query}],
    }


def copy_kgc_triples(native_repo: Path, out: Path) -> int:
    """One JSON line per NativE triple: head/tail entity ids, relation id, KGC split."""
    n_total = 0
    with out.open("w", encoding="utf-8") as f:
        for split, fname in KGC_FILES.items():
            lines = (native_repo / "benchmarks" / "MKG-W" / fname).read_text().splitlines()
            n = int(lines[0])
            for line in lines[1 : n + 1]:
                h, t, r = map(int, line.split())     # OpenKE order: head tail relation
                f.write(json.dumps({"h": h, "t": t, "r": r, "kgc_split": split}) + "\n")
            n_total += n
    return n_total


def main() -> None:
    cfg = load_config()
    mcfg = cfg["mkgw"]
    lang, width = mcfg["query_lang"], mcfg["image_width"]
    cache = load_cache(REPO_ROOT / mcfg["wikidata_cache"])
    author_index = json.loads(
        (REPO_ROOT / "data" / "raw" / "mkgw" / "author_images_index.json").read_text(encoding="utf-8"))

    usable, dropped = [], {"missing": 0, "no_desc": 0, "no_author_image": 0, "empty_query": 0}
    for row in cache:
        if row.get("missing"):
            dropped["missing"] += 1
        elif not row.get(f"desc_{lang}"):
            dropped["no_desc"] += 1
        elif row["qid"] not in author_index:
            dropped["no_author_image"] += 1
        elif not mask_name(row[f"desc_{lang}"], [row.get(f"label_{lang}")] + row.get(f"aliases_{lang}", [])):
            dropped["empty_query"] += 1
        else:
            usable.append(row)

    splits = split_entities(usable, mcfg["split_fractions"], cfg["seed"])
    out_dir = resolve(cfg, "splits")
    for split, rows in splits.items():
        payload = {"dataset": DATASET, "split": split,
                   "images": [to_image_entry(r, lang, width) for r in rows]}
        (out_dir / f"{DATASET}_{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    n_triples = copy_kgc_triples(Path(mcfg["native_repo"]).expanduser(),
                                 resolve(cfg, "processed") / "mkgw_triples.jsonl")

    stats = [
        f"\n## MKG-W retrieval split (built {json.dumps(dict(seed=cfg['seed'], lang=lang))})\n",
        f"- Cache: {len(cache)} entities; usable (description + author image + non-empty masked query): {len(usable)}",
        f"- Dropped: {dropped}",
        f"- Split sizes: " + ", ".join(f"{s} {len(r)}" for s, r in splits.items()),
        f"- KGC triples copied with their NativE split: {n_triples}",
    ]
    with (resolve(cfg, "experiments") / "data_stats.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(stats) + "\n")
    print("\n".join(stats))


if __name__ == "__main__":
    main()
