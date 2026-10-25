"""Build the Vietnamese MMKG (`vnmmkg`) from the raw cache and the review decisions (spec section 4, step 5).

No network. Inputs: data/raw/vnmmkg/ (steps 1-3) and vietnamese-mmkg/review_*.json (step 4).
1. Landmark candidates (ranked by sitelinks, then QID) keep their reviewed images: a landmark
   needs at least `min_images` kept images and keeps at most `max_images` (P18 first, then
   by file title). An image already used by a higher-ranked landmark is not reused.
2. Triples are proposed from Wikidata, infobox and the two rules (src/data/vnmmkg_triples.py),
   then filtered by review: infobox/derived ones need an explicit "keep".
3. Context entities are capped at `max_entities` in total, by degree then QID.
4. Auto queries: masked lead sentences (src/data/vnmmkg_text.py).
5. Landmarks are split 50/50 into val/test with the config seed (setting A1 of the spec).
Every output is sorted and written as UTF-8/LF, so rerunning on the same cache gives
byte-identical files.

Usage:  python -m src.data.build_vnmmkg
Output: vietnamese-mmkg/{entities,triples,images,queries_auto}.jsonl, data/splits/vnmmkg_{val,test}.json,
        data/raw/vnmmkg/images/<image_id>.jpg, a block in experiments/data_stats.md.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.data.vnmmkg_cache import infobox_rows, landmark_candidates, load_commons, load_link_qids, load_viwiki
from src.data.vnmmkg_text import contains_name, make_auto_queries, name_variants
from src.data.vnmmkg_triples import (cap_entities, categorize, gold_from_pattern, propose_triples, triple_key,
                                     validate_periods)
from src.data.vnmmkg_wikidata import qid_key, truthy
from src.utils.config import REPO_ROOT, load_config, resolve

DATASET = "vnmmkg"
STATS_BEGIN, STATS_END = "<!-- vnmmkg:begin -->", "<!-- vnmmkg:end -->"
SCALE = {"entities": (50, 150), "triples": (300, 1000), "images": (200, 500)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def read_decisions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["decisions"] if path.exists() else {}


def select_images(ranked: list[str], commons: dict, decisions: dict, min_images: int,
                  max_images: int) -> tuple[dict[str, list[dict]], dict[str, int]]:
    selected, stats, used = {}, Counter(), set()
    for qid in ranked:
        kept = []
        for c in commons.get(qid, {}).get("candidates", []):
            d = decisions.get(c["file_title"])
            if d is None:
                stats["images_undecided"] += 1
            elif not d["keep"]:
                stats["images_rejected"] += 1
            elif c["file_title"] in used:
                stats["images_shared_with_earlier_landmark"] += 1
            else:
                kept.append(c)
        if len(kept) >= min_images:
            selected[qid] = kept[:max_images]
            used.update(c["file_title"] for c in selected[qid])
            stats["images_over_max"] += len(kept) - len(selected[qid])
        else:
            stats["landmarks_too_few_images"] += 1
    return selected, dict(sorted(stats.items()))


def apply_triple_review(triples: list[dict], decisions: dict) -> tuple[list[dict], dict[str, int]]:
    kept, stats = [], Counter()
    for t in triples:
        d = decisions.get(triple_key(t))
        if d is None and t["source"] != "wikidata":
            stats[f"{t['source']}_undecided"] += 1
        elif d is not None and not d["keep"]:
            stats[f"{t['source']}_rejected"] += 1
        else:
            kept.append({**t, "reviewed": d is not None})
    return kept, dict(sorted(stats.items()))


def split_landmarks(landmarks: list[str], seed: int, val_fraction: float) -> dict[str, list[str]]:
    order = sorted(landmarks, key=qid_key)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    return {"val": sorted(order[:n_val], key=qid_key), "test": sorted(order[n_val:], key=qid_key)}


def resolve_hand_queries(queries: list[dict], triples: list[dict], entity_rows: list[dict], cfg_v: dict,
                         allow_update: bool) -> tuple[list[dict], dict]:
    """Compute gold_qids of indirect queries from their pattern and validate the whole set (spec 4.5)."""
    landmarks = [e["qid"] for e in entity_rows if e["kind"] == "landmark"]
    by_qid = {e["qid"]: e for e in entity_rows}
    non_visual = set(cfg_v["non_visual_pids"])
    problems, out, seen, sizes = [], [], set(), Counter()
    for q in queries:
        qid, group, pattern = q["query_id"], q["group"], [list(p) for p in q.get("pattern") or []]
        if qid in seen:
            problems.append(f"{qid}: duplicate query_id")
        seen.add(qid)
        if not q["text"].strip():
            problems.append(f"{qid}: empty text")
        gold = q.get("gold_qids")
        if group == "direct":
            if pattern:
                problems.append(f"{qid}: a direct query has no pattern")
            if not gold or len(gold) != 1 or gold[0] not in by_qid or by_qid[gold[0]]["kind"] != "landmark":
                problems.append(f"{qid}: a direct query needs exactly one landmark in gold_qids")
        elif group == "indirect":
            if not pattern or not all(len(p) == 2 for p in pattern):
                problems.append(f"{qid}: an indirect query needs a pattern of [pid, qid] pairs")
            elif not {p[0] for p in pattern} & non_visual:
                problems.append(f"{qid}: pattern needs at least one non-visual relation {sorted(non_visual)}")
            computed = gold_from_pattern(pattern, triples, landmarks)
            if not computed:
                problems.append(f"{qid}: pattern has no gold landmark")
            elif len(computed) > cfg_v["max_pattern_golds"]:
                problems.append(f"{qid}: too many gold landmarks ({len(computed)}); narrow the pattern")
            if gold is not None and gold != computed and not allow_update:
                problems.append(f"{qid}: gold changed {gold} -> {computed} (locked set; rerun with --update-hand-gold "
                                "only if Khang approves)")
            gold = computed
            sizes[str(len(computed))] += 1
        else:
            problems.append(f"{qid}: group must be direct or indirect")
        for g in gold or []:
            e = by_qid.get(g)
            if e and contains_name(q["text"], name_variants(e["label_vi"], e["aliases_vi"], e["viwiki_title"],
                                                            cfg_v["type_prefixes"])):
                problems.append(f"{qid}: text names its own answer {g}")
        out.append({"query_id": qid, "group": group, "text": q["text"], "pattern": pattern, "gold_qids": gold})
    if problems:
        raise ValueError("hand-written queries:\n" + "\n".join(problems))
    stats = {"direct": sum(q["group"] == "direct" for q in out), "indirect": sum(q["group"] == "indirect" for q in out),
             "indirect_gold_sizes": dict(sorted(sizes.items()))}
    return out, stats


def _entity_row(qid: str, kind: str, category: str, e: dict, page: dict | None) -> dict:
    return {"qid": qid, "kind": kind, "category": category, "label_vi": e["label_vi"], "aliases_vi": e["aliases_vi"],
            "label_en": e["label_en"], "desc_vi": e["desc_vi"], "viwiki_title": e["viwiki_title"],
            "viwiki_revid": page["revid"] if page else None,
            "commons_category": (truthy(e, "P373") or [None])[0], "sitelinks": e["sitelinks"]}


def _split_payload(name: str, qids: set[str], images: list[dict], queries: list[dict]) -> dict:
    return {"dataset": DATASET, "split": name,
            "images": [{"image_id": im["image_id"], "qid": im["qid"], "file_name": f"{im['image_id']}.jpg",
                        "url": im["thumb_url"], "width": im["width"], "height": im["height"], "captions": []}
                       for im in images if im["qid"] in qids],
            "queries": [{"query_id": q["query_id"], "text": q["text"], "gold_qid": q["qid"]}
                        for q in queries if q["qid"] in qids]}


def _counts(values) -> dict:
    return dict(sorted(Counter(values).items()))


def build_dataset(cfg_v: dict, raw_dir: Path, review_dir: Path, seed: int) -> dict:
    validate_periods(cfg_v["periods"])
    ranked, entities = landmark_candidates(raw_dir, cfg_v)
    viwiki, commons = load_viwiki(raw_dir), load_commons(raw_dir)
    selected, image_stats = select_images(ranked, commons, read_decisions(review_dir / "review_images.json"),
                                          cfg_v["min_images"], cfg_v["max_images"])
    landmarks = [q for q in ranked if q in selected and q in viwiki]

    proposed, proposal_counts = propose_triples(landmarks, entities, infobox_rows(viwiki, cfg_v),
                                                load_link_qids(raw_dir), cfg_v)
    reviewed, review_stats = apply_triple_review(proposed, read_decisions(review_dir / "review_triples.json"))
    context, triples, n_drop_ent, n_drop_tri = cap_entities(landmarks, reviewed, cfg_v["max_entities"])

    landmark_set = set(landmarks)
    entity_rows = ([_entity_row(q, "landmark", "landmark", entities[q], viwiki[q]) for q in landmarks]
                   + [_entity_row(q, "context", categorize(q, landmark_set, entities, triples, cfg_v), entities[q], None)
                      for q in context])

    images, copies = [], []
    for q in landmarks:
        for c in selected[q]:
            images.append({"image_id": len(images), "qid": q, "file_title": c["file_title"],
                           "source_url": c["source_url"], "thumb_url": c["thumb_url"], "width": c["width"],
                           "height": c["height"], "license": c["license"], "license_url": c["license_url"],
                           "artist": c["artist"], "sha256": c["sha256"]})
            copies.append((c["local_file"], images[-1]["image_id"], c["sha256"]))

    queries, query_drops = [], Counter()
    for q in landmarks:
        e = entities[q]
        names = name_variants(e["label_vi"], [*e["aliases_vi"], *([e["label_en"]] if e["label_en"] else [])],
                              e["viwiki_title"], cfg_v["type_prefixes"])
        rows, drops = make_auto_queries(q, viwiki[q]["extract"], viwiki[q]["revid"], names,
                                        cfg_v["max_query_sentences"], cfg_v["min_query_words"])
        queries += rows
        query_drops.update(drops)
        if not rows:
            query_drops["landmarks_without_query"] += 1

    split_qids = split_landmarks(landmarks, seed, cfg_v["val_fraction"])
    splits = {name: _split_payload(name, set(qids), images, queries) for name, qids in split_qids.items()}
    stats = {
        "seed": seed, "valid_candidates": len(ranked),
        "entities": len(entity_rows), "landmarks": len(landmarks), "context_entities": len(context),
        "categories": _counts(r["category"] for r in entity_rows),
        "triples": len(triples), "triples_by_source": _counts(t["source"] for t in triples),
        "triples_by_relation": _counts(t["r"] for t in triples),
        "triple_proposal_counters": proposal_counts, "triple_review": review_stats,
        "dropped_by_entity_cap": {"entities": n_drop_ent, "triples": n_drop_tri},
        "images": len(images), "image_selection": image_stats, "licenses": _counts(im["license"] for im in images),
        "queries_auto": len(queries), "query_drops": dict(sorted(query_drops.items())),
        "split": {name: {"landmarks": len(qids), "images": len(splits[name]["images"]),
                         "queries": len(splits[name]["queries"])} for name, qids in split_qids.items()},
    }
    return {"landmarks": landmarks, "entities": entity_rows, "triples": triples, "images": images,
            "queries_auto": queries, "splits": splits, "copies": copies, "stats": stats}


def check_and_copy_images(copies: list[tuple[str, int, str]], candidates_dir: Path, images_dir: Path,
                          dry_run: bool) -> None:
    """Verify every reviewed file is unchanged; with dry_run=False copy it to <image_id>.jpg."""
    expected = set()
    for local, image_id, sha in copies:
        src = candidates_dir / local
        if not src.exists():
            raise FileNotFoundError(f"candidate image missing: {local}")
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"sha256 mismatch for {local}: the file changed after it was fetched and reviewed")
        if dry_run:
            continue
        dest = images_dir / f"{image_id}.jpg"
        expected.add(dest.name)
        images_dir.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_bytes() != data:
            dest.write_bytes(data)
    if not dry_run:
        for stale in images_dir.glob("*.jpg"):
            if stale.name not in expected:
                stale.unlink()


def scale_problems(stats: dict) -> list[str]:
    return [f"{k} = {stats[k]} outside {lo}-{hi}" for k, (lo, hi) in SCALE.items() if not lo <= stats[k] <= hi]


def stats_markdown(stats: dict) -> str:
    lines = [STATS_BEGIN, "## MMKG tiếng Việt (`vnmmkg`)", "",
             f"Sinh bởi `python -m src.data.build_vnmmkg` (seed {stats['seed']}). Không sửa tay.", "",
             "| Thành phần | Số lượng | Khung của đề |", "|---|---|---|",
             f"| Thực thể | {stats['entities']} (di tích {stats['landmarks']}, ngữ cảnh {stats['context_entities']}) | 50–150 |",
             f"| Triple | {stats['triples']} | 300–1 000 |",
             f"| Ảnh đã duyệt | {stats['images']} | 200–500 |", ""]
    for key in ["valid_candidates", "categories", "triples_by_source", "triples_by_relation",
                "triple_proposal_counters", "triple_review", "dropped_by_entity_cap", "image_selection",
                "licenses", "queries_auto", "query_drops", "split", "hand_queries"]:
        if key in stats:
            lines.append(f"- `{key}`: {json.dumps(stats[key], ensure_ascii=False)}")
    lines.append(STATS_END)
    return "\n".join(lines) + "\n"


def replace_stats_block(md_path: Path, block: str) -> None:
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    if STATS_BEGIN in text:
        start, end = text.index(STATS_BEGIN), text.index(STATS_END) + len(STATS_END)
        text = text[:start] + block.rstrip("\n") + text[end:]
    else:
        text = (text.rstrip("\n") + "\n\n" if text else "") + block
    md_path.write_text(text, encoding="utf-8", newline="\n")


def run_build(cfg_v: dict, raw_dir: Path, out_dir: Path, splits_dir: Path, stats_md: Path, seed: int,
              update_hand_gold: bool = False) -> dict:
    result = build_dataset(cfg_v, raw_dir, out_dir, seed)
    check_and_copy_images(result["copies"], raw_dir / "candidates", raw_dir / "images", dry_run=True)
    hand_path = out_dir / "queries_hand.json"
    if hand_path.exists():
        hand, result["stats"]["hand_queries"] = resolve_hand_queries(
            json.loads(hand_path.read_text(encoding="utf-8")), result["triples"], result["entities"], cfg_v,
            allow_update=update_hand_gold)
    write_jsonl(out_dir / "entities.jsonl", result["entities"])
    write_jsonl(out_dir / "triples.jsonl", result["triples"])
    write_jsonl(out_dir / "images.jsonl", result["images"])
    write_jsonl(out_dir / "queries_auto.jsonl", result["queries_auto"])
    for name, payload in result["splits"].items():
        write_json(splits_dir / f"{DATASET}_{name}.json", payload)
    if hand_path.exists():
        write_json(hand_path, hand)
    check_and_copy_images(result["copies"], raw_dir / "candidates", raw_dir / "images", dry_run=False)
    replace_stats_block(stats_md, stats_markdown(result["stats"]))
    return result["stats"]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update-hand-gold", action="store_true",
                    help="accept recomputed gold of indirect hand queries (only before the set is locked)")
    args = ap.parse_args()
    cfg = load_config()
    v = cfg["vnmmkg"]
    stats = run_build(v, REPO_ROOT / v["raw_dir"], REPO_ROOT / v["out_dir"], resolve(cfg, "splits"),
                      resolve(cfg, "experiments") / "data_stats.md", cfg["seed"], args.update_hand_gold)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    problems = scale_problems(stats)
    print("SCALE OK" if not problems else "SCALE OUT OF RANGE: " + "; ".join(problems))


if __name__ == "__main__":
    main()
