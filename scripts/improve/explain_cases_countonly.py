"""Explanations of the improvement phase's main model: 5 success and 5 failure queries on VAL, as level 1.

Same rule and code path as scripts/explain_cases.py (level 1, main_r3_seed0), applied to countonly_factual_r3_seed0
(best val R@1 of its three seeds) on its data version count_v1. Selection: a val query is eligible when its FACTUAL
caption graph has at least one relation; with numpy default_rng(2026), 5 success (CLIP rank > 1, model rank 1), 3
broken (CLIP rank 1, model rank > 1) and 2 both wrong. Category counts are given for the three countonly seeds.

Every number comes from the ranking pass (src/explain/graph_explainer.py score / matched_pairs). Under the raw fusion
a row's channel_term (channel weight * w_i * sum_j a_ij sim_ij) is its exact part of the final score, and object rows
carry the count term g. The same part splits exactly by image part, term_j = channel weight * sum_i w_i a_ij sim_ij
(matched_pairs image_terms). The deletion test removes the image part with the largest term_j, then 10 random parts of
the same kind, and re-ranks the full top-50. (Level 1 removed the part of the largest best-pair contribution; with the
diffuse triple attention of this model, a_ij about 0.1, the best pair of a row holds only a small piece of it.) Count buckets of the image are recomputed after a
removal (src/graph/counts.py image_counts, as in data/graphs/count_v1), since the image then has one instance less.

Comparison with level 1 (main_r3, seed k against seed k): val queries the new model fixes (main_r3 rank > 1, new rank
1) and breaks (main_r3 rank 1, new rank > 1), in all and among captions with an explicit count; then examples drawn by
rule with the same rng: one fixed caption with an explicit count, one fixed caption whose gold image's top object row
matches a merged node (boxes > 1), and the bad count match 151938_1 ("A lone woman ...", woman -> hair, found by
scripts/improve/count_match_changes.py). Writes experiments/improve/explanations/cases.json and img/.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.explain_cases import PLAN, RANDOM_REPEATS, SEED, TOP_K, categories, labels, ranking, val_ranks  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.explain.graph_explainer import remove_parts, score  # noqa: E402
from src.graph.counts import CAPTION_CLASSES, image_bucket, image_counts  # noqa: E402
from src.graph.dedup import same_type  # noqa: E402
from src.models.graph_reranker import fill_known_mean  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

RUNS = REPO_ROOT / "experiments" / "improve" / "count-only"
LEVEL1 = REPO_ROOT / "experiments" / "level1" / "gat"
OUT = REPO_ROOT / "experiments" / "improve" / "explanations"
IMAGES = REPO_ROOT / "data" / "raw" / "images"
EXPLICIT = {CAPTION_CLASSES.index(c) for c in ("1", "2", "3-4", ">=5")}
LONE_WOMAN = "151938_1"


def recount(graph):
    """Count and bucket of every node from the graph as it is (after a removal), as in data/graphs/count_v1."""
    out = copy.deepcopy(graph)
    plain = [{k: v for k, v in o.items() if k not in ("count", "count_bucket")} for o in out["objects"]]
    for obj, count in zip(out["objects"], image_counts(plain, same_type)):
        obj["count"], obj["count_bucket"] = count, image_bucket(count)
    return out


@torch.no_grad()
def explain(model, store, caption_id, image_id, counted=True):
    """Rows of score(); for the raw fusion also the CLIP part and the filled part of a channel without rows."""
    result = score(model, store, caption_id, image_id)
    query, image, rel = labels(store, caption_id, image_id)
    for row in result["objects"]:
        row["query_part"] = query["objects"][row["query_index"]]["name"]
        row["image_part"] = image["objects"][row["image_index"]]["name"]
        if counted:
            row["image_boxes"] = image["objects"][row["image_index"]].get("boxes", 1)
    for row in result["triples"]:
        row["query_part"] = rel(query, row["query_index"])
        row["image_part"] = rel(image, row["image_index"])
    if model.options["fusion"] == "raw":
        batch = store.batch([caption_id], next(model.parameters()).device)
        k = batch["candidate_ids"][0].tolist().index(image_id)
        _, detail = model(batch)
        a, b, c = result["weights"]
        result["clip"] = float(batch["clip"][0, k])
        result["clip_part"] = a * result["clip"]
        result["fills"] = {key: (w * float(fill_known_mean(detail[key])[0, k])
                                 if not bool(torch.isfinite(detail[key][0, k])) else None)
                           for key, w in (("objects", b), ("triples", c))}
    return result


def deletion_test(model, store, caption_id, image_id, explanation, rng, device, counted=True):
    """Remove the image part with the largest exact term, then random parts of the same kind, and re-rank."""
    ranked, scores, _ = ranking(model, store, caption_id, device)
    parts = [(kind, r["image_index"], r["term"]) for kind in ("objects", "triples")
             for r in explanation["image_terms"][kind]]
    if not parts:
        return None
    kind, index, contribution = max(parts, key=lambda p: p[2])
    graph = store.graphs[image_id]

    def removed(kind, index):
        modified = remove_parts(graph, [index], []) if kind == "objects" else remove_parts(graph, [], [index])
        modified = recount(modified) if counted else modified
        new_ranked, new_scores, _ = ranking(model, store, caption_id, device, {image_id: modified})
        return {"rank": new_ranked.index(image_id) + 1, "score": new_scores[image_id]}
    top = removed(kind, index)
    pool = [i for i in range(len(graph["objects" if kind == "objects" else "relations"])) if i != index]
    randoms = [removed(kind, int(i)) for i in rng.choice(pool, size=min(RANDOM_REPEATS, len(pool)), replace=False)]
    _, image, rel = labels(store, caption_id, image_id)
    return {"image_id": image_id, "removed_kind": kind, "removed_index": int(index),
            "removed_part": image["objects"][index]["name"] if kind == "objects" else rel(image, index),
            "removed_contribution": contribution,
            "before": {"rank": ranked.index(image_id) + 1, "score": scores[image_id]}, "after_top_part": top,
            "after_random_parts": {"n": len(randoms), "mean_rank": float(np.mean([r["rank"] for r in randoms])) if randoms else None,
                                   "mean_score": float(np.mean([r["score"] for r in randoms])) if randoms else None}}


def thumbnail(image_id):
    target = OUT / "img" / f"{image_id}.jpg"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(IMAGES / f"{image_id}.jpg") as image:
            image = image.convert("RGB")
            image.thumbnail((256, 256))
            image.save(target, quality=85)
    return target.relative_to(OUT).as_posix()


def query_graph(store, cid, gold):
    graph = store.queries[cid]
    return {"objects": [o["name"] for o in graph["objects"]],
            "count_classes": [CAPTION_CLASSES[o.get("count_class", 0)] for o in graph["objects"]],
            "relations": [labels(store, cid, gold)[2](graph, i) for i in range(len(graph["relations"]))]}


def case_of(model, store, arrays, row, cid, category, text, rng, device, clip_rank, model_rank):
    gold = int(arrays["gold"][row])
    ranked, scores, weights = ranking(model, store, cid, device)
    top1 = ranked[0]
    clip_order = [int(i) for i in arrays["candidate_ids"][row]]
    case = {"category": category, "caption_id": cid, "caption": text[cid], "query_graph": query_graph(store, cid, gold),
            "gold": gold, "clip_rank": int(clip_rank[row]), "model_rank": int(model_rank[row]), "weights_abc": weights,
            "top": [{"rank": k + 1, "image_id": i, "score": scores[i], "clip_rank": clip_order.index(i) + 1,
                     "is_gold": i == gold, "thumbnail": thumbnail(i)} for k, i in enumerate(ranked[:TOP_K])],
            "gold_thumbnail": thumbnail(gold),
            # A gold image outside CLIP's top-50 is not in the pool the model re-ranks: nothing to explain.
            "explain_gold": explain(model, store, cid, gold) if gold in clip_order else None}
    if top1 != gold:
        case["explain_top1"] = explain(model, store, cid, top1)
    case["deletion_top1"] = deletion_test(model, store, cid, top1, case.get("explain_top1", case["explain_gold"]),
                                          rng, device)
    if top1 != gold and case["explain_gold"] is not None:
        case["deletion_gold"] = deletion_test(model, store, cid, gold, case["explain_gold"], rng, device)
    return case


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(load_config(REPO_ROOT / "configs" / "count_v1.yaml"), splits=("val",))
    level1 = GraphStore(splits=("val",))
    arrays = store.candidates("val")
    assert np.array_equal(arrays["caption_ids"], level1.candidates("val")["caption_ids"])
    assert np.array_equal(arrays["candidate_ids"], level1.candidates("val")["candidate_ids"])
    split = json.loads((REPO_ROOT / "data" / "splits" / "vg_coco_val.json").read_text(encoding="utf-8"))
    text = {c["caption_id"]: c["text"] for image in split["images"] for c in image["captions"]}
    counts, ranks = {}, {}
    for name, root, data in [(f"countonly_factual_r3_seed{k}", RUNS, store) for k in range(3)] + \
                            [(f"main_r3_seed{k}", LEVEL1, level1) for k in range(3)]:
        model, _, val = load_run_model(root / name, device, store=data)
        caption_ids, clip_rank, model_rank = val_ranks(model, data, device)
        cats = categories(clip_rank, model_rank)
        counts[name] = {**{k: int(v.sum()) for k, v in cats.items()}, "n": len(caption_ids),
                        "model_better": int((model_rank < clip_rank).sum()),
                        "model_worse": int((model_rank > clip_rank).sum()),
                        "recall@1": float((model_rank == 1).mean()), "stored_recall@1": val["recall@1"]}
        ranks[name] = model_rank
        print(name, json.dumps(counts[name]), flush=True)
    explicit = np.array([any(o.get("count_class", 0) in EXPLICIT for o in store.queries[c]["objects"]) for c in caption_ids])
    comparison = {"explicit_share": float(explicit.mean()), "pairs": {}}
    for k in range(3):
        new, old = ranks[f"countonly_factual_r3_seed{k}"], ranks[f"main_r3_seed{k}"]
        fixed, broken = (old > 1) & (new == 1), (old == 1) & (new > 1)
        comparison["pairs"][f"seed{k}"] = {"fixed": int(fixed.sum()), "broken": int(broken.sum()),
                                           "both_right": int(((old == 1) & (new == 1)).sum()),
                                           "both_wrong": int(((old > 1) & (new > 1)).sum()),
                                           "fixed_explicit": int((fixed & explicit).sum()),
                                           "broken_explicit": int((broken & explicit).sum()),
                                           "explicit": int(explicit.sum())}
    print("comparison", json.dumps(comparison), flush=True)

    model = load_run_model(RUNS / "countonly_factual_r3_seed0", device, store=store)[0]
    old_model = load_run_model(LEVEL1 / "main_r3_seed0", device, store=level1)[0]
    clip_rank = np.where(arrays["gold_rank"] > 0, arrays["gold_rank"], 51)
    new_rank, old_rank = ranks["countonly_factual_r3_seed0"], ranks["main_r3_seed0"]
    cats = categories(clip_rank, new_rank)
    eligible = np.array([len(store.queries[c]["relations"]) > 0 for c in caption_ids])
    rng = np.random.default_rng(SEED)
    cases = []
    for category, n in PLAN:
        pool = np.flatnonzero(cats[category] & eligible)
        for row in rng.choice(pool, size=n, replace=False):
            cid = caption_ids[row]
            cases.append(case_of(model, store, arrays, row, cid, category, text, rng, device, clip_rank, new_rank))
            print(category, cid, "clip", cases[-1]["clip_rank"], "model", cases[-1]["model_rank"], flush=True)

    # Comparison examples, drawn by rule after the 10 cases with the same generator.
    fixed = (old_rank > 1) & (new_rank == 1)
    picks = [("count_fixed", int(rng.choice(np.flatnonzero(fixed & explicit))))]
    merged_pool = []
    for row in rng.permutation(np.flatnonzero(fixed)):
        cid, gold = caption_ids[row], int(arrays["gold"][row])
        rows = explain(model, store, cid, gold)["objects"]
        if rows and max(rows, key=lambda r: r["channel_term"])["image_boxes"] > 1:
            merged_pool.append(int(row))
            break
    picks.append(("merge_fixed", merged_pool[0]))
    picks.append(("count_bad_match", caption_ids.index(LONE_WOMAN)))
    examples = []
    for kind, row in picks:
        cid = caption_ids[row]
        gold = int(arrays["gold"][row])
        example = case_of(model, store, arrays, row, cid, kind, text, rng, device, clip_rank, new_rank)
        old_ranked, old_scores, old_weights = ranking(old_model, level1, cid, device)
        example["level1"] = {"model_rank": int(old_rank[row]), "weights_abc": old_weights,
                             "top1": old_ranked[0], "top1_thumbnail": thumbnail(old_ranked[0]),
                             "explain_gold": explain(old_model, level1, cid, gold, counted=False)
                             if gold in arrays["candidate_ids"][row] else None,
                             "query_graph": {"objects": [o["name"] for o in level1.queries[cid]["objects"]],
                                             "relations": [labels(level1, cid, gold)[2](level1.queries[cid], i)
                                                           for i in range(len(level1.queries[cid]["relations"]))]}}
        if old_ranked[0] != gold:
            example["level1"]["explain_top1"] = explain(old_model, level1, cid, old_ranked[0], counted=False)
        examples.append(example)
        print(kind, cid, "clip", example["clip_rank"], "main_r3", old_rank[row], "countonly", new_rank[row], flush=True)
    comparison["examples"] = examples
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps({"rule": __doc__, "model": "countonly_factual_r3_seed0", "counts": counts,
                                                "eligible_share": float(eligible.mean()), "cases": cases,
                                                "comparison": comparison}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
