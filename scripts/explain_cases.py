"""Level 1 explanations: pick 5 success and 5 failure queries on VAL by a fixed rule and explain them.

Everything comes from the ranking mechanism itself (src/explain/graph_explainer.py): the matched part pairs,
their attention and weights are the ones the model used to score the image. For each case a deletion test
removes the image part with the largest contribution (score_without / remove_parts) and re-ranks the whole
top-50, against removing random parts of the same kind.

Selection rule (fixed, reproducible): model = main_r3_seed0 (best-on-val checkpoint); a query is eligible when its
parsed graph has at least one relation, so both channels can be shown. With numpy default_rng(2026):
  success       5 queries where CLIP ranks the gold image > 1 and the model ranks it 1;
  broken        3 queries where CLIP ranks the gold image 1 and the model ranks it > 1;
  both_wrong    2 queries where both rank the gold image > 1.
Counts of every category are reported over all val queries for the three main_r3 seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.explain.graph_explainer import remove_parts, score  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402

RUNS = Path("experiments/level1/gat")
OUT = Path("experiments/level1/explanations")
IMAGES = Path("data/raw/images")
SEED, RANDOM_REPEATS, TOP_K = 2026, 10, 5
PLAN = (("success", 5), ("broken", 3), ("both_wrong", 2))


@torch.no_grad()
def ranking(model, store, caption_id, device, overrides=None):
    batch = store.batch([caption_id], device, overrides)
    fused, details = model(batch)
    fused = fused[0].cpu().numpy()
    order = np.argsort(-fused, kind="stable")
    ids = batch["candidate_ids"][0]
    return [int(ids[k]) for k in order], {int(ids[k]): float(fused[k]) for k in order}, details["weights"][0].cpu().tolist()


@torch.no_grad()
def val_ranks(model, store, device, batch_size=256):
    arrays = store.candidates("val")
    caption_ids = [str(c) for c in arrays["caption_ids"]]
    model_rank = np.zeros(len(caption_ids), dtype=np.int64)
    for start in range(0, len(caption_ids), batch_size):
        batch = store.batch(caption_ids[start:start + batch_size], device)
        fused, _ = model(batch)
        order = torch.argsort(fused, dim=-1, descending=True, stable=True).cpu().numpy()
        ranked = np.take_along_axis(batch["candidate_ids"], order, axis=1)
        for row, gold in enumerate(batch["gold"]):
            hits = np.flatnonzero(ranked[row] == gold)
            model_rank[start + row] = hits[0] + 1 if len(hits) else 51
    clip_rank = np.where(arrays["gold_rank"] > 0, arrays["gold_rank"], 51)
    return caption_ids, clip_rank, model_rank


def categories(clip_rank, model_rank):
    return {"success": (clip_rank > 1) & (model_rank == 1), "broken": (clip_rank == 1) & (model_rank > 1),
            "both_wrong": (clip_rank > 1) & (model_rank > 1), "both_right": (clip_rank == 1) & (model_rank == 1)}


def labels(store, caption_id, image_id, overrides=None):
    query, image = store.queries[caption_id], (overrides or {}).get(image_id, store.graphs[image_id])
    def rel(graph, index):
        names = {o["id"]: o["name"] for o in graph["objects"]}
        r = graph["relations"][index]
        return f'{names[r["subject"]]} {r["predicate"]} {names[r["object"]]}'
    return query, image, rel


def explain(model, store, caption_id, image_id):
    result = score(model, store, caption_id, image_id)
    query, image, rel = labels(store, caption_id, image_id)
    for row in result["objects"]:
        row["query_part"] = query["objects"][row["query_index"]]["name"]
        row["image_part"] = image["objects"][row["image_index"]]["name"]
    for row in result["triples"]:
        row["query_part"] = rel(query, row["query_index"])
        row["image_part"] = rel(image, row["image_index"])
    return result


def deletion_test(model, store, caption_id, image_id, explanation, rng, device):
    """Remove the top-contributing image part of image_id, then random parts of the same kind, and re-rank."""
    ranked, scores, _ = ranking(model, store, caption_id, device)
    parts = [("objects", r["image_index"], r["contribution"]) for r in explanation["objects"]] + \
            [("triples", r["image_index"], r["contribution"]) for r in explanation["triples"]]
    if not parts:
        return None
    kind, index, contribution = max(parts, key=lambda p: p[2])
    graph = store.graphs[image_id]
    def removed(kind, index):
        modified = remove_parts(graph, [index], []) if kind == "objects" else remove_parts(graph, [], [index])
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(splits=("val",))
    arrays = store.candidates("val")
    split = json.loads(Path("data/splits/vg_coco_val.json").read_text(encoding="utf-8"))
    text = {c["caption_id"]: c["text"] for image in split["images"] for c in image["captions"]}
    counts = {}
    for seed in (0, 1, 2):
        model, _, _ = load_run_model(RUNS / f"main_r3_seed{seed}", device)
        caption_ids, clip_rank, model_rank = val_ranks(model, store, device)
        cats = categories(clip_rank, model_rank)
        counts[f"main_r3_seed{seed}"] = {**{k: int(v.sum()) for k, v in cats.items()}, "n": len(caption_ids),
                                         "model_better": int((model_rank < clip_rank).sum()),
                                         "model_worse": int((model_rank > clip_rank).sum())}
        if seed == 0:
            chosen = (model, caption_ids, clip_rank, model_rank, cats)
    model, caption_ids, clip_rank, model_rank, cats = chosen
    eligible = np.array([len(store.queries[c]["relations"]) > 0 for c in caption_ids])
    rng = np.random.default_rng(SEED)
    cases = []
    for category, n in PLAN:
        pool = np.flatnonzero(cats[category] & eligible)
        for row in rng.choice(pool, size=n, replace=False):
            cid = caption_ids[row]
            gold = int(arrays["gold"][row])
            ranked, scores, weights = ranking(model, store, cid, device)
            top1 = ranked[0]
            clip_order = [int(i) for i in arrays["candidate_ids"][row]]
            case = {"category": category, "caption_id": cid, "caption": text[cid],
                    "query_graph": {"objects": [o["name"] for o in store.queries[cid]["objects"]],
                                    "relations": [labels(store, cid, gold)[2](store.queries[cid], i)
                                                  for i in range(len(store.queries[cid]["relations"]))]},
                    "gold": gold, "clip_rank": int(clip_rank[row]), "model_rank": int(model_rank[row]),
                    "weights_abc": weights,
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
            cases.append(case)
            print(category, cid, "clip", case["clip_rank"], "model", case["model_rank"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps({"rule": __doc__, "model": "main_r3_seed0", "counts": counts,
                                                "eligible_share": float(eligible.mean()), "cases": cases},
                                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
