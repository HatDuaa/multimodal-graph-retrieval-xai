"""Is it the count term or the merged graph? Eval-time counterfactuals for the comparison examples (val, no training).

For countonly_factual_r3 (trained on count_v1), two changes at evaluation only:
- g = 0: the learned count table zeroed, so object similarities lose the count term;
- unmerged: every val image graph rebuilt from the level-1 graph and its VG boxes exactly as merge_v1
  (scripts/improve/package_merged.merged_graph) but with no IoU merge (threshold above 1), count buckets recomputed.
Per seed: val R@1 and MRR as trained and under each change, overall and on captions with an explicit count. For the three
comparison examples of experiments/improve/explanations/cases.json (seed 0): the gold image's rank and the score gap to
the best other image as trained and under each change. Writes experiments/improve/explanations/counterfactuals.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.explain_cases import ranking, val_ranks  # noqa: E402
from scripts.improve.explain_cases_countonly import EXPLICIT, OUT, RUNS, recount  # noqa: E402
from scripts.improve.package_merged import merged_graph  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

NO_MERGE = 1.01  # IoU never reaches it: same pipeline as merge_v1, nothing merged


def unmerged_graphs(split="val"):
    base = load_config()
    objects = json.loads((resolve(base, "processed") / "geometry" / f"objects_{split}.json").read_text(encoding="utf-8"))
    import scripts.improve.package_merged as package_merged
    threshold, package_merged.THRESHOLD = package_merged.THRESHOLD, NO_MERGE
    try:
        out = {}
        with (resolve(base, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                if graph["split"] == split:
                    out[int(graph["image_id"])] = recount(merged_graph((graph, objects[str(graph["image_id"])])))
    finally:
        package_merged.THRESHOLD = threshold
    return out


def gold_rank(model, store, caption_id, gold, device):
    ranked, scores, _ = ranking(model, store, caption_id, device)
    best_other = max(s for i, s in scores.items() if i != gold)
    return {"rank": ranked.index(gold) + 1, "gold_score": scores[gold], "gap_to_best_other": scores[gold] - best_other}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(load_config(REPO_ROOT / "configs" / "count_v1.yaml"), splits=("val",))
    merged = dict(store.graphs)
    unmerged = unmerged_graphs()
    assert set(unmerged) == set(merged)
    arrays = store.candidates("val")
    explicit = np.array([any(o.get("count_class", 0) in EXPLICIT for o in store.queries[str(c)]["objects"])
                         for c in arrays["caption_ids"]])
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))["comparison"]["examples"]
    result = {"rule": __doc__, "nodes": {"merged": sum(len(g["objects"]) for g in merged.values()),
                                         "unmerged": sum(len(g["objects"]) for g in unmerged.values())},
              "seeds": {}, "examples": {c["caption_id"]: {"category": c["category"]} for c in cases}}
    for seed in range(3):
        run = RUNS / f"countonly_factual_r3_seed{seed}"
        model = load_run_model(run, device, store=store)[0]
        table = model.count_table.detach().clone()
        rows = {}
        for change in ("trained", "g_zero", "unmerged"):
            store.replace_image_graphs(unmerged if change == "unmerged" else merged)
            with torch.no_grad():
                model.count_table.copy_(torch.zeros_like(table) if change == "g_zero" else table)
            _, _, rank = val_ranks(model, store, device)
            reciprocal = np.where(rank <= 50, 1 / rank, 0.)
            rows[change] = {"recall@1": float((rank == 1).mean()), "mrr": float(reciprocal.mean()),
                            "recall@1_explicit": float((rank[explicit] == 1).mean()),
                            "recall@1_no_explicit": float((rank[~explicit] == 1).mean())}
            if seed == 0:
                for case in cases:
                    result["examples"][case["caption_id"]][change] = gold_rank(model, store, case["caption_id"],
                                                                               case["gold"], device)
            print(run.name, change, json.dumps(rows[change]), flush=True)
        with torch.no_grad():
            model.count_table.copy_(table)
        store.replace_image_graphs(merged)
        result["seeds"][run.name] = rows
    (OUT / "counterfactuals.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["examples"], indent=1))


if __name__ == "__main__":
    main()
