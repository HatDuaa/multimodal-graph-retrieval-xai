"""Val results of the no-edges ablation by caption group (the groups of experiment 7), next to the full models.

Models: CLIP (level-1 top-50 order); countonly_factual_r3 and countonly_noedges_r3 on count_v1; main_r3 and
no_edges_r3 on level-1 data (3 seeds each, checkpoint_best). Groups: caption_groups of
scripts/improve/alpha_groups_countonly.py (explicit count, relations, object count). Per group: N, R@1 and MRR
(mean +- std over seeds); countonly - noedges and main_r3 - no_edges_r3 with a caption-level paired bootstrap 95% CI
(seed-averaged hits, 2,000 resamples, rng 2026) and Welch's t-test over the 3 + 3 seeds; and the edges' share of the
gain over CLIP, (full - no edges) / (full - CLIP), for each pair. Every run's R@1 must equal its metrics.json.
Writes experiments/improve/no-edges/groups.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.explain_cases import val_ranks  # noqa: E402
from scripts.improve.alpha_groups_countonly import bootstrap_ci, caption_groups, mean_std  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

IMPROVE, LEVEL1 = REPO_ROOT / "experiments" / "improve", REPO_ROOT / "experiments" / "level1" / "gat"
MODELS = {"countonly": (IMPROVE / "count-only", "countonly_factual_r3", "count_v1"),
          "noedges": (IMPROVE / "no-edges", "countonly_noedges_r3", "count_v1"),
          "main_r3": (LEVEL1, "main_r3", "level1"),
          "no_edges_r3": (LEVEL1, "no_edges_r3", "level1")}
PAIRS = {"countonly_minus_noedges": ("countonly", "noedges"), "main_r3_minus_no_edges_r3": ("main_r3", "no_edges_r3")}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stores = {"count_v1": GraphStore(load_config(REPO_ROOT / "configs" / "count_v1.yaml"), splits=("val",)),
              "level1": GraphStore(splits=("val",))}
    arrays = stores["count_v1"].candidates("val")
    assert np.array_equal(arrays["caption_ids"], stores["level1"].candidates("val")["caption_ids"])
    clip_rank = np.where(arrays["gold_rank"] > 0, arrays["gold_rank"], 51)
    ranks = {}
    for name, (root, prefix, data) in MODELS.items():
        ranks[name] = []
        for seed in range(3):
            run = root / f"{prefix}_seed{seed}"
            model, _, saved = load_run_model(run, device, store=stores[data])
            rank = val_ranks(model, stores[data], device)[2]
            assert abs((rank == 1).mean() - saved["recall@1"]) < 1e-9, (run.name, (rank == 1).mean(), saved["recall@1"])
            ranks[name].append(rank)
            print(run.name, round(100 * (rank == 1).mean(), 2), flush=True)
    hit = {name: np.stack([r == 1 for r in rs]).astype(np.float64) for name, rs in ranks.items()}
    recip = {name: np.stack([np.where(r <= 50, 1 / r, 0.) for r in rs]) for name, rs in ranks.items()}
    hit["clip"], recip["clip"] = (clip_rank == 1).astype(np.float64)[None], np.where(clip_rank <= 50, 1 / clip_rank, 0.)[None]
    rng = np.random.default_rng(2026)
    result = {"rule": __doc__, "groups": {}}
    for group, mask in caption_groups(stores["count_v1"], arrays).items():
        if not mask.any():
            continue
        row = {"n": int(mask.sum()), "share": float(mask.mean())}
        for name in ("clip", *MODELS):
            r1, mrr = 100 * hit[name][:, mask].mean(1), 100 * recip[name][:, mask].mean(1)
            row[name] = {"recall@1": mean_std(r1) if len(r1) > 1 else [float(r1[0]), 0.],
                         "mrr": mean_std(mrr) if len(mrr) > 1 else [float(mrr[0]), 0.], "recall@1_per_seed": r1.tolist()}
        for pair, (full, ablated) in PAIRS.items():
            diff = hit[full][:, mask].mean(0) - hit[ablated][:, mask].mean(0)
            welch = stats.ttest_ind(row[full]["recall@1_per_seed"], row[ablated]["recall@1_per_seed"], equal_var=False)
            gain = row[full]["recall@1"][0] - row["clip"]["recall@1"][0]
            row[pair] = {"recall@1": 100 * float(diff.mean()), "ci95": [100 * x for x in bootstrap_ci(diff, rng)],
                         "welch_p": float(welch.pvalue), "mrr": row[full]["mrr"][0] - row[ablated]["mrr"][0],
                         "gain_over_clip": gain,
                         # undefined when the full model does not beat CLIP on the group
                         "edge_share_of_gain": 100 * float(diff.mean()) / gain if gain > 0 else None}
        result["groups"][group] = row
        print(group, row["n"], json.dumps({name: round(row[name]["recall@1"][0], 2) for name in ("clip", *MODELS)}),
              json.dumps({pair: [round(row[pair]["recall@1"], 2), [round(x, 2) for x in row[pair]["ci95"]],
                                 round(row[pair]["welch_p"], 4), round(row[pair]["edge_share_of_gain"] or 0, 2)]
                          for pair in PAIRS}), flush=True)
    (IMPROVE / "no-edges" / "groups.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
