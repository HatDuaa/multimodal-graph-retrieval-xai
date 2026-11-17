"""Compare the scene ranker (Nhi's personal repo, test run 0070) with countonly_factual_r3 on test.

The scene ranker was trained and tested once in a separate repository on the same VG ∩ COCO
Karpathy split (2,138 test images; 10,690 test captions there against 10,696 here). Its
frozen test ranks are in experiments/improve/scene-ranker/personal-repo/. This script
recomputes R@1/5/10 and both MRR conventions from those ranks (full-gallery MRR as reported
there, and MRR with ranks beyond 50 counted as 0 as reported here), then a Welch t-test per
metric against the three countonly seeds. Nothing is retrained or re-scored.

Usage:
    python scripts/improve/compare_scene_ranker.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PERSONAL = ROOT / "experiments/improve/scene-ranker/personal-repo"
METRICS = ["recall@1", "recall@5", "recall@10", "mrr"]


def rank_metrics(ranks, top=50):
    """R@k on full-gallery ranks, MRR over the full gallery and with ranks beyond ``top`` as 0."""
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "n_queries": int(len(ranks)),
        "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float(np.where(ranks <= top, 1 / ranks, 0).mean()),
        "mrr_full_gallery": float((1 / ranks).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", type=Path, default=PERSONAL / "test_eval_0070_test_ranks.npz")
    parser.add_argument(
        "--countonly",
        type=Path,
        default=ROOT / "experiments/improve/test/metrics_countonly_factual_r3.json",
    )
    parser.add_argument("--out", type=Path, default=PERSONAL / "comparison_test.json")
    args = parser.parse_args()

    with np.load(args.ranks) as archive:
        seeds = sorted(k for k in archive.files if k.startswith("ranker_"))
        scene = {k: rank_metrics(archive[k]) for k in seeds}
        clip = rank_metrics(archive["clip"])
    countonly = json.loads(args.countonly.read_text())
    group = [countonly["per_seed"][s] for s in sorted(countonly["per_seed"])]

    comparison = {}
    for metric in METRICS:
        a = np.array([scene[k][metric] for k in seeds])
        b = np.array([g[metric] for g in group])
        test = stats.ttest_ind(a, b, equal_var=False)
        comparison[metric] = {
            "scene_ranker": {"mean": float(a.mean()), "std": float(a.std(ddof=1))},
            "countonly_factual_r3": {"mean": float(b.mean()), "std": float(b.std(ddof=1))},
            "difference": float(a.mean() - b.mean()),
            "welch_t": float(test.statistic),
            "welch_p": float(test.pvalue),
        }
    result = {
        "split": "test",
        "note": (
            "Scene ranker: personal repo test run 0070, 10,690 captions; countonly: this repo, "
            "10,696 captions. Same 2,138-image test gallery and Karpathy split. 'mrr' counts "
            "ranks beyond 50 as 0, as this repo does."
        ),
        "clip_scene_ranker_repo": clip,
        "scene_ranker_per_seed": scene,
        "comparison": comparison,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"CLIP (personal repo): R@1 {100 * clip['recall@1']:.2f}  MRR {100 * clip['mrr']:.2f}")
    for metric, row in comparison.items():
        s, c = row["scene_ranker"], row["countonly_factual_r3"]
        print(
            f"{metric:10s} scene {100 * s['mean']:.2f} ± {100 * s['std']:.2f}  "
            f"countonly {100 * c['mean']:.2f} ± {100 * c['std']:.2f}  "
            f"diff {100 * row['difference']:+.2f}  p {row['welch_p']:.3f}"
        )


if __name__ == "__main__":
    main()
