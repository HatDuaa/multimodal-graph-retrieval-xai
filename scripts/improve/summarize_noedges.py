"""Summarize the no-edges ablation of countonly_factual_r3 next to level 1's no_edges_r3 (validation only).

Per run: best epoch (selection by val R@1, as in training), its val R@1/5/10/MRR, epochs run, training seconds (sum
of the epochs' seconds in log.jsonl) and peak GPU memory. Per pair (full model, no edges): mean +- std over the three
seeds, the difference and Welch's t-test per metric. Writes experiments/improve/no-edges/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import REPO_ROOT  # noqa: E402

KEYS = ("recall@1", "recall@5", "recall@10", "mrr")
PAIRS = {"countonly": (REPO_ROOT / "experiments" / "improve" / "count-only", "countonly_factual_r3",
                       REPO_ROOT / "experiments" / "improve" / "no-edges", "countonly_noedges_r3"),
         "level1": (REPO_ROOT / "experiments" / "level1" / "gat", "main_r3",
                    REPO_ROOT / "experiments" / "level1" / "gat", "no_edges_r3")}


def run_summary(run):
    best = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["best"]
    log = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"best": {k: 100 * best[k] for k in KEYS}, "best_epoch": best["epoch"], "epochs_run": len(log),
            "train_seconds": sum(e["seconds"] for e in log),
            "seconds_per_epoch": float(np.mean([e["seconds"] for e in log])),
            "gpu_memory_peak_bytes": max(e.get("gpu_memory_peak_bytes", 0) for e in log),
            "weights_mean": best["weights_mean"]}


def main() -> None:
    out = {}
    for name, (full_root, full, ablated_root, ablated) in PAIRS.items():
        runs = {label: [run_summary(root / f"{prefix}_seed{k}") for k in range(3)]
                for label, root, prefix in (("full", full_root, full), ("no_edges", ablated_root, ablated))}
        pair = {"runs": {full: runs["full"], ablated: runs["no_edges"]}, "metrics": {}}
        for key in KEYS:
            a = np.array([r["best"][key] for r in runs["full"]])
            b = np.array([r["best"][key] for r in runs["no_edges"]])
            test = stats.ttest_ind(b, a, equal_var=False)
            pair["metrics"][key] = {"full": [float(a.mean()), float(a.std(ddof=1))],
                                    "no_edges": [float(b.mean()), float(b.std(ddof=1))],
                                    "difference": float(b.mean() - a.mean()), "welch_t": float(test.statistic),
                                    "welch_p": float(test.pvalue)}
        out[name] = pair
        print(name, json.dumps({k: [round(x, 2) for x in v["full"] + v["no_edges"]] + [round(v["difference"], 2), round(v["welch_p"], 4)]
                                for k, v in pair["metrics"].items()}), flush=True)
    (REPO_ROOT / "experiments" / "improve" / "no-edges" / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
