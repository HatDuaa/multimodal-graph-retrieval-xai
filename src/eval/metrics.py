"""Retrieval metrics shared by every experiment: Recall@k, MRR, and mean ± std over seeds.

Conventions (keep them fixed so that all numbers in the report are comparable):
- A query has one gold id, or several (e.g. an entity with many images); with several,
  the best-ranked gold counts.
- Ranks are 1-based. A gold id that is absent from the ranked list has no rank: it counts
  as a miss for every Recall@k and contributes 0 to MRR. Lists are usually cut at top-k,
  so MRR here is MRR@len(list); pass long enough lists if that matters.
- Std over seeds is the sample standard deviation (ddof=1); with one run it is 0.
"""
import json
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

DEFAULT_KS = (1, 5, 10)


def gold_rank(ranked: Sequence, gold) -> int | None:
    """1-based rank of the best-ranked gold id in `ranked`, or None if no gold id is present."""
    golds = set(gold) if isinstance(gold, Collection) and not isinstance(gold, (str, bytes)) else {gold}
    for i, item in enumerate(ranked, start=1):
        if item in golds:
            return i
    return None


def evaluate(ranked_lists: Sequence[Sequence], golds: Sequence, ks: Sequence[int] = DEFAULT_KS) -> dict:
    """Metrics over a set of queries. Returns {"n_queries", "recall@k"..., "mrr"} as floats in [0, 1]."""
    assert len(ranked_lists) == len(golds) and len(golds) > 0
    ranks = [gold_rank(r, g) for r, g in zip(ranked_lists, golds)]
    out = {"n_queries": len(golds)}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([r is not None and r <= k for r in ranks]))
    out["mrr"] = float(np.mean([0.0 if r is None else 1.0 / r for r in ranks]))
    return out


def aggregate_seeds(runs: Sequence[dict]) -> dict:
    """{"metric": {"mean", "std"}} over per-seed results of evaluate(); n_queries must agree."""
    assert len(runs) > 0
    assert len({r["n_queries"] for r in runs}) == 1, "runs were evaluated on different query sets"
    out = {"n_queries": runs[0]["n_queries"], "n_runs": len(runs)}
    for key in runs[0]:
        if key == "n_queries":
            continue
        values = np.array([r[key] for r in runs], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


def write_metrics(path: str | Path, *, method: str, dataset: str, split: str,
                  per_seed: dict[int, dict], config: dict | None = None) -> dict:
    """Write experiments/.../metrics.json in the one schema used by the whole project."""
    payload = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "per_seed": {str(s): m for s, m in per_seed.items()},
        "aggregate": aggregate_seeds(list(per_seed.values())),
        "config": config or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def format_row(agg: dict, ks: Sequence[int] = DEFAULT_KS) -> str:
    """'R@1 | R@5 | R@10 | MRR' cells as percentages, 'mean ± std', for markdown tables."""
    keys = [f"recall@{k}" for k in ks] + ["mrr"]
    return " | ".join(f"{100 * agg[k]['mean']:.2f} ± {100 * agg[k]['std']:.2f}" for k in keys)
