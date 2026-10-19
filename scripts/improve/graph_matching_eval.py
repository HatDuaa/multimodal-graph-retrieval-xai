"""Evaluate the whole-graph matching channels of graph_matching_probe.py on val (improvement step 2). CPU only.

For each channel (vectors x method):
- alone: rank the top-50 by the channel only, missing scores last; ties in CLIP order ("alone") or in a seeded
  random order ("alone_random_ties", which removes the CLIP signal hidden in ties of coarse channels);
- correlation per query with the object channel of main_factual_r3 (Pearson over candidates both score, as
  scripts/improve/channel_correlation.py); low means the channel carries information the object channel does not;
- fourth channel: fused_main_factual_r3 + w * z(channel), w swept on val and also picked by 2-fold (by gold image);
- all val, and the subset of captions with >= 2 triples sharing a node (where joint matching can matter).
Channels from the encoded parts of seed k are paired with main_factual_r3 seed k; frozen ones with every seed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks.three_channel_probe import zscore_rows  # noqa: E402
from scripts.improve.graph_matching_probe import METHODS, query_structure, structures  # noqa: E402
from src.eval.metrics import aggregate_seeds, evaluate  # noqa: E402
from src.graph.graph_matching import shared_node_pairs  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "graph-matching"
WEIGHTS = [0., .05, .1, .15, .2, .3, .4, .5, .75, 1.]
SEEDS = (0, 1, 2)


def ranked(scores, candidates, shuffle=None):
    """Descending scores, missing ones last; ties keep the CLIP order, or a seeded random order when shuffle is set."""
    scores = np.nan_to_num(scores, nan=-np.inf)
    if shuffle is not None:
        perm = np.random.default_rng(shuffle).permuted(np.tile(np.arange(scores.shape[1]), (len(scores), 1)), axis=1)
        order = np.take_along_axis(perm, np.argsort(-np.take_along_axis(scores, perm, 1), axis=1, kind="stable"), 1)
    else:
        order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1).tolist()


def metrics(scores, candidates, golds, masks, shuffle=None):
    lists = ranked(scores, candidates, shuffle)
    return {v: evaluate([r for r, m in zip(lists, mask) if m], [g for g, m in zip(golds, mask) if m]) for v, mask in masks.items()}


def row_correlation(a, b):
    out = []
    for x, y in zip(a, b):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3 and x[mask].std() > 1e-12 and y[mask].std() > 1e-12:
            out.append(np.corrcoef(x[mask], y[mask])[0, 1])
    return float(np.mean(out)) if out else float("nan"), len(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/factual_v1.yaml"))
    parser.add_argument("--vectors", nargs="+", default=["frozen", "seed0", "seed1", "seed2"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    dump_dir = resolve(load_config(), "processed") / "graph_matching"
    dumps = {k: dict(np.load(dump_dir / f"main_factual_r3_seed{k}.npz")) for k in SEEDS}
    caption_ids = [str(c) for c in dumps[0]["caption_ids"]]
    candidates, golds = dumps[0]["candidate_ids"], [int(g) for g in dumps[0]["gold"]]
    raw_queries, _ = structures(cfg)
    pair_counts = [len(shared_node_pairs(query_structure(raw_queries[c])["edges"])) for c in caption_ids]
    masks = {"all": [True] * len(caption_ids), "shared_node": [n > 0 for n in pair_counts]}
    fold = np.array([g % 2 for g in golds])
    base = {k: metrics(dumps[k]["fused"], candidates, golds, masks) for k in SEEDS}
    result = {"split": "val", "baseline": "main_factual_r3", "n": {v: int(sum(m)) for v, m in masks.items()},
              "weights": WEIGHTS, "baseline_metrics": {v: aggregate_seeds([base[k][v] for k in SEEDS]) for v in masks},
              "channels": {}}
    for vectors in args.vectors:
        for method in METHODS:
            path = dump_dir / f"scores_{vectors}_{method}.npy"
            if not path.exists():
                continue
            scores = np.load(path)
            seeds = SEEDS if vectors == "frozen" else (int(vectors[-1]),)
            alone = metrics(scores, candidates, golds, masks)
            alone_random_ties = metrics(scores, candidates, golds, masks, shuffle=0)
            correlation = [row_correlation(scores, dumps[k]["objects"]) for k in seeds]
            z, _ = zscore_rows(scores)
            sweep = {}
            for w in WEIGHTS:
                per_seed = [metrics(dumps[k]["fused"] + w * z, candidates, golds, masks) for k in seeds]
                sweep[str(w)] = {v: (aggregate_seeds([m[v] for m in per_seed]) if len(per_seed) > 1 else
                                     {key: {"mean": val, "std": 0.} for key, val in per_seed[0][v].items() if key != "n_queries"})
                                 for v in masks}
            best = max(WEIGHTS, key=lambda w: sweep[str(w)]["all"]["recall@1"]["mean"])
            cross, chosen = [], {}
            for k in seeds:
                final = dumps[k]["fused"].copy()
                for f in (0, 1):
                    train = fold != f
                    w = max(WEIGHTS, key=lambda w: evaluate(ranked((dumps[k]["fused"] + w * z)[train], candidates[train]),
                                                               [g for g, t in zip(golds, train) if t])["recall@1"])
                    final[fold == f] = (dumps[k]["fused"] + w * z)[fold == f]
                    chosen[f"seed{k}:fold{f}"] = w
                cross.append(metrics(final, candidates, golds, masks))
            result["channels"][f"{vectors}/{method}"] = {
                "finite_share": float(np.isfinite(scores).mean()), "alone": alone,
                "alone_random_ties": alone_random_ties,
                "object_correlation": [c[0] for c in correlation], "object_correlation_queries": [c[1] for c in correlation],
                "sweep": sweep, "best_weight_on_val": best, "two_fold_weights": chosen,
                "two_fold": {v: (aggregate_seeds([m[v] for m in cross]) if len(cross) > 1 else
                                 {key: {"mean": val, "std": 0.} for key, val in cross[0][v].items() if key != "n_queries"})
                             for v in masks}}
            print(vectors, method, "alone R@1", round(alone["all"]["recall@1"], 4), "corr", np.round([c[0] for c in correlation], 3),
                  "best w", best, round(sweep[str(best)]["all"]["recall@1"]["mean"], 4), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graph_matching.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
