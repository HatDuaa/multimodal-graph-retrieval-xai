"""Alpha sweep (experiment 6) and results by caption group (experiment 7) for countonly_factual_r3, on val only.

Channel scores of every val caption's 50 candidates are computed once per seed (in memory, nothing dumped). The raw
fusion is linear in them, so every alpha is re-fused with numpy exactly as fuse_raw does: a missing channel score takes
the mean of the query's known ones (0 when none), and fused = a * clip + b * S_obj + c * S_tri. Sweep: a = alpha for
every caption, b and c keep the learned per-caption ratio, b' = (1 - alpha) b / (b + c), c' = (1 - alpha) c / (b + c).
The learned weights re-fused this way must reproduce each run's val metrics.

Groups are defined once from the FACTUAL caption graphs of count_v1, as the ranking sees them
(GraphStore._graph_parts(query=True): pronouns dropped, empty labels masked), and used for every model:
explicit count (a scorable node of class 1 / 2 / 3-4 / >=5) or not; at least one relation or objects only; number of
scorable objects 0 / 1 / 2 / >=3. Per group: N, R@1 and MRR of CLIP (level-1 top-50 order), main_r3 and countonly
(3 seeds each, mean +- std), countonly - CLIP and countonly - main_r3 with a caption-level paired bootstrap 95% CI
(seed-averaged hits, 2,000 resamples, rng 2026) and, for countonly - main_r3, Welch's t-test over the 3 + 3 seeds.
Writes experiments/improve/alpha-groups/{alpha_sweep.json, groups.json, alpha_sweep.png}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.explain_cases import val_ranks  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.graph.counts import CAPTION_CLASSES  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

RUNS = REPO_ROOT / "experiments" / "improve" / "count-only"
LEVEL1 = REPO_ROOT / "experiments" / "level1"
OUT = REPO_ROOT / "experiments" / "improve" / "alpha-groups"
ALPHAS = [round(0.1 * k, 1) for k in range(11)]
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
EXPLICIT = {CAPTION_CLASSES.index(c) for c in ("1", "2", "3-4", ">=5")}


@torch.no_grad()
def channel_scores(model, store, device, batch_size=256):
    arrays = store.candidates("val")
    caption_ids = [str(c) for c in arrays["caption_ids"]]
    out = {"clip": [], "objects": [], "triples": [], "weights": [], "fused": []}
    for start in range(0, len(caption_ids), batch_size):
        batch = store.batch(caption_ids[start:start + batch_size], device)
        fused, detail = model(batch)
        out["clip"].append(batch["clip"].cpu().numpy())
        for key in ("objects", "triples", "weights"):
            out[key].append(detail[key].cpu().numpy())
        out["fused"].append(fused.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def fill_known_mean(values):
    known = np.isfinite(values)
    mean = np.where(known, values, 0.).sum(-1, keepdims=True) / np.maximum(known.sum(-1, keepdims=True), 1)
    return np.where(known, values, mean)


def fuse(scores, a, b, c):
    return a[:, None] * scores["clip"] + b[:, None] * fill_known_mean(scores["objects"]) + \
        c[:, None] * fill_known_mean(scores["triples"])


def metrics_of(fused, candidates, gold):
    order = np.argsort(-fused, axis=1, kind="stable")
    return evaluate(np.take_along_axis(candidates, order, axis=1).tolist(), gold.tolist())


def ranks_of(fused, candidates, gold):
    order = np.argsort(-fused, axis=1, kind="stable")
    hits = np.take_along_axis(candidates, order, axis=1) == gold[:, None]
    return np.where(hits.any(1), hits.argmax(1) + 1, 51)


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return [float(values.mean()), float(values.std(ddof=1))]


def bootstrap_ci(diff, rng, resamples=2000, chunk=200):
    means = []
    for _ in range(resamples // chunk):
        idx = rng.integers(0, len(diff), size=(chunk, len(diff)))
        means.append(diff[idx].mean(1))
    means = np.concatenate(means)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def shared_grid(level1):
    """Alpha values on every level-1 grid (each grid also holds its run's own learned alpha)."""
    return sorted(set.intersection(*({g["a"] for g in run["grid"]} for run in level1)))


def plot(sweep, level1, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    alphas = np.array(ALPHAS)
    mean = np.array([sweep["grid"][str(a)]["recall@1"][0] for a in ALPHAS])
    std = np.array([sweep["grid"][str(a)]["recall@1"][1] for a in ALPHAS])
    ax = axes[0]
    ax.plot(alphas, mean, marker="o", color="#2563eb", label="countonly, a = α (mean of 3 seeds)")
    ax.fill_between(alphas, mean - std, mean + std, color="#2563eb", alpha=.2)
    learned = sweep["learned"]["recall@1"][0]
    ax.axhline(learned, color="#ea580c", linestyle="--", label=f"learned per-caption a: {learned:.2f}")
    ax.axhline(sweep["clip_only"]["recall@1"], color="#6b7280", linestyle=":", label=f"CLIP only: {sweep['clip_only']['recall@1']:.2f}")
    ax.set(title="countonly_factual_r3 (raw fusion)", xlabel="α = weight of the raw CLIP cosine", ylabel="val R@1 (%)")
    ax.legend(fontsize=8, loc="lower center")
    ax = axes[1]
    grid = shared_grid(level1)
    l1 = np.array([[100 * next(g["recall@1"] for g in run["grid"] if g["a"] == a) for a in grid] for run in level1])
    ax.plot(grid, l1.mean(0), marker="o", color="#9333ea", label="fixed_weights_r3, a = α (mean of 3 seeds)")
    ax.fill_between(grid, l1.mean(0) - l1.std(0, ddof=1), l1.mean(0) + l1.std(0, ddof=1), color="#9333ea", alpha=.2)
    l1_learned = np.mean([100 * run["learned_a_val"]["recall@1"] for run in level1])
    ax.axhline(l1_learned, color="#ea580c", linestyle="--", label=f"learned fixed a (0.48-0.50): {l1_learned:.2f}")
    ax.set(title="level 1 fixed_weights_r3 (z-score fusion)", xlabel="α = weight of the z-scored CLIP", ylabel="val R@1 (%)")
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(load_config(REPO_ROOT / "configs" / "count_v1.yaml"), splits=("val",))
    level1 = GraphStore(splits=("val",))
    arrays = store.candidates("val")
    assert np.array_equal(arrays["caption_ids"], level1.candidates("val")["caption_ids"])
    candidates, gold = arrays["candidate_ids"], arrays["gold"].astype(np.int64)
    clip = metrics_of(arrays["clip_scores"].astype(np.float64), candidates, gold)
    clip_rank = np.where(arrays["gold_rank"] > 0, arrays["gold_rank"], 51)
    OUT.mkdir(parents=True, exist_ok=True)

    # experiment 6: alpha sweep
    sweep = {"rule": __doc__, "clip_only": {k: 100 * clip[k] for k in METRICS}, "per_seed": {}, "grid": {}}
    ranks = {"countonly": [], "main_r3": []}
    for seed in range(3):
        run = f"countonly_factual_r3_seed{seed}"
        model, _, saved = load_run_model(RUNS / run, device, store=store)
        scores = channel_scores(model, store, device)
        a, b, c = scores["weights"].T
        learned = fuse(scores, a, b, c)
        check = metrics_of(learned, candidates, gold)
        row = {"learned": {k: 100 * check[k] for k in METRICS}, "learned_a_mean": float(a.mean()),
               "refuse_max_abs_diff": float(np.abs(learned - scores["fused"]).max()),
               "stored_recall@1": 100 * saved["recall@1"], "grid": {}}
        assert abs(check["recall@1"] - saved["recall@1"]) < 1e-9, (run, check["recall@1"], saved["recall@1"])
        for alpha in ALPHAS:
            rest = (1 - alpha) / (b + c)
            m = metrics_of(fuse(scores, np.full_like(a, alpha), rest * b, rest * c), candidates, gold)
            row["grid"][str(alpha)] = {k: 100 * m[k] for k in METRICS}
        sweep["per_seed"][run] = row
        ranks["countonly"].append(ranks_of(learned, candidates, gold))
        print(run, json.dumps({k: v for k, v in row.items() if k != "grid"}),
              json.dumps({a_: round(r["recall@1"], 2) for a_, r in row["grid"].items()}), flush=True)
    for alpha in ALPHAS:
        sweep["grid"][str(alpha)] = {k: mean_std([r["grid"][str(alpha)][k] for r in sweep["per_seed"].values()])
                                     for k in METRICS}
    sweep["learned"] = {k: mean_std([r["learned"][k] for r in sweep["per_seed"].values()]) for k in METRICS}
    sweep["learned_a_mean"] = mean_std([r["learned_a_mean"] for r in sweep["per_seed"].values()])
    best = max(ALPHAS, key=lambda a_: sweep["grid"][str(a_)]["recall@1"][0])
    sweep["best_alpha_on_val"] = {"alpha": best, **sweep["grid"][str(best)],
                                  "note": "chosen on val, so its val numbers are optimistic"}
    level1_runs = [json.loads((LEVEL1 / "controls" / f"alpha_sweep_fixed_weights_r3_seed{k}.json").read_text(encoding="utf-8"))
                   for k in range(3)]
    sweep["level1_fixed_weights"] = {"grid": {str(a_): mean_std([100 * next(x["recall@1"] for x in run["grid"] if x["a"] == a_)
                                                                 for run in level1_runs])
                                              for a_ in shared_grid(level1_runs)},
                                     "learned": mean_std([100 * run["learned_a_val"]["recall@1"] for run in level1_runs]),
                                     "learned_a": [run["learned_a_val"]["a"] for run in level1_runs],
                                     "source": "experiments/level1/controls/alpha_sweep_fixed_weights_r3_seed*.json"}
    (OUT / "alpha_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    plot(sweep, level1_runs, OUT / "alpha_sweep.png")

    # experiment 7: results by caption group
    for seed in range(3):
        model = load_run_model(LEVEL1 / "gat" / f"main_r3_seed{seed}", device, store=level1)[0]
        ranks["main_r3"].append(val_ranks(model, level1, device)[2])
    for seed in range(3):
        stored = json.loads((LEVEL1 / "gat" / f"main_r3_seed{seed}" / "metrics.json").read_text(encoding="utf-8"))["best"]
        assert abs((ranks["main_r3"][seed] == 1).mean() - stored["recall@1"]) < 1e-9
    parts = [GraphStore._graph_parts(store.queries[str(c)], query=True) for c in arrays["caption_ids"]]
    objects = np.array([sum(p["part_mask"]) for p in parts])
    relations = np.array([len(p["edges"]) for p in parts])
    explicit = np.array([any(cls in EXPLICIT for cls, keep in zip(p["count_index"], p["part_mask"]) if keep) for p in parts])
    groups = {"all": np.ones(len(parts), dtype=bool),
              "explicit_count": explicit, "no_explicit_count": ~explicit,
              "with_relation": relations > 0, "objects_only": relations == 0,
              "objects_0": objects == 0, "objects_1": objects == 1, "objects_2": objects == 2, "objects_3plus": objects >= 3}
    hit = {name: np.stack([r == 1 for r in rs]).astype(np.float64) for name, rs in ranks.items()}
    recip = {name: np.stack([np.where(r <= 50, 1 / r, 0.) for r in rs]) for name, rs in ranks.items()}
    hit["clip"], recip["clip"] = (clip_rank == 1).astype(np.float64)[None], np.where(clip_rank <= 50, 1 / clip_rank, 0.)[None]
    rng = np.random.default_rng(2026)
    result = {"rule": __doc__, "groups": {}}
    for name, mask in groups.items():
        if not mask.any():
            continue
        row = {"n": int(mask.sum()), "share": float(mask.mean())}
        for model_name in ("clip", "main_r3", "countonly"):
            r1 = 100 * hit[model_name][:, mask].mean(1)
            mrr = 100 * recip[model_name][:, mask].mean(1)
            row[model_name] = {"recall@1": mean_std(r1) if len(r1) > 1 else [float(r1[0]), 0.],
                               "mrr": mean_std(mrr) if len(mrr) > 1 else [float(mrr[0]), 0.],
                               "recall@1_per_seed": r1.tolist()}
        new, old, base = hit["countonly"][:, mask].mean(0), hit["main_r3"][:, mask].mean(0), hit["clip"][0, mask]
        row["countonly_minus_clip"] = {"recall@1": 100 * float((new - base).mean()),
                                       "ci95": [100 * x for x in bootstrap_ci(new - base, rng)]}
        welch = stats.ttest_ind(row["countonly"]["recall@1_per_seed"], row["main_r3"]["recall@1_per_seed"], equal_var=False)
        row["countonly_minus_main_r3"] = {"recall@1": 100 * float((new - old).mean()),
                                          "ci95": [100 * x for x in bootstrap_ci(new - old, rng)],
                                          "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
                                          "mrr": float(row["countonly"]["mrr"][0] - row["main_r3"]["mrr"][0])}
        result["groups"][name] = row
        print(name, row["n"], json.dumps({k: [round(x, 2) for x in row[k]["recall@1"]] for k in ("clip", "main_r3", "countonly")}),
              round(row["countonly_minus_clip"]["recall@1"], 2), round(row["countonly_minus_main_r3"]["recall@1"], 2),
              [round(x, 2) for x in row["countonly_minus_main_r3"]["ci95"]], round(welch.pvalue, 4), flush=True)
    (OUT / "groups.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
