"""Summarise level 1 validation results: best-epoch metrics per configuration (mean ± std over seeds) and controls."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import aggregate_seeds, format_row  # noqa: E402

RUNS = Path("experiments/level1/gat")
CONTROLS = Path("experiments/level1/controls")
ROWS = [("main", "Đầy đủ (GAT, 3 kênh, a/b/c theo từng câu)"), ("fixed_weights", "a, b, c cố định (học trên train)"),
        ("no_gat", "Bỏ GAT"), ("no_triple", "Tắt kênh bộ ba"), ("no_edges", "Bỏ hẳn quan hệ"),
        ("text_query", "Câu là text thuần"), ("rewire", "Huấn luyện trên cạnh nối ngẫu nhiên"),
        ("lambda", "Loss LambdaRank (MRR) thay cross-entropy")]
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def best_metrics(run_dir):
    best = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["best"]
    return {"n_queries": best["n_queries"], **{k: best[k] for k in METRICS},
            "a": best["weights_mean"][0], "b": best["weights_mean"][1], "c": best["weights_mean"][2],
            "epoch": best["epoch"]}


def main(tag="r3"):
    summary = {"baselines": {"clip_only": 0.391328881783128, "training_free_three_channel": 0.43496661337346}}
    lines = ["| Cấu hình | Seed xong | R@1 | R@5 | R@10 | MRR | a / b / c | Epoch tốt nhất |", "|---|---|---|---|---|---|---|---|"]
    for key, label in ROWS:
        per_seed = {}
        for seed in (0, 1, 2):
            run_dir = RUNS / f"{key}_{tag}_seed{seed}"
            if (run_dir / "metrics.json").exists():
                per_seed[seed] = best_metrics(run_dir)
        if not per_seed:
            lines.append(f"| {label} | 0/3 | — | — | — | — | — | — |")
            continue
        agg = aggregate_seeds([{k: m[k] for k in ("n_queries",) + METRICS + ("a", "b", "c", "epoch")}
                               for m in per_seed.values()])
        summary[key] = {"seeds": sorted(per_seed), "per_seed": per_seed, "aggregate": agg}
        abc = " / ".join(f"{agg[k]['mean']:.3f}" for k in ("a", "b", "c"))
        epochs = ", ".join(str(per_seed[s]["epoch"]) for s in sorted(per_seed))
        # One seed has no spread; do not print a misleading "± 0".
        cells = format_row(agg) if len(per_seed) > 1 else " | ".join(f"{100 * agg[k]['mean']:.2f}" for k in METRICS)
        lines.append(f"| {label} | {len(per_seed)}/3 | {cells} | {abc} | {epochs} |")
    controls = {}
    for path in sorted(CONTROLS.glob("*.json")):
        controls[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    summary["controls"] = controls
    Path(f"experiments/level1/summary_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    print("\n".join(lines))
    for name, result in controls.items():
        if name.startswith("alpha_sweep"):
            print(name, "learned", result["learned_a_val"], "best", result["best_a_val"])
        else:
            corrupted = [v["recall@1"] for v in result["corrupted_val"].values()]
            print(name, "clean R@1 %.4f" % result["clean_val"]["recall@1"],
                  "corrupted R@1", ["%.4f" % v for v in corrupted])


if __name__ == "__main__":
    main(*sys.argv[1:])
