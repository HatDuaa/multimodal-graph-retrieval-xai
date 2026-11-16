"""Summarize one improvement experiment against the level-1 baselines (validation only).

Reads each run's metrics.json / log.jsonl (best epoch = selection by val R@1, as in training), the experiment's
triple_perturbation.json and part_similarity.json, and writes summary.json plus summary.md tables.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import aggregate_seeds  # noqa: E402

LEVEL1 = Path("experiments/level1/gat")
BASELINES = {"main_r3": "main_r3", "no_triple_r3": "no_triple_r3", "no_gat_r3": "no_gat_r3"}
KEYS = ("recall@1", "recall@5", "recall@10", "mrr")
CONDITIONS = {"baseline": "c. giữ nguyên", "verb_swap": "a. đổi động từ", "subject_object_swap": "b. đổi chủ ngữ/tân ngữ"}


def runs_of(root, prefix):
    return [root / f"{prefix}_seed{k}" for k in range(3)]


def run_summary(run):
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    log = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    best = metrics["best"]
    return {"best": {k: best[k] for k in ("n_queries",) + KEYS}, "best_epoch": best["epoch"], "epochs_run": len(log),
            "train_seconds": sum(e["seconds"] for e in log), "gradient_norms": best["gradient_norms"],
            "weights_mean": best["weights_mean"], "tau_triples": best["tau_triples"],
            "gpu_memory_peak_bytes": max(e["gpu_memory_peak_bytes"] for e in log)}


def row(agg):
    return " | ".join(f"{100 * agg[k]['mean']:.2f} ± {100 * agg[k]['std']:.2f}" for k in KEYS)


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline", action="append", metavar="NAME=DIR",
                        help="Comparison runs DIR/NAME_seed{0,1,2} (repeatable); default: the level-1 main_r3, "
                             "no_triple_r3 and no_gat_r3.")
    parser.add_argument("--reference", default="main_r3=experiments/improve/triple-perturbation/main_r3.json",
                        metavar="NAME=PATH", help="Perturbation diagnosis of the reference model.")
    parser.add_argument("--spread-reference", default="main_r3=experiments/checks/part_similarity.json",
                        metavar="NAME=PATH", help="part_similarity.json holding NAME_seed{0,1,2}.")
    args = parser.parse_args()
    groups = {args.label: runs_of(args.exp, args.prefix)}
    if args.baseline:
        for item in args.baseline:
            name, folder = item.split("=", 1)
            groups[name] = runs_of(Path(folder), name)
    else:
        groups.update({name: runs_of(LEVEL1, prefix) for name, prefix in BASELINES.items()})
    summary = {name: [run_summary(r) for r in runs] for name, runs in groups.items()}
    out = {"per_run": summary, "aggregate": {n: aggregate_seeds([r["best"] for r in rs]) for n, rs in summary.items()}}
    lines = ["## Val (checkpoint tốt nhất theo val R@1, 3 seed)", "",
             "| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |",
             "|---|---|---|---|---|---|---|---|"]
    for name, runs in summary.items():
        lines.append(f"| {name} | {row(out['aggregate'][name])} | {', '.join(str(r['best_epoch']) for r in runs)} | "
                     f"{', '.join(str(r['epochs_run']) for r in runs)} | "
                     f"{sum(r['train_seconds'] for r in runs) / 3600:.2f} h (3 seed) |")
    perturbation = json.loads((args.exp / "triple_perturbation.json").read_text(encoding="utf-8"))
    reference_name, reference_path = args.reference.split("=", 1)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    for view, title in (("all", "toàn val"), ("with_triple", "câu có ≥1 bộ ba"),
                        ("triple_only_with_triple", "chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba"),
                        ("third_only_all", "chỉ xếp theo điểm kênh thứ ba, toàn val")):
        lines += ["", f"## Chẩn đoán nhiễu phía câu — {title}", "", "| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |",
                  "|---|---|---|---|---|---|"]
        for name, result in ((args.label, perturbation), (reference_name, reference)):
            for condition, label in CONDITIONS.items():
                if view in result["aggregate"][condition]:
                    lines.append(f"| {name} | {label} | {row(result['aggregate'][condition][view])} |")
    spread = json.loads((args.exp / "part_similarity.json").read_text(encoding="utf-8"))
    spread_name, spread_path = args.spread_reference.split("=", 1)
    level1_spread = json.loads(Path(spread_path).read_text(encoding="utf-8"))
    lines += ["", "## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)", ""]
    names = [k for k in spread if k not in ("images", "pairs")]
    for name in names:
        lines.append(f"- {name}: " + "; ".join(
            f"{part} {value['mean']:.3f}" if "mean" in value else
            f"{part} (" + ", ".join(f"{s} {v['mean']:.3f}" for s, v in value.items()) + ")"
            for part, value in spread[name].items()))
    lines.append(f"- {spread_name} (mốc): triples " + mean_std([level1_spread[f"{spread_name}_seed{k}"]["triples"]["mean"] for k in range(3)])
                 + "; objects " + mean_std([level1_spread[f"{spread_name}_seed{k}"]["objects"]["mean"] for k in range(3)]))
    modules = sorted({m for rs in summary.values() for r in rs for m in r["gradient_norms"]})
    lines += ["", "## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)", "",
              "| Module | " + " | ".join(summary) + " |", "|---" * (len(summary) + 1) + "|"]
    for module in modules:
        cells = [mean_std([r["gradient_norms"][module] for r in rs]) if all(module in r["gradient_norms"] for r in rs)
                 else "—" for rs in summary.values()]
        lines.append(f"| {module} | " + " | ".join(cells) + " |")
    lines += ["", "## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)", ""]
    for name, runs in summary.items():
        w = np.asarray([r["weights_mean"] for r in runs])
        lines.append(f"- {name}: a/b/c = " + " / ".join(f"{x:.3f}" for x in w.mean(0)) +
                     f"; τ_triples = {np.mean([r['tau_triples'] for r in runs]):.3f}; GPU peak "
                     f"{max(r['gpu_memory_peak_bytes'] for r in runs) / 1e9:.1f} GB")
    lines += ["", f"## Diễn biến theo epoch của {args.label}: R@1 val, a/b/c (trung bình trên val), nhiệt độ loss T", ""]
    for run in runs_of(args.exp, args.prefix):
        log = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        lines.append(f"- {run.name}: " + "; ".join(
            f"e{e['epoch']} {100 * e['recall@1']:.2f} ({'/'.join(f'{w:.3f}' for w in e['weights_mean'])}, T {e['temperature']:.4f})"
            for e in log))
    (args.exp / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.exp / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
