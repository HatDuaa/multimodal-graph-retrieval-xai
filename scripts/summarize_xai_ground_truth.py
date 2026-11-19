"""Mean +- std over seeds of experiments/xai-ground-truth/<benchmark>/metrics.json (scripts/xai_ground_truth.py).

Writes <dir>/summary.json and prints the markdown tables of <dir>/REPORT.md (saved as <dir>/logs/tables.md by run.sh).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_xai_metrics import EXPLAINERS, number, row  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-ground-truth"
BENCHMARKS = (("planted", "C. Cài phần đúng vào ảnh thật", ("main",)),
              ("vg-regions", "A. Region graph Visual Genome", ("main", "label")),
              ("synthetic", "B. Dữ liệu tổng hợp", ("main",)))
METRICS = ("gea", "precision", "auroc", "first", "decoy_percentile")


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
            "seeds": values}


def summarize(metrics):
    """{model: {"runs", "cases", explainer: {metric: stats over seeds}}}."""
    out = {}
    for model in sorted({r["model"] for r in metrics["runs"].values()}):
        runs = sorted((r for r in metrics["runs"].values() if r["model"] == model), key=lambda r: r["seed"])
        entry = {"seeds": [r["seed"] for r in runs], "cases": [r["summary"]["cases"] for r in runs],
                 "answer_size": stats([r["summary"].get("answer_size") for r in runs]),
                 "parts": stats([r["summary"].get("parts") for r in runs])}
        if "val_recall@1" in runs[0]:
            entry["val_recall@1"] = stats([r["val_recall@1"] for r in runs])
        if "gold_rank1" in runs[0]:
            entry["gold_rank1_share"] = stats([r["gold_rank1"] / r["queries_tried"] for r in runs])
        for name in EXPLAINERS:
            keys = sorted({k for r in runs for k in r["summary"].get(name, {}) if k != "cases"})
            if keys:
                entry[name] = {k: stats([r["summary"].get(name, {}).get(k) for r in runs]) for k in keys}
        out[model] = entry
    return out


def cell(s, scale=1., digits=3):
    if s is None:
        return "n/a"
    return f"{number(s['mean'] * scale, digits)} ± {number(s['std'] * scale, digits)}"


def tables(summaries):
    lines = []
    for benchmark, title, answers in BENCHMARKS:
        if benchmark not in summaries:
            continue
        lines += [f"## {title} (`{benchmark}`)", ""]
        for model, entry in summaries[benchmark].items():
            facts = [f"ca theo seed: {', '.join(str(c) for c in entry['cases'])}",
                     f"cỡ đáp án {cell(entry['answer_size'], digits=1)}", f"số phần {cell(entry['parts'], digits=1)}"]
            if "val_recall@1" in entry:
                facts.append(f"R@1 val tổng hợp {cell(entry['val_recall@1'], 100., 1)}%")
            if "gold_rank1_share" in entry:
                facts.append(f"ảnh có đáp án ở hạng 1: {cell(entry['gold_rank1_share'], 100., 1)}% truy vấn")
            lines += [f"### {model}", "", "; ".join(facts) + ".", ""]
            for answer in answers:
                label = {"main": "đáp án chính", "label": "đáp án theo nhãn (mọi node cùng tên)"}[answer]
                columns = ["GEA", "Precision@g", "AUROC", "Vị trí phần đúng đầu tiên"]
                decoy = benchmark == "planted" and answer == "main"
                if decoy:
                    columns.append("Phân vị của mồi nhử (cao tốt)")
                lines += [f"**{label}.**", "", row("Cách", *columns), row(*["---"] * (len(columns) + 1))]
                for name in EXPLAINERS:
                    if name not in entry:
                        continue
                    e = entry[name]
                    cells = [cell(e.get(f"{answer}_gea")), cell(e.get(f"{answer}_precision")),
                             cell(e.get(f"{answer}_auroc")), cell(e.get(f"{answer}_first"), digits=2)]
                    if decoy:
                        cells.append(cell(e.get(f"{answer}_decoy_percentile")))
                    lines.append(row(name, *cells))
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=OUT)
    args = parser.parse_args()
    summaries = {}
    for benchmark, _, _ in BENCHMARKS:
        path = args.dir / benchmark / "metrics.json"
        if path.exists():
            summaries[benchmark] = summarize(json.loads(path.read_text(encoding="utf-8")))
    (args.dir / "summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    print(tables(summaries))


if __name__ == "__main__":
    main()
