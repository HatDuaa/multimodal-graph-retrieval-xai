"""Mean +- std over seeds of experiments/xai-metrics/metrics.json, and the evidence table of cases.json (seed 0).

Writes <dir>/summary.json and prints the markdown tables of REPORT.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-metrics"
MODELS = ("countonly_factual_r3", "main_r3")
GROUPS = ("all", "correct", "wrong")
KS = ("1", "3", "5")


def flatten(tree, prefix=""):
    out = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def summarize(metrics):
    out = {}
    for model in MODELS:
        runs = [r for r in metrics["runs"].values() if r["model"] == model]
        if not runs:
            continue
        out[model] = {"seeds": [r["seed"] for r in runs], "recall@1": [r["recall@1"] for r in runs],
                      "seconds": [r["seconds"] for r in runs], "groups": {}}
        for group in GROUPS:
            flat = [flatten(r["groups"][group]) for r in runs]
            keys = sorted(set.intersection(*(set(f) for f in flat)))
            out[model]["groups"][group] = {
                k: {"mean": float(np.mean([f[k] for f in flat])),
                    "std": float(np.std([f[k] for f in flat], ddof=1)) if len(flat) > 1 else 0.,
                    "seeds": [f[k] for f in flat]} for k in keys}
    return out


def number(value, digits):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def cell(stats, key, scale=1., digits=2, std=True):
    if key not in stats:
        return "—"
    text = number(stats[key]["mean"] * scale, digits)
    return f"{text} ± {number(stats[key]['std'] * scale, digits)}" if std else text


def row(*cells):
    return "| " + " | ".join(cells) + " |"


def header(*names):
    return [row(*names), row(*["---"] * len(names))]


def fidelity_table(data, sign):
    word = "bị bỏ" if sign == "plus" else "giữ lại"
    lines = header("Nhóm", "Số câu", f"Fid{'+' if sign == 'plus' else '−'} k", "Δ điểm top", "Δ điểm ngẫu nhiên",
                   "Hạng top", "Hạng ngẫu nhiên", "Mất hạng 1 top (%)", "Mất hạng 1 ngẫu nhiên (%)",
                   f"Hiệu ứng {word}, top: tỉ số tổng / trung vị (%)", "Ngẫu nhiên: tỉ số tổng / trung vị (%)")
    for group in GROUPS:
        g = data["groups"][group]
        for k in KS:
            p = f"fidelity.{k}.{sign}."
            share = lambda who: (f"{cell(g, p + who + '.share_pooled', 100, 1, False)} / "  # noqa: E731
                                 f"{cell(g, p + who + '.share_median', 100, 1, False)}")
            lines.append(row(group, cell(g, "captions", digits=0, std=False), k, cell(g, p + "top.delta", digits=4),
                             cell(g, p + "random.delta", digits=4), cell(g, p + "top.rank"), cell(g, p + "random.rank"),
                             cell(g, p + "top.lost_rank1", 100, 1), cell(g, p + "random.lost_rank1", 100, 1),
                             share("top"), share("random")))
    return lines


def top1_table(data):
    lines = header("Nhóm", "Top-1 của E", "Số câu (TB/seed)", "Bỏ top-1: mất hạng 1 (%)", "Ngẫu nhiên (%)",
                   "Bỏ top-1: Δ điểm", "Ngẫu nhiên", "Chỉ giữ top-1: hạng", "Ngẫu nhiên",
                   "Chỉ giữ top-1: mất hạng 1 (%)", "Ngẫu nhiên (%)")
    for group in GROUPS:
        g = data["groups"][group]
        for kind in ("node", "triple"):
            p = f"by_top1_kind.{kind}."
            if g.get(p + "captions", {"mean": 0})["mean"] == 0:
                continue
            lines.append(row(group, kind, cell(g, p + "captions", digits=0, std=False),
                             cell(g, p + "plus_top_lost_rank1", 100, 1), cell(g, p + "plus_random_lost_rank1", 100, 1),
                             cell(g, p + "plus_top_delta", digits=4), cell(g, p + "plus_random_delta", digits=4),
                             cell(g, p + "minus_top_rank"), cell(g, p + "minus_random_rank"),
                             cell(g, p + "minus_top_lost_rank1", 100, 1), cell(g, p + "minus_random_lost_rank1", 100, 1)))
    return lines


def validity_table(data):
    lines = header("Nhóm", "Dòng object", "Khớp nhãn (any, %)", "alias", "WordNet", "CLIP ≥ 0,955",
                   "Cùng danh từ chính", "any hoặc danh từ chính", "Dòng có số", "Khớp nhóm số lượng (%)")
    for group in GROUPS:
        g, v = data["groups"][group], "validity."
        lines.append(row(group, cell(g, v + "object_rows", digits=0, std=False),
                         *(cell(g, v + k, 100, 1) for k in ("any", "alias", "wordnet", "cosine", "head", "any_or_head")),
                         cell(g, v + "count_rows", digits=0, std=False), cell(g, v + "count_compatible", 100, 1)))
    return lines


def sparsity_table(data):
    lines = header("Nhóm", "Biến thể", "Câu có biên dương", "Số dòng", "Dòng cho 80%", "Dòng cho 90%",
                   "% số dòng (80 / 90)", "Dòng đồ thị (80 / 90)", "% phần của đồ thị ảnh (80 / 90)")
    for group in GROUPS:
        g = data["groups"][group]
        for variant in ("graph", "with_clip"):
            p = f"sparsity.{variant}."
            if p + "defined" not in g:
                continue
            pair = lambda m, scale=1., digits=2: (f"{cell(g, p + m + '80', scale, digits, False)} / "  # noqa: E731
                                                  f"{cell(g, p + m + '90', scale, digits, False)}")
            lines.append(row(group, variant, cell(g, p + "defined", digits=0, std=False), cell(g, p + "rows_mean"),
                             cell(g, p + "n80"), cell(g, p + "n90"), pair("share_of_rows", 100, 1), pair("graph_rows"),
                             pair("share_of_parts", 100, 1)))
    return lines


def subgraph_table(data):
    lines = header("Nhóm", "Tập câu", "Số câu", "Cạnh bỏ / câu", "Cạnh vào / câu", "Loại cạnh (có nhãn / kNN / tổng, %)",
                   "Δ điểm top", "Δ điểm ngẫu nhiên", "Hạng top", "Hạng ngẫu nhiên", "Mất hạng 1 top (%)",
                   "Ngẫu nhiên (%)")
    for group in GROUPS:
        g = data["groups"][group]
        kinds = " / ".join(cell(g, f"subgraph.kinds.{k}", 100, 0, False) if f"subgraph.kinds.{k}" in g else "0"
                           for k in ("labelled", "completion", "global"))
        for subset, p in (("có cạnh vào", "subgraph."), ("có lựa chọn", "subgraph.with_choice.")):
            lines.append(row(group, subset, cell(g, p + "captions", digits=0, std=False), cell(g, p + "edges_mean"),
                             cell(g, p + "pool_mean") if p + "pool_mean" in g else "—",
                             kinds if subset == "có cạnh vào" else "", cell(g, p + "top_delta", digits=4),
                             cell(g, p + "random_delta", digits=4), cell(g, p + "top_rank"), cell(g, p + "random_rank"),
                             cell(g, p + "top_lost_rank1", 100, 1), cell(g, p + "random_lost_rank1", 100, 1)))
    return lines


def shape_table(data):
    lines = header("Nhóm", "Node / ảnh", "Bộ ba / ảnh", "Bộ ba trong top-1 / top-3 / top-5 của E (%)",
                   "Top-3 node chiếm (% term dương của kênh)", "Top-3 bộ ba chiếm (%)", "Cách biệt hạng 1 – hạng 2")
    for group in GROUPS:
        g = data["groups"][group]
        lines.append(row(group, cell(g, "parts.nodes", digits=1, std=False), cell(g, "parts.triples", digits=1, std=False),
                         " / ".join(cell(g, f"composition.{k}", 100, 0, False) for k in KS),
                         cell(g, "concentration.objects.mean", 100, 1), cell(g, "concentration.triples.mean", 100, 1),
                         cell(g, "gap_mean", digits=4)))
    return lines


def tables(summary):
    lines = []
    for model, data in summary.items():
        recalls = ", ".join(number(100 * r, 2) for r in data["recall@1"])
        lines.append(f"\n### {model} (R@1 theo seed: {recalls})\n")
        for block in (fidelity_table(data, "plus"), fidelity_table(data, "minus"), top1_table(data), validity_table(data),
                      sparsity_table(data), subgraph_table(data), shape_table(data)):
            lines += block + [""]
    return "\n".join(lines)


def evidence(cases):
    lines = header("Câu (caption_id)", "Mô hình", "Ảnh hạng 1", "E top-3 (term)", "Bỏ top-3: Δ điểm, hạng (ngẫu nhiên)",
                   "Chỉ giữ top-3: Δ điểm, hạng (ngẫu nhiên)", "Dòng cho 80% biên",
                   "Đồ thị con: 3 cạnh vào đầu (alpha); Δ điểm top (ngẫu nhiên), số cạnh vào")
    for cid, case in cases.items():
        for model in MODELS:
            if model not in case:
                continue
            r = case[model]
            top = "; ".join(f"{e['part']} ({number(e['term'], 3)})" for e in r["explanation"][:3])
            fid = lambda f: (f"{number(f['top']['delta'], 3)}, {f['top']['rank']} "  # noqa: E731
                             f"({number(f['random']['delta'], 3)}, {number(f['random']['rank'], 1)})")
            sparse = r["sparsity"]["graph"]
            chosen = r["subgraph"].get("chosen", [])[:3]
            sub = ", ".join(f"{c['source']}→{c['target']} [{c['kind']}] ({number(c['alpha'], 2)})" for c in chosen)
            s = r["subgraph"]
            sub = (sub or "không có") + (f"; {number(s['top']['delta'], 4)} ({number(s['random']['delta'], 4)}), "
                                         f"{s['edges']}/{s['pool']}")
            image = f"{r['image_id']} ({'đúng' if r['correct'] else 'sai'})"
            lines.append(row(f"{case['caption'].replace('|', '/')} ({cid})", model, image, top,
                             fid(r["fidelity"]["3"]["plus"]), fid(r["fidelity"]["3"]["minus"]),
                             str(sparse["n80"]) if sparse else "—", sub))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=OUT)
    out = parser.parse_args().dir
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    summary = summarize(metrics)
    (out / "summary.json").write_text(json.dumps({"captions": metrics["captions"], "sample": metrics["sample"],
                                                  "torch": metrics["torch"], "models": summary}, indent=2),
                                      encoding="utf-8")
    print(tables(summary))
    print()
    print(evidence(json.loads((out / "cases.json").read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
