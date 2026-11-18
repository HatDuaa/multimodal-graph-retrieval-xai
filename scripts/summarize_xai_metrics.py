"""Mean +- std over seeds of experiments/xai-metrics-v2/metrics.json, and the evidence table of cases.json (seed 0).

Writes <dir>/summary.json and prints the markdown tables of <dir>/REPORT.md (saved as <dir>/logs/tables.md by run.sh).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-metrics-v2"
MODELS = ("countonly_factual_r3", "main_r3")
GROUPS = ("all", "correct", "wrong")
KS = ("1", "3", "5")
EXPLAINERS = ("model", "random", "lowest", "clip", "gradient", "integrated_gradients", "occlusion", "gnnexplainer",
              "pgexplainer", "gnn_lrp", "graphsvx", "subgraphx", "shapley")
DECOMPOSITIONS = ("model", "gradient", "integrated_gradients", "occlusion")


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
        runs = sorted((r for r in metrics["runs"].values() if r["model"] == model), key=lambda r: r["seed"])
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
        return "n/a"
    text = number(stats[key]["mean"] * scale, digits)
    return f"{text} ± {number(stats[key]['std'] * scale, digits)}" if std else text


def pct(stats, key, std=True):
    return cell(stats, key, 100., 1, std)


def worst_p(stats, key):
    """The largest value over seeds (a p value or an error), so the table never overstates a result."""
    if key not in stats:
        return "n/a"
    return f"{max(stats[key]['seeds']):.1e}".replace(".", ",").replace("-", "−")


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def header(*names):
    return [row(*names), row(*["---"] * len(names))]


def model_tables(name, summary):
    s = summary["groups"]
    a = s["all"]
    lines = [f"### {name} (R@1 theo seed: {', '.join(number(100 * r, 2) for r in summary['recall@1'])})", ""]

    lines += ["**Chạy.**", ""]
    lines += header("Seed", "Giây (CPU)", "Câu", "E rỗng", "noop khác hạng 1", "Sai số noop tối đa")
    runs = zip(summary["seeds"], summary["seconds"], a["captions"]["seeds"], a["empty_explanations"]["seeds"],
               a["noop_not_rank1"]["seeds"], a["noop_error_max"]["seeds"])
    lines += [row(seed, f"{sec:.0f}", int(n), int(e), int(x), f"{err:.1e}") for seed, sec, n, e, x, err in runs]
    lines.append("")

    for side, title, share in (("plus", "Fid+ (bỏ top-k)", "Hiệu ứng bị bỏ"), ("minus", "Fid− (chỉ giữ top-k)",
                                                                              "Hiệu ứng giữ lại")):
        lines += [f"**{title}.**", ""]
        lines += header("Nhóm", "k", "Mất hạng 1: top (%)", "Ngẫu nhiên (%)", "Hạng: top", "Ngẫu nhiên", "Δ điểm: top",
                        "Ngẫu nhiên", f"{share}, top: tỉ số tổng / trung vị (%)", "Ngẫu nhiên (%)")
        for group in GROUPS:
            g = s[group]
            for k in KS:
                p = f"fidelity.{k}.{side}"
                lines.append(row(group, k, pct(g, f"{p}.top.lost_rank1"), pct(g, f"{p}.random.lost_rank1"),
                                 cell(g, f"{p}.top.rank"), cell(g, f"{p}.random.rank"),
                                 cell(g, f"{p}.top.delta", digits=3), cell(g, f"{p}.random.delta", digits=3),
                                 f"{pct(g, f'{p}.top.share_pooled', False)} / {pct(g, f'{p}.top.share_median', False)}",
                                 f"{pct(g, f'{p}.random.share_pooled', False)} / "
                                 f"{pct(g, f'{p}.random.share_median', False)}"))
        lines.append("")

    lines += ["**Fid k = 1 theo loại phần đứng đầu E** (ngẫu nhiên: một phần cùng loại).", ""]
    lines += header("Nhóm", "Top-1 của E", "Câu (TB/seed)", "Bỏ top-1: mất hạng 1 (%)", "Ngẫu nhiên (%)",
                    "Chỉ giữ top-1: hạng", "Ngẫu nhiên")
    for group in GROUPS:
        for kind in ("node", "triple"):
            p = f"by_top1_kind.{kind}"
            lines.append(row(group, kind, cell(s[group], f"{p}.captions", digits=0, std=False),
                             pct(s[group], f"{p}.plus_top_lost_rank1"), pct(s[group], f"{p}.plus_random_lost_rank1"),
                             cell(s[group], f"{p}.minus_top_rank"), cell(s[group], f"{p}.minus_random_rank")))
    lines.append("")

    lines += ["**Hình dạng của E.**", ""]
    lines += header("Nhóm", "Node / ảnh", "Bộ ba / ảnh", "Bộ ba trong top-1 / 3 / 5 của E (%)",
                    "Top-3 node chiếm (% term dương kênh vật thể)", "Top-3 bộ ba chiếm (% kênh bộ ba)",
                    "Cách biệt hạng 1 – hạng 2")
    for group in GROUPS:
        g = s[group]
        lines.append(row(group, cell(g, "parts.nodes", digits=1, std=False), cell(g, "parts.triples", digits=1,
                                                                                   std=False),
                         " / ".join(pct(g, f"composition.{k}", False) for k in KS),
                         pct(g, "concentration.objects.mean"), pct(g, "concentration.triples.mean"),
                         cell(g, "gap_mean", digits=3)))
    lines.append("")

    lines += ["**Tính cộng của E:** tổng term của top-k (dự đoán) so với điểm mất thật khi bỏ top-k.", ""]
    lines += header("Nhóm", "k", "Câu", "Dự đoán (TB)", "Đo được (TB)", "Tỉ số tổng", "Tỉ số trung vị", "Spearman",
                    "Sai lệch tuyệt đối / hiệu ứng")
    for group in GROUPS:
        for k in KS:
            p = f"additivity.{k}"
            g = s[group]
            lines.append(row(group, k, cell(g, f"{p}.captions", digits=0, std=False), cell(g, f"{p}.predicted_mean"),
                             cell(g, f"{p}.measured_mean", digits=3), cell(g, f"{p}.pooled_ratio", digits=3),
                             cell(g, f"{p}.median_ratio", digits=3), cell(g, f"{p}.spearman", digits=3),
                             cell(g, f"{p}.gap_share_of_effect", digits=1)))
    lines.append("")

    lines += ["**Sparsity.**", ""]
    lines += header("Nhóm", "Phần cho 80% / 90% khối term dương", "% số phần (80 / 90)",
                    "k nhỏ nhất giữ 80% hiệu ứng", "k cho 90%", "Đạt trong k ≤ 10: 80% / 90% (% câu)",
                    "Dòng demo cho 80% / 90% biên")
    for group in GROUPS:
        g = s[group]
        lines.append(row(group, f"{cell(g, 'sparsity.terms.n80', digits=1)} / {cell(g, 'sparsity.terms.n90', digits=1)}",
                         f"{pct(g, 'sparsity.terms.share80', False)} / {pct(g, 'sparsity.terms.share90', False)}",
                         cell(g, "sparsity.fidelity.k80"), cell(g, "sparsity.fidelity.k90"),
                         f"{pct(g, 'sparsity.fidelity.reached80', False)} / "
                         f"{pct(g, 'sparsity.fidelity.reached90', False)}",
                         f"{cell(g, 'sparsity.caption_rows.n80')} / {cell(g, 'sparsity.caption_rows.n90')}"))
    lines.append("")

    lines += ["**Phản thực tế.**", ""]
    lines += header("Nhóm", "k_flip đạt (≤ 10, %)", "k_flip TB", "k_flip trung vị", "Câu có ảnh vàng khác hạng 1",
                    "Hạng ảnh vàng trước", "Sau bỏ top-1 / 3 / 5", "Ảnh vàng lên hạng: k = 1 / 3 / 5 (%)",
                    "Ảnh vàng lên hạng 1: k = 1 / 3 / 5 (%)")
    for group in GROUPS:
        g = s[group]
        c = "counterfactual"
        if f"{c}.gold.captions" not in g:
            lines.append(row(group, pct(g, f"{c}.k_flip_reached"), cell(g, f"{c}.k_flip_mean"),
                             cell(g, f"{c}.k_flip_median", digits=1, std=False), 0, *["n/a"] * 4))
            continue
        lines.append(row(group, pct(g, f"{c}.k_flip_reached"), cell(g, f"{c}.k_flip_mean"),
                         cell(g, f"{c}.k_flip_median", digits=1, std=False),
                         cell(g, f"{c}.gold.captions", digits=0, std=False), cell(g, f"{c}.gold.rank_before"),
                         " / ".join(cell(g, f"{c}.gold.rank_after_{k}", std=False) for k in KS),
                         " / ".join(pct(g, f"{c}.gold.rose_{k}", False) for k in KS),
                         " / ".join(pct(g, f"{c}.gold.first_{k}", False) for k in KS)))
    lines.append("")

    lines += ["**Validity ngữ nghĩa** (dòng vật thể và dòng bộ ba của E).", ""]
    lines += header("Nhóm", "Dòng vật thể", "strict (%)", "alias", "WordNet strict", "lenient (như v1)",
                    "WordNet lenient", "CLIP ≥ 0,955", "Cùng danh từ chính", "Dòng có số", "Khớp nhóm số (%)",
                    "Dòng bộ ba", "Chủ thể", "Đối tượng", "Vị từ", "Cả ba (%)")
    for group in GROUPS:
        g = s[group]
        v = "validity"
        lines.append(row(group, cell(g, f"{v}.object_rows", digits=0, std=False), pct(g, f"{v}.strict"),
                         pct(g, f"{v}.alias", False), pct(g, f"{v}.wordnet_strict", False), pct(g, f"{v}.lenient"),
                         pct(g, f"{v}.wordnet_lenient", False), pct(g, f"{v}.cosine", False), pct(g, f"{v}.head", False),
                         cell(g, f"{v}.count_rows", digits=0, std=False), pct(g, f"{v}.count_compatible"),
                         cell(g, f"{v}.triple_rows", digits=0, std=False), pct(g, f"{v}.triple_subject", False),
                         pct(g, f"{v}.triple_object", False), pct(g, f"{v}.triple_predicate", False),
                         pct(g, f"{v}.triple_all")))
    lines.append("")

    lines += ["**Đồ thị con theo alpha GAT** (messages: cắt đường truyền tin, giữ bộ ba; edges: cách v1, bỏ cả bộ ba).",
              ""]
    lines += header("Nhóm", "Tập câu", "Câu (TB/seed)", "Cạnh bỏ / cạnh vào", "Dạng", "Δ điểm: top", "Ngẫu nhiên",
                    "Mất hạng 1: top (%)", "Ngẫu nhiên (%)")
    for group in GROUPS:
        for subset, label in (("all", "có cạnh vào"), ("with_choice", "có lựa chọn")):
            g = s[group]
            p = f"subgraph.{subset}"
            for form in ("messages", "edges"):
                lines.append(row(group, label, cell(g, f"{p}.captions", digits=0, std=False),
                                 f"{cell(g, f'{p}.edges_mean', std=False)} / {cell(g, f'{p}.pool_mean', std=False)}",
                                 form, cell(g, f"{p}.{form}_top_delta", digits=4),
                                 cell(g, f"{p}.{form}_random_delta", digits=4), pct(g, f"{p}.{form}_top_lost_rank1"),
                                 pct(g, f"{p}.{form}_random_lost_rank1")))
    lines.append("")

    lines += ["**So sánh các cách giải thích** (mẫu cố định trong val; GNNExplainer, GraphSVX, SubgraphX và Shapley "
              "trên tập con).", ""]
    lines += header("Nhóm", "Cách", "Câu", "Giây / câu (CPU)", "Deletion AUC (thấp tốt)", "Insertion AUC (cao tốt)",
                    "k_flip đạt (%)",
                    "Bỏ top-1 / 3 / 5: mất hạng 1 (%)", "Chỉ giữ top-1 / 3 / 5: hạng",
                    "Fid_α+ / Fid_α− (k = 5, α = 0,1)", "Fid_α+ / Fid_α− (k = 5, α = 0,5)", "Khớp Shapley: Spearman",
                    "Top-5 trùng Shapley (%)", "p Wilcoxon deletion so với model (lớn nhất qua seed)")
    for group in GROUPS:
        for name in EXPLAINERS:
            g, p = s[group], f"explainers.{name}"
            if f"{p}.captions" not in g:
                continue
            lines.append(row(group, name, cell(g, f"{p}.captions", digits=0, std=False),
                             cell(g, f"{p}.seconds", digits=3, std=False) if f"{p}.seconds" in g
                             else "0" if name in ("model", "random", "lowest") else "n/a",
                             cell(g, f"{p}.deletion_auc", digits=3), cell(g, f"{p}.insertion_auc", digits=3),
                             pct(g, f"{p}.k_flip_reached"),
                             " / ".join(pct(g, f"{p}.plus_{k}_lost_rank1", False) for k in KS),
                             " / ".join(cell(g, f"{p}.minus_{k}_rank", std=False) for k in KS),
                             f"{cell(g, f'{p}.fid_alpha.plus_5_0.1.share', digits=3)} / "
                             f"{cell(g, f'{p}.fid_alpha.minus_5_0.1.share', digits=3)}",
                             f"{cell(g, f'{p}.fid_alpha.plus_5_0.5.share', digits=3)} / "
                             f"{cell(g, f'{p}.fid_alpha.minus_5_0.5.share', digits=3)}",
                             cell(g, f"{p}.agreement_with_shapley.spearman", digits=3),
                             pct(g, f"{p}.agreement_with_shapley.top5_overlap"),
                             worst_p(g, f"{p}.vs_model.deletion_auc.p")))
    lines.append("")

    lines += ["**Tính cộng của các cách phân rã** (cùng các câu so sánh).", ""]
    lines += header("Nhóm", "Cách", "k", "Tỉ số tổng", "Tỉ số trung vị", "Spearman", "Sai lệch tuyệt đối / hiệu ứng")
    for group in GROUPS:
        for name in DECOMPOSITIONS:
            for k in KS:
                g, p = s[group], f"explainers.{name}.additivity.{k}"
                lines.append(row(group, name, k, cell(g, f"{p}.pooled_ratio", digits=3),
                                 cell(g, f"{p}.median_ratio", digits=3), cell(g, f"{p}.spearman", digits=3),
                                 cell(g, f"{p}.gap_share_of_effect", digits=2)))
    lines.append("")

    lines += ["**Kiểm tra:** IG completeness, |Σ attribution − (f(ảnh) − f(trung tính))| / hiệu ứng: "
              f"{pct(a, 'ig_completeness.mean_abs_gap_share')}%. Shapley efficiency, sai lệch tối đa: "
              f"{worst_p(a, 'shapley_efficiency.max_abs_gap')}."
              + (f" GraphSVX efficiency, sai lệch tối đa: {worst_p(a, 'graphsvx_efficiency.max_abs_gap')}; "
                 f"TB {cell(a, 'graphsvx_efficiency.evaluations_mean', digits=0, std=False)} lượt chấm mỗi câu."
                 if "graphsvx_efficiency.max_abs_gap" in a else "")
              + (f" GNN-LRP: dựng lại đầu ra hai lớp GAT sai tối đa {worst_p(a, 'gnn_lrp_check.max_layer_gap')}, "
                 f"tổng relevance các walk lệch relevance đầu ra tối đa {worst_p(a, 'gnn_lrp_check.max_conservation_gap')}."
                 if "gnn_lrp_check.max_layer_gap" in a else "")
              + (f" SubgraphX: TB {cell(a, 'subgraphx_evaluations_mean', digits=0, std=False)} lượt chấm mỗi câu."
                 if "subgraphx_evaluations_mean" in a else ""), ""]
    return lines


def short(value, digits=3):
    return number(value, digits) if value is not None else "n/a"


def evidence(cases, model):
    lines = [f"### {model} (seed 0)", ""]
    lines += header("Câu (caption_id)", "Ảnh hạng 1", "E top-3 (term)", "Bỏ top-3: Δ điểm, hạng (ngẫu nhiên)",
                    "Chỉ giữ top-3: Δ điểm, hạng (ngẫu nhiên)", "k_flip", "k giữ 80%", "Phần cho 80% term",
                    "Tính cộng k = 3: dự đoán / đo", "Vật thể khớp strict", "Bộ ba khớp đủ",
                    "Cắt truyền tin: Δ top (ngẫu nhiên)")
    for cid, case in cases.items():
        c = case.get(model)
        if c is None:
            continue
        caption = f"{case['caption']} ({cid})"
        image = f"{c['image_id']} ({'đúng' if c['correct'] else 'sai'})"
        if c["empty_explanation"]:
            lines.append(row(caption, image, "E rỗng", *["n/a"] * 9))
            continue
        top = "; ".join(f"{e['part']} ({short(e['term'])})" for e in c["explanation"][:3])
        f3 = c["fidelity"]["3"]

        def edit(side):
            t, r = f3[side]["top"], f3[side]["random"]
            return f"{short(t['delta'])}, {t['rank']} ({short(r['delta'])}, {number(r['rank'], 1)})"

        add = c["additivity"]["3"]
        strict = sum(v["strict"] for v in c["validity"])
        triples = c["triple_validity"]
        sub = c["subgraph"]
        cut = (f"{short(sub['messages']['top']['delta'], 4)} ({short(sub['messages']['random']['delta'], 4)})"
               if sub and sub.get("messages") else "không có cạnh vào")
        lines.append(row(caption, image, top, edit("plus"), edit("minus"), c["k_flip"] or "> 10",
                         c["sparsity"]["fidelity"]["k80"] or "> 10", c["sparsity"]["terms"]["n80"],
                         f"{short(add['predicted'])} / {short(add['measured'])}", f"{strict}/{len(c['validity'])}",
                         f"{sum(t['all'] for t in triples)}/{len(triples)}", cut))
    lines.append("")
    return lines


def tables(summary, cases):
    lines = ["## Phụ lục: bảng đầy đủ theo nhóm", ""]
    for model in MODELS:
        if model in summary:
            lines += model_tables(model, summary[model])
    lines += ["## Phụ lục: 13 câu bằng chứng", ""]
    for model in MODELS:
        if model in summary:
            lines += evidence(cases, model)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=OUT)
    args = parser.parse_args()
    metrics = json.loads((args.dir / "metrics.json").read_text(encoding="utf-8"))
    cases = json.loads((args.dir / "cases.json").read_text(encoding="utf-8"))
    summary = summarize(metrics)
    (args.dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(tables(summary, cases))


if __name__ == "__main__":
    main()
