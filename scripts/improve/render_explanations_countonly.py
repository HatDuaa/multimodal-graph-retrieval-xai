"""Render experiments/improve/explanations/README.md from cases.json, counterfactuals.json and analysis_vi.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.render_explanations import TITLES, contribution_table as level1_table, num  # noqa: E402

OUT = Path("experiments/improve/explanations")
EXAMPLE_TITLES = {"count_fixed": "Số lượng: một câu có số nêu rõ được sửa đúng",
                  "merge_fixed": "Đồ thị gộp: một câu được sửa đúng, phần khớp lớn nhất là node đã gộp",
                  "count_bad_match": "Số lượng làm khớp sai: \"lone woman\" → `hair`"}


def pct(part, total):
    return f"{100 * part / total:.0f}%".replace(".", ",") if total > 0 else "—"


def raw_table(e):
    """Raw fusion: every row an exact part of the final score; object rows with the count term g."""
    score = e["score"]
    a = e["weights"][0]
    rows = ["| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |",
            "|---|---|---|---|---|---|---|---|---|---|",
            f"| CLIP {num(a, 2)} | cả câu | cả ảnh |  |  | cosine {num(e['clip'], 4)} |  |  | {num(e['clip_part'], 4)} | "
            f"{pct(e['clip_part'], score)} |"]
    for channel, label, weight in (("objects", "vật thể", e["weights"][1]), ("triples", "bộ ba", e["weights"][2])):
        for r in sorted(e[channel], key=lambda r: -r["channel_term"]):
            counted = r.get("query_count", "unknown") != "unknown"
            query = f"`{r['query_part']}`" + (f" ({r['query_count']})" if counted else "")
            image = f"`{r['image_part']}`" + (f" ({r['image_count']})" if counted else "")
            g = f"{r['count_bias']:+.3f}".replace(".", ",").replace("-", "−") if counted else ""
            rows.append(f"| {label} {num(weight, 2)} | {query} | {image} | {num(r['w_i'], 2)} | {num(r['a_ij'], 2)} | "
                        f"{num(r['sim_ij'])} | {g} | {num(r['row_score'])} | {num(r['channel_term'], 4)} | "
                        f"{pct(r['channel_term'], score)} |")
        if e["fills"][channel] is not None:
            rows.append(f"| {label} {num(weight, 2)} | cả kênh | — ảnh không có {label}: trung bình của các ứng viên |  |  |  |  | "
                        f"{num(e['fills'][channel] / weight)} | {num(e['fills'][channel], 4)} | {pct(e['fills'][channel], score)} |")
    rows.append(f"| **Tổng** |  |  |  |  |  |  |  | **{num(score, 4)}** | 100% |")
    return "\n".join(rows)


def channel_parts(e):
    return {"clip": e["clip_part"], "objects": e["fills"]["objects"] if e["fills"]["objects"] is not None
            else sum(r["channel_term"] for r in e["objects"]),
            "triples": e["fills"]["triples"] if e["fills"]["triples"] is not None
            else sum(r["channel_term"] for r in e["triples"])}


def deletion_line(d, who):
    kind = "vật thể" if d["removed_kind"] == "objects" else "bộ ba"
    return (f"- Xoá phần đóng góp lớn nhất của {who} (`{d['removed_part']}`, {kind}, phần điểm {num(d['removed_contribution'], 4)}): "
            f"hạng {d['before']['rank']} → **{d['after_top_part']['rank']}**, điểm {num(d['before']['score'], 4)} → "
            f"{num(d['after_top_part']['score'], 4)}. Xoá ngẫu nhiên {d['after_random_parts']['n']} phần cùng loại: hạng trung "
            f"bình {num(d['after_random_parts']['mean_rank'], 1)}, điểm trung bình {num(d['after_random_parts']['mean_score'], 4)}.")


def images_row(case, extra=()):
    cells = [("Ảnh đúng", case["gold_thumbnail"], f"`{case['gold']}`")]
    for t in case["top"][:3]:
        cells.append((f"Mô hình hạng {t['rank']}", t["thumbnail"],
                      f"`{t['image_id']}` (CLIP hạng {t['clip_rank']}){' — ảnh đúng' if t['is_gold'] else ''}"))
    cells += list(extra)
    return ["| " + " | ".join(c[0] for c in cells) + " |", "|" + "---|" * len(cells),
            "| " + " | ".join(f"![]({c[1]})" for c in cells) + " |", "| " + " | ".join(c[2] for c in cells) + " |"]


def graph_line(q):
    objects = ", ".join(f"`{o}`" + (f" ({c})" if c != "unknown" else "") for o, c in zip(q["objects"], q["count_classes"]))
    return f"Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể {objects}; bộ ba {', '.join(f'`{r}`' for r in q['relations'])}."


def main() -> None:
    data = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    analysis = json.loads((OUT / "analysis_vi.json").read_text(encoding="utf-8"))
    counter = json.loads((OUT / "counterfactuals.json").read_text(encoding="utf-8"))
    counts, comparison = data["counts"], data["comparison"]
    c0 = counts["countonly_factual_r3_seed0"]
    lines = ["# Giải thích và 10 truy vấn phân tích — mô hình chính `countonly_factual_r3` (val)", "",
             "Sinh bởi `scripts/improve/explain_cases_countonly.py` (chọn ca, giải thích, phép thử xoá, so với cấp 1), "
             "`scripts/improve/explain_counterfactuals.py` (bỏ g / gỡ gộp lúc eval) và `scripts/improve/render_explanations_countonly.py` "
             "(trang này). Dữ liệu gốc: `cases.json`, `counterfactuals.json`. Lời phân tích: `analysis_vi.json`, viết sau khi xem ảnh "
             "thật. Mô hình: checkpoint tốt nhất theo val của `countonly_factual_r3_seed0` (seed có R@1 val cao nhất trong 3 seed). "
             "Test không được dùng ở đây. Bản tương ứng của mô hình cấp 1: `experiments/level1/explanations/`.", "",
             "## Cách chọn truy vấn", "",
             "Giống hệt cấp 1 (`scripts/explain_cases.py`, dùng lại các hàm của nó):", "",
             "- Chỉ xét câu val mà đồ thị câu FACTUAL có ít nhất một quan hệ; "
             f"{num(100 * data['eligible_share'], 1)}% số câu val thoả điều kiện này.",
             "- Bốc ngẫu nhiên với `numpy.random.default_rng(2026)`: 5 câu **thành công** (CLIP xếp ảnh đúng dưới hạng 1, mô hình "
             "xếp hạng 1), 3 câu **mô hình làm tệ đi** (CLIP hạng 1, mô hình dưới hạng 1), 2 câu **cả hai đều sai**.", "",
             "Số câu mỗi loại trên toàn bộ 10 633 câu val (hạng tính trên top-50 của CLIP; ảnh đúng ngoài top-50 tính là sai). "
             "R@1 tính lại ở đây trùng với `metrics.json` của từng run:", "",
             "| Checkpoint | Thành công | Mô hình làm tệ đi | Cả hai sai | Cả hai đúng | Mô hình xếp ảnh đúng cao hơn CLIP | Thấp hơn CLIP | R@1 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, c in counts.items():
        lines.append(f"| `{name}` | {c['success']} | {c['broken']} | {c['both_wrong']} | {c['both_right']} | {c['model_better']} | "
                     f"{c['model_worse']} | {num(100 * c['recall@1'], 2)} |")
    m0 = counts["main_r3_seed0"]
    lines += ["", f"Với seed 0, mô hình mới sửa đúng {c0['success']} câu CLIP sai và làm hỏng {c0['broken']} câu CLIP đúng, "
              f"tỉ lệ {num(c0['success'] / c0['broken'], 1)} lần. `main_r3_seed0` là {m0['success']} / {m0['broken']} "
              f"({num(m0['success'] / m0['broken'], 1)} lần). R@1 = ({c0['success']} + {c0['both_right']}) / 10 633 = "
              f"{num(100 * c0['recall@1'], 2)}%.", "",
              "## Cách đọc bảng giải thích", "",
              "Mọi số lấy thẳng từ lần xếp hạng (`src/explain/graph_explainer.py`). Mô hình dùng fusion raw:", "",
              "điểm cuối = a × cosine CLIP + b × S(vật thể) + c × S(bộ ba), với S = Σ_i w_i · Σ_j a_ij · sim_ij.", "",
              "Vì không có z-score trong top-50, mỗi dòng là một phần **chính xác** của điểm cuối, khác với bảng cấp 1:", "",
              "- Một dòng = trọng số kênh × w_i × (Σ_j a_ij · sim_ij) (cột \"Σ a·sim\"). Các phần cộng lại đúng bằng điểm cuối.",
              "- Cột \"Phần của ảnh\" ghi phần j có attention lớn nhất, kèm a_ij và sim_ij của nó.",
              "- Với vật thể có số lượng nêu trong câu, sim_ij đã gồm g[lớp số lượng của câu, nhóm số lượng của node ảnh]. Lớp và "
              "nhóm ghi trong ngoặc, g ở cột riêng.",
              "- Ảnh không có phần nào cho một kênh: fusion raw lấy điểm kênh trung bình của các ứng viên, ghi thành dòng \"cả kênh\".",
              "- Cạnh kNN `[none]` và node tổng chỉ dùng lúc truyền tin trong GAT, không thành dòng.", "",
              "Phép thử xoá chọn **phần của ảnh** có đóng góp chính xác lớn nhất, term_j = trọng số kênh × Σ_i w_i · a_ij · sim_ij. "
              "Đây là cùng phần điểm đó, nhưng chia theo phần của ảnh thay vì phần của câu. Sau khi xoá một vật thể, nhóm số lượng "
              "của ảnh được tính lại theo đúng luật count_v1.", ""]
    # finding: diffuse triple attention, triple parts nearly equal between gold and top-1
    a_obj = [r["a_ij"] for case in data["cases"] for key in ("explain_gold", "explain_top1") if case.get(key)
             for r in case[key]["objects"]]
    a_tri = [r["a_ij"] for case in data["cases"] for key in ("explain_gold", "explain_top1") if case.get(key)
             for r in case[key]["triples"]]
    lines += ["## Phát hiện: kênh bộ ba nặng ký nhưng khớp trải đều", "",
              f"Trong 10 ca, attention của cặp khớp nhất trung bình là {num(np.mean(a_obj), 2)} ở kênh vật thể nhưng chỉ "
              f"{num(np.mean(a_tri), 2)} ở kênh bộ ba ({len(a_obj)} và {len(a_tri)} dòng). Một dòng bộ ba vì vậy là trung bình "
              "của gần như mọi bộ ba trong ảnh, nên cặp \"khớp nhất\" thường vô nghĩa (`cow standing ear`, `man wearing pant`). "
              "Phần điểm của kênh này lớn, nhưng gần như bằng nhau giữa ảnh đúng và ảnh mô hình xếp hạng 1:", "",
              "| Ca | Ảnh | CLIP | Vật thể | Bộ ba | Điểm cuối |", "|---|---|---|---|---|---|"]
    for n, case in enumerate(data["cases"], 1):
        if case.get("explain_top1") and case.get("explain_gold"):
            for who, key in (("đúng", "explain_gold"), ("hạng 1", "explain_top1")):
                p = channel_parts(case[key])
                lines.append(f"| {n} | {who} | {num(p['clip'], 4)} | {num(p['objects'], 4)} | {num(p['triples'], 4)} | "
                             f"{num(case[key]['score'], 4)} |")
    lines += ["", "Chênh lệch giữa hai ảnh chủ yếu nằm ở CLIP và kênh vật thể. Phép thử xoá cho thấy cùng điều đó: xoá một vật "
              "thể thường đổi hạng, còn xoá bộ ba \"đóng góp lớn nhất\" thì gần như không. Điều này khớp với phép đo trước "
              "(`experiments/improve/count-only/part_similarity.json`): vector bộ ba của countonly vẫn bị ép sát nhau, cosine "
              "trung bình giữa hai bộ ba bất kỳ 0,875 / 0,899 / 0,867 ở 3 seed, so với 0,684 của CLIP đông cứng và khoảng 0,97 ở "
              "`main_r3`.", ""]
    deletions = []
    for n, case in enumerate(data["cases"], 1):
        if n == 1 or case["category"] != data["cases"][n - 2]["category"]:
            lines += [f"## {TITLES[case['category']]}", ""]
        gold_rank = "ngoài top-50" if case["model_rank"] == 51 else case["model_rank"]
        clip_rank = "ngoài top-50" if case["clip_rank"] == 51 else case["clip_rank"]
        a, b, c = case["weights_abc"]
        lines += [f"### Ca {n}. \"{case['caption']}\"", "",
                  f"`{case['caption_id']}` · ảnh đúng `{case['gold']}` · hạng theo CLIP: {clip_rank} · hạng theo mô hình: "
                  f"**{gold_rank}** · a / b / c = {num(a, 2)} / {num(b, 2)} / {num(c, 2)}", "",
                  graph_line(case["query_graph"]), "", *images_row(case), ""]
        if case["explain_gold"]:
            lines += ["Giải thích cho **ảnh đúng**:", "", raw_table(case["explain_gold"]), ""]
        if "explain_top1" in case:
            lines += [f"Giải thích cho **ảnh mô hình xếp hạng 1** (`{case['top'][0]['image_id']}`):", "",
                      raw_table(case["explain_top1"]), ""]
        lines.append("Phép thử xoá:")
        lines.append(deletion_line(case["deletion_top1"], "ảnh đúng" if case["top"][0]["is_gold"] else "ảnh mô hình xếp hạng 1"))
        deletions.append((n, case["deletion_top1"]))
        if case.get("deletion_gold"):
            lines.append(deletion_line(case["deletion_gold"], "ảnh đúng"))
        lines += ["", "**Phân tích.** " + analysis[case["caption_id"]], ""]
    lines += ["## Tổng hợp phép thử xoá", "",
              "Xoá trên ảnh mô hình xếp hạng 1 của mỗi ca: phần của ảnh có đóng góp lớn nhất, so với trung bình 10 phần ngẫu "
              "nhiên cùng loại (ít hơn nếu ảnh không đủ 11 phần loại đó).", "",
              "| Ca | Phần bị xoá | Loại | Hạng trước | Hạng sau (phần lớn nhất) | Hạng sau (ngẫu nhiên, trung bình) | Giảm điểm (lớn nhất) | Giảm điểm (ngẫu nhiên) |",
              "|---|---|---|---|---|---|---|---|"]
    moved = larger = 0
    by_kind = {"objects": [0, 0], "triples": [0, 0]}
    for n, d in deletions:
        drop_top = d["before"]["score"] - d["after_top_part"]["score"]
        drop_rand = d["before"]["score"] - d["after_random_parts"]["mean_score"]
        moved += d["after_top_part"]["rank"] > d["before"]["rank"]
        larger += drop_top > drop_rand
        by_kind[d["removed_kind"]][0] += 1
        by_kind[d["removed_kind"]][1] += d["after_top_part"]["rank"] > d["before"]["rank"]
        lines.append(f"| {n} | `{d['removed_part']}` | {'vật thể' if d['removed_kind'] == 'objects' else 'bộ ba'} | "
                     f"{d['before']['rank']} | {d['after_top_part']['rank']} | {num(d['after_random_parts']['mean_rank'], 1)} | "
                     f"{num(drop_top, 4)} | {num(drop_rand, 4)} |")
    lines += ["", f"- Ở {larger}/{len(deletions)} ca, xoá phần có đóng góp lớn nhất làm điểm giảm nhiều hơn xoá ngẫu nhiên.",
              f"- Ở {moved}/{len(deletions)} ca, riêng việc xoá một phần này đã làm ảnh rơi khỏi hạng 1.",
              f"- Chia theo loại phần bị xoá: vật thể {by_kind['objects'][1]}/{by_kind['objects'][0]} ca rơi hạng, bộ ba "
              f"{by_kind['triples'][1]}/{by_kind['triples'][0]} ca. Ca bộ ba duy nhất rơi hạng là ca 7, nơi hai ảnh đầu chỉ "
              "cách nhau 0,0003.",
              "- Cấp 1 (`main_r3`) dùng tiêu chí khác: cặp khớp nhất của dòng có đóng góp lớn nhất, nên hai bảng không so trực "
              "tiếp từng số được.", ""]
    # comparison with level 1
    pairs = comparison["pairs"]
    lines += ["## So với mô hình cấp 1 (`main_r3`)", "",
              "Mỗi seed của mô hình mới so với cùng seed của `main_r3`, trên 10 633 câu val. \"Sửa đúng\": `main_r3` xếp ảnh "
              "đúng dưới hạng 1, mô hình mới xếp hạng 1. \"Làm hỏng\": ngược lại. \"Có số nêu rõ\": câu có ít nhất một node "
              f"thuộc lớp 1 / 2 / 3–4 / ≥5 ({pairs['seed0']['explicit']} câu, {num(100 * comparison['explicit_share'], 1)}%).", "",
              "| Cặp seed | Sửa đúng | Làm hỏng | Chênh | Cả hai đúng | Cả hai sai | Sửa đúng (có số nêu rõ) | Làm hỏng (có số nêu rõ) |",
              "|---|---|---|---|---|---|---|---|"]
    for seed, p in pairs.items():
        lines.append(f"| {seed} | {p['fixed']} | {p['broken']} | +{p['fixed'] - p['broken']} | {p['both_right']} | "
                     f"{p['both_wrong']} | {p['fixed_explicit']} | {p['broken_explicit']} |")
    fixed = [p["fixed"] for p in pairs.values()]
    broken = [p["broken"] for p in pairs.values()]
    lines += ["", f"Trung bình 3 cặp: sửa đúng {num(np.mean(fixed), 0)} ± {num(np.std(fixed, ddof=1), 0)}, làm hỏng "
              f"{num(np.mean(broken), 0)} ± {num(np.std(broken, ddof=1), 0)} câu. Mô hình mới không chỉ cộng thêm câu đúng: "
              "nó đổi khá nhiều câu theo cả hai chiều, và phần lời ròng khoảng 260 câu, tức khoảng 2,4 điểm R@1.", "",
              "### Bỏ g hoặc gỡ gộp lúc eval (không train lại)", "",
              f"Đếm bằng `explain_counterfactuals.py`: {counter['nodes']['merged']} node ảnh val khi gộp, "
              f"{counter['nodes']['unmerged']} node khi gỡ gộp.", "",
              "| Run | Thay đổi lúc eval | R@1 | MRR | R@1 có số nêu rõ | R@1 không có số nêu rõ |", "|---|---|---|---|---|---|"]
    labels = {"trained": "như đã train", "g_zero": "g = 0", "unmerged": "gỡ gộp"}
    for run, rows in counter["seeds"].items():
        for change, r in rows.items():
            lines.append(f"| `{run}` | {labels[change]} | {num(100 * r['recall@1'], 2)} | {num(100 * r['mrr'], 2)} | "
                         f"{num(100 * r['recall@1_explicit'], 2)} | {num(100 * r['recall@1_no_explicit'], 2)} |")
    delta = lambda change, key: [100 * (rows[change][key] - rows["trained"][key]) for rows in counter["seeds"].values()]  # noqa: E731
    dz, dz_e, du = delta("g_zero", "recall@1"), delta("g_zero", "recall@1_explicit"), delta("unmerged", "recall@1")
    lines += ["", f"- **Bỏ g:** R@1 giảm {num(-np.mean(dz), 2)} ± {num(np.std(dz, ddof=1), 2)} điểm. Trên câu có số nêu rõ, R@1 giảm "
              f"{num(-np.mean(dz_e), 2)} ± {num(np.std(dz_e, ddof=1), 2)} điểm. Mô hình đã train thật sự dùng g lúc xếp hạng, và "
              "dùng nhiều nhất ở câu có số.",
              f"- **Gỡ gộp:** R@1 đổi {num(np.mean(du), 2)} ± {num(np.std(du, ddof=1), 2)} điểm, tức gần như không đổi. Mô hình đã "
              "train trên đồ thị gộp không nhạy với việc gộp lúc eval. Phần tăng của thí nghiệm 3 vì vậy khó quy cho việc gộp. "
              "Thí nghiệm 3 gộp 4 thay đổi mà chưa có ablation, như đã ghi ở phần hạn chế.", ""]
    for k, ex in enumerate(comparison["examples"], 1):
        l1 = ex["level1"]
        cf = counter["examples"][ex["caption_id"]]
        extra = [] if l1["top1"] == ex["gold"] else [("main_r3 hạng 1", l1["top1_thumbnail"], f"`{l1['top1']}`")]
        lines += [f"### Ví dụ {k}. {EXAMPLE_TITLES[ex['category']]}", "",
                  f"\"{ex['caption']}\" · `{ex['caption_id']}` · ảnh đúng `{ex['gold']}` · hạng theo CLIP: {ex['clip_rank']} · "
                  f"`main_r3_seed0`: {l1['model_rank']} · `countonly_factual_r3_seed0`: **{ex['model_rank']}**", "",
                  graph_line(ex["query_graph"]), "",
                  f"Đồ thị câu của cấp 1 (sng_parser): vật thể {', '.join(f'`{o}`' for o in l1['query_graph']['objects'])}; bộ ba "
                  f"{', '.join(f'`{r}`' for r in l1['query_graph']['relations'])}.", "",
                  *images_row(ex, extra), "",
                  "Giải thích của **countonly** cho ảnh đúng:", "", raw_table(ex["explain_gold"]), ""]
        if l1.get("explain_gold"):
            lines += ["Giải thích của **main_r3** cho ảnh đúng (bảng cấp 1: cặp khớp nhất, trước z-score):", "",
                      level1_table(l1["explain_gold"]), ""]
        if l1.get("explain_top1"):
            lines += [f"Giải thích của **main_r3** cho ảnh nó xếp hạng 1 (`{l1['top1']}`):", "", level1_table(l1["explain_top1"]), ""]
        lines += ["Bỏ g hoặc gỡ gộp lúc eval (countonly seed 0), hạng của ảnh đúng và cách biệt với ảnh tốt nhất còn lại:", "",
                  "| Thay đổi | Hạng | Điểm ảnh đúng | Cách biệt |", "|---|---|---|---|"]
        for change in ("trained", "g_zero", "unmerged"):
            r = cf[change]
            lines.append(f"| {labels[change]} | {r['rank']} | {num(r['gold_score'], 4)} | {num(r['gap_to_best_other'], 4)} |")
        lines += ["", "**Phân tích.** " + analysis[ex["caption_id"]], ""]
    lines += ["## Ghi chú về cách làm", "",
              "- Ví dụ 1 và 2 bốc theo luật, tiếp tục cùng bộ sinh số sau 10 ca:",
              "  - ví dụ 1: ngẫu nhiên trong các câu được sửa đúng có số nêu rõ;",
              "  - ví dụ 2: câu đầu tiên, theo thứ tự ngẫu nhiên, trong các câu được sửa đúng mà dòng vật thể có phần điểm lớn "
              "nhất ở ảnh đúng khớp một node đã gộp (boxes > 1).",
              "- Ví dụ 3 (151938_1) do nhóm chỉ định trước, vì `scripts/improve/count_match_changes.py` đã tìm thấy nó như một ca "
              "g kéo sang node sai.",
              "- Lần chạy đầu, phép thử xoá chọn phần theo dòng có channel_term lớn nhất. Mình đổi sang term_j theo phần của ảnh "
              "trước khi viết phân tích, vì attention bộ ba trải đều nên cặp khớp nhất của dòng chỉ giữ một phần nhỏ của dòng. "
              "Cách chọn ca không đổi: chạy lại cho đúng 10 ca và 3 ví dụ như lần đầu.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
