"""Render experiments/level1/explanations/README.md from cases.json, analysis_vi.json and part_similarity.json."""
import json
from pathlib import Path

OUT = Path("experiments/level1/explanations")
TITLES = {"success": "Thành công: đồ thị kéo ảnh đúng lên hạng 1",
          "broken": "Thất bại: CLIP đúng, mô hình làm tệ đi",
          "both_wrong": "Thất bại: cả CLIP và mô hình đều sai"}


def num(x, digits=3):
    return f"{x:.{digits}f}".replace(".", ",")


def contribution_table(explanation):
    rows = ["| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |", "|---|---|---|---|---|---|---|"]
    for channel, label in (("objects", "vật thể"), ("triples", "bộ ba")):
        for r in explanation[channel]:
            rows.append(f"| {label} | `{r['query_part']}` | `{r['image_part']}` | {num(r['w_i'], 2)} | {num(r['a_ij'], 2)} | "
                        f"{num(r['sim_ij'])} | {num(r['contribution'], 4)} |")
    return "\n".join(rows)


def deletion_line(d, who):
    return (f"- Xoá phần đóng góp lớn nhất của {who} (`{d['removed_part']}`, {'vật thể' if d['removed_kind'] == 'objects' else 'bộ ba'}): "
            f"hạng {d['before']['rank']} → **{d['after_top_part']['rank']}**, điểm {num(d['before']['score'])} → "
            f"{num(d['after_top_part']['score'])}. Xoá ngẫu nhiên {d['after_random_parts']['n']} phần cùng loại: hạng trung "
            f"bình {num(d['after_random_parts']['mean_rank'], 1)}, điểm trung bình {num(d['after_random_parts']['mean_score'])}.")


def main() -> None:
    data = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    analysis = json.loads((OUT / "analysis_vi.json").read_text(encoding="utf-8"))
    spread = json.loads(Path("experiments/checks/part_similarity.json").read_text(encoding="utf-8"))
    lines = ["# Giải thích và 10 truy vấn phân tích (cấp độ 1, val)", "",
             "Sinh bởi `scripts/explain_cases.py` (chọn ca, tính giải thích, phép thử xoá) và `scripts/render_explanations.py` "
             "(trang này). Dữ liệu gốc: `cases.json`. Lời phân tích: `analysis_vi.json`, viết sau khi xem ảnh thật. "
             "Mô hình: checkpoint tốt nhất theo val của `main_r3_seed0`, là mô hình chính đã chốt. Test không được dùng ở đây.", "",
             "## Cách chọn truy vấn", "",
             "- Chỉ xét câu val mà đồ thị câu có ít nhất một quan hệ (để hiện được cả kênh vật thể lẫn kênh bộ ba); "
             f"{num(100 * data['eligible_share'], 1)}% số câu val thoả điều kiện này.",
             "- Bốc ngẫu nhiên với `numpy.random.default_rng(2026)`: 5 câu **thành công** (CLIP xếp ảnh đúng dưới hạng 1, mô hình "
             "xếp hạng 1), 3 câu **mô hình làm tệ đi** (CLIP hạng 1, mô hình dưới hạng 1), 2 câu **cả hai đều sai**.", "",
             "Số câu mỗi loại trên toàn bộ 10 633 câu val (hạng tính trên top-50 của CLIP; ảnh đúng ngoài top-50 tính là sai):", "",
             "| Checkpoint | Thành công | Mô hình làm tệ đi | Cả hai sai | Cả hai đúng | Mô hình xếp ảnh đúng cao hơn CLIP | Thấp hơn CLIP |",
             "|---|---|---|---|---|---|---|"]
    for name, c in data["counts"].items():
        lines.append(f"| `{name}` | {c['success']} | {c['broken']} | {c['both_wrong']} | {c['both_right']} | {c['model_better']} | {c['model_worse']} |")
    lines += ["", "Với `main_r3_seed0`, số câu được sửa đúng gấp khoảng 2,8 lần số câu bị làm hỏng (1 235 so với 434), và R@1 bằng "
              "(1 235 + 3 727) / 10 633 = 46,67%, khớp số đã báo.", "",
              "## Cách đọc bảng giải thích", "",
              "Mọi số lấy thẳng từ các đại lượng mô hình dùng để xếp hạng (`src/explain/graph_explainer.py`), không tìm đường đi "
              "sau khi có kết quả. Với mỗi phần i của câu (vật thể hoặc bộ ba), bảng ghi phần j của ảnh có độ tương đồng lớn nhất, "
              "`w_i` (trọng số phần i trong câu), `a_ij` (attention của i lên j), `sim_ij` (cosine sau khi mã hoá), và đóng góp "
              "= trọng số kênh (b hoặc c) × w_i × a_ij × sim_ij. Mỗi dòng chỉ là cặp có attention lớn nhất, nên các đóng góp không "
              "cộng lại thành điểm cuối (điểm cuối còn qua z-score trong top-50).", "",
              "## Một phát hiện từ giải thích: vector bộ ba bị ép sát nhau", "",
              "Trong mọi ca, `sim_ij` của kênh bộ ba đều quanh 0,95–0,99, kể cả với cặp không liên quan (`woman on table` → "
              "`man wearing tie`). Đo trên đồ thị của 300 ảnh val (`scripts/checks/part_similarity.py`), cosine trung bình giữa "
              "hai phần bất kỳ:", "",
              "| Mô hình | Vật thể | Bộ ba |", "|---|---|---|"]
    for name in ("step0_frozen_clip", "main_r3_seed0", "main_r3_seed1", "main_r3_seed2", "no_gat_r3_seed0"):
        s = spread[name]
        lines.append(f"| `{name}` | {num(s['objects']['mean'])} | {num(s['triples']['mean'])} |")
    lines += ["", "Huấn luyện làm vector vật thể tách xa nhau hơn (0,79 → khoảng 0,25), nhưng lại dồn vector bộ ba về gần một hướng "
              "(0,68 → khoảng 0,97). Như vậy kênh bộ ba chỉ còn xếp hạng nhờ những khác biệt rất nhỏ được z-score phóng lên. Điều "
              "này khớp với các ablation: nối lại cạnh ngẫu nhiên gần như không làm giảm điểm, và phần lớn giải thích có ý nghĩa nằm "
              "ở kênh vật thể. Một nguyên nhân có thể là lớp `U` dùng chung cho vật thể và bộ ba (xem `docs/level1-model.md`). "
              "Chưa sửa, vì thiết kế đã chốt; ghi vào phần hạn chế.", ""]
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
                  f"Đồ thị câu: vật thể {', '.join(f'`{o}`' for o in case['query_graph']['objects'])}; bộ ba "
                  f"{', '.join(f'`{r}`' for r in case['query_graph']['relations'])}.", "",
                  "| Ảnh đúng | " + " | ".join(f"Mô hình hạng {t['rank']}" for t in case["top"][:3]) + " |",
                  "|---|" + "---|" * 3,
                  f"| ![]({case['gold_thumbnail']}) | " + " | ".join(f"![]({t['thumbnail']})" for t in case["top"][:3]) + " |",
                  "| " + f"`{case['gold']}`" + " | " + " | ".join(
                      f"`{t['image_id']}` (CLIP hạng {t['clip_rank']}){' — ảnh đúng' if t['is_gold'] else ''}" for t in case["top"][:3]) + " |", ""]
        if case["explain_gold"]:
            lines += ["Giải thích cho **ảnh đúng**:", "", contribution_table(case["explain_gold"]), ""]
        if "explain_top1" in case:
            lines += [f"Giải thích cho **ảnh mô hình xếp hạng 1** (`{case['top'][0]['image_id']}`):", "",
                      contribution_table(case["explain_top1"]), ""]
        lines.append("Phép thử xoá:")
        top1_who = "ảnh đúng" if case["top"][0]["is_gold"] else "ảnh mô hình xếp hạng 1"
        lines.append(deletion_line(case["deletion_top1"], top1_who))
        deletions.append((n, case["deletion_top1"]))
        if case.get("deletion_gold"):
            lines.append(deletion_line(case["deletion_gold"], "ảnh đúng"))
        lines += ["", "**Phân tích.** " + analysis[case["caption_id"]], ""]
    lines += ["## Tổng hợp phép thử xoá", "",
              "Xoá trên ảnh mô hình xếp hạng 1 của mỗi ca: phần có đóng góp lớn nhất so với trung bình 10 phần ngẫu nhiên cùng loại.", "",
              "| Ca | Phần bị xoá | Hạng trước | Hạng sau (phần lớn nhất) | Hạng sau (ngẫu nhiên, trung bình) | Giảm điểm (lớn nhất) | Giảm điểm (ngẫu nhiên) |",
              "|---|---|---|---|---|---|---|"]
    moved = larger = 0
    for n, d in deletions:
        drop_top = d["before"]["score"] - d["after_top_part"]["score"]
        drop_rand = d["before"]["score"] - d["after_random_parts"]["mean_score"]
        moved += d["after_top_part"]["rank"] > d["before"]["rank"]
        larger += drop_top > drop_rand
        lines.append(f"| {n} | `{d['removed_part']}` | {d['before']['rank']} | {d['after_top_part']['rank']} | "
                     f"{num(d['after_random_parts']['mean_rank'], 1)} | {num(drop_top)} | {num(drop_rand)} |")
    lines += ["", f"Ở {larger}/{len(deletions)} ca, xoá phần có đóng góp lớn nhất làm điểm giảm nhiều hơn xoá ngẫu nhiên. "
              f"Ở {moved}/{len(deletions)} ca, riêng việc xoá một phần này đã làm ảnh rơi khỏi hạng 1. Các ca còn lại giữ hạng vì "
              "bằng chứng trải trên nhiều vật thể. Đây là phép thử fidelity mức cơ bản; đo đầy đủ trên ≥10 truy vấn với các chỉ số "
              "fidelity / validity / sparsity thuộc cấp độ 3.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
