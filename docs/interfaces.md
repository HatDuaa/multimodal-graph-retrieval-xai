# Hợp đồng interface giữa các gói

Bốn format thống nhất để các gói làm song song và tự tạo dữ liệu giả (mock) mà không phải chờ nhau (xem `plan.md`, mục 3). **Mục 1 và 2 do gói 1 phát hành, đã có code. Mục 3 và 4 là bản ĐỀ XUẤT, chốt trong buổi họp tuần 1** với gói 2 và gói 4.

## 1. File split — `data/splits/<dataset>_<split>.json`

Gói 1 phát hành, các gói khác chỉ đọc. Ví dụ: `vg_coco_train.json`, `vg_coco_val.json`, `vg_coco_test.json`.

```json
{
  "dataset": "vg_coco",
  "split": "test",
  "images": [
    {
      "image_id": 2368620,
      "coco_id": 391895,
      "file_name": "2368620.jpg",
      "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2368620.jpg",
      "width": 500,
      "height": 375,
      "captions": [
        {"caption_id": "391895_0", "text": "A man riding a motorcycle on a dirt road."}
      ]
    }
  ]
}
```

- `image_id` là id của Visual Genome, dùng làm khoá chính ở mọi nơi (đặc trưng, node ảnh trong đồ thị, kết quả).
- Mỗi caption là một truy vấn; đáp án đúng là ảnh chứa nó. `caption_id` = `<coco_id>_<thứ tự>`.
- Pool ứng viên khi đánh giá split nào là toàn bộ ảnh của chính split đó.
- Lấy **toàn bộ** phần giao Visual Genome ∩ COCO, không lấy mẫu: train 46 944 ảnh (đã gộp `restval` của Karpathy), val 2 126, test 2 138; đa số ảnh có 5 caption, một ít có 6–7. 290 ảnh Visual Genome bị bỏ vì trùng `coco_id` với ảnh khác. Số liệu đầy đủ: `experiments/data_stats.md`.

## 2. Đặc trưng CLIP — `data/features/<dataset>_<loại>.npy` + `.ids.json`

Ghi và đọc bằng `save_features()` / `load_features()` trong `src/features/extract_clip.py`.

| File | Mỗi dòng là | Khoá trong `ids` |
|---|---|---|
| `vg_coco_image.npy` | một ảnh | `image_id` |
| `vg_coco_caption.npy` | một caption | `caption_id` |
| `vg_coco_concept.npy` | một nhãn VG150 (đối tượng theo mẫu `a photo of a {tên}`, quan hệ mã hoá nguyên văn); chưa trích | tên nhãn |

- Ma trận `float32`, mỗi dòng đã chuẩn hoá L2, nên tích vô hướng chính là cosine.
- `<tên>.ids.json` = `{"encoder": "ViT-B-32-quickgelu/openai", "dim": 512, "ids": [...]}`; dòng `i` của ma trận thuộc về `ids[i]`.
- Một file chứa cả train, val và test; lọc theo split bằng id lấy từ file split.
- Tìm top-k: `CosineIndex(feats, ids).search(queries, k)` trong `src/retrieval/faiss_index.py`.

## 2b. Đánh giá — `src/eval/metrics.py`

Mọi con số trong báo cáo phải đi qua module này để so sánh được với nhau.

- `evaluate(ranked_lists, golds, ks=(1, 5, 10))` → `{"n_queries", "recall@1", "recall@5", "recall@10", "mrr"}`. `golds[i]` là một id, hoặc một tập id khi truy vấn có nhiều đáp án đúng (khi đó tính theo đáp án xếp cao nhất).
- Hạng tính từ 1. Đáp án không nằm trong danh sách trả về thì tính trượt ở mọi Recall@k và đóng góp 0 vào MRR; vì danh sách thường cắt ở top-k nên MRR ở đây là MRR@độ-dài-danh-sách.
- `aggregate_seeds(runs)` → mean và **độ lệch chuẩn mẫu** (ddof=1) qua các seed; một lần chạy thì std = 0.
- `write_metrics(path, method=..., dataset=..., split=..., per_seed={seed: metrics}, config=...)` ghi `metrics.json` theo một schema duy nhất: `method`, `dataset`, `split`, `per_seed`, `aggregate`, `config`.
- Lộc viết bản đầu (2026-09-18) để baseline không bị chặn; người nhận gói 4 giữ và mở rộng module này (nDCG, tách nhóm truy vấn trực tiếp / gián tiếp).

## 3. Đồ thị — ĐỀ XUẤT (đổi 2026-09-19)

Mỗi ảnh một đồ thị nhỏ riêng, truy vấn cũng được tách thành một đồ thị cùng dạng; **không** dựng một đồ thị chung chứa mọi ảnh. Lý do và các bài tham khảo: `docs/research/graph-enhanced-image-retrieval-approaches.md`; chi tiết từng bước: sheet "Cap do 1" của `task-assignment.xlsx`. Thiết kế còn chờ số đo độ phủ (`scripts/checks/scene_graph_coverage.py`). ConceptNet không dùng ở cấp độ 1.

Format JSON thuần, không phụ thuộc thư viện, để gói 4 và demo đọc được mà không cần PyTorch Geometric:

- `data/graphs/<dataset>_scene_graphs.jsonl`, mỗi dòng một ảnh:

```json
{"image_id": 2368620,
 "objects": [{"id": 0, "name": "man"}, {"id": 1, "name": "motorcycle"}],
 "relations": [{"subject": 0, "predicate": "riding", "object": 1}]}
```

- `data/processed/query_graphs_<split>.json`: `{caption_id: {"objects": [...], "relations": [...]}}`, cùng cấu trúc như trên.

Tên đối tượng và quan hệ của ảnh đã chuẩn hoá theo alias và lọc theo VG150. Đồ thị của ảnh nào chỉ chứa chú thích của chính ảnh đó, không có cạnh gộp từ nhiều ảnh.

## 4. API điểm đồ thị — ĐỀ XUẤT

```python
def score(query: str, candidate_image_id: int) -> tuple[float, list[Match]]: ...

# Match = (bộ ba của truy vấn, bộ ba của ảnh, đóng góp vào điểm); bộ ba = (head, relation, tail)
```

- Các cặp bộ ba trả về phải là thứ thật sự tham gia tính điểm (đề cấm tìm đường đi sau khi đã xếp hạng).
- Cần thêm hàm `score_without(query, candidate, removed_triples)` để gói 4 đo fidelity và để demo có nút "bỏ bộ ba này".
- Demo và đánh giá chỉ gọi hai hàm này, không đọc trực tiếp bên trong mô hình.
