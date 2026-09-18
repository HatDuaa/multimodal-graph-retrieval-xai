# Hợp đồng interface giữa các gói

Bốn format thống nhất để các gói làm song song và tự tạo dữ liệu giả (mock) mà không phải chờ nhau (xem `plan.md`, mục 3). **Mục 1 và 2 do gói 1 phát hành, đã có code. Mục 3 và 4 là bản ĐỀ XUẤT, chốt trong buổi họp tuần 1** với gói 2 và gói 4.

## 1. File split — `data/splits/<dataset>_<split>.json`

Gói 1 phát hành, các gói khác chỉ đọc. Ví dụ: `vg_coco_train.json`, `vg_coco_val.json`, `vg_coco_test.json`.

```json
{
  "dataset": "vg_coco",
  "split": "test",
  "subset_seed": 0,
  "images": [
    {
      "image_id": 2368620,
      "coco_id": 391895,
      "file_name": "2368620.jpg",
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

## 2. Đặc trưng CLIP — `data/features/<dataset>_<loại>.npy` + `.ids.json`

Ghi và đọc bằng `save_features()` / `load_features()` trong `src/features/extract_clip.py`.

| File | Mỗi dòng là | Khoá trong `ids` |
|---|---|---|
| `vg_coco_image.npy` | một ảnh | `image_id` |
| `vg_coco_caption.npy` | một caption | `caption_id` |
| `vg_coco_concept.npy` | một khái niệm, mã hoá theo mẫu `a photo of a {tên}` | tên khái niệm |

- Ma trận `float32`, mỗi dòng đã chuẩn hoá L2, nên tích vô hướng chính là cosine.
- `<tên>.ids.json` = `{"encoder": "ViT-B-32/openai", "dim": 512, "ids": [...]}`; dòng `i` của ma trận thuộc về `ids[i]`.
- Một file chứa cả train, val và test; lọc theo split bằng id lấy từ file split.
- Tìm top-k: `CosineIndex(feats, ids).search(queries, k)` trong `src/retrieval/faiss_index.py`.

## 3. Đồ thị — ĐỀ XUẤT

Danh sách cạnh dạng bảng, không phụ thuộc thư viện, để gói 4 và demo đọc được mà không cần PyTorch Geometric:

- `data/graphs/<dataset>_nodes.json`: `[{"node_id": "img:2368620", "type": "image"}, {"node_id": "c:ball", "type": "concept"}]`
- `data/graphs/<dataset>_edges.tsv`: các cột `head`, `relation`, `tail`, `source`, `weight`, trong đó `source` ∈ {`scene_graph`, `conceptnet`, `contains`}.

Tiền tố `img:` và `c:` giúp id không trùng nhau giữa các loại node. Cạnh gộp từ thống kê chỉ tính trên ảnh train.

## 4. API điểm đồ thị — ĐỀ XUẤT

```python
def score(query: str, candidate_image_id: int) -> tuple[float, list[Path]]: ...

# Path = danh sách cạnh (head, relation, tail) kèm trọng số đóng góp vào điểm
```

- Đường đi trả về phải là thứ thật sự tham gia tính điểm (đề cấm tìm đường đi sau khi đã xếp hạng).
- Cần thêm hàm `score_without(query, candidate, removed_edges)` để gói 4 đo fidelity và để demo có nút "bỏ cạnh này".
- Demo và đánh giá chỉ gọi hai hàm này, không đọc trực tiếp bên trong mô hình.
