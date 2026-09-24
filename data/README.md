# Dữ liệu

Dữ liệu lớn **không commit**. Git chỉ giữ: file split (`data/splits/`), file này và `MANIFEST.md`.

## Ba bộ dữ liệu

| Bộ | Dùng cho | Nguồn | Trạng thái |
|---|---|---|---|
| Visual Genome ∩ MS-COCO | Cấp độ 1 | Chú thích Visual Genome v1.4 (objects, relationships, alias) + split Karpathy của COCO (kèm 5 caption mỗi ảnh). Hai bộ dùng chung ảnh, nối bằng trường `coco_id` trong `image_data.json`. | Script tải đã có |
| MKG-W | Cấp độ 2 | Kho mã của NativE / MMRNS | Chưa có script |
| MMKG tiếng Việt | Cấp độ 3 | Nhóm tự xây | Chưa bắt đầu |

## Tải chú thích Visual Genome ∩ COCO

```bash
python scripts/download_data.py
```

Tải khoảng 170 MB vào `data/raw/visual_genome/` và `data/raw/coco/`, giải nén, rồi in ra các dòng checksum để đối chiếu với `MANIFEST.md`. Script **không tải ảnh**: ảnh chỉ tải cho tập con cố định sau khi đã có file split (cả bộ ảnh Visual Genome nặng khoảng 15 GB, không cần).

## Tải ảnh

Link từng ảnh nằm sẵn trong file split (`data/splits/vg_coco_*.json`, trường `url`, máy chủ `cs.stanford.edu/people/rak248/VG_100K*`). Tải bằng script, **không tải tay**:

```bash
python scripts/download_images.py --splits test val     # 4 264 ảnh, ~600 MB, vài phút
python scripts/download_images.py                        # cả 51 208 ảnh, ~7,3 GB, khoảng 1 giờ 15 phút
```

- Script chạy lại được: ảnh đã có và đọc được thì bỏ qua; ảnh lỗi ghi vào `data/raw/images_failed.txt`, chạy lại đến khi file này rỗng.
- **Hầu hết mọi người chỉ cần `--splits test val`** (để xem ảnh khi phân tích kết quả và chạy demo). Ảnh train chỉ cần trên máy trích đặc trưng CLIP; sau bước đó mọi thực nghiệm chạy trên vector, không đọc ảnh nữa.
- Ảnh lưu tại `data/raw/images/<image_id>.jpg`, trùng với trường `file_name` trong file split.

## Lấy đặc trưng CLIP

**Không tự trích lại.** Đặc trưng được trích một lần (Lộc, máy RTX 4090) và chia sẻ qua Drive, để cả nhóm chạy trên đúng một bản; tự trích trên máy khác có thể lệch số ở chữ số thập phân cuối và làm kết quả các gói không khớp nhau.

| File | Nội dung | Dung lượng ước tính |
|---|---|---|
| `data/features/vg_coco_image.npy` + `.ids.json` | 51 208 vector ảnh, 512 chiều | ~100 MB |
| `data/features/vg_coco_caption.npy` + `.ids.json` | 256 183 vector caption | ~525 MB |

**Link Drive của nhóm:** <https://drive.google.com/drive/folders/11xG7r4oX884RmKfmLjtFQeLVpcSrv6cV?usp=sharing> → thư mục `data/features/`. Tải cả 4 file về đặt đúng vào `data/features/` trong repo, rồi đối chiếu checksum với `MANIFEST.md`.

Chỉ chạy `scripts/extract_features.py` khi đổi encoder; khi đó phải tải đủ 51 208 ảnh trước (script từ chối chạy nếu thiếu ảnh) và phát hành file mới với tên mới.

## Cấu trúc thư mục

```
data/
├── raw/          # file tải về nguyên bản (bỏ qua trong git)
├── splits/       # file split train/val/test — COMMIT, là nguồn chân lý duy nhất
├── features/     # đặc trưng CLIP .npy + .ids.json (bỏ qua trong git, chia sẻ qua Drive)
├── graphs/       # đồ thị đã dựng (bỏ qua trong git, chia sẻ qua Drive)
└── processed/    # sản phẩm trung gian khác (bỏ qua trong git)
```

## File nặng dùng chung

Ảnh của tập con, đặc trưng CLIP, đồ thị và checkpoint đặt trên Google Drive của nhóm, cấu trúc thư mục giống hệt `data/` và `experiments/` để tải về là đặt đúng chỗ. Link Drive: <https://drive.google.com/drive/folders/11xG7r4oX884RmKfmLjtFQeLVpcSrv6cV?usp=sharing>.

Quy tắc: file trên Drive **không sửa tại chỗ**. Khi cần đổi thì tạo tên mới (ví dụ `image_feats_v2.npy`) và cập nhật `MANIFEST.md`, để mọi người luôn chạy trên cùng một bản.

## Gói dữ liệu re-ranker đồ thị

Script `scripts/build_graph_data.py` sinh đồ thị không lọc VG150, đồ thị truy vấn, bảng đếm nhãn, bảng vector CLIP của các mảnh và ứng viên top-50 theo nhóm train cùng ứng viên val lấy từ baseline đã có. Không có tập dev; val dùng để dừng sớm và chọn mô hình, test không được chạm tới. Các file lớn được lưu ngoài Git; kích thước và SHA-256 sẽ được điền sau lần chạy đầy đủ (`TODO(run)`).

| File | Kích thước | SHA-256 |
|---|---:|---|
| `data/splits/vg_coco_train_groups.json` | TODO(run) | TODO(run) |
| `data/graphs/vg_coco_scene_graphs.jsonl` | TODO(run) | TODO(run) |
| `data/processed/query_graphs_{train,val,test}.json` | TODO(run) | TODO(run) |
| `data/features/vg_coco_graph_parts.npy` + `.ids.json` | TODO(run) | TODO(run) |
| `data/processed/candidates_train.npz` | TODO(run) | TODO(run) |
| `data/processed/candidates_val.npz` | TODO(run) | TODO(run) |

Visual Genome `attributes.json.zip` (SHA-256 `7f71c80fb5396c67a3022e0748de16a3253b7fe5759e151215e0f1a3de3f38c7`, 83 280 561 bytes) được tải bởi `scripts/download_data.py` để bổ sung thuộc tính màu/kích thước cho phép thử. File giải nén khoảng 0,46 GB và không commit.
