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

Ảnh của tập con, đặc trưng CLIP, đồ thị và checkpoint đặt trên Google Drive của nhóm, cấu trúc thư mục giống hệt `data/` và `experiments/` để tải về là đặt đúng chỗ. Link Drive: *(chưa tạo)*.

Quy tắc: file trên Drive **không sửa tại chỗ**. Khi cần đổi thì tạo tên mới (ví dụ `image_feats_v2.npy`) và cập nhật `MANIFEST.md`, để mọi người luôn chạy trên cùng một bản.
