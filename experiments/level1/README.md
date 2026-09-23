# Kết quả re-ranker đồ thị

## Kiểm tra bước 0

Mô hình mới khởi tạo được kiểm tra ở chế độ đánh giá trên đúng các ứng viên top-50 của tập val.
Các vector phần tử, điểm thô, z-score, điểm `Graph`, điểm trộn và thứ hạng của 200 truy vấn đầu
được đối chiếu với `experiments/checks/three_channel_probe_reference_scores.npz`, sai số tuyệt đối
tối đa cho phép là `1e-4`. R@1 toàn bộ val phải khớp `0.43496661337346` của phép thử ba kênh
không huấn luyện, với trọng số ban đầu `(a,b,c) = (0,60; 0,28; 0,12)` và nhiệt độ `0,05`.

Lệnh chạy:

```bash
~/venvs/mgrx/bin/python scripts/checks/step0_reference_check.py
```

Kết quả lần chạy kiểm tra được ghi trong log của script; kết quả huấn luyện sẽ bổ sung sau.

## Tăng tốc đường dữ liệu và đo bộ nhớ (2026-09-23)

`GraphStore` tính sẵn một lần, cho mọi ảnh và câu truy vấn của train/val, chỉ số dòng trong bảng vector phần tử
và `edge_index` cục bộ (mảng số nguyên). Bảng vector nằm hẳn trong RAM (không memory-map), được chép một lần lên GPU,
và mỗi batch chỉ lấy vector bằng torch indexing. Val được chấm theo batch 256 câu. Kiểm tra bước 0 chạy trên
đúng đường mới: R@1 val = 0,43496661337346, 200 truy vấn đầu lệch tối đa 5,9e-6 (ngưỡng 1e-4), thứ tự giống hệt.

Giây mỗi epoch (gồm cả chấm val), batch hiệu dụng 256:

| Đường dữ liệu | Micro-batch | Giây/epoch |
|---|---|---|
| Cũ, dựng đồ thị từ JSON mỗi bước (`main_r2_seed2`) | 32 | 2 779 – 2 794 |
| Cũ (`main_seed0`, chẩn đoán) | 256 | 486 – 494 |
| Mới (`check_fastpath_main_seed2_mb128`) | 128 | 285 – 287 |

Run mới lặp lại cấu hình của `main_r2_seed2` (seed 2). Epoch 1: R@1 0,4524 so với 0,4515 cũ, R@5 0,7273 so với 0,7271,
MRR 0,5772 so với 0,5766. Epoch 2: R@1 0,4589 so với 0,4585. Mọi chênh lệch ≤ 1e-3.

Bộ nhớ theo micro-batch, 30 bước tối ưu của epoch 1 (`scripts/checks/memory_probe.py`, kết quả trong
`experiments/checks/memory_probe*.jsonl`). Lúc đo, GPU không có tiến trình nào của người khác; RAM máy còn trống ~111 GB / 125 GB.

| Micro-batch | RAM đỉnh | GPU cấp phát đỉnh | GPU giữ đỉnh (mặc định) | GPU giữ đỉnh (`expandable_segments`) | Giây/bước |
|---|---|---|---|---|---|
| 32 | 4,95 GB | 4,79 GB | 10,23 GB | — | 0,360 |
| 64 | 4,95 GB | 8,22 GB | 18,27 GB | 9,58 GB | 0,350 |
| 128 | 4,95 GB | 14,92 GB | 22,91 GB | 17,70 GB | 0,338 |
| 256 | — | tràn bộ nhớ GPU (OOM) | — | — | — |

Script huấn luyện giờ mặc định dùng `expandable_segments` và micro-batch 128, mức lớn nhất không tràn bộ nhớ
(~17,7 GB trên 24 GB; GPU không có người khác dùng nên không cần chừa). Batch hiệu dụng vẫn là 256.
