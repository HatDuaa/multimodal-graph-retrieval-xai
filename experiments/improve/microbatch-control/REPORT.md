# Đối chứng micro-batch 32

Công thức main_r3 (kênh bộ ba dạng cụm, mọi giá trị mặc định), seed 0, micro-batch 32 thay cho 128, chạy 2 epoch.
Batch hiệu dụng vẫn là 256. Lệnh chạy: `run.sh`. Log: `main_mb32_seed0/log.jsonl`. Allocator:
`expandable_segments:False` (xem `../README.md`). Main_r3_seed0 cấp 1 chạy micro-batch 128 với `expandable_segments:True`.

| Epoch | Run | R@1 | R@5 | R@10 | MRR | giây/epoch | GPU đỉnh (GB) |
|---|---|---|---|---|---|---|---|
| 1 | micro-batch 32 (đối chứng) | 45.28 | 72.89 | 82.92 | 57.78 | 313 | 4.9 |
| 1 | micro-batch 128 (main_r3_seed0) | 45.20 | 72.67 | 83.01 | 57.77 | 431 | 15.3 |
| 1 | chênh lệch | +0.08 | +0.22 | -0.08 | +0.01 | | |
| 2 | micro-batch 32 (đối chứng) | 45.73 | 73.19 | 83.43 | 58.25 | 312 | 4.9 |
| 2 | micro-batch 128 (main_r3_seed0) | 45.73 | 73.17 | 82.91 | 58.29 | 449 | 15.3 |
| 2 | chênh lệch | +0.01 | +0.02 | +0.52 | -0.04 | | |

Ở cả hai epoch, R@1 và MRR lệch dưới 0,1 điểm. R@1 là chỉ số dùng để chọn checkpoint. Để so sánh: std R@1 giữa
các seed của main_r3 là 0,24 điểm. Chênh lệch lớn nhất là R@10 ở epoch 2 (+0,52 điểm, bản micro-batch 32 cao hơn).
Mức này lớn hơn std R@10 giữa seed ở epoch tốt nhất (0,24), nhưng đây là một run duy nhất ở epoch 2. Kết luận:
micro-batch 32 không làm R@1/MRR đổi đáng kể, nên có thể so A, B với main_r3. Mỗi epoch mất khoảng 312 giây, so với
khoảng 285 giây ở các epoch ổn định của main_r3 (hai epoch đầu của main_r3_seed0 mất 431 và 449 giây). GPU đỉnh
4,9 GB, so với 15,3 GB.
