# Thí nghiệm 1 — Đổi bước chuẩn hoá để sửa vector bộ ba bị dồn (dữ liệu factual_v1)

Hai biến thể, mỗi biến thể 3 seed, theo công thức main_factual_r3; chỉ đổi bước chuẩn hoá điểm. Hằng số của cả hai
lấy từ thống kê **train** ở step 0 (`train_channel_stats.json`, `scripts/improve/channel_stats.py`), không dò trên
val. Lệnh chạy lại: `run.sh`. Chi tiết: `raw/REPORT.md` (1a) và `floor/REPORT.md` (1b).

| Mô hình (val, 3 seed) | R@1 | MRR | cosine vector bộ ba | chỉ xếp theo kênh bộ ba, R@1 | tương quan vật thể ~ bộ ba | a / b / c |
|---|---|---|---|---|---|---|
| main_factual_r3 (mốc) | 47.23 ± 0.36 | 59.49 ± 0.29 | 0.976 | 9.52 ± 2.15 | 0.238 | 0.478 / 0.350 / 0.172 |
| 1a `--fusion raw` | 47.48 ± 0.10 | 59.72 ± 0.00 | 0.961 | 7.22 ± 0.34 | 0.206 | 0.467 / 0.078 / 0.455 (thang điểm thô) |
| 1b `--zscore-floor` | 47.31 ± 0.17 | 59.58 ± 0.15 | 0.876 | 9.18 ± 3.00 | 0.236 | 0.474 / 0.334 / 0.192 |

Kết luận:

- **Giả thuyết 1 được xác nhận một nửa.**
  - Ở main_factual_r3, kênh bộ ba có std theo câu khoảng 0,003–0,006, nằm dưới sàn train ở hơn 99% câu val: z-score
    đúng là kéo giãn một kênh đã dồn (`floor/floor_activity.json`).
  - Bỏ z-score (1a) không gỡ được dồn: mô hình tự học lại phép phóng to qua trọng số c và nhiệt độ T.
  - Đặt sàn (1b) gỡ được một phần (cosine 0,98 → 0,88), nhưng kênh bộ ba không mạnh hơn khi đứng riêng, và điểm cuối
    không đổi.
- **Không biến thể nào hơn mốc rõ ràng trên val.** 1a +0,25 R@1, với độ lệch giữa seed nhỏ hơn nhiều và vẫn đang
  tăng ở trần 10 epoch; 1b +0,08.
