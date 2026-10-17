# Bước 1 — Chẩn đoán kênh bộ ba bằng nhiễu phía câu (chỉ val, không train)

Script: `scripts/improve/triple_perturbation.py`; lệnh chạy lại: `experiments/improve/triple-perturbation/run.sh`.
Kết quả thô (mỗi seed, mỗi điều kiện): `main_r3.json`, `no_triple_r3.json`, `no_gat_r3.json`, `step0_frozen_clip.json`.
Mỗi file chạy khoảng 30 giây trên RTX 4090.

## Cách làm

Chỉ sửa đồ thị phía câu của val, đồ thị ảnh giữ nguyên. Checkpoint là `checkpoint_best.pt` của cấp 1, không train lại.

- **a. đổi động từ**: mỗi quan hệ trong đồ thị câu nhận một vị từ ngẫu nhiên lấy từ từ vựng vị từ của các đồ thị câu train (3 361 vị từ). Vị từ được rút theo tần suất trong train, không bao giờ trùng vị từ gốc, `--verb-seed 0`. Rút theo tần suất để vị từ thay thế giống phân bố câu thật ("on", "with", "holding"...), không phải các vị từ hiếm lạ. Đổi cả đặc trưng cạnh `rel:<p>` (đi vào GAT) lẫn cụm `triple:<s> <p> <o>`.
- **b. đổi chủ ngữ/tân ngữ**: đổi chỗ subject và object của mọi quan hệ phía câu. Cạnh bị đảo hướng, cụm thành `<o> <p> <s>`.
- **c. giữ nguyên**: mốc.
- Có 25 381 cụm chữ mới chưa có trong bảng part. Chúng được mã hoá bằng CLIP đóng băng (`ViT-B-32-quickgelu/openai`), quan hệ và bộ ba mã hoá nguyên văn, giống `src/service/graph_mode.py`.
- "Câu có ≥1 bộ ba": đồ thị câu còn ít nhất một quan hệ sau khi bỏ đại từ (`GraphStore._graph_parts(query=True)`), gồm 9 123/10 633 câu.
- "Chỉ xếp theo kênh bộ ba": xếp top-50 theo riêng điểm `channel_score` của kênh bộ ba, trước z-score. Ảnh không có bộ ba bị xếp cuối. Bảng này bỏ no_triple_r3 vì kênh của nó tắt.
- Mốc tham chiếu: no_triple_r3 (không có kênh bộ ba, nhiễu chỉ đi vào qua cạnh GAT phía câu); no_gat_r3; step0 (mô hình chưa train: part CLIP đóng băng, a/b/c cố định 0,60/0,28/0,12, τ = 0,05).
- Hạ tầng: driver NVIDIA tự cập nhật ngày 15/10 mà chưa reboot nên NVML lỗi. `expandable_segments:True` gọi NVML, vì vậy mọi lệnh chạy với `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Biến này chỉ đổi cách cấp phát bộ nhớ. Kiểm tra: mốc main_r3 ra đúng 46,48 ± 0,24 như cấp 1.

## Kết quả (mean ± std qua 3 seed; step0 chỉ có một mô hình)

### Toàn val (10 633 câu)

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_r3 | c. giữ nguyên | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 |
| main_r3 | a. đổi động từ | 46.06 ± 0.15 | 73.41 ± 0.10 | 83.20 ± 0.22 | 58.49 ± 0.13 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 45.91 ± 0.15 | 73.60 ± 0.08 | 83.39 ± 0.12 | 58.46 ± 0.06 |
| no_triple_r3 | c. giữ nguyên | 45.99 ± 0.05 | 73.33 ± 0.13 | 82.84 ± 0.10 | 58.34 ± 0.05 |
| no_triple_r3 | a. đổi động từ | 45.94 ± 0.07 | 73.25 ± 0.16 | 82.85 ± 0.08 | 58.29 ± 0.07 |
| no_triple_r3 | b. đổi chủ ngữ/tân ngữ | 45.81 ± 0.10 | 73.19 ± 0.14 | 82.86 ± 0.06 | 58.21 ± 0.09 |
| no_gat_r3 | c. giữ nguyên | 46.71 ± 0.07 | 73.81 ± 0.29 | 83.56 ± 0.24 | 59.04 ± 0.09 |
| no_gat_r3 | a. đổi động từ | 46.23 ± 0.11 | 73.46 ± 0.29 | 83.25 ± 0.21 | 58.61 ± 0.03 |
| no_gat_r3 | b. đổi chủ ngữ/tân ngữ | 46.30 ± 0.18 | 73.57 ± 0.13 | 83.44 ± 0.06 | 58.70 ± 0.06 |
| step0 (CLIP đóng băng) | c. giữ nguyên | 43.50 ± 0.00 | 70.95 ± 0.00 | 81.59 ± 0.00 | 56.10 ± 0.00 |
| step0 (CLIP đóng băng) | a. đổi động từ | 43.21 ± 0.00 | 70.89 ± 0.00 | 81.58 ± 0.00 | 55.98 ± 0.00 |
| step0 (CLIP đóng băng) | b. đổi chủ ngữ/tân ngữ | 43.33 ± 0.00 | 70.82 ± 0.00 | 81.43 ± 0.00 | 55.99 ± 0.00 |

### Câu có ≥1 bộ ba (9 123 câu)

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_r3 | c. giữ nguyên | 47.37 ± 0.29 | 74.72 ± 0.21 | 84.26 ± 0.30 | 59.72 ± 0.21 |
| main_r3 | a. đổi động từ | 46.89 ± 0.18 | 74.29 ± 0.13 | 83.98 ± 0.23 | 59.30 ± 0.14 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 46.71 ± 0.16 | 74.51 ± 0.11 | 84.20 ± 0.13 | 59.27 ± 0.04 |
| no_triple_r3 | c. giữ nguyên | 46.74 ± 0.05 | 74.07 ± 0.11 | 83.54 ± 0.09 | 59.07 ± 0.08 |
| no_triple_r3 | a. đổi động từ | 46.68 ± 0.13 | 73.98 ± 0.15 | 83.56 ± 0.08 | 59.01 ± 0.11 |
| no_triple_r3 | b. đổi chủ ngữ/tân ngữ | 46.53 ± 0.14 | 73.91 ± 0.12 | 83.57 ± 0.05 | 58.92 ± 0.13 |
| no_gat_r3 | c. giữ nguyên | 47.63 ± 0.09 | 74.77 ± 0.35 | 84.35 ± 0.24 | 59.92 ± 0.11 |
| no_gat_r3 | a. đổi động từ | 47.07 ± 0.10 | 74.37 ± 0.34 | 83.99 ± 0.23 | 59.43 ± 0.04 |
| no_gat_r3 | b. đổi chủ ngữ/tân ngữ | 47.14 ± 0.18 | 74.49 ± 0.16 | 84.21 ± 0.09 | 59.54 ± 0.07 |
| step0 (CLIP đóng băng) | c. giữ nguyên | 44.40 ± 0.00 | 71.61 ± 0.00 | 82.28 ± 0.00 | 56.90 ± 0.00 |
| step0 (CLIP đóng băng) | a. đổi động từ | 44.08 ± 0.00 | 71.54 ± 0.00 | 82.26 ± 0.00 | 56.76 ± 0.00 |
| step0 (CLIP đóng băng) | b. đổi chủ ngữ/tân ngữ | 44.21 ± 0.00 | 71.46 ± 0.00 | 82.09 ± 0.00 | 56.78 ± 0.00 |

### Chỉ xếp theo điểm kênh bộ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_r3 | c. giữ nguyên | 6.24 ± 0.14 | 24.97 ± 0.64 | 41.96 ± 0.59 | 17.37 ± 0.23 |
| main_r3 | a. đổi động từ | 4.77 ± 0.26 | 21.34 ± 0.50 | 37.46 ± 0.31 | 15.12 ± 0.25 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 4.69 ± 0.11 | 20.07 ± 0.37 | 35.82 ± 0.43 | 14.58 ± 0.23 |
| no_gat_r3 | c. giữ nguyên | 7.24 ± 0.72 | 27.18 ± 1.72 | 43.85 ± 1.68 | 18.66 ± 1.01 |
| no_gat_r3 | a. đổi động từ | 5.55 ± 0.68 | 22.85 ± 1.65 | 39.25 ± 1.98 | 16.07 ± 0.97 |
| no_gat_r3 | b. đổi chủ ngữ/tân ngữ | 5.16 ± 0.76 | 21.58 ± 2.06 | 37.44 ± 2.38 | 15.40 ± 1.21 |
| step0 (CLIP đóng băng) | c. giữ nguyên | 18.84 ± 0.00 | 42.27 ± 0.00 | 56.42 ± 0.00 | 30.96 ± 0.00 |
| step0 (CLIP đóng băng) | a. đổi động từ | 16.31 ± 0.00 | 40.02 ± 0.00 | 54.50 ± 0.00 | 28.62 ± 0.00 |
| step0 (CLIP đóng băng) | b. đổi chủ ngữ/tân ngữ | 17.70 ± 0.00 | 40.50 ± 0.00 | 55.33 ± 0.00 | 29.74 ± 0.00 |

## Nhận xét

1. **Kênh bộ ba có dùng vị từ và thứ tự, nhưng tác động lên điểm cuối nhỏ.** Trên toàn val, R@1 của main_r3 giảm 0,42 điểm khi đổi động từ và 0,57 điểm khi đổi chủ ngữ/tân ngữ. Ở no_triple_r3, nhiễu chỉ đi qua cạnh GAT phía câu, mức giảm là 0,05 và 0,18. Phần chênh còn lại khoảng 0,4 điểm đến từ kênh bộ ba. Sau nhiễu, main_r3 (46,06 và 45,91) rơi về mức no_triple_r3 (45,99). Tức là toàn bộ lợi ích 0,49 điểm của kênh bộ ba phụ thuộc vào vị từ đúng và chiều đúng.
2. **Kênh bộ ba sau train là bộ xếp hạng độc lập rất yếu.** Khi chỉ xếp theo điểm kênh bộ ba: main_r3 đạt R@1 6,24 ± 0,14, no_gat_r3 đạt 7,24 ± 0,72, còn cụm CLIP đóng băng ở step0 đạt 18,84 (ngẫu nhiên trong 50 ảnh là 2%). Train làm kênh này kém đi rõ rệt như một bộ xếp hạng tự thân. Điều này khớp với việc vector bộ ba bị dồn (cosine 0,97): giá trị tuyệt đối gần như không còn thông tin, mô hình chỉ dựa vào chênh lệch nhỏ mà z-score trong top-50 phóng to (giả thuyết 1).
3. **Cụm CLIP gần với túi từ hơn khi đổi thứ tự so với khi đổi động từ.** Ở step0, xếp riêng theo kênh bộ ba thì đổi chủ ngữ/tân ngữ làm R@1 mất 1,14 điểm (18,84 → 17,70), còn đổi động từ mất 2,53 điểm (→ 16,31). CLIP có nhận ra thứ tự, nhưng yếu hơn nhận ra động từ. Điều này ủng hộ một phần giả thuyết 2. Sau train, main_r3 lại nhạy với thứ tự hơn (−1,55, so với −1,47 khi đổi động từ), có lẽ vì vector đã cộng thêm `U·triple_mlp([h_s; e; h_o])`, phần này có phân biệt chiều.
