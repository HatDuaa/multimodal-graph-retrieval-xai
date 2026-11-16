# Bước 1d — Train lại công thức main_r3 với đồ thị câu của FACTUAL-T5 large

Chỉ dùng val, không chấm test. Thứ duy nhất thay đổi so với main_r3 là parser câu: sng_parser được thay bằng
FACTUAL-T5 large (`lizhuang144/flan-t5-large-VG-factual-sg`). Đồ thị ảnh (VG gốc), ứng viên top-50 của CLIP, vector
caption, công thức train và 3 seed đều giữ nguyên. Như mọi run cải tiến khác, micro-batch là 32 và allocator dùng
`expandable_segments:False` (xem `../README.md` và `../microbatch-control/REPORT.md`).

Lệnh chạy lại:

- `run_parse.sh`: parse train và test. Test chỉ để sẵn dữ liệu, không đánh giá.
- `run.sh`: đóng gói, kiểm tra, train, chẩn đoán.

Bảng dưới sinh bằng `python scripts/improve/summarize_improve.py --exp experiments/improve/factual-retrain --prefix main_factual_r3 --label main_factual_r3`
(ghi `summary.md`, `summary.json`). Các file kết quả khác: `triple_perturbation.json`, `part_similarity.json`,
`channel_correlation.json`, `welch_vs_main_r3.json`, `check_step0.json`, `check_probe.json`, `data_manifest.json`.
Log của từng run: `main_factual_r3_seed*/log.jsonl`. Timeline: `logs/timeline.log`.

## Dữ liệu: phiên bản `factual_v1`

- **Parse.** FACTUAL-large, beam 5, batch 64, cùng thiết lập với val ở `../parser-compare`.
  - Train: 234 854 câu, mất 8 311 giây (35,4 giây / 1 000 câu).
  - Test: 10 696 câu, mất 384 giây.
  - Không câu nào lỗi. Parse theo chunk 8 192 câu, chạy tiếp được; kiểm tra trên 300 câu val cho kết quả trùng lần
    parse val trước.
- **Đóng gói** (`scripts/improve/package_factual.py`, đọc qua `configs/factual_v1.yaml`):
  - `data/processed/factual_v1/query_graphs_{train,val,test}.json`.
  - Bảng part `data/features/factual_v1/vg_coco_graph_parts`: bảng cấp 1 (582 085 khoá) cộng 155 659 khoá mới
    (1 828 node, 5 269 quan hệ, 148 562 bộ ba), mã hoá bằng CLIP đóng băng. Mã hoá lại 1 000 khoá cấp 1 cho sai số
    tuyệt đối tối đa 0,0.
  - `candidates_*.npz` và vector caption là symlink tới dữ liệu cấp 1. Không file cấp 1 nào bị ghi đè.
  - Câu có ≥1 bộ ba: train 228 524/234 854 (97,3%), val 10 342/10 633.
- **Kiểm tra:**
  - Probe 3 kênh không train trên dữ liệu đã đóng gói: R@1 0,43618922 (τ 0,02, α 0,5, β 0,6), trùng số đã báo ở 1c.
  - Mô hình step 0 chấm qua đúng đường dữ liệu của train: R@1 0,434590426032164, trùng từng chữ số với điểm tham
    chiếu (τ 0,05, α 0,6, β 0,7) của probe.

## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| main_factual_r3 | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 | 5, 3, 10 | 7, 5, 10 | 1.91 h (3 seed) |
| main_r3 | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 | 8, 10, 6 | 10, 10, 8 | 2.30 h (3 seed) |
| no_triple_r3 | 45.99 ± 0.05 | 73.33 ± 0.13 | 82.84 ± 0.10 | 58.34 ± 0.05 | 4, 4, 6 | 6, 6, 8 | 1.25 h (3 seed) |
| no_gat_r3 | 46.71 ± 0.07 | 73.81 ± 0.29 | 83.56 ± 0.24 | 59.04 ± 0.09 | 5, 5, 3 | 7, 7, 5 | 1.01 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_factual_r3 | c. giữ nguyên | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 |
| main_factual_r3 | a. đổi động từ | 46.36 ± 0.40 | 73.72 ± 0.19 | 83.33 ± 0.21 | 58.68 ± 0.24 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.37 ± 0.23 | 73.85 ± 0.19 | 83.59 ± 0.10 | 58.78 ± 0.14 |
| main_r3 | c. giữ nguyên | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 |
| main_r3 | a. đổi động từ | 46.06 ± 0.15 | 73.41 ± 0.10 | 83.20 ± 0.22 | 58.49 ± 0.13 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 45.91 ± 0.15 | 73.60 ± 0.08 | 83.39 ± 0.12 | 58.46 ± 0.06 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_factual_r3 | c. giữ nguyên | 47.65 ± 0.35 | 74.82 ± 0.34 | 84.18 ± 0.16 | 59.86 ± 0.29 |
| main_factual_r3 | a. đổi động từ | 46.74 ± 0.40 | 74.03 ± 0.20 | 83.49 ± 0.23 | 59.02 ± 0.25 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.75 ± 0.22 | 74.17 ± 0.20 | 83.75 ± 0.12 | 59.13 ± 0.14 |
| main_r3 | c. giữ nguyên | 47.37 ± 0.29 | 74.72 ± 0.21 | 84.26 ± 0.30 | 59.72 ± 0.21 |
| main_r3 | a. đổi động từ | 46.89 ± 0.18 | 74.29 ± 0.13 | 83.98 ± 0.23 | 59.30 ± 0.14 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 46.71 ± 0.16 | 74.51 ± 0.11 | 84.20 ± 0.13 | 59.27 ± 0.04 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_factual_r3 | c. giữ nguyên | 9.52 ± 2.15 | 30.15 ± 3.77 | 46.88 ± 3.71 | 21.15 ± 2.65 |
| main_factual_r3 | a. đổi động từ | 6.09 ± 1.46 | 23.54 ± 2.67 | 40.05 ± 3.19 | 16.80 ± 1.86 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 5.85 ± 1.33 | 22.75 ± 2.33 | 38.84 ± 2.75 | 16.34 ± 1.64 |
| main_r3 | c. giữ nguyên | 6.24 ± 0.14 | 24.97 ± 0.64 | 41.96 ± 0.59 | 17.37 ± 0.23 |
| main_r3 | a. đổi động từ | 4.77 ± 0.26 | 21.34 ± 0.50 | 37.46 ± 0.31 | 15.12 ± 0.25 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 4.69 ± 0.11 | 20.07 ± 0.37 | 35.82 ± 0.43 | 14.58 ± 0.23 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| main_factual_r3 | c. giữ nguyên | 10.16 ± 2.09 | 30.99 ± 3.67 | 47.62 ± 3.61 | 21.83 ± 2.58 |
| main_factual_r3 | a. đổi động từ | 6.82 ± 1.42 | 24.56 ± 2.60 | 40.98 ± 3.10 | 17.59 ± 1.81 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 6.60 ± 1.29 | 23.79 ± 2.27 | 39.79 ± 2.68 | 17.15 ± 1.59 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.789; triples 0.684
- main_factual_r3_seed0: objects 0.383; triples 0.987
- main_factual_r3_seed1: objects 0.423; triples 0.967
- main_factual_r3_seed2: objects 0.193; triples 0.975
- main_r3 (cấp 1): triples 0.9661 ± 0.0083; objects 0.2546 ± 0.0308

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | main_factual_r3 | main_r3 | no_triple_r3 | no_gat_r3 |
|---|---|---|---|---|
| gat1 | 0.2386 ± 0.0690 | 0.1486 ± 0.0122 | 0.1697 ± 0.0380 | — |
| gat2 | 0.1878 ± 0.0474 | 0.1201 ± 0.0099 | 0.1305 ± 0.0169 | — |
| log_tau_objects | 0.0070 ± 0.0045 | 0.0439 ± 0.0076 | 0.0354 ± 0.0330 | 0.0575 ± 0.0084 |
| log_tau_triples | 0.0026 ± 0.0023 | 0.0017 ± 0.0014 | 0.0000 ± 0.0000 | 0.0034 ± 0.0027 |
| log_temperature | 0.0243 ± 0.0155 | 0.0873 ± 0.0716 | 0.0188 ± 0.0107 | 0.0156 ± 0.0127 |
| p_edge | 0.0628 ± 0.0127 | 0.0304 ± 0.0071 | 0.0070 ± 0.0027 | 0.0783 ± 0.0040 |
| p_node | 0.3714 ± 0.0646 | 0.3221 ± 0.0447 | 0.2843 ± 0.0499 | 0.2834 ± 0.0202 |
| triple_mlp | 0.2014 ± 0.0390 | 0.1353 ± 0.0050 | — | 0.1648 ± 0.0051 |
| u | 0.1266 ± 0.0458 | 0.0804 ± 0.0131 | 0.0965 ± 0.0122 | 0.0925 ± 0.0018 |
| u_objects | 0.0249 ± 0.0062 | 0.0267 ± 0.0063 | 0.0250 ± 0.0045 | 0.0223 ± 0.0029 |
| u_triples | 0.0016 ± 0.0002 | 0.0024 ± 0.0008 | 0.0000 ± 0.0000 | 0.0018 ± 0.0008 |
| weight_bias | 0.0759 ± 0.0439 | 0.0429 ± 0.0243 | 0.0271 ± 0.0152 | 0.0295 ± 0.0126 |
| weight_map | 0.0739 ± 0.0255 | 0.0537 ± 0.0136 | 0.0430 ± 0.0056 | 0.0477 ± 0.0019 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- main_factual_r3: a/b/c = 0.478 / 0.350 / 0.172; τ_triples = 0.081; GPU peak 5.3 GB
- main_r3: a/b/c = 0.485 / 0.355 / 0.160; τ_triples = 0.138; GPU peak 15.5 GB
- no_triple_r3: a/b/c = 0.504 / 0.496 / 0.000; τ_triples = 0.050; GPU peak 13.0 GB
- no_gat_r3: a/b/c = 0.473 / 0.353 / 0.174; τ_triples = 0.077; GPU peak 12.4 GB

## Kiểm định Welch so với main_r3 (3 seed mỗi bên, hai phía)

| Chỉ số | chênh lệch (điểm) | t | df | p |
|---|---|---|---|---|
| recall@1 | +0.76 | 3.02 | 3.50 | 0.046 |
| recall@5 | +0.71 | 3.32 | 2.93 | 0.046 |
| recall@10 | +0.56 | 3.40 | 3.54 | 0.033 |
| mrr | +0.64 | 3.23 | 3.36 | 0.041 |

## Tương quan theo câu giữa kênh vật thể và kênh bộ ba (`channel_correlation.json`)

- main_factual_r3: 0.238 ± 0.034 (CLIP ~ bộ ba 0.162; tỉ lệ câu có kênh bộ ba hoạt động 97.3%)
- main_r3: 0.197 ± 0.009 (CLIP ~ bộ ba 0.140; 85.8%)

## Thời gian chạy (RTX 4090, 18/10/2026)

| Bước | Thời gian |
|---|---|
| Parse train (FACTUAL-large) | 2 giờ 18 phút (16:18–18:37) |
| Parse test | 6,5 phút |
| Đóng gói | 58 giây |
| Kiểm tra (step 0 + probe) | 41 giây |
| Train 3 seed | 1,91 giờ (312 giây/epoch; 7, 5 và 10 epoch; GPU đỉnh 5,3 GB; không lần nào lỗi) |
| Chẩn đoán | 47 giây |

## Nhận xét

1. **Val tăng rõ và đều.** R@1 đạt 47,23 ± 0,36, so với 46,48 ± 0,24 của main_r3 (+0,76); MRR 59,49 ± 0,29, so với
   58,85 ± 0,18 (+0,64); R@5 tăng +0,71; R@10 tăng +0,56. Seed thấp nhất của FACTUAL (R@1 46,98) vẫn cao hơn seed cao
   nhất của main_r3 (46,67). Kiểm định Welch hai phía cho p ≈ 0,03–0,05 ở cả bốn chỉ số. Với 3 seed mỗi bên, đây là
   bằng chứng nhất quán nhưng còn mỏng. Mức tăng cũng lớn hơn hẳn mọi thí nghiệm trước của phase này, và cao hơn
   no_gat_r3 (46,71).
2. **Probe không train dự báo thấp lợi ích.** Trên probe, FACTUAL-large chỉ hơn sng +0,12 R@1. Khi chấm main_r3 với đồ
   thị FACTUAL mà không train lại, mức hơn là +0,13. Train lại trên đồ thị sạch hơn cho +0,76. Lợi ích đến từ việc mô
   hình *học* trên đồ thị tốt hơn, chứ không phải từ việc so khớp tốt hơn lúc chấm.
3. **Mô hình mới dựa vào vị từ và chiều nhiều hơn.** Đổi động từ làm R@1 toàn val mất 0,87 điểm, đổi chủ ngữ/tân
   ngữ mất 0,86 (main_r3: 0,42 và 0,57). Sau nhiễu, mô hình còn khoảng 46,36–46,37, gần mức main_r3 chưa nhiễu.
   Khi chỉ xếp theo kênh bộ ba: 9,52 ± 2,15, so với 6,24 của main_r3; đổi động từ còn 6,09, đổi chủ ngữ/tân ngữ còn
   5,85.
4. **Độ dồn của vector bộ ba vẫn còn.** Cosine trung bình giữa các vector bộ ba là 0,967–0,987, ngang main_r3
   (0,966). Parser tốt hơn không gỡ được hiện tượng này; đó vẫn là vấn đề của cơ chế (bước 1–3 của phase).
5. **Tương quan giữa kênh vật thể và kênh bộ ba** chỉ tăng nhẹ (0,238 so với 0,197), nên kênh bộ ba vẫn mang tín
   hiệu riêng. Kênh bộ ba hoạt động ở 97,3% số câu, so với 85,8%. Trọng số a/b/c = 0,478 / 0,350 / 0,172 (main_r3:
   0,485 / 0,355 / 0,160).
6. **Gradient.** GAT, `p_edge` và `triple_mlp` nhận gradient lớn hơn main_r3 (gat1 0,239 so với 0,149; `p_edge` 0,063
   so với 0,030; `triple_mlp` 0,201 so với 0,135). Đồ thị câu có nhiều cạnh hơn, nên các module đồ thị học nhiều hơn.
7. **Điểm cần lưu ý khi so sánh:**
   - main_r3 chạy micro-batch 128 với `expandable_segments:True`; bản này chạy micro-batch 32. Đối chứng ở
     `../microbatch-control` cho thấy đổi micro-batch làm R@1/MRR lệch dưới 0,1 điểm.
   - Tập "câu có ≥1 bộ ba" của mỗi mô hình là tập của parser tương ứng: 10 342 câu so với 9 123 câu.
   - Test chưa được chấm. Có đổi mô hình chính và chấm test một lần hay không là việc nhóm quyết.
