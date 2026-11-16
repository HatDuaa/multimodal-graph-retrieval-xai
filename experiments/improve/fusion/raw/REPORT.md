# Thí nghiệm 1a — Bỏ z-score: `--fusion raw` (dữ liệu factual_v1)

Mốc so sánh: main_factual_r3, cùng dữ liệu, cùng công thức (micro-batch 32, allocator override). Thay đổi duy nhất
là bước chuẩn hoá điểm (`src/models/graph_reranker.py`, `fuse_raw`):

- Điểm = a·CLIP + b·S_obj + c·S_tri, dùng điểm thô. a, b, c vẫn là softmax như cũ. Điểm thiếu (NaN) lấy trung bình
  các điểm đã biết của câu đó, giống cách z-score xử lý. Kênh không hoạt động thành hằng số, nên không ảnh hưởng
  cross-entropy.
- Sàn của nhiệt độ loss T hạ từ 0,05 xuống 0,005.
- Khởi tạo tương đương bản z-score: trọng số tỉ lệ với (0,60; 0,28; 0,12) / std_k, T = 1 / Σ(base_k / std_k), với
  std_k = trung vị std theo câu của từng kênh trên 20 000 câu **train** ở step 0 (`../train_channel_stats.json`):
  CLIP 0,0186, vật thể 0,0298, bộ ba 0,0436. Lúc bắt đầu a/b/c = 0,726 / 0,212 / 0,062, T = 0,0225.
- Phép tương đương chỉ gần đúng: bỏ qua bước z-score lại phần trộn đồ thị và biến thiên std giữa các câu. Kiểm tra
  nhanh: với mô hình gần như chưa học, R@1 val là 43,20 (bản z-score 43,46).

Lệnh chạy lại: `../run.sh`. Bảng dưới sinh bằng `scripts/improve/summarize_improve.py` với mốc main_factual_r3
(`summary.md`). Train 3 seed mất 2,59 giờ (khoảng 312 giây/epoch, GPU 5,3 GB, không lỗi), chạy từ 20:15 đến 22:52
ngày 19/10/2026; chẩn đoán mất 44 giây.

## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| raw_factual_r3 | 47.48 ± 0.10 | 74.70 ± 0.18 | 84.10 ± 0.08 | 59.72 ± 0.00 | 9, 10, 9 | 10, 10, 10 | 2.59 h (3 seed) |
| main_factual_r3 | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 | 5, 3, 10 | 7, 5, 10 | 1.91 h (3 seed) |
| main_r3 | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 | 8, 10, 6 | 10, 10, 8 | 2.30 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| raw_factual_r3 | c. giữ nguyên | 47.48 ± 0.10 | 74.70 ± 0.18 | 84.10 ± 0.08 | 59.72 ± 0.00 |
| raw_factual_r3 | a. đổi động từ | 46.56 ± 0.10 | 73.70 ± 0.08 | 83.36 ± 0.23 | 58.89 ± 0.08 |
| raw_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.62 ± 0.16 | 73.84 ± 0.12 | 83.63 ± 0.12 | 59.05 ± 0.11 |
| main_factual_r3 | c. giữ nguyên | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 |
| main_factual_r3 | a. đổi động từ | 46.36 ± 0.40 | 73.72 ± 0.19 | 83.33 ± 0.21 | 58.68 ± 0.24 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.37 ± 0.23 | 73.85 ± 0.19 | 83.59 ± 0.10 | 58.78 ± 0.14 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| raw_factual_r3 | c. giữ nguyên | 47.90 ± 0.11 | 75.02 ± 0.18 | 84.28 ± 0.09 | 60.07 ± 0.02 |
| raw_factual_r3 | a. đổi động từ | 46.95 ± 0.12 | 74.00 ± 0.08 | 83.52 ± 0.26 | 59.23 ± 0.09 |
| raw_factual_r3 | b. đổi chủ ngữ/tân ngữ | 47.01 ± 0.16 | 74.14 ± 0.12 | 83.79 ± 0.12 | 59.39 ± 0.11 |
| main_factual_r3 | c. giữ nguyên | 47.65 ± 0.35 | 74.82 ± 0.34 | 84.18 ± 0.16 | 59.86 ± 0.29 |
| main_factual_r3 | a. đổi động từ | 46.74 ± 0.40 | 74.03 ± 0.20 | 83.49 ± 0.23 | 59.02 ± 0.25 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.75 ± 0.22 | 74.17 ± 0.20 | 83.75 ± 0.12 | 59.13 ± 0.14 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| raw_factual_r3 | c. giữ nguyên | 7.22 ± 0.34 | 26.76 ± 0.57 | 43.92 ± 0.87 | 18.54 ± 0.43 |
| raw_factual_r3 | a. đổi động từ | 5.04 ± 0.26 | 21.69 ± 0.51 | 37.74 ± 0.29 | 15.42 ± 0.37 |
| raw_factual_r3 | b. đổi chủ ngữ/tân ngữ | 5.00 ± 0.40 | 21.11 ± 0.89 | 36.77 ± 1.23 | 15.18 ± 0.59 |
| main_factual_r3 | c. giữ nguyên | 9.52 ± 2.15 | 30.15 ± 3.77 | 46.88 ± 3.71 | 21.15 ± 2.65 |
| main_factual_r3 | a. đổi động từ | 6.09 ± 1.46 | 23.54 ± 2.67 | 40.05 ± 3.19 | 16.80 ± 1.86 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 5.85 ± 1.33 | 22.75 ± 2.33 | 38.84 ± 2.75 | 16.34 ± 1.64 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| raw_factual_r3 | c. giữ nguyên | 7.92 ± 0.33 | 27.70 ± 0.56 | 44.74 ± 0.85 | 19.29 ± 0.42 |
| raw_factual_r3 | a. đổi động từ | 5.81 ± 0.26 | 22.76 ± 0.50 | 38.73 ± 0.29 | 16.25 ± 0.36 |
| raw_factual_r3 | b. đổi chủ ngữ/tân ngữ | 5.77 ± 0.39 | 22.20 ± 0.87 | 37.78 ± 1.20 | 16.01 ± 0.57 |
| main_factual_r3 | c. giữ nguyên | 10.16 ± 2.09 | 30.99 ± 3.67 | 47.62 ± 3.61 | 21.83 ± 2.58 |
| main_factual_r3 | a. đổi động từ | 6.82 ± 1.42 | 24.56 ± 2.60 | 40.98 ± 3.10 | 17.59 ± 1.81 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 6.60 ± 1.29 | 23.79 ± 2.27 | 39.79 ± 2.68 | 17.15 ± 1.59 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.789; triples 0.684
- raw_factual_r3_seed0: objects 0.184; triples 0.960
- raw_factual_r3_seed1: objects 0.172; triples 0.962
- raw_factual_r3_seed2: objects 0.248; triples 0.960
- main_factual_r3 (mốc): triples 0.9762 ± 0.0101; objects 0.3330 ± 0.1232

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | raw_factual_r3 | main_factual_r3 | main_r3 |
|---|---|---|---|
| gat1 | 0.1785 ± 0.0331 | 0.2386 ± 0.0690 | 0.1486 ± 0.0122 |
| gat2 | 0.1350 ± 0.0220 | 0.1878 ± 0.0474 | 0.1201 ± 0.0099 |
| log_tau_objects | 0.0459 ± 0.0088 | 0.0070 ± 0.0045 | 0.0439 ± 0.0076 |
| log_tau_triples | 0.0058 ± 0.0085 | 0.0026 ± 0.0023 | 0.0017 ± 0.0014 |
| log_temperature | 0.0336 ± 0.0113 | 0.0243 ± 0.0155 | 0.0873 ± 0.0716 |
| p_edge | 0.0527 ± 0.0145 | 0.0628 ± 0.0127 | 0.0304 ± 0.0071 |
| p_node | 0.3872 ± 0.0733 | 0.3714 ± 0.0646 | 0.3221 ± 0.0447 |
| triple_mlp | 0.1857 ± 0.0452 | 0.2014 ± 0.0390 | 0.1353 ± 0.0050 |
| u | 0.0784 ± 0.0088 | 0.1266 ± 0.0458 | 0.0804 ± 0.0131 |
| u_objects | 0.0242 ± 0.0079 | 0.0249 ± 0.0062 | 0.0267 ± 0.0063 |
| u_triples | 0.0025 ± 0.0002 | 0.0016 ± 0.0002 | 0.0024 ± 0.0008 |
| weight_bias | 0.0875 ± 0.0105 | 0.0759 ± 0.0439 | 0.0429 ± 0.0243 |
| weight_map | 0.0785 ± 0.0046 | 0.0739 ± 0.0255 | 0.0537 ± 0.0136 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- raw_factual_r3: a/b/c = 0.467 / 0.078 / 0.455; τ_triples = 0.232; GPU peak 5.3 GB
- main_factual_r3: a/b/c = 0.478 / 0.350 / 0.172; τ_triples = 0.081; GPU peak 5.3 GB
- main_r3: a/b/c = 0.485 / 0.355 / 0.160; τ_triples = 0.138; GPU peak 15.5 GB

## Diễn biến theo epoch của raw_factual_r3: R@1 val, a/b/c (trung bình trên val), nhiệt độ loss T

- raw_factual_r3_seed0: e1 46.29 (0.726/0.212/0.062, T 0.0126); e2 46.34 (0.713/0.161/0.127, T 0.0115); e3 46.69 (0.635/0.132/0.233, T 0.0102); e4 46.52 (0.569/0.107/0.323, T 0.0091); e5 46.89 (0.526/0.100/0.373, T 0.0084); e6 46.63 (0.510/0.090/0.401, T 0.0080); e7 46.99 (0.499/0.086/0.416, T 0.0078); e8 46.93 (0.491/0.080/0.429, T 0.0075); e9 47.36 (0.472/0.078/0.450, T 0.0073); e10 47.28 (0.470/0.073/0.457, T 0.0072)
- raw_factual_r3_seed1: e1 45.82 (0.726/0.212/0.062, T 0.0126); e2 46.21 (0.705/0.167/0.128, T 0.0115); e3 46.77 (0.630/0.132/0.238, T 0.0101); e4 46.86 (0.569/0.113/0.318, T 0.0090); e5 46.70 (0.528/0.100/0.372, T 0.0084); e6 46.95 (0.511/0.091/0.398, T 0.0080); e7 46.95 (0.490/0.084/0.426, T 0.0078); e8 47.38 (0.481/0.083/0.436, T 0.0075); e9 47.40 (0.470/0.080/0.450, T 0.0073); e10 47.56 (0.462/0.076/0.462, T 0.0072)
- raw_factual_r3_seed2: e1 46.06 (0.726/0.212/0.062, T 0.0126); e2 46.39 (0.710/0.167/0.124, T 0.0115); e3 46.81 (0.629/0.131/0.241, T 0.0102); e4 46.74 (0.569/0.111/0.319, T 0.0090); e5 46.89 (0.525/0.099/0.376, T 0.0084); e6 47.28 (0.506/0.091/0.403, T 0.0079); e7 47.33 (0.485/0.091/0.424, T 0.0077); e8 47.28 (0.477/0.084/0.440, T 0.0075); e9 47.52 (0.466/0.080/0.454, T 0.0073); e10 47.38 (0.463/0.077/0.460, T 0.0072)

## Tương quan theo câu (`channel_correlation.json`)

- raw_factual_r3: vật thể ~ bộ ba 0.206 ± 0.004; CLIP ~ bộ ba 0.134. main_factual_r3: 0,238 ± 0,034; 0,162.

## Nhận xét

1. **Val nhích lên nhưng còn trong nhiễu; độ lệch giữa seed giảm mạnh.** R@1 47,48 ± 0,10, so với 47,23 ± 0,36
   (+0,25); MRR 59,72 ± 0,00, so với 59,49 ± 0,29 (+0,23); R@5 tăng +0,21; R@10 tăng +0,10. Theo seed: 47,36 /
   47,56 / 47,52, so với 47,08 / 46,98 / 47,64.
2. **Chưa hội tụ.** Cả 3 seed chạy hết 10 epoch, epoch tốt nhất là 9, 10, 9, và R@1 vẫn đang tăng. Bản z-score dừng
   sớm ở epoch 5, 3, 10. Mức trần 10 epoch có thể đang giới hạn bản raw; muốn biết thì phải chạy lâu hơn, nhưng đó
   là đổi thêm một thứ nên chưa làm.
3. **Không sửa được độ dồn (chỉ số chính).** Cosine trung bình giữa các vector bộ ba là 0,960–0,962, so với 0,976 ở
   main_factual_r3. Khi chỉ xếp theo kênh bộ ba: 7,22 ± 0,34, so với 9,52.
4. **Mô hình tự học lại việc phóng to.** Trọng số kênh bộ ba c tăng từ 0,062 lên 0,455, trọng số kênh vật thể b giảm
   từ 0,212 xuống 0,078, và T giảm từ 0,0225 xuống 0,0072. Kênh bộ ba vẫn dồn, chênh lệch thô rất nhỏ, nên mô hình
   bù bằng trọng số và nhiệt độ: thay cho phép chia cho std của z-score là một hệ số học được, dùng chung cho mọi
   câu. Lưu ý: a/b/c của bản raw nhân với điểm thô, nên không so trực tiếp được với a/b/c của bản z-score.
5. **Độ nhạy vị từ và chiều giữ nguyên.** Đổi động từ làm R@1 mất 0,92 điểm, đổi chủ ngữ/tân ngữ mất 0,86
   (main_factual_r3: 0,87 và 0,86).
6. **Tương quan vật thể–bộ ba** giảm nhẹ: 0,206 so với 0,238.
