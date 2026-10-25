# Thí nghiệm 1b — Sàn cho z-score: `--zscore-floor` (dữ liệu factual_v1)

Mốc so sánh: main_factual_r3. Thay đổi duy nhất: mỗi kênh được chia cho max(std, s_min) thay vì std
(`zscore_tensor(..., floor)`). Bước z-score lại phần trộn đồ thị và mọi thứ khác giữ nguyên.

- **s_min** = phân vị 5% của std theo câu của từng kênh, đo trên 20 000 câu **train** ở step 0
  (`../train_channel_stats.json`): CLIP 0,0114, vật thể 0,0185, bộ ba 0,0259.
- **Lý do chọn:** đó là độ trải mà 95% câu train đã có ngay lúc khởi tạo. Một kênh sau train bị ép xuống dưới mức đó
  sẽ không còn được kéo giãn lại về độ trải 1. Không dò s_min trên val.

Lệnh chạy lại: `../run.sh`. Bảng dưới sinh bằng `scripts/improve/summarize_improve.py` (`summary.md`). Mức độ sàn
thật sự tác động: `floor_activity.json` (`scripts/improve/floor_activity.py`). Train 3 seed mất 2,26 giờ (khoảng 312
giây/epoch, GPU 5,3 GB, không lỗi), chạy từ 22:52 ngày 19/10 đến 01:09 ngày 20/10/2026; chẩn đoán mất 42 giây.

## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| floor_factual_r3 | 47.31 ± 0.17 | 74.56 ± 0.35 | 84.24 ± 0.19 | 59.58 ± 0.15 | 4, 10, 9 | 6, 10, 10 | 2.26 h (3 seed) |
| main_factual_r3 | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 | 5, 3, 10 | 7, 5, 10 | 1.91 h (3 seed) |
| main_r3 | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 | 8, 10, 6 | 10, 10, 8 | 2.30 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| floor_factual_r3 | c. giữ nguyên | 47.31 ± 0.17 | 74.56 ± 0.35 | 84.24 ± 0.19 | 59.58 ± 0.15 |
| floor_factual_r3 | a. đổi động từ | 46.26 ± 0.14 | 73.77 ± 0.17 | 83.35 ± 0.09 | 58.63 ± 0.11 |
| floor_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.43 ± 0.26 | 73.94 ± 0.20 | 83.60 ± 0.23 | 58.89 ± 0.15 |
| main_factual_r3 | c. giữ nguyên | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 |
| main_factual_r3 | a. đổi động từ | 46.36 ± 0.40 | 73.72 ± 0.19 | 83.33 ± 0.21 | 58.68 ± 0.24 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.37 ± 0.23 | 73.85 ± 0.19 | 83.59 ± 0.10 | 58.78 ± 0.14 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| floor_factual_r3 | c. giữ nguyên | 47.74 ± 0.18 | 74.87 ± 0.37 | 84.42 ± 0.20 | 59.96 ± 0.16 |
| floor_factual_r3 | a. đổi động từ | 46.66 ± 0.14 | 74.06 ± 0.17 | 83.50 ± 0.07 | 58.98 ± 0.11 |
| floor_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.84 ± 0.27 | 74.23 ± 0.22 | 83.76 ± 0.21 | 59.25 ± 0.15 |
| main_factual_r3 | c. giữ nguyên | 47.65 ± 0.35 | 74.82 ± 0.34 | 84.18 ± 0.16 | 59.86 ± 0.29 |
| main_factual_r3 | a. đổi động từ | 46.74 ± 0.40 | 74.03 ± 0.20 | 83.49 ± 0.23 | 59.02 ± 0.25 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 46.75 ± 0.22 | 74.17 ± 0.20 | 83.75 ± 0.12 | 59.13 ± 0.14 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| floor_factual_r3 | c. giữ nguyên | 9.18 ± 3.00 | 29.94 ± 4.38 | 46.54 ± 4.01 | 20.79 ± 3.40 |
| floor_factual_r3 | a. đổi động từ | 5.90 ± 1.89 | 23.08 ± 3.51 | 39.34 ± 3.73 | 16.41 ± 2.42 |
| floor_factual_r3 | b. đổi chủ ngữ/tân ngữ | 6.45 ± 2.36 | 23.98 ± 4.38 | 39.49 ± 4.29 | 17.01 ± 3.01 |
| main_factual_r3 | c. giữ nguyên | 9.52 ± 2.15 | 30.15 ± 3.77 | 46.88 ± 3.71 | 21.15 ± 2.65 |
| main_factual_r3 | a. đổi động từ | 6.09 ± 1.46 | 23.54 ± 2.67 | 40.05 ± 3.19 | 16.80 ± 1.86 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 5.85 ± 1.33 | 22.75 ± 2.33 | 38.84 ± 2.75 | 16.34 ± 1.64 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| floor_factual_r3 | c. giữ nguyên | 9.83 ± 2.92 | 30.78 ± 4.26 | 47.29 ± 3.90 | 21.48 ± 3.31 |
| floor_factual_r3 | a. đổi động từ | 6.64 ± 1.84 | 24.12 ± 3.41 | 40.29 ± 3.63 | 17.21 ± 2.36 |
| floor_factual_r3 | b. đổi chủ ngữ/tân ngữ | 7.18 ± 2.30 | 24.99 ± 4.26 | 40.43 ± 4.17 | 17.80 ± 2.93 |
| main_factual_r3 | c. giữ nguyên | 10.16 ± 2.09 | 30.99 ± 3.67 | 47.62 ± 3.61 | 21.83 ± 2.58 |
| main_factual_r3 | a. đổi động từ | 6.82 ± 1.42 | 24.56 ± 2.60 | 40.98 ± 3.10 | 17.59 ± 1.81 |
| main_factual_r3 | b. đổi chủ ngữ/tân ngữ | 6.60 ± 1.29 | 23.79 ± 2.27 | 39.79 ± 2.68 | 17.15 ± 1.59 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.789; triples 0.684
- floor_factual_r3_seed0: objects 0.403; triples 0.875
- floor_factual_r3_seed1: objects 0.247; triples 0.872
- floor_factual_r3_seed2: objects 0.267; triples 0.881
- main_factual_r3 (mốc): triples 0.9762 ± 0.0101; objects 0.3330 ± 0.1232

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | floor_factual_r3 | main_factual_r3 | main_r3 |
|---|---|---|---|
| gat1 | 0.2341 ± 0.0318 | 0.2386 ± 0.0690 | 0.1486 ± 0.0122 |
| gat2 | 0.1776 ± 0.0293 | 0.1878 ± 0.0474 | 0.1201 ± 0.0099 |
| log_tau_objects | 0.0292 ± 0.0180 | 0.0070 ± 0.0045 | 0.0439 ± 0.0076 |
| log_tau_triples | 0.0035 ± 0.0044 | 0.0026 ± 0.0023 | 0.0017 ± 0.0014 |
| log_temperature | 0.0705 ± 0.0641 | 0.0243 ± 0.0155 | 0.0873 ± 0.0716 |
| p_edge | 0.0639 ± 0.0057 | 0.0628 ± 0.0127 | 0.0304 ± 0.0071 |
| p_node | 0.4169 ± 0.0196 | 0.3714 ± 0.0646 | 0.3221 ± 0.0447 |
| triple_mlp | 0.2080 ± 0.0145 | 0.2014 ± 0.0390 | 0.1353 ± 0.0050 |
| u | 0.1086 ± 0.0292 | 0.1266 ± 0.0458 | 0.0804 ± 0.0131 |
| u_objects | 0.0261 ± 0.0084 | 0.0249 ± 0.0062 | 0.0267 ± 0.0063 |
| u_triples | 0.0029 ± 0.0010 | 0.0016 ± 0.0002 | 0.0024 ± 0.0008 |
| weight_bias | 0.0661 ± 0.0168 | 0.0759 ± 0.0439 | 0.0429 ± 0.0243 |
| weight_map | 0.0638 ± 0.0117 | 0.0739 ± 0.0255 | 0.0537 ± 0.0136 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- floor_factual_r3: a/b/c = 0.474 / 0.334 / 0.192; τ_triples = 0.331; GPU peak 5.3 GB
- main_factual_r3: a/b/c = 0.478 / 0.350 / 0.172; τ_triples = 0.081; GPU peak 5.3 GB
- main_r3: a/b/c = 0.485 / 0.355 / 0.160; τ_triples = 0.138; GPU peak 15.5 GB

## Diễn biến theo epoch của floor_factual_r3: R@1 val, a/b/c (trung bình trên val), nhiệt độ loss T

- floor_factual_r3_seed0: e1 45.17 (0.600/0.280/0.120, T 0.5351); e2 46.26 (0.521/0.306/0.174, T 0.4471); e3 46.92 (0.496/0.314/0.190, T 0.4298); e4 47.16 (0.485/0.328/0.187, T 0.4263); e5 46.99 (0.480/0.332/0.187, T 0.4243); e6 47.00 (0.491/0.333/0.176, T 0.4221)
- floor_factual_r3_seed1: e1 45.51 (0.600/0.280/0.120, T 0.5366); e2 46.10 (0.508/0.300/0.191, T 0.4473); e3 46.51 (0.492/0.308/0.200, T 0.4303); e4 46.75 (0.486/0.322/0.193, T 0.4223); e5 46.67 (0.481/0.331/0.188, T 0.4212); e6 46.91 (0.483/0.331/0.187, T 0.4185); e7 46.57 (0.474/0.342/0.184, T 0.4158); e8 46.94 (0.478/0.344/0.178, T 0.4166); e9 46.73 (0.472/0.343/0.185, T 0.4087); e10 47.26 (0.474/0.332/0.195, T 0.4068)
- floor_factual_r3_seed2: e1 45.68 (0.600/0.280/0.120, T 0.5360); e2 46.67 (0.506/0.301/0.193, T 0.4468); e3 46.61 (0.490/0.322/0.189, T 0.4294); e4 46.69 (0.493/0.315/0.191, T 0.4253); e5 46.93 (0.478/0.339/0.183, T 0.4222); e6 46.99 (0.485/0.336/0.179, T 0.4188); e7 47.18 (0.471/0.351/0.177, T 0.4152); e8 47.31 (0.472/0.339/0.189, T 0.4132); e9 47.49 (0.463/0.344/0.193, T 0.4078); e10 47.12 (0.470/0.333/0.197, T 0.4059)

## Sàn có thật sự tác động không (val)

| Run | std kênh bộ ba (trung vị theo câu) | % câu dưới sàn bộ ba 0,0259 | std kênh vật thể (trung vị) | % câu dưới sàn vật thể |
|---|---|---|---|---|
| main_factual_r3_seed0 | 0.0033 | 99.9% | 0.0977 | 0.0% |
| main_factual_r3_seed1 | 0.0062 | 99.1% | 0.0936 | 0.1% |
| main_factual_r3_seed2 | 0.0056 | 100.0% | 0.1312 | 0.0% |
| floor_factual_r3_seed0 | 0.0196 | 79.8% | 0.0955 | 0.0% |
| floor_factual_r3_seed1 | 0.0250 | 55.1% | 0.1154 | 0.0% |
| floor_factual_r3_seed2 | 0.0244 | 58.7% | 0.1145 | 0.0% |

## Tương quan theo câu

- floor_factual_r3: vật thể ~ bộ ba 0.236 ± 0.036. main_factual_r3: 0,238 ± 0,034.

## Nhận xét

1. **Val không đổi.** R@1 47,31 ± 0,17, so với 47,23 ± 0,36 (+0,08); MRR 59,58 ± 0,15, so với 59,49 ± 0,29.
   Theo seed: 47,16 / 47,26 / 47,49.
2. **Độ dồn giảm rõ (chỉ số chính).** Cosine trung bình giữa các vector bộ ba là 0,872–0,881, so với 0,976 ở
   main_factual_r3 (lúc chưa train là 0,684).
3. **Sàn đúng là đang tác động.**
   - Ở main_factual_r3, std theo câu của kênh bộ ba chỉ khoảng 0,003–0,006 và nằm dưới sàn ở hơn 99% câu val: z-score
     đang kéo giãn các chênh lệch rất nhỏ lên 5–8 lần. Đây là bằng chứng trực tiếp cho giả thuyết 1 của phase
     (z-score che việc dồn).
   - Có sàn, mô hình phải tự trải điểm ra: std tăng lên khoảng 0,020–0,025, và chỉ còn 55–80% câu dưới sàn.
   - Kênh vật thể có std khoảng 0,1, không bao giờ chạm sàn.
4. **Nhưng gỡ dồn không làm kênh bộ ba tốt hơn.** Chỉ xếp theo kênh bộ ba: 9,18 ± 3,00, so với 9,52; tương quan với
   kênh vật thể 0,236 (không đổi); đổi động từ / đổi chủ–tân làm R@1 mất 1,05 / 0,88. Mô hình trải điểm ra một phần
   bằng cách làm attention mềm hơn: τ của kênh bộ ba tăng lên 0,331, so với 0,081.
5. **Trọng số** a/b/c = 0,474 / 0,334 / 0,192 (mốc 0,478 / 0,350 / 0,172). Không seed nào bị đẩy ra cực trị.
