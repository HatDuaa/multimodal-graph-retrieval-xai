## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| factorized | 46.00 ± 0.02 | 73.33 ± 0.14 | 82.88 ± 0.02 | 58.35 ± 0.04 | 4, 4, 7 | 6, 6, 9 | 2.50 h (3 seed) |
| main_r3 | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 | 8, 10, 6 | 10, 10, 8 | 2.30 h (3 seed) |
| no_triple_r3 | 45.99 ± 0.05 | 73.33 ± 0.13 | 82.84 ± 0.10 | 58.34 ± 0.05 | 4, 4, 6 | 6, 6, 8 | 1.25 h (3 seed) |
| no_gat_r3 | 46.71 ± 0.07 | 73.81 ± 0.29 | 83.56 ± 0.24 | 59.04 ± 0.09 | 5, 5, 3 | 7, 7, 5 | 1.01 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| factorized | c. giữ nguyên | 46.00 ± 0.02 | 73.33 ± 0.14 | 82.88 ± 0.02 | 58.35 ± 0.04 |
| factorized | a. đổi động từ | 45.94 ± 0.06 | 73.21 ± 0.04 | 82.85 ± 0.04 | 58.31 ± 0.08 |
| factorized | b. đổi chủ ngữ/tân ngữ | 45.86 ± 0.13 | 73.32 ± 0.05 | 82.92 ± 0.13 | 58.23 ± 0.11 |
| main_r3 | c. giữ nguyên | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 |
| main_r3 | a. đổi động từ | 46.06 ± 0.15 | 73.41 ± 0.10 | 83.20 ± 0.22 | 58.49 ± 0.13 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 45.91 ± 0.15 | 73.60 ± 0.08 | 83.39 ± 0.12 | 58.46 ± 0.06 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| factorized | c. giữ nguyên | 46.81 ± 0.04 | 74.14 ± 0.20 | 83.57 ± 0.06 | 59.11 ± 0.06 |
| factorized | a. đổi động từ | 46.74 ± 0.09 | 74.00 ± 0.05 | 83.54 ± 0.06 | 59.07 ± 0.10 |
| factorized | b. đổi chủ ngữ/tân ngữ | 46.65 ± 0.17 | 74.13 ± 0.09 | 83.62 ± 0.18 | 58.97 ± 0.14 |
| main_r3 | c. giữ nguyên | 47.37 ± 0.29 | 74.72 ± 0.21 | 84.26 ± 0.30 | 59.72 ± 0.21 |
| main_r3 | a. đổi động từ | 46.89 ± 0.18 | 74.29 ± 0.13 | 83.98 ± 0.23 | 59.30 ± 0.14 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 46.71 ± 0.16 | 74.51 ± 0.11 | 84.20 ± 0.13 | 59.27 ± 0.04 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| factorized | c. giữ nguyên | 17.64 ± 0.21 | 39.70 ± 0.23 | 53.72 ± 0.22 | 29.27 ± 0.18 |
| factorized | a. đổi động từ | 15.92 ± 0.44 | 38.14 ± 0.18 | 52.92 ± 0.23 | 27.73 ± 0.34 |
| factorized | b. đổi chủ ngữ/tân ngữ | 12.19 ± 0.37 | 31.84 ± 0.44 | 46.82 ± 0.19 | 23.33 ± 0.11 |
| main_r3 | c. giữ nguyên | 6.24 ± 0.14 | 24.97 ± 0.64 | 41.96 ± 0.59 | 17.37 ± 0.23 |
| main_r3 | a. đổi động từ | 4.77 ± 0.26 | 21.34 ± 0.50 | 37.46 ± 0.31 | 15.12 ± 0.25 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 4.69 ± 0.11 | 20.07 ± 0.37 | 35.82 ± 0.43 | 14.58 ± 0.23 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| factorized | c. giữ nguyên | 20.17 ± 0.18 | 43.17 ± 0.19 | 56.88 ± 0.19 | 32.05 ± 0.15 |
| factorized | a. đổi động từ | 18.69 ± 0.38 | 41.83 ± 0.15 | 56.19 ± 0.20 | 30.73 ± 0.29 |
| factorized | b. đổi chủ ngữ/tân ngữ | 15.49 ± 0.32 | 36.42 ± 0.38 | 50.96 ± 0.16 | 26.95 ± 0.10 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.789; triples 0.684
- step0_factorized: objects 0.789; triples 0.771; triple_slots (subject 0.801, predicate 0.906, object 0.798)
- factorized_seed0: objects 0.431; triples 0.503; triple_slots (subject 0.446, predicate 0.906, object 0.460)
- factorized_seed1: objects 0.513; triples 0.565; triple_slots (subject 0.542, predicate 0.906, object 0.538)
- factorized_seed2: objects 0.396; triples 0.479; triple_slots (subject 0.408, predicate 0.906, object 0.427)
- main_r3 (cấp 1): triples 0.9661 ± 0.0083; objects 0.2546 ± 0.0308

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | factorized | main_r3 | no_triple_r3 | no_gat_r3 |
|---|---|---|---|---|
| gat1 | 0.1379 ± 0.0333 | 0.1486 ± 0.0122 | 0.1697 ± 0.0380 | — |
| gat2 | 0.1133 ± 0.0251 | 0.1201 ± 0.0099 | 0.1305 ± 0.0169 | — |
| log_tau_objects | 0.0162 ± 0.0126 | 0.0439 ± 0.0076 | 0.0354 ± 0.0330 | 0.0575 ± 0.0084 |
| log_tau_triples | 0.0026 ± 0.0009 | 0.0017 ± 0.0014 | 0.0000 ± 0.0000 | 0.0034 ± 0.0027 |
| log_temperature | 0.0313 ± 0.0297 | 0.0873 ± 0.0716 | 0.0188 ± 0.0107 | 0.0156 ± 0.0127 |
| p_edge | 0.0034 ± 0.0020 | 0.0304 ± 0.0071 | 0.0070 ± 0.0027 | 0.0783 ± 0.0040 |
| p_node | 0.2419 ± 0.0354 | 0.3221 ± 0.0447 | 0.2843 ± 0.0499 | 0.2834 ± 0.0202 |
| triple_mlp | — | 0.1353 ± 0.0050 | — | 0.1648 ± 0.0051 |
| u | 0.0865 ± 0.0201 | 0.0804 ± 0.0131 | 0.0965 ± 0.0122 | 0.0925 ± 0.0018 |
| u_objects | 0.0233 ± 0.0042 | 0.0267 ± 0.0063 | 0.0250 ± 0.0045 | 0.0223 ± 0.0029 |
| u_triples | 0.0011 ± 0.0001 | 0.0024 ± 0.0008 | 0.0000 ± 0.0000 | 0.0018 ± 0.0008 |
| weight_bias | 0.0194 ± 0.0119 | 0.0429 ± 0.0243 | 0.0271 ± 0.0152 | 0.0295 ± 0.0126 |
| weight_map | 0.0381 ± 0.0037 | 0.0537 ± 0.0136 | 0.0430 ± 0.0056 | 0.0477 ± 0.0019 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- factorized: a/b/c = 0.500 / 0.429 / 0.071; τ_triples = 0.082; GPU peak 6.9 GB
- main_r3: a/b/c = 0.485 / 0.355 / 0.160; τ_triples = 0.138; GPU peak 15.5 GB
- no_triple_r3: a/b/c = 0.504 / 0.496 / 0.000; τ_triples = 0.050; GPU peak 13.0 GB
- no_gat_r3: a/b/c = 0.473 / 0.353 / 0.174; τ_triples = 0.077; GPU peak 12.4 GB
