## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| graph_channel | 46.23 ± 0.13 | 73.97 ± 0.18 | 83.55 ± 0.16 | 58.76 ± 0.06 | 5, 4, 4 | 7, 6, 6 | 1.46 h (3 seed) |
| main_r3 | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 | 8, 10, 6 | 10, 10, 8 | 2.30 h (3 seed) |
| no_triple_r3 | 45.99 ± 0.05 | 73.33 ± 0.13 | 82.84 ± 0.10 | 58.34 ± 0.05 | 4, 4, 6 | 6, 6, 8 | 1.25 h (3 seed) |
| no_gat_r3 | 46.71 ± 0.07 | 73.81 ± 0.29 | 83.56 ± 0.24 | 59.04 ± 0.09 | 5, 5, 3 | 7, 7, 5 | 1.01 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| graph_channel | c. giữ nguyên | 46.23 ± 0.13 | 73.97 ± 0.18 | 83.55 ± 0.16 | 58.76 ± 0.06 |
| graph_channel | a. đổi động từ | 46.27 ± 0.10 | 73.90 ± 0.24 | 83.59 ± 0.14 | 58.77 ± 0.03 |
| graph_channel | b. đổi chủ ngữ/tân ngữ | 45.96 ± 0.08 | 73.91 ± 0.14 | 83.60 ± 0.02 | 58.60 ± 0.03 |
| main_r3 | c. giữ nguyên | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 |
| main_r3 | a. đổi động từ | 46.06 ± 0.15 | 73.41 ± 0.10 | 83.20 ± 0.22 | 58.49 ± 0.13 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 45.91 ± 0.15 | 73.60 ± 0.08 | 83.39 ± 0.12 | 58.46 ± 0.06 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| graph_channel | c. giữ nguyên | 47.06 ± 0.13 | 74.77 ± 0.18 | 84.16 ± 0.20 | 59.54 ± 0.04 |
| graph_channel | a. đổi động từ | 47.11 ± 0.11 | 74.68 ± 0.25 | 84.20 ± 0.20 | 59.54 ± 0.04 |
| graph_channel | b. đổi chủ ngữ/tân ngữ | 46.75 ± 0.07 | 74.69 ± 0.12 | 84.22 ± 0.05 | 59.34 ± 0.01 |
| main_r3 | c. giữ nguyên | 47.37 ± 0.29 | 74.72 ± 0.21 | 84.26 ± 0.30 | 59.72 ± 0.21 |
| main_r3 | a. đổi động từ | 46.89 ± 0.18 | 74.29 ± 0.13 | 83.98 ± 0.23 | 59.30 ± 0.14 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 46.71 ± 0.16 | 74.51 ± 0.11 | 84.20 ± 0.13 | 59.27 ± 0.04 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| graph_channel | c. giữ nguyên | 15.43 ± 0.36 | 40.35 ± 0.44 | 57.94 ± 0.42 | 28.40 ± 0.39 |
| graph_channel | a. đổi động từ | 15.33 ± 0.41 | 40.30 ± 0.60 | 57.80 ± 0.39 | 28.35 ± 0.40 |
| graph_channel | b. đổi chủ ngữ/tân ngữ | 15.43 ± 0.54 | 40.47 ± 0.51 | 57.88 ± 0.42 | 28.40 ± 0.52 |
| main_r3 | c. giữ nguyên | 6.24 ± 0.14 | 24.97 ± 0.64 | 41.96 ± 0.59 | 17.37 ± 0.23 |
| main_r3 | a. đổi động từ | 4.77 ± 0.26 | 21.34 ± 0.50 | 37.46 ± 0.31 | 15.12 ± 0.25 |
| main_r3 | b. đổi chủ ngữ/tân ngữ | 4.69 ± 0.11 | 20.07 ± 0.37 | 35.82 ± 0.43 | 14.58 ± 0.23 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| graph_channel | c. giữ nguyên | 15.23 ± 0.43 | 39.85 ± 0.36 | 57.31 ± 0.33 | 28.08 ± 0.43 |
| graph_channel | a. đổi động từ | 15.15 ± 0.48 | 39.81 ± 0.50 | 57.19 ± 0.32 | 28.04 ± 0.44 |
| graph_channel | b. đổi chủ ngữ/tân ngữ | 15.23 ± 0.58 | 39.95 ± 0.42 | 57.26 ± 0.39 | 28.08 ± 0.53 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.789; triples 0.684
- step0_graph: objects 0.789; graph_vectors 0.960
- graph_channel_seed0: objects 0.110; graph_vectors 0.314
- graph_channel_seed1: objects 0.133; graph_vectors 0.408
- graph_channel_seed2: objects 0.202; graph_vectors 0.473
- main_r3 (cấp 1): triples 0.9661 ± 0.0083; objects 0.2546 ± 0.0308

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | graph_channel | main_r3 | no_triple_r3 | no_gat_r3 |
|---|---|---|---|---|
| gat1 | 0.1798 ± 0.0063 | 0.1486 ± 0.0122 | 0.1697 ± 0.0380 | — |
| gat2 | 0.1382 ± 0.0082 | 0.1201 ± 0.0099 | 0.1305 ± 0.0169 | — |
| log_tau_objects | 0.0066 ± 0.0033 | 0.0439 ± 0.0076 | 0.0354 ± 0.0330 | 0.0575 ± 0.0084 |
| log_tau_triples | — | 0.0017 ± 0.0014 | 0.0000 ± 0.0000 | 0.0034 ± 0.0027 |
| log_temperature | 0.0506 ± 0.0353 | 0.0873 ± 0.0716 | 0.0188 ± 0.0107 | 0.0156 ± 0.0127 |
| p_edge | 0.0093 ± 0.0047 | 0.0304 ± 0.0071 | 0.0070 ± 0.0027 | 0.0783 ± 0.0040 |
| p_node | 0.3231 ± 0.0348 | 0.3221 ± 0.0447 | 0.2843 ± 0.0499 | 0.2834 ± 0.0202 |
| triple_mlp | — | 0.1353 ± 0.0050 | — | 0.1648 ± 0.0051 |
| u | 0.1036 ± 0.0095 | 0.0804 ± 0.0131 | 0.0965 ± 0.0122 | 0.0925 ± 0.0018 |
| u_objects | 0.0270 ± 0.0025 | 0.0267 ± 0.0063 | 0.0250 ± 0.0045 | 0.0223 ± 0.0029 |
| u_pool_image | 0.0160 ± 0.0015 | — | — | — |
| u_pool_query | 0.0156 ± 0.0019 | — | — | — |
| u_triples | — | 0.0024 ± 0.0008 | 0.0000 ± 0.0000 | 0.0018 ± 0.0008 |
| weight_bias | 0.0541 ± 0.0517 | 0.0429 ± 0.0243 | 0.0271 ± 0.0152 | 0.0295 ± 0.0126 |
| weight_map | 0.0614 ± 0.0296 | 0.0537 ± 0.0136 | 0.0430 ± 0.0056 | 0.0477 ± 0.0019 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- graph_channel: a/b/c = 0.485 / 0.334 / 0.181; τ_triples = 0.050; GPU peak 4.8 GB
- main_r3: a/b/c = 0.485 / 0.355 / 0.160; τ_triples = 0.138; GPU peak 15.5 GB
- no_triple_r3: a/b/c = 0.504 / 0.496 / 0.000; τ_triples = 0.050; GPU peak 13.0 GB
- no_gat_r3: a/b/c = 0.473 / 0.353 / 0.174; τ_triples = 0.077; GPU peak 12.4 GB
