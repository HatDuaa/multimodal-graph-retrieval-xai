## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| groupcount_factual_r3 | 47.75 ± 0.22 | 74.82 ± 0.08 | 84.32 ± 0.17 | 59.97 ± 0.14 | 9, 4, 7 | 10, 6, 9 | 4.79 h (3 seed) |
| fullgraph_factual_r3 | 48.38 ± 0.17 | 75.25 ± 0.09 | 84.71 ± 0.09 | 60.51 ± 0.17 | 4, 8, 5 | 6, 10, 7 | 4.40 h (3 seed) |
| raw_factual_r3 | 47.48 ± 0.10 | 74.70 ± 0.18 | 84.10 ± 0.08 | 59.72 ± 0.00 | 9, 10, 9 | 10, 10, 10 | 2.59 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| groupcount_factual_r3 | c. giữ nguyên | 47.75 ± 0.22 | 74.82 ± 0.08 | 84.32 ± 0.17 | 59.97 ± 0.14 |
| groupcount_factual_r3 | a. đổi động từ | 46.62 ± 0.33 | 74.25 ± 0.05 | 83.77 ± 0.07 | 59.05 ± 0.17 |
| groupcount_factual_r3 | b. đổi chủ ngữ/tân ngữ | 47.42 ± 0.27 | 74.73 ± 0.09 | 84.15 ± 0.22 | 59.69 ± 0.16 |
| fullgraph_factual_r3 | c. giữ nguyên | 48.38 ± 0.17 | 75.25 ± 0.09 | 84.71 ± 0.09 | 60.51 ± 0.17 |
| fullgraph_factual_r3 | a. đổi động từ | 47.36 ± 0.18 | 74.55 ± 0.14 | 84.06 ± 0.06 | 59.69 ± 0.13 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 48.00 ± 0.40 | 74.93 ± 0.25 | 84.46 ± 0.23 | 60.14 ± 0.31 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| groupcount_factual_r3 | c. giữ nguyên | 48.16 ± 0.25 | 75.16 ± 0.05 | 84.56 ± 0.20 | 60.34 ± 0.17 |
| groupcount_factual_r3 | a. đổi động từ | 46.99 ± 0.32 | 74.56 ± 0.03 | 84.00 ± 0.06 | 59.39 ± 0.17 |
| groupcount_factual_r3 | b. đổi chủ ngữ/tân ngữ | 47.81 ± 0.26 | 75.06 ± 0.06 | 84.39 ± 0.25 | 60.06 ± 0.17 |
| fullgraph_factual_r3 | c. giữ nguyên | 48.82 ± 0.20 | 75.57 ± 0.06 | 84.93 ± 0.12 | 60.88 ± 0.17 |
| fullgraph_factual_r3 | a. đổi động từ | 47.77 ± 0.21 | 74.84 ± 0.11 | 84.27 ± 0.06 | 60.04 ± 0.14 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 48.42 ± 0.44 | 75.24 ± 0.22 | 84.68 ± 0.25 | 60.50 ± 0.32 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| groupcount_factual_r3 | c. giữ nguyên | 11.28 ± 2.17 | 34.52 ± 3.81 | 50.96 ± 3.38 | 23.67 ± 2.54 |
| groupcount_factual_r3 | a. đổi động từ | 7.79 ± 1.90 | 27.98 ± 3.47 | 44.65 ± 3.81 | 19.28 ± 2.43 |
| groupcount_factual_r3 | b. đổi chủ ngữ/tân ngữ | 9.64 ± 2.08 | 32.03 ± 4.09 | 48.66 ± 3.55 | 21.81 ± 2.64 |
| fullgraph_factual_r3 | c. giữ nguyên | 13.14 ± 1.25 | 37.01 ± 1.75 | 53.17 ± 1.41 | 25.71 ± 1.31 |
| fullgraph_factual_r3 | a. đổi động từ | 9.02 ± 0.98 | 30.04 ± 1.62 | 46.74 ± 1.44 | 20.79 ± 1.06 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 10.57 ± 0.76 | 33.51 ± 1.25 | 49.70 ± 1.21 | 22.93 ± 0.85 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| groupcount_factual_r3 | c. giữ nguyên | 11.88 ± 2.11 | 35.24 ± 3.71 | 51.59 ± 3.29 | 24.27 ± 2.48 |
| groupcount_factual_r3 | a. đổi động từ | 8.48 ± 1.85 | 28.88 ± 3.37 | 45.45 ± 3.70 | 20.00 ± 2.36 |
| groupcount_factual_r3 | b. đổi chủ ngữ/tân ngữ | 10.28 ± 2.03 | 32.82 ± 3.98 | 49.35 ± 3.46 | 22.46 ± 2.57 |
| fullgraph_factual_r3 | c. giữ nguyên | 13.68 ± 1.22 | 37.67 ± 1.70 | 53.74 ± 1.37 | 26.26 ± 1.28 |
| fullgraph_factual_r3 | a. đổi động từ | 9.67 ± 0.95 | 30.88 ± 1.58 | 47.49 ± 1.40 | 21.48 ± 1.04 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 11.18 ± 0.74 | 34.26 ± 1.21 | 50.36 ± 1.18 | 23.55 ± 0.83 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.788; triples 0.688
- groupcount_factual_r3_seed0: objects 0.268; triples 0.881
- groupcount_factual_r3_seed1: objects 0.358; triples 0.763
- groupcount_factual_r3_seed2: objects 0.314; triples 0.869
- fullgraph_factual_r3 (mốc): triples 0.8693 ± 0.0445; objects 0.3339 ± 0.0769

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | groupcount_factual_r3 | fullgraph_factual_r3 | raw_factual_r3 |
|---|---|---|---|
| count_table | 0.1677 ± 0.0517 | — | — |
| gat1 | 0.4169 ± 0.0505 | 0.4020 ± 0.0515 | 0.1785 ± 0.0331 |
| gat2 | 0.3240 ± 0.0261 | 0.3214 ± 0.0431 | 0.1350 ± 0.0220 |
| geometry_mlp | 0.1079 ± 0.0130 | 0.1084 ± 0.0159 | — |
| global_predicate | 0.0403 ± 0.0210 | 0.0415 ± 0.0167 | — |
| log_tau_objects | 0.0053 ± 0.0056 | 0.0133 ± 0.0108 | 0.0459 ± 0.0088 |
| log_tau_triples | 0.0124 ± 0.0022 | 0.0092 ± 0.0082 | 0.0058 ± 0.0085 |
| log_temperature | 0.0251 ± 0.0216 | 0.0909 ± 0.0689 | 0.0336 ± 0.0113 |
| none_predicate | 0.0492 ± 0.0083 | 0.0496 ± 0.0207 | — |
| p_edge | 0.0896 ± 0.0081 | 0.0811 ± 0.0159 | 0.0527 ± 0.0145 |
| p_node | 0.5597 ± 0.0772 | 0.5754 ± 0.0535 | 0.3872 ± 0.0733 |
| triple_mlp | 0.3455 ± 0.1006 | 0.2571 ± 0.0672 | 0.1857 ± 0.0452 |
| u | 0.2394 ± 0.0180 | 0.2306 ± 0.0241 | 0.0784 ± 0.0088 |
| u_objects | 0.0180 ± 0.0021 | 0.0199 ± 0.0016 | 0.0242 ± 0.0079 |
| u_triples | 0.0037 ± 0.0008 | 0.0036 ± 0.0004 | 0.0025 ± 0.0002 |
| weight_bias | 0.0648 ± 0.0217 | 0.0665 ± 0.0210 | 0.0875 ± 0.0105 |
| weight_map | 0.0648 ± 0.0130 | 0.0657 ± 0.0123 | 0.0785 ± 0.0046 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- groupcount_factual_r3: a/b/c = 0.578 / 0.115 / 0.307; τ_triples = 0.053; GPU peak 5.6 GB
- fullgraph_factual_r3: a/b/c = 0.618 / 0.135 / 0.247; τ_triples = 0.055; GPU peak 7.6 GB
- raw_factual_r3: a/b/c = 0.467 / 0.078 / 0.455; τ_triples = 0.232; GPU peak 5.3 GB

## Diễn biến theo epoch của groupcount_factual_r3: R@1 val, a/b/c (trung bình trên val), nhiệt độ loss T

- groupcount_factual_r3_seed0: e1 46.42 (0.726/0.212/0.062, T 0.0127); e2 47.04 (0.744/0.157/0.100, T 0.0119); e3 46.96 (0.704/0.140/0.157, T 0.0113); e4 47.26 (0.645/0.130/0.226, T 0.0105); e5 47.01 (0.604/0.116/0.280, T 0.0097); e6 47.55 (0.573/0.107/0.319, T 0.0090); e7 47.63 (0.544/0.103/0.353, T 0.0086); e8 47.70 (0.525/0.098/0.378, T 0.0081); e9 47.93 (0.499/0.097/0.404, T 0.0078); e10 46.96 (0.492/0.093/0.416, T 0.0075)
- groupcount_factual_r3_seed1: e1 46.12 (0.726/0.212/0.062, T 0.0127); e2 46.63 (0.744/0.166/0.090, T 0.0120); e3 46.96 (0.718/0.154/0.128, T 0.0115); e4 47.50 (0.687/0.138/0.175, T 0.0109); e5 47.48 (0.642/0.128/0.230, T 0.0103); e6 47.31 (0.605/0.119/0.277, T 0.0096)
- groupcount_factual_r3_seed2: e1 46.05 (0.726/0.212/0.062, T 0.0127); e2 46.61 (0.742/0.164/0.093, T 0.0119); e3 47.00 (0.713/0.151/0.135, T 0.0115); e4 47.36 (0.672/0.135/0.193, T 0.0107); e5 47.00 (0.625/0.120/0.255, T 0.0101); e6 47.79 (0.590/0.114/0.296, T 0.0093); e7 47.82 (0.548/0.111/0.341, T 0.0088); e8 47.61 (0.533/0.105/0.362, T 0.0084); e9 47.30 (0.506/0.098/0.395, T 0.0080)
