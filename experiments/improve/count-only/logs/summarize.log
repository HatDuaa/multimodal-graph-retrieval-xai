## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | epoch tốt nhất | số epoch | thời gian train |
|---|---|---|---|---|---|---|---|
| countonly_factual_r3 | 48.91 ± 0.21 | 75.22 ± 0.05 | 84.96 ± 0.25 | 60.86 ± 0.22 | 8, 9, 5 | 10, 10, 7 | 7.03 h (3 seed) |
| fullgraph_factual_r3 | 48.38 ± 0.17 | 75.25 ± 0.09 | 84.71 ± 0.09 | 60.51 ± 0.17 | 4, 8, 5 | 6, 10, 7 | 4.40 h (3 seed) |
| groupcount_factual_r3 | 47.75 ± 0.22 | 74.82 ± 0.08 | 84.32 ± 0.17 | 59.97 ± 0.14 | 9, 4, 7 | 10, 6, 9 | 4.79 h (3 seed) |

## Chẩn đoán nhiễu phía câu — toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| countonly_factual_r3 | c. giữ nguyên | 48.91 ± 0.21 | 75.22 ± 0.05 | 84.96 ± 0.25 | 60.86 ± 0.22 |
| countonly_factual_r3 | a. đổi động từ | 47.75 ± 0.22 | 74.53 ± 0.16 | 84.21 ± 0.12 | 59.90 ± 0.18 |
| countonly_factual_r3 | b. đổi chủ ngữ/tân ngữ | 48.48 ± 0.08 | 75.13 ± 0.22 | 84.70 ± 0.21 | 60.49 ± 0.06 |
| fullgraph_factual_r3 | c. giữ nguyên | 48.38 ± 0.17 | 75.25 ± 0.09 | 84.71 ± 0.09 | 60.51 ± 0.17 |
| fullgraph_factual_r3 | a. đổi động từ | 47.36 ± 0.18 | 74.55 ± 0.14 | 84.06 ± 0.06 | 59.69 ± 0.13 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 48.00 ± 0.40 | 74.93 ± 0.25 | 84.46 ± 0.23 | 60.14 ± 0.31 |

## Chẩn đoán nhiễu phía câu — câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| countonly_factual_r3 | c. giữ nguyên | 49.38 ± 0.20 | 75.53 ± 0.02 | 85.19 ± 0.26 | 61.26 ± 0.21 |
| countonly_factual_r3 | a. đổi động từ | 48.19 ± 0.22 | 74.82 ± 0.18 | 84.42 ± 0.13 | 60.28 ± 0.19 |
| countonly_factual_r3 | b. đổi chủ ngữ/tân ngữ | 48.94 ± 0.08 | 75.45 ± 0.18 | 84.93 ± 0.23 | 60.89 ± 0.04 |
| fullgraph_factual_r3 | c. giữ nguyên | 48.82 ± 0.20 | 75.57 ± 0.06 | 84.93 ± 0.12 | 60.88 ± 0.17 |
| fullgraph_factual_r3 | a. đổi động từ | 47.77 ± 0.21 | 74.84 ± 0.11 | 84.27 ± 0.06 | 60.04 ± 0.14 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 48.42 ± 0.44 | 75.24 ± 0.22 | 84.68 ± 0.25 | 60.50 ± 0.32 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, câu có ≥1 bộ ba

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| countonly_factual_r3 | c. giữ nguyên | 12.12 ± 2.12 | 35.55 ± 2.59 | 51.69 ± 2.49 | 24.49 ± 2.22 |
| countonly_factual_r3 | a. đổi động từ | 8.23 ± 1.60 | 28.66 ± 2.94 | 44.76 ± 3.09 | 19.79 ± 1.96 |
| countonly_factual_r3 | b. đổi chủ ngữ/tân ngữ | 10.18 ± 1.80 | 32.29 ± 2.73 | 48.54 ± 3.06 | 22.19 ± 2.09 |
| fullgraph_factual_r3 | c. giữ nguyên | 13.14 ± 1.25 | 37.01 ± 1.75 | 53.17 ± 1.41 | 25.71 ± 1.31 |
| fullgraph_factual_r3 | a. đổi động từ | 9.02 ± 0.98 | 30.04 ± 1.62 | 46.74 ± 1.44 | 20.79 ± 1.06 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 10.57 ± 0.76 | 33.51 ± 1.25 | 49.70 ± 1.21 | 22.93 ± 0.85 |

## Chẩn đoán nhiễu phía câu — chỉ xếp theo điểm kênh thứ ba, toàn val

| Mô hình | Điều kiện | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| countonly_factual_r3 | c. giữ nguyên | 12.69 ± 2.06 | 36.24 ± 2.52 | 52.29 ± 2.42 | 25.07 ± 2.16 |
| countonly_factual_r3 | a. đổi động từ | 8.90 ± 1.55 | 29.54 ± 2.86 | 45.56 ± 3.01 | 20.50 ± 1.91 |
| countonly_factual_r3 | b. đổi chủ ngữ/tân ngữ | 10.81 ± 1.75 | 33.07 ± 2.65 | 49.23 ± 2.98 | 22.84 ± 2.03 |
| fullgraph_factual_r3 | c. giữ nguyên | 13.68 ± 1.22 | 37.67 ± 1.70 | 53.74 ± 1.37 | 26.26 ± 1.28 |
| fullgraph_factual_r3 | a. đổi động từ | 9.67 ± 0.95 | 30.88 ± 1.58 | 47.49 ± 1.40 | 21.48 ± 1.04 |
| fullgraph_factual_r3 | b. đổi chủ ngữ/tân ngữ | 11.18 ± 0.74 | 34.26 ± 1.21 | 50.36 ± 1.18 | 23.55 ± 0.83 |

## Độ dồn (cosine trung bình giữa các cặp ngẫu nhiên, 300 ảnh val)

- step0_frozen_clip: objects 0.790; triples 0.683
- countonly_factual_r3_seed0: objects 0.309; triples 0.875
- countonly_factual_r3_seed1: objects 0.259; triples 0.899
- countonly_factual_r3_seed2: objects 0.336; triples 0.867
- fullgraph_factual_r3 (mốc): triples 0.8693 ± 0.0445; objects 0.3339 ± 0.0769

## Gradient norm theo module (epoch tốt nhất, batch cuối, mean ± std qua 3 seed)

| Module | countonly_factual_r3 | fullgraph_factual_r3 | groupcount_factual_r3 |
|---|---|---|---|
| count_table | 0.1840 ± 0.0605 | — | 0.1677 ± 0.0517 |
| gat1 | 0.5527 ± 0.0704 | 0.4020 ± 0.0515 | 0.4169 ± 0.0505 |
| gat2 | 0.3905 ± 0.0229 | 0.3214 ± 0.0431 | 0.3240 ± 0.0261 |
| geometry_mlp | 0.1242 ± 0.0235 | 0.1084 ± 0.0159 | 0.1079 ± 0.0130 |
| global_predicate | 0.0701 ± 0.0477 | 0.0415 ± 0.0167 | 0.0403 ± 0.0210 |
| log_tau_objects | 0.0173 ± 0.0163 | 0.0133 ± 0.0108 | 0.0053 ± 0.0056 |
| log_tau_triples | 0.0148 ± 0.0154 | 0.0092 ± 0.0082 | 0.0124 ± 0.0022 |
| log_temperature | 0.0865 ± 0.0453 | 0.0909 ± 0.0689 | 0.0251 ± 0.0216 |
| none_predicate | 0.0644 ± 0.0137 | 0.0496 ± 0.0207 | 0.0492 ± 0.0083 |
| p_edge | 0.1176 ± 0.0357 | 0.0811 ± 0.0159 | 0.0896 ± 0.0081 |
| p_node | 0.8862 ± 0.2020 | 0.5754 ± 0.0535 | 0.5597 ± 0.0772 |
| triple_mlp | 0.4021 ± 0.0587 | 0.2571 ± 0.0672 | 0.3455 ± 0.1006 |
| u | 0.2453 ± 0.0133 | 0.2306 ± 0.0241 | 0.2394 ± 0.0180 |
| u_objects | 0.0205 ± 0.0056 | 0.0199 ± 0.0016 | 0.0180 ± 0.0021 |
| u_triples | 0.0038 ± 0.0006 | 0.0036 ± 0.0004 | 0.0037 ± 0.0008 |
| weight_bias | 0.0794 ± 0.0444 | 0.0665 ± 0.0210 | 0.0648 ± 0.0217 |
| weight_map | 0.0796 ± 0.0206 | 0.0657 ± 0.0123 | 0.0648 ± 0.0130 |

## Trọng số kênh a/b/c và τ kênh thứ ba (epoch tốt nhất)

- countonly_factual_r3: a/b/c = 0.571 / 0.125 / 0.304; τ_triples = 0.041; GPU peak 7.6 GB
- fullgraph_factual_r3: a/b/c = 0.618 / 0.135 / 0.247; τ_triples = 0.055; GPU peak 7.6 GB
- groupcount_factual_r3: a/b/c = 0.578 / 0.115 / 0.307; τ_triples = 0.053; GPU peak 5.6 GB

## Diễn biến theo epoch của countonly_factual_r3: R@1 val, a/b/c (trung bình trên val), nhiệt độ loss T

- countonly_factual_r3_seed0: e1 47.00 (0.726/0.212/0.062, T 0.0128); e2 47.55 (0.737/0.162/0.100, T 0.0120); e3 48.01 (0.708/0.155/0.137, T 0.0116); e4 48.37 (0.662/0.148/0.190, T 0.0110); e5 48.30 (0.630/0.136/0.234, T 0.0104); e6 48.51 (0.599/0.129/0.272, T 0.0098); e7 48.49 (0.568/0.124/0.308, T 0.0092); e8 49.15 (0.548/0.119/0.333, T 0.0088); e9 48.72 (0.528/0.112/0.360, T 0.0085); e10 48.58 (0.517/0.107/0.376, T 0.0081)
- countonly_factual_r3_seed1: e1 46.79 (0.726/0.212/0.062, T 0.0129); e2 47.46 (0.738/0.170/0.092, T 0.0121); e3 47.91 (0.713/0.162/0.125, T 0.0117); e4 48.33 (0.678/0.153/0.169, T 0.0111); e5 47.80 (0.640/0.144/0.217, T 0.0105); e6 48.60 (0.610/0.135/0.255, T 0.0099); e7 48.56 (0.571/0.130/0.299, T 0.0094); e8 48.69 (0.546/0.122/0.332, T 0.0089); e9 48.85 (0.527/0.121/0.352, T 0.0085); e10 48.78 (0.507/0.116/0.378, T 0.0081)
- countonly_factual_r3_seed2: e1 46.95 (0.726/0.212/0.062, T 0.0129); e2 47.22 (0.738/0.165/0.096, T 0.0121); e3 48.22 (0.704/0.158/0.138, T 0.0116); e4 47.98 (0.677/0.146/0.177, T 0.0111); e5 48.74 (0.636/0.137/0.227, T 0.0105); e6 48.63 (0.609/0.130/0.261, T 0.0098); e7 48.42 (0.564/0.127/0.309, T 0.0093)
