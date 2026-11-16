# Bước 3 — Thí nghiệm B: thay kênh bộ ba bằng kênh mức đồ thị

Lệnh chạy lại: `run.sh`. Bảng dưới sinh bằng `python scripts/improve/summarize_improve.py --exp experiments/improve/graph-channel --prefix graph_channel --label graph_channel`
(ghi `summary.md`, `summary.json`). Chẩn đoán: `triple_perturbation.json`. Độ dồn: `part_similarity.json`. Log từng run:
`graph_channel_seed*/log.jsonl`. Timeline: `logs/timeline.log`. Chỉ dùng val.

## Thay đổi duy nhất so với main_r3

Cờ `--graph-channel` (`src/models/graph_reranker.py`) thay kênh bộ ba bằng kênh mức đồ thị:

- Mỗi đồ thị (ảnh hoặc câu) được thu về một vector bằng attention pooling trên các vector node sau GAT, cũng chính
  là vector của kênh vật thể: `g = normalize(Σ_n α_n·node_n)`, `α = softmax_n(node_n · u_pool)`.
- Phía ảnh và phía câu có `u_pool` riêng (`u_pool_image`, `u_pool_query`). Cả hai khởi tạo bằng 0, tức ban đầu là
  mean pooling.
- Điểm kênh là `cos(g_câu, g_ảnh)`.
- Trọng số pooling α được giữ trong `details["pool_query"]`, `details["pool_image"]` để làm giải thích. Mục
  `graph_level` của `matched_pairs` (`src/explain/graph_explainer.py`) liệt kê α của từng node ở hai phía và phần
  đóng góp `c · cos` (trước z-score).
- Mọi thứ khác giữ nguyên: kênh CLIP, kênh vật thể, z-score, a/b/c (c giờ là trọng số của kênh đồ thị), loss,
  công thức train, 3 seed.

Hạ tầng: micro-batch 32, batch hiệu dụng 256 (`../README.md`, `../microbatch-control/REPORT.md`). Mỗi epoch mất
khoảng 276 giây, GPU đỉnh 4,8 GB. Train 3 seed mất tổng 1,46 giờ, chạy từ 04:23 đến 05:52 ngày 18/10/2026, không
lần nào bị lỗi.

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

## Tương quan với kênh vật thể (`../channel-correlation/REPORT.md`)

Tương quan theo câu giữa điểm kênh vật thể và kênh thứ ba: graph_channel 0,518 ± 0,023; factorized 0,587 ± 0,004;
main_r3 0,197 ± 0,009.

## Nhận xét

1. **Điểm cuối nằm giữa no_triple và main_r3.** R@1 val là 46,23 ± 0,13, thấp hơn main_r3 0,25 điểm (khoảng 1 std
   của main_r3) và cao hơn no_triple_r3 0,24 điểm. R@5 (73,97 so với 73,78) và R@10 (83,55 so với 83,44) nhỉnh hơn
   main_r3 một chút, MRR thấp hơn 0,09. Không có cải thiện rõ ràng.
2. **Kênh đồ thị khi đứng riêng tốt hơn kênh cụm sau train, nhưng mù vị từ và mù thứ tự, đúng như cấu trúc của nó.**
   Chỉ xếp theo cosine đồ thị: R@1 15,23 ± 0,43 trên toàn val, 15,43 trên câu có ≥1 bộ ba. Main_r3 đạt 6,24; cụm
   CLIP đóng băng đạt 18,84. Đổi động từ hoặc đổi chủ ngữ/tân ngữ không làm điểm kênh này đổi (15,33 và 15,43),
   vì pooling cộng các node lại, không nhìn cạnh. Nhiễu chỉ đi vào qua cạnh GAT phía câu. Trên điểm cuối, đổi chủ
   ngữ/tân ngữ làm R@1 giảm 0,27 (giống mức −0,18 của no_triple), còn đổi động từ không có tác dụng (+0,04).
3. **Vector đồ thị dồn rất mạnh lúc khởi tạo nhưng train đã gỡ ra.** Với mean pooling trên node CLIP đóng băng,
   cosine trung bình giữa các ảnh là 0,960. Sau train còn 0,31–0,47. Vector node cũng tách xa hơn (0,11–0,20, so
   với 0,25 ở main_r3).
4. **Vì sao không giúp nhiều.** Kênh đồ thị tương quan 0,52 với kênh vật thể, gấp 2,6 lần kênh cụm của main_r3. Cả
   hai kênh dựng từ cùng vector node, nên sau z-score phần lớn thông tin bị lặp. Mô hình vẫn cho kênh này trọng số
   c = 0,181, cao hơn main_r3 (0,160), và lấy bớt trọng số của kênh vật thể (b = 0,334 so với 0,355).
5. **Gradient.** `u_pool_image` và `u_pool_query` nhận gradient khoảng 0,016, cùng cỡ `u_objects` (0,027). GAT nhận
   gradient lớn hơn main_r3 một chút (gat1 0,180 so với 0,149), vì giờ hai kênh cùng đẩy ngược về vector node.
   `triple_mlp`, `u_triples`, `log_tau_triples` không được dùng: τ_triples đứng yên ở 0,050.
6. **Giải thích.** Trọng số pooling α cho biết node nào chi phối vector đồ thị. Nhưng điểm kênh là một cosine duy
   nhất, nên không ghép được từng phần của câu với từng phần của ảnh như kênh vật thể hay kênh bộ ba. Về mặt giải
   thích, kênh này kém chi tiết hơn kênh bộ ba.
