# Phase cải tiến — kênh bộ ba

Mọi số trong thư mục này là của **val**. Test chưa được chạm. Mỗi thí nghiệm chỉ đổi đúng một thứ so với `main_r3`
(`experiments/level1/gat/main_r3_seed*`) và chạy 3 seed (0, 1, 2). Số được tổng hợp bằng `src/eval/metrics.py`
(mean ± std, ddof = 1).

| Thư mục | Nội dung |
|---|---|
| `triple-perturbation/` | Bước 1: đổi động từ hoặc đổi chủ ngữ/tân ngữ phía câu, trên các checkpoint cấp 1 (không train) |
| `microbatch-control/` | Đối chứng cho việc giảm micro-batch 128 → 32 (xem mục Hạ tầng) |
| `triple-factorized/` | Bước 2, thí nghiệm A: so khớp bộ ba tách theo thành phần, `sim = c(s)·c(p)·c(o)` |
| `graph-channel/` | Bước 3, thí nghiệm B: thay kênh bộ ba bằng kênh mức đồ thị (attention pooling + cosine) |
| `channel-correlation/` | Tương quan theo câu giữa các kênh (không train) |
| `parser-compare/` | Đợt 2, phần 1: sng / FACTUAL-T5 base, large / Qwen2.5-3B trên val (nội tại, probe, main_r3) |
| `geometry-channel/` | Đợt 2, phần 2: kênh số lượng + vị trí từ khung VG, cộng vào main_r3 (không train) |
| `factual-retrain/` | Bước 1d: train lại công thức main_r3 với đồ thị câu FACTUAL-T5 large (dữ liệu `factual_v1`) |
| `fusion/` | Đợt 3, thí nghiệm 1: bỏ z-score (1a) và sàn cho z-score (1b), mốc main_factual_r3 |
| `graph-matching/` | Đợt 3, thí nghiệm 2: ghép cả đồ thị (FGW, nhất quán node chung), probe không train |
| `graph-dedup-audit/` | Audit làm sạch đồ thị ảnh: gộp khung trùng, gom nhóm theo vai trò (CPU, không train) |
| `full-graph/` | Đợt 4, thí nghiệm 3: train trên graph ảnh gộp trùng + hình học cạnh + kNN liền + node tổng, mốc 1a |

Mỗi thư mục có `run.sh` (lệnh chạy lại), `REPORT.md` (kết quả và nhận xét), log và file kết quả. Checkpoint `*.pt`
không commit (`.gitignore`). Ở mỗi run, `config.json` lưu đủ cờ, seed và commit.

## Hạ tầng (máy RTX 4090, 18/10/2026)

- **NVML.** Driver NVIDIA tự cập nhật ngày 15/10 mà chưa reboot: kernel module 595.84, thư viện 595.99. Vì vậy
  NVML không khởi tạo được. Chế độ `expandable_segments:True` mà script train đặt mặc định lại gọi NVML, nên mọi
  job CUDA đều chết. Mọi lệnh ở đây chạy với `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Biến này chỉ đổi
  cách cấp phát bộ nhớ, không đổi phép tính. Kiểm tra: chấm lại `main_r3_seed0` ra R@1 val 0,46666039687764505,
  trùng từng chữ số với `metrics.json` cấp 1.
- **Micro-batch.** Không có `expandable_segments` thì bộ nhớ cache phân mảnh. Ngay công thức main_r3 ở
  micro-batch 128 đã giữ tới 22,9/24 GB (bảng bộ nhớ trong `experiments/level1/README.md`), và bản factorized tràn
  bộ nhớ ở batch đầu tiên (`triple-factorized/archive-oom-mb128/`). Vì thế mọi run cải tiến dùng micro-batch 32,
  batch hiệu dụng vẫn 256 (gradient cộng dồn đúng tỉ lệ). Mô hình không có batch-norm, z-score và loss tính theo
  từng câu, nên gradient mỗi bước tối ưu chỉ khác ở thứ tự cộng số thực và ở cách rút mặt nạ dropout.
  `microbatch-control/` kiểm tra lại điều này trên công thức main_r3.

## Kết quả tổng hợp (val, 3 seed, mean ± std)

| Mô hình | R@1 | R@5 | R@10 | MRR | kênh 3 đứng riêng, R@1 (c / a / b) | tương quan vật thể ~ kênh 3 |
|---|---|---|---|---|---|---|
| main_r3 | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 | 6.24 / 4.77 / 4.69 | 0.197 |
| no_triple_r3 | 45.99 ± 0.05 | 73.33 ± 0.13 | 82.84 ± 0.10 | 58.34 ± 0.05 | — | — |
| no_gat_r3 | 46.71 ± 0.07 | 73.81 ± 0.29 | 83.56 ± 0.24 | 59.04 ± 0.09 | — | 0.209 |
| A: factorized | 46.00 ± 0.02 | 73.33 ± 0.14 | 82.88 ± 0.02 | 58.35 ± 0.04 | 17.64 / 15.92 / 12.19 | 0.587 |
| B: graph_channel | 46.23 ± 0.13 | 73.97 ± 0.18 | 83.55 ± 0.16 | 58.76 ± 0.06 | 15.43 / 15.33 / 15.43 | 0.518 |

"Kênh 3 đứng riêng": xếp top-50 chỉ theo điểm kênh thứ ba, trên các câu có ≥1 bộ ba. c là giữ nguyên, a là đổi
động từ, b là đổi chủ ngữ/tân ngữ. Mốc tham chiếu: cụm CLIP đóng băng ở step 0 đạt 18,84 / 16,31 / 17,70.
Tương quan được tính theo từng câu, trên top-50 (`channel-correlation/`).

Tóm tắt:

- **Bước 1.** Kênh bộ ba của main_r3 có dùng vị từ và chiều, nhưng khi đứng riêng rất yếu (R@1 6,24, do vector dồn
  về cosine 0,97). Toàn bộ 0,49 điểm R@1 nó đóng góp phụ thuộc vào vị từ và chiều đúng.
- **Bước 2 (A).** Tách bộ ba theo thành phần đã gỡ được độ dồn, kênh đứng riêng mạnh gấp 3 và nhạy thứ tự gấp 3,5.
  Nhưng điểm cuối giảm về mức no_triple, vì kênh mới lặp lại kênh vật thể (tương quan 0,59) và mô hình hạ trọng số
  c xuống 0,07.
- **Bước 3 (B).** Kênh mức đồ thị đứng giữa no_triple và main_r3 về R@1, nhỉnh hơn main_r3 một chút ở R@5/R@10.
  Kênh này mù vị từ và thứ tự, lặp kênh vật thể (tương quan 0,52), và giải thích kém chi tiết hơn.
- Trên val, chưa thí nghiệm nào trong phase này vượt main_r3 về R@1/MRR. Chọn mô hình nào để chấm test một lần cuối
  là việc nhóm quyết định.

## Đợt 2: parser và kênh hình học (18/10/2026, branch `feat/parser-and-geometry`)

- **Parser** (`parser-compare/REPORT.md`): FACTUAL-T5 sửa đúng các lỗi của sng. Node đại từ giảm 3,9% → 0,1%; câu
  có ≥1 bộ ba tăng 85,8% → 97,2%; vật thể có trong ảnh đúng tăng 46,4% → 53,9%. Nhưng trên probe 3 kênh không train,
  R@1 chỉ hơn sng +0,02 (base), +0,12 (large), +0,16 (Qwen2.5-3B), đều dưới ngưỡng +0,3. Vì vậy dừng ở 1c, không
  parse train, không train lại. main_r3 với đồ thị FACTUAL-large đạt 46,61 ± 0,12, so với 46,48 ± 0,24 (nằm trong
  nhiễu).
- **Kênh hình học** (`geometry-channel/REPORT.md`): kênh số lượng tăng R@1 trên 1 121 câu có số đếm thêm +0,74 (w
  chọn trên val) hoặc +0,44 (chọn 2-fold); trên toàn val chỉ +0,10 hoặc +0,05. Kênh vị trí không có bằng chứng giúp
  (2-fold −0,05 trên 756 câu).

## Bước 1d: train lại với FACTUAL-T5 large (18/10/2026, branch `feat/factual-retrain`)

`factual-retrain/REPORT.md`: chỉ đổi parser câu. Val R@1 đạt 47,23 ± 0,36, so với 46,48 ± 0,24 của main_r3 (+0,76);
MRR 59,49 ± 0,29, so với 58,85 ± 0,18 (+0,64). Welch p ≈ 0,05 với 3 seed mỗi bên, và mọi seed đều cao hơn mọi seed
của main_r3. Probe không train chỉ báo +0,12, nên lợi ích đến từ việc train trên đồ thị sạch hơn. Độ dồn của vector
bộ ba vẫn còn (cosine 0,97–0,99). Test chưa chấm.

## Đợt 3: chuẩn hoá, ghép cả đồ thị, làm sạch đồ thị ảnh (19–20/10/2026, branch `feat/fusion-and-graph-matching`)

Mốc: main_factual_r3, R@1 47,23 ± 0,36.

- **Thí nghiệm 1** (`fusion/REPORT.md`): bỏ z-score cho 47,48 ± 0,10; sàn z-score cho 47,31 ± 0,17. Sàn gỡ được một
  phần độ dồn của vector bộ ba (cosine 0,98 → 0,88), nhưng thứ hạng không đổi.
- **Thí nghiệm 2** (`graph-matching/REPORT.md`): FGW và nhất quán node chung, không train, đều không giúp (tối đa
  +0,12 R@1).
- **Audit** (`graph-dedup-audit/REPORT.md`): 11,8% node train có khung trùng cùng loại ở IoU ≥ 0,7. Gộp lại làm đồ
  thị liền hơn và tăng tỉ lệ khớp số lượng (24,9% → 30,2%). Điểm nhất quán gần như không đổi. Chưa train.

## Đợt 4: graph ảnh đầy đủ (22–23/10/2026, branch `feat/full-image-graph`)

Mốc: 1a `raw_factual_r3`, R@1 47,48 ± 0,10. Theo kế hoạch Lộc chốt ngày 22/10, chỉ chạy một bản gộp cả 4 thay đổi:

- gộp khung trùng (dữ liệu `merge_v1`);
- 6 đặc trưng hình học trên mọi cạnh;
- nối mỗi node tới 5 node gần nhất cùng cầu nối cho graph liền, các cạnh này mang vị từ `[none]` và không phải bộ ba;
- một node tổng cho mỗi ảnh và mỗi câu.

Kết quả (`full-graph/REPORT.md`):

- `fullgraph_factual_r3` đạt R@1 48,38 ± 0,17 (+0,90, Welch p = 0,003) và MRR 60,51 ± 0,17 (+0,79). Mọi seed đều
  cao hơn mọi seed của mốc.
- GAT nhận gradient gấp 2,5 lần.
- Vector bộ ba bớt dồn (cosine 0,96 → 0,87).
- Kênh bộ ba đứng riêng mạnh gần gấp đôi (R@1 7,2 → 13,1).
- Mỗi epoch chậm gấp khoảng 2,2 lần.
- Chưa tách được phần nào có tác dụng (chưa có ablation, chưa có đối chứng tắt GAT). Test chưa chấm.
