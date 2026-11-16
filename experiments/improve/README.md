# Phase cải tiến — kênh bộ ba

Mọi số trong thư mục này là của **val**, trừ `test/`. Test chỉ được chấm một lần, cho mô hình chính, vào cuối phase
(25/10/2026). Mỗi thí nghiệm đổi đúng một thứ so với mô hình tốt nhất lúc đó; riêng thí nghiệm 3 gộp bốn thay đổi theo
quyết định của Lộc. Mọi thí nghiệm chạy 3 seed (0, 1, 2), số được tổng hợp bằng `src/eval/metrics.py` (mean ± std,
ddof = 1).

## Tổng kết phase cải tiến (18–25/10/2026)

**Mô hình chính:** `countonly_factual_r3` (`count-only/`).

Kết quả test: R@1 **47,95 ± 0,15**, MRR 60,15 ± 0,15. So với main_r3 cấp 1 (45,77 ± 0,23), R@1 tăng +2,18 (p = 0,0005);
so với CLIP thuần (39,32), tăng +8,63.

Mô hình gồm:

- **Đồ thị câu:** FACTUAL-T5 large thay cho parser sng.
- **Trộn điểm:** trộn điểm thô, bỏ z-score (`--fusion raw`).
- **Đồ thị ảnh:**
  - gộp các khung trùng cùng loại (IoU ≥ 0,7);
  - 6 đặc trưng hình học trên mọi cạnh;
  - nối mỗi node tới 5 node gần nhất cùng cầu nối cho graph liền, vị từ `[none]`, không tính là bộ ba;
  - một node tổng.
- **Số lượng:** số lượng cho node câu và node ảnh, kèm bảng học được g cộng vào độ giống trong kênh vật thể.

### Chuỗi các bước được giữ lại (val, 3 seed)

| Bước | Mô hình | Thay đổi so với bước trước | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| Cấp 1 | main_r3 | — | 46.48 ± 0.24 | 73.78 ± 0.16 | 83.44 ± 0.24 | 58.85 ± 0.18 |
| 1d | main_factual_r3 | Đồ thị câu từ FACTUAL-T5 large (`factual_v1`) | 47.23 ± 0.36 | 74.49 ± 0.33 | 84.00 ± 0.16 | 59.49 ± 0.29 |
| 1a | raw_factual_r3 | Bỏ z-score, trộn điểm thô | 47.48 ± 0.10 | 74.70 ± 0.18 | 84.10 ± 0.08 | 59.72 ± 0.00 |
| 3 | fullgraph_factual_r3 | Gộp trùng + hình học cạnh + kNN liền + node tổng (`merge_v1`) | 48.38 ± 0.17 | 75.25 ± 0.09 | 84.71 ± 0.09 | 60.51 ± 0.17 |
| 5 | **countonly_factual_r3** | Số lượng + bảng g, không gom nhóm (`count_v1`) | **48.91 ± 0.21** | 75.22 ± 0.05 | **84.96 ± 0.25** | **60.86 ± 0.22** |

Mức tăng R@1 val ở từng bước, so với bước liền trước (Welch, 3 seed mỗi bên):

| Bước | Chênh R@1 | p |
|---|---|---|
| 1d | +0,76 | ≈ 0,05 |
| 1a | +0,25 | không có ý nghĩa thống kê; giữ vì std thấp nhất và vẫn còn tăng ở epoch 10 |
| 3 | +0,90 | 0,003 |
| 5 | +0,53 | 0,032 |

Cộng lại, val tăng +2,43 so với main_r3. Thí nghiệm 3 gộp bốn thay đổi, nên chưa biết phần nào có tác dụng; Lộc quyết
không chạy ablation.

### Test cuối cùng (`test/README.md`, chấm một lần, 25/10/2026)

| Mô hình | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần | 39.32 | 66.39 | 77.32 | 52.00 |
| main_r3 (cấp 1) | 45.77 ± 0.23 | 72.87 ± 0.31 | 82.69 ± 0.31 | 58.14 ± 0.18 |
| **countonly_factual_r3** | **47.95 ± 0.15** | **74.79 ± 0.03** | **84.38 ± 0.08** | **60.15 ± 0.15** |

- Chênh so với main_r3: R@1 +2,18, R@5 +1,91, R@10 +1,70, MRR +2,00. Mọi p đều ≤ 0,008.
- Chênh val → test: −0,96 R@1 (main_r3: −0,71).

### Các thí nghiệm không giúp (không đưa vào mô hình chính)

| Thí nghiệm | Mốc lúc đó | Kết quả val | Ghi chú |
|---|---|---|---|
| A: so khớp bộ ba tách thành phần (`triple-factorized/`) | main_r3 46,48 | R@1 46.00 ± 0.02 | Gỡ được độ dồn, nhưng kênh bộ ba lặp kênh vật thể (tương quan 0,59) |
| B: kênh mức đồ thị thay kênh bộ ba (`graph-channel/`) | main_r3 46,48 | R@1 46.23 ± 0.13 | Mù vị từ và chiều |
| Đổi parser, không train lại (`parser-compare/`) | main_r3 46,48 | +0,02 đến +0,16 | Cần train lại; chính là bước 1d |
| Kênh hình học theo luật: số lượng + vị trí (`geometry-channel/`) | main_r3 46,48 | toàn val +0,10 / +0,05 | Kênh vị trí không có bằng chứng giúp |
| 1b: sàn cho z-score (`fusion/floor/`) | main_factual_r3 47,23 | R@1 47.31 ± 0.17 | Bớt dồn (0,98 → 0,88) nhưng không tăng điểm |
| 2: ghép cả đồ thị, FGW / nhất quán node chung (`graph-matching/`) | main_factual_r3 47,23 | tối đa +0,12 | Không train |
| 4: node nhóm + số lượng (`group-count/`) | fullgraph 48,38 | R@1 47.75 ± 0.22 (−0,63, p = 0,019) | Gom nhóm làm mất thông tin; bảng g vẫn giúp câu có số nêu rõ |

Các thư mục chẩn đoán và kiểm soát hạ tầng: `triple-perturbation/`, `channel-correlation/`, `microbatch-control/`,
`graph-dedup-audit/`, `test-prep/`.

### Hạn chế cần ghi trong báo cáo môn học

- **Thí nghiệm 3 gộp bốn thay đổi**, không có ablation. Vì vậy chưa biết phần nào có tác dụng, và chưa rõ GAT có thật sự
  đóng góp vào thứ hạng hay không.
- **Đếm số lượng gộp mọi người thành một loại.** "A lone woman" trong ảnh nhiều người bị coi là sai số lượng. Bảng g đôi
  khi kéo node câu ghép sang node ảnh khác nhãn, ở khoảng 8% node có số lượng.
- **Val bị dùng nhiều lần để chọn mô hình,** nên val hơi lạc quan. Chênh val → test là −0,96 R@1.
- **Mỗi epoch chậm gấp khoảng 3 lần bản 1a** (931 giây so với 312), do cạnh kNN và bảng g.

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
| `group-count/` | Đợt 4, thí nghiệm 4: node nhóm + số lượng + bảng g trong kênh vật thể, trên nền fullgraph |
| `count-only/` | Đợt 4, thí nghiệm 5: chỉ số lượng + bảng g trên fullgraph, không gom nhóm — **mô hình chính** |
| `test-prep/` | Dữ liệu test đóng gói theo chuỗi count_v1, kiểm tra và lệnh chấm test một lần |
| `test/` | Kết quả test, chấm một lần ngày 25/10/2026: countonly_factual_r3 và CLIP thuần |

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
bộ ba vẫn còn (cosine 0,97–0,99). Lúc đó test chưa chấm.

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
- Chưa tách được phần nào có tác dụng (chưa có ablation, chưa có đối chứng tắt GAT). Lúc đó test chưa chấm.

### Thí nghiệm 4: node nhóm và số lượng (23–24/10/2026, branch `feat/group-count`)

Nền là fullgraph_factual_r3, thêm ba thứ:

- gom các vật cùng loại có cùng node neo và cùng quan hệ thành node nhóm, đặt tên theo nhãn chiếm đa số (dữ liệu
  `group_v1`);
- số lượng cho node ảnh (1 / 2 / 3–4 / ≥5) và node câu (luật trên câu FACTUAL);
- bảng học được g[lớp số lượng câu, nhóm số lượng ảnh] cộng vào độ giống trong kênh vật thể.

Kết quả (`group-count/REPORT.md`):

- `groupcount_factual_r3` đạt R@1 47,75 ± 0,22, thấp hơn fullgraph 0,63 (p = 0,019).
- g học đúng cấu trúc: đường chéo dương, các ô lệch số lượng âm.
- Trên 1 235 câu có số nêu rõ, R@1 tăng +1,65 và MRR tăng +1,05 (p = 0,044).
- Câu không có số lượng tụt −0,97, câu chỉ có số nhiều tụt −0,87. Nhiều khả năng do graph đã gom làm mất thông tin
  (−24% node, khớp số lượng ở ảnh đúng 40,7% → 37,3%).
- fullgraph vẫn là bản tốt nhất trên val. Lúc đó test chưa chấm.

### Thí nghiệm 5: chỉ số lượng, không gom nhóm (24/10/2026, branch `feat/count-only`)

Nền là fullgraph_factual_r3. Thêm số lượng cho từng node ảnh (số vật cùng loại trong ảnh, giữ nhãn gốc), cùng luật số
lượng phía câu và bảng g như thí nghiệm 4 (`count-only/REPORT.md`).

- `countonly_factual_r3` đạt R@1 48,91 ± 0,21 (+0,53 so với fullgraph, Welch p = 0,032) và MRR 60,86 ± 0,22 (+0,35).
  Cả 3 seed đều cao hơn mọi seed của fullgraph.
- Câu có số nêu rõ tăng +1,48. Các câu còn lại không tụt nữa.
- g làm đổi node ảnh được ghép ở 8,4% node câu có số lượng (thí nghiệm 4: 14%).
- Theo quyết định của Lộc, bản này là **mô hình chính**, và phase cải tiến đóng lại.
- Test đã được đóng gói và có script chấm một lần (`test-prep/README.md`). Script đã chạy thử trên val và cho lại đúng
  số. Test được chấm một lần ngày 25/10 (`test/README.md`): R@1 47,95 ± 0,15.
