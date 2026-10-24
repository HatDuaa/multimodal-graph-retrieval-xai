# Thí nghiệm 5 — Chỉ thêm số lượng và bảng g lên fullgraph, không gom nhóm (`countonly_factual_r3`)

Mọi số trong báo cáo là của **val**. Test chưa chấm.

Mốc so sánh là `fullgraph_factual_r3` (`../full-graph/`), R@1 48,38 ± 0,17. Thí nghiệm 4 `groupcount_factual_r3`
(`../group-count/`, R@1 47,75 ± 0,22) được đưa vào để đối chiếu.

**Kết luận:** bản này cao hơn fullgraph +0,53 R@1 (48,91 ± 0,21, Welch p = 0,032), và cả 3 seed đều cao hơn mọi seed
của fullgraph. Lộc đã chốt từ trước: nếu điều kiện này đạt thì bản này thành mô hình chính và phase cải tiến đóng lại.

## Thay đổi so với fullgraph

Công thức giữ nguyên như fullgraph:

- dữ liệu merge_v1;
- `--fusion raw` với cùng hằng số;
- hình học trên cạnh;
- kNN-5 cùng cầu nối;
- node tổng;
- micro-batch 32, `expandable_segments:False`;
- tối đa 10 epoch, dừng sớm như cũ.

Chỉ thêm số lượng và bảng g (`--count-table`), **không gom nhóm**.

- **Phía ảnh** (`scripts/improve/package_counts.py`, dữ liệu `count_v1`):
  - Mỗi node của merge_v1 vẫn là một vật riêng và giữ nhãn gốc.
  - Số lượng của node là số vật cùng loại trong ảnh sau khi gộp trùng, tính cả chính nó. "Cùng loại" theo `same_type`
    của audit: synset hoặc hypernym, và mọi người (man, woman, skier…) tính là một loại.
  - Chia nhóm 1 / 2 / 3–4 / ≥5. Tỉ lệ node trên val: 50,6% / 19,3% / 14,6% / 15,5%; train gần như vậy.
  - Graph không chép lại. Số lượng nằm trong file phụ `data/graphs/count_v1/count_buckets.json` (9 MB), GraphStore gắn
    vào qua `paths.count_buckets` (`configs/count_v1.yaml`).
- **Phía câu và bảng g:** giữ y như thí nghiệm 4.
  - Cùng luật số lượng; query graph lấy lại của group_v1, chỉ đọc.
  - g khởi tạo bằng 0; hàng "không rõ" cố định bằng 0.
  - Chỉ cộng vào độ giống của từng cặp node trong kênh vật thể, dùng cho cả attention lẫn điểm.

## Kiểm tra trước khi train

**g = 0** (`check_zero.json`, `logs/check_zero_repeat.log`)

- Checkpoint `fullgraph_factual_r3_seed0` với g = 0, chạy trên count_v1, cho R@1 0,48208408 và MRR 0,60380839, trùng
  `metrics.json`. Chạy lại 5 lần đều như vậy.
- Cùng trọng số nhưng không có bảng g: 2 trong 14 lần chạy cho MRR lệch 2,7e-7 (một câu đổi hạng 18↔19); R@1/5/10
  không đổi. Lệch này xuất hiện cả trên merge_v1 khi không có g. Nguyên nhân là phép cộng dồn trên GPU trong GAT không
  tất định, không phải do g.

**Khớp số lượng ở ảnh đúng** (`check_match.json`, 5 871 node câu có số lượng)

| | Có node ảnh khớp nhãn | Khớp cả số lượng |
|---|---|---|
| count_v1 (thí nghiệm 5) | 66,1% | 40,7% |
| group_v1 (thí nghiệm 4) | 62,3% | 37,3% |

count_v1 trùng hàng "merge_v1, chưa gom" của audit.

## Hạ tầng

- **Thời gian:** một epoch 931 giây, so với 681 giây của fullgraph. Phần chậm thêm là phép tra bảng g cho từng cặp
  node: backward của nó cộng dồn vào một bảng 28 ô. Kết quả không bị ảnh hưởng. Mình không sửa giữa chừng để 3 seed
  chạy cùng một code (commit `c9102e9`).
- **Tổng thời gian train:** 7,03 giờ. Peak GPU 7,6 GB. Seed 0 và seed 2 mỗi seed có 1 cảnh báo phân mảnh bộ nhớ lúc chấm
  val, đều tự hồi phục.
- **Dòng thời gian:**
  - seed 0: 11:23–13:59 ngày 24/10;
  - seed 1: 13:59–16:36;
  - seed 2: bắt đầu 16:36. KG dừng chuỗi lúc 17:41 vì GPU cần cho việc khác, sau khi xong epoch 4 (`checkpoint_last.pt`
    lưu lúc 17:39). Seed 2 chạy tiếp lúc 22:40 bằng `--resume` từ epoch 4 (`resume.jsonl`), dừng sớm sau epoch 7.
  - Phần dở của epoch 5 trong lần chạy đầu bị bỏ. Epoch 5 được train lại từ trạng thái đã lưu, kể cả trạng thái sinh số
    ngẫu nhiên.
  - Chẩn đoán xong lúc 23:31.
- **Ổ đĩa:** còn 21 GB suốt quá trình (sàn 20 GB). `run.sh` kiểm tra đĩa trước mỗi seed và trước bước chẩn đoán. Phần
  đĩa bị hao trước thí nghiệm là của người dùng khác; mình không xoá gì.

## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | Epoch tốt nhất | Số epoch |
|---|---|---|---|---|---|---|
| **countonly_factual_r3** | **48.91 ± 0.21** | 75.22 ± 0.05 | **84.96 ± 0.25** | **60.86 ± 0.22** | 8, 9, 5 | 10, 10, 7 |
| fullgraph_factual_r3 (mốc) | 48.38 ± 0.17 | 75.25 ± 0.09 | 84.71 ± 0.09 | 60.51 ± 0.17 | 4, 8, 5 | 6, 10, 7 |
| groupcount_factual_r3 (thí nghiệm 4) | 47.75 ± 0.22 | 74.82 ± 0.08 | 84.32 ± 0.17 | 59.97 ± 0.14 | 9, 4, 7 | 10, 6, 9 |

- **Từng seed của countonly:** R@1 49,15 / 48,85 / 48,74; MRR 61,10 / 60,80 / 60,68. Fullgraph: R@1 48,21 / 48,56 /
  48,39.
- **So với fullgraph** (Welch, 3 seed mỗi bên): R@1 +0,53 (p = 0,032), MRR +0,35 (p = 0,093), R@10 +0,25 (p = 0,21),
  R@5 −0,03 (p = 0,61).

Theo epoch, R@1 (MRR):

- seed 0: 47,00 → 47,55 → 48,01 → 48,37 → 48,30 → 48,51 → 48,49 → **49,15** (61,10) → 48,72 → 48,58
- seed 1: 46,79 → 47,46 → 47,91 → 48,33 → 47,80 → 48,60 → 48,56 → 48,69 → **48,85** (60,80) → 48,78
- seed 2: 46,95 → 47,22 → 48,22 → 47,98 → **48,74** (60,68) → 48,63 → 48,42, dừng sớm

## R@1 theo tập con câu (`count_subset.json`)

Các tập con được xác định một lần từ lớp số lượng của câu, dùng chung cho cả ba bản.

| Tập con val | Số câu | countonly | fullgraph | groupcount | countonly − fullgraph, R@1 (p) | MRR (p) |
|---|---|---|---|---|---|---|
| Tất cả | 10 633 | 48.91 ± 0.21 | 48.38 ± 0.17 | 47.75 ± 0.22 | +0,53 (0,032) | +0,35 (0,093) |
| Có số lượng | 4 543 | 50.85 ± 0.45 | 50.28 ± 0.18 | 50.10 ± 0.36 | +0,57 (0,15) | +0,42 (0,18) |
| — có số nêu rõ | 1 235 | 54.06 ± 1.09 | 52.58 ± 0.28 | 54.22 ± 1.13 | +1,48 (0,14) | +0,98 (0,16) |
| — chỉ số nhiều / group | 3 308 | 49.66 ± 0.39 | 49.43 ± 0.31 | 48.56 ± 0.14 | +0,23 (0,47) | +0,21 (0,35) |
| Không rõ số lượng | 6 090 | 47.46 ± 0.14 | 46.97 ± 0.37 | 46.00 ± 0.12 | +0,49 (0,14) | +0,30 (0,24) |

- **Câu có số nêu rõ:** lợi ích giữ ngang thí nghiệm 4 (54,06 so với 54,22).
- **Các tập con khác không còn bị tụt.** Thí nghiệm 4 tụt −0,87 / −0,97 ở hai tập này; countonly tăng nhẹ.
- **Không tập con nào đủ ý nghĩa thống kê khi xét riêng,** vì mỗi tập nhỏ hơn và chỉ có 3 seed.
- **Tập "không rõ số lượng" (+0,49):** g không tác động trực tiếp lên các câu này, nên mức tăng ở đây chỉ có thể đến gián
  tiếp qua việc train cùng g, hoặc là dao động giữa các seed. Hai khả năng này không tách được với 3 seed.

## Bảng g đã học (mean ± std qua 3 seed)

Hàng là lớp số lượng phía câu, cột là nhóm số lượng phía ảnh.

| Câu \ Ảnh | 1 | 2 | 3–4 | ≥5 |
|---|---|---|---|---|
| không rõ (cố định) | 0 | 0 | 0 | 0 |
| 1 | **+0.099 ± 0.011** | −0.012 ± 0.015 | −0.095 ± 0.008 | −0.165 ± 0.018 |
| 2 | −0.071 ± 0.009 | **+0.074 ± 0.002** | +0.037 ± 0.006 | −0.034 ± 0.006 |
| 3–4 | −0.090 ± 0.012 | −0.073 ± 0.013 | **+0.104 ± 0.002** | +0.016 ± 0.002 |
| ≥5 | −0.111 ± 0.012 | −0.088 ± 0.006 | −0.018 ± 0.003 | **+0.127 ± 0.011** |
| ≥2 | −0.035 ± 0.002 | +0.002 ± 0.007 | +0.025 ± 0.009 | +0.030 ± 0.007 |
| ≥3 | −0.070 ± 0.003 | −0.045 ± 0.002 | +0.029 ± 0.003 | +0.065 ± 0.002 |

Cấu trúc giống thí nghiệm 4: đường chéo dương, lệch càng xa càng âm, các hàng "≥" tăng dần. Ba seed gần như trùng nhau.
Hàng "≥2" yếu hơn thí nghiệm 4: ở đây node đơn đếm cả các vật cùng loại khác, nên "nhiều hơn 1" ít khi phân biệt được
ảnh này với ảnh khác.

## Ca g làm đổi node được ghép (`count_match_changes.json/.md`; thí nghiệm 4: `exp4-reference/`)

Xét mọi node câu có số lượng đã biết, trên ảnh đúng của câu (5 582 node). So node ảnh được ghép theo sim_ij (bỏ g) với
node được ghép theo sim_ij + g. Một lần đổi được tính là:

- "tốt" nếu chỉ node mới khớp nhãn với node câu;
- "xấu" nếu chỉ node cũ khớp nhãn.

"Khớp nhãn" nghĩa là trùng nhãn sau khi gộp bí danh VG, hoặc cosine CLIP ≥ 0,955.

| Mean ± std qua 3 seed | countonly (thí nghiệm 5) | groupcount (thí nghiệm 4) |
|---|---|---|
| Node đổi node được ghép | 469 ± 42 (8,4%) | 783 ± 1 (14,0%) |
| — tốt / xấu / không đổi trạng thái nhãn | 12,7 / 135,3 / 321,3 | 16,0 / 205,3 / 561,3 |
| Khớp nhãn: bỏ g → có g | 65,0% → 62,8% | 62,0% → 58,6% |
| Khớp số lượng: bỏ g → có g | 54,8% → 62,1% | 52,9% → 64,1% |

g làm đổi node được ghép ở 8,4% node câu có số lượng, ít hơn thí nghiệm 4 (14%). Khi đổi, g thường đổi lấy khớp số
lượng (+7,3 điểm) bằng cách hy sinh một ít khớp nhãn (−2,2 điểm).

Tiêu chí nhãn khá chặt, nên nhiều ca "xấu" vẫn hợp lý về nghĩa. Ví dụ từ seed 0, ghi cả node bỏ g và node có g:

- **Hợp lý về nghĩa** (người → man/woman):
  - "A group of people crowded around a picnic blanket": câu ≥3 people — bỏ g: 1 people (0,902) → có g: 3–4 woman
    (0,877, g +0,031).
  - "a big bus that has some people by it": ≥2 people — 1 people (0,879) → ≥5 man (0,908, g +0,037).
- **Sai thật:**
  - "A lone woman huddles under her umbrella…": câu 1 woman — 3–4 woman (0,925) → 1 hair (0,917, g +0,087). Mọi người
    trong ảnh tính chung một loại, nên người phụ nữ mang nhóm 3–4. g thưởng cho node "hair" chỉ vì nó thuộc nhóm 1.
  - "An empty kitchen, including a refrigerator, sink, and cupboards": ≥2 cupboard — 1 cupboard (0,917) → 2 cabinet
    (0,882). Bản thân hai node này gần nghĩa, nên ca này không hẳn sai.
- **Ca tốt:**
  - "Two giraffes standing next to each other in a holding pen": câu 2 giraffe — 1 baby giraffe (0,876) → 2 giraffe
    (0,905, g +0,071).
  - "A market has many fruits and vegetables": ≥3 fruit — 1 orange (0,864) → ≥5 fruit (0,880, g +0,064).

Danh sách 16 ca ở `count_match_changes.md`.

**Cho demo:** nên hiện cả hai dòng, ghép khi bỏ g và khi có g, mỗi khi hai node khác nhau, như trong file trên.

## Ví dụ giải thích có dòng số lượng (`count_examples.md`, seed 0)

6 câu val có số nêu rõ, chọn ngẫu nhiên (seed 0) — cùng tập câu với thí nghiệm 4 nên so sánh được. Ví dụ:

- "Three children dressed in white…": câu 3–4 child — ảnh: 3–4 child → g = +0,105.
- "…two pillow cases…": câu 2 pillow case — ảnh: 2 bed pillows → g = +0,071. Ở thí nghiệm 4, node câu này bị ghép nhầm
  sang "3–4 quilt".
- "A lone surfer…": câu 1 surfer — ảnh: 1 surfboard → g = +0,087. Vẫn ghép nhầm như thí nghiệm 4: ảnh này không có node
  "surfer".
- "…two windows": câu 2 window — ảnh: 1 window → g = −0,081. VG chỉ đánh một khung cửa sổ, nên g phạt ảnh đúng.

## Các phép đo khác

| | countonly | fullgraph | groupcount |
|---|---|---|---|
| Đổi động từ / đổi chủ–tân, Δ R@1 điểm cuối | −1,16 / −0,43 | −1,02 / −0,38 | −1,13 / −0,33 |
| Kênh bộ ba đứng riêng, R@1 câu có ≥1 bộ ba | 12.69 ± 2.06 | 13.14 ± 1.25 | 11.28 ± 2.17 |
| Độ dồn vector bộ ba (cosine) | 0,880 ± 0,017 | 0,869 ± 0,045 | 0,838 ± 0,065 |
| Độ dồn vector vật thể (cosine) | 0,301 ± 0,039 | 0,334 ± 0,077 | 0,314 ± 0,045 |
| Tương quan vật thể ~ bộ ba | 0,309 ± 0,037 | 0,323 ± 0,015 | 0,317 ± 0,040 |
| Tương quan CLIP ~ bộ ba / CLIP ~ vật thể | 0,187 / 0,333 | 0,202 / 0,326 | 0,192 / 0,314 |
| Gradient gat1 / gat2 (probe) | 0,344 / 0,261 | 0,335 / 0,256 | 0,325 / 0,240 |
| Gradient bảng g (probe) | 0,157 ± 0,009 | — | 0,140 ± 0,005 |
| a / b / c ở epoch tốt nhất | 0,571 / 0,125 / 0,304 | 0,618 / 0,135 / 0,247 | 0,578 / 0,115 / 0,307 |

Ngoài kênh vật thể, mô hình gần như không đổi so với fullgraph. Bảng đầy đủ ở `summary.md`.

## Nhận xét

1. **Số lượng giúp được khi không gom nhóm.** R@1 +0,53 so với fullgraph (p = 0,032), và mọi seed đều cao hơn mọi seed
   của fullgraph. Đây là kết quả val tốt nhất của phase cải tiến.
2. **Thí nghiệm 4 kém là do gom nhóm, không phải do bảng g.** Cùng bảng g và cùng luật phía câu, bỏ gom nhóm thì:
   - lợi ích ở câu có số nêu rõ vẫn giữ nguyên (54,06 so với 54,22);
   - mức tụt ở các câu khác biến mất.
3. **Hạn chế còn lại:**
   - Mọi người tính chung một loại. Vì vậy "a lone woman" trong ảnh có nhiều người bị phạt, và g có thể kéo sang node
     sai (woman → hair).
   - g thưởng cho cùng nhóm số lượng bất kể nhãn, nên đôi khi đổi node được ghép.
   - Cả hai điểm này nên ghi vào phần hạn chế của báo cáo môn học.
4. **Test chưa chấm.** Phần chuẩn bị chấm test ở `../test-prep/` và trong báo cáo gửi KG.

## File

- **Run:** `countonly_factual_r3_seed{0,1,2}/`, gồm `config.json` (commit `c9102e9`), `log.jsonl`, `metrics.json`, và
  `resume.jsonl` của seed 2. Checkpoint không commit.
- **Kiểm tra:** `check_zero.json`, `check_match.json`, `logs/check_zero_repeat.log`.
- **Kết quả:**
  - `count_subset.json`: tập con và bảng g; `count_subset_seed0_cpu.json`: bản tính sớm cho seed 0;
  - `count_match_changes.json/.md` và `exp4-reference/`;
  - `count_examples.md`;
  - `summary.md`, `summary.json`;
  - `triple_perturbation.json`, `part_similarity.json`, `channel_correlation.json`, `gradient_probe.json`.
- **Báo cáo nhanh seed 0:** `seed0.md`.
- **Log:** `logs/`. Lệnh chạy lại: `run.sh`, rồi `diagnostics.sh`.
