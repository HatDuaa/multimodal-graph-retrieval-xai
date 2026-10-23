# Thí nghiệm 4 — Node nhóm và số lượng trên graph liền (`groupcount_factual_r3`)

Mọi số ở đây là của **val**. Test chưa chấm.

Mốc so sánh là `fullgraph_factual_r3` (`../full-graph/`): R@1 48,38 ± 0,17. Bản mới giữ nguyên toàn bộ công thức của
fullgraph:

- dữ liệu merge_v1 làm nền;
- `--fusion raw` với cùng hằng số khởi tạo;
- hình học trên cạnh;
- kNN-5 cùng cầu nối;
- node tổng;
- micro-batch 32, `expandable_segments:False`;
- tối đa 10 epoch, seed 0, 1, 2.

Chỉ đổi đúng ba thứ, theo yêu cầu của Lộc ngày 23/10:

1. **Graph ảnh có node nhóm** (dữ liệu `group_v1`). kNN, cầu nối và node tổng được tính lại trên graph mới.
2. **Số lượng** cho node ảnh và node câu.
3. **Bảng g** học được, cộng vào độ giống của từng cặp node trong kênh vật thể.

## Thay đổi

### 1. Node nhóm (`scripts/improve/package_group.py`, `configs/group_v1.yaml`)

Xuất phát từ merge_v1, tức là đã gộp các khung trùng cùng loại ở IoU ≥ 0,7.

- **Gom nhóm:** các vật cùng loại được gom khi có chung một node neo và cùng quan hệ với node đó, theo một trong hai
  cách: (a) cùng vị từ có nhãn, hoặc (b) cùng lớp hình học (trái, phải, trên, dưới, chồng lên, cạnh). Đây là tiêu chí
  "ab" của bản audit (`src/graph/dedup.py`, `group_by_role`), khép bắc cầu.
- **Đặt tên node nhóm** (`majority_name`):
  - lấy nhãn xuất hiện nhiều nhất trong các thành viên, mỗi thành viên một phiếu;
  - nếu hoà, lấy hypernym chung gần nhất trong WordNet của các nhãn hoà, ví dụ man + woman → adult, man + skier → person;
  - nếu không có hypernym chung, lấy nhãn hoà đứng đầu theo thứ tự chữ cái.

  Không còn lấy nhãn của khung lớn nhất. Lưu ý: bước gộp trùng của merge_v1 vẫn lấy nhãn khung lớn nhất, vì bản này chỉ
  đổi phần nhóm.
- **Khung** của node nhóm là khung bao các thành viên. **Cạnh** là hợp các cạnh của thành viên; bộ ba trùng và vòng tự
  thân bị bỏ.

Kết quả đóng gói (`data/graphs/group_v1/MANIFEST.json`):

| | train | val |
|---|---|---|
| Node trước → sau khi gom | 1 168 021 → 883 256 (−24%) | 52 230 → 39 495 |
| Cạnh trước → sau khi gom | 822 515 → 670 055 | 36 954 → 30 106 |
| Node nhóm / ảnh, cỡ nhóm trung bình | 2,90, 3,09 | 2,87, 3,08 |
| Cách đặt tên node nhóm: đa số / hypernym / hoà lấy nhãn đầu | 87,0% / 8,0% / 5,0% | 86,7% / 8,0% / 5,3% |

- Số node nhóm mỗi ảnh khớp với audit (2,90 ở train).
- Có 33 386 cụm part mới, gồm 21 nhãn node (do hypernym) và 33 365 bộ ba. Các cụm này được mã hoá bằng đúng CLIP đóng
  băng; mã hoá lại 1 000 khoá cũ để kiểm tra cho sai lệch lớn nhất bằng 0.
- Test chưa được đóng gói.

### 2. Số lượng (`src/graph/counts.py`)

**Phía ảnh.** Mỗi node có một số lượng:

- node nhóm: số thành viên;
- node đơn: số vật cùng loại trong ảnh sau khi gộp trùng, tính cả chính nó; mỗi thành viên của một nhóm cũng tính là
  một vật. "Cùng loại" theo `same_type` của audit, nên mọi người (man, woman, skier…) cùng một loại.

Số lượng được xếp vào 4 nhóm: 1 / 2 / 3–4 / ≥5. Trên val, tỉ lệ node ở từng nhóm là 67% / 18% / 9% / 6%.

**Phía câu.** Lấy graph FACTUAL của `factual_v1`. Mỗi node được nối với danh từ tương ứng trong bản phân tích spaCy
của câu, rồi áp luật đầu tiên khớp:

1. Số nêu rõ 2–10 gắn vào danh từ (chữ hoặc số), hoặc "a pair of X" → đúng số đó. 2 cho lớp "2", 3–4 cho lớp "3–4",
   5–10 cho lớp "≥5".
2. "a couple of X", "both X" → "2".
3. "lone X", "single X" → "1".
4. "a group / herd / crowd of X", "lots of X" hoặc "a lot of X", "several X", "many X" → "≥3". Luật này cũng áp khi
   thuộc tính FACTUAL chứa một trong các từ đó, hoặc khi chính node là group / herd / crowd.
5. Danh từ số nhiều (NNS/NNPS) → "≥2", trừ một danh sách danh từ luôn ở dạng số nhiều (pants, glasses, scissors…).
6. Còn lại, kể cả "a X" → "không rõ".

Nếu không tìm thấy danh từ trong câu, dùng số lượng FACTUAL (2–10) nếu có; nếu không có thì xét thuộc tính theo luật 4.

Trên val có 4 543 / 10 633 câu (42,7%) có ít nhất một node với số lượng khác "không rõ". Tính trên 29 766 node chấm
điểm được:

| Lớp | không rõ | ≥2 | 2 | ≥3 | 3–4 | ≥5 | 1 |
|---|---|---|---|---|---|---|---|
| % node val | 80,3 | 12,5 | 3,3 | 2,9 | 0,84 | 0,12 | 0,07 |

Số node do từng luật bắt được trên val:

| Luật | Số node |
|---|---|
| số nhiều | 3 725 |
| số nêu rõ | 1 037 |
| group / herd / many… | 871 |
| couple / both | 147 |
| pair | 41 |
| số FACTUAL (dự phòng) | 30 |
| lone / single | 20 |

Train có tỉ lệ gần như vậy: 44,6% câu có số lượng đã biết.

### 3. Bảng g (`--count-table`)

- g là bảng 7 × 4: hàng là lớp số lượng phía câu, cột là nhóm số lượng phía ảnh. Khởi tạo bằng 0; hàng "không rõ" cố
  định bằng 0.
- Trong kênh vật thể, độ giống của mỗi cặp node đổi thành sim'_ij = sim_ij + g[lớp câu của i, nhóm ảnh của j]. sim'
  được dùng cho cả attention lẫn điểm.
- Kênh bộ ba giữ nguyên.
- `matched_pairs` (`src/explain/graph_explainer.py`) có thêm `query_count`, `image_count` và `count_bias` cho mỗi dòng
  vật thể. Ví dụ ở `count_examples.md`.

## Kiểm tra trước khi train

**g = 0** (`check_zero.json`), dùng checkpoint `fullgraph_factual_r3_seed0`:

- Chạy trên merge_v1 với code mới cho kết quả trùng từng chữ số với `metrics.json` (R@1 0,48208).
- Trên group_v1, cùng trọng số đó: có bảng g bằng 0 và không có bảng g cho kết quả giống hệt nhau (R@1 0,47503,
  MRR 0,59619); chạy hai lần đều như vậy.
- Mô hình fullgraph không train lại mà chấm thẳng trên graph đã gom thì tụt 0,7 R@1 (48,21 → 47,50).

**Khớp số lượng ở ảnh đúng** (`check_match.json`). Tính trên 5 871 node câu val có số lượng đã biết. Node ảnh "khớp
nhãn" khi trùng nhãn sau khi gộp bí danh VG, hoặc cosine CLIP ≥ 0,955, như trong audit. "Khớp cả số lượng" khi nhóm
số lượng của node ảnh hợp với lớp của node câu.

| Graph ảnh đúng | Có node khớp nhãn | Khớp cả số lượng | Riêng lớp "2" | "3–4" | "≥2" | "≥3" |
|---|---|---|---|---|---|---|
| Cấp 1 | 66,6% | 40,7% | 23,3% | 34,5% | 42,8% | 52,1% |
| merge_v1, chưa gom | 66,1% | 40,7% | 30,0% | 39,4% | 41,0% | 51,0% |
| group_v1, đã gom | 62,3% | 37,3% | 31,3% | 36,1% | 37,2% | 44,1% |

Gom nhóm làm tỉ lệ khớp giảm, vì hai lý do:

- Node đơn vốn đã đếm số vật cùng loại trong cả ảnh, còn node nhóm chỉ đếm thành viên của nhóm, nên con số thường nhỏ
  hơn.
- Một phần node nhóm đổi sang nhãn hypernym, nên ít khớp nhãn hơn.

Riêng lớp "2" thì tăng.

**Độ liền graph ảnh** (`graph_structure.json`):

| Graph ảnh | Node / ảnh | Mảnh rời / ảnh | % node ở mảnh lớn nhất |
|---|---|---|---|
| merge_v1 (thí nghiệm 3) | 24,9 | 9,8 | 48,4 |
| group_v1 | 18,8 | 7,3 | 55,8 |
| group_v1 + kNN-5 + cầu nối + node tổng | 19,6 | 1,0 | 100 |

Số liệu trên là của val; train gần như vậy. Sau khi gom, mỗi ảnh có khoảng 85 cạnh tự thêm (merge_v1: 114), và 0,8%
ảnh cần cầu nối.

**Hạ tầng.** Mỗi epoch mất 685–726 giây. Train 3 seed mất 4,79 giờ, từ 21:15 ngày 23/10 đến 02:04 ngày 24/10; peak
GPU 5,6 GB. Mỗi seed có 1 cảnh báo phân mảnh bộ nhớ lúc chấm val, đều tự hồi phục; không run nào lỗi. Test đơn vị:
`tests/test_group_count.py` có 19 test; toàn bộ 128 test pass.

## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | Epoch tốt nhất | Số epoch |
|---|---|---|---|---|---|---|
| groupcount_factual_r3 | 47.75 ± 0.22 | 74.82 ± 0.08 | 84.32 ± 0.17 | 59.97 ± 0.14 | 9, 4, 7 | 10, 6, 9 |
| fullgraph_factual_r3 (mốc) | **48.38 ± 0.17** | **75.25 ± 0.09** | **84.71 ± 0.09** | **60.51 ± 0.17** | 4, 8, 5 | 6, 10, 7 |
| raw_factual_r3 (1a) | 47.48 ± 0.10 | 74.70 ± 0.18 | 84.10 ± 0.08 | 59.72 ± 0.00 | 9, 10, 9 | 10, 10, 10 |

- R@1 từng seed của bản mới: 47,93 / 47,50 / 47,82.
- So với fullgraph (Welch, 3 seed mỗi bên): R@1 −0,63 (p = 0,019), MRR −0,54 (p = 0,014), R@5 −0,43 (p = 0,004),
  R@10 −0,39 (p = 0,040).
- So với 1a: R@1 +0,27 (p = 0,16), không có ý nghĩa thống kê.

## R@1 trên tập con câu có số lượng (`count_subset.json`)

Các tập con được xác định một lần từ lớp số lượng của câu, dùng chung cho cả hai bản. Mỗi bản chạy trên dữ liệu và
cấu trúc graph của chính nó.

| Tập con val | Số câu | groupcount R@1 | fullgraph R@1 | Chênh (p) | Chênh MRR (p) |
|---|---|---|---|---|---|
| Tất cả | 10 633 | 47.75 ± 0.22 | 48.38 ± 0.17 | −0,63 (0,019) | −0,54 (0,014) |
| Có số lượng đã biết | 4 543 | 50.10 ± 0.36 | 50.28 ± 0.18 | −0,18 (0,49) | −0,18 (0,16) |
| — có số nêu rõ (1 / 2 / 3–4 / ≥5) | 1 235 | **54.22 ± 1.13** | 52.58 ± 0.28 | **+1,65** (0,12) | **+1,05** (0,044) |
| — chỉ có số nhiều hoặc group/herd… | 3 308 | 48.56 ± 0.14 | 49.43 ± 0.31 | −0,87 (0,025) | −0,64 (0,025) |
| Không rõ số lượng | 6 090 | 46.00 ± 0.12 | 46.97 ± 0.37 | −0,97 (0,036) | −0,80 (0,030) |

`count_subset_seed0_cpu.json` là bản tính sớm cho riêng seed 0, chạy trên CPU trong lúc seed 1 đang train; số khớp với
bản 3 seed.

## Bảng g đã học (checkpoint tốt nhất, mean ± std qua 3 seed)

Hàng là lớp số lượng phía câu, cột là nhóm số lượng phía ảnh.

| Câu \ Ảnh | 1 | 2 | 3–4 | ≥5 |
|---|---|---|---|---|
| không rõ (cố định) | 0 | 0 | 0 | 0 |
| 1 | **+0.110 ± 0.015** | −0.015 ± 0.018 | −0.134 ± 0.008 | −0.172 ± 0.029 |
| 2 | −0.092 ± 0.003 | **+0.077 ± 0.006** | +0.056 ± 0.005 | −0.015 ± 0.006 |
| 3–4 | −0.098 ± 0.017 | −0.068 ± 0.009 | **+0.103 ± 0.008** | +0.031 ± 0.006 |
| ≥5 | −0.117 ± 0.013 | −0.084 ± 0.002 | −0.013 ± 0.011 | **+0.133 ± 0.008** |
| ≥2 | −0.053 ± 0.004 | +0.001 ± 0.003 | +0.034 ± 0.005 | +0.053 ± 0.008 |
| ≥3 | −0.088 ± 0.018 | −0.054 ± 0.006 | +0.037 ± 0.009 | +0.087 ± 0.004 |

Từng seed ở `count_subset.json`. Để so độ lớn: CLIP cosine giữa các node khác nhau dao động khoảng 0,6–0,95.

- Ba seed học ra gần như cùng một bảng (std ≤ 0,03).
- Mọi ô trên đường chéo đều dương; các ô lệch số lượng âm, lệch càng xa càng âm.
- "≥2" và "≥3" tăng dần theo số lượng ảnh, đúng nghĩa "ít nhất".
- Ô "2" → "3–4" dương (+0,056): câu nói "two" thì ảnh có 3–4 vật vẫn được cộng điểm. Một cách hiểu là VG hay chú thích
  thừa, hoặc các vật cùng loại bị đếm gộp.

## Các phép đo khác (so với fullgraph)

| | groupcount | fullgraph |
|---|---|---|
| Độ dồn vector bộ ba (cosine; từng seed 0,881 / 0,763 / 0,869) | 0,838 ± 0,065 | 0,869 ± 0,045 |
| Độ dồn vector vật thể (cosine) | 0,314 ± 0,045 | 0,334 ± 0,077 |
| Tương quan vật thể ~ bộ ba | 0,317 ± 0,040 | 0,323 ± 0,015 |
| Tương quan CLIP ~ bộ ba | 0,192 ± 0,027 | 0,202 ± 0,015 |
| Tương quan CLIP ~ vật thể | 0,314 ± 0,008 | 0,326 ± 0,006 |
| Kênh bộ ba đứng riêng, R@1 câu có ≥1 bộ ba | 11.28 ± 2.17 | 13.14 ± 1.25 |
| Đổi động từ / đổi chủ–tân, Δ R@1 điểm cuối | −1,13 / −0,33 | −1,02 / −0,38 |
| Gradient GAT gat1 / gat2 (probe 10 batch train) | 0,325 ± 0,009 / 0,240 ± 0,012 | 0,335 ± 0,045 / 0,256 ± 0,019 |
| Gradient bảng g (probe) | 0,140 ± 0,005 | — |
| a / b / c ở epoch tốt nhất | 0,578 / 0,115 / 0,307 | 0,618 / 0,135 / 0,247 |

Bảng đầy đủ ở `summary.md`.

## Ví dụ giải thích có dòng số lượng (`count_examples.md`, seed 0)

`scripts/improve/count_explain_examples.py` chọn ngẫu nhiên (seed 0) 6 câu val có số nêu rõ, và in giải thích kênh vật
thể cho ảnh đúng của mỗi câu. Ví dụ:

- "there are two elephants…": câu 2 elephant — ảnh 2 elephant → g = +0,073 (sim_ij 0,957).
- "Three children dressed in white…": câu 3–4 child — ảnh 3–4 kid → g = +0,111.
- "A lone surfer…": câu 1 surfer — ảnh 1 surfboard → g = +0,102. Ở ví dụ này g kéo node "surfer" ghép nhầm sang
  "surfboard", vì node đó có đúng nhóm số lượng.
- "…two pillow cases…": câu 2 pillow case — ảnh 3–4 quilt → g = +0,057, cũng là ghép nhầm.

Như vậy g không chỉ cộng điểm cho cặp đúng mà còn có thể đổi node ảnh được ghép; khi ghép nhầm, lời giải thích hiển thị
dòng sai. Ngoài ra, khi cộng g thì sim_ij có thể vượt 1 (ví dụ 1,042).

## Nhận xét

1. **Bản gộp node nhóm + số lượng kém fullgraph** 0,63 R@1 (p = 0,019) và 0,54 MRR. Cả 3 seed đều thấp hơn mọi seed của
   fullgraph. Bản này vẫn nhỉnh hơn 1a, nhưng không có ý nghĩa thống kê.
2. **Bảng g có tác dụng đúng chỗ.** Bảng học được có cấu trúc rõ và ổn định qua 3 seed. Trên 1 235 câu có số nêu rõ,
   R@1 tăng +1,65 (p = 0,12) và MRR tăng +1,05 (p = 0,044).
3. **Phần còn lại tụt khoảng 1 điểm,** và có lẽ do graph đã gom:
   - Trên 6 090 câu không có số lượng, R@1 −0,97 (p = 0,036). Với các câu này g luôn bằng 0, nên mức tụt không đến
     trực tiếp từ g.
   - Câu chỉ có số nhiều tụt −0,87, vì hàng "≥2" của g rất nhỏ (−0,05 đến +0,05) nên không bù được.
   - Lời giải thích hợp lý nhất là graph đã gom làm mất thông tin: 24% node biến mất, các vật trong một nhóm dùng chung
     một vector, và 13% node nhóm mang nhãn hypernym hoặc nhãn hoà. Kiểm tra trước khi train cũng thấy đúng hướng đó:
     khớp số lượng ở ảnh đúng giảm (40,7% → 37,3%), và mô hình fullgraph chấm thẳng trên graph đã gom tụt 0,7.
   - Thí nghiệm gộp nhiều thay đổi nên không tách chắc chắn được: g còn ảnh hưởng tới quá trình train, nên không hoàn
     toàn vô can với các câu "không rõ".
4. **Kênh bộ ba đứng riêng yếu đi** (13,1 → 11,3, std lớn). Graph gom có ít cạnh có nhãn hơn (14,2 so với 17,4 mỗi
   ảnh). Độ dồn và tương quan giữa các kênh gần như không đổi.

## Việc tiếp theo (Lộc quyết)

- **Giữ fullgraph_factual_r3 làm mô hình tốt nhất trên val.** Bản này không nên thay nó.
- **Nếu muốn giữ phần số lượng:** cách tách tự nhiên là một bản chỉ thêm bảng g trên merge_v1, không gom nhóm; node
  đơn đếm số vật cùng loại như ở dòng "merge_v1, chưa gom" của bảng khớp số lượng. Bản này giữ được lợi ích ở câu có số
  nêu rõ mà không mất thông tin do gom. Ước tính khoảng 4,5 giờ cho 3 seed.
- **Nếu dùng g trong demo giải thích,** nên hiện cả hai: node ảnh được ghép khi có g và khi không có g, để thấy được
  trường hợp g làm đổi node ghép.

## File

- Run: `groupcount_factual_r3_seed{0,1,2}/`, gồm `config.json` (commit `e3be011`), `log.jsonl`, `metrics.json`.
  Checkpoint không commit.
- Báo cáo seed 0 viết lúc vừa xong: `seed0.md`. Báo cáo này chưa gửi được cho KG vì mã liên lạc của phiên KG đã hết hạn.
- Kiểm tra: `check_zero.json`, `check_match.json`, `graph_structure.json`.
- Kết quả: `count_subset.json` (tập con và bảng g), `count_subset_seed0_cpu.json`, `summary.md`, `summary.json`,
  `triple_perturbation.json`, `part_similarity.json`, `channel_correlation.json`, `gradient_probe.json`,
  `count_examples.md`.
- Log: `logs/`. Lệnh chạy lại: `run.sh`, rồi `diagnostics.sh`.
