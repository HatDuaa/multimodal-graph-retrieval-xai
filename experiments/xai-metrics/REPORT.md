# Đo chất lượng giải thích cấp 3: fidelity, validity, sparsity, đồ thị con (val)

Việc này thuộc phần của Nhi trong bảng phân công (dòng 27–29); Lộc nhờ làm, KG giao đặc tả ngày 16/11/2026. Chỉ dùng
val, không train, không chấm test.

> **Đính chính (17/11/2026).** Bản đo lại ở `experiments/xai-metrics-v2/` (PR #38) sửa các điểm sau; số liệu dưới đây giữ nguyên như lúc chạy.
>
> - Script của bản này nay là `scripts/xai_metrics_v1.py` và `scripts/summarize_xai_metrics_v1.py`; `run.sh` đã đổi theo.
> - E của `main_r3` xếp term thô của hai kênh, trong khi fusion z-score cân mỗi kênh theo trọng số / độ phân tán trên 50 ứng viên.
>   Đưa term về thang điểm cuối thì bộ ba đứng đầu E ở 70% câu (ở đây 1%), và bỏ top-3 làm mất hạng 1 ở 34,9% câu thay vì 65,7%.
>   Số của countonly (fusion raw) không bị ảnh hưởng bởi điểm này.
> - Mục 4 (sparsity): "6–8% số phần của đồ thị ảnh" chia số dòng demo cho số phần của ảnh, hai đơn vị khác nhau; không nên dùng con số này.
>   v2 đo trên chính E: với `main_r3`, 2,65 phần đầu giữ 80% hiệu ứng, đạt ở 56% câu.
> - Mục 3 (validity): tiêu chí ở đây không khắt khe mà dễ dãi: `dedup.same_type` coi mọi người là cùng loại ("woman" ~ "man") và chấp nhận hypernym hai chiều.
>   Tiêu chí strict của v2 cho 58,3% với `main_r3`, so với 63,8% ở đây.
> - Mục 3 và kết luận 3: so sánh khớp nhóm số lượng giữa countonly và `main_r3` không công bằng, vì `main_r3` được chấm trên đồ thị cấp 1 chưa gộp trùng, nơi các hộp trùng làm tăng số đếm.
> - Mục 5 (đồ thị con): với cạnh có nhãn, bỏ cạnh ở đây là bỏ luôn bộ ba, và lần bốc ngẫu nhiên lấy từ cạnh gộp của 3 node.
>   Chỉ cắt đường truyền tin GAT (giữ bộ ba) thì mất hạng 1 ở 1,0% câu so với 0,7% khi cắt ngẫu nhiên (`main_r3`); kết luận "alpha GAT không phải lời giải thích" vẫn đúng.
> - Fidelity ở đây tính cả các câu có E rỗng (như "không đổi" cho cả E và ngẫu nhiên) và lấy mốc là lượt chấm thường; v2 loại các câu đó và lấy mốc là ảnh không sửa chấm lại cùng đường.

- **Mô hình:**
  - `countonly_factual_r3` (mô hình chính, dữ liệu count_v1, fusion raw);
  - `main_r3` (cấp 1, fusion z-score);
  - mỗi mô hình 3 seed, dùng `checkpoint_best.pt`.
- **Câu:** toàn bộ 10 633 câu val, không lấy mẫu. Cả 6 run mất 18,5 phút GPU trên RTX 4090: khoảng 208 giây mỗi seed
  countonly và 161 giây mỗi seed `main_r3`.
- **Script:**
  - `src/explain/xai_metrics.py`: sửa đồ thị một ảnh và chấm lại;
  - `scripts/xai_metrics.py`: đo, ghi `metrics.json` (chỉ số gộp theo run và nhóm) và `cases.json` (chi tiết 13 câu);
  - `scripts/summarize_xai_metrics.py`: tính mean ± std qua seed, ghi `summary.json` và in mọi bảng dưới đây
    (`logs/tables.md`);
  - `run.sh`: lệnh chạy lại cả hai bước; log ở `logs/run.log`.
- **Môi trường:** venv `~/venvs/mgrx` được cài lại ngày 16/11/2026 theo `requirements.txt`, với torch 2.14.0+cu130 và
  transformers 5.20.1. Sau khi cài:
  - `pytest` đạt 195 test, bỏ qua 1 test (`tests/test_graph_matching.py` cần gói `ot`, không có trong requirements);
  - chấm lại `countonly_factual_r3_seed0` trên val được R@1 = 49,15, trùng tuyệt đối `metrics.json` (0,491488761403);
  - R@1 của cả 6 run, tính lại trong lúc đo, đều trùng `metrics.json` (script kiểm bằng assert): 49,15 / 48,85 / 48,74
    và 46,67 / 46,56 / 46,21.
- **Lưu trữ:** mọi điểm tính trong RAM, không ghi cache vector hay dump điểm ra đĩa. Cả thư mục này dưới 1 MB. Đĩa còn
  24 GB suốt quá trình.

## Định nghĩa

### Giải thích E và các phép sửa đồ thị

**Giải thích E(q, ảnh)** xếp các phần của ảnh, gồm node vật thể và bộ ba có nhãn, theo term chính xác của từng phần:

term_j = trọng số kênh × Σ_i w_i · a_ij · sim_ij

Đây là `image_terms` trong `matched_pairs`, cũng là cột dùng cho phép thử xoá ở `experiments/improve/explanations/`.

- Với fusion raw (countonly), các term cộng lại đúng bằng phần đồ thị của điểm cuối.
- Với `main_r3`, đó là term thô trước khi z-score giữa các ứng viên.
- Ảnh được giải thích là **ảnh mô hình xếp hạng 1** cho mỗi câu. Kết quả báo riêng hai nhóm: ảnh hạng 1 đúng (correct)
  và sai (wrong).
- Ảnh hạng 1 không có phần nào thì E rỗng: 31–36 câu mỗi seed với countonly, 39–45 câu với `main_r3`. Các câu này vẫn
  được tính trong số câu, nhưng không có phần nào để bỏ.

**Sửa đồ thị** (`src/explain/xai_metrics.py`):

- Bỏ một node là bỏ node đó cùng mọi cạnh chạm vào nó: cạnh có nhãn, cạnh kNN `[none]` và cạnh nối node tổng. Cạnh kNN
  **không** được dựng lại cho các node còn lại. Như vậy bỏ một node cũng bỏ luôn các bộ ba chạm vào nó.
- Bỏ một bộ ba là bỏ đúng cạnh có nhãn đó.
- Sau khi sửa, ảnh được mã hoá lại bằng cả mô hình, kể cả GAT. 49 ứng viên còn lại giữ nguyên điểm kênh; fusion được tính
  lại trên cả 50 ứng viên. Vì vậy với các ứng viên kia, chỉ giá trị lấp (trung bình của các ứng viên đã biết) của
  `fuse_raw` có thể đổi, hoặc z-score với `main_r3`.
- **Chỉ giữ top-k** giữ các phần đó cùng node tổng. Một bộ ba được giữ thì giữ luôn hai node đầu–cuối làm "carrier": node
  vẫn ở trong đồ thị và vẫn truyền tin qua GAT, nhưng không được chấm ở kênh vật thể.
- **Ngẫu nhiên:** 10 lần bốc, mỗi lần cùng số node và cùng số bộ ba với top-k, bốc đều trong mọi node / bộ ba của ảnh.
  rng = `default_rng([2026, seed, image_id, chỉ số câu])`.

**Unit test** (`tests/test_xai_metrics.py`, 7 test, đều đạt):

- bỏ tập rỗng hoặc giữ đủ mọi phần thì điểm và hạng không đổi (sai số ≤ 1e-6);
- bỏ hết bộ ba thì kênh bộ ba nhận đúng giá trị lấp, và tương tự khi bỏ hết node;
- bỏ node thì bỏ đúng các cạnh chạm vào nó, không dựng lại kNN;
- bộ ba được giữ thì giữ hai node carrier không chấm;
- attention có đúng một alpha cho mỗi cạnh, và cạnh tự nối nhận NaN (xem mục đồ thị con);
- tổng các dòng `row_terms` bằng điểm kênh.

Trên val, chênh lệch tối đa giữa điểm chấm lại khi không sửa gì và điểm gốc là 1,8e-7 với countonly và 1,8e-5 với
`main_r3` (sai số float32 sau z-score).

### Các độ đo

**Fidelity, k = 1, 3, 5:**

- **Fid+** bỏ top-k phần của E, ghi Δ điểm cuối, hạng mới và tỉ lệ mất hạng 1.
- **Fid−** chỉ giữ top-k phần, ghi cùng các số đó.
- Cả hai đều so với trung bình của 10 lần chọn ngẫu nhiên.
- **Phần hiệu ứng:** "rỗng" là ảnh bị bỏ mọi phần; khi đó cả hai kênh nhận giá trị lấp.
  - Fid+: phần bị bỏ = (gốc − sau khi bỏ) / (gốc − rỗng).
  - Fid−: phần giữ lại = (sau khi giữ − rỗng) / (gốc − rỗng).
  - Báo tỉ số tổng qua các câu, kèm trung vị. Chỉ tính trên các câu có gốc − rỗng > 1e-4: khoảng 10 520 câu mỗi seed với
    countonly, 10 450 với `main_r3`.
  - Trung bình của tỉ số từng câu cũng có trong `metrics.json`, nhưng không dùng: nó bị vài câu có mẫu số rất nhỏ kéo
    lệch.

**Validity:**

- **Phản thực tế (counterfactual):** tỉ lệ câu mà ảnh mất hạng 1 khi bỏ top-k, so với khi bỏ ngẫu nhiên. Đây là cột "mất
  hạng 1" của bảng Fid+.
- **Ngữ nghĩa:** với mỗi dòng vật thể của E, node ảnh có attention cao nhất có mang nhãn của node câu không. Tiêu chí như
  `count_match_changes`, cộng thêm WordNet:
  - cùng nhãn sau alias VG;
  - hoặc WordNet cùng loại (`src/graph/dedup.same_type`, sense danh từ đầu tiên của từ trong câu);
  - hoặc cosine CLIP đóng băng của hai node ≥ 0,955.
- Ghi riêng thêm cột "cùng danh từ chính" sau alias, ví dụ "dining table" ~ "table". Cột này **không** tính vào "any":
  WordNet 3.0 xếp `dining_table.n.01` dưới `table.n.03` (bàn đã bày bữa), trong khi bàn của VG là `table.n.02`.
- **Khớp nhóm số lượng:** với node câu có lớp số lượng nêu rõ, nhóm số lượng của node ảnh có tương thích không
  (`src/graph/counts.compatible`).
  - Ảnh cấp 1 lấy synset từ `data/processed/geometry/objects_val.json` và nhóm số lượng theo cùng luật đếm của count_v1.
  - Node câu cấp 1 lấy lớp số lượng bằng cùng luật spaCy.

**Sparsity** dùng các dòng của cột "So với TB top-50" trong demo: mỗi dòng trừ trung bình của nó trên 50 ứng viên, và
các dòng cộng lại đúng bằng biên của ảnh so với trung bình.

- Đếm số dòng ít nhất, xếp theo Δ giảm dần, để đạt 80% / 90% biên.
- Biến thể "graph": chỉ các dòng đồ thị, với biên của phần đồ thị (term thô; với `main_r3` là trước z-score).
- Biến thể "with_clip": thêm dòng CLIP, với biên của điểm cuối. Chỉ có ở fusion raw.
- Báo cả số dòng tuyệt đối, tỉ lệ trên số dòng, và tỉ lệ trên số phần của đồ thị ảnh (node + bộ ba có nhãn). Chỉ tính khi
  biên dương.

**Đồ thị con giải thích:**

- Với 3 node đầu của E, lấy 3 cạnh đi vào có alpha GAT lớn nhất, alpha lấy trung bình qua 2 lớp × 4 head.
- Edge fidelity = bỏ các cạnh này, so với bỏ cùng số cạnh vào ngẫu nhiên của chính các node đó (10 lần).
- Khi các node chỉ có đúng số cạnh vào bằng số cạnh bỏ, lần bốc ngẫu nhiên trùng với top, nên phép so sánh không có
  nghĩa. Vì vậy báo thêm tập "có lựa chọn": các câu mà số cạnh vào lớn hơn số cạnh bỏ.
- `GATv2Conv` bỏ cạnh tự nối có sẵn trong đồ thị (quan hệ có subject = object, có trong đồ thị cấp 1) rồi thêm self-loop
  của nó. Vì vậy alpha được ghép với các cạnh không tự nối theo thứ tự, còn cạnh tự nối không được chọn.

**Đơn vị:** Δ điểm của countonly tính bằng đơn vị điểm raw (khoảng 0,01). Với `main_r3`, Δ tính bằng đơn vị z-score
(khoảng 1). Vì vậy hai mô hình chỉ so được qua hạng, tỉ lệ mất hạng 1 và tỉ lệ hiệu ứng.

## Kết quả (val, 10 633 câu, mean ± std qua 3 seed)

Các bảng dưới đây lấy nhóm "tất cả câu". Bảng đầy đủ theo nhóm đúng / sai nằm ở phụ lục.

### 1. Fidelity

**Fid+: bỏ top-k phần của E, so với bỏ k phần ngẫu nhiên cùng loại.**

| Mô hình | k | Mất hạng 1: top (%) | Ngẫu nhiên (%) | Hạng mới: top | Ngẫu nhiên | Phần hiệu ứng bị bỏ, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên (%) |
|---|---|---|---|---|---|---|---|
| countonly | 1 | 34,3 ± 4,8 | 4,4 ± 0,5 | 2,30 ± 0,22 | 1,14 ± 0,03 | 30,6 / 18,9 | 2,0 / 0,0 |
| countonly | 3 | 58,0 ± 4,1 | 9,4 ± 0,9 | 4,73 ± 0,57 | 1,36 ± 0,03 | 63,4 / 56,6 | 5,9 / 1,0 |
| countonly | 5 | 68,0 ± 3,0 | 12,9 ± 0,7 | 6,39 ± 0,58 | 1,54 ± 0,02 | 80,4 / 77,8 | 8,8 / 2,2 |
| main_r3 | 1 | 40,4 ± 0,6 | 5,0 ± 0,0 | 2,76 ± 0,06 | 1,18 ± 0,00 | 44,4 / 34,8 | 3,5 / 0,5 |
| main_r3 | 3 | 65,7 ± 0,3 | 12,0 ± 0,1 | 6,46 ± 0,02 | 1,52 ± 0,01 | 96,0 / 92,7 | 10,6 / 3,8 |
| main_r3 | 5 | 75,1 ± 0,4 | 16,6 ± 0,1 | 9,22 ± 0,04 | 1,78 ± 0,01 | 126,0 / 126,5 | 15,6 / 7,3 |

**Fid−: chỉ giữ top-k phần (và node tổng), so với giữ k phần ngẫu nhiên.**

| Mô hình | k | Hạng: top | Ngẫu nhiên | Mất hạng 1: top (%) | Ngẫu nhiên (%) | Phần hiệu ứng giữ lại, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên (%) |
|---|---|---|---|---|---|---|---|
| countonly | 1 | 4,18 ± 0,83 | 10,13 ± 2,26 | 51,0 ± 6,3 | 69,2 ± 7,1 | 45,7 / 47,4 | −2,9 / −7,4 |
| countonly | 3 | 2,22 ± 0,74 | 8,13 ± 1,40 | 21,7 ± 6,7 | 66,3 ± 5,0 | 99,7 / 106,4 | 12,2 / 9,9 |
| countonly | 5 | 1,86 ± 0,54 | 6,93 ± 0,96 | 14,2 ± 5,7 | 62,1 ± 3,6 | 116,6 / 122,6 | 24,0 / 21,8 |
| main_r3 | 1 | 6,37 ± 0,34 | 18,91 ± 0,34 | 64,8 ± 1,9 | 87,2 ± 0,6 | 3,0 / 6,3 | −109,4 / −109,6 |
| main_r3 | 3 | 2,07 ± 0,05 | 11,09 ± 0,23 | 22,7 ± 0,9 | 75,2 ± 0,8 | 85,4 / 95,5 | −40,4 / −41,3 |
| main_r3 | 5 | 1,45 ± 0,01 | 8,14 ± 0,19 | 11,2 ± 0,7 | 66,2 ± 0,9 | 107,9 / 110,7 | −9,5 / −9,6 |

Δ điểm theo đơn vị của từng mô hình nằm ở phụ lục. Ví dụ, bỏ top-3 làm điểm countonly giảm 0,0196, còn bỏ ngẫu nhiên
giảm 0,0018. Với `main_r3`, hai con số là 0,834 và 0,091.

- **Hai chiều của fidelity đều cách xa ngẫu nhiên.**
  - Bỏ top-3 làm ảnh mất hạng 1 ở 58% câu (countonly) và 66% (`main_r3`), so với 9–12% khi bỏ ngẫu nhiên.
  - Chỉ giữ top-3 thì ảnh còn ở hạng 2,2 (countonly) và 2,1 (`main_r3`), so với hạng 8–11 khi giữ ngẫu nhiên.
- **Term của E cộng đúng ra điểm, nhưng hiệu ứng khi bỏ phần không cộng tuyến tính.**
  - Chỉ giữ top-5 cho điểm **cao hơn** ảnh gốc: 117% hiệu ứng với countonly, 108% với `main_r3`.
  - Lý do: Σ_j a_ij · sim_ij là trung bình có trọng số softmax trên các phần của ảnh (Σ_j a_ij = 1), không phải tổng. Bỏ
    các phần khớp kém làm trung bình tăng lên.
  - Vì vậy E chỉ nói phần nào mang điểm. E không nói các phần khớp kém đang kéo điểm xuống bao nhiêu.
- **Fidelity theo tỉ lệ hiệu ứng của `main_r3` cần đọc thận trọng.**
  - Với z-score, ảnh "rỗng" không phải mốc trung tính: kênh rỗng nhận z = 0, tức ngang trung bình top-50.
  - Vì vậy giữ k phần ngẫu nhiên thường cho điểm thấp hơn cả ảnh rỗng, nên tỉ lệ hiệu ứng ngẫu nhiên âm (−109%). Hạng và
    tỉ lệ mất hạng 1 không bị vấn đề này.
- **Fid của countonly dao động giữa các seed nhiều hơn** (±4,8 so với ±0,6), vì tỉ lệ bộ ba đứng đầu E khác nhau theo
  seed. Xem mục 2.

### 2. Kênh bộ ba: nặng ký trong E nhưng bỏ đi ít tác dụng

**Hình dạng của E.**

| Mô hình | Node / ảnh | Bộ ba / ảnh | Bộ ba trong top-1 / top-3 / top-5 của E (%) | Top-3 node chiếm (% term dương của kênh vật thể) | Top-3 bộ ba chiếm (% term dương của kênh bộ ba) |
|---|---|---|---|---|---|
| countonly | 25,3 | 18,1 | 44 / 50 / 56 | 68,0 ± 0,5 | 38,7 ± 1,3 |
| main_r3 | 27,4 | 18,7 | 1 / 5 / 11 | 63,1 ± 0,9 | 22,8 ± 0,2 |

**Fid k = 1 tách theo loại phần đứng đầu E.** Số câu là trung bình mỗi seed. Ngẫu nhiên ở đây là một phần cùng loại.

| Mô hình | Top-1 của E | Số câu | Bỏ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) | Chỉ giữ top-1: hạng | Ngẫu nhiên |
|---|---|---|---|---|---|---|
| countonly | node | 5 982 | 46,5 ± 0,8 | 4,7 ± 0,1 | 4,21 ± 0,73 | 14,48 ± 1,71 |
| countonly | bộ ba | 4 618 | 19,7 ± 3,2 | 4,2 ± 1,0 | 3,70 ± 1,80 | 4,02 ± 1,95 |
| main_r3 | node | 10 466 | 40,8 ± 0,6 | 4,9 ± 0,0 | 6,45 ± 0,33 | 19,17 ± 0,30 |
| main_r3 | bộ ba | 124 | 18,6 ± 6,1 | 12,0 ± 2,7 | 1,83 ± 0,15 | 2,59 ± 0,31 |

- **Ở countonly, bộ ba đứng đầu E trong 44% câu, nhưng bỏ nó chỉ làm mất hạng 1 ở 20% câu.** Khi node đứng đầu, con số
  là 47%.
- **Chỉ giữ bộ ba đứng đầu gần như không hơn giữ một bộ ba bất kỳ:** hạng 3,70 so với 4,02. Với node, chênh lệch là 4,21
  so với 14,48.
  - So sánh hạng giữa node và bộ ba không công bằng: khi chỉ giữ một bộ ba, kênh vật thể bị lấp bằng trung bình. Chỉ nên
    so top với ngẫu nhiên trong cùng loại.
- **Kênh bộ ba dàn trải:**
  - top-3 bộ ba chỉ chiếm 39% term dương của kênh bộ ba (countonly) và 23% (`main_r3`);
  - với kênh vật thể, top-3 node chiếm 63–68%;
  - điều này khớp với nhận xét ở `experiments/improve/explanations/`: attention của cặp khớp nhất là 0,17 ở kênh bộ ba,
    so với 0,51 ở kênh vật thể.
  - Cách giải thích phù hợp với số liệu, nhưng chưa kiểm riêng: bộ ba đứng đầu E vì trọng số kênh c lớn và nhiều bộ ba
    cùng khớp vừa phải; bỏ một bộ ba thì attention dồn sang các bộ ba tương tự, nên điểm gần như không đổi.
- **Mức này phụ thuộc seed:**
  - seed 2 học c = 0,227 và đặt bộ ba đứng đầu E ở 26% câu;
  - seed 0 và 1 học c = 0,333 / 0,352 và đặt bộ ba đứng đầu ở 51% / 54% câu;
  - vì vậy Fid+ k = 1 của seed 2 cao hơn: mất hạng 1 ở 39,8%, so với 31,9% và 31,3%.

### 3. Validity

**Phản thực tế:** xem cột "mất hạng 1" của bảng Fid+. Bỏ top-1 / top-3 / top-5 làm mất hạng 1 ở 34 / 58 / 68% câu
(countonly), so với 4 / 9 / 13% khi bỏ ngẫu nhiên.

**Ngữ nghĩa** (dòng vật thể của E; số dòng là trung bình mỗi seed):

| Mô hình | Dòng | Khớp nhãn, any (%) | alias | WordNet | CLIP ≥ 0,955 | Cùng danh từ chính | any hoặc danh từ chính | Dòng có số nêu rõ | Khớp nhóm số lượng (%) |
|---|---|---|---|---|---|---|---|---|---|
| countonly | 29 687 | 70,0 ± 1,0 | 57,0 ± 1,0 | 54,6 ± 0,7 | 63,7 ± 1,0 | 62,1 ± 1,1 | 71,7 ± 1,0 | 5 857 | 66,7 ± 0,8 |
| main_r3 | 32 382 | 63,8 ± 0,2 | 51,6 ± 0,0 | 49,5 ± 0,1 | 57,4 ± 0,1 | 58,1 ± 0,0 | 65,6 ± 0,2 | 6 817 | 52,3 ± 0,6 |

- Node ảnh mà mô hình chú ý nhất mang đúng nhãn của node câu ở 70% dòng với countonly, so với 64% ở `main_r3`.
- Với node câu có số lượng nêu rõ, nhóm số lượng của node ảnh tương thích ở 67% dòng, so với 52% ở `main_r3`.
- Bảng g và việc gộp trùng node làm giải thích đúng nghĩa hơn, nhất là phần số lượng.
- Khoảng 30% dòng vẫn khớp một node mang nhãn khác. Tiêu chí ở đây khắt khe: thêm "cùng danh từ chính" chỉ tăng thêm
  1,7 điểm.

### 4. Sparsity

| Mô hình | Biến thể | Số dòng TB | Dòng cho 80% biên | Dòng cho 90% | % số dòng (80 / 90) | Dòng đồ thị (80 / 90) | % phần của đồ thị ảnh (80 / 90) |
|---|---|---|---|---|---|---|---|
| countonly | graph | 4,57 | 2,52 ± 0,04 | 3,06 ± 0,05 | 58,9 / 70,7 | 2,52 / 3,06 | 8,1 / 9,7 |
| countonly | with_clip | 5,56 | 2,68 ± 0,04 | 3,42 ± 0,03 | 50,6 / 64,1 | 1,71 / 2,44 | 5,5 / 7,8 |
| main_r3 | graph | 4,71 | 1,95 ± 0,01 | 2,28 ± 0,01 | 46,5 / 53,6 | 1,95 / 2,28 | 5,7 / 6,6 |

- Khoảng 2,5 dòng (countonly) và 2 dòng (`main_r3`) đủ cho 80% biên của phần đồ thị, tức 6–8% số phần của đồ thị ảnh
  (trung bình 43 phần với countonly, 46 với `main_r3`).
- Tính cả CLIP thì 2,7 dòng đủ cho 80% biên của điểm cuối; trong đó trung bình 1,7 dòng là dòng đồ thị.
- Giải thích của `main_r3` tập trung hơn vì E của nó gần như chỉ có node. Giải thích của countonly trải trên nhiều dòng
  hơn một chút, do có bộ ba.

### 5. Đồ thị con theo attention GAT

Tập "có lựa chọn": các câu mà số cạnh vào lớn hơn số cạnh bỏ.

| Mô hình | Câu (có lựa chọn / có cạnh vào) | Cạnh bỏ / cạnh vào | Loại cạnh được chọn (có nhãn / kNN / tổng, %) | Δ điểm: top | Ngẫu nhiên | Mất hạng 1: top (%) | Ngẫu nhiên (%) |
|---|---|---|---|---|---|---|---|
| countonly | 10 481 / 10 600 | 9,00 / 21,11 | 17 / 50 / 33 | +0,0013 ± 0,0005 | +0,0013 ± 0,0002 | 6,6 ± 1,8 | 3,8 ± 0,9 |
| main_r3 | 2 801 / 9 560 | 5,25 / 9,50 | 100 / 0 / 0 | −0,0163 ± 0,0022 | −0,0129 ± 0,0018 | 5,7 ± 0,6 | 5,0 ± 0,5 |

- **Bỏ các cạnh có alpha cao nhất gần như không khác bỏ cạnh ngẫu nhiên.**
  - countonly: Δ điểm bằng nhau (+0,0013). Tỉ lệ mất hạng 1 là 6,6% so với 3,8%.
  - `main_r3`: Δ là −0,016 so với −0,013. Tỉ lệ mất hạng 1 là 5,7% so với 5,0%.
- Với countonly, một nửa số cạnh được chọn là cạnh kNN `[none]` và một phần ba là cạnh từ node tổng. Chỉ 17% là quan hệ
  có nhãn.
- Ở `main_r3`, chỉ 29% số câu có lựa chọn: phần lớn node đứng đầu E có ≤ 3 cạnh vào.
- **Kết luận:** alpha GAT cho biết node lấy tin từ đâu, nhưng không cho biết cạnh nào quyết định điểm.
  - Ablation `experiments/improve/no-edges/` cho thấy cạnh quan trọng ở tổng thể: bỏ cạnh lúc train làm R@1 giảm 2,39.
  - Nhưng lúc chấm, không có nhóm nhỏ cạnh alpha cao nào mang hiệu ứng đó: tác dụng của cạnh trải đều.

### 6. Ảnh hạng 1 đúng và sai

| Mô hình | Nhóm | Số câu (TB/seed) | Bỏ top-3: mất hạng 1 (%) | Ngẫu nhiên (%) | Chỉ giữ top-3: hạng | Ngẫu nhiên | Khớp nhãn, any (%) | Bộ ba ở top-1 của E (%) | Cách biệt hạng 1 – hạng 2 |
|---|---|---|---|---|---|---|---|---|---|
| countonly | đúng | 5 201 | 45,3 ± 3,7 | 6,1 ± 0,6 | 1,56 ± 0,31 | 6,11 ± 0,92 | 71,8 ± 0,8 | 37 | 0,0271 ± 0,0019 |
| countonly | sai | 5 432 | 70,1 ± 4,4 | 12,5 ± 1,1 | 2,85 ± 1,15 | 10,06 ± 1,85 | 68,1 ± 1,3 | 50 | 0,0070 ± 0,0007 |
| main_r3 | đúng | 4 942 | 49,2 ± 0,4 | 7,8 ± 0,2 | 1,52 ± 0,03 | 7,28 ± 0,15 | 66,4 ± 0,3 | 1 | 1,0043 ± 0,0051 |
| main_r3 | sai | 5 691 | 79,9 ± 0,5 | 15,7 ± 0,1 | 2,55 ± 0,08 | 14,39 ± 0,32 | 61,4 ± 0,3 | 1 | 0,3015 ± 0,0038 |

- Khi ảnh hạng 1 sai, E vẫn trung thực như khi đúng: top-3 vẫn cách xa ngẫu nhiên.
- Ảnh hạng 1 sai mất hạng 1 dễ hơn vì cách biệt với hạng 2 nhỏ hơn khoảng 4 lần (countonly) và 3 lần (`main_r3`).
- E của ảnh sai vì vậy cho thấy đúng phần nào đã đẩy ảnh sai lên. Với countonly, đó thường là bộ ba: 50% so với 37% ở ảnh
  đúng. Đây là tương quan, chưa phải nguyên nhân.

## Bảng bằng chứng: 13 câu (seed 0)

Gồm 10 ca của `experiments/improve/explanations/` và 3 ví dụ so sánh với cấp 1 ở đó.

- Ảnh được giải thích là ảnh **hạng 1** của từng mô hình, không phải ảnh đúng như ở báo cáo kia.
- Các cột Fid dùng k = 3. "Ngẫu nhiên" là trung bình 10 lần bốc.
- Cột đồ thị con ghi 3 trong 9 cạnh được chọn, Δ điểm khi bỏ cả 9 cạnh, và "số cạnh bỏ / số cạnh vào".
- Chi tiết đầy đủ nằm trong `cases.json`: top-5 của E, các dòng vật thể kèm kết quả khớp nhãn, Fid với k = 1, 3, 5, và cả
  9 cạnh.

| Câu (caption_id) | Mô hình | Ảnh hạng 1 | E top-3 (term) | Bỏ top-3: Δ điểm, hạng (ngẫu nhiên) | Chỉ giữ top-3: Δ điểm, hạng (ngẫu nhiên) | Dòng cho 80% biên | Đồ thị con: 3 cạnh vào đầu (alpha); Δ điểm top (ngẫu nhiên), số cạnh |
|---|---|---|---|---|---|---|---|
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | countonly | 2377477 (đúng) | table (0,028); room (0,025); stove (0,016) | −0,016, 4 (0,000, 1,0) | 0,016, 1 (−0,020, 6,7) | 3 | [global]→table [global] (0,20), chair→table [completion] (0,11), rug→table [completion] (0,07); 0,0044 (0,0039), 9/24 |
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | main_r3 | 2377477 (đúng) | stove (0,065); table (0,053); room (0,034) | −0,429, 2 (0,012, 1,0) | 0,271, 1 (−0,825, 7,9) | 2 | fire→stove [labelled] (0,02), wood→stove [labelled] (0,02), stove pipe→stove [labelled] (0,01); 0,0092 (0,0085), 6/12 |
| A cow standing along side of a large river. (422698_1) | countonly | 2351201 (đúng) | cow standing ear (0,058); plant along water side (0,035); cow standing nose (0,034) | −0,008, 1 (0,001, 1,0) | 0,004, 1 (0,001, 1,0) | 2 | [global]→river [global] (0,19), water→river [completion] (0,06), river bank→river [completion] (0,06); −0,0028 (0,0005), 9/19 |
| A cow standing along side of a large river. (422698_1) | main_r3 | 2351201 (đúng) | river (0,092); river bank (0,059); cow (0,057) | −0,432, 1 (0,051, 1,0) | 0,135, 1 (−1,343, 11,2) | 2 | không có cạnh vào |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | countonly | 2329851 (đúng) | pen (0,052); trainer observing dog (0,032); dog (0,031) | −0,020, 2 (−0,004, 1,3) | −0,015, 2 (−0,055, 6,0) | 3 | [global]→pen [global] (0,27), dog→pen [completion] (0,03), ground→pen [completion] (0,03); −0,0051 (−0,0000), 9/18 |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | main_r3 | 2329851 (đúng) | pen (0,186); trainer (0,031); sheep (0,028) | −0,994, 5 (−0,158, 2,0) | 0,399, 1 (−1,194, 11,5) | 2 | pen→sheep [labelled] (0,01); −0,0595 (−0,0595), 1/1 |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | countonly | 2402283 (đúng) | donut (0,030); bite (0,026); person holding donut (0,023) | −0,026, 4 (−0,001, 1,4) | 0,007, 1 (−0,023, 4,7) | 4 | [global]→donut [global] (0,21), hand→donut [labelled] (0,04), person→donut [labelled] (0,03); 0,0013 (0,0016), 9/37 |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | main_r3 | 2402283 (đúng) | donut (0,156); bite (0,120); chocolate cream (0,008) | −1,410, 11 (−0,226, 2,4) | 0,143, 1 (−1,095, 8,0) | 2 | chocolate cream→donut [labelled] (0,03), bite→donut [labelled] (0,03), thumb→donut [labelled] (0,02); 0,0112 (0,0118), 8/21 |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | countonly | 1159610 (đúng) | dog (0,047); umbrella (0,040); umbrella (0,026) | −0,069, 22 (−0,013, 1,8) | 0,000, 1 (−0,049, 14,1) | 3 | [global]→dog [global] (0,19), man→dog [completion] (0,05), jean→dog [completion] (0,04); 0,0011 (0,0014), 9/18 |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | main_r3 | 1159610 (đúng) | dog (0,056); umbrella (0,042); umbrella (0,042) | −1,159, 5 (−0,220, 1,4) | −0,126, 1 (−0,960, 3,9) | 2 | không có cạnh vào |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | countonly | 2366682 (sai) | banana in plate (0,069); banana in paper plates (0,061); banana in paper plates (0,059) | −0,006, 1 (0,000, 1,0) | −0,008, 1 (−0,008, 1,0) | 3 | [global]→plate [global] (0,19), banana→plate [labelled] (0,07), counter→plate [completion] (0,06); 0,0043 (0,0031), 9/22 |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | main_r3 | 2366682 (sai) | plate (0,048); plate (0,047); banana in plate (0,020) | −0,352, 2 (−0,022, 1,1) | −0,535, 2 (−0,477, 2,8) | 2 | banana→plate [labelled] (0,04), banana→plate [labelled] (0,03), banana→plate [labelled] (0,03); −0,1806 (−0,1806), 3/3 |
| A train riding along the tracks along a bank of dirt. (279933_1) | countonly | 2357910 (sai) | windshield wiper on train (0,024); gravel underneath track (0,020); logo on train (0,017) | −0,002, 2 (0,000, 1,3) | −0,007, 3 (−0,007, 3,1) | 3 | [global]→train [global] (0,23), train→train [completion] (0,07), forest→train [completion] (0,06); 0,0007 (0,0004), 9/21 |
| A train riding along the tracks along a bank of dirt. (279933_1) | main_r3 | 2378096 (sai) | dirt (0,101); track (0,056); 413 (0,039) | −0,756, 4 (−0,040, 1,0) | 0,500, 1 (−1,724, 20,2) | 2 | không có cạnh vào |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | countonly | 2353248 (sai) | bus (0,035); street (0,021); bus has bumper (0,014) | −0,020, 9 (−0,001, 1,5) | 0,008, 1 (−0,015, 7,4) | 2 | [global]→bus [global] (0,26), door→bus [completion] (0,03), windshield→bus [completion] (0,03); 0,0002 (0,0015), 9/20 |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | main_r3 | 2355061 (đúng) | bus (0,034); bus (0,034); bus (0,034) | −0,399, 3 (−0,005, 1,0) | −0,315, 3 (−0,572, 5,8) | 1 | sign→bus [labelled] (0,00), trim→bus [labelled] (0,00), sign→bus [labelled] (0,00); 0,0001 (0,0001), 5/6 |
| Two men standing in a living room playing video game with remotes. (34471_0) | countonly | 2395602 (sai) | wii controller (0,060); man wearing wrist band (0,048); man wearing short (0,042) | −0,022, 2 (−0,003, 1,1) | −0,005, 1 (−0,027, 6,0) | 3 | [global]→wii controller [global] (0,21), man→wii controller [labelled] (0,14), shirt→wii controller [completion] (0,05); −0,0044 (−0,0021), 9/19 |
| Two men standing in a living room playing video game with remotes. (34471_0) | main_r3 | 2395602 (sai) | wii controller (0,163); man (0,025); man (0,025) | −0,718, 3 (−0,299, 1,8) | −0,121, 1 (−1,033, 8,7) | 3 | man→wii controller [labelled] (0,00), glass→man [labelled] (0,01); −0,0204 (−0,0204), 2/2 |
| A batter is preparing to hit the ball at a game. (526446_0) | countonly | 2417935 (sai) | helmet has black/face mask (0,068); woman swinging bat (0,066); black bar on helmet (0,063) | −0,007, 1 (−0,001, 1,0) | −0,043, 44 (−0,044, 44,4) | 2 | [global]→ball [global] (0,28), bat→ball [completion] (0,05), thread→ball [completion] (0,03); −0,0019 (−0,0009), 9/19 |
| A batter is preparing to hit the ball at a game. (526446_0) | main_r3 | 2397612 (sai) | catcher (0,360); dirt (0,013); guard (0,010) | −2,260, 45 (−0,843, 21,9) | 0,127, 1 (−0,822, 22,0) | 2 | a→guard [labelled] (0,01); 0,0014 (0,0014), 1/1 |
| The two people are walking in the snow with their ski equipment. (472772_2) | countonly | 2384371 (đúng) | person carrying ski (0,083); woman (0,017); person (0,016) | −0,017, 3 (−0,002, 1,0) | 0,025, 1 (−0,023, 14,1) | 1 | [global]→woman [global] (0,30), people→woman [completion] (0,06), combat→woman [completion] (0,04); −0,0083 (−0,0009), 9/19 |
| The two people are walking in the snow with their ski equipment. (472772_2) | main_r3 | 2352632 (sai) | people (0,072); snow (0,071); ski (0,028) | −1,186, 18 (−0,009, 1,1) | 0,497, 1 (−1,490, 23,8) | 1 | people→snow [labelled] (0,04), people→ski [labelled] (0,00); −0,1533 (−0,1533), 2/2 |
| A snow board sitting upright with a view of a mountain. (237041_4) | countonly | 2320074 (đúng) | logo on snowboard (0,050); design on snowboard (0,039); snowboard (0,033) | −0,024, 2 (−0,004, 1,2) | −0,026, 3 (−0,024, 4,1) | 3 | [global]→snowboard [global] (0,29), figure→snowboard [labelled] (0,12), man→snowboard [completion] (0,04); −0,0044 (0,0010), 9/19 |
| A snow board sitting upright with a view of a mountain. (237041_4) | main_r3 | 2412723 (sai) | snowboard (0,116); mountain (0,051); mountain peaks (0,031) | −0,704, 3 (−0,144, 1,0) | −0,220, 1 (−1,169, 11,3) | 2 | man→snowboard [labelled] (0,00), man→snowboard [labelled] (0,00), man→snowboard [labelled] (0,00); 0,1122 (0,0738), 6/7 |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | countonly | 2385674 (đúng) | umbrella (0,037); rain (0,023); table (0,014) | −0,025, 8 (−0,005, 2,5) | −0,002, 1 (−0,027, 8,9) | 4 | [global]→umbrella [global] (0,23), woman→umbrella [labelled] (0,08), man→umbrella [labelled] (0,06); −0,0016 (0,0006), 9/25 |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | main_r3 | 2360307 (sai) | umbrella (0,032); umbrella (0,032); lady (0,031) | −0,404, 5 (−0,067, 1,4) | 0,061, 1 (−0,384, 3,5) | 2 | trim→umbrella [labelled] (0,03), coat→umbrella [labelled] (0,00), red shoes→lady [labelled] (0,01); −0,0033 (−0,0033), 4/4 |

Đọc bảng:

- **E gồm node thì vừa trung thực vừa đủ.** Câu 574350_4 (countonly), E là dog, umbrella, umbrella:
  - bỏ ba node này làm ảnh đúng rơi từ hạng 1 xuống hạng 22, còn bỏ ngẫu nhiên chỉ xuống hạng 1,8;
  - chỉ giữ ba node này thì ảnh vẫn hạng 1, còn giữ ngẫu nhiên thì xuống hạng 14,1.
- **E gồm bộ ba thì yếu.** Câu 526446_0 (countonly, ảnh sai), E là ba bộ ba về mũ bảo hiểm và gậy:
  - bỏ chúng thì ảnh vẫn hạng 1;
  - chỉ giữ chúng thì ảnh rơi xuống hạng 44, ngang giữ ngẫu nhiên (44,4).
  - Câu 1153_3 và 279933_1 cũng vậy: các bộ ba "banana in plate" hay "windshield wiper on train" đứng đầu E, nhưng bỏ hay
    giữ chúng đều ra kết quả gần như ngẫu nhiên.
- **Cùng một ảnh, hai mô hình có thể giải thích khác nhau.** Câu 472772_2:
  - countonly chọn đúng ảnh nhờ một bộ ba nổi trội, "person carrying ski" (term 0,083, gấp 5 lần phần thứ hai). Đây là ca
    hiếm mà một dòng đủ cho 80% biên.
  - `main_r3` chọn ảnh sai nhờ people, snow và ski; bỏ ba node đó thì ảnh sai rơi xuống hạng 18.
- **Đồ thị con theo attention phần lớn là cạnh từ node tổng và cạnh kNN.** Bỏ chúng làm điểm đổi rất ít: |Δ| < 0,01 ở cả
  13 câu của countonly.

## Kết luận

1. **Giải thích sinh từ term của điểm là trung thực.**
   - Bỏ 3 phần đứng đầu E làm ảnh mất hạng 1 ở 58% câu với countonly và 66% với `main_r3`, so với 9–12% khi bỏ 3 phần
     ngẫu nhiên.
   - Chỉ giữ 3 phần đó (cùng node tổng) thì ảnh còn ở hạng 2,2 với countonly, so với hạng 8,1 khi giữ ngẫu nhiên.
   - Giữ 3 phần đó cũng giữ 100% (countonly) và 85% (`main_r3`) hiệu ứng của đồ thị theo tỉ số tổng.
   - Kết quả này có ý nghĩa vì nó không hiển nhiên: bỏ một node làm đổi tin GAT của các node hàng xóm và làm softmax chia
     lại attention. Thứ tự của E vẫn giữ được qua các tương tác phi tuyến đó.
2. **Giải thích gọn:** khoảng 2,5 dòng (countonly) và 2 dòng (`main_r3`) đủ cho 80% biên của phần đồ thị, tức khoảng 8% và
   6% số phần của đồ thị ảnh.
3. **Giải thích của countonly đúng nghĩa hơn cấp 1.**
   - Node ảnh được chú ý nhất mang đúng nhãn ở 70,0% dòng vật thể, so với 63,8% ở `main_r3`.
   - Nhóm số lượng tương thích ở 66,7% dòng có số nêu rõ, so với 52,3%.
4. **Điểm yếu 1: kênh bộ ba dàn trải.**
   - Ở countonly, bộ ba đứng đầu E trong 44% câu, nhưng bỏ bộ ba đó chỉ làm mất hạng 1 ở 20% câu, so với 47% khi node đứng
     đầu.
   - Giữ riêng bộ ba đó gần như không hơn giữ một bộ ba bất kỳ: hạng 3,70 so với 4,02.
   - Top-3 bộ ba chỉ chiếm 39% term dương của kênh bộ ba, so với 68% của top-3 node ở kênh vật thể.
   - Một dòng bộ ba đứng đầu trong demo vì vậy dễ gây hiểu nhầm: nó nặng ký trong tổng điểm, nhưng không phải nguyên nhân
     riêng lẻ. Demo nên ghi chú điều này, hoặc xếp dòng node lên trước khi trình bày.
   - Mức dàn trải phụ thuộc trọng số c mà mỗi seed học được (0,23–0,35). Đây cũng là lý do Fid của countonly dao động giữa
     các seed nhiều hơn `main_r3`.
5. **Điểm yếu 2: attention GAT không phải lời giải thích.**
   - Bỏ 9 cạnh vào có alpha cao nhất của 3 node đứng đầu E làm điểm đổi y như bỏ 9 cạnh ngẫu nhiên.
   - Tỉ lệ mất hạng 1 chỉ cao hơn chút ít: 6,6% so với 3,8% (countonly), 5,7% so với 5,0% (`main_r3`).
   - Đồ thị con theo attention có thể minh hoạ node lấy tin từ đâu, nhưng không nên trình bày như nguyên nhân của thứ hạng.
   - Cạnh có ích ở tổng thể (ablation bỏ cạnh: −2,39 R@1), nhưng tác dụng trải trên nhiều cạnh chứ không nằm ở vài cạnh.
6. **Điểm yếu 3: E không kể các phần kéo điểm xuống.**
   - Chỉ giữ top-5 cho điểm cao hơn ảnh gốc: 117% hiệu ứng với countonly. Lý do là điểm kênh là trung bình softmax trên các
     phần của ảnh, nên các phần khớp kém làm loãng điểm.
   - Term cộng đúng ra điểm, nhưng không cho biết bỏ một phần sẽ đổi điểm bao nhiêu.
7. **Ảnh hạng 1 sai:** E vẫn trung thực như với ảnh đúng. Ảnh sai mất hạng 1 dễ hơn vì cách biệt với hạng 2 nhỏ hơn
   khoảng 4 lần. Với countonly, bộ ba đứng đầu E ở 50% câu sai, so với 37% câu đúng. Đây là tương quan, chưa phải nguyên
   nhân.

**Giới hạn:**

- Validity ngữ nghĩa dùng tiêu chí nhãn khắt khe, nên bỏ sót các cặp như "dining table" ~ "table".
- Bỏ node cũng bỏ các bộ ba chạm vào nó, theo đúng đặc tả. Vì vậy hiệu ứng của một node đã gồm phần của các bộ ba đó.
- Với `main_r3`, mốc "rỗng" không trung tính dưới z-score. Tỉ lệ hiệu ứng của nó chỉ nên đọc cùng hạng.
- Chỉ đo trên ảnh hạng 1, chỉ trên val.

## Phụ lục: bảng đầy đủ theo nhóm (in bởi `scripts/summarize_xai_metrics.py`)

### countonly_factual_r3 (R@1 theo seed: 49,15, 48,85, 48,74)

| Nhóm | Số câu | Fid+ k | Δ điểm top | Δ điểm ngẫu nhiên | Hạng top | Hạng ngẫu nhiên | Mất hạng 1 top (%) | Mất hạng 1 ngẫu nhiên (%) | Hiệu ứng bị bỏ, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên: tỉ số tổng / trung vị (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10633 | 1 | −0,0095 ± 0,0019 | −0,0006 ± 0,0002 | 2,30 ± 0,22 | 1,14 ± 0,03 | 34,3 ± 4,8 | 4,4 ± 0,5 | 30,6 / 18,9 | 2,0 / 0,0 |
| all | 10633 | 3 | −0,0196 ± 0,0033 | −0,0018 ± 0,0004 | 4,73 ± 0,57 | 1,36 ± 0,03 | 58,0 ± 4,1 | 9,4 ± 0,9 | 63,4 / 56,6 | 5,9 / 1,0 |
| all | 10633 | 5 | −0,0248 ± 0,0035 | −0,0027 ± 0,0004 | 6,39 ± 0,58 | 1,54 ± 0,02 | 68,0 ± 3,0 | 12,9 ± 0,7 | 80,4 / 77,8 | 8,8 / 2,2 |
| correct | 5201 | 1 | −0,0114 ± 0,0019 | −0,0007 ± 0,0002 | 1,70 ± 0,10 | 1,06 ± 0,01 | 23,9 ± 2,8 | 2,5 ± 0,3 | 30,1 / 20,9 | 1,9 / 0,1 |
| correct | 5201 | 3 | −0,0233 ± 0,0033 | −0,0022 ± 0,0005 | 3,32 ± 0,28 | 1,17 ± 0,02 | 45,3 ± 3,7 | 6,1 ± 0,6 | 61,6 / 57,3 | 6,0 / 1,6 |
| correct | 5201 | 5 | −0,0293 ± 0,0034 | −0,0034 ± 0,0006 | 4,53 ± 0,26 | 1,27 ± 0,02 | 55,4 ± 2,7 | 8,7 ± 0,6 | 77,4 / 76,3 | 8,9 / 3,0 |
| wrong | 5432 | 1 | −0,0076 ± 0,0019 | −0,0005 ± 0,0002 | 2,88 ± 0,33 | 1,23 ± 0,04 | 44,3 ± 6,5 | 6,2 ± 0,6 | 31,3 / 17,0 | 2,1 / −0,1 |
| wrong | 5432 | 3 | −0,0161 ± 0,0033 | −0,0014 ± 0,0003 | 6,08 ± 0,85 | 1,54 ± 0,05 | 70,1 ± 4,4 | 12,5 ± 1,1 | 66,2 / 55,5 | 5,7 / 0,6 |
| wrong | 5432 | 5 | −0,0206 ± 0,0036 | −0,0021 ± 0,0003 | 8,17 ± 0,87 | 1,80 ± 0,02 | 80,0 ± 3,2 | 17,0 ± 0,7 | 85,0 / 79,8 | 8,7 / 1,3 |

| Nhóm | Số câu | Fid− k | Δ điểm top | Δ điểm ngẫu nhiên | Hạng top | Hạng ngẫu nhiên | Mất hạng 1 top (%) | Mất hạng 1 ngẫu nhiên (%) | Hiệu ứng giữ lại, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên: tỉ số tổng / trung vị (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10633 | 1 | −0,0167 ± 0,0040 | −0,0319 ± 0,0077 | 4,18 ± 0,83 | 10,13 ± 2,26 | 51,0 ± 6,3 | 69,2 ± 7,1 | 45,7 / 47,4 | −2,9 / −7,4 |
| all | 10633 | 3 | 0,0000 ± 0,0026 | −0,0272 ± 0,0049 | 2,22 ± 0,74 | 8,13 ± 1,40 | 21,7 ± 6,7 | 66,3 ± 5,0 | 99,7 / 106,4 | 12,2 / 9,9 |
| all | 10633 | 5 | 0,0052 ± 0,0027 | −0,0235 ± 0,0031 | 1,86 ± 0,54 | 6,93 ± 0,96 | 14,2 ± 5,7 | 62,1 ± 3,6 | 116,6 / 122,6 | 24,0 / 21,8 |
| correct | 5201 | 1 | −0,0229 ± 0,0047 | −0,0392 ± 0,0078 | 3,24 ± 0,57 | 8,11 ± 1,60 | 44,3 ± 6,6 | 64,2 ± 7,3 | 39,6 / 41,6 | −3,4 / −9,0 |
| correct | 5201 | 3 | −0,0010 ± 0,0020 | −0,0329 ± 0,0049 | 1,56 ± 0,31 | 6,11 ± 0,92 | 14,4 ± 4,4 | 58,6 ± 4,9 | 97,1 / 103,7 | 13,2 / 11,3 |
| correct | 5201 | 5 | 0,0058 ± 0,0023 | −0,0278 ± 0,0030 | 1,32 ± 0,20 | 4,93 ± 0,57 | 8,0 ± 3,5 | 52,6 ± 3,4 | 115,1 / 120,7 | 26,3 / 24,5 |
| wrong | 5432 | 1 | −0,0109 ± 0,0033 | −0,0249 ± 0,0077 | 5,08 ± 1,13 | 12,06 ± 2,87 | 57,5 ± 6,0 | 74,0 ± 6,9 | 55,0 / 55,9 | −2,0 / −5,9 |
| wrong | 5432 | 3 | 0,0010 ± 0,0030 | −0,0217 ± 0,0049 | 2,85 ± 1,15 | 10,06 ± 1,85 | 28,8 ± 8,9 | 73,6 ± 5,1 | 103,7 / 110,0 | 10,7 / 7,6 |
| wrong | 5432 | 5 | 0,0047 ± 0,0031 | −0,0193 ± 0,0033 | 2,37 ± 0,88 | 8,84 ± 1,34 | 20,1 ± 7,7 | 71,2 ± 3,8 | 118,7 / 125,0 | 20,5 / 17,8 |

| Nhóm | Top-1 của E | Số câu (TB/seed) | Bỏ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) | Bỏ top-1: Δ điểm | Ngẫu nhiên | Chỉ giữ top-1: hạng | Ngẫu nhiên | Chỉ giữ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | node | 5982 | 46,5 ± 0,8 | 4,7 ± 0,1 | −0,0144 ± 0,0006 | −0,0008 ± 0,0000 | 4,21 ± 0,73 | 14,48 ± 1,71 | 52,7 ± 4,6 | 83,8 ± 2,3 |
| all | triple | 4618 | 19,7 ± 3,2 | 4,2 ± 1,0 | −0,0035 ± 0,0013 | −0,0004 ± 0,0002 | 3,70 ± 1,80 | 4,02 ± 1,95 | 46,4 ± 12,5 | 48,8 ± 11,9 |
| correct | node | 3247 | 32,4 ± 0,5 | 2,9 ± 0,2 | −0,0161 ± 0,0006 | −0,0010 ± 0,0001 | 3,45 ± 0,55 | 11,09 ± 1,23 | 46,1 ± 4,2 | 76,9 ± 2,4 |
| correct | triple | 1937 | 10,1 ± 1,1 | 1,8 ± 0,3 | −0,0041 ± 0,0017 | −0,0003 ± 0,0002 | 2,61 ± 1,12 | 2,85 ± 1,29 | 39,5 ± 12,8 | 42,1 ± 12,4 |
| wrong | node | 2735 | 63,6 ± 3,3 | 6,8 ± 0,2 | −0,0124 ± 0,0005 | −0,0006 ± 0,0000 | 5,11 ± 0,82 | 18,58 ± 1,96 | 60,7 ± 4,0 | 92,3 ± 1,3 |
| wrong | triple | 2681 | 26,7 ± 4,8 | 5,9 ± 1,6 | −0,0031 ± 0,0011 | −0,0004 ± 0,0003 | 4,49 ± 2,30 | 4,86 ± 2,43 | 51,4 ± 12,8 | 53,6 ± 12,0 |

| Nhóm | Dòng object | Khớp nhãn (any, %) | alias | WordNet | CLIP ≥ 0,955 | Cùng danh từ chính | any hoặc danh từ chính | Dòng có số | Khớp nhóm số lượng (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 29687 | 70,0 ± 1,0 | 57,0 ± 1,0 | 54,6 ± 0,7 | 63,7 ± 1,0 | 62,1 ± 1,1 | 71,7 ± 1,0 | 5857 | 66,7 ± 0,8 |
| correct | 15191 | 71,8 ± 0,8 | 59,2 ± 0,8 | 56,8 ± 0,7 | 65,6 ± 0,8 | 64,5 ± 0,8 | 73,8 ± 0,8 | 3078 | 66,1 ± 0,9 |
| wrong | 14496 | 68,1 ± 1,3 | 54,7 ± 1,3 | 52,4 ± 0,8 | 61,6 ± 1,3 | 59,7 ± 1,4 | 69,5 ± 1,3 | 2779 | 67,3 ± 0,7 |

| Nhóm | Biến thể | Câu có biên dương | Số dòng | Dòng cho 80% | Dòng cho 90% | % số dòng (80 / 90) | Dòng đồ thị (80 / 90) | % phần của đồ thị ảnh (80 / 90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | graph | 10539 | 4,57 ± 0,00 | 2,52 ± 0,04 | 3,06 ± 0,05 | 58,9 / 70,7 | 2,52 / 3,06 | 8,1 / 9,7 |
| all | with_clip | 10633 | 5,56 ± 0,00 | 2,68 ± 0,04 | 3,42 ± 0,03 | 50,6 / 64,1 | 1,71 / 2,44 | 5,5 / 7,8 |
| correct | graph | 5161 | 4,81 ± 0,00 | 2,66 ± 0,05 | 3,24 ± 0,07 | 58,7 / 71,0 | 2,66 / 3,24 | 8,1 / 9,9 |
| correct | with_clip | 5201 | 5,81 ± 0,01 | 2,75 ± 0,04 | 3,55 ± 0,05 | 49,5 / 63,6 | 1,77 / 2,57 | 5,3 / 7,8 |
| wrong | graph | 5378 | 4,33 ± 0,01 | 2,39 ± 0,03 | 2,88 ± 0,03 | 59,1 / 70,4 | 2,39 / 2,88 | 8,0 / 9,6 |
| wrong | with_clip | 5432 | 5,33 ± 0,01 | 2,61 ± 0,03 | 3,30 ± 0,03 | 51,6 / 64,6 | 1,65 / 2,32 | 5,6 / 7,8 |

| Nhóm | Tập câu | Số câu | Cạnh bỏ / câu | Cạnh vào / câu | Loại cạnh (có nhãn / kNN / tổng, %) | Δ điểm top | Δ điểm ngẫu nhiên | Hạng top | Hạng ngẫu nhiên | Mất hạng 1 top (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | có cạnh vào | 10600 | 8,97 ± 0,00 | — | 17 / 50 / 33 | 0,0012 ± 0,0004 | 0,0012 ± 0,0002 | 1,18 ± 0,05 | 1,11 ± 0,02 | 6,9 ± 1,8 | 4,1 ± 0,9 |
| all | có lựa chọn | 10481 | 9,00 ± 0,00 | 21,11 ± 0,04 |  | 0,0013 ± 0,0005 | 0,0013 ± 0,0002 | 1,14 ± 0,04 | 1,07 ± 0,01 | 6,6 ± 1,8 | 3,8 ± 0,9 |
| correct | có cạnh vào | 5184 | 8,98 ± 0,00 | — | 17 / 49 / 33 | 0,0008 ± 0,0004 | 0,0011 ± 0,0001 | 1,05 ± 0,01 | 1,03 ± 0,00 | 3,4 ± 0,7 | 1,8 ± 0,4 |
| correct | có lựa chọn | 5151 | 9,00 ± 0,00 | 21,23 ± 0,03 |  | 0,0008 ± 0,0004 | 0,0011 ± 0,0001 | 1,05 ± 0,01 | 1,03 ± 0,01 | 3,4 ± 0,8 | 1,8 ± 0,4 |
| wrong | có cạnh vào | 5416 | 8,96 ± 0,00 | — | 16 / 50 / 33 | 0,0015 ± 0,0005 | 0,0013 ± 0,0002 | 1,30 ± 0,09 | 1,19 ± 0,03 | 10,3 ± 2,8 | 6,3 ± 1,4 |
| wrong | có lựa chọn | 5330 | 9,00 ± 0,00 | 21,00 ± 0,05 |  | 0,0017 ± 0,0005 | 0,0015 ± 0,0002 | 1,23 ± 0,08 | 1,11 ± 0,02 | 9,8 ± 2,8 | 5,7 ± 1,3 |

| Nhóm | Node / ảnh | Bộ ba / ảnh | Bộ ba trong top-1 / top-3 / top-5 của E (%) | Top-3 node chiếm (% term dương của kênh) | Top-3 bộ ba chiếm (%) | Cách biệt hạng 1 – hạng 2 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 25,3 | 18,1 | 44 / 50 / 56 | 68,0 ± 0,5 | 38,7 ± 1,3 | 0,0168 ± 0,0013 |
| correct | 25,3 | 18,3 | 37 / 44 / 51 | 68,9 ± 0,5 | 40,2 ± 0,9 | 0,0271 ± 0,0019 |
| wrong | 25,2 | 17,9 | 50 / 56 / 61 | 67,1 ± 0,6 | 37,2 ± 1,7 | 0,0070 ± 0,0007 |


### main_r3 (R@1 theo seed: 46,67, 46,56, 46,21)

| Nhóm | Số câu | Fid+ k | Δ điểm top | Δ điểm ngẫu nhiên | Hạng top | Hạng ngẫu nhiên | Mất hạng 1 top (%) | Mất hạng 1 ngẫu nhiên (%) | Hiệu ứng bị bỏ, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên: tỉ số tổng / trung vị (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10633 | 1 | −0,3868 ± 0,0042 | −0,0305 ± 0,0007 | 2,76 ± 0,06 | 1,18 ± 0,00 | 40,4 ± 0,6 | 5,0 ± 0,0 | 44,4 / 34,8 | 3,5 / 0,5 |
| all | 10633 | 3 | −0,8341 ± 0,0033 | −0,0910 ± 0,0007 | 6,46 ± 0,02 | 1,52 ± 0,01 | 65,7 ± 0,3 | 12,0 ± 0,1 | 96,0 / 92,7 | 10,6 / 3,8 |
| all | 10633 | 5 | −1,0954 ± 0,0033 | −0,1347 ± 0,0016 | 9,22 ± 0,04 | 1,78 ± 0,01 | 75,1 ± 0,4 | 16,6 ± 0,1 | 126,0 / 126,5 | 15,6 / 7,3 |
| correct | 4942 | 1 | −0,4053 ± 0,0024 | −0,0340 ± 0,0009 | 1,85 ± 0,02 | 1,08 ± 0,01 | 26,1 ± 0,3 | 2,9 ± 0,1 | 40,5 / 34,0 | 3,4 / 0,6 |
| correct | 4942 | 3 | −0,8603 ± 0,0038 | −0,1013 ± 0,0012 | 4,01 ± 0,09 | 1,27 ± 0,01 | 49,2 ± 0,4 | 7,8 ± 0,2 | 86,4 / 84,3 | 10,3 / 3,9 |
| correct | 4942 | 5 | −1,1199 ± 0,0059 | −0,1479 ± 0,0025 | 5,74 ± 0,11 | 1,41 ± 0,02 | 60,0 ± 0,7 | 10,8 ± 0,2 | 112,4 / 114,0 | 14,9 / 7,3 |
| wrong | 5691 | 1 | −0,3706 ± 0,0066 | −0,0275 ± 0,0007 | 3,55 ± 0,11 | 1,25 ± 0,01 | 52,9 ± 0,7 | 6,8 ± 0,0 | 48,8 / 35,7 | 3,6 / 0,5 |
| wrong | 5691 | 3 | −0,8113 ± 0,0088 | −0,0821 ± 0,0004 | 8,58 ± 0,09 | 1,74 ± 0,00 | 79,9 ± 0,5 | 15,7 ± 0,1 | 107,0 / 101,8 | 10,9 / 3,7 |
| wrong | 5691 | 5 | −1,0740 ± 0,0086 | −0,1233 ± 0,0011 | 12,24 ± 0,12 | 2,10 ± 0,01 | 88,3 ± 0,4 | 21,6 ± 0,1 | 141,5 / 140,4 | 16,3 / 7,3 |

| Nhóm | Số câu | Fid− k | Δ điểm top | Δ điểm ngẫu nhiên | Hạng top | Hạng ngẫu nhiên | Mất hạng 1 top (%) | Mất hạng 1 ngẫu nhiên (%) | Hiệu ứng giữ lại, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên: tỉ số tổng / trung vị (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10633 | 1 | −0,8382 ± 0,0343 | −1,8194 ± 0,0216 | 6,37 ± 0,34 | 18,91 ± 0,34 | 64,8 ± 1,9 | 87,2 ± 0,6 | 3,0 / 6,3 | −109,4 / −109,6 |
| all | 10633 | 3 | −0,1239 ± 0,0153 | −1,2197 ± 0,0197 | 2,07 ± 0,05 | 11,09 ± 0,23 | 22,7 ± 0,9 | 75,2 ± 0,8 | 85,4 / 95,5 | −40,4 / −41,3 |
| all | 10633 | 5 | 0,0719 ± 0,0113 | −0,9512 ± 0,0180 | 1,45 ± 0,01 | 8,14 ± 0,19 | 11,2 ± 0,7 | 66,2 ± 0,9 | 107,9 / 110,7 | −9,5 / −9,6 |
| correct | 4942 | 1 | −0,9037 ± 0,0313 | −1,8529 ± 0,0178 | 4,13 ± 0,16 | 13,53 ± 0,16 | 51,6 ± 1,2 | 78,3 ± 0,5 | 9,0 / 10,4 | −86,0 / −87,5 |
| correct | 4942 | 3 | −0,1464 ± 0,0170 | −1,2508 ± 0,0184 | 1,52 ± 0,03 | 7,28 ± 0,15 | 14,3 ± 0,5 | 62,5 ± 0,5 | 85,1 / 93,9 | −25,5 / −27,9 |
| correct | 4942 | 5 | 0,0579 ± 0,0119 | −0,9808 ± 0,0170 | 1,19 ± 0,02 | 5,24 ± 0,16 | 6,5 ± 0,4 | 52,5 ± 0,6 | 105,7 / 107,9 | 1,6 / 0,0 |
| wrong | 5691 | 1 | −0,7814 ± 0,0378 | −1,7903 ± 0,0258 | 8,32 ± 0,48 | 23,58 ± 0,49 | 76,3 ± 2,3 | 94,8 ± 0,6 | −3,7 / 2,0 | −136,1 / −136,8 |
| wrong | 5691 | 3 | −0,1044 ± 0,0140 | −1,1927 ± 0,0215 | 2,55 ± 0,08 | 14,39 ± 0,32 | 29,9 ± 1,3 | 86,1 ± 1,0 | 85,7 / 97,6 | −57,4 / −56,8 |
| wrong | 5691 | 5 | 0,0840 ± 0,0113 | −0,9256 ± 0,0210 | 1,67 ± 0,01 | 10,66 ± 0,28 | 15,3 ± 0,9 | 78,1 ± 1,1 | 110,6 / 114,2 | −22,2 / −20,6 |

| Nhóm | Top-1 của E | Số câu (TB/seed) | Bỏ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) | Bỏ top-1: Δ điểm | Ngẫu nhiên | Chỉ giữ top-1: hạng | Ngẫu nhiên | Chỉ giữ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | node | 10466 | 40,8 ± 0,6 | 4,9 ± 0,0 | −0,3916 ± 0,0038 | −0,0304 ± 0,0007 | 6,45 ± 0,33 | 19,17 ± 0,30 | 65,5 ± 1,8 | 88,0 ± 0,5 |
| all | triple | 124 | 18,6 ± 6,1 | 12,0 ± 2,7 | −0,1104 ± 0,0038 | −0,0523 ± 0,0091 | 1,83 ± 0,15 | 2,59 ± 0,31 | 31,2 ± 4,8 | 43,6 ± 5,3 |
| correct | node | 4863 | 26,4 ± 0,4 | 2,9 ± 0,1 | −0,4109 ± 0,0028 | −0,0341 ± 0,0009 | 4,17 ± 0,16 | 13,72 ± 0,17 | 52,2 ± 1,2 | 79,3 ± 0,5 |
| correct | triple | 55 | 8,6 ± 4,0 | 5,9 ± 2,9 | −0,0911 ± 0,0114 | −0,0404 ± 0,0094 | 1,71 ± 0,10 | 1,90 ± 0,12 | 22,6 ± 3,1 | 29,2 ± 2,9 |
| wrong | node | 5604 | 53,3 ± 0,7 | 6,7 ± 0,1 | −0,3748 ± 0,0056 | −0,0272 ± 0,0007 | 8,42 ± 0,46 | 23,90 ± 0,41 | 77,0 ± 2,1 | 95,6 ± 0,5 |
| wrong | triple | 69 | 27,0 ± 5,2 | 17,4 ± 1,4 | −0,1296 ± 0,0136 | −0,0657 ± 0,0242 | 1,93 ± 0,27 | 3,12 ± 0,47 | 39,1 ± 7,8 | 55,9 ± 3,8 |

| Nhóm | Dòng object | Khớp nhãn (any, %) | alias | WordNet | CLIP ≥ 0,955 | Cùng danh từ chính | any hoặc danh từ chính | Dòng có số | Khớp nhóm số lượng (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 32382 | 63,8 ± 0,2 | 51,6 ± 0,0 | 49,5 ± 0,1 | 57,4 ± 0,1 | 58,1 ± 0,0 | 65,6 ± 0,2 | 6817 | 52,3 ± 0,6 |
| correct | 15515 | 66,4 ± 0,3 | 54,0 ± 0,1 | 52,4 ± 0,2 | 59,7 ± 0,1 | 61,0 ± 0,0 | 68,4 ± 0,2 | 3293 | 54,8 ± 0,7 |
| wrong | 16867 | 61,4 ± 0,3 | 49,3 ± 0,0 | 46,7 ± 0,2 | 55,2 ± 0,1 | 55,5 ± 0,0 | 63,1 ± 0,3 | 3523 | 49,9 ± 0,7 |

| Nhóm | Biến thể | Câu có biên dương | Số dòng | Dòng cho 80% | Dòng cho 90% | % số dòng (80 / 90) | Dòng đồ thị (80 / 90) | % phần của đồ thị ảnh (80 / 90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | graph | 10435 | 4,71 ± 0,00 | 1,95 ± 0,01 | 2,28 ± 0,01 | 46,5 / 53,6 | 1,95 / 2,28 | 5,7 / 6,6 |
| correct | graph | 4863 | 4,90 ± 0,01 | 2,02 ± 0,01 | 2,39 ± 0,01 | 45,9 / 53,4 | 2,02 / 2,39 | 5,9 / 6,9 |
| wrong | graph | 5572 | 4,54 ± 0,01 | 1,88 ± 0,01 | 2,19 ± 0,02 | 46,9 / 53,8 | 1,88 / 2,19 | 5,5 / 6,4 |

| Nhóm | Tập câu | Số câu | Cạnh bỏ / câu | Cạnh vào / câu | Loại cạnh (có nhãn / kNN / tổng, %) | Δ điểm top | Δ điểm ngẫu nhiên | Hạng top | Hạng ngẫu nhiên | Mất hạng 1 top (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | có cạnh vào | 9560 | 3,40 ± 0,01 | — | 100 / 0 / 0 | −0,0209 ± 0,0014 | −0,0199 ± 0,0012 | 1,10 ± 0,00 | 1,10 ± 0,00 | 6,3 ± 0,0 | 6,1 ± 0,1 |
| all | có lựa chọn | 2801 | 5,25 ± 0,03 | 9,50 ± 0,03 |  | −0,0163 ± 0,0022 | −0,0129 ± 0,0018 | 1,08 ± 0,02 | 1,07 ± 0,01 | 5,7 ± 0,6 | 5,0 ± 0,5 |
| correct | có cạnh vào | 4452 | 3,52 ± 0,02 | — | 100 / 0 / 0 | −0,0185 ± 0,0011 | −0,0173 ± 0,0009 | 1,04 ± 0,00 | 1,04 ± 0,00 | 2,7 ± 0,1 | 2,6 ± 0,2 |
| correct | có lựa chọn | 1418 | 5,34 ± 0,03 | 9,52 ± 0,03 |  | −0,0163 ± 0,0014 | −0,0127 ± 0,0004 | 1,04 ± 0,00 | 1,03 ± 0,00 | 2,7 ± 0,4 | 2,4 ± 0,2 |
| wrong | có cạnh vào | 5108 | 3,30 ± 0,01 | — | 100 / 0 / 0 | −0,0229 ± 0,0017 | −0,0221 ± 0,0016 | 1,16 ± 0,01 | 1,16 ± 0,00 | 9,5 ± 0,2 | 9,3 ± 0,2 |
| wrong | có lựa chọn | 1383 | 5,16 ± 0,04 | 9,47 ± 0,07 |  | −0,0162 ± 0,0031 | −0,0131 ± 0,0032 | 1,13 ± 0,03 | 1,11 ± 0,02 | 8,7 ± 1,1 | 7,8 ± 1,0 |

| Nhóm | Node / ảnh | Bộ ba / ảnh | Bộ ba trong top-1 / top-3 / top-5 của E (%) | Top-3 node chiếm (% term dương của kênh) | Top-3 bộ ba chiếm (%) | Cách biệt hạng 1 – hạng 2 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 27,4 | 18,7 | 1 / 5 / 11 | 63,1 ± 0,9 | 22,8 ± 0,2 | 0,6281 ± 0,0027 |
| correct | 27,1 | 18,7 | 1 / 4 / 11 | 64,0 ± 0,8 | 22,6 ± 0,2 | 1,0043 ± 0,0051 |
| wrong | 27,6 | 18,6 | 1 / 5 / 12 | 62,3 ± 1,1 | 23,0 ± 0,2 | 0,3015 ± 0,0038 |
