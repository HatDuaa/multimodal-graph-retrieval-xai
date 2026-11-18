# Đo chất lượng giải thích cấp 3, phiên bản 2: fidelity, validity, sparsity và so sánh 9 cách giải thích (val)

Bản này làm lại `experiments/xai-metrics/` (v1, PR #36) sau khi soát lại cách đo, và thêm phần so sánh với các cách giải thích khác.
Chỉ dùng val, không train, không chấm test.
v1 và kết quả của nó giữ nguyên; các điểm cần đính chính ở v1 được ghi ở đầu `experiments/xai-metrics/REPORT.md`.

- **Mô hình đã chạy:** `main_r3` (cấp 1, fusion z-score), 3 seed, `checkpoint_best.pt`.
- **Chưa chạy:** `countonly_factual_r3`.
  Máy này chưa có dữ liệu count_v1 (`data/features/factual_v1/`, `data/features/merge_v1/`, `data/graphs/merge_v1/`, `data/graphs/count_v1/`, `data/processed/group_v1/`) và checkpoint countonly.
  Khi có, chạy `bash experiments/xai-metrics-v2/run.sh countonly_factual_r3`; kết quả được gộp vào `metrics.json` và các bảng ở phụ lục tự có thêm mô hình này.
- **Câu:** toàn bộ 10 633 câu val cho fidelity, validity, sparsity, đồ thị con.
  Phần so sánh các cách giải thích dùng một mẫu cố định 1 000 câu (rng 2026), trong đó giữ các câu có hiệu ứng đồ thị dương (972–980 câu mỗi seed).
  GNNExplainer và Shapley chạy trên 200 câu đầu của mẫu đó (192–197 câu sau khi lọc).
- **Thời gian:** chạy trên CPU (MacBook), 989 / 1 096 / 871 giây cho seed 0 / 1 / 2.
- **Kiểm tra tái lập:**
  - R@1 tính lại trong lúc đo trùng `metrics.json` của từng checkpoint (script kiểm bằng assert): 46,67 / 46,56 / 46,21;
  - ảnh hạng 1, chấm lại khi không sửa gì, vẫn hạng 1 ở mọi câu; sai số điểm tối đa 2,0e−5 (float32 sau z-score);
  - Integrated Gradients thoả completeness với sai số trung bình 1,1% hiệu ứng; Shapley thoả efficiency với sai số tối đa 1,3e−7.
- **Dữ liệu:** dữ liệu cấp 1 và checkpoint `main_r3` lấy từ Drive của nhóm, đối chiếu với `MANIFEST`.
  `data/processed/geometry/objects_val.json` được dựng lại bằng `scripts/improve/build_boxes.py --split val --synsets`.
- **Script:**
  - `src/explain/xai_metrics.py`: sửa đồ thị một ảnh và chấm lại; thêm `message_drop` (cắt đường truyền tin GAT của một cạnh mà vẫn giữ bộ ba) và đạo hàm theo đầu vào;
  - `scripts/xai_metrics.py`: đo, ghi `metrics.json` và `cases.json` (13 câu bằng chứng, seed 0); docstring ghi đầy đủ định nghĩa;
  - `scripts/summarize_xai_metrics.py`: mean ± std qua seed, ghi `summary.json` và in các bảng phụ lục (`logs/tables.md`);
  - `run.sh`: chạy lại cả hai bước; log ở `logs/run.log`.
  - Bản v1 giữ ở `scripts/xai_metrics_v1.py` và `scripts/summarize_xai_metrics_v1.py`.

## v2 khác v1 ở đâu

| Chỗ | v1 | v2 |
|---|---|---|
| Thang của E với `main_r3` | So term thô của hai kênh, trong khi fusion z-score cân mỗi kênh theo trọng số / độ phân tán trên 50 ứng viên | Mỗi term nhân với đạo hàm chính xác của điểm cuối theo kênh đó (autograd), nên hai kênh cùng thang điểm cuối; với fusion raw hệ số là 1, E giữ như v1 |
| Mốc so sánh | Lượt chấm thường | Ảnh không sửa, chấm lại bằng đúng đường chấm sau khi sửa (noop); câu có noop khác hạng 1 bị loại (0 câu) |
| E rỗng | Tính như "không đổi" cho cả E và ngẫu nhiên | Báo riêng, không tính vào trung bình fidelity |
| Sparsity | Đếm dòng demo rồi chia cho số phần của đồ thị ảnh (hai đơn vị khác nhau) | Đo trên chính E, theo phần của ảnh: số phần cho 80% / 90% khối term, và k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% / 90% hiệu ứng; số dòng demo vẫn báo nhưng không chia |
| Validity ngữ nghĩa | `dedup.same_type`: mọi người coi như cùng loại, chấp nhận hypernym hai chiều ("woman" ~ "man", "object" ~ "table") | strict: synset ảnh bằng hoặc là hyponym của sense danh từ đầu của từ trong câu, người chỉ khớp với từ chung (person, people...); tiêu chí v1 giữ làm "lenient"; thêm kiểm dòng bộ ba |
| Khớp nhóm số lượng | Chấm cả `main_r3` trên đồ thị cấp 1 chưa gộp trùng | Chỉ chấm mô hình có bảng số lượng (countonly) |
| Đồ thị con | Bỏ cạnh có nhãn là bỏ luôn bộ ba; ngẫu nhiên bốc từ cạnh gộp của 3 node | Dạng chính "messages": chỉ cắt đường truyền tin GAT, giữ bộ ba; dạng v1 giữ là "edges"; ngẫu nhiên bốc theo từng node, cùng số cạnh |
| Mới | | k_flip, hạng ảnh vàng sau khi bỏ top-k, tính cộng của E, so sánh 9 cách giải thích, Fid_α |

## Định nghĩa

**Giải thích E(q, ảnh)** xếp các phần của ảnh hạng 1, gồm node vật thể và bộ ba có nhãn, theo term đã đưa về thang điểm cuối:

scaled_term_j = trọng số kênh × Σ_i w_i · a_ij · sim_ij × ∂(điểm cuối)/∂(điểm kênh)

**Sửa đồ thị** giữ như v1: bỏ node là bỏ node cùng mọi cạnh chạm vào nó; bỏ bộ ba là bỏ cạnh có nhãn đó; "chỉ giữ top-k" giữ các phần đó cùng node tổng và node carrier của bộ ba; ảnh được mã hoá lại bằng cả mô hình, kể cả GAT.

**Fidelity** (k = 1, 3, 5), so với trung bình 10 lần chọn ngẫu nhiên cùng số node và bộ ba:

- **Fid+**: bỏ top-k của E; báo tỉ lệ mất hạng 1, hạng mới, Δ điểm và phần hiệu ứng bị bỏ = (gốc − sau) / (gốc − rỗng).
- **Fid−**: chỉ giữ top-k; báo cùng các số đó với phần hiệu ứng giữ lại = (sau − rỗng) / (gốc − rỗng).
- **Fid_α** (Zheng và cộng sự, ICLR 2024): với E là k = 5 hoặc 10 phần đầu, bỏ mỗi phần trong E (plus) hoặc ngoài E (minus) với xác suất α = 0,1 hoặc 0,5, 10 lần bốc dùng chung cho mọi cách giải thích.
  Đồ thị sau khi sửa vẫn gần đồ thị gốc, nên tránh được việc đánh giá trên đồ thị lệch phân phối.
  Giải thích trung thực thì plus làm mất nhiều điểm hơn minus.

**Đường cong deletion / insertion** (k = 0..10): bỏ dần (hoặc chỉ giữ dần) k phần đầu theo thứ tự của từng cách giải thích.
AUC là trung bình phần hiệu ứng còn lại qua k = 1..10; deletion thấp là tốt, insertion cao là tốt.
Giá trị có thể âm hoặc lớn hơn 1, vì bỏ hoặc giữ một số phần có thể làm điểm vượt khỏi khoảng [rỗng, gốc].

**Tính cộng (additivity):** E là phép phân rã điểm, nên tổng term của top-k là "dự đoán" điểm mất khi bỏ top-k.
So với điểm mất "đo được": tỉ số tổng qua câu, tỉ số trung vị, Spearman giữa hai đại lượng, và sai lệch tuyệt đối chia cho hiệu ứng.

**Phản thực tế:** k_flip là k nhỏ nhất (≤ 10) mà bỏ top-k làm ảnh mất hạng 1.
Với ảnh hạng 1 sai, ghi thêm hạng của ảnh vàng sau khi bỏ top-k.

**Validity ngữ nghĩa:** với mỗi dòng vật thể của E, node ảnh có attention cao nhất có cùng loại với node câu không (strict, lenient, alias, CLIP ≥ 0,955, cùng danh từ chính).
Với dòng bộ ba, chủ thể và đối tượng có khớp strict không, vị từ có khớp sau alias VG không.

**Sparsity:** số phần đầu của E để đạt 80% / 90% khối term dương, và k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% / 90% hiệu ứng.

**Đồ thị con:** với 3 node đầu của E, lấy các cạnh vào có alpha GAT lớn nhất (trung bình 2 lớp × 4 head), cắt đường truyền tin của chúng, so với cắt cùng số cạnh vào ngẫu nhiên của chính từng node.

**9 cách giải thích**, đều xếp cùng các phần của ảnh và được đo cùng một cách:

| Cách | Loại | Chi phí mỗi câu |
|---|---|---|
| model (E) | Nội tại, phân rã điểm | 0 lượt chấm thêm |
| random | Mốc dưới | 0 |
| lowest | E đảo ngược, mốc dưới nữa | 0 |
| clip | Cosine CLIP đóng băng giữa câu và từng phần, không dùng mô hình | 0 |
| gradient | Gradient × input của điểm cuối theo vector đầu vào của từng phần | 1 lượt lan truyền ngược |
| integrated_gradients | Phân rã hậu nghiệm (Sundararajan và cộng sự, 2017) từ ảnh trung tính (mỗi phần là vector trung bình của loại đó), 20 bước | 20 lượt lan truyền ngược |
| occlusion | Bỏ từng phần một (leave-one-out), điểm mất là độ quan trọng | 1 lượt chấm mỗi phần (TB 46) |
| gnnexplainer | Mặt nạ mềm trên vector đầu vào, nội suy về vector trung tính, học để giữ điểm với phạt kích thước và entropy (Ying và cộng sự, 2019) | 100 bước tối ưu |
| shapley | Giá trị Shapley lấy mẫu hoán vị (16 thứ tự, antithetic), hàm giá trị là biên với hạng 2 khi chỉ giữ một tập phần (ý tưởng của RankingSHAP / RankSHAP) | 16 × số phần lượt chấm (TB khoảng 740) |

Shapley được dùng làm chuẩn tham chiếu: với mỗi cách khác, báo Spearman giữa thứ tự của nó và giá trị Shapley, và tỉ lệ trùng top-5.
Kích thước phạt của GNNExplainer (2,0) được chọn trên 30 câu val khác (rng 7) trước khi chạy.

## Kết quả (`main_r3`, val, mean ± std qua 3 seed)

Bảng dưới lấy nhóm "tất cả câu"; bảng đầy đủ theo nhóm đúng / sai ở phụ lục.

### 1. Fidelity của E

| k | Bỏ top-k: mất hạng 1, E (%) | Ngẫu nhiên (%) | Hạng mới: E | Ngẫu nhiên | Chỉ giữ top-k: hạng, E | Ngẫu nhiên | Chỉ giữ top-k: phần hiệu ứng giữ lại, E (%) | Ngẫu nhiên (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | 20,5 ± 1,8 | 3,0 ± 0,2 | 1,82 ± 0,11 | 1,08 ± 0,01 | 3,00 ± 0,03 | 9,19 ± 0,55 | 46,1 | −25,7 |
| 3 | 34,9 ± 2,8 | 6,5 ± 0,3 | 3,17 ± 0,20 | 1,24 ± 0,01 | 2,16 ± 0,08 | 7,38 ± 0,36 | 64,3 | −11,1 |
| 5 | 43,6 ± 3,1 | 9,8 ± 0,5 | 4,24 ± 0,24 | 1,43 ± 0,02 | 2,17 ± 0,06 | 6,89 ± 0,33 | 64,6 | −6,0 |

- E vượt xa ngẫu nhiên ở cả hai chiều: bỏ top-3 làm mất hạng 1 gấp khoảng 5 lần ngẫu nhiên, và chỉ giữ top-3 cho hạng 2,2 so với 7,4.
- Nhưng E của v2 yếu hơn E của v1 trên cùng mô hình: ở v1, bỏ top-3 làm mất hạng 1 ở 65,7% câu.
  Lý do ở mục 2.
- Fid− gần như không tăng từ k = 3 lên k = 5 (64,3% lên 64,6% hiệu ứng).
  Điều này phù hợp với việc các phần thứ 4 và 5 của E phần lớn là bộ ba (bộ ba chiếm 78% top-5), mà giữ thêm bộ ba ít đổi điểm (mục 2).

### 2. Đưa term về thang điểm cuối làm bộ ba chiếm đầu E

| | Bộ ba ở top-1 / top-3 / top-5 của E (%) | Câu có top-1 là node / bộ ba (TB/seed) | Bỏ top-1 node: mất hạng 1 (%) | Bỏ top-1 bộ ba: mất hạng 1 (%) |
|---|---|---|---|---|
| v1 (term thô) | 1 / 5 / 11 | 10 466 / 124 | 40,8 (ngẫu nhiên 4,9) | 18,6 (ngẫu nhiên 12,0) |
| v2 (thang điểm cuối) | 70,5 / 76,4 / 77,6 | 3 121 / 7 469 | 49,3 (ngẫu nhiên 4,4) | 8,4 (ngẫu nhiên 2,5) |

- Theo đạo hàm, v2 đúng: kênh bộ ba có độ phân tán nhỏ trên 50 ứng viên, nên z-score khuếch đại mỗi đơn vị term của nó.
- Nhưng bỏ một bộ ba gần như không đổi điểm, vì attention softmax dồn sang các bộ ba tương tự còn lại (v1 đã thấy điều này ở countonly).
  Khi bộ ba đứng đầu E, bỏ nó chỉ làm mất hạng 1 ở 8,4% câu; khi node đứng đầu, con số là 49,3%.
- Hệ quả thấy rõ ở bảng bằng chứng: câu 574350_4 (chó và ô) có E top-3 là "frame has frame", "man has pant", "woman wearing pant"; ở v1, E là dog, umbrella, umbrella.
- Vậy đưa E về đúng đơn vị không làm E trung thực hơn khi đo bằng can thiệp.
  Đây là trường hợp cụ thể của mục 3: độ lớn của term không phản ánh hiệu ứng khi bỏ phần.

### 3. Độ lớn của term không phản ánh hiệu ứng khi bỏ phần

| k | Tổng term top-k (dự đoán) | Điểm mất khi bỏ top-k (đo được) | Tỉ số tổng | Spearman |
|---|---|---|---|---|
| 1 | 2,31 ± 0,35 | 0,189 ± 0,021 | 0,084 ± 0,022 | 0,183 ± 0,033 |
| 3 | 6,12 ± 0,96 | 0,375 ± 0,036 | 0,063 ± 0,016 | −0,035 ± 0,016 |
| 5 | 9,22 ± 1,46 | 0,501 ± 0,042 | 0,056 ± 0,013 | −0,097 ± 0,007 |

- Điểm mất thật chỉ bằng khoảng 6% tổng term của top-3, và gần như không tương quan với tổng đó.
- Cách giải thích phù hợp với số liệu, nhưng chưa kiểm riêng: term đo mức khớp tuyệt đối của từng phần, phần lớn là mức nền mà mọi ứng viên đều có; z-score chỉ giữ chênh lệch giữa các ứng viên, nên phần nền này không quyết định thứ hạng.
  Các dòng "So với TB top-50" của demo trừ đúng mức nền đó, và chỉ cần 1,95 dòng cho 80% biên (mục 5).
- So sánh: occlusion (tỉ số 1,15, Spearman 0,89 ở k = 3) và Integrated Gradients (1,43, 0,82) cộng tính tốt hơn nhiều (mục 7).

### 4. Validity

**Phản thực tế:**

- Bỏ tối đa 10 phần đầu của E làm ảnh mất hạng 1 ở 60,3 ± 2,9% câu, trung vị k_flip là 3.
  Với ảnh hạng 1 sai, con số là 74,4%; với ảnh đúng, 44,1%.
- Với 5 325 câu có ảnh hạng 1 sai (TB/seed), bỏ top-5 của E làm ảnh vàng lên hạng ở 30,7% câu và lên hạng 1 ở 13,1% câu.
  Hạng trung bình của ảnh vàng chỉ đổi từ 8,52 xuống 8,23, nên E của ảnh sai chỉ ra phần đã đẩy ảnh sai lên, nhưng bỏ chúng thường chưa đủ để ảnh đúng thắng.

**Ngữ nghĩa** (TB 32 382 dòng vật thể và 17 311 dòng bộ ba mỗi seed):

| Tiêu chí | Khớp (%) |
|---|---|
| Vật thể, strict | 58,3 ± 0,3 |
| Vật thể, lenient (tiêu chí v1) | 63,8 ± 0,2 |
| Vật thể, cùng nhãn sau alias VG | 51,6 |
| Vật thể, cùng danh từ chính | 58,1 |
| Bộ ba, chủ thể khớp | 30,1 |
| Bộ ba, đối tượng khớp | 28,8 |
| Bộ ba, vị từ khớp | 20,7 |
| Bộ ba, cả ba khớp | 5,4 ± 0,2 |

- Node ảnh được chú ý nhất mang đúng loại của node câu ở 58% dòng vật thể theo tiêu chí strict.
  Tiêu chí lenient của v1 cho 64%, trùng với số v1 báo; khoảng 5,5 điểm chênh lệch là các cặp như "woman" ~ "man" mà lenient chấp nhận.
- Dòng bộ ba hầu như không khớp nghĩa: chỉ 5,4% dòng khớp đủ chủ thể, đối tượng và vị từ.
  Vì bộ ba đứng đầu E ở 70% câu (mục 2), phần lớn dòng đầu tiên mà người dùng thấy là một bộ ba không khớp nghĩa với câu.
- Khớp nhóm số lượng không chấm cho `main_r3` (xem bảng v2 khác v1).

### 5. Sparsity

| Đo | Giá trị |
|---|---|
| Số phần đầu của E cho 80% khối term dương | 15,4 ± 0,2 phần, 33% số phần của ảnh |
| k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% hiệu ứng | 2,65 ± 0,14, đạt trong k ≤ 10 ở 55,7% câu |
| k cho 90% hiệu ứng | 2,90 ± 0,14, đạt ở 50,4% câu |
| Dòng demo ("So với TB top-50") cho 80% biên | 1,95 ± 0,01 |

- Khối term dàn trải: cần 15 phần để đạt 80%, do nhiều bộ ba có term gần bằng nhau.
- Nhưng theo hiệu ứng, E gọn: ở hơn nửa số câu, 2–3 phần đầu đã giữ 80% hiệu ứng.
  Ở các câu còn lại, kể cả 10 phần đầu cũng không giữ được 80% hiệu ứng.
- Con số "6–8% số phần" của v1 chia số dòng demo cho số phần của ảnh, hai đơn vị khác nhau; v2 không dùng con số đó.

### 6. Đồ thị con theo attention GAT

Tập "có lựa chọn" (2 803 câu/seed, số cạnh vào lớn hơn số cạnh bỏ):

| Dạng | Δ điểm: top | Ngẫu nhiên | Mất hạng 1: top (%) | Ngẫu nhiên (%) |
|---|---|---|---|---|
| messages (cắt truyền tin, giữ bộ ba) | −0,0011 ± 0,0007 | −0,0010 ± 0,0005 | 1,0 ± 0,6 | 0,7 ± 0,4 |
| edges (cách v1, bỏ cả bộ ba) | −0,0163 ± 0,0022 | −0,0145 ± 0,0019 | 5,7 ± 0,6 | 5,1 ± 0,5 |

- Cắt đường truyền tin của các cạnh có alpha cao nhất gần như không đổi điểm (Δ −0,001 trên thang z-score), và không khác cắt ngẫu nhiên.
- Hiệu ứng mà v1 đo được (5,7%) phần lớn đến từ việc bỏ luôn bộ ba của cạnh, không phải từ việc cắt truyền tin.
- Kết luận của v1 vẫn đúng và nay rõ hơn: alpha GAT không phải lời giải thích cho thứ hạng.

### 7. So sánh 9 cách giải thích

| Cách | Câu | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | k_flip đạt (%) | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Khớp Shapley: Spearman | Top-5 trùng Shapley (%) |
|---|---|---|---|---|---|---|---|
| shapley | 195 | **−0,683 ± 0,226** | **1,532 ± 0,541** | **85,3** | **0,118 / −0,046** | chuẩn | chuẩn |
| occlusion | 977 | −0,244 ± 0,027 | 1,234 ± 0,141 | 74,8 | 0,106 / −0,046 | **0,476 ± 0,007** | 58,5 |
| integrated_gradients | 977 | −0,445 ± 0,019 | 0,646 ± 0,167 | 81,9 | 0,090 / −0,024 | 0,288 ± 0,010 | **62,9** |
| gnnexplainer | 195 | −0,169 ± 0,072 | −0,336 ± 0,951 | 81,8 | 0,091 / −0,000 | 0,255 ± 0,010 | 56,7 |
| **model (E)** | 977 | 0,237 ± 0,066 | 0,764 ± 0,102 | 57,7 | 0,050 / 0,016 | 0,460 ± 0,007 | 31,8 |
| clip | 977 | 0,283 ± 0,030 | 0,234 ± 0,137 | 61,8 | 0,047 / 0,021 | 0,352 ± 0,016 | 35,4 |
| gradient | 977 | 0,826 ± 0,043 | −1,038 ± 0,275 | 27,9 | 0,012 / 0,057 | −0,133 ± 0,007 | 17,8 |
| random | 977 | 0,838 ± 0,005 | −0,621 ± 0,105 | 25,5 | 0,013 / 0,053 | −0,012 ± 0,009 | 14,3 |
| lowest | 977 | 1,009 ± 0,008 | −2,300 ± 0,279 | 10,9 | 0,002 / 0,068 | −0,460 ± 0,007 | 2,1 |

Số câu là trung bình mỗi seed.
Kiểm định Wilcoxon ghép cặp với E trên deletion AUC: mọi cách khác đều khác E với p < 1e−11 ở cả 3 seed, trừ clip (p lớn nhất 0,96).

- **Ba cách can thiệp hoặc phân rã có mốc (Shapley, occlusion, IG) trung thực nhất trên mọi trục.**
  Bỏ dần theo thứ tự của chúng làm điểm giảm xuống dưới mức rỗng (deletion AUC âm), và đều có Fid_α+ > Fid_α−.
- **E đứng giữa:** tốt hơn hẳn ngẫu nhiên và gradient × input, ngang CLIP trên deletion, tốt hơn CLIP trên insertion (0,76 so với 0,23).
- **Về thứ tự, E gần như khớp Shapley bằng occlusion** (Spearman 0,46 so với 0,48), mà không tốn lượt chấm nào.
  Nhưng top-5 của E chỉ trùng 32% với top-5 của Shapley, so với 59% của occlusion: E đúng ở thứ tự tổng thể nhưng sai ở các phần đầu, đúng chỗ bộ ba chen vào (mục 2).
- **Gradient × input gần như ngẫu nhiên** (Spearman với Shapley −0,13): đạo hàm cục bộ không nói gì về việc bỏ hẳn một phần.
- **GNNExplainer chọn đúng phần quan trọng** (top-5 trùng Shapley 57%, k_flip 82%), nhưng thứ tự tổng thể khớp Shapley kém (Spearman 0,26) và insertion AUC dao động mạnh giữa các seed (−0,34 ± 0,95).
- **Tính cộng** (k = 3, cùng các câu):

  | Cách | Tỉ số tổng | Spearman |
  |---|---|---|
  | occlusion | 1,153 ± 0,017 | 0,888 ± 0,011 |
  | integrated_gradients | 1,434 ± 0,037 | 0,815 ± 0,009 |
  | gradient | 1,730 ± 0,172 | 0,189 ± 0,019 |
  | model (E) | 0,058 ± 0,013 | −0,010 ± 0,018 |

  Occlusion ở k = 1 cộng đúng tuyệt đối theo định nghĩa; ở k = 3 và 5, tỉ số lớn hơn 1 cho thấy các phần đầu bổ trợ nhau.

### 8. Ảnh hạng 1 đúng và sai

- Ảnh sai mất hạng 1 dễ hơn: bỏ top-3 của E làm mất hạng 1 ở 44,9% câu sai và 23,4% câu đúng, vì cách biệt với hạng 2 nhỏ hơn khoảng 3 lần (0,30 so với 1,00).
- Ở cả hai nhóm, Shapley, IG, occlusion và GNNExplainer đều có deletion AUC âm và tốt hơn E; E tốt hơn ngẫu nhiên và gradient × input (bảng phụ lục).
- Validity strict thấp hơn ở ảnh sai: 56,1% so với 60,8%.

## Kết luận

1. **E của `main_r3` trung thực theo thứ tự, nhưng không theo độ lớn.**
   - Bỏ top-3 của E làm mất hạng 1 ở 35% câu, so với 6,5% khi bỏ ngẫu nhiên; chỉ giữ top-3 cho hạng 2,2 so với 7,4.
   - Thứ tự của E khớp Shapley gần bằng occlusion (Spearman 0,46 so với 0,48).
   - Nhưng điểm mất khi bỏ top-3 chỉ bằng 6% tổng term của chúng và không tương quan với tổng đó.
2. **Đưa term về thang điểm cuối làm E kém trung thực hơn khi đo bằng can thiệp** (Fid+ k = 3: 65,7% ở v1 xuống 34,9% ở v2), vì bộ ba chiếm đầu E ở 70% câu mà bỏ một bộ ba gần như không đổi điểm.
   Demo vốn đã nhóm dòng vật thể trước dòng bộ ba, nên không cần đổi thứ tự.
   Demo nay thêm cột "Bỏ phần này": với mỗi dòng, chấm lại ảnh bằng chính mô hình sau khi bỏ phần khớp (occlusion của phần đó) và ghi Δ điểm cùng hạng mới, kèm ghi chú rằng % điểm cuối chỉ là cách chia điểm.
3. **Nếu cần giải thích trung thực nhất, dùng occlusion hoặc Integrated Gradients.**
   - Occlusion: khớp Shapley tốt nhất (0,48), cộng tính tốt (0,89), chi phí khoảng 46 lượt chấm mỗi câu.
   - IG: deletion AUC tốt thứ hai sau Shapley, thoả completeness (sai số 1,1%), chi phí 20 lượt lan truyền ngược.
   - Shapley là chuẩn tốt nhất nhưng tốn khoảng 740 lượt chấm mỗi câu.
4. **Validity ngữ nghĩa vừa phải ở node, kém ở bộ ba:** 58% dòng vật thể khớp strict, chỉ 5% dòng bộ ba khớp đủ.
5. **Sparsity theo hiệu ứng tốt:** ở hơn nửa số câu, 2–3 phần đầu của E giữ 80% hiệu ứng đồ thị.
6. **Alpha GAT không phải lời giải thích:** cắt truyền tin của các cạnh alpha cao nhất làm mất hạng 1 ở 1,0% câu, so với 0,7% khi cắt ngẫu nhiên.

**Giới hạn:**

- Chưa có kết quả cho `countonly_factual_r3`, mô hình chính của nhóm (thiếu dữ liệu count_v1 trên máy này).
  Với fusion raw, E của countonly giữ như v1, nên mục 2 không áp dụng; các mục còn lại cần chạy để có số.
- So sánh các cách giải thích dùng 1 000 câu, GNNExplainer và Shapley chỉ khoảng 195 câu; std qua seed của hai cách này lớn hơn.
- Ảnh trung tính của IG và GNNExplainer là vector trung bình của từng loại phần; chọn mốc khác có thể đổi kết quả của hai cách này.
- Chỉ đo trên ảnh hạng 1, chỉ trên val.

## Phụ lục: bảng đầy đủ theo nhóm

### main_r3 (R@1 theo seed: 46,67, 46,56, 46,21)

**Chạy.**

| Seed | Giây (CPU) | Câu | E rỗng | noop khác hạng 1 | Sai số noop tối đa |
| --- | --- | --- | --- | --- | --- |
| 0 | 989 | 10633 | 45 | 0 | 1.8e-05 |
| 1 | 1096 | 10633 | 39 | 0 | 2.0e-05 |
| 2 | 871 | 10633 | 44 | 0 | 2.0e-05 |

**Fid+ (bỏ top-k).**

| Nhóm | k | Mất hạng 1: top (%) | Ngẫu nhiên (%) | Hạng: top | Ngẫu nhiên | Δ điểm: top | Ngẫu nhiên | Hiệu ứng bị bỏ, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 20,5 ± 1,8 | 3,0 ± 0,2 | 1,82 ± 0,11 | 1,08 ± 0,01 | −0,188 ± 0,021 | −0,012 ± 0,001 | 21,5 / 5,2 | 1,4 / 0,1 |
| all | 3 | 34,9 ± 2,8 | 6,5 ± 0,3 | 3,17 ± 0,20 | 1,24 ± 0,01 | −0,372 ± 0,036 | −0,037 ± 0,003 | 42,6 / 16,8 | 4,3 / 0,4 |
| all | 5 | 43,6 ± 3,1 | 9,8 ± 0,5 | 4,24 ± 0,24 | 1,43 ± 0,02 | −0,497 ± 0,042 | −0,065 ± 0,004 | 56,9 / 29,4 | 7,5 / 0,7 |
| correct | 1 | 12,6 ± 1,6 | 1,6 ± 0,3 | 1,39 ± 0,07 | 1,04 ± 0,01 | −0,203 ± 0,024 | −0,013 ± 0,001 | 20,1 / 4,5 | 1,3 / 0,1 |
| correct | 3 | 23,4 ± 2,4 | 3,6 ± 0,2 | 2,15 ± 0,14 | 1,11 ± 0,01 | −0,388 ± 0,041 | −0,039 ± 0,003 | 38,7 / 14,7 | 4,0 / 0,3 |
| correct | 5 | 30,1 ± 2,8 | 5,6 ± 0,2 | 2,79 ± 0,18 | 1,21 ± 0,01 | −0,510 ± 0,048 | −0,068 ± 0,003 | 50,9 / 25,8 | 6,8 / 0,6 |
| wrong | 1 | 27,3 ± 1,9 | 4,3 ± 0,2 | 2,18 ± 0,14 | 1,12 ± 0,01 | −0,176 ± 0,018 | −0,011 ± 0,001 | 23,1 / 5,8 | 1,4 / 0,1 |
| wrong | 3 | 44,9 ± 3,1 | 9,0 ± 0,5 | 4,06 ± 0,27 | 1,36 ± 0,02 | −0,358 ± 0,032 | −0,035 ± 0,003 | 46,9 / 18,5 | 4,7 / 0,4 |
| wrong | 5 | 55,2 ± 3,5 | 13,4 ± 1,0 | 5,50 ± 0,30 | 1,62 ± 0,04 | −0,486 ± 0,038 | −0,063 ± 0,006 | 63,7 / 32,2 | 8,3 / 0,8 |

**Fid− (chỉ giữ top-k).**

| Nhóm | k | Mất hạng 1: top (%) | Ngẫu nhiên (%) | Hạng: top | Ngẫu nhiên | Δ điểm: top | Ngẫu nhiên | Hiệu ứng giữ lại, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 43,9 ± 2,5 | 70,9 ± 0,1 | 3,00 ± 0,03 | 9,19 ± 0,55 | −0,464 ± 0,021 | −1,094 ± 0,037 | 46,1 / 53,1 | −25,7 / 4,5 |
| all | 3 | 35,8 ± 3,3 | 69,5 ± 0,2 | 2,16 ± 0,08 | 7,38 ± 0,36 | −0,306 ± 0,043 | −0,967 ± 0,022 | 64,3 / 64,4 | −11,1 / 8,8 |
| all | 5 | 35,9 ± 2,9 | 67,9 ± 0,2 | 2,17 ± 0,06 | 6,89 ± 0,33 | −0,304 ± 0,043 | −0,922 ± 0,020 | 64,6 / 63,7 | −6,0 / 11,0 |
| correct | 1 | 35,4 ± 1,3 | 60,0 ± 1,1 | 2,27 ± 0,02 | 6,85 ± 0,52 | −0,598 ± 0,023 | −1,218 ± 0,040 | 39,8 / 43,9 | −21,8 / 3,0 |
| correct | 3 | 28,0 ± 2,2 | 57,5 ± 0,6 | 1,77 ± 0,04 | 5,48 ± 0,35 | −0,417 ± 0,049 | −1,096 ± 0,024 | 58,0 / 54,9 | −9,6 / 6,9 |
| correct | 5 | 27,9 ± 2,3 | 55,7 ± 0,6 | 1,78 ± 0,05 | 5,16 ± 0,37 | −0,405 ± 0,051 | −1,049 ± 0,023 | 59,2 / 54,7 | −4,9 / 9,2 |
| wrong | 1 | 51,2 ± 3,6 | 80,2 ± 0,6 | 3,64 ± 0,04 | 11,22 ± 0,58 | −0,348 ± 0,021 | −0,987 ± 0,033 | 53,3 / 63,0 | −30,1 / 6,5 |
| wrong | 3 | 42,4 ± 4,2 | 79,9 ± 0,7 | 2,50 ± 0,12 | 9,02 ± 0,38 | −0,210 ± 0,039 | −0,856 ± 0,020 | 71,5 / 72,3 | −12,9 / 11,1 |
| wrong | 5 | 42,9 ± 3,4 | 78,4 ± 0,7 | 2,51 ± 0,07 | 8,39 ± 0,31 | −0,216 ± 0,037 | −0,812 ± 0,017 | 70,8 / 70,5 | −7,2 / 13,4 |

**Fid k = 1 theo loại phần đứng đầu E** (ngẫu nhiên: một phần cùng loại).

| Nhóm | Top-1 của E | Câu (TB/seed) | Bỏ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) | Chỉ giữ top-1: hạng | Ngẫu nhiên |
| --- | --- | --- | --- | --- | --- | --- |
| all | node | 3121 | 49,3 ± 0,3 | 4,4 ± 0,1 | 5,27 ± 0,22 | 20,38 ± 0,42 |
| all | triple | 7469 | 8,4 ± 0,4 | 2,5 ± 0,2 | 2,05 ± 0,16 | 4,52 ± 0,01 |
| correct | node | 1532 | 32,7 ± 0,7 | 2,7 ± 0,1 | 3,44 ± 0,14 | 14,80 ± 0,39 |
| correct | triple | 3386 | 3,5 ± 0,6 | 1,1 ± 0,3 | 1,75 ± 0,10 | 3,26 ± 0,08 |
| wrong | node | 1589 | 65,2 ± 0,5 | 6,0 ± 0,2 | 7,03 ± 0,32 | 25,74 ± 0,37 |
| wrong | triple | 4083 | 12,5 ± 0,4 | 3,7 ± 0,1 | 2,31 ± 0,22 | 5,56 ± 0,09 |

**Hình dạng của E.**

| Nhóm | Node / ảnh | Bộ ba / ảnh | Bộ ba trong top-1 / 3 / 5 của E (%) | Top-3 node chiếm (% term dương kênh vật thể) | Top-3 bộ ba chiếm (% kênh bộ ba) | Cách biệt hạng 1 – hạng 2 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 27,5 | 18,7 | 70,5 / 76,4 / 77,6 | 63,1 ± 0,9 | 22,8 ± 0,2 | 0,628 ± 0,003 |
| correct | 27,3 | 18,8 | 68,9 / 75,6 / 77,4 | 64,0 ± 0,8 | 22,6 ± 0,2 | 1,004 ± 0,005 |
| wrong | 27,7 | 18,7 | 72,0 / 77,0 / 77,7 | 62,3 ± 1,1 | 23,0 ± 0,2 | 0,301 ± 0,004 |

**Tính cộng của E:** tổng term của top-k (dự đoán) so với điểm mất thật khi bỏ top-k.

| Nhóm | k | Câu | Dự đoán (TB) | Đo được (TB) | Tỉ số tổng | Tỉ số trung vị | Spearman | Sai lệch tuyệt đối / hiệu ứng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 10455 | 2,31 ± 0,35 | 0,189 ± 0,021 | 0,084 ± 0,022 | 0,019 ± 0,006 | 0,183 ± 0,033 | 4,3 ± 1,0 |
| all | 3 | 10455 | 6,12 ± 0,96 | 0,375 ± 0,036 | 0,063 ± 0,016 | 0,019 ± 0,006 | −0,035 ± 0,016 | 11,6 ± 2,7 |
| all | 5 | 10455 | 9,22 ± 1,46 | 0,501 ± 0,042 | 0,056 ± 0,013 | 0,021 ± 0,007 | −0,097 ± 0,007 | 17,5 ± 4,0 |
| correct | 1 | 4864 | 2,12 ± 0,33 | 0,203 ± 0,024 | 0,099 ± 0,026 | 0,020 ± 0,007 | 0,185 ± 0,028 | 3,4 ± 0,7 |
| correct | 3 | 4864 | 5,53 ± 0,88 | 0,391 ± 0,041 | 0,073 ± 0,019 | 0,021 ± 0,007 | −0,029 ± 0,015 | 8,7 ± 1,7 |
| correct | 5 | 4864 | 8,30 ± 1,35 | 0,514 ± 0,048 | 0,064 ± 0,016 | 0,023 ± 0,008 | −0,093 ± 0,009 | 12,9 ± 2,4 |
| wrong | 1 | 5590 | 2,47 ± 0,38 | 0,177 ± 0,018 | 0,074 ± 0,019 | 0,018 ± 0,005 | 0,189 ± 0,042 | 5,1 ± 1,3 |
| wrong | 3 | 5590 | 6,63 ± 1,02 | 0,360 ± 0,033 | 0,056 ± 0,013 | 0,018 ± 0,006 | −0,036 ± 0,022 | 14,1 ± 3,5 |
| wrong | 5 | 5590 | 10,03 ± 1,55 | 0,489 ± 0,038 | 0,050 ± 0,011 | 0,019 ± 0,006 | −0,098 ± 0,011 | 21,5 ± 5,3 |

**Sparsity.**

| Nhóm | Phần cho 80% / 90% khối term dương | % số phần (80 / 90) | k nhỏ nhất giữ 80% hiệu ứng | k cho 90% | Đạt trong k ≤ 10: 80% / 90% (% câu) | Dòng demo cho 80% / 90% biên |
| --- | --- | --- | --- | --- | --- | --- |
| all | 15,4 ± 0,2 / 18,0 ± 0,3 | 33,0 / 38,7 | 2,65 ± 0,14 | 2,90 ± 0,14 | 55,7 / 50,4 | 1,95 ± 0,01 / 2,28 ± 0,01 |
| correct | 15,7 ± 0,2 / 18,4 ± 0,3 | 33,8 / 39,6 | 3,09 ± 0,16 | 3,37 ± 0,14 | 49,3 / 44,6 | 2,02 ± 0,01 / 2,39 ± 0,01 |
| wrong | 15,1 ± 0,1 / 17,7 ± 0,2 | 32,3 / 37,9 | 2,34 ± 0,13 | 2,57 ± 0,15 | 61,2 / 55,5 | 1,88 ± 0,01 / 2,19 ± 0,02 |

**Phản thực tế.**

| Nhóm | k_flip đạt (≤ 10, %) | k_flip TB | k_flip trung vị | Câu có ảnh vàng khác hạng 1 | Hạng ảnh vàng trước | Sau bỏ top-1 / 3 / 5 | Ảnh vàng lên hạng: k = 1 / 3 / 5 (%) | Ảnh vàng lên hạng 1: k = 1 / 3 / 5 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 60,3 ± 2,9 | 3,71 ± 0,12 | 3,0 | 5325 | 8,52 ± 0,04 | 8,41 / 8,30 / 8,23 | 12,6 / 23,8 / 30,7 | 6,6 / 10,6 / 13,1 |
| correct | 44,1 ± 3,0 | 4,03 ± 0,15 | 3,0 | 0 | n/a | n/a | n/a | n/a |
| wrong | 74,4 ± 2,9 | 3,54 ± 0,11 | 2,3 | 5325 | 8,52 ± 0,04 | 8,41 / 8,30 / 8,23 | 12,6 / 23,8 / 30,7 | 6,6 / 10,6 / 13,1 |

**Validity ngữ nghĩa** (dòng vật thể và dòng bộ ba của E).

| Nhóm | Dòng vật thể | strict (%) | alias | WordNet strict | lenient (như v1) | WordNet lenient | CLIP ≥ 0,955 | Cùng danh từ chính | Dòng có số | Khớp nhóm số (%) | Dòng bộ ba | Chủ thể | Đối tượng | Vị từ | Cả ba (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 32382 | 58,3 ± 0,3 | 51,6 | 46,7 | 63,8 ± 0,2 | 49,5 | 57,4 | 58,1 | 0 | n/a | 17311 | 30,1 | 28,8 | 20,7 | 5,4 ± 0,2 |
| correct | 15515 | 60,8 ± 0,3 | 54,0 | 49,4 | 66,4 ± 0,3 | 52,4 | 59,7 | 61,0 | 0 | n/a | 8443 | 32,7 | 31,0 | 21,8 | 6,5 ± 0,2 |
| wrong | 16867 | 56,1 ± 0,3 | 49,3 | 44,3 | 61,4 ± 0,3 | 46,8 | 55,2 | 55,5 | 0 | n/a | 8868 | 27,5 | 26,7 | 19,6 | 4,3 ± 0,3 |

**Đồ thị con theo alpha GAT** (messages: cắt đường truyền tin, giữ bộ ba; edges: cách v1, bỏ cả bộ ba).

| Nhóm | Tập câu | Câu (TB/seed) | Cạnh bỏ / cạnh vào | Dạng | Δ điểm: top | Ngẫu nhiên | Mất hạng 1: top (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | có cạnh vào | 9565 | 3,40 / 4,65 | messages | −0,0005 ± 0,0001 | −0,0005 ± 0,0001 | 0,7 ± 0,4 | 0,6 ± 0,3 |
| all | có cạnh vào | 9565 | 3,40 / 4,65 | edges | −0,0209 ± 0,0014 | −0,0203 ± 0,0012 | 6,3 ± 0,0 | 6,2 ± 0,1 |
| all | có lựa chọn | 2803 | 5,25 / 9,50 | messages | −0,0011 ± 0,0007 | −0,0010 ± 0,0005 | 1,0 ± 0,6 | 0,7 ± 0,4 |
| all | có lựa chọn | 2803 | 5,25 / 9,50 | edges | −0,0163 ± 0,0022 | −0,0145 ± 0,0019 | 5,7 ± 0,6 | 5,1 ± 0,5 |
| correct | có cạnh vào | 4453 | 3,52 / 4,86 | messages | −0,0008 ± 0,0004 | −0,0007 ± 0,0003 | 0,3 ± 0,1 | 0,3 ± 0,1 |
| correct | có cạnh vào | 4453 | 3,52 / 4,86 | edges | −0,0185 ± 0,0012 | −0,0181 ± 0,0010 | 2,7 ± 0,2 | 2,6 ± 0,1 |
| correct | có lựa chọn | 1417 | 5,34 / 9,53 | messages | −0,0014 ± 0,0010 | −0,0012 ± 0,0007 | 0,5 ± 0,3 | 0,3 ± 0,2 |
| correct | có lựa chọn | 1417 | 5,34 / 9,53 | edges | −0,0163 ± 0,0014 | −0,0152 ± 0,0008 | 2,8 ± 0,4 | 2,4 ± 0,3 |
| wrong | có cạnh vào | 5112 | 3,30 / 4,47 | messages | −0,0003 ± 0,0004 | −0,0003 ± 0,0002 | 1,1 ± 0,6 | 0,9 ± 0,5 |
| wrong | có cạnh vào | 5112 | 3,30 / 4,47 | edges | −0,0230 ± 0,0016 | −0,0223 ± 0,0015 | 9,5 ± 0,1 | 9,3 ± 0,1 |
| wrong | có lựa chọn | 1386 | 5,16 / 9,47 | messages | −0,0009 ± 0,0014 | −0,0007 ± 0,0008 | 1,6 ± 1,0 | 1,1 ± 0,7 |
| wrong | có lựa chọn | 1386 | 5,16 / 9,47 | edges | −0,0162 ± 0,0031 | −0,0137 ± 0,0032 | 8,7 ± 1,1 | 7,8 ± 1,0 |

**So sánh các cách giải thích** (mẫu cố định trong val; GNNExplainer và Shapley trên tập con).

| Nhóm | Cách | Câu | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | k_flip đạt (%) | Bỏ top-1 / 3 / 5: mất hạng 1 (%) | Chỉ giữ top-1 / 3 / 5: hạng | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Fid_α+ / Fid_α− (k = 5, α = 0,5) | Khớp Shapley: Spearman | Top-5 trùng Shapley (%) | p Wilcoxon deletion so với model (lớn nhất qua seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | model | 977 | 0,237 ± 0,066 | 0,764 ± 0,102 | 57,7 ± 2,5 | 19,5 / 33,7 / 41,4 | 2,82 / 2,12 / 2,14 | 0,050 ± 0,004 / 0,016 ± 0,011 | 0,289 ± 0,035 / 0,211 ± 0,034 | 0,460 ± 0,007 | 31,8 ± 2,6 | n/a |
| all | random | 977 | 0,838 ± 0,005 | −0,621 ± 0,105 | 25,5 ± 1,4 | 3,7 / 8,6 / 12,1 | 12,88 / 12,83 / 10,18 | 0,013 ± 0,001 / 0,053 ± 0,002 | 0,070 ± 0,004 / 0,409 ± 0,002 | −0,012 ± 0,009 | 14,3 ± 0,7 | 1,4e−105 |
| all | lowest | 977 | 1,009 ± 0,008 | −2,300 ± 0,279 | 10,9 ± 1,0 | 1,1 / 2,5 / 4,2 | 28,46 / 25,29 / 23,36 | 0,002 ± 0,002 / 0,068 ± 0,003 | 0,005 ± 0,005 / 0,535 ± 0,010 | −0,460 ± 0,007 | 2,1 ± 0,4 | 4,7e−157 |
| all | clip | 977 | 0,283 ± 0,030 | 0,234 ± 0,137 | 61,8 ± 1,0 | 10,9 / 29,6 / 42,8 | 5,17 / 5,34 / 4,83 | 0,047 ± 0,002 / 0,021 ± 0,004 | 0,277 ± 0,019 / 0,222 ± 0,099 | 0,352 ± 0,016 | 35,4 ± 1,5 | 9,6e−01 |
| all | gradient | 977 | 0,826 ± 0,043 | −1,038 ± 0,275 | 27,9 ± 2,9 | 8,8 / 14,1 / 17,8 | 13,30 / 12,17 / 10,94 | 0,012 ± 0,002 / 0,057 ± 0,004 | 0,071 ± 0,018 / 0,415 ± 0,018 | −0,133 ± 0,007 | 17,8 ± 1,4 | 7,6e−84 |
| all | integrated_gradients | 977 | −0,445 ± 0,019 | 0,646 ± 0,167 | 81,9 ± 0,4 | 39,3 / 64,6 / 74,0 | 7,23 / 2,46 / 1,78 | 0,090 ± 0,001 / −0,024 ± 0,005 | 0,555 ± 0,014 / −0,089 ± 0,012 | 0,288 ± 0,010 | 62,9 ± 1,6 | 8,1e−118 |
| all | occlusion | 977 | −0,244 ± 0,027 | 1,234 ± 0,141 | 74,8 ± 0,1 | 44,9 / 63,0 / 68,8 | 7,68 / 1,99 / 1,39 | 0,106 ± 0,004 / −0,046 ± 0,008 | 0,585 ± 0,027 / −0,264 ± 0,053 | 0,476 ± 0,007 | 58,5 ± 1,0 | 9,5e−77 |
| all | gnnexplainer | 195 | −0,169 ± 0,072 | −0,336 ± 0,951 | 81,8 ± 2,4 | 34,6 / 63,7 / 73,5 | 10,01 / 3,59 / 2,42 | 0,091 ± 0,017 / −0,000 ± 0,010 | 0,491 ± 0,060 / 0,105 ± 0,127 | 0,255 ± 0,010 | 56,7 ± 1,6 | 1,3e−12 |
| all | shapley | 195 | −0,683 ± 0,226 | 1,532 ± 0,541 | 85,3 ± 1,7 | 43,1 / 67,1 / 77,4 | 6,14 / 1,72 / 1,14 | 0,118 ± 0,016 / −0,046 ± 0,022 | 0,749 ± 0,192 / −0,199 ± 0,096 | n/a | n/a | 8,1e−21 |
| correct | model | 456 | 0,348 ± 0,037 | 0,654 ± 0,060 | 42,0 ± 1,6 | 11,5 / 23,6 / 29,9 | 2,22 / 1,86 / 1,84 | 0,043 ± 0,003 / 0,019 ± 0,008 | 0,250 ± 0,023 / 0,183 ± 0,035 | 0,448 ± 0,014 | 31,3 ± 2,5 | n/a |
| correct | random | 456 | 0,850 ± 0,027 | −0,373 ± 0,071 | 17,2 ± 0,3 | 1,8 / 5,5 / 7,8 | 9,03 / 8,68 / 6,74 | 0,014 ± 0,004 / 0,048 ± 0,002 | 0,069 ± 0,017 / 0,358 ± 0,028 | 0,002 ± 0,027 | 14,8 ± 1,5 | 2,0e−46 |
| correct | lowest | 456 | 1,002 ± 0,012 | −1,675 ± 0,022 | 6,5 ± 0,7 | 0,3 / 1,5 / 2,9 | 22,70 / 19,33 / 17,28 | 0,006 ± 0,002 / 0,062 ± 0,001 | 0,007 ± 0,007 / 0,460 ± 0,008 | −0,448 ± 0,014 | 2,3 ± 0,2 | 4,7e−73 |
| correct | clip | 456 | 0,411 ± 0,032 | 0,321 ± 0,060 | 45,7 ± 0,9 | 6,1 / 18,7 / 27,7 | 3,32 / 3,55 / 3,07 | 0,043 ± 0,002 / 0,020 ± 0,001 | 0,233 ± 0,006 / 0,159 ± 0,011 | 0,326 ± 0,032 | 33,1 ± 1,4 | 4,4e−01 |
| correct | gradient | 456 | 0,801 ± 0,037 | −0,613 ± 0,070 | 18,9 ± 3,4 | 5,3 / 8,8 / 11,7 | 8,92 / 7,81 / 7,01 | 0,013 ± 0,005 / 0,053 ± 0,008 | 0,079 ± 0,021 / 0,350 ± 0,020 | −0,134 ± 0,013 | 18,1 ± 1,4 | 2,7e−40 |
| correct | integrated_gradients | 456 | −0,218 ± 0,025 | 0,665 ± 0,079 | 69,5 ± 0,6 | 24,8 / 49,1 / 60,1 | 4,72 / 1,75 / 1,31 | 0,077 ± 0,001 / −0,017 ± 0,001 | 0,477 ± 0,004 / −0,065 ± 0,022 | 0,278 ± 0,009 | 63,5 ± 2,0 | 1,7e−58 |
| correct | occlusion | 456 | −0,060 ± 0,008 | 1,101 ± 0,040 | 58,4 ± 0,7 | 27,7 / 46,5 / 51,6 | 4,93 / 1,45 / 1,18 | 0,092 ± 0,003 / −0,034 ± 0,001 | 0,495 ± 0,007 / −0,191 ± 0,012 | 0,455 ± 0,016 | 58,4 ± 1,3 | 1,2e−39 |
| correct | gnnexplainer | 80 | −0,102 ± 0,060 | 0,458 ± 0,257 | 67,5 ± 2,1 | 20,4 / 50,0 / 59,4 | 5,09 / 1,71 / 1,39 | 0,080 ± 0,024 / −0,006 ± 0,011 | 0,446 ± 0,070 / 0,001 ± 0,069 | 0,272 ± 0,022 | 57,5 ± 1,8 | 3,9e−06 |
| correct | shapley | 80 | −0,409 ± 0,034 | 1,201 ± 0,051 | 74,1 ± 2,4 | 29,6 / 52,4 / 64,5 | 3,46 / 1,20 / 1,03 | 0,087 ± 0,007 / −0,027 ± 0,006 | 0,568 ± 0,014 / −0,122 ± 0,010 | n/a | n/a | 2,2e−10 |
| wrong | model | 520 | 0,139 ± 0,094 | 0,861 ± 0,138 | 71,6 ± 3,3 | 26,6 / 42,5 / 51,5 | 3,34 / 2,35 / 2,40 | 0,055 ± 0,006 / 0,013 ± 0,013 | 0,324 ± 0,045 / 0,236 ± 0,033 | 0,468 ± 0,003 | 32,1 ± 2,9 | n/a |
| wrong | random | 520 | 0,829 ± 0,022 | −0,838 ± 0,227 | 32,7 ± 2,4 | 5,4 / 11,2 / 15,8 | 16,26 / 16,48 / 13,19 | 0,013 ± 0,002 / 0,057 ± 0,005 | 0,071 ± 0,010 / 0,454 ± 0,026 | −0,022 ± 0,015 | 13,9 ± 0,6 | 9,4e−61 |
| wrong | lowest | 520 | 1,015 ± 0,005 | −2,847 ± 0,535 | 14,7 ± 1,6 | 1,9 / 3,3 / 5,3 | 33,51 / 30,52 / 28,68 | −0,000 ± 0,002 / 0,072 ± 0,005 | 0,003 ± 0,003 / 0,600 ± 0,015 | −0,468 ± 0,003 | 1,9 ± 0,7 | 1,0e−85 |
| wrong | clip | 520 | 0,171 ± 0,029 | 0,159 ± 0,274 | 75,8 ± 1,0 | 15,1 / 39,2 / 55,9 | 6,79 / 6,91 / 6,36 | 0,051 ± 0,003 / 0,022 ± 0,007 | 0,316 ± 0,033 / 0,277 ± 0,186 | 0,371 ± 0,019 | 37,1 ± 1,6 | 9,1e−01 |
| wrong | gradient | 520 | 0,849 ± 0,058 | −1,410 ± 0,524 | 35,8 ± 2,6 | 11,8 / 18,8 / 23,1 | 17,15 / 16,01 / 14,39 | 0,011 ± 0,002 / 0,060 ± 0,001 | 0,063 ± 0,026 / 0,473 ± 0,043 | −0,132 ± 0,005 | 17,6 ± 1,7 | 2,8e−45 |
| wrong | integrated_gradients | 520 | −0,643 ± 0,052 | 0,630 ± 0,266 | 92,8 ± 1,0 | 52,1 / 78,3 / 86,1 | 9,42 / 3,08 / 2,19 | 0,102 ± 0,003 / −0,030 ± 0,009 | 0,624 ± 0,025 / −0,110 ± 0,008 | 0,295 ± 0,010 | 62,5 ± 2,0 | 6,5e−60 |
| wrong | occlusion | 520 | −0,405 ± 0,056 | 1,350 ± 0,232 | 89,2 ± 0,6 | 60,1 / 77,6 / 83,8 | 10,09 / 2,47 / 1,56 | 0,119 ± 0,005 / −0,057 ± 0,014 | 0,664 ± 0,045 / −0,328 ± 0,089 | 0,490 ± 0,003 | 58,6 ± 1,4 | 1,2e−39 |
| wrong | gnnexplainer | 115 | −0,216 ± 0,099 | −0,906 ± 1,717 | 91,8 ± 3,1 | 44,4 / 73,3 / 83,2 | 13,43 / 4,90 / 3,12 | 0,098 ± 0,012 / 0,004 ± 0,013 | 0,522 ± 0,071 / 0,180 ± 0,228 | 0,244 ± 0,015 | 56,1 ± 1,6 | 4,9e−07 |
| wrong | shapley | 115 | −0,878 ± 0,415 | 1,773 ± 0,936 | 93,0 ± 2,7 | 52,6 / 77,3 / 86,3 | 8,00 / 2,08 / 1,22 | 0,140 ± 0,026 / −0,060 ± 0,038 | 0,879 ± 0,326 / −0,255 ± 0,160 | n/a | n/a | 5,0e−12 |

**Tính cộng của các cách phân rã** (cùng các câu so sánh).

| Nhóm | Cách | k | Tỉ số tổng | Tỉ số trung vị | Spearman | Sai lệch tuyệt đối / hiệu ứng |
| --- | --- | --- | --- | --- | --- | --- |
| all | model | 1 | 0,083 ± 0,021 | 0,018 ± 0,005 | 0,180 ± 0,029 | 3,88 ± 0,52 |
| all | model | 3 | 0,058 ± 0,013 | 0,018 ± 0,006 | −0,010 ± 0,018 | 10,50 ± 1,20 |
| all | model | 5 | 0,052 ± 0,012 | 0,020 ± 0,006 | −0,060 ± 0,012 | 15,63 ± 1,72 |
| all | gradient | 1 | 1,913 ± 0,342 | −0,143 ± 0,094 | 0,169 ± 0,086 | 0,15 ± 0,03 |
| all | gradient | 3 | 1,730 ± 0,172 | −0,053 ± 0,095 | 0,189 ± 0,019 | 0,28 ± 0,05 |
| all | gradient | 5 | 1,776 ± 0,197 | 0,051 ± 0,192 | 0,174 ± 0,028 | 0,35 ± 0,05 |
| all | integrated_gradients | 1 | 1,154 ± 0,023 | 1,047 ± 0,013 | 0,770 ± 0,015 | 0,28 ± 0,00 |
| all | integrated_gradients | 3 | 1,434 ± 0,037 | 1,391 ± 0,027 | 0,815 ± 0,009 | 0,49 ± 0,02 |
| all | integrated_gradients | 5 | 1,602 ± 0,049 | 1,590 ± 0,054 | 0,817 ± 0,024 | 0,64 ± 0,02 |
| all | occlusion | 1 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,00 ± 0,00 |
| all | occlusion | 3 | 1,153 ± 0,017 | 1,077 ± 0,007 | 0,888 ± 0,011 | 0,24 ± 0,02 |
| all | occlusion | 5 | 1,245 ± 0,021 | 1,141 ± 0,023 | 0,837 ± 0,005 | 0,34 ± 0,02 |
| correct | model | 1 | 0,101 ± 0,030 | 0,021 ± 0,008 | 0,167 ± 0,026 | 3,09 ± 0,78 |
| correct | model | 3 | 0,072 ± 0,019 | 0,022 ± 0,008 | −0,042 ± 0,023 | 8,16 ± 1,88 |
| correct | model | 5 | 0,063 ± 0,017 | 0,023 ± 0,009 | −0,098 ± 0,012 | 11,65 ± 2,95 |
| correct | gradient | 1 | 2,390 ± 0,719 | −0,145 ± 0,130 | 0,174 ± 0,115 | 0,14 ± 0,02 |
| correct | gradient | 3 | 2,027 ± 0,428 | −0,063 ± 0,105 | 0,188 ± 0,030 | 0,25 ± 0,02 |
| correct | gradient | 5 | 2,081 ± 0,414 | 0,042 ± 0,220 | 0,154 ± 0,034 | 0,29 ± 0,02 |
| correct | integrated_gradients | 1 | 1,159 ± 0,019 | 1,080 ± 0,028 | 0,777 ± 0,016 | 0,24 ± 0,02 |
| correct | integrated_gradients | 3 | 1,454 ± 0,037 | 1,414 ± 0,046 | 0,811 ± 0,008 | 0,42 ± 0,02 |
| correct | integrated_gradients | 5 | 1,611 ± 0,048 | 1,603 ± 0,066 | 0,823 ± 0,016 | 0,54 ± 0,02 |
| correct | occlusion | 1 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,00 ± 0,00 |
| correct | occlusion | 3 | 1,161 ± 0,005 | 1,092 ± 0,007 | 0,890 ± 0,005 | 0,22 ± 0,03 |
| correct | occlusion | 5 | 1,241 ± 0,005 | 1,149 ± 0,022 | 0,854 ± 0,010 | 0,27 ± 0,01 |
| wrong | model | 1 | 0,070 ± 0,016 | 0,017 ± 0,004 | 0,204 ± 0,045 | 4,57 ± 0,34 |
| wrong | model | 3 | 0,049 ± 0,010 | 0,017 ± 0,005 | 0,036 ± 0,037 | 12,55 ± 0,67 |
| wrong | model | 5 | 0,045 ± 0,009 | 0,018 ± 0,005 | −0,015 ± 0,032 | 19,12 ± 0,69 |
| wrong | gradient | 1 | 1,550 ± 0,091 | −0,138 ± 0,075 | 0,168 ± 0,075 | 0,17 ± 0,04 |
| wrong | gradient | 3 | 1,504 ± 0,068 | −0,045 ± 0,090 | 0,197 ± 0,011 | 0,30 ± 0,07 |
| wrong | gradient | 5 | 1,538 ± 0,073 | 0,076 ± 0,163 | 0,195 ± 0,034 | 0,39 ± 0,09 |
| wrong | integrated_gradients | 1 | 1,149 ± 0,041 | 1,028 ± 0,033 | 0,762 ± 0,031 | 0,32 ± 0,01 |
| wrong | integrated_gradients | 3 | 1,416 ± 0,037 | 1,373 ± 0,013 | 0,818 ± 0,009 | 0,55 ± 0,02 |
| wrong | integrated_gradients | 5 | 1,594 ± 0,050 | 1,572 ± 0,034 | 0,811 ± 0,034 | 0,72 ± 0,03 |
| wrong | occlusion | 1 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,00 ± 0,00 |
| wrong | occlusion | 3 | 1,146 ± 0,029 | 1,063 ± 0,008 | 0,886 ± 0,021 | 0,27 ± 0,01 |
| wrong | occlusion | 5 | 1,249 ± 0,037 | 1,131 ± 0,019 | 0,820 ± 0,020 | 0,39 ± 0,02 |

**Kiểm tra:** IG completeness, |Σ attribution − (f(ảnh) − f(trung tính))| / hiệu ứng: 1,1 ± 0,1%. Shapley efficiency, sai lệch tối đa: 1,3e−07.

## Phụ lục: 13 câu bằng chứng

### main_r3 (seed 0)

| Câu (caption_id) | Ảnh hạng 1 | E top-3 (term) | Bỏ top-3: Δ điểm, hạng (ngẫu nhiên) | Chỉ giữ top-3: Δ điểm, hạng (ngẫu nhiên) | k_flip | k giữ 80% | Phần cho 80% term | Tính cộng k = 3: dự đoán / đo | Vật thể khớp strict | Bộ ba khớp đủ | Cắt truyền tin: Δ top (ngẫu nhiên) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | 2377477 (đúng) | room with stove (1,227); wood under stove (1,203); stove burning wood (1,195) | −0,067, 1 (−0,004, 1,0) | −0,125, 1 (−0,437, 3,2) | 7 | 1 | 19 | 3,624 / 0,067 | 1/3 | 0/1 | −0,0272 (−0,0135) |
| A cow standing along side of a large river. (422698_1) | 2351201 (đúng) | moss on water (2,541); river surrounded by plant (2,527); shade area along edge of water (2,477) | −0,104, 1 (0,011, 1,0) | −1,286, 11 (−1,574, 12,2) | > 10 | > 10 | 12 | 7,544 / 0,104 | 2/3 | 0/2 | 0,0000 (0,0000) |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | 2329851 (đúng) | pen of sheep (2,336); pen of sheep (2,336); pen of sheep (2,336) | −0,221, 1 (0,004, 1,0) | −0,830, 2 (−1,411, 11,7) | 5 | > 10 | 11 | 7,009 / 0,221 | 2/4 | 0/2 | 0,0006 (0,0006) |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | 2402283 (đúng) | donut (1,241); bite (0,954); chocolate cream (0,064) | −1,410, 11 (−0,226, 2,4) | 0,143, 1 (−1,095, 8,0) | 1 | 2 | 2 | 2,259 / 1,410 | 1/2 | 0/0 | 0,0112 (0,0112) |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | 1159610 (đúng) | frame has frame (0,942); man has pant (0,912); woman wearing pant (0,899) | −0,026, 1 (−0,001, 1,0) | −0,613, 2 (−0,648, 2,0) | 7 | 9 | 9 | 2,754 / 0,026 | 5/6 | 0/4 | 0,0000 (0,0000) |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | 2366682 (sai) | banana in plate (4,007); banana in plate (3,984); banana in plate (3,984) | −0,181, 1 (0,012, 1,0) | −0,611, 2 (−0,794, 2,8) | 9 | 10 | 7 | 11,974 / 0,181 | 2/3 | 0/2 | 0,0210 (0,0210) |
| A train riding along the tracks along a bank of dirt. (279933_1) | 2378096 (sai) | cfv under headlight (2,736); headlight on train (2,657); grass on ground (2,629) | −0,165, 1 (−0,014, 1,0) | −0,716, 4 (−1,253, 10,7) | 8 | > 10 | 14 | 8,022 / 0,165 | 2/3 | 0/1 | 0,0000 (0,0000) |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | 2355061 (đúng) | bus (0,393); bus (0,393); bus (0,393) | −0,399, 3 (−0,005, 1,0) | −0,315, 3 (−0,572, 5,8) | 3 | 4 | 6 | 1,180 / 0,399 | 1/2 | 0/0 | 0,0001 (0,0001) |
| Two men standing in a living room playing video game with remotes. (34471_0) | 2395602 (sai) | man wearing shoe (2,049); man has leg (2,040); writing on jean (2,028) | −0,051, 1 (−0,011, 1,0) | −0,414, 1 (−0,567, 2,2) | > 10 | > 10 | 13 | 6,117 / 0,051 | 1/4 | 0/3 | 0,0011 (0,0011) |
| A batter is preparing to hit the ball at a game. (526446_0) | 2397612 (sai) | catcher (4,147); dirt (0,145); guard (0,113) | −2,260, 45 (−0,843, 21,9) | 0,127, 1 (−0,822, 22,0) | 1 | 1 | 1 | 4,405 / 2,260 | 0/3 | 0/0 | 0,0014 (0,0014) |
| The two people are walking in the snow with their ski equipment. (472772_2) | 2352632 (sai) | people in snow (3,051); people has ski (2,783); tree near building (2,671) | −0,180, 1 (0,006, 1,0) | 0,116, 1 (−0,420, 2,1) | 4 | 1 | 14 | 8,505 / 0,180 | 2/3 | 1/2 | −0,0272 (−0,0272) |
| A snow board sitting upright with a view of a mountain. (237041_4) | 2412723 (sai) | mountain range in distance (2,811); snowboard has writing (2,802); mountain on horizon (2,795) | −0,117, 1 (−0,002, 1,0) | 0,100, 1 (−0,689, 5,5) | > 10 | 1 | 22 | 8,407 / 0,117 | 1/3 | 0/2 | 0,0165 (0,0131) |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | 2360307 (sai) | lady sitting on outside bench (0,676); lady in coat (0,655); trim on umbrella (0,639) | −0,044, 1 (−0,000, 1,0) | −0,110, 1 (−0,246, 2,5) | 7 | > 10 | 13 | 1,970 / 0,044 | 2/4 | 0/3 | −0,0027 (−0,0027) |
