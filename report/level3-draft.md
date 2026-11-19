# Nháp báo cáo: Cấp độ 3, đánh giá định lượng lời giải thích

Nháp để nhóm duyệt rồi ghép vào bản báo cáo chính.
Cùng quy ước với `report/level1-draft.md`: thuật ngữ tiếng Anh giữ nguyên, kèm giải thích lần đầu; mọi con số lấy từ file kết quả trong `experiments/`, đường dẫn ghi ở cuối từng mục và ở mục 12.
Phần này thuộc dòng 27–29 của bảng phân công (đo fidelity, validity, sparsity).
Người soạn: Nhi cùng công cụ AI (Claude).
Bản nháp ngày **2026-11-18**.

> **Số liệu.** Cả hai mô hình đều đo bằng bản đo thứ hai (**v2**, `experiments/xai-metrics-v2/`), gồm cả phần so sánh 13 cách giải thích.
> Bản đo thứ nhất (**v1**, `experiments/xai-metrics/`) giữ nguyên làm tham khảo; mục 3.3 ghi v2 sửa gì so với v1.

## Tóm tắt

Cấp độ 3 yêu cầu đo **fidelity, validity, sparsity** của lời giải thích một cách định lượng, trên ít nhất 10 truy vấn.
Nhóm đo trên **toàn bộ 10 633 câu val**, mỗi mô hình 3 seed, và giữ thêm 13 câu làm bằng chứng chi tiết (mục 7).

Lời giải thích được đánh giá là **E**: các phần của ảnh hạng 1 (node vật thể và bộ ba có nhãn) xếp theo term của chính phép tính điểm (mục 2).
E sinh từ cơ chế xếp hạng, không tốn thêm lượt chấm nào.
Để biết E tốt đến đâu, nhóm so nó với 12 cách giải thích khác trên cùng các câu, gồm các cách giải thích GNN phổ biến (GNNExplainer, PGExplainer, GNN-LRP, SubgraphX, GraphSVX), trong đó **Shapley** làm chuẩn tham chiếu và **occlusion** (bỏ từng phần rồi chấm lại) là cách can thiệp rẻ nhất.

Kết quả chính (val, mean ± std qua 3 seed):

| | `countonly_factual_r3` | `main_r3` | Ngẫu nhiên (countonly / `main_r3`) |
|---|---|---|---|
| Bỏ top-3 của E: ảnh mất hạng 1 | 58,1 ± 4,1% | 34,9 ± 2,8% | 9,4% / 6,5% |
| Chỉ giữ top-3 của E: hạng của ảnh | 2,22 ± 0,74 | 2,16 ± 0,08 | 8,15 / 7,38 |
| Thứ tự của E khớp Shapley (Spearman) | 0,61 (occlusion: 0,45; GraphSVX: 0,75) | 0,46 (occlusion: 0,48; GraphSVX: 0,72) | 0,00 / −0,01 |
| Điểm mất khi bỏ top-3 / tổng term của top-3 | 0,19 | 0,06 | |
| Node ảnh được chú ý nhất cùng loại với node câu (strict / lenient) | 63,8% / 70,0% | 58,3% / 63,8% | |

Bốn kết luận:

1. **E trung thực theo thứ tự**: bỏ hay giữ các phần đầu của E làm thứ hạng đổi xa hơn hẳn ngẫu nhiên.
   Với mô hình chính countonly, thứ tự của E khớp Shapley tốt hơn mọi cách được so, trừ GraphSVX (vốn là một phép ước lượng Shapley); E cũng khớp Shapley tốt hơn nhiều so với PGExplainer, GNN-LRP và SubgraphX.
2. **E không trung thực theo độ lớn**: term của một phần không cho biết bỏ phần đó thì mất bao nhiêu điểm; với bộ ba, gần như không mất gì.
3. **Attention của GAT không phải lời giải thích tốt**: cắt đường truyền tin của các cạnh có attention cao nhất không khác cắt ngẫu nhiên ở `main_r3`, và chỉ nhỉnh hơn ngẫu nhiên một chút ở countonly.
4. **Các cách giải thích GNN phổ biến không vượt E trên mô hình này**: PGExplainer, GNN-LRP và SubgraphX khớp Shapley kém hơn E nhiều; GraphSVX tái tạo được Shapley nhưng tốn khoảng 700 lượt chấm mỗi câu.

Từ kết luận 2, demo thêm một cột kiểm chứng bằng chính mô hình cho mỗi dòng giải thích (mục 8).

## 1. Yêu cầu của đề và phạm vi

Đề (`assignments/Project03_Multimodal_XAI.pdf`, cấp độ 3) yêu cầu hai việc:

- fidelity, validity, sparsity định lượng trên ít nhất 10 truy vấn: **báo cáo này**;
- tự xây MMKG nhỏ tiếng Việt, chạy phương pháp tốt nhất và phân tích chuyển giao: phần của Khang, xem mục 11.

Phạm vi phần đo giải thích:

- **Chỉ dùng val.** Không train, không chấm test, không chỉnh mô hình theo kết quả đo.
- **Mô hình:** `countonly_factual_r3` (mô hình chính, fusion raw) và `main_r3` (cấp 1, fusion z-score), mỗi mô hình 3 seed, `checkpoint_best.pt`.
- **Ảnh được giải thích:** ảnh mô hình xếp hạng 1 cho mỗi câu, dù đúng hay sai; kết quả tách riêng hai nhóm ở phụ lục của `experiments/xai-metrics-v2/REPORT.md`.
- **Câu:** toàn bộ 10 633 câu val cho fidelity, validity, sparsity và đồ thị con.
  Phần so sánh các cách giải thích dùng một mẫu cố định 1 000 câu (rng 2026), giữ các câu có hiệu ứng đồ thị dương (TB 983 câu mỗi seed với countonly, 977 với `main_r3`); GNNExplainer, GraphSVX, SubgraphX và Shapley chạy trên 200 câu đầu của mẫu (TB 196 và 195 câu sau khi lọc); PGExplainer học trên 200 câu val khác, ngoài mẫu so sánh và ngoài 13 câu bằng chứng.

## 2. Lời giải thích được đánh giá

Mô hình chấm điểm một ảnh bằng cách khớp từng phần i của câu với các phần j của ảnh, trong hai kênh: vật thể và bộ ba.
Điểm kênh là Σ_i w_i · Σ_j a_ij · sim_ij, với w_i là trọng số của phần câu, a_ij là attention softmax và sim_ij là độ giống.
Đổi thứ tự tổng thì điểm kênh tách được theo phần của ảnh:

term_j = trọng số kênh × Σ_i w_i · a_ij · sim_ij

**E(q, ảnh)** là danh sách các phần của ảnh xếp theo term_j giảm dần (`src/explain/graph_explainer.matched_pairs`, `image_terms`).

- Với fusion raw (countonly), các term cộng lại đúng bằng phần đồ thị của điểm cuối.
- Với fusion z-score (`main_r3`), điểm cuối chuẩn hoá từng kênh trên 50 ứng viên.
  Bản v2 nhân term của mỗi kênh với đạo hàm chính xác của điểm cuối theo kênh đó (autograd), để node và bộ ba được so trên cùng thang điểm cuối.

E là giải thích **nội tại**: lấy thẳng từ lần xếp hạng, đúng yêu cầu "giải thích sinh từ chính cơ chế xếp hạng" của nhóm.
Bảng giải thích trong demo (`report/level1-draft.md` mục 8.1) trình bày cùng các đại lượng theo từng phần của câu.

## 3. Cách đo

### 3.1. Sửa đồ thị ảnh và chấm lại

Mọi độ đo đều dựa trên việc sửa đồ thị của ảnh hạng 1 rồi chấm lại bằng chính mô hình (`src/explain/xai_metrics.py`):

- **Bỏ một node:** bỏ node cùng mọi cạnh chạm vào nó, nên bỏ luôn các bộ ba chạm vào nó; cạnh kNN không được dựng lại.
- **Bỏ một bộ ba:** bỏ đúng cạnh có nhãn đó.
- **Chỉ giữ top-k:** giữ các phần đó cùng node tổng; một bộ ba được giữ thì giữ hai node đầu–cuối làm "carrier", tức vẫn truyền tin qua GAT nhưng không được chấm ở kênh vật thể.
- **Cắt đường truyền tin:** giữ nguyên node và bộ ba, chỉ cho một số cạnh ngừng truyền tin trong GAT.
- Ảnh sau khi sửa được mã hoá lại bằng cả mô hình, kể cả GAT; 49 ứng viên còn lại giữ nguyên điểm kênh, fusion tính lại trên cả 50.

### 3.2. Mốc so sánh

- **Ngẫu nhiên:** 10 lần bốc, mỗi lần cùng số node và cùng số bộ ba với top-k của E, rng cố định theo (seed, ảnh, câu).
- **Ảnh không sửa:** chấm lại theo đúng đường chấm sau khi sửa.
  Ở cả 3 seed của cả hai mô hình, ảnh này vẫn hạng 1 ở mọi câu; sai số điểm tối đa 2,4e−7 với countonly và 2,0e−5 với `main_r3`.
- **Ảnh rỗng:** bỏ mọi phần; hai kênh nhận giá trị lấp.
  "Hiệu ứng đồ thị" của một câu là điểm của ảnh gốc trừ điểm của ảnh rỗng.
- R@1 tính lại trong lúc đo trùng `metrics.json` của từng checkpoint (script kiểm bằng assert).

### 3.3. Bản v1 và v2

Bản v1 (PR #36) được soát lại và làm lại thành v2 (PR #38); v1 giữ nguyên, kèm mục đính chính ở đầu `experiments/xai-metrics/REPORT.md`.

| Điểm sửa ở v2 | Hệ quả |
|---|---|
| Đưa term của `main_r3` về thang điểm cuối | Đổi E của `main_r3` (mục 6.2); E của countonly giữ như v1 vì fusion raw có hệ số 1 |
| Loại câu có E rỗng khỏi trung bình fidelity, lấy mốc là ảnh không sửa chấm lại | Rất nhỏ với countonly: 31–36 câu mỗi seed (0,3%); Fid+ k = 3 là 58,1% ở v2, 58,0% ở v1 |
| Sparsity đo trên chính E, theo phần của ảnh | Con số "6–8% số phần" của v1 không dùng nữa |
| Validity strict; tiêu chí v1 đổi tên thành lenient | Số lenient của v2 trùng số v1 (70,0% và 63,8%) |
| Khớp nhóm số lượng chỉ chấm mô hình có bảng số lượng | Chỉ báo cho countonly |
| Đồ thị con: cắt đường truyền tin, giữ bộ ba; ngẫu nhiên bốc theo từng node | Dạng v1 (bỏ cả bộ ba) giữ làm tham khảo |
| Mới: k_flip, hạng ảnh đúng, tính cộng, so sánh 13 cách giải thích, Fid_α | |

## 4. Các độ đo

**Fidelity** (độ trung thực: E có chỉ đúng các phần mô hình thật sự dựa vào không), với k = 1, 3, 5:

- **Fid+:** bỏ top-k của E; ghi tỉ lệ ảnh mất hạng 1, hạng mới, và phần hiệu ứng đồ thị bị mất.
- **Fid−:** chỉ giữ top-k của E; ghi các số đó với phần hiệu ứng còn lại.
- **Fid_α** (Zheng và cộng sự, ICLR 2024): với E là 5 phần đầu, bỏ mỗi phần trong E (plus) hoặc ngoài E (minus) với xác suất α = 0,1 hoặc 0,5.
  Đồ thị sau khi sửa vẫn gần đồ thị gốc, nên tránh đánh giá trên đồ thị lệch phân phối; giải thích trung thực thì plus mất nhiều điểm hơn minus.
- **Đường cong deletion / insertion** (k = 0..10): bỏ dần, hoặc chỉ giữ dần, k phần đầu; AUC là trung bình phần hiệu ứng còn lại.
  Deletion thấp là tốt, insertion cao là tốt.
- **Tính cộng (additivity):** E là một phép phân rã điểm, nên tổng term của top-k là "dự đoán" điểm mất khi bỏ top-k; so với điểm mất đo được.

**Validity** (độ hợp lệ: E có đúng nghĩa và có tính phản thực tế không):

- **Phản thực tế (counterfactual):** k_flip là k nhỏ nhất (≤ 10) mà bỏ top-k làm ảnh mất hạng 1; với ảnh hạng 1 sai, ghi hạng của ảnh đúng sau khi bỏ top-k.
- **Ngữ nghĩa:** với mỗi phần của câu, node ảnh có attention cao nhất có cùng loại với nó không.
  - strict: cùng nhãn sau alias VG, hoặc synset của node ảnh trùng hay là hyponym của nghĩa danh từ đầu tiên của từ trong câu; người chỉ khớp với từ chung (person, people...).
  - lenient (tiêu chí v1): cùng nhãn sau alias VG, hoặc cùng loại theo `dedup.same_type` (nên "woman" ~ "man" cũng tính là khớp), hoặc cosine CLIP đóng băng của hai node ≥ 0,955.
  - Cả hai đều lấy "hoặc" của các điều kiện (`scripts/xai_metrics.py`, `LabelMatcher`); riêng alias đã khớp 57,0% dòng của countonly và 51,6% của `main_r3`.
  - Với bộ ba: chủ thể và đối tượng có khớp strict không, vị từ có khớp sau alias VG không.
- **Khớp nhóm số lượng:** với node câu có số lượng nêu rõ, nhóm số lượng (1, 2, 3–4, ≥ 5) của node ảnh có tương thích không (chỉ countonly).

**Sparsity** (độ gọn):

- số phần đầu của E chiếm 80% / 90% khối term dương;
- k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% / 90% hiệu ứng đồ thị;
- số dòng của bảng demo đủ cho 80% biên so với trung bình 50 ứng viên.

**Đồ thị con:** với 3 node đầu của E, lấy các cạnh vào có attention GAT lớn nhất, cắt đường truyền tin của chúng, so với cắt cùng số cạnh vào ngẫu nhiên của chính từng node.

## 5. Mười ba cách giải thích để so sánh

Một con số fidelity đứng một mình khó đọc; nhóm đo thêm 12 cách giải thích khác, xếp **cùng các phần của ảnh** và đo **cùng một cách**.
Danh sách theo các họ phương pháp chính của giải thích GNN (Yuan và cộng sự, TPAMI 2022) và giải thích cho xếp hạng (RankingSHAP / RankSHAP):

| Cách | Loại | Chi phí mỗi câu |
|---|---|---|
| **E** (`model`) | Nội tại, phân rã điểm | 0 lượt chấm thêm |
| random | Mốc dưới | 0 |
| lowest | E đảo ngược | 0 |
| clip | Cosine CLIP giữa câu và từng phần, không dùng mô hình | 0 |
| gradient | Gradient × input theo vector đầu vào của từng phần | 1 lượt lan truyền ngược |
| integrated gradients (IG) | Phân rã hậu nghiệm từ ảnh trung tính (Sundararajan và cộng sự, 2017), 20 bước | 20 lượt lan truyền ngược |
| occlusion | Bỏ từng phần một, điểm mất là độ quan trọng | 1 lượt chấm mỗi phần (TB 43 với countonly, 46 với `main_r3`) |
| GNNExplainer | Mặt nạ mềm trên vector đầu vào, học để giữ điểm (Ying và cộng sự, 2019) | 100 bước tối ưu |
| PGExplainer | Mặt nạ của GNNExplainer do một MLP chung sinh ra từ vector đã mã hoá của từng phần (Luo và cộng sự, 2020), học một lần cho mỗi checkpoint trên 200 câu val khác | 1 lượt mã hoá, sau 42–85 giây học |
| GNN-LRP | Relevance của các walk qua hai lớp GAT (Schnake và cộng sự, 2022), bằng LRP-ε; điểm của một phần là tổng relevance các walk đi qua nó | 2 lượt lan truyền ngược |
| GraphSVX | Surrogate tuyến tính theo trọng số Kernel SHAP, ước lượng giá trị Shapley (Duval và Malliaros, 2021), cùng ngân sách với Shapley | khoảng 700 lượt chấm |
| SubgraphX | Tìm kiếm cây Monte Carlo trên đồ thị con liên thông, thưởng bằng Shapley cục bộ (Yuan và cộng sự, 2021) | TB 5 769 lượt chấm (countonly), 838 (`main_r3`) |
| **Shapley** | Giá trị Shapley lấy mẫu 16 hoán vị, hàm giá trị là biên giữa hạng 1 và hạng 2 | khoảng 700–740 lượt chấm |

Shapley là chuẩn tham chiếu: với mỗi cách khác, đo tương quan Spearman giữa thứ tự của nó và giá trị Shapley, và tỉ lệ trùng top-5.
Bốn cách PGExplainer, GNN-LRP, GraphSVX và SubgraphX vốn viết cho bài toán phân loại node hoặc đồ thị; nhóm chỉnh chúng cho các phần ảnh của reranker, và cách chỉnh ghi ở `src/explain/xai_baselines.py`.
Kiểm tra cài đặt:
- IG thoả completeness với sai số trung bình 0,7% (countonly) và 1,1% (`main_r3`) hiệu ứng;
- Shapley và GraphSVX thoả efficiency với sai số tối đa 1,3e−7 và 4,4e−16;
- GNN-LRP dựng lại đầu ra hai lớp GAT với sai số tối đa 2,1e−6, và tổng relevance của các walk khớp relevance ở đầu ra GAT, chỉ lệch do hằng số ε của quy tắc chia;
- thêm bốn cách mới không làm đổi số nào của 9 cách có trước (chạy lại cả 6 run, mọi số trùng tuyệt đối).

## 6. Kết quả (val, mean ± std qua 3 seed)

R@1 tính lại trong lúc đo: countonly 49,15 / 48,85 / 48,74; `main_r3` 46,67 / 46,56 / 46,21.
Δ điểm của hai mô hình khác thang (countonly: điểm raw, `main_r3`: z-score), nên giữa hai mô hình chỉ so tỉ lệ, hạng và tương quan.

### 6.1. Fidelity

**Bỏ top-k của E (Fid+): tỉ lệ ảnh mất hạng 1.**

| k | countonly: E | Ngẫu nhiên | `main_r3`: E | Ngẫu nhiên |
|---|---|---|---|---|
| 1 | 34,4 ± 4,8% | 4,4% | 20,5 ± 1,8% | 3,0% |
| 3 | 58,1 ± 4,1% | 9,4% | 34,9 ± 2,8% | 6,5% |
| 5 | 68,2 ± 3,0% | 12,9% | 43,6 ± 3,1% | 9,8% |

**Chỉ giữ top-k của E (Fid−): hạng của ảnh, và phần hiệu ứng đồ thị giữ lại.**

| k | countonly: hạng E | Ngẫu nhiên | Hiệu ứng giữ lại, E | `main_r3`: hạng E | Ngẫu nhiên | Hiệu ứng giữ lại, E |
|---|---|---|---|---|---|---|
| 1 | 4,19 ± 0,84 | 10,16 | 45,7% | 3,00 ± 0,03 | 9,19 | 46,1% |
| 3 | 2,22 ± 0,74 | 8,15 | 99,7% | 2,16 ± 0,08 | 7,38 | 64,3% |
| 5 | 1,86 ± 0,54 | 6,95 | 116,6% | 2,17 ± 0,06 | 6,89 | 64,6% |

- Ở cả hai mô hình, E cách xa ngẫu nhiên theo cả hai chiều: bỏ top-3 làm mất hạng 1 gấp 5–6 lần ngẫu nhiên, và chỉ giữ top-3 cho hạng khoảng 2,2 so với 7–8.
- Fid+ của countonly cao hơn `main_r3` vì E của countonly ít bộ ba ở đầu hơn (mục 6.2).
- Với countonly, chỉ giữ top-3 đã giữ đủ hiệu ứng đồ thị, và top-5 vượt quá hiệu ứng gốc (117%): bỏ các phần khớp kém làm điểm tăng, vì attention softmax không còn chia phần cho chúng.
- Fid− của `main_r3` gần như không đổi từ k = 3 lên k = 5: phần thứ 4 và 5 của E phần lớn là bộ ba, mà giữ thêm bộ ba ít đổi điểm.

### 6.2. Kênh bộ ba: nặng ký trong E nhưng bỏ đi ít tác dụng

| | Bộ ba ở top-1 / top-3 / top-5 của E | Bỏ top-1 khi nó là node: mất hạng 1 | Khi nó là bộ ba |
|---|---|---|---|
| countonly | 43,6 / 50,2 / 56,1% | 46,5% (ngẫu nhiên 4,7%) | 19,7% (ngẫu nhiên 4,2%) |
| `main_r3` | 70,5 / 76,4 / 77,6% | 49,3% (ngẫu nhiên 4,4%) | 8,4% (ngẫu nhiên 2,5%) |

- Ở cả hai mô hình, khi bộ ba đứng đầu E, bỏ nó ít khi làm ảnh mất hạng 1.
  Với countonly, chỉ giữ bộ ba đứng đầu cho hạng 3,70, gần như giữ một bộ ba bất kỳ (4,02).
- Lý do phù hợp với số liệu: điểm kênh là trung bình softmax trên các phần của ảnh, và kênh bộ ba dàn trải (top-3 bộ ba chỉ chiếm 23–39% term của kênh, so với 63–68% của top-3 node).
  Bỏ một bộ ba thì attention dồn sang các bộ ba tương tự còn lại, nên điểm gần như không đổi.
- Với `main_r3`, đưa term về thang điểm cuối làm bộ ba lên đầu E ở 70% câu (term thô: 1%), vì z-score khuếch đại kênh có độ phân tán nhỏ.
  Đúng theo đạo hàm, nhưng làm E kém trung thực hơn khi đo bằng can thiệp: Fid+ k = 3 là 34,9%, so với 65,7% khi xếp theo term thô.

### 6.3. Tính cộng: độ lớn của term không phản ánh điểm mất

Bỏ top-k của E (toàn val):

| Mô hình | k | Tổng term top-k (dự đoán) | Điểm mất đo được | Tỉ số | Spearman |
|---|---|---|---|---|---|
| countonly | 1 | 0,05 ± 0,00 | 0,010 ± 0,002 | 0,20 | 0,43 |
| countonly | 3 | 0,11 ± 0,01 | 0,020 ± 0,003 | 0,19 | 0,40 |
| countonly | 5 | 0,14 ± 0,01 | 0,025 ± 0,004 | 0,18 | 0,35 |
| `main_r3` | 1 | 2,31 ± 0,35 | 0,189 ± 0,021 | 0,084 | 0,18 |
| `main_r3` | 3 | 6,12 ± 0,96 | 0,375 ± 0,036 | 0,063 | −0,04 |
| `main_r3` | 5 | 9,22 ± 1,46 | 0,501 ± 0,042 | 0,056 | −0,10 |

- Điểm mất thật chỉ bằng khoảng 19% (countonly) và 6% (`main_r3`) tổng term của top-3.
- Nguyên nhân chính, chung cho cả hai mô hình: term là a_ij · sim_ij, gồm cả mức khớp tuyệt đối; khi bỏ một phần, attention softmax chia lại cho các phần còn lại, nên phần lớn mức khớp đó được các phần còn lại bù lại.
- Với `main_r3`, z-score còn bỏ thêm mức nền chung của 50 ứng viên, nên tổng term gần như không tương quan với điểm mất (Spearman −0,04); với countonly, tương quan vẫn vừa phải (0,40).
  Cách giải thích này phù hợp với số liệu, nhưng chưa kiểm riêng.
- Trên cùng các câu so sánh (mục 6.7), ở k = 3:

  | Cách | countonly: tỉ số | Spearman | `main_r3`: tỉ số | Spearman |
  |---|---|---|---|---|
  | E | 0,19 | 0,44 | 0,058 | −0,01 |
  | occlusion | 0,97 | 0,91 | 1,15 | 0,89 |
  | IG | 1,32 | 0,81 | 1,43 | 0,82 |

  Occlusion và IG cộng tính tốt hơn E nhiều ở cả hai mô hình.

### 6.4. Validity

**Phản thực tế:**

| | countonly | `main_r3` |
|---|---|---|
| Bỏ tối đa 10 phần đầu của E làm ảnh mất hạng 1 | 80,2 ± 1,5% | 60,3 ± 2,9% |
| Như trên, ảnh hạng 1 sai / đúng | 91,8% / 68,0% | 74,4% / 44,1% |
| Trung vị k_flip | 2 | 3 |
| Ảnh hạng 1 sai: bỏ top-5, ảnh đúng lên hạng / lên hạng 1 | 51,9% / 20,6% | 30,7% / 13,1% |
| Ảnh hạng 1 sai: hạng TB của ảnh đúng, trước → sau bỏ top-5 | 8,16 → 7,64 | 8,52 → 8,23 |

- Với ảnh hạng 1 sai, E chỉ ra phần đã đẩy ảnh sai lên: bỏ top-5 làm ảnh đúng lên hạng ở khoảng nửa số câu với countonly.
  Nhưng bỏ chúng thường chưa đủ để ảnh đúng thắng, và hạng trung bình của ảnh đúng chỉ đổi ít.

**Ngữ nghĩa** (TB mỗi seed: countonly 29 687 dòng vật thể và 18 532 dòng bộ ba; `main_r3` 32 382 và 17 311):

| Tiêu chí | countonly | `main_r3` |
|---|---|---|
| Vật thể, strict | 63,8 ± 0,9% | 58,3 ± 0,3% |
| Vật thể, lenient | 70,0 ± 1,0% | 63,8 ± 0,2% |
| Bộ ba, chủ thể / đối tượng / vị từ khớp | 35,8 / 25,8 / 10,5% | 30,1 / 28,8 / 20,7% |
| Bộ ba, khớp đủ cả ba | 2,9 ± 0,6% | 5,4 ± 0,2% |
| Khớp nhóm số lượng (5 857 dòng có số nêu rõ) | 66,7 ± 0,8% | không chấm |

- Node ảnh được chú ý nhất cùng loại với node câu ở 64% (countonly) và 58% (`main_r3`) dòng theo tiêu chí strict.
  Tiêu chí lenient cộng thêm khoảng 6 điểm, từ hai điều kiện strict không có: cùng loại theo `same_type` (ví dụ "woman" ~ "man") và cosine CLIP ≥ 0,955.
  Bảng hiện chưa tách phần đóng góp riêng của từng điều kiện.
- Với countonly, 67% dòng vật thể có số lượng trỏ tới node ảnh cùng nhóm số lượng.
- Dòng bộ ba hầu như không khớp nghĩa: chỉ 3–5% khớp đủ.
  Countonly khớp vị từ kém hơn (10,5% so với 20,7%); một giả thuyết chưa kiểm là query graph của countonly do parser FACTUAL sinh ra, với vị từ khác từ vựng VG nhiều hơn.

### 6.5. Sparsity

| Đo | countonly | `main_r3` |
|---|---|---|
| k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% hiệu ứng | 2,78 ± 0,27, đạt ở 94,5% câu | 2,65 ± 0,14, đạt ở 55,7% câu |
| Số phần đầu của E cho 80% khối term | 15,0 phần (35% số phần) | 15,4 phần (33% số phần) |
| Số dòng demo cho 80% biên so với trung bình top-50 | 2,52 | 1,95 |

- Theo hiệu ứng, E gọn: với countonly, khoảng 3 phần đầu đã giữ 80% hiệu ứng đồ thị, và mức này đạt được ở 94,5% câu; với `main_r3`, 2–3 phần đầu làm được điều đó ở hơn nửa số câu.
- Theo khối term thì E dàn trải (15 phần cho 80%), do nhiều phần có term gần bằng nhau; đây lại là hệ quả của mục 6.3.

### 6.6. Đồ thị con theo attention GAT

Các câu mà số cạnh vào lớn hơn số cạnh bỏ: 10 481 câu mỗi seed với countonly (bỏ 9,0 trong 21,1 cạnh vào), 2 803 với `main_r3` (bỏ 5,3 trong 9,5).

| Cắt đường truyền tin của các cạnh vào | countonly: Δ điểm | Mất hạng 1 | `main_r3`: Δ điểm | Mất hạng 1 |
|---|---|---|---|---|
| Top theo attention | +0,0025 ± 0,0006 | 3,5 ± 1,5% | −0,0011 ± 0,0007 | 1,0 ± 0,6% |
| Ngẫu nhiên | +0,0021 ± 0,0003 | 1,8 ± 0,9% | −0,0010 ± 0,0005 | 0,7 ± 0,4% |

- Với `main_r3`, cắt đường truyền tin của các cạnh có attention cao nhất không khác cắt ngẫu nhiên.
- Với countonly, cắt các cạnh attention cao làm mất hạng 1 nhiều gấp khoảng 2 lần cắt ngẫu nhiên, ở cả 3 seed (2,7 / 5,2 / 2,6% so với 1,3 / 2,8 / 1,3%).
  Nhưng hiệu ứng nhỏ: Δ điểm trung bình như cắt ngẫu nhiên (cả hai đều làm điểm tăng nhẹ), và nhỏ hơn nhiều so với bỏ phần (mục 6.1).
- Ablation bỏ cạnh lúc train làm R@1 giảm 2,39 (`experiments/improve/no-edges/`): cạnh có ích ở tổng thể, nhưng tác dụng trải trên nhiều cạnh, không nằm ở vài cạnh có attention cao.

### 6.7. So sánh 13 cách giải thích

**countonly** (983 câu mỗi seed; Shapley, GraphSVX, SubgraphX và GNNExplainer 196 câu; thời gian trên CPU):

| Cách | Giây / câu | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | k_flip đạt | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Khớp Shapley (Spearman) | Top-5 trùng Shapley |
|---|---|---|---|---|---|---|---|
| Shapley | 0,28 | −0,19 | 1,54 | 88,1% | 0,081 / −0,051 | chuẩn | chuẩn |
| GraphSVX | 0,28 | **−0,24** | **1,56** | **89,3%** | 0,084 / −0,055 | **0,75** | **73,2%** |
| IG | 0,03 | −0,22 | 1,24 | 86,2% | 0,087 / −0,045 | 0,34 | 60,4% |
| occlusion | 0,03 | −0,01 | 1,45 | 78,1% | **0,102 / −0,065** | 0,45 | 55,3% |
| **E** | 0 | 0,07 | 1,26 | 79,8% | 0,076 / −0,032 | 0,61 | 56,5% |
| GNNExplainer | 0,37 | 0,11 | 1,06 | 83,5% | 0,070 / −0,043 | 0,37 | 57,1% |
| GNN-LRP | 0,01 | 0,11 | 0,88 | 76,8% | 0,063 / −0,025 | 0,08 | 47,4% |
| PGExplainer | 0,002 | 0,21 | 0,64 | 77,2% | 0,051 / −0,009 | −0,03 | 42,8% |
| clip | 0,000 | 0,39 | 0,90 | 68,6% | 0,049 / −0,006 | 0,42 | 40,3% |
| SubgraphX | 2,29 | 0,39 | 0,95 | 59,8% | 0,051 / −0,019 | 0,02 | 38,1% |
| gradient | 0,004 | 0,76 | 0,24 | 39,4% | 0,020 / 0,022 | −0,21 | 24,0% |
| random | 0 | 0,89 | 0,21 | 28,5% | 0,011 / 0,028 | 0,00 | 16,9% |
| lowest (E đảo ngược) | 0 | 1,06 | −1,31 | 11,0% | −0,003 / 0,044 | −0,61 | 3,4% |

Kiểm định Wilcoxon ghép cặp với E trên deletion AUC: mọi cách khác đều khác E với p < 0,05 ở cả 3 seed (p lớn nhất: occlusion 0,042, GNNExplainer 0,030), trừ PGExplainer (p lớn nhất 0,70).

**`main_r3`** (977 câu mỗi seed; Shapley, GraphSVX, SubgraphX và GNNExplainer 195 câu):

| Cách | Giây / câu | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | k_flip đạt | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Khớp Shapley (Spearman) | Top-5 trùng Shapley |
|---|---|---|---|---|---|---|---|
| Shapley | 0,14 | −0,68 | **1,53** | 85,3% | **0,118 / −0,046** | chuẩn | chuẩn |
| GraphSVX | 0,14 | **−0,71** | 1,50 | **86,1%** | 0,114 / −0,047 | **0,72** | **71,9%** |
| IG | 0,01 | −0,45 | 0,65 | 81,9% | 0,090 / −0,024 | 0,29 | 62,9% |
| occlusion | 0,02 | −0,24 | 1,23 | 74,8% | 0,106 / −0,046 | 0,48 | 58,5% |
| GNNExplainer | 0,27 | −0,17 | −0,34 | 81,8% | 0,091 / 0,000 | 0,26 | 56,7% |
| GNN-LRP | 0,007 | 0,04 | 0,39 | 69,7% | 0,068 / 0,001 | 0,07 | 39,5% |
| PGExplainer | 0,001 | 0,08 | 0,20 | 70,8% | 0,050 / 0,018 | 0,07 | 43,4% |
| **E** | 0 | 0,24 | 0,76 | 57,7% | 0,050 / 0,016 | 0,46 | 31,8% |
| SubgraphX | 0,11 | 0,28 | 0,62 | 49,5% | 0,083 / −0,009 | 0,11 | 33,9% |
| clip | 0,000 | 0,28 | 0,23 | 61,8% | 0,047 / 0,021 | 0,35 | 35,4% |
| gradient | 0,003 | 0,83 | −1,04 | 27,9% | 0,012 / 0,057 | −0,13 | 17,8% |
| random | 0 | 0,84 | −0,62 | 25,5% | 0,013 / 0,053 | −0,01 | 14,3% |
| lowest (E đảo ngược) | 0 | 1,01 | −2,30 | 10,9% | 0,002 / 0,068 | −0,46 | 2,1% |

Kiểm định Wilcoxon ghép cặp với E trên deletion AUC: mọi cách khác đều khác E với p < 1e−7 ở cả 3 seed, trừ clip (p lớn nhất 0,96) và SubgraphX (0,51).

- **Shapley, GraphSVX, occlusion và IG dẫn đầu ở deletion AUC và Fid_α, ở cả hai mô hình**: bỏ dần theo thứ tự của chúng làm điểm xuống dưới cả mức ảnh rỗng (deletion AUC âm hoặc gần 0), và Fid_α+ > Fid_α−.
  Chúng không dẫn đầu mọi trục: với `main_r3`, insertion AUC của IG (0,65) và độ khớp với Shapley (0,29) đều thấp hơn E (0,76 và 0,46).
- **GraphSVX tái tạo Shapley với cùng ngân sách** (khoảng 700 lượt chấm mỗi câu): Spearman 0,75 và 0,72, top-5 trùng 72–73%, và các đường cong ngang hoặc nhỉnh hơn chính Shapley lấy mẫu hoán vị.
- **Với countonly, E gần sát nhóm dẫn đầu.**
  Thứ tự của E khớp Shapley tốt hơn mọi cách không ước lượng Shapley (0,61, so với 0,45 của occlusion và 0,34 của IG); insertion AUC ngang IG (1,26 so với 1,24); Fid_α+ > Fid_α− như nhóm dẫn đầu.
  E chỉ kém rõ ở deletion AUC (0,07 so với −0,01 của occlusion), mà không tốn lượt chấm nào.
- **Với `main_r3`, E đứng giữa**: tốt hơn hẳn ngẫu nhiên và gradient, ngang clip trên deletion, tốt hơn clip trên insertion.
  Thứ tự tổng thể của E khớp Shapley gần bằng occlusion (0,46 so với 0,48), nhưng top-5 của E chỉ trùng 32% với Shapley, đúng ở chỗ bộ ba chen vào đầu E (mục 6.2).
- **PGExplainer, GNN-LRP và SubgraphX khớp Shapley kém** (Spearman ≤ 0,11 ở cả hai mô hình, so với 0,46–0,61 của E).
  - Với countonly, PGExplainer và GNN-LRP kém E ở deletion, insertion và độ khớp Shapley.
  - Với `main_r3`, hai cách này hơn E ở deletion AUC (0,04 và 0,08 so với 0,24) và k_flip (70% và 71% so với 58%), nhưng kém E nhiều ở insertion AUC (0,39 và 0,20 so với 0,76).
  - GNN-LRP vẫn tốt hơn hẳn gradient × input, cách cùng dựa trên phép tuyến tính hoá cục bộ.
  - SubgraphX có k_flip thấp nhất trong bốn cách mới (60% và 50%) và tốn nhiều lượt chấm nhất với countonly; một lý do có thể, chưa kiểm riêng, là nó chọn cả một đồ thị con liên thông, trong khi các độ đo ở đây bỏ hoặc giữ từng phần một.
- **lowest** (E đảo ngược) kém nhất trên mọi trục ở cả hai mô hình, đúng như mong đợi của một phép kiểm tra: thứ tự của E mang thông tin thật.
- **Gradient × input gần như ngẫu nhiên** ở cả hai mô hình: đạo hàm cục bộ không nói gì về việc bỏ hẳn một phần.

## 7. Bằng chứng trên 13 truy vấn

Đề yêu cầu ít nhất 10 truy vấn; nhóm dùng 13 câu đã phân tích ở `report/level1-draft.md` mục 8.3 (10 ca và 3 ví dụ so với cấp 1).
Hai bảng dưới là seed 0 của mỗi mô hình; ảnh được giải thích là ảnh hạng 1 của mô hình đó, nên hai mô hình có thể giải thích hai ảnh khác nhau.
Cột "Bỏ top-3" và "Chỉ giữ top-3" ghi hạng của ảnh, trong ngoặc là trung bình 10 lần chọn ngẫu nhiên.
Chi tiết (top-5 của E, từng dòng vật thể và kết quả khớp nghĩa, Fid với k = 1, 3, 5, đường cong, đồ thị con) ở `experiments/xai-metrics-v2/cases.json`.

**countonly, seed 0:**

| Câu (caption_id) | Ảnh hạng 1 | E top-3 | Bỏ top-3: hạng (ngẫu nhiên) | Chỉ giữ top-3: hạng (ngẫu nhiên) | k_flip | Vật thể khớp strict |
|---|---|---|---|---|---|---|
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | 2377477 (đúng) | table; room; stove | 4 (1,0) | 1 (6,7) | 1 | 1/3 |
| A cow standing along side of a large river. (422698_1) | 2351201 (đúng) | cow standing ear; plant along water side; cow standing nose | 1 (1,0) | 1 (1,0) | 4 | 2/2 |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | 2329851 (đúng) | pen; trainer observing dog; dog | 2 (1,3) | 2 (6,0) | 1 | 3/5 |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | 2402283 (đúng) | donut; bite; person holding donut | 4 (1,4) | 1 (4,7) | 1 | 2/3 |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | 1159610 (đúng) | dog; umbrella; umbrella | 22 (1,8) | 1 (14,1) | 1 | 4/5 |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | 2366682 (sai) | banana in plate; banana in paper plates; banana in paper plates | 1 (1,0) | 1 (1,0) | 9 | 2/2 |
| A train riding along the tracks along a bank of dirt. (279933_1) | 2357910 (sai) | windshield wiper on train; gravel underneath track; logo on train | 2 (1,3) | 3 (3,1) | 1 | 2/3 |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | 2353248 (sai) | bus; street; bus has bumper | 9 (1,5) | 1 (7,4) | 1 | 2/2 |
| Two men standing in a living room playing video game with remotes. (34471_0) | 2395602 (sai) | wii controller; man wearing wrist band; man wearing short | 2 (1,1) | 1 (6,0) | 1 | 1/4 |
| A batter is preparing to hit the ball at a game. (526446_0) | 2417935 (sai) | helmet has black/face mask; woman swinging bat; black bar on helmet | 1 (1,0) | 44 (44,4) | 4 | 2/2 |
| The two people are walking in the snow with their ski equipment. (472772_2) | 2384371 (đúng) | person carrying ski; woman; person | 3 (1,0) | 1 (14,1) | 3 | 1/3 |
| A snow board sitting upright with a view of a mountain. (237041_4) | 2320074 (đúng) | logo on snowboard; design on snowboard; snowboard | 2 (1,2) | 3 (4,1) | 3 | 1/2 |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | 2385674 (đúng) | umbrella; rain; table | 8 (2,5) | 1 (8,9) | 1 | 2/4 |

**`main_r3`, seed 0:**

| Câu (caption_id) | Ảnh hạng 1 | E top-3 | Bỏ top-3: hạng (ngẫu nhiên) | Chỉ giữ top-3: hạng (ngẫu nhiên) | k_flip | Vật thể khớp strict |
|---|---|---|---|---|---|---|
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | 2377477 (đúng) | room with stove; wood under stove; stove burning wood | 1 (1,0) | 1 (3,2) | 7 | 1/3 |
| A cow standing along side of a large river. (422698_1) | 2351201 (đúng) | moss on water; river surrounded by plant; shade area along edge of water | 1 (1,0) | 11 (12,2) | > 10 | 2/3 |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | 2329851 (đúng) | pen of sheep; pen of sheep; pen of sheep | 1 (1,0) | 2 (11,7) | 5 | 2/4 |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | 2402283 (đúng) | donut; bite; chocolate cream | 11 (2,4) | 1 (8,0) | 1 | 1/2 |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | 1159610 (đúng) | frame has frame; man has pant; woman wearing pant | 1 (1,0) | 2 (2,0) | 7 | 5/6 |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | 2366682 (sai) | banana in plate; banana in plate; banana in plate | 1 (1,0) | 2 (2,8) | 9 | 2/3 |
| A train riding along the tracks along a bank of dirt. (279933_1) | 2378096 (sai) | cfv under headlight; headlight on train; grass on ground | 1 (1,0) | 4 (10,7) | 8 | 2/3 |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | 2355061 (đúng) | bus; bus; bus | 3 (1,0) | 3 (5,8) | 3 | 1/2 |
| Two men standing in a living room playing video game with remotes. (34471_0) | 2395602 (sai) | man wearing shoe; man has leg; writing on jean | 1 (1,0) | 1 (2,2) | > 10 | 1/4 |
| A batter is preparing to hit the ball at a game. (526446_0) | 2397612 (sai) | catcher; dirt; guard | 45 (21,9) | 1 (22,0) | 1 | 0/3 |
| The two people are walking in the snow with their ski equipment. (472772_2) | 2352632 (sai) | people in snow; people has ski; tree near building | 1 (1,0) | 1 (2,1) | 4 | 2/3 |
| A snow board sitting upright with a view of a mountain. (237041_4) | 2412723 (sai) | mountain range in distance; snowboard has writing; mountain on horizon | 1 (1,0) | 1 (5,5) | > 10 | 1/3 |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | 2360307 (sai) | lady sitting on outside bench; lady in coat; trim on umbrella | 1 (1,0) | 1 (2,5) | 7 | 2/4 |

Đọc bảng:

- **Khi E gồm node, E vừa trung thực vừa đủ.**
  Với countonly, ba phần đầu của E có ít nhất một node ở 9/13 câu; bỏ top-3 làm ảnh mất hạng 1 ở 10/13 câu, và k_flip = 1 ở 8/13 câu.
  Câu 574350_4 (chó và ô) là ví dụ rõ nhất: E của countonly là dog, umbrella, umbrella; bỏ ba node này đưa ảnh từ hạng 1 xuống hạng 22 (ngẫu nhiên: 1,8), còn chỉ giữ chúng thì ảnh vẫn hạng 1 (ngẫu nhiên: 14,1).
  Với `main_r3`, câu 526446_0 cho cùng kết luận: E là catcher, dirt, guard; bỏ ba node này làm ảnh sai rơi xuống hạng 45 (ngẫu nhiên: 21,9).
- **Khi E gồm bộ ba, bỏ đi gần như không đổi gì.**
  Với `main_r3`, ba phần đầu của E đều là bộ ba ở 10/13 câu, và ở cả 10 câu đó, bỏ cả ba thì ảnh vẫn hạng 1.
  Cùng câu 574350_4, E của `main_r3` là "frame has frame", "man has pant", "woman wearing pant", không liên quan tới câu, và bỏ cả ba không đổi hạng.
  Với countonly, ba phần đầu của E toàn là bộ ba ở 4/13 câu, và ba trong bốn câu đó (422698_1, 1153_3, 526446_0) là ba câu duy nhất mà bỏ top-3 không đổi hạng.
- **Chỉ giữ top-3, ảnh vẫn ở hạng 1–2 ở 10/13 câu với cả hai mô hình**, kể cả khi top-3 là bộ ba.
  Điều này phù hợp với việc điểm kênh là trung bình softmax: giữ ít phần thì cũng bỏ đi các phần khớp kém đang làm loãng trung bình.
- Các ca này khớp với số trên toàn val ở mục 6: E đúng ở chỗ phần nào mang điểm, nhưng các bộ ba đứng đầu E không phải là thứ quyết định thứ hạng.

## 8. Áp dụng vào demo

Từ mục 6.2 và 6.3, cột "% điểm cuối" của bảng giải thích trong demo dễ bị đọc nhầm là "bỏ phần này thì mất chừng ấy điểm".
Demo nay có thêm cột **Bỏ phần này** (`app/demo_app.py`, `src/service/graph_mode.py`):

- với mỗi dòng khớp, ảnh được chấm lại bằng chính mô hình sau khi bỏ phần khớp đầu tiên của dòng đó, 49 ảnh khác giữ nguyên;
- ô ghi thay đổi của điểm cuối và hạng mới, in đậm khi hạng đổi;
- hai ghi chú dưới bảng nói rõ % điểm cuối chỉ là cách chia điểm, không phải mức điểm mất khi bỏ phần.

Đây là phép kiểm chứng bằng chính mô hình (occlusion của phần đang hiển thị), không phải một lời giải thích tìm sau khi đã có kết quả; các dòng giải thích vẫn lấy từ lần xếp hạng.
Ví dụ trên val với `main_r3_seed0`: câu "A batter is preparing to hit the ball at a game.", bỏ node `catcher` đưa ảnh từ hạng 1 xuống hạng 43 (khớp đường cong deletion ở k = 1 trong `experiments/xai-metrics-v2/cases.json`); câu về chó và ô, bỏ các bộ ba khớp nhiều % nhất không đổi hạng.

## 9. Kết luận

1. **E trung thực theo thứ tự.**
   Bỏ top-3 của E làm mất hạng 1 ở 58% câu (countonly) và 35% câu (`main_r3`), so với 6–9% khi bỏ ngẫu nhiên.
   Với countonly, thứ tự của E khớp Shapley tốt hơn mọi cách không ước lượng Shapley (Spearman 0,61; chỉ GraphSVX khớp hơn, 0,75), và chỉ giữ top-3 của E đã giữ đủ hiệu ứng đồ thị.
2. **E không trung thực theo độ lớn**, nhất là ở kênh bộ ba.
   Điểm mất khi bỏ top-3 chỉ bằng 19% (countonly) và 6% (`main_r3`) tổng term của chúng; bộ ba đứng đầu E nhưng bỏ đi ít khi đổi hạng.
   Nên trình bày E như một thứ tự, và kiểm chứng bằng cột "Bỏ phần này" của demo.
3. **Nếu cần lời giải thích trung thực nhất, dùng occlusion hoặc IG** làm kiểm chứng: occlusion cộng tính tốt nhất và có Fid_α tốt, chỉ tốn khoảng 43–46 lượt chấm mỗi câu; IG có deletion AUC tốt và thoả completeness.
   Shapley là chuẩn tốt nhất nhưng tốn khoảng 700–740 lượt chấm mỗi câu; GraphSVX với cùng ngân sách cho kết quả ngang Shapley.
   Với countonly, E đã gần bằng các cách này mà không tốn thêm lượt chấm nào.
4. **Validity ngữ nghĩa vừa phải ở node, kém ở bộ ba:** 64% (countonly) và 58% (`main_r3`) dòng vật thể khớp strict, chỉ 3–5% dòng bộ ba khớp đủ.
   Với countonly, 67% dòng vật thể có số lượng trỏ tới node ảnh cùng nhóm số lượng.
5. **Sparsity theo hiệu ứng tốt:** với countonly, khoảng 3 phần đầu của E giữ 80% hiệu ứng đồ thị ở 94,5% câu; với `main_r3`, 2–3 phần đầu làm được điều đó ở hơn nửa số câu.
6. **Attention GAT không phải lời giải thích tốt:** ở `main_r3`, cắt đường truyền tin của các cạnh có attention cao nhất không khác cắt ngẫu nhiên.
   Ở countonly, cách cắt đó làm mất hạng 1 nhiều gấp khoảng 2 lần ngẫu nhiên, nhưng chỉ ở 3,5% câu.
7. **Các cách giải thích GNN phổ biến không vượt E trên mô hình này**, trừ GraphSVX vốn là một phép ước lượng Shapley.
   PGExplainer, GNN-LRP và SubgraphX khớp Shapley kém hơn E nhiều (Spearman ≤ 0,11 so với 0,46–0,61); với `main_r3`, PGExplainer và GNN-LRP hơn E ở deletion AUC và k_flip nhưng kém E nhiều ở insertion AUC.

## 10. Hạn chế

- So sánh các cách giải thích dùng 1 000 câu; GNNExplainer, GraphSVX, SubgraphX và Shapley chỉ khoảng 195 câu, std qua seed của các cách này lớn hơn.
- PGExplainer, GNN-LRP, GraphSVX và SubgraphX được chỉnh cho reranker; kết quả phụ thuộc cách chỉnh và ngân sách đã chọn (PGExplainer học trên 200 câu và dùng lại hệ số phạt của GNNExplainer; SubgraphX 10 rollout, 10 mẫu), không phải kết quả của bản cài gốc trên tác vụ gốc.
- Ảnh trung tính của IG và GNNExplainer là vector trung bình của từng loại phần; chọn mốc khác có thể đổi kết quả của hai cách này.
- Δ điểm và tổng term của hai mô hình khác thang (raw và z-score), nên giữa hai mô hình chỉ so được tỉ lệ, hạng và tương quan.
- Validity ngữ nghĩa dựa trên WordNet và alias VG; nhãn VG thiếu hoặc khác từ vựng với caption COCO (`report/level1-draft.md` mục 8.4) làm tỉ lệ khớp thấp hơn thực tế.
- Hai giải thích chưa kiểm riêng: phần nền của term dưới z-score (mục 6.3) và lý do countonly khớp vị từ kém (mục 6.4).
- Bỏ node cũng bỏ các bộ ba chạm vào nó, nên hiệu ứng của một node đã gồm phần của các bộ ba đó.
- Chỉ đo trên ảnh hạng 1, chỉ trên val.

## 11. MMKG tiếng Việt và phân tích chuyển giao

Phần của Khang (dòng 30–32 của bảng phân công): bộ dữ liệu ở `vietnamese-mmkg/` (data card `vietnamese-mmkg/DATA_CARD.md`), chuyển giao ở `scripts/eval_vnmmkg_transfer.py`.
Nội dung mục này do Khang viết.

## 12. Nguồn số liệu

| Nội dung | File |
|---|---|
| Cả hai mô hình, mọi độ đo v2, 3 seed | `experiments/xai-metrics-v2/metrics.json`, `summary.json`, bảng đầy đủ theo nhóm đúng / sai ở `experiments/xai-metrics-v2/REPORT.md` |
| 13 câu bằng chứng, seed 0 của mỗi mô hình | `experiments/xai-metrics-v2/cases.json` |
| Số v1 (tham khảo, đọc kèm mục đính chính ở đầu) | `experiments/xai-metrics/metrics.json`, `summary.json`, `REPORT.md` |
| Script đo | `scripts/xai_metrics.py` (v2), `scripts/xai_metrics_v1.py` (v1), `src/explain/xai_metrics.py` |
| Script tổng hợp bảng | `scripts/summarize_xai_metrics.py` (v2), `scripts/summarize_xai_metrics_v1.py` (v1) |
| Lệnh chạy lại | `experiments/xai-metrics-v2/run.sh`, `experiments/xai-metrics/run.sh` |

**Khai báo công cụ AI:** code đo, phần chạy thí nghiệm và bản nháp này được viết cùng Claude (Anthropic); Nhi duyệt định nghĩa các độ đo, kết quả và kết luận.
Mọi con số lấy từ các file ở bảng trên.
