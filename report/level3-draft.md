# Nháp báo cáo: Cấp độ 3, đánh giá định lượng lời giải thích

Nháp để nhóm duyệt rồi ghép vào bản báo cáo chính.
Cùng quy ước với `report/level1-draft.md`: thuật ngữ tiếng Anh giữ nguyên, kèm giải thích lần đầu; mọi con số lấy từ file kết quả trong `experiments/`, đường dẫn ghi ở cuối từng mục và ở mục 12.
Phần này thuộc dòng 27–29 của bảng phân công (đo fidelity, validity, sparsity).
Người soạn: Nhi cùng công cụ AI (Claude).
Bản nháp ngày **2026-11-18**.

> **Hai phiên bản số liệu.**
> - Mô hình cấp 1 `main_r3`: đo đầy đủ bằng bản đo thứ hai (**v2**, `experiments/xai-metrics-v2/`), gồm cả phần so sánh 9 cách giải thích.
> - Mô hình chính `countonly_factual_r3`: hiện chỉ có số của bản đo thứ nhất (**v1**, `experiments/xai-metrics/`), chạy trên GPU ngày 16/11.
>   Máy chạy v2 chưa có dữ liệu count_v1 và checkpoint của countonly.
>   Mục 6 chỉ dùng những số v1 vẫn đúng sau khi soát lại (mục 3.3); các ô ghi `__,__` sẽ điền khi chạy v2 cho countonly bằng `bash experiments/xai-metrics-v2/run.sh countonly_factual_r3`.
>   Không điền số bịa vào các ô đó.

## Tóm tắt

Cấp độ 3 yêu cầu đo **fidelity, validity, sparsity** của lời giải thích một cách định lượng, trên ít nhất 10 truy vấn.
Nhóm đo trên **toàn bộ 10 633 câu val**, mỗi mô hình 3 seed, và giữ thêm 13 câu làm bằng chứng chi tiết (mục 7).

Lời giải thích được đánh giá là **E**: các phần của ảnh hạng 1 (node vật thể và bộ ba có nhãn) xếp theo term của chính phép tính điểm (mục 2).
E sinh từ cơ chế xếp hạng, không tốn thêm lượt chấm nào.
Để biết E tốt đến đâu, nhóm so nó với 8 cách giải thích khác trên cùng các câu, trong đó **Shapley** làm chuẩn tham chiếu và **occlusion** (bỏ từng phần rồi chấm lại) là cách can thiệp rẻ nhất.

Kết quả chính (val, mean ± std qua 3 seed):

| | `main_r3` (v2) | `countonly_factual_r3` (v1) | Ngẫu nhiên (`main_r3` / countonly) |
|---|---|---|---|
| Bỏ top-3 của E: ảnh mất hạng 1 | 34,9 ± 2,8% | 58,0 ± 4,1% | 6,5% / 9,4% |
| Chỉ giữ top-3 của E: hạng của ảnh | 2,16 ± 0,08 | 2,22 ± 0,74 | 7,38 / 8,13 |
| Thứ tự của E khớp Shapley (Spearman) | 0,46 (occlusion: 0,48) | __,__ | −0,01 / __,__ |
| Điểm mất khi bỏ top-3 / tổng term của top-3 | 0,06 | __,__ | |
| Node ảnh được chú ý nhất cùng loại với node câu (strict / lenient) | 58,3% / 63,8% | __,__ / 70,0% | |

Ba kết luận:

1. **E trung thực theo thứ tự**: bỏ hay giữ các phần đầu của E làm thứ hạng đổi xa hơn hẳn ngẫu nhiên, và thứ tự của E khớp Shapley gần bằng occlusion.
2. **E không trung thực theo độ lớn**: term của một phần không cho biết bỏ phần đó thì mất bao nhiêu điểm; với bộ ba, gần như không mất gì.
3. **Attention của GAT không phải lời giải thích**: cắt đường truyền tin của các cạnh có attention cao nhất không khác cắt ngẫu nhiên.

Từ kết luận 2, demo thêm một cột kiểm chứng bằng chính mô hình cho mỗi dòng giải thích (mục 8).

## 1. Yêu cầu của đề và phạm vi

Đề (`assignments/Project03_Multimodal_XAI.pdf`, cấp độ 3) yêu cầu hai việc:

- fidelity, validity, sparsity định lượng trên ít nhất 10 truy vấn: **báo cáo này**;
- tự xây MMKG nhỏ tiếng Việt, chạy phương pháp tốt nhất và phân tích chuyển giao: phần của Khang, xem mục 11.

Phạm vi phần đo giải thích:

- **Chỉ dùng val.** Không train, không chấm test, không chỉnh mô hình theo kết quả đo.
- **Mô hình:** `main_r3` (cấp 1, fusion z-score) và `countonly_factual_r3` (mô hình chính, fusion raw), mỗi mô hình 3 seed, `checkpoint_best.pt`.
- **Ảnh được giải thích:** ảnh mô hình xếp hạng 1 cho mỗi câu, dù đúng hay sai; kết quả tách riêng hai nhóm ở phụ lục của từng báo cáo thí nghiệm.
- **Câu:** toàn bộ 10 633 câu val cho fidelity, validity, sparsity và đồ thị con.
  Phần so sánh các cách giải thích dùng một mẫu cố định 1 000 câu (rng 2026), giữ các câu có hiệu ứng đồ thị dương (972–980 câu mỗi seed); GNNExplainer và Shapley chạy trên 200 câu đầu của mẫu (192–197 câu sau khi lọc).

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
  Ở cả 3 seed của `main_r3`, ảnh này vẫn hạng 1 ở mọi câu, sai số điểm tối đa 2,0e−5.
- **Ảnh rỗng:** bỏ mọi phần; hai kênh nhận giá trị lấp.
  "Hiệu ứng đồ thị" của một câu là điểm của ảnh gốc trừ điểm của ảnh rỗng.
- R@1 tính lại trong lúc đo trùng `metrics.json` của từng checkpoint (script kiểm bằng assert).

### 3.3. Bản v1 và v2

Bản v1 (PR #36) được soát lại và làm lại thành v2 (PR #38); v1 giữ nguyên, kèm mục đính chính.
Các điểm v2 sửa, và hệ quả cho số của countonly hiện lấy từ v1:

| Điểm sửa ở v2 | Ảnh hưởng tới số v1 của countonly |
|---|---|
| Đưa term của `main_r3` về thang điểm cuối | Không: fusion raw có hệ số 1, E như nhau |
| Loại câu có E rỗng khỏi trung bình fidelity, lấy mốc là ảnh không sửa chấm lại | Rất nhỏ: 31–36 câu mỗi seed (0,3%), sai số chấm lại 1,8e−7 |
| Sparsity đo trên chính E, theo phần của ảnh | Con số "8% số phần" của v1 không dùng; số dòng demo cho 80% biên vẫn dùng được |
| Validity strict; tiêu chí v1 đổi tên thành lenient | Số v1 dùng được, ghi là lenient |
| Khớp nhóm số lượng chỉ chấm mô hình có bảng số lượng | Số của countonly dùng được; không so với `main_r3` |
| Đồ thị con: cắt đường truyền tin, giữ bộ ba | Số v1 lẫn việc bỏ bộ ba, chỉ dùng tham khảo |

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
  - strict: synset của node ảnh trùng hoặc là hyponym của nghĩa danh từ đầu tiên của từ trong câu; người chỉ khớp với từ chung (person, people...).
  - lenient (tiêu chí v1): cùng loại theo `dedup.same_type`, nên "woman" ~ "man" cũng tính là khớp.
  - Với bộ ba: chủ thể và đối tượng có khớp strict không, vị từ có khớp sau alias VG không.
- **Khớp nhóm số lượng:** với node câu có số lượng nêu rõ, nhóm số lượng của node ảnh có tương thích không (chỉ countonly).

**Sparsity** (độ gọn):

- số phần đầu của E chiếm 80% / 90% khối term dương;
- k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% / 90% hiệu ứng đồ thị;
- số dòng của bảng demo đủ cho 80% biên so với trung bình 50 ứng viên.

**Đồ thị con:** với 3 node đầu của E, lấy các cạnh vào có attention GAT lớn nhất, cắt đường truyền tin của chúng, so với cắt cùng số cạnh vào ngẫu nhiên của chính từng node.

## 5. Chín cách giải thích để so sánh

Một con số fidelity đứng một mình khó đọc; nhóm đo thêm 8 cách giải thích khác, xếp **cùng các phần của ảnh** và đo **cùng một cách**.
Danh sách theo các họ phương pháp chính của giải thích GNN (Yuan và cộng sự, TPAMI 2022) và giải thích cho xếp hạng (RankingSHAP / RankSHAP):

| Cách | Loại | Chi phí mỗi câu |
|---|---|---|
| **E** (`model`) | Nội tại, phân rã điểm | 0 lượt chấm thêm |
| random | Mốc dưới | 0 |
| lowest | E đảo ngược | 0 |
| clip | Cosine CLIP giữa câu và từng phần, không dùng mô hình | 0 |
| gradient | Gradient × input theo vector đầu vào của từng phần | 1 lượt lan truyền ngược |
| integrated gradients (IG) | Phân rã hậu nghiệm từ ảnh trung tính (Sundararajan và cộng sự, 2017), 20 bước | 20 lượt lan truyền ngược |
| occlusion | Bỏ từng phần một, điểm mất là độ quan trọng | 1 lượt chấm mỗi phần (TB 46) |
| GNNExplainer | Mặt nạ mềm trên vector đầu vào, học để giữ điểm (Ying và cộng sự, 2019) | 100 bước tối ưu |
| **Shapley** | Giá trị Shapley lấy mẫu 16 hoán vị, hàm giá trị là biên giữa hạng 1 và hạng 2 | khoảng 740 lượt chấm |

Shapley là chuẩn tham chiếu: với mỗi cách khác, đo tương quan Spearman giữa thứ tự của nó và giá trị Shapley, và tỉ lệ trùng top-5.
Kiểm tra cài đặt: IG thoả completeness với sai số trung bình 1,1% hiệu ứng; Shapley thoả efficiency với sai số tối đa 1,3e−7.

## 6. Kết quả (val, mean ± std qua 3 seed)

R@1 tính lại trong lúc đo: `main_r3` 46,67 / 46,56 / 46,21; countonly 49,15 / 48,85 / 48,74.

### 6.1. Fidelity

**Bỏ top-k của E (Fid+): tỉ lệ ảnh mất hạng 1.**

| k | `main_r3`: E | Ngẫu nhiên | countonly (v1): E | Ngẫu nhiên |
|---|---|---|---|---|
| 1 | 20,5 ± 1,8% | 3,0% | 34,3 ± 4,8% | 4,4% |
| 3 | 34,9 ± 2,8% | 6,5% | 58,0 ± 4,1% | 9,4% |
| 5 | 43,6 ± 3,1% | 9,8% | 68,0 ± 3,0% | 12,9% |

**Chỉ giữ top-k của E (Fid−): hạng của ảnh.**

| k | `main_r3`: E | Ngẫu nhiên | countonly (v1): E | Ngẫu nhiên |
|---|---|---|---|---|
| 1 | 3,00 ± 0,03 | 9,19 | 4,18 ± 0,83 | 10,13 |
| 3 | 2,16 ± 0,08 | 7,38 | 2,22 ± 0,74 | 8,13 |
| 5 | 2,17 ± 0,06 | 6,89 | 1,86 ± 0,54 | 6,93 |

- Ở cả hai mô hình, E cách xa ngẫu nhiên theo cả hai chiều: bỏ top-3 làm mất hạng 1 gấp 5–6 lần ngẫu nhiên, và chỉ giữ top-3 cho hạng khoảng 2,2 so với 7–8.
- Fid+ của countonly cao hơn `main_r3` vì E của countonly ít bộ ba ở đầu hơn (mục 6.2).
- Fid− của `main_r3` gần như không đổi từ k = 3 lên k = 5: phần thứ 4 và 5 của E phần lớn là bộ ba, mà giữ thêm bộ ba ít đổi điểm.

### 6.2. Kênh bộ ba: nặng ký trong E nhưng bỏ đi ít tác dụng

| | Bộ ba ở top-1 / top-3 / top-5 của E | Bỏ top-1 khi nó là node: mất hạng 1 | Khi nó là bộ ba |
|---|---|---|---|
| `main_r3` (v2) | 70,5 / 76,4 / 77,6% | 49,3% (ngẫu nhiên 4,4%) | 8,4% (ngẫu nhiên 2,5%) |
| countonly (v1) | 44 / 50 / 56% | 46,5% (ngẫu nhiên 4,7%) | 19,7% (ngẫu nhiên 4,2%) |

- Ở cả hai mô hình, khi bộ ba đứng đầu E, bỏ nó ít khi làm ảnh mất hạng 1.
  Với countonly, chỉ giữ bộ ba đứng đầu cho hạng 3,70, gần như giữ một bộ ba bất kỳ (4,02).
- Lý do phù hợp với số liệu: điểm kênh là trung bình softmax trên các phần của ảnh, và kênh bộ ba dàn trải (top-3 bộ ba chỉ chiếm 23–39% term của kênh, so với 63–68% của top-3 node).
  Bỏ một bộ ba thì attention dồn sang các bộ ba tương tự còn lại, nên điểm gần như không đổi.
- Với `main_r3`, đưa term về thang điểm cuối làm bộ ba lên đầu E ở 70% câu (term thô: 1%), vì z-score khuếch đại kênh có độ phân tán nhỏ.
  Đúng theo đạo hàm, nhưng làm E kém trung thực hơn khi đo bằng can thiệp: Fid+ k = 3 là 34,9%, so với 65,7% khi xếp theo term thô.

### 6.3. Tính cộng: độ lớn của term không phản ánh điểm mất

`main_r3`, bỏ top-k của E (toàn val):

| k | Tổng term top-k (dự đoán) | Điểm mất đo được | Tỉ số | Spearman |
|---|---|---|---|---|
| 1 | 2,31 ± 0,35 | 0,189 ± 0,021 | 0,084 | 0,18 |
| 3 | 6,12 ± 0,96 | 0,375 ± 0,036 | 0,063 | −0,04 |
| 5 | 9,22 ± 1,46 | 0,501 ± 0,042 | 0,056 | −0,10 |

- Điểm mất thật chỉ bằng khoảng 6% tổng term của top-3, và gần như không tương quan với tổng đó.
- Cách giải thích phù hợp với số liệu, chưa kiểm riêng: term đo mức khớp tuyệt đối, phần lớn là mức nền mà mọi ứng viên đều có; z-score chỉ giữ chênh lệch giữa các ứng viên.
- Occlusion (tỉ số 1,15, Spearman 0,89 ở k = 3) và IG (1,43, 0,82) cộng tính tốt hơn nhiều.
- countonly: __,__ (chưa đo; v1 chỉ thấy cùng hiện tượng ở dạng chỉ giữ top-5 cho 117% hiệu ứng, vì bỏ phần khớp kém làm trung bình softmax tăng).

### 6.4. Validity

**Phản thực tế** (`main_r3`):

- Bỏ tối đa 10 phần đầu của E làm ảnh mất hạng 1 ở 60,3 ± 2,9% câu, trung vị k_flip là 3 (ảnh sai 74,4%, ảnh đúng 44,1%).
- Với ảnh hạng 1 sai, bỏ top-5 của E làm ảnh đúng lên hạng ở 30,7% câu và lên hạng 1 ở 13,1% câu.
  E chỉ ra phần đã đẩy ảnh sai lên, nhưng bỏ chúng thường chưa đủ để ảnh đúng thắng.

**Ngữ nghĩa:**

| Tiêu chí | `main_r3` (v2) | countonly (v1) |
|---|---|---|
| Vật thể, strict | 58,3 ± 0,3% | __,__ |
| Vật thể, lenient | 63,8 ± 0,2% | 70,0 ± 1,0% |
| Bộ ba, khớp đủ chủ thể, đối tượng, vị từ | 5,4 ± 0,2% | __,__ |
| Khớp nhóm số lượng (dòng có số nêu rõ) | không chấm | 66,7 ± 0,8% |

- Node ảnh được chú ý nhất cùng loại với node câu ở khoảng 58% dòng theo tiêu chí strict; tiêu chí lenient cộng thêm khoảng 5,5 điểm, chủ yếu là các cặp người khác loại như "woman" ~ "man".
- Dòng bộ ba hầu như không khớp nghĩa: chỉ 5,4% khớp đủ.

### 6.5. Sparsity

| Đo | `main_r3` (v2) | countonly (v1) |
|---|---|---|
| k nhỏ nhất mà chỉ giữ top-k còn ≥ 80% hiệu ứng | 2,65 ± 0,14, đạt ở 55,7% câu | __,__ |
| Số phần đầu của E cho 80% khối term | 15,4 phần (33% số phần) | __,__ |
| Số dòng demo cho 80% biên so với trung bình top-50 | 1,95 | 2,52 |

- Theo hiệu ứng, E gọn: ở hơn nửa số câu, 2–3 phần đầu đã giữ 80% hiệu ứng đồ thị.
- Theo khối term thì E dàn trải (15 phần cho 80%), do nhiều bộ ba có term gần bằng nhau; đây lại là hệ quả của mục 6.3.

### 6.6. Đồ thị con theo attention GAT

`main_r3`, các câu mà số cạnh vào lớn hơn số cạnh bỏ (2 803 câu mỗi seed):

| Cắt các cạnh vào có attention cao nhất | Δ điểm | Mất hạng 1 |
|---|---|---|
| Top theo attention | −0,0011 ± 0,0007 | 1,0 ± 0,6% |
| Ngẫu nhiên | −0,0010 ± 0,0005 | 0,7 ± 0,4% |

- Cắt đường truyền tin của các cạnh có attention cao nhất không khác cắt ngẫu nhiên.
- countonly (v1, dạng bỏ cả cạnh lẫn bộ ba): 6,6% so với 3,8%; khoảng một nửa số cạnh được chọn là cạnh kNN và một phần ba là cạnh từ node tổng.
- Ablation bỏ cạnh lúc train làm R@1 giảm 2,39 (`experiments/improve/no-edges/`): cạnh có ích ở tổng thể, nhưng tác dụng trải trên nhiều cạnh, không nằm ở vài cạnh có attention cao.

### 6.7. So sánh 9 cách giải thích (`main_r3`)

| Cách | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Khớp Shapley (Spearman) | Top-5 trùng Shapley |
|---|---|---|---|---|---|
| Shapley (195 câu) | **−0,68** | **1,53** | **0,118 / −0,046** | chuẩn | chuẩn |
| occlusion | −0,24 | 1,23 | 0,106 / −0,046 | **0,48** | 58,5% |
| IG | −0,45 | 0,65 | 0,090 / −0,024 | 0,29 | **62,9%** |
| GNNExplainer (195 câu) | −0,17 | −0,34 | 0,091 / 0,000 | 0,26 | 56,7% |
| **E** | 0,24 | 0,76 | 0,050 / 0,016 | 0,46 | 31,8% |
| clip | 0,28 | 0,23 | 0,047 / 0,021 | 0,35 | 35,4% |
| gradient | 0,83 | −1,04 | 0,012 / 0,057 | −0,13 | 17,8% |
| random | 0,84 | −0,62 | 0,013 / 0,053 | −0,01 | 14,3% |

Số câu so sánh: 972–980 mỗi seed, trừ hai dòng ghi riêng.
Kiểm định Wilcoxon ghép cặp với E trên deletion AUC: mọi cách khác đều khác E với p < 1e−11 ở cả 3 seed, trừ clip.

- **Shapley, occlusion và IG trung thực nhất trên mọi trục**; bỏ dần theo thứ tự của chúng làm điểm xuống dưới cả mức ảnh rỗng (deletion AUC âm).
- **E đứng giữa**: tốt hơn hẳn ngẫu nhiên và gradient, ngang clip trên deletion, tốt hơn clip trên insertion.
- **Về thứ tự tổng thể, E khớp Shapley gần bằng occlusion (0,46 so với 0,48)** mà không tốn lượt chấm nào; nhưng top-5 của E chỉ trùng 32% với Shapley, đúng ở chỗ bộ ba chen vào đầu E (mục 6.2).
- **Gradient × input gần như ngẫu nhiên**: đạo hàm cục bộ không nói gì về việc bỏ hẳn một phần.
- countonly: __,__ (chưa đo).

## 7. Bằng chứng trên 13 truy vấn

Đề yêu cầu ít nhất 10 truy vấn; nhóm dùng 13 câu đã phân tích ở `report/level1-draft.md` mục 8.3 (10 ca và 3 ví dụ so với cấp 1).
Bảng dưới là `main_r3` seed 0 (v2); ảnh được giải thích là ảnh hạng 1 của mô hình.
Cột "Bỏ top-3" và "Chỉ giữ top-3" ghi hạng của ảnh, trong ngoặc là trung bình 10 lần chọn ngẫu nhiên.
Chi tiết (top-5 của E, từng dòng vật thể và kết quả khớp nghĩa, Fid với k = 1, 3, 5, đường cong, đồ thị con) ở `experiments/xai-metrics-v2/cases.json`; bảng tương ứng cho countonly (v1) ở `experiments/xai-metrics/REPORT.md`.

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
  Câu 526446_0: E là catcher, dirt, guard; bỏ ba node này làm ảnh sai rơi từ hạng 1 xuống hạng 45 (ngẫu nhiên: 21,9), còn chỉ giữ chúng thì ảnh vẫn hạng 1 (ngẫu nhiên: 22,0).
  Câu 445695_1 (donut, bite) cũng vậy: bỏ thì xuống hạng 11, giữ thì vẫn hạng 1.
- **Khi E gồm bộ ba, bỏ đi gần như không đổi gì.**
  Ở 10/13 câu, ba phần đầu của E đều là bộ ba, và ở cả 10 câu đó, bỏ cả ba thì ảnh vẫn hạng 1.
  Câu 574350_4 (chó và ô) là ví dụ rõ nhất: E là "frame has frame", "man has pant", "woman wearing pant", không liên quan tới câu; trong khi bảng demo cho thấy bỏ node `dog` đưa ảnh xuống hạng 2.
- **Chỉ giữ top-3, ảnh vẫn ở hạng 1–2 ở 10/13 câu**, kể cả khi top-3 là bộ ba.
  Điều này phù hợp với việc điểm kênh là trung bình softmax: giữ ít phần thì cũng bỏ đi các phần khớp kém đang làm loãng trung bình.
- Các ca này khớp với số trên toàn val ở mục 6: E đúng ở chỗ phần nào mang điểm, nhưng các bộ ba đứng đầu E không phải là thứ quyết định thứ hạng.

## 8. Áp dụng vào demo

Từ mục 6.2 và 6.3, cột "% điểm cuối" của bảng giải thích trong demo dễ bị đọc nhầm là "bỏ phần này thì mất chừng ấy điểm".
Demo nay có thêm cột **Bỏ phần này** (`app/demo_app.py`, `src/service/graph_mode.py`):

- với mỗi dòng khớp, ảnh được chấm lại bằng chính mô hình sau khi bỏ phần khớp đầu tiên của dòng đó, 49 ảnh khác giữ nguyên;
- ô ghi thay đổi của điểm cuối và hạng mới, in đậm khi hạng đổi;
- hai ghi chú dưới bảng nói rõ % điểm cuối chỉ là cách chia điểm, không phải mức điểm mất khi bỏ phần.

Đây là phép kiểm chứng bằng chính mô hình (occlusion của phần đang hiển thị), không phải một lời giải thích tìm sau khi đã có kết quả; các dòng giải thích vẫn lấy từ lần xếp hạng.
Ví dụ trên val với `main_r3_seed0`: câu "A batter is preparing to hit the ball at a game.", bỏ node `catcher` đưa ảnh từ hạng 1 xuống hạng 43; câu về chó và ô, bỏ các bộ ba khớp nhiều % nhất không đổi hạng.

## 9. Kết luận

1. **E trung thực theo thứ tự.**
   Bỏ top-3 của E làm mất hạng 1 ở 35% câu (`main_r3`) và 58% câu (countonly), so với 6–9% khi bỏ ngẫu nhiên; thứ tự của E khớp Shapley gần bằng occlusion (0,46 so với 0,48).
2. **E không trung thực theo độ lớn**, nhất là ở kênh bộ ba.
   Điểm mất khi bỏ top-3 chỉ bằng 6% tổng term của chúng (`main_r3`); bộ ba đứng đầu E nhưng bỏ đi ít khi đổi hạng.
   Nên trình bày E như một thứ tự, và kiểm chứng bằng cột "Bỏ phần này" của demo.
3. **Nếu cần lời giải thích trung thực nhất, dùng occlusion hoặc IG** làm kiểm chứng: occlusion khớp Shapley tốt nhất và cộng tính tốt, chỉ tốn khoảng 46 lượt chấm mỗi câu; IG thoả completeness.
   Shapley là chuẩn tốt nhất nhưng tốn khoảng 740 lượt chấm mỗi câu.
4. **Validity ngữ nghĩa vừa phải ở node, kém ở bộ ba:** 58% dòng vật thể khớp strict, chỉ 5% dòng bộ ba khớp đủ.
5. **Sparsity theo hiệu ứng tốt:** ở hơn nửa số câu, 2–3 phần đầu của E giữ 80% hiệu ứng đồ thị.
6. **Attention GAT không phải lời giải thích:** cắt đường truyền tin của các cạnh có attention cao nhất không khác cắt ngẫu nhiên.

## 10. Hạn chế

- Mô hình chính countonly mới có số v1; phần so sánh 9 cách giải thích, tính cộng, validity strict và sparsity theo E chưa đo cho nó.
  Cơ chế giải thích của hai mô hình giống nhau, chỉ khác cách trộn điểm, nên các kết luận nhiều khả năng vẫn đúng, nhưng chưa có số.
- So sánh các cách giải thích dùng 1 000 câu; GNNExplainer và Shapley chỉ khoảng 195 câu, std qua seed của hai cách này lớn hơn.
- Ảnh trung tính của IG và GNNExplainer là vector trung bình của từng loại phần; chọn mốc khác có thể đổi kết quả của hai cách này.
- Validity ngữ nghĩa dựa trên WordNet và alias VG; nhãn VG thiếu hoặc khác từ vựng với caption COCO (`report/level1-draft.md` mục 8.4) làm tỉ lệ khớp thấp hơn thực tế.
- Bỏ node cũng bỏ các bộ ba chạm vào nó, nên hiệu ứng của một node đã gồm phần của các bộ ba đó.
- Chỉ đo trên ảnh hạng 1, chỉ trên val.

## 11. MMKG tiếng Việt và phân tích chuyển giao

Phần của Khang (dòng 30–32 của bảng phân công): bộ dữ liệu ở `vietnamese-mmkg/` (data card `vietnamese-mmkg/DATA_CARD.md`), chuyển giao ở `scripts/eval_vnmmkg_transfer.py`.
Nội dung mục này do Khang viết.

## 12. Nguồn số liệu

| Nội dung | File |
|---|---|
| `main_r3`, mọi độ đo v2, 3 seed | `experiments/xai-metrics-v2/metrics.json`, `summary.json`, bảng đầy đủ theo nhóm đúng / sai ở `experiments/xai-metrics-v2/REPORT.md` |
| 13 câu bằng chứng, `main_r3` seed 0 | `experiments/xai-metrics-v2/cases.json` |
| countonly, số v1, 3 seed | `experiments/xai-metrics/metrics.json`, `summary.json`, `REPORT.md` (đọc kèm mục đính chính ở đầu) |
| Script đo | `scripts/xai_metrics.py` (v2), `scripts/xai_metrics_v1.py` (v1), `src/explain/xai_metrics.py` |
| Script tổng hợp bảng | `scripts/summarize_xai_metrics.py` (v2), `scripts/summarize_xai_metrics_v1.py` (v1) |
| Lệnh chạy lại | `experiments/xai-metrics-v2/run.sh`, `experiments/xai-metrics/run.sh` |

**Khai báo công cụ AI:** code đo, phần chạy thí nghiệm và bản nháp này được viết cùng Claude (Anthropic); Nhi duyệt định nghĩa các độ đo, kết quả và kết luận.
Mọi con số lấy từ các file ở bảng trên.
