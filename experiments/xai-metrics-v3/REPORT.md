# Đo chất lượng giải thích cấp 3, phiên bản 3: thêm phản thực tế, GraphMask và GStarX (val)

Bản này là v2 (`experiments/xai-metrics-v2/`, PR #50) cộng ba cách giải thích mới và hai nhóm độ đo mới.
Mô hình, câu, phép sửa đồ thị, mốc so sánh và định nghĩa các độ đo cũ giữ y như v2; đọc định nghĩa ở `experiments/xai-metrics-v2/REPORT.md`.
Chỉ dùng val, không train mô hình xếp hạng, không chấm test.

- **Mới trong v3:**
  - ba cách giải thích, thành 16 cách: CF-GNNExplainer (Lucic và cộng sự, AISTATS 2022), GraphMask (Schlichtkrull và cộng sự, ICLR 2021), GStarX (Zhang và cộng sự, NeurIPS 2022);
  - phản thực tế nhỏ nhất: duyệt mọi tập ≤ 2 phần, so với CF-GNNExplainer và với phản thực tế tham lam của từng cách giải thích;
  - phản thực tế có đích: với ảnh hạng 1 sai, tập ≤ 2 phần nhỏ nhất mà bỏ đi thì ảnh vàng lên hạng 1;
  - mức message: GraphMask so với attention GAT và ngẫu nhiên khi ngừng truyền từng message.
- **Script:** `src/explain/xai_counterfactual.py` (mới), `src/explain/xai_baselines.py` (GraphMask, GStarX), `scripts/xai_metrics.py` (v3), `scripts/summarize_xai_metrics.py`; `run.sh` chạy lại cả hai bước, log ở `logs/run.log`.
- **Thời gian:** CPU (MacBook), chạy song song với `experiments/xai-ground-truth/`, nên số giây mỗi câu cao hơn v2 và chỉ dùng để so giữa các cách trong cùng bản này.
  Mỗi seed khoảng 80–97 phút với countonly (4 122 / 5 839 / 4 835 giây; seed 0 là lần chạy lại sau review PR #52) và khoảng 21 phút với `main_r3`.

## Kiểm tra tái lập

- **v3 không đổi số nào của v2:** mọi số trong `metrics.json` của v2 (trừ thời gian) có ở v3 với giá trị trùng tuyệt đối, 14 542 giá trị; chỉ đoạn mô tả `rule` khác.
  13 câu bằng chứng trong `cases.json` cũng trùng (8 780 giá trị), trừ 12 số giây; v3 chỉ thêm khoá `cf_exact`, `graphmask`, `messages`.
  Ba cách mới dùng bộ sinh số ngẫu nhiên riêng và GraphMask học trong `torch.random.fork_rng`, nên không chạm vào các lần bốc của v2.
- R@1 tính lại trong lúc đo trùng checkpoint như v2: countonly 49,15 / 48,85 / 48,74, `main_r3` 46,67 / 46,56 / 46,21.
- GStarX: giá trị Hamiache-Navarro tính chính xác trên mỗi tập liên thông được lấy mẫu (test: trùng Shapley trên đồ thị đầy đủ, efficient trên đường thẳng, `tests/test_xai_extra.py`); TB 542 lượt chấm mỗi câu (countonly) và 318 (`main_r3`).
- CF-GNNExplainer: sau mỗi bước, các phần có mặt nạ < 0,5 bị bỏ thật và ảnh được chấm lại; chỉ tập bỏ thật làm mất hạng 1 mới tính là phản thực tế.

## Ba cách giải thích mới, chỉnh cho reranker

- **CF-GNNExplainer:** bản gốc học một nhiễu thưa của ma trận kề, loss = L_pred + β · L_dist, L_pred chỉ bật khi dự đoán chưa đổi.
  Ở đây mỗi phần có mặt nạ m, vector của nó là m · vector + (1 − m) · vector trung tính của loại phần đó (như GNNExplainer ở v2); L_pred là hinge trên biên hạng 1 chia cho hiệu ứng đồ thị, L_dist là tỉ lệ phần bị bỏ, β = 0,5, 200 bước.
  Thứ tự để vẽ đường cong là 1 − mặt nạ cuối.
- **GraphMask:** một cổng hard concrete cho mỗi message của mỗi lớp GAT, do một MLP chung (amortized) theo lớp sinh ra từ vector nguồn, vector đích, vector cạnh và chính message; message bị đóng thay bằng vector baseline học được của lớp đó.
  Huấn luyện bằng Lagrange như bài gốc: tối thiểu L0 với ràng buộc điểm đổi ≤ 5% hiệu ứng đồ thị, trên cùng 200 câu val của PGExplainer.
  Cổng gắn vào mô hình bằng hook của PyG (`register_message_forward_hook`), không sửa code mô hình; bài gốc sửa lớp GNN, kết quả tương đương nhưng cách cài khác.
  Cổng khởi tạo ở trạng thái mở (bias 3): với khởi tạo ngẫu nhiên, countonly seed 0 đóng hết cổng cùng lúc và clamp của hard concrete không còn gradient để mở lại, dù điểm đổi tới 0,61 hiệu ứng (vượt xa 0,05).
  Độ quan trọng của một message là P(cổng ≠ 0); để so cùng bảng, bộ ba lấy giá trị của cạnh có nhãn của nó, node lấy giá trị lớn nhất trong các message nó gửi đi.
- **GStarX:** giá trị Hamiache-Navarro, một giá trị Shapley có cấu trúc chỉ cho các liên minh liên thông tương tác.
  Đồ thị các phần: hai node kề nhau khi có cạnh có nhãn hoặc cạnh hoàn thiện; bộ ba kề hai node đầu cuối.
  Hàm giá trị giống Shapley của v2 (biên hạng 1 khi chỉ giữ liên minh); lấy mẫu các tập liên thông ≤ 5 phần mọc từ từng phần, tính HN chính xác trên mỗi tập (τ = 1/n, bình phương ma trận đến khi đổi < 1e−12), rồi lấy trung bình cho từng phần.

**Tình trạng học của GraphMask** (từ `metrics.json`, lần học cuối, 200 câu):

| Mô hình | Seed | L0 (tỉ lệ cổng mở) | Điểm đổi / hiệu ứng (ràng buộc 0,05) |
|---|---|---|---|
| countonly | 0 / 1 / 2 | 0,82 / 1,00 / 1,00 | 0,242 / 0,000 / 0,000 |
| `main_r3` | 0 / 1 / 2 | 0,000 / 0,000 / 0,001 | 0,113 / 0,079 / 0,177 |

Với countonly seed 1 và 2, GraphMask giữ gần hết message và thoả ràng buộc; với `main_r3`, nó đóng gần hết message và vượt ràng buộc (8–18% hiệu ứng thay vì 5%).
**Việc học GraphMask của countonly seed 0 không ổn định** (`scripts/checks/graphmask_stability.py`, `graphmask_stability.json`): cùng code, cùng seed, cùng 200 câu, 9 lần học (1, 4 và 8 luồng CPU, mỗi mức 3 lần) cho điểm đổi từ 0,07 đến 0,74 hiệu ứng và L0 từ 0,57 đến 0,89; với 1 luồng kết quả lặp lại y hệt (0,74), với nhiều luồng thì thứ tự cộng số thực đổi và kết quả đổi theo.
Lần chạy đầu của v3 cho 0,001, lần chạy lại sau review PR #52 cho 0,242 (số trong bảng); không lần nào trong 9 lần thử đạt ràng buộc 0,05.
Cùng phép thử, `main_r3` seed 0 cho đúng 0,113 ở cả 9 lần.
Vì vậy các số GraphMask của countonly ở dưới (seed 0) là một lần rơi của một bài toán tối ưu nhạy với nhiễu số học, không phải giá trị cố định; kết luận "GraphMask gần ngẫu nhiên" đúng với cả lần chạy đầu và lần chạy lại.
Cả hai trường hợp đều cho rất ít thông tin để xếp các phần: các xác suất gần bằng nhau (gần 1 hoặc gần 0), nên thứ tự chủ yếu do chênh lệch rất nhỏ.
Kết quả mức message bên dưới khớp với việc này: message chỉ mang một phần nhỏ của điểm.

## Kết quả (val, mean ± std qua 3 seed)

### Phản thực tế nhỏ nhất

Trên tập con so sánh (TB 196 câu countonly, 195 câu `main_r3`), duyệt mọi tập 1 hoặc 2 phần của ảnh hạng 1:

| | countonly | `main_r3` |
|---|---|---|
| Có phản thực tế ≤ 2 phần | 74,2 ± 1,9% | 70,7 ± 1,3% |
| Cỡ 1 / cỡ 2 | 53,5% / 20,7% | 51,5% / 19,2% |
| Ảnh đúng / ảnh sai có phản thực tế ≤ 2 phần | 62,5% / 84,2% | 52,5% / 83,4% |
| Ảnh sai: bỏ ≤ 2 phần thì ảnh vàng lên hạng 1 | 19,4 ± 4,2% | 15,3 ± 3,6% |
| CF-GNNExplainer tìm được phản thực tế | 45,7 ± 4,3% | 34,6 ± 2,9% |
| Cỡ TB của nó, và tỉ lệ bằng cỡ nhỏ nhất | 2,3; 65,8% | 2,1; 74,4% |

- Hơn nửa số ảnh hạng 1 mất hạng 1 khi bỏ đúng một phần; ảnh hạng 1 sai dễ lật hơn ảnh đúng (84% so với 53–63% với ≤ 2 phần).
- CF-GNNExplainer chỉ tìm được phản thực tế ở 35–46% câu, ít hơn hẳn mức có thật (71–74% chỉ với ≤ 2 phần); khi tìm được, nó thường đúng cỡ nhỏ nhất.

**Phản thực tế tham lam** (bỏ dần theo thứ tự của từng cách, k ≤ 10; "bằng cỡ nhỏ nhất" tính trên câu có phản thực tế ≤ 2 phần):

| Cách | countonly: tìm được | Cỡ TB | Bằng cỡ nhỏ nhất | `main_r3`: tìm được | Cỡ TB | Bằng cỡ nhỏ nhất |
|---|---|---|---|---|---|---|
| occlusion | 83,0% | **2,00** | **90,6%** | 80,5% | **1,89** | **92,4%** |
| GraphSVX | 89,3% | 2,21 | 81,3% | **86,1%** | 2,27 | 82,3% |
| IG | **89,6%** | 2,32 | 81,2% | 84,9% | 2,20 | 84,0% |
| Shapley | 88,1% | 2,28 | 77,5% | 85,3% | 2,42 | 77,1% |
| **E** (`model`) | 84,2% | 2,65 | 59,7% | 63,9% | 3,74 | 46,7% |
| GNNExplainer | 83,5% | 2,65 | 56,2% | 81,8% | 2,53 | 64,9% |
| GStarX | 73,0% | 3,23 | 46,9% | 76,9% | 3,23 | 47,9% |
| CF-GNNExplainer | 69,3% | 2,59 | 70,6% | 50,8% | 2,30 | 74,9% |
| GraphMask | 35,3% | 5,50 | 15,2% | 33,1% | 4,57 | 28,9% |
| random | 32,1% | 5,02 | 18,0% | 29,8% | 5,01 | 15,8% |

- **Occlusion là phản thực tế tham lam tốt nhất:** đúng cỡ nhỏ nhất ở 91–92% câu, vì phản thực tế cỡ 1 chính là phần có điểm mất lớn nhất khi bỏ một mình.
- **E là phản thực tế tốt với countonly** (tìm được 84%, đúng cỡ nhỏ nhất 60%), kém hơn với `main_r3` (64%, 47%), đúng chỗ bộ ba chen vào đầu E (v2, mục 6.2 của báo cáo).
- **Thứ tự của chính CF-GNNExplainer tìm được phản thực tế ít hơn E và occlusion** (69% và 51%), dù nó được thiết kế cho việc này; khi tìm được, nó đúng cỡ nhỏ nhất nhiều hơn E (71% và 75%).

### GraphMask và GStarX trong bảng so sánh chung

| Cách | countonly: deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | Spearman với Shapley | `main_r3`: deletion | Insertion | Spearman |
|---|---|---|---|---|---|---|
| Shapley | −0,19 | 1,54 | chuẩn | −0,68 | 1,53 | chuẩn |
| GraphSVX | −0,24 | 1,56 | 0,75 | −0,71 | 1,50 | 0,72 |
| **GStarX** | 0,16 | 1,39 | **0,65** | −0,13 | 1,26 | **0,66** |
| occlusion | −0,01 | 1,45 | 0,45 | −0,24 | 1,23 | 0,48 |
| **E** | 0,07 | 1,26 | 0,61 | 0,24 | 0,76 | 0,46 |
| **CF-GNNExplainer** | 0,33 | 0,68 | 0,18 | 0,41 | −0,73 | 0,01 |
| **GraphMask** | 0,88 | −0,28 | −0,17 | 0,77 | −0,86 | 0,01 |
| random | 0,89 | 0,21 | 0,00 | 0,84 | −0,62 | −0,01 |

- **GStarX khớp Shapley tốt thứ hai** sau GraphSVX ở cả hai mô hình (0,65 và 0,66), với khoảng nửa số lượt chấm (542 và 318 mỗi câu, so với khoảng 700 của Shapley).
  Insertion AUC của nó cao (1,39 và 1,26), nhưng deletion AUC kém occlusion và Shapley: các liên minh liên thông nhỏ đánh giá tốt việc "giữ lại", kém hơn ở việc "bỏ đi".
- **CF-GNNExplainer là thứ tự kém:** loss của nó chỉ đòi lật hạng 1 với ít phần nhất, không đòi xếp đúng các phần còn lại; một lý do có thể, chưa kiểm riêng, là mặt nạ ngoài tập phản thực tế không mang thông tin xếp hạng.
- **GraphMask ở mức các phần gần như ngẫu nhiên** ở cả hai mô hình (deletion AUC 0,88 và 0,77, random 0,89 và 0,84; insertion AUC còn dưới random), đúng như tình trạng học ở trên.

### Mức message: GraphMask, attention GAT và ngẫu nhiên

Ngừng truyền top-k message của ảnh hạng 1 (giữ bộ ba), theo thứ tự của GraphMask, của attention GAT (trung bình 2 lớp × 4 head) và ngẫu nhiên; phần trăm tính trên điểm mất khi ngừng mọi message, chỉ ở câu mà message giúp ảnh.

| | countonly | `main_r3` |
|---|---|---|
| Câu mà ngừng mọi message làm ảnh mất điểm | 12,2% | 48,2% |
| Điểm đổi TB khi ngừng mọi message (sau − trước) | +0,0079 ± 0,0014 | −0,0010 ± 0,0013 |
| Message TB mỗi ảnh | 188 | 19 |
| AUC mất điểm (cao tốt): GraphMask / attention / random | −0,12 / −0,23 / −0,19 | 0,40 / 0,41 / 0,32 |

- **Message mang rất ít điểm:** với countonly, ngừng mọi message chỉ làm ảnh mất điểm ở 12% câu, còn trung bình điểm lại tăng (+0,0079); với `main_r3`, ngừng mọi message làm ảnh mất điểm ở 48% câu, nhưng trung bình chỉ giảm 0,0010.
- **Với countonly, không thứ tự nào tốt hơn ngẫu nhiên** (AUC âm cho cả ba, Wilcoxon GraphMask so với random p lớn nhất 0,66): bỏ vài message đầu thường làm điểm tăng chứ không giảm.
- **Với `main_r3`, GraphMask và attention ngang nhau** (0,40 và 0,41) và đều hơn random (0,32; p ≤ 1,2e−5 so với GraphMask).
- Kết luận của v2 về attention GAT vẫn đứng: attention không phải lời giải thích tốt cho các phần, và ở mức message, cả cách học riêng cho việc này (GraphMask) cũng không hơn attention.

## Kết luận

1. **Phản thực tế ngắn có thật:** khoảng 3/4 ảnh hạng 1 mất hạng 1 khi bỏ ≤ 2 phần, hơn nửa chỉ với 1 phần.
   Occlusion tìm được phản thực tế nhỏ nhất ở hơn 90% câu đó; E làm được 60% với countonly.
2. **Trong ba cách mới, chỉ GStarX đáng dùng:** nó khớp Shapley tốt thứ hai với ít lượt chấm hơn.
   CF-GNNExplainer tìm được ít phản thực tế hơn mức có thật, và GraphMask không phân biệt được các phần.
3. **Truyền tin GAT không phải nơi mô hình đặt điểm:** ngừng mọi message đổi điểm rất ít, nên các cách giải thích theo message (attention, GraphMask) có ít thứ để giải thích.
4. Thứ hạng các cách cũ giữ nguyên như v2.

## Hạn chế

- Phản thực tế nhỏ nhất chỉ duyệt tới cỡ 2; câu không có phản thực tế ≤ 2 phần có thể có phản thực tế cỡ 3 trở lên.
- GraphMask với `main_r3` không thoả ràng buộc 5%; tăng bước Lagrange hoặc số epoch có thể đổi kết quả.
- Việc học GraphMask của countonly seed 0 không lặp lại được khi chạy nhiều luồng (điểm đổi 0,07–0,74 qua 9 lần thử); chưa thử cách làm cho nó ổn định (bước học nhỏ hơn, nhiều lần khởi tạo).
- GStarX lấy mẫu tập liên thông ≤ 5 phần, không duyệt hết; CF-GNNExplainer dùng β và số bước cố định.
- Các tập con, mốc trung tính và ngân sách khác giữ như v2.

Các bảng đầy đủ theo nhóm đúng / sai ở phụ lục dưới, in bằng `scripts/summarize_xai_metrics.py --dir experiments/xai-metrics-v3` (cũng ở `logs/tables.md`).

## Phụ lục: bảng đầy đủ theo nhóm

### countonly_factual_r3 (R@1 theo seed: 49,15, 48,85, 48,74)

**Chạy.**

| Seed | Giây (CPU) | Câu | E rỗng | noop khác hạng 1 | Sai số noop tối đa |
| --- | --- | --- | --- | --- | --- |
| 0 | 4122 | 10633 | 31 | 0 | 2.4e-07 |
| 1 | 5839 | 10633 | 31 | 0 | 2.4e-07 |
| 2 | 4835 | 10633 | 36 | 0 | 2.4e-07 |

**Fid+ (bỏ top-k).**

| Nhóm | k | Mất hạng 1: top (%) | Ngẫu nhiên (%) | Hạng: top | Ngẫu nhiên | Δ điểm: top | Ngẫu nhiên | Hiệu ứng bị bỏ, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 34,4 ± 4,8 | 4,4 ± 0,5 | 2,31 ± 0,22 | 1,14 ± 0,03 | −0,010 ± 0,002 | −0,001 ± 0,000 | 30,6 / 18,9 | 2,0 / 0,0 |
| all | 3 | 58,1 ± 4,1 | 9,4 ± 0,9 | 4,74 ± 0,58 | 1,36 ± 0,03 | −0,020 ± 0,003 | −0,002 ± 0,000 | 63,4 / 56,6 | 5,9 / 1,0 |
| all | 5 | 68,2 ± 3,0 | 12,9 ± 0,7 | 6,41 ± 0,58 | 1,54 ± 0,02 | −0,025 ± 0,004 | −0,003 ± 0,000 | 80,4 / 77,8 | 8,8 / 2,2 |
| correct | 1 | 24,0 ± 2,9 | 2,5 ± 0,3 | 1,70 ± 0,10 | 1,06 ± 0,01 | −0,011 ± 0,002 | −0,001 ± 0,000 | 30,1 / 20,9 | 1,9 / 0,1 |
| correct | 3 | 45,4 ± 3,7 | 6,1 ± 0,6 | 3,33 ± 0,28 | 1,17 ± 0,02 | −0,023 ± 0,003 | −0,002 ± 0,000 | 61,6 / 57,3 | 6,0 / 1,6 |
| correct | 5 | 55,6 ± 2,7 | 8,7 ± 0,6 | 4,54 ± 0,26 | 1,27 ± 0,02 | −0,029 ± 0,003 | −0,003 ± 0,001 | 77,4 / 76,3 | 8,9 / 3,0 |
| wrong | 1 | 44,4 ± 6,6 | 6,2 ± 0,7 | 2,89 ± 0,33 | 1,23 ± 0,04 | −0,008 ± 0,002 | −0,001 ± 0,000 | 31,3 / 17,0 | 2,1 / −0,1 |
| wrong | 3 | 70,3 ± 4,4 | 12,5 ± 1,1 | 6,09 ± 0,85 | 1,54 ± 0,05 | −0,016 ± 0,003 | −0,001 ± 0,000 | 66,2 / 55,5 | 5,7 / 0,6 |
| wrong | 5 | 80,3 ± 3,3 | 17,0 ± 0,7 | 8,19 ± 0,88 | 1,81 ± 0,02 | −0,021 ± 0,004 | −0,002 ± 0,000 | 85,0 / 79,8 | 8,7 / 1,3 |

**Fid− (chỉ giữ top-k).**

| Nhóm | k | Mất hạng 1: top (%) | Ngẫu nhiên (%) | Hạng: top | Ngẫu nhiên | Δ điểm: top | Ngẫu nhiên | Hiệu ứng giữ lại, top: tỉ số tổng / trung vị (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 51,2 ± 6,3 | 69,4 ± 7,2 | 4,19 ± 0,84 | 10,16 ± 2,27 | −0,017 ± 0,004 | −0,032 ± 0,008 | 45,7 / 47,4 | −2,9 / −7,4 |
| all | 3 | 21,8 ± 6,7 | 66,5 ± 5,0 | 2,22 ± 0,74 | 8,15 ± 1,41 | 0,000 ± 0,003 | −0,027 ± 0,005 | 99,7 / 106,4 | 12,2 / 9,9 |
| all | 5 | 14,2 ± 5,7 | 62,3 ± 3,6 | 1,86 ± 0,54 | 6,95 ± 0,96 | 0,005 ± 0,003 | −0,024 ± 0,003 | 116,6 / 122,6 | 24,0 / 21,8 |
| correct | 1 | 44,5 ± 6,6 | 64,4 ± 7,3 | 3,25 ± 0,58 | 8,14 ± 1,60 | −0,023 ± 0,005 | −0,039 ± 0,008 | 39,6 / 41,6 | −3,4 / −9,0 |
| correct | 3 | 14,4 ± 4,5 | 58,8 ± 4,9 | 1,56 ± 0,31 | 6,12 ± 0,93 | −0,001 ± 0,002 | −0,033 ± 0,005 | 97,1 / 103,7 | 13,2 / 11,3 |
| correct | 5 | 8,0 ± 3,5 | 52,8 ± 3,4 | 1,32 ± 0,20 | 4,94 ± 0,57 | 0,006 ± 0,002 | −0,028 ± 0,003 | 115,1 / 120,7 | 26,3 / 24,5 |
| wrong | 1 | 57,6 ± 6,1 | 74,2 ± 7,0 | 5,09 ± 1,14 | 12,10 ± 2,88 | −0,011 ± 0,003 | −0,025 ± 0,008 | 55,0 / 55,9 | −2,0 / −5,9 |
| wrong | 3 | 28,9 ± 9,0 | 73,9 ± 5,1 | 2,86 ± 1,15 | 10,09 ± 1,85 | 0,001 ± 0,003 | −0,022 ± 0,005 | 103,7 / 110,0 | 10,7 / 7,6 |
| wrong | 5 | 20,1 ± 7,7 | 71,4 ± 3,9 | 2,37 ± 0,88 | 8,86 ± 1,34 | 0,005 ± 0,003 | −0,019 ± 0,003 | 118,7 / 125,0 | 20,5 / 17,8 |

**Fid k = 1 theo loại phần đứng đầu E** (ngẫu nhiên: một phần cùng loại).

| Nhóm | Top-1 của E | Câu (TB/seed) | Bỏ top-1: mất hạng 1 (%) | Ngẫu nhiên (%) | Chỉ giữ top-1: hạng | Ngẫu nhiên |
| --- | --- | --- | --- | --- | --- | --- |
| all | node | 5982 | 46,5 ± 0,8 | 4,7 ± 0,1 | 4,21 ± 0,73 | 14,48 ± 1,71 |
| all | triple | 4618 | 19,7 ± 3,2 | 4,2 ± 1,0 | 3,70 ± 1,80 | 4,02 ± 1,95 |
| correct | node | 3247 | 32,4 ± 0,5 | 2,9 ± 0,2 | 3,45 ± 0,55 | 11,09 ± 1,23 |
| correct | triple | 1937 | 10,1 ± 1,1 | 1,8 ± 0,3 | 2,61 ± 1,12 | 2,85 ± 1,29 |
| wrong | node | 2735 | 63,6 ± 3,3 | 6,8 ± 0,2 | 5,11 ± 0,82 | 18,58 ± 1,96 |
| wrong | triple | 2681 | 26,7 ± 4,8 | 5,9 ± 1,6 | 4,49 ± 2,30 | 4,86 ± 2,43 |

**Hình dạng của E.**

| Nhóm | Node / ảnh | Bộ ba / ảnh | Bộ ba trong top-1 / 3 / 5 của E (%) | Top-3 node chiếm (% term dương kênh vật thể) | Top-3 bộ ba chiếm (% kênh bộ ba) | Cách biệt hạng 1 – hạng 2 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 25,3 | 18,1 | 43,6 / 50,2 / 56,1 | 68,0 ± 0,5 | 38,7 ± 1,3 | 0,017 ± 0,001 |
| correct | 25,4 | 18,3 | 37,3 / 44,0 / 50,7 | 68,9 ± 0,5 | 40,2 ± 0,9 | 0,027 ± 0,002 |
| wrong | 25,3 | 17,9 | 49,5 / 56,1 / 61,2 | 67,1 ± 0,6 | 37,2 ± 1,7 | 0,007 ± 0,001 |

**Tính cộng của E:** tổng term của top-k (dự đoán) so với điểm mất thật khi bỏ top-k.

| Nhóm | k | Câu | Dự đoán (TB) | Đo được (TB) | Tỉ số tổng | Tỉ số trung vị | Spearman | Sai lệch tuyệt đối / hiệu ứng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1 | 10520 | 0,05 ± 0,00 | 0,010 ± 0,002 | 0,203 ± 0,052 | 0,115 ± 0,076 | 0,430 ± 0,138 | 2,1 ± 0,2 |
| all | 3 | 10520 | 0,11 ± 0,01 | 0,020 ± 0,003 | 0,189 ± 0,045 | 0,169 ± 0,048 | 0,402 ± 0,149 | 4,8 ± 0,7 |
| all | 5 | 10520 | 0,14 ± 0,01 | 0,025 ± 0,004 | 0,175 ± 0,038 | 0,164 ± 0,042 | 0,354 ± 0,142 | 6,7 ± 1,0 |
| correct | 1 | 5152 | 0,05 ± 0,00 | 0,012 ± 0,002 | 0,251 ± 0,048 | 0,170 ± 0,081 | 0,449 ± 0,134 | 1,4 ± 0,2 |
| correct | 3 | 5152 | 0,10 ± 0,00 | 0,024 ± 0,003 | 0,230 ± 0,041 | 0,212 ± 0,041 | 0,502 ± 0,139 | 3,2 ± 0,4 |
| correct | 5 | 5152 | 0,14 ± 0,01 | 0,030 ± 0,003 | 0,212 ± 0,035 | 0,201 ± 0,038 | 0,488 ± 0,135 | 4,4 ± 0,6 |
| wrong | 1 | 5368 | 0,05 ± 0,00 | 0,008 ± 0,002 | 0,159 ± 0,053 | 0,082 ± 0,061 | 0,425 ± 0,127 | 2,8 ± 0,3 |
| wrong | 3 | 5368 | 0,11 ± 0,01 | 0,016 ± 0,003 | 0,151 ± 0,046 | 0,126 ± 0,054 | 0,339 ± 0,135 | 6,4 ± 0,9 |
| wrong | 5 | 5368 | 0,15 ± 0,02 | 0,021 ± 0,004 | 0,141 ± 0,040 | 0,129 ± 0,045 | 0,272 ± 0,120 | 8,9 ± 1,4 |

**Sparsity.**

| Nhóm | Phần cho 80% / 90% khối term dương | % số phần (80 / 90) | k nhỏ nhất giữ 80% hiệu ứng | k cho 90% | Đạt trong k ≤ 10: 80% / 90% (% câu) | Dòng demo cho 80% / 90% biên |
| --- | --- | --- | --- | --- | --- | --- |
| all | 15,0 ± 0,3 / 19,6 ± 0,5 | 35,4 / 45,9 | 2,78 ± 0,27 | 3,06 ± 0,26 | 94,5 / 92,9 | 2,52 ± 0,04 / 3,06 ± 0,05 |
| correct | 14,7 ± 0,2 / 19,6 ± 0,3 | 34,8 / 45,8 | 2,88 ± 0,23 | 3,20 ± 0,22 | 96,2 / 94,7 | 2,66 ± 0,05 / 3,25 ± 0,07 |
| wrong | 15,2 ± 0,5 / 19,5 ± 0,7 | 36,0 / 46,1 | 2,68 ± 0,32 | 2,92 ± 0,31 | 92,8 / 91,1 | 2,39 ± 0,03 / 2,89 ± 0,03 |

**Phản thực tế.**

| Nhóm | k_flip đạt (≤ 10, %) | k_flip TB | k_flip trung vị | Câu có ảnh vàng khác hạng 1 | Hạng ảnh vàng trước | Sau bỏ top-1 / 3 / 5 | Ảnh vàng lên hạng: k = 1 / 3 / 5 (%) | Ảnh vàng lên hạng 1: k = 1 / 3 / 5 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80,2 ± 1,5 | 2,78 ± 0,22 | 2,0 | 5069 | 8,16 ± 0,10 | 7,94 / 7,74 / 7,64 | 22,3 / 42,4 / 51,9 | 11,8 / 18,2 / 20,6 |
| correct | 68,0 ± 1,9 | 3,14 ± 0,17 | 2,0 | 0 | n/a | n/a | n/a | n/a |
| wrong | 91,8 ± 1,1 | 2,53 ± 0,25 | 1,7 | 5069 | 8,16 ± 0,10 | 7,94 / 7,74 / 7,64 | 22,3 / 42,4 / 51,9 | 11,8 / 18,2 / 20,6 |

**Phản thực tế nhỏ nhất** (tập con so sánh: mọi tập ≤ 2 phần, CF-GNNExplainer, và k_flip của từng cách giải thích như một phản thực tế tham lam).

| Nhóm | Câu | Có tập ≤ 2 phần (%) | Cỡ 1 / cỡ 2 (%) | Số tập nhỏ nhất (TB) | CF-GNNExplainer tìm được (%) | Cỡ TB / trung vị | % số phần | Bằng cỡ nhỏ nhất (%) | Thừa so với nhỏ nhất | Ảnh sai: gold lên hạng 1 với ≤ 2 phần (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 196 | 74,2 ± 1,9 | 53,5 / 20,7 | 2,7 ± 0,3 | 45,7 ± 4,3 | 2,3 ± 0,2 / 1,3 | 6,2 ± 0,3 | 65,8 ± 4,6 | 0,71 ± 0,26 | 19,4 ± 4,2 |
| correct | 91 | 62,5 ± 2,8 | 37,9 / 24,6 | 2,0 ± 0,1 | 41,6 ± 2,2 | 2,4 ± 0,1 / 2,0 | 6,0 ± 0,2 | 63,8 ± 7,2 | 0,61 ± 0,24 | n/a |
| wrong | 106 | 84,2 ± 3,3 | 66,9 / 17,3 | 3,2 ± 0,4 | 49,1 ± 6,0 | 2,2 ± 0,3 / 1,0 | 6,4 ± 0,5 | 67,4 ± 3,1 | 0,79 ± 0,29 | 19,4 ± 4,2 |

**Phản thực tế tham lam theo thứ tự của từng cách** (bỏ dần theo thứ tự đến khi mất hạng 1, k ≤ 10).

| Nhóm | Cách | Tìm được (%) | Cỡ TB | Bằng cỡ nhỏ nhất (%) | Thừa so với nhỏ nhất |
| --- | --- | --- | --- | --- | --- |
| all | model | 84,2 ± 3,6 | 2,65 ± 0,25 | 59,7 ± 13,8 | 0,95 ± 0,32 |
| all | random | 32,1 ± 1,4 | 5,02 ± 0,53 | 18,0 ± 5,9 | 3,81 ± 0,59 |
| all | lowest | 13,2 ± 1,7 | 5,39 ± 0,68 | 15,1 ± 11,2 | 4,24 ± 0,88 |
| all | clip | 75,0 ± 1,4 | 4,05 ± 0,10 | 22,0 ± 1,1 | 2,50 ± 0,03 |
| all | gradient | 44,2 ± 9,3 | 3,70 ± 0,57 | 36,2 ± 4,2 | 2,20 ± 0,43 |
| all | integrated_gradients | 89,6 ± 0,3 | 2,32 ± 0,10 | 81,2 ± 3,9 | 0,32 ± 0,06 |
| all | occlusion | 83,0 ± 2,7 | 2,00 ± 0,01 | 90,6 ± 3,2 | 0,27 ± 0,11 |
| all | gnnexplainer | 83,5 ± 3,0 | 2,65 ± 0,11 | 56,2 ± 3,8 | 0,90 ± 0,08 |
| all | pgexplainer | 81,5 ± 2,9 | 3,15 ± 0,11 | 48,5 ± 0,6 | 1,40 ± 0,06 |
| all | gnn_lrp | 79,6 ± 3,8 | 2,91 ± 0,14 | 57,1 ± 4,3 | 1,18 ± 0,21 |
| all | graphsvx | 89,3 ± 2,2 | 2,21 ± 0,02 | 81,3 ± 4,6 | 0,26 ± 0,05 |
| all | subgraphx | 59,8 ± 2,9 | 2,04 ± 0,09 | 78,3 ± 2,1 | 0,55 ± 0,03 |
| all | shapley | 88,1 ± 2,5 | 2,28 ± 0,02 | 77,5 ± 3,6 | 0,41 ± 0,11 |
| all | cf_gnnexplainer | 69,3 ± 2,0 | 2,59 ± 0,15 | 70,6 ± 5,2 | 0,99 ± 0,19 |
| all | graphmask | 35,3 ± 4,0 | 5,50 ± 0,35 | 15,2 ± 6,1 | 4,23 ± 0,32 |
| all | gstarx | 73,0 ± 8,4 | 3,23 ± 0,28 | 46,9 ± 6,2 | 1,56 ± 0,38 |
| correct | model | 74,3 ± 7,9 | 2,95 ± 0,27 | 59,7 ± 12,1 | 0,95 ± 0,34 |
| correct | random | 20,9 ± 4,7 | 5,20 ± 0,42 | 15,0 ± 4,3 | 3,87 ± 0,36 |
| correct | lowest | 3,7 ± 1,4 | 4,00 ± 1,98 | 19,4 ± 17,3 | 3,19 ± 2,12 |
| correct | clip | 64,0 ± 3,0 | 4,49 ± 0,27 | 24,0 ± 2,5 | 2,76 ± 0,13 |
| correct | gradient | 30,6 ± 11,4 | 3,30 ± 0,33 | 39,5 ± 14,2 | 1,80 ± 0,25 |
| correct | integrated_gradients | 83,1 ± 2,3 | 2,76 ± 0,03 | 78,6 ± 4,1 | 0,34 ± 0,08 |
| correct | occlusion | 71,4 ± 5,4 | 2,32 ± 0,21 | 90,2 ± 2,0 | 0,30 ± 0,15 |
| correct | gnnexplainer | 74,4 ± 7,7 | 2,91 ± 0,33 | 54,2 ± 4,2 | 0,86 ± 0,19 |
| correct | pgexplainer | 72,8 ± 3,1 | 3,72 ± 0,20 | 45,1 ± 3,4 | 1,62 ± 0,12 |
| correct | gnn_lrp | 71,4 ± 3,8 | 3,12 ± 0,38 | 57,7 ± 8,3 | 1,07 ± 0,38 |
| correct | graphsvx | 82,8 ± 3,6 | 2,58 ± 0,04 | 80,2 ± 4,4 | 0,27 ± 0,07 |
| correct | subgraphx | 47,5 ± 3,0 | 2,19 ± 0,24 | 76,1 ± 7,1 | 0,54 ± 0,23 |
| correct | shapley | 81,0 ± 3,9 | 2,55 ± 0,14 | 79,0 ± 4,2 | 0,36 ± 0,04 |
| correct | cf_gnnexplainer | 60,7 ± 2,2 | 2,82 ± 0,08 | 69,2 ± 5,0 | 0,92 ± 0,06 |
| correct | graphmask | 25,7 ± 2,9 | 5,92 ± 0,64 | 10,1 ± 4,2 | 4,52 ± 0,76 |
| correct | gstarx | 65,3 ± 8,6 | 3,69 ± 0,50 | 43,0 ± 13,7 | 1,74 ± 0,73 |
| wrong | model | 92,7 ± 1,3 | 2,44 ± 0,27 | 60,1 ± 15,4 | 0,94 ± 0,33 |
| wrong | random | 41,6 ± 6,2 | 4,93 ± 0,63 | 18,8 ± 8,4 | 3,80 ± 0,69 |
| wrong | lowest | 21,4 ± 2,6 | 5,60 ± 0,51 | 14,4 ± 11,5 | 4,41 ± 0,68 |
| wrong | clip | 84,5 ± 0,6 | 3,76 ± 0,04 | 21,0 ± 3,1 | 2,33 ± 0,12 |
| wrong | gradient | 55,8 ± 7,2 | 3,85 ± 0,78 | 35,1 ± 4,3 | 2,37 ± 0,60 |
| wrong | integrated_gradients | 95,3 ± 1,8 | 1,99 ± 0,20 | 82,9 ± 4,1 | 0,31 ± 0,11 |
| wrong | occlusion | 93,1 ± 0,4 | 1,79 ± 0,13 | 91,1 ± 5,4 | 0,25 ± 0,14 |
| wrong | gnnexplainer | 91,5 ± 2,0 | 2,47 ± 0,28 | 57,2 ± 5,9 | 0,93 ± 0,23 |
| wrong | pgexplainer | 88,9 ± 2,5 | 2,76 ± 0,29 | 50,6 ± 1,7 | 1,26 ± 0,17 |
| wrong | gnn_lrp | 86,7 ± 6,6 | 2,76 ± 0,04 | 56,6 ± 1,7 | 1,25 ± 0,10 |
| wrong | graphsvx | 94,9 ± 1,2 | 1,93 ± 0,05 | 82,0 ± 4,8 | 0,25 ± 0,05 |
| wrong | subgraphx | 70,3 ± 2,3 | 1,95 ± 0,01 | 79,4 ± 4,5 | 0,57 ± 0,09 |
| wrong | shapley | 94,3 ± 1,1 | 2,08 ± 0,13 | 76,6 ± 4,0 | 0,45 ± 0,16 |
| wrong | cf_gnnexplainer | 76,6 ± 3,7 | 2,44 ± 0,30 | 71,3 ± 5,4 | 1,03 ± 0,28 |
| wrong | graphmask | 43,6 ± 5,5 | 5,28 ± 0,83 | 17,7 ± 10,8 | 4,09 ± 0,83 |
| wrong | gstarx | 79,7 ± 9,6 | 2,90 ± 0,10 | 49,7 ± 3,9 | 1,42 ± 0,17 |

**Mức message: GraphMask so với attention GAT** (ngừng truyền top-k message; phần trăm của điểm mất khi ngừng mọi message, chỉ ở câu mà message giúp ảnh).

| Nhóm | Câu | Message giúp ảnh (%) | Δ điểm (sau − trước) khi ngừng mọi message | Message / ảnh | Thứ tự | AUC mất điểm (cao tốt) | Mất k = 1 / 3 / 5 (%) | Điểm mất (trước − sau) k = 1 / 3 / 5 | p Wilcoxon so với GraphMask (lớn nhất qua seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 983 | 12,2 | 0,0079 ± 0,0014 | 188 | graphmask | −0,123 ± 0,080 | −1,3 / −13,2 / −17,2 | −0,0000 / −0,0001 / −0,0002 |  |
| all | 983 | 12,2 | 0,0079 ± 0,0014 | 188 | attention | −0,232 ± 0,703 | −9,5 / −24,4 / −32,2 | −0,0005 / −0,0014 / −0,0020 | 1,3e−01 |
| all | 983 | 12,2 | 0,0079 ± 0,0014 | 188 | random | −0,187 ± 0,089 | −3,8 / −4,7 / −16,2 | −0,0001 / −0,0003 / −0,0005 | 6,6e−01 |
| correct | 476 | 15,2 | 0,0069 ± 0,0012 | 188 | graphmask | −0,147 ± 0,096 | −1,5 / −11,7 / −18,6 | −0,0000 / −0,0001 / −0,0002 |  |
| correct | 476 | 15,2 | 0,0069 ± 0,0012 | 188 | attention | −0,092 ± 0,581 | −8,9 / −24,2 / −24,9 | −0,0005 / −0,0013 / −0,0018 | 2,3e−01 |
| correct | 476 | 15,2 | 0,0069 ± 0,0012 | 188 | random | −0,166 ± 0,039 | −6,2 / −5,3 / −16,7 | −0,0001 / −0,0003 / −0,0005 | 7,6e−01 |
| wrong | 507 | 9,5 | 0,0089 ± 0,0016 | 187 | graphmask | −0,126 ± 0,325 | −0,8 / −20,4 / −18,0 | −0,0000 / −0,0001 / −0,0001 |  |
| wrong | 507 | 9,5 | 0,0089 ± 0,0016 | 187 | attention | −0,498 ± 0,901 | −8,2 / −31,3 / −47,9 | −0,0005 / −0,0014 / −0,0022 | 9,0e−01 |
| wrong | 507 | 9,5 | 0,0089 ± 0,0016 | 187 | random | −0,258 ± 0,311 | −0,4 / −2,0 / −18,9 | −0,0001 / −0,0003 / −0,0005 | 7,8e−01 |

**Validity ngữ nghĩa** (dòng vật thể và dòng bộ ba của E).

| Nhóm | Dòng vật thể | strict (%) | alias | WordNet strict | lenient (như v1) | WordNet lenient | CLIP ≥ 0,955 | Cùng danh từ chính | Dòng có số | Khớp nhóm số (%) | Dòng bộ ba | Chủ thể | Đối tượng | Vị từ | Cả ba (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 29687 | 63,8 ± 0,9 | 57,0 | 51,2 | 70,0 ± 1,0 | 54,6 | 63,7 | 62,1 | 5857 | 66,7 ± 0,8 | 18532 | 35,8 | 25,8 | 10,5 | 2,9 ± 0,6 |
| correct | 15191 | 65,8 ± 0,6 | 59,2 | 53,3 | 71,8 ± 0,8 | 56,8 | 65,6 | 64,5 | 3078 | 66,1 ± 0,9 | 9649 | 38,1 | 28,9 | 10,9 | 3,2 ± 0,7 |
| wrong | 14496 | 61,8 ± 1,3 | 54,7 | 49,0 | 68,1 ± 1,3 | 52,4 | 61,6 | 59,7 | 2779 | 67,3 ± 0,7 | 8884 | 33,4 | 22,4 | 10,1 | 2,5 ± 0,6 |

**Đồ thị con theo alpha GAT** (messages: cắt đường truyền tin, giữ bộ ba; edges: cách v1, bỏ cả bộ ba).

| Nhóm | Tập câu | Câu (TB/seed) | Cạnh bỏ / cạnh vào | Dạng | Δ điểm: top | Ngẫu nhiên | Mất hạng 1: top (%) | Ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | có cạnh vào | 10600 | 8,97 / 20,95 | messages | 0,0025 ± 0,0006 | 0,0021 ± 0,0003 | 3,6 ± 1,5 | 2,0 ± 0,9 |
| all | có cạnh vào | 10600 | 8,97 / 20,95 | edges | 0,0012 ± 0,0004 | 0,0013 ± 0,0002 | 6,9 ± 1,8 | 4,0 ± 0,9 |
| all | có lựa chọn | 10481 | 9,00 / 21,11 | messages | 0,0025 ± 0,0006 | 0,0021 ± 0,0003 | 3,5 ± 1,5 | 1,8 ± 0,9 |
| all | có lựa chọn | 10481 | 9,00 / 21,11 | edges | 0,0013 ± 0,0005 | 0,0014 ± 0,0002 | 6,6 ± 1,8 | 3,6 ± 0,9 |
| correct | có cạnh vào | 5184 | 8,98 / 21,13 | messages | 0,0023 ± 0,0006 | 0,0020 ± 0,0003 | 1,6 ± 0,8 | 0,8 ± 0,5 |
| correct | có cạnh vào | 5184 | 8,98 / 21,13 | edges | 0,0008 ± 0,0004 | 0,0012 ± 0,0001 | 3,4 ± 0,7 | 1,7 ± 0,4 |
| correct | có lựa chọn | 5151 | 9,00 / 21,23 | messages | 0,0023 ± 0,0006 | 0,0020 ± 0,0003 | 1,6 ± 0,8 | 0,8 ± 0,4 |
| correct | có lựa chọn | 5151 | 9,00 / 21,23 | edges | 0,0008 ± 0,0004 | 0,0012 ± 0,0001 | 3,4 ± 0,8 | 1,7 ± 0,4 |
| wrong | có cạnh vào | 5416 | 8,96 / 20,77 | messages | 0,0027 ± 0,0006 | 0,0021 ± 0,0003 | 5,5 ± 2,3 | 3,1 ± 1,4 |
| wrong | có cạnh vào | 5416 | 8,96 / 20,77 | edges | 0,0015 ± 0,0005 | 0,0014 ± 0,0002 | 10,3 ± 2,8 | 6,1 ± 1,4 |
| wrong | có lựa chọn | 5330 | 9,00 / 21,00 | messages | 0,0027 ± 0,0006 | 0,0021 ± 0,0003 | 5,3 ± 2,2 | 2,8 ± 1,3 |
| wrong | có lựa chọn | 5330 | 9,00 / 21,00 | edges | 0,0017 ± 0,0005 | 0,0016 ± 0,0002 | 9,8 ± 2,8 | 5,4 ± 1,3 |

**So sánh các cách giải thích** (mẫu cố định trong val; GNNExplainer, GraphSVX, SubgraphX và Shapley trên tập con).

| Nhóm | Cách | Câu | Giây / câu (CPU) | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | k_flip đạt (%) | Bỏ top-1 / 3 / 5: mất hạng 1 (%) | Chỉ giữ top-1 / 3 / 5: hạng | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Fid_α+ / Fid_α− (k = 5, α = 0,5) | Khớp Shapley: Spearman | Top-5 trùng Shapley (%) | p Wilcoxon deletion so với model (lớn nhất qua seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | model | 983 | 0 | 0,070 ± 0,128 | 1,262 ± 0,175 | 79,8 ± 2,0 | 33,5 / 57,2 / 67,8 | 3,95 / 2,11 / 1,77 | 0,076 ± 0,007 / −0,032 ± 0,008 | 0,409 ± 0,055 / −0,141 ± 0,066 | 0,605 ± 0,048 | 56,5 ± 7,9 | n/a |
| all | random | 983 | 0 | 0,891 ± 0,012 | 0,206 ± 0,141 | 28,5 ± 1,2 | 4,2 / 9,4 / 14,1 | 9,87 / 8,40 / 6,48 | 0,011 ± 0,002 / 0,028 ± 0,004 | 0,050 ± 0,010 / 0,280 ± 0,043 | 0,004 ± 0,009 | 16,9 ± 0,7 | 4,0e−154 |
| all | lowest | 983 | 0 | 1,064 ± 0,014 | −1,309 ± 0,151 | 11,0 ± 2,5 | 1,4 / 3,4 / 4,5 | 24,40 / 22,64 / 21,52 | −0,003 ± 0,002 / 0,044 ± 0,005 | −0,025 ± 0,008 / 0,426 ± 0,046 | −0,605 ± 0,048 | 3,4 ± 1,2 | 1,3e−161 |
| all | clip | 983 | 0,000 | 0,387 ± 0,031 | 0,899 ± 0,143 | 68,6 ± 0,8 | 14,3 / 35,9 / 48,4 | 3,58 / 3,12 / 2,47 | 0,049 ± 0,002 / −0,006 ± 0,003 | 0,264 ± 0,019 / 0,030 ± 0,030 | 0,424 ± 0,040 | 40,3 ± 1,5 | 5,9e−54 |
| all | gradient | 983 | 0,011 | 0,759 ± 0,165 | 0,241 ± 0,160 | 39,4 ± 9,5 | 11,2 / 19,9 / 25,9 | 11,20 / 6,97 / 5,82 | 0,020 ± 0,010 / 0,022 ± 0,008 | 0,104 ± 0,073 / 0,206 ± 0,066 | −0,207 ± 0,077 | 24,0 ± 5,3 | 4,6e−136 |
| all | integrated_gradients | 983 | 0,066 | −0,220 ± 0,068 | 1,238 ± 0,052 | 86,2 ± 1,1 | 43,2 / 67,9 / 76,8 | 4,84 / 1,63 / 1,24 | 0,087 ± 0,003 / −0,045 ± 0,004 | 0,492 ± 0,016 / −0,212 ± 0,017 | 0,335 ± 0,057 | 60,4 ± 1,9 | 1,1e−89 |
| all | occlusion | 983 | 0,054 | −0,007 ± 0,056 | 1,447 ± 0,070 | 78,1 ± 1,3 | 49,9 / 65,6 / 70,7 | 5,81 / 1,56 / 1,13 | 0,102 ± 0,003 / −0,065 ± 0,007 | 0,496 ± 0,010 / −0,329 ± 0,030 | 0,446 ± 0,041 | 55,3 ± 1,6 | 4,2e−02 |
| all | gnnexplainer | 196 | 0,953 | 0,107 ± 0,014 | 1,059 ± 0,194 | 83,5 ± 3,0 | 31,9 / 61,8 / 73,7 | 7,24 / 2,82 / 1,82 | 0,070 ± 0,006 / −0,043 ± 0,011 | 0,396 ± 0,011 / −0,142 ± 0,020 | 0,365 ± 0,018 | 57,1 ± 1,0 | 3,0e−02 |
| all | pgexplainer | 983 | 0,006 | 0,213 ± 0,050 | 0,636 ± 0,093 | 77,2 ± 1,0 | 24,2 / 47,9 / 59,2 | 8,68 / 3,95 / 2,75 | 0,051 ± 0,006 / −0,009 ± 0,003 | 0,291 ± 0,017 / 0,030 ± 0,022 | −0,026 ± 0,055 | 42,8 ± 3,1 | 7,0e−01 |
| all | gnn_lrp | 983 | 0,024 | 0,114 ± 0,100 | 0,880 ± 0,135 | 76,8 ± 3,4 | 30,5 / 53,4 / 63,0 | 7,88 / 3,17 / 2,25 | 0,063 ± 0,003 / −0,025 ± 0,005 | 0,368 ± 0,035 / −0,068 ± 0,063 | 0,078 ± 0,050 | 47,4 ± 4,0 | 1,2e−11 |
| all | graphsvx | 196 | 0,414 | −0,243 ± 0,022 | 1,562 ± 0,111 | 89,3 ± 2,2 | 45,7 / 74,9 / 82,4 | 3,86 / 1,23 / 1,05 | 0,084 ± 0,003 / −0,055 ± 0,007 | 0,509 ± 0,028 / −0,252 ± 0,023 | 0,754 ± 0,007 | 73,2 ± 0,4 | 9,2e−18 |
| all | subgraphx | 196 | 4,482 | 0,392 ± 0,046 | 0,949 ± 0,104 | 59,8 ± 2,9 | 38,0 / 50,3 / 52,8 | 5,25 / 2,71 / 2,19 | 0,051 ± 0,003 / −0,019 ± 0,009 | 0,281 ± 0,016 / −0,081 ± 0,030 | 0,024 ± 0,029 | 38,1 ± 1,7 | 1,0e−08 |
| all | shapley | 196 | 0,412 | −0,187 ± 0,027 | 1,542 ± 0,109 | 88,1 ± 2,5 | 44,0 / 71,1 / 80,7 | 3,97 / 1,21 / 1,03 | 0,081 ± 0,001 / −0,051 ± 0,007 | 0,492 ± 0,027 / −0,229 ± 0,019 | n/a | n/a | 2,5e−07 |
| all | cf_gnnexplainer | 196 | 1,939 | 0,332 ± 0,049 | 0,679 ± 0,156 | 69,3 ± 2,0 | 36,8 / 52,1 / 58,1 | 8,38 / 5,10 / 3,94 | 0,051 ± 0,009 / −0,023 ± 0,011 | 0,294 ± 0,026 / −0,031 ± 0,037 | 0,182 ± 0,033 | 38,4 ± 2,3 | 5,1e−03 |
| all | graphmask | 983 | 0,008 | 0,883 ± 0,036 | −0,279 ± 0,168 | 32,0 ± 4,8 | 4,5 / 9,9 / 14,4 | 17,40 / 12,93 / 10,30 | 0,009 ± 0,004 / 0,030 ± 0,003 | 0,038 ± 0,026 / 0,309 ± 0,047 | −0,172 ± 0,038 | 12,3 ± 2,4 | 2,1e−151 |
| all | gstarx | 196 | 0,108 | 0,161 ± 0,149 | 1,385 ± 0,127 | 73,0 ± 8,4 | 23,1 / 47,7 / 58,1 | 2,53 / 1,45 / 1,24 | 0,046 ± 0,005 / −0,022 ± 0,009 | 0,324 ± 0,051 / −0,023 ± 0,029 | 0,653 ± 0,008 | 51,6 ± 5,6 | 2,1e−02 |
| correct | model | 476 | 0 | 0,092 ± 0,153 | 1,243 ± 0,123 | 67,1 ± 2,0 | 23,3 / 44,4 / 54,5 | 2,98 / 1,44 / 1,28 | 0,069 ± 0,005 / −0,031 ± 0,006 | 0,395 ± 0,065 / −0,147 ± 0,062 | 0,624 ± 0,045 | 59,1 ± 7,8 | n/a |
| correct | random | 476 | 0 | 0,904 ± 0,008 | 0,235 ± 0,109 | 19,9 ± 0,4 | 2,8 / 5,7 / 9,1 | 7,19 / 5,60 / 4,34 | 0,007 ± 0,002 / 0,030 ± 0,005 | 0,040 ± 0,002 / 0,294 ± 0,055 | 0,006 ± 0,021 | 15,9 ± 1,5 | 1,5e−76 |
| correct | lowest | 476 | 0 | 1,064 ± 0,005 | −0,990 ± 0,238 | 5,3 ± 1,4 | 0,4 / 1,3 / 1,9 | 19,04 / 17,33 / 16,28 | −0,005 ± 0,000 / 0,043 ± 0,001 | −0,029 ± 0,000 / 0,416 ± 0,066 | −0,624 ± 0,045 | 2,4 ± 0,3 | 1,0e−77 |
| correct | clip | 476 | 0,000 | 0,461 ± 0,047 | 0,835 ± 0,062 | 54,4 ± 0,9 | 8,7 / 21,7 / 33,5 | 2,54 / 2,30 / 1,87 | 0,040 ± 0,005 / −0,001 ± 0,004 | 0,230 ± 0,041 / 0,064 ± 0,004 | 0,438 ± 0,041 | 39,3 ± 3,9 | 7,4e−40 |
| correct | gradient | 476 | 0,011 | 0,758 ± 0,190 | 0,315 ± 0,135 | 25,8 ± 7,5 | 6,1 / 12,7 / 17,8 | 8,04 / 5,07 / 4,15 | 0,019 ± 0,013 / 0,024 ± 0,008 | 0,106 ± 0,089 / 0,197 ± 0,067 | −0,214 ± 0,082 | 23,6 ± 7,6 | 3,7e−69 |
| correct | integrated_gradients | 476 | 0,066 | −0,113 ± 0,143 | 1,203 ± 0,022 | 76,6 ± 1,5 | 28,6 / 53,5 / 65,1 | 3,55 / 1,31 / 1,10 | 0,075 ± 0,004 / −0,039 ± 0,002 | 0,446 ± 0,036 / −0,187 ± 0,029 | 0,364 ± 0,048 | 63,4 ± 2,7 | 8,5e−43 |
| correct | occlusion | 476 | 0,054 | 0,093 ± 0,098 | 1,352 ± 0,050 | 63,2 ± 1,3 | 32,7 / 50,7 / 55,5 | 4,05 / 1,28 / 1,06 | 0,085 ± 0,003 / −0,053 ± 0,005 | 0,439 ± 0,036 / −0,270 ± 0,019 | 0,432 ± 0,045 | 57,4 ± 4,1 | 1,7e−01 |
| correct | gnnexplainer | 91 | 0,952 | 0,211 ± 0,014 | 1,101 ± 0,116 | 74,4 ± 7,7 | 21,7 / 53,7 / 63,6 | 4,37 / 1,59 / 1,07 | 0,065 ± 0,001 / −0,041 ± 0,013 | 0,373 ± 0,013 / −0,154 ± 0,046 | 0,341 ± 0,025 | 61,4 ± 2,6 | 3,0e−01 |
| correct | pgexplainer | 476 | 0,006 | 0,280 ± 0,033 | 0,615 ± 0,110 | 66,3 ± 1,5 | 16,4 / 35,0 / 46,8 | 6,09 / 2,83 / 2,02 | 0,045 ± 0,001 / −0,006 ± 0,001 | 0,265 ± 0,005 / 0,051 ± 0,043 | −0,013 ± 0,049 | 44,1 ± 3,9 | 3,7e−02 |
| correct | gnn_lrp | 476 | 0,024 | 0,186 ± 0,066 | 0,893 ± 0,116 | 65,1 ± 3,6 | 21,5 / 42,4 / 51,6 | 5,22 / 1,97 / 1,50 | 0,057 ± 0,000 / −0,024 ± 0,001 | 0,337 ± 0,011 / −0,051 ± 0,060 | 0,105 ± 0,057 | 51,4 ± 3,5 | 8,9e−06 |
| correct | graphsvx | 91 | 0,415 | −0,109 ± 0,017 | 1,463 ± 0,117 | 82,8 ± 3,6 | 31,7 / 64,7 / 74,4 | 3,20 / 1,17 / 1,01 | 0,065 ± 0,005 / −0,053 ± 0,007 | 0,447 ± 0,025 / −0,217 ± 0,016 | 0,758 ± 0,011 | 75,1 ± 2,0 | 3,6e−08 |
| correct | subgraphx | 91 | 4,477 | 0,497 ± 0,048 | 0,942 ± 0,110 | 47,5 ± 3,0 | 28,7 / 39,0 / 42,4 | 4,04 / 2,13 / 1,85 | 0,039 ± 0,005 / −0,020 ± 0,008 | 0,230 ± 0,004 / −0,054 ± 0,028 | 0,018 ± 0,025 | 38,3 ± 2,0 | 1,3e−06 |
| correct | shapley | 91 | 0,407 | −0,073 ± 0,019 | 1,453 ± 0,104 | 81,0 ± 3,9 | 32,0 / 61,4 / 73,3 | 3,32 / 1,17 / 1,00 | 0,065 ± 0,003 / −0,046 ± 0,008 | 0,439 ± 0,026 / −0,214 ± 0,011 | n/a | n/a | 3,3e−05 |
| correct | cf_gnnexplainer | 91 | 1,964 | 0,380 ± 0,065 | 0,796 ± 0,101 | 60,7 ± 2,2 | 25,3 / 44,4 / 49,5 | 5,95 / 3,25 / 2,36 | 0,040 ± 0,006 / −0,028 ± 0,019 | 0,276 ± 0,027 / −0,048 ± 0,052 | 0,182 ± 0,037 | 41,4 ± 0,7 | 3,0e−02 |
| correct | graphmask | 476 | 0,008 | 0,877 ± 0,040 | −0,112 ± 0,181 | 22,5 ± 5,2 | 2,4 / 6,2 / 9,2 | 12,62 / 9,35 / 7,58 | 0,006 ± 0,003 / 0,030 ± 0,006 | 0,039 ± 0,025 / 0,303 ± 0,087 | −0,189 ± 0,044 | 11,2 ± 2,6 | 5,3e−77 |
| correct | gstarx | 91 | 0,106 | 0,239 ± 0,191 | 1,274 ± 0,126 | 65,3 ± 8,6 | 14,0 / 38,0 / 49,2 | 2,17 / 1,38 / 1,17 | 0,037 ± 0,011 / −0,024 ± 0,009 | 0,287 ± 0,091 / −0,019 ± 0,070 | 0,665 ± 0,012 | 52,9 ± 5,2 | 1,5e−01 |
| wrong | model | 507 | 0 | 0,049 ± 0,108 | 1,280 ± 0,276 | 91,7 ± 1,6 | 43,1 / 69,4 / 80,3 | 4,86 / 2,75 / 2,25 | 0,082 ± 0,010 / −0,033 ± 0,009 | 0,423 ± 0,047 / −0,135 ± 0,084 | 0,588 ± 0,051 | 54,3 ± 8,0 | n/a |
| wrong | random | 507 | 0 | 0,879 ± 0,019 | 0,179 ± 0,201 | 36,6 ± 2,3 | 5,6 / 13,0 / 18,9 | 12,40 / 11,02 / 8,49 | 0,014 ± 0,003 / 0,026 ± 0,011 | 0,059 ± 0,018 / 0,267 ± 0,040 | 0,002 ± 0,011 | 17,8 ± 2,3 | 3,9e−77 |
| wrong | lowest | 507 | 0 | 1,065 ± 0,024 | −1,611 ± 0,107 | 16,5 ± 3,7 | 2,2 / 5,4 / 7,0 | 29,44 / 27,64 / 26,47 | −0,001 ± 0,005 / 0,045 ± 0,010 | −0,021 ± 0,016 / 0,436 ± 0,038 | −0,588 ± 0,051 | 4,1 ± 1,9 | 2,3e−83 |
| wrong | clip | 507 | 0,000 | 0,317 ± 0,016 | 0,960 ± 0,227 | 81,9 ± 0,6 | 19,5 / 49,4 / 62,4 | 4,57 / 3,89 / 3,04 | 0,058 ± 0,003 / −0,012 ± 0,010 | 0,295 ± 0,013 / −0,003 ± 0,062 | 0,411 ± 0,042 | 41,2 ± 1,3 | 1,5e−18 |
| wrong | gradient | 507 | 0,011 | 0,760 ± 0,144 | 0,171 ± 0,189 | 52,2 ± 10,8 | 15,9 / 26,7 / 33,5 | 14,17 / 8,76 / 7,40 | 0,021 ± 0,008 / 0,020 ± 0,008 | 0,102 ± 0,059 / 0,214 ± 0,066 | −0,202 ± 0,074 | 24,3 ± 3,7 | 5,2e−67 |
| wrong | integrated_gradients | 507 | 0,067 | −0,323 ± 0,012 | 1,272 ± 0,121 | 95,1 ± 0,5 | 56,9 / 81,4 / 87,9 | 6,06 / 1,92 / 1,37 | 0,099 ± 0,006 / −0,051 ± 0,010 | 0,535 ± 0,006 / −0,237 ± 0,042 | 0,309 ± 0,068 | 57,8 ± 1,8 | 1,0e−48 |
| wrong | occlusion | 507 | 0,055 | −0,101 ± 0,018 | 1,539 ± 0,158 | 92,0 ± 0,9 | 66,0 / 79,7 / 84,9 | 7,47 / 1,83 / 1,19 | 0,117 ± 0,006 / −0,076 ± 0,017 | 0,550 ± 0,015 / −0,386 ± 0,071 | 0,458 ± 0,040 | 53,5 ± 1,5 | 6,4e−05 |
| wrong | gnnexplainer | 106 | 0,954 | 0,017 ± 0,031 | 1,023 ± 0,279 | 91,5 ± 2,0 | 40,6 / 68,7 / 82,2 | 9,69 / 3,87 / 2,46 | 0,073 ± 0,012 / −0,045 ± 0,011 | 0,415 ± 0,029 / −0,130 ± 0,023 | 0,386 ± 0,014 | 53,4 ± 2,0 | 1,2e−01 |
| wrong | pgexplainer | 507 | 0,006 | 0,150 ± 0,077 | 0,656 ± 0,082 | 87,4 ± 0,7 | 31,6 / 60,0 / 70,9 | 11,09 / 5,00 / 3,43 | 0,057 ± 0,012 / −0,011 ± 0,005 | 0,315 ± 0,031 / 0,010 ± 0,006 | −0,036 ± 0,060 | 41,8 ± 2,6 | 1,9e−01 |
| wrong | gnn_lrp | 507 | 0,024 | 0,046 ± 0,136 | 0,869 ± 0,160 | 87,8 ± 3,8 | 39,0 / 63,8 / 73,7 | 10,38 / 4,28 / 2,95 | 0,068 ± 0,007 / −0,026 ± 0,010 | 0,398 ± 0,059 / −0,085 ± 0,073 | 0,055 ± 0,046 | 44,0 ± 4,4 | 3,5e−07 |
| wrong | graphsvx | 106 | 0,414 | −0,358 ± 0,051 | 1,647 ± 0,132 | 94,9 ± 1,2 | 57,7 / 83,6 / 89,2 | 4,42 / 1,27 / 1,08 | 0,100 ± 0,003 / −0,058 ± 0,008 | 0,563 ± 0,036 / −0,283 ± 0,054 | 0,750 ± 0,012 | 71,6 ± 1,2 | 3,6e−11 |
| wrong | subgraphx | 106 | 4,487 | 0,301 ± 0,063 | 0,955 ± 0,101 | 70,3 ± 2,3 | 46,0 / 60,0 / 61,7 | 6,28 / 3,21 / 2,49 | 0,061 ± 0,002 / −0,019 ± 0,013 | 0,325 ± 0,026 / −0,105 ± 0,036 | 0,029 ± 0,039 | 38,0 ± 2,0 | 7,9e−04 |
| wrong | shapley | 106 | 0,417 | −0,285 ± 0,062 | 1,618 ± 0,127 | 94,3 ± 1,1 | 54,3 / 79,5 / 87,0 | 4,51 / 1,24 / 1,06 | 0,094 ± 0,003 / −0,055 ± 0,007 | 0,537 ± 0,032 / −0,243 ± 0,040 | n/a | n/a | 1,3e−03 |
| wrong | cf_gnnexplainer | 106 | 1,920 | 0,290 ± 0,042 | 0,580 ± 0,205 | 76,6 ± 3,7 | 46,6 / 58,6 / 65,4 | 10,46 / 6,68 / 5,27 | 0,060 ± 0,011 / −0,020 ± 0,005 | 0,309 ± 0,024 / −0,018 ± 0,028 | 0,181 ± 0,034 | 35,8 ± 3,9 | 6,8e−02 |
| wrong | graphmask | 507 | 0,008 | 0,889 ± 0,046 | −0,438 ± 0,216 | 41,0 ± 4,9 | 6,4 / 13,4 / 19,3 | 21,87 / 16,29 / 12,85 | 0,012 ± 0,004 / 0,030 ± 0,003 | 0,036 ± 0,031 / 0,316 ± 0,013 | −0,157 ± 0,037 | 13,3 ± 2,7 | 9,4e−76 |
| wrong | gstarx | 106 | 0,109 | 0,092 ± 0,113 | 1,482 ± 0,153 | 79,7 ± 9,6 | 30,9 / 56,0 / 65,7 | 2,83 / 1,51 / 1,29 | 0,055 ± 0,007 / −0,021 ± 0,013 | 0,357 ± 0,014 / −0,028 ± 0,010 | 0,643 ± 0,006 | 50,5 ± 6,1 | 8,7e−02 |

**Tính cộng của các cách phân rã** (cùng các câu so sánh).

| Nhóm | Cách | k | Tỉ số tổng | Tỉ số trung vị | Spearman | Sai lệch tuyệt đối / hiệu ứng |
| --- | --- | --- | --- | --- | --- | --- |
| all | model | 1 | 0,207 ± 0,052 | 0,116 ± 0,078 | 0,464 ± 0,113 | 1,94 ± 0,41 |
| all | model | 3 | 0,191 ± 0,046 | 0,170 ± 0,058 | 0,441 ± 0,129 | 4,48 ± 0,99 |
| all | model | 5 | 0,177 ± 0,040 | 0,161 ± 0,047 | 0,390 ± 0,131 | 6,30 ± 1,40 |
| all | gradient | 1 | 0,229 ± 0,112 | −0,010 ± 0,033 | 0,075 ± 0,064 | 0,64 ± 0,12 |
| all | gradient | 3 | 0,214 ± 0,123 | 0,036 ± 0,084 | 0,081 ± 0,068 | 1,26 ± 0,24 |
| all | gradient | 5 | 0,235 ± 0,139 | 0,080 ± 0,111 | 0,135 ± 0,086 | 1,64 ± 0,36 |
| all | integrated_gradients | 1 | 1,136 ± 0,037 | 1,066 ± 0,041 | 0,746 ± 0,018 | 0,23 ± 0,00 |
| all | integrated_gradients | 3 | 1,320 ± 0,015 | 1,304 ± 0,039 | 0,814 ± 0,014 | 0,38 ± 0,02 |
| all | integrated_gradients | 5 | 1,462 ± 0,036 | 1,443 ± 0,059 | 0,830 ± 0,013 | 0,49 ± 0,01 |
| all | occlusion | 1 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,00 ± 0,00 |
| all | occlusion | 3 | 0,968 ± 0,008 | 0,990 ± 0,010 | 0,910 ± 0,008 | 0,21 ± 0,02 |
| all | occlusion | 5 | 1,004 ± 0,008 | 0,992 ± 0,009 | 0,875 ± 0,015 | 0,26 ± 0,03 |
| correct | model | 1 | 0,255 ± 0,044 | 0,163 ± 0,079 | 0,498 ± 0,121 | 1,32 ± 0,15 |
| correct | model | 3 | 0,235 ± 0,045 | 0,213 ± 0,055 | 0,525 ± 0,143 | 3,01 ± 0,39 |
| correct | model | 5 | 0,216 ± 0,039 | 0,201 ± 0,044 | 0,505 ± 0,142 | 4,16 ± 0,53 |
| correct | gradient | 1 | 0,262 ± 0,139 | −0,018 ± 0,035 | 0,046 ± 0,047 | 0,57 ± 0,11 |
| correct | gradient | 3 | 0,254 ± 0,158 | 0,057 ± 0,128 | 0,095 ± 0,083 | 1,05 ± 0,05 |
| correct | gradient | 5 | 0,277 ± 0,166 | 0,107 ± 0,148 | 0,153 ± 0,106 | 1,30 ± 0,11 |
| correct | integrated_gradients | 1 | 1,131 ± 0,033 | 1,069 ± 0,043 | 0,752 ± 0,020 | 0,21 ± 0,01 |
| correct | integrated_gradients | 3 | 1,292 ± 0,019 | 1,304 ± 0,020 | 0,824 ± 0,003 | 0,33 ± 0,03 |
| correct | integrated_gradients | 5 | 1,429 ± 0,033 | 1,443 ± 0,065 | 0,835 ± 0,011 | 0,41 ± 0,01 |
| correct | occlusion | 1 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,00 ± 0,00 |
| correct | occlusion | 3 | 0,967 ± 0,020 | 0,986 ± 0,003 | 0,925 ± 0,003 | 0,17 ± 0,01 |
| correct | occlusion | 5 | 0,998 ± 0,023 | 0,986 ± 0,012 | 0,893 ± 0,006 | 0,21 ± 0,02 |
| wrong | model | 1 | 0,167 ± 0,059 | 0,088 ± 0,071 | 0,457 ± 0,092 | 2,53 ± 0,67 |
| wrong | model | 3 | 0,154 ± 0,047 | 0,129 ± 0,063 | 0,408 ± 0,100 | 5,89 ± 1,61 |
| wrong | model | 5 | 0,143 ± 0,040 | 0,131 ± 0,046 | 0,337 ± 0,105 | 8,34 ± 2,27 |
| wrong | gradient | 1 | 0,199 ± 0,092 | −0,002 ± 0,029 | 0,103 ± 0,083 | 0,71 ± 0,25 |
| wrong | gradient | 3 | 0,179 ± 0,098 | 0,019 ± 0,053 | 0,072 ± 0,064 | 1,47 ± 0,46 |
| wrong | gradient | 5 | 0,198 ± 0,119 | 0,062 ± 0,095 | 0,122 ± 0,073 | 1,97 ± 0,61 |
| wrong | integrated_gradients | 1 | 1,141 ± 0,040 | 1,059 ± 0,037 | 0,730 ± 0,053 | 0,26 ± 0,01 |
| wrong | integrated_gradients | 3 | 1,357 ± 0,019 | 1,299 ± 0,062 | 0,796 ± 0,032 | 0,44 ± 0,02 |
| wrong | integrated_gradients | 5 | 1,506 ± 0,039 | 1,442 ± 0,056 | 0,815 ± 0,019 | 0,57 ± 0,01 |
| wrong | occlusion | 1 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 0,00 ± 0,00 |
| wrong | occlusion | 3 | 0,970 ± 0,011 | 0,994 ± 0,020 | 0,894 ± 0,016 | 0,25 ± 0,02 |
| wrong | occlusion | 5 | 1,012 ± 0,016 | 1,001 ± 0,015 | 0,856 ± 0,027 | 0,30 ± 0,03 |

**Kiểm tra:** IG completeness, |Σ attribution − (f(ảnh) − f(trung tính))| / hiệu ứng: 0,7 ± 0,4%. Shapley efficiency, sai lệch tối đa: 0,0e+00. GraphSVX efficiency, sai lệch tối đa: 1,7e−17; TB 685 lượt chấm mỗi câu. GNN-LRP: dựng lại đầu ra hai lớp GAT sai tối đa 2,1e−06, tổng relevance các walk lệch relevance đầu ra tối đa 2,7e−02. SubgraphX: TB 5769 lượt chấm mỗi câu. GStarX: TB 542 lượt chấm mỗi câu.

### main_r3 (R@1 theo seed: 46,67, 46,56, 46,21)

**Chạy.**

| Seed | Giây (CPU) | Câu | E rỗng | noop khác hạng 1 | Sai số noop tối đa |
| --- | --- | --- | --- | --- | --- |
| 0 | 1421 | 10633 | 45 | 0 | 1.8e-05 |
| 1 | 1169 | 10633 | 39 | 0 | 2.0e-05 |
| 2 | 1270 | 10633 | 44 | 0 | 2.0e-05 |

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

**Phản thực tế nhỏ nhất** (tập con so sánh: mọi tập ≤ 2 phần, CF-GNNExplainer, và k_flip của từng cách giải thích như một phản thực tế tham lam).

| Nhóm | Câu | Có tập ≤ 2 phần (%) | Cỡ 1 / cỡ 2 (%) | Số tập nhỏ nhất (TB) | CF-GNNExplainer tìm được (%) | Cỡ TB / trung vị | % số phần | Bằng cỡ nhỏ nhất (%) | Thừa so với nhỏ nhất | Ảnh sai: gold lên hạng 1 với ≤ 2 phần (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 195 | 70,7 ± 1,3 | 51,5 / 19,2 | 3,1 ± 0,5 | 34,6 ± 2,9 | 2,1 ± 0,4 / 1,3 | 4,9 ± 0,7 | 74,4 ± 5,5 | 0,63 ± 0,35 | 15,3 ± 3,6 |
| correct | 80 | 52,5 ± 2,1 | 32,9 / 19,5 | 2,3 ± 0,6 | 25,0 ± 1,9 | 2,5 ± 0,7 / 2,0 | 6,0 ± 1,4 | 70,0 ± 9,1 | 0,78 ± 0,48 | n/a |
| wrong | 115 | 83,4 ± 2,0 | 64,5 / 18,9 | 3,4 ± 0,8 | 41,3 ± 6,4 | 1,9 ± 0,4 / 1,3 | 4,4 ± 0,7 | 75,6 ± 8,7 | 0,59 ± 0,36 | 15,3 ± 3,6 |

**Phản thực tế tham lam theo thứ tự của từng cách** (bỏ dần theo thứ tự đến khi mất hạng 1, k ≤ 10).

| Nhóm | Cách | Tìm được (%) | Cỡ TB | Bằng cỡ nhỏ nhất (%) | Thừa so với nhỏ nhất |
| --- | --- | --- | --- | --- | --- |
| all | model | 63,9 ± 2,5 | 3,74 ± 0,16 | 46,7 ± 3,0 | 2,13 ± 0,21 |
| all | random | 29,8 ± 3,1 | 5,01 ± 0,26 | 15,8 ± 2,6 | 3,69 ± 0,10 |
| all | lowest | 13,0 ± 3,3 | 6,17 ± 0,74 | 7,6 ± 6,7 | 4,86 ± 0,75 |
| all | clip | 64,2 ± 2,0 | 4,03 ± 0,04 | 20,6 ± 3,1 | 2,59 ± 0,07 |
| all | gradient | 30,0 ± 2,6 | 4,14 ± 0,10 | 38,9 ± 2,8 | 2,73 ± 0,09 |
| all | integrated_gradients | 84,9 ± 0,6 | 2,20 ± 0,06 | 84,0 ± 3,2 | 0,31 ± 0,05 |
| all | occlusion | 80,5 ± 0,5 | 1,89 ± 0,09 | 92,4 ± 2,2 | 0,11 ± 0,04 |
| all | gnnexplainer | 81,8 ± 2,4 | 2,53 ± 0,14 | 64,9 ± 3,2 | 0,72 ± 0,08 |
| all | pgexplainer | 75,3 ± 3,9 | 3,64 ± 0,28 | 36,5 ± 3,6 | 1,96 ± 0,25 |
| all | gnn_lrp | 75,7 ± 1,5 | 3,31 ± 0,23 | 44,8 ± 3,8 | 1,70 ± 0,27 |
| all | graphsvx | 86,1 ± 2,5 | 2,27 ± 0,07 | 82,3 ± 2,2 | 0,36 ± 0,09 |
| all | subgraphx | 49,5 ± 1,7 | 2,23 ± 0,10 | 76,4 ± 3,6 | 0,91 ± 0,06 |
| all | shapley | 85,3 ± 1,7 | 2,42 ± 0,07 | 77,1 ± 2,6 | 0,45 ± 0,03 |
| all | cf_gnnexplainer | 50,8 ± 2,4 | 2,30 ± 0,22 | 74,9 ± 3,9 | 0,91 ± 0,20 |
| all | graphmask | 33,1 ± 1,3 | 4,57 ± 0,10 | 28,9 ± 2,0 | 3,23 ± 0,07 |
| all | gstarx | 76,9 ± 0,8 | 3,23 ± 0,33 | 47,9 ± 7,7 | 1,58 ± 0,44 |
| correct | model | 45,0 ± 2,6 | 4,17 ± 0,24 | 47,8 ± 8,5 | 2,24 ± 0,16 |
| correct | random | 19,5 ± 4,7 | 5,11 ± 0,65 | 12,4 ± 13,4 | 3,80 ± 0,79 |
| correct | lowest | 5,0 ± 1,3 | 6,54 ± 0,94 | 0,0 ± 0,0 | 4,91 ± 1,72 |
| correct | clip | 46,2 ± 0,1 | 4,42 ± 0,21 | 23,0 ± 2,6 | 2,68 ± 0,34 |
| correct | gradient | 20,5 ± 3,5 | 4,02 ± 0,38 | 44,7 ± 3,0 | 2,73 ± 0,46 |
| correct | integrated_gradients | 72,9 ± 1,6 | 2,67 ± 0,08 | 82,8 ± 8,2 | 0,27 ± 0,15 |
| correct | occlusion | 63,7 ± 1,1 | 2,24 ± 0,23 | 95,2 ± 2,4 | 0,10 ± 0,07 |
| correct | gnnexplainer | 67,5 ± 2,1 | 2,77 ± 0,12 | 65,3 ± 6,5 | 0,63 ± 0,15 |
| correct | pgexplainer | 61,6 ± 2,8 | 4,05 ± 0,13 | 29,3 ± 6,2 | 2,07 ± 0,33 |
| correct | gnn_lrp | 62,0 ± 4,8 | 3,84 ± 0,47 | 38,8 ± 7,2 | 1,90 ± 0,57 |
| correct | graphsvx | 73,7 ± 3,9 | 2,70 ± 0,15 | 85,5 ± 3,0 | 0,26 ± 0,14 |
| correct | subgraphx | 33,3 ± 2,9 | 2,27 ± 0,26 | 75,7 ± 9,5 | 0,77 ± 0,10 |
| correct | shapley | 74,1 ± 2,4 | 3,00 ± 0,16 | 81,0 ± 1,3 | 0,43 ± 0,19 |
| correct | cf_gnnexplainer | 34,2 ± 0,8 | 2,56 ± 0,32 | 71,9 ± 9,3 | 0,92 ± 0,23 |
| correct | graphmask | 22,5 ± 0,7 | 5,38 ± 0,51 | 20,1 ± 7,3 | 3,99 ± 0,49 |
| correct | gstarx | 63,3 ± 1,2 | 3,69 ± 0,35 | 44,6 ± 9,8 | 1,72 ± 0,60 |
| wrong | model | 77,1 ± 3,0 | 3,57 ± 0,20 | 46,2 ± 5,8 | 2,09 ± 0,29 |
| wrong | random | 36,9 ± 3,3 | 5,02 ± 0,55 | 16,3 ± 6,2 | 3,70 ± 0,28 |
| wrong | lowest | 18,6 ± 4,6 | 6,09 ± 0,89 | 8,9 ± 7,8 | 4,81 ± 0,81 |
| wrong | clip | 76,7 ± 3,2 | 3,86 ± 0,08 | 19,7 ± 3,4 | 2,55 ± 0,05 |
| wrong | gradient | 36,6 ± 2,6 | 4,19 ± 0,07 | 36,6 ± 3,4 | 2,72 ± 0,26 |
| wrong | integrated_gradients | 93,3 ± 1,0 | 1,94 ± 0,07 | 84,6 ± 2,0 | 0,32 ± 0,10 |
| wrong | occlusion | 92,2 ± 0,9 | 1,73 ± 0,10 | 91,3 ± 2,4 | 0,12 ± 0,04 |
| wrong | gnnexplainer | 91,8 ± 3,1 | 2,40 ± 0,20 | 64,8 ± 6,4 | 0,75 ± 0,12 |
| wrong | pgexplainer | 84,8 ± 7,2 | 3,44 ± 0,38 | 39,6 ± 2,7 | 1,93 ± 0,27 |
| wrong | gnn_lrp | 85,2 ± 1,1 | 3,04 ± 0,11 | 47,6 ± 2,4 | 1,61 ± 0,14 |
| wrong | graphsvx | 94,8 ± 2,3 | 2,04 ± 0,14 | 80,9 ± 4,4 | 0,40 ± 0,18 |
| wrong | subgraphx | 60,8 ± 1,7 | 2,23 ± 0,12 | 76,5 ± 1,3 | 0,96 ± 0,06 |
| wrong | shapley | 93,0 ± 2,7 | 2,09 ± 0,04 | 75,3 ± 3,5 | 0,46 ± 0,06 |
| wrong | cf_gnnexplainer | 62,5 ± 4,3 | 2,20 ± 0,19 | 75,9 ± 6,0 | 0,91 ± 0,19 |
| wrong | graphmask | 40,4 ± 2,5 | 4,25 ± 0,22 | 32,2 ± 3,9 | 2,95 ± 0,13 |
| wrong | gstarx | 86,3 ± 1,5 | 2,99 ± 0,32 | 49,4 ± 8,4 | 1,51 ± 0,45 |

**Mức message: GraphMask so với attention GAT** (ngừng truyền top-k message; phần trăm của điểm mất khi ngừng mọi message, chỉ ở câu mà message giúp ảnh).

| Nhóm | Câu | Message giúp ảnh (%) | Δ điểm (sau − trước) khi ngừng mọi message | Message / ảnh | Thứ tự | AUC mất điểm (cao tốt) | Mất k = 1 / 3 / 5 (%) | Điểm mất (trước − sau) k = 1 / 3 / 5 | p Wilcoxon so với GraphMask (lớn nhất qua seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 969 | 48,2 | −0,0010 ± 0,0013 | 19 | graphmask | 0,403 ± 0,166 | 6,7 / 22,6 / 37,0 | −0,0002 / −0,0002 / 0,0003 |  |
| all | 969 | 48,2 | −0,0010 ± 0,0013 | 19 | attention | 0,411 ± 0,189 | 7,3 / 21,6 / 33,2 | −0,0008 / −0,0011 / −0,0009 | 1,4e−05 |
| all | 969 | 48,2 | −0,0010 ± 0,0013 | 19 | random | 0,323 ± 0,033 | 6,9 / 20,5 / 31,6 | −0,0000 / 0,0000 / 0,0003 | 1,2e−05 |
| correct | 451 | 48,4 | −0,0008 ± 0,0006 | 19 | graphmask | 0,284 ± 0,225 | 5,3 / 10,3 / 19,0 | −0,0000 / −0,0002 / 0,0006 |  |
| correct | 451 | 48,4 | −0,0008 ± 0,0006 | 19 | attention | 0,359 ± 0,120 | 4,4 / 16,2 / 25,9 | −0,0007 / −0,0012 / −0,0006 | 2,9e−01 |
| correct | 451 | 48,4 | −0,0008 ± 0,0006 | 19 | random | 0,301 ± 0,061 | 4,5 / 13,9 / 30,4 | 0,0001 / −0,0002 / 0,0003 | 4,7e−04 |
| wrong | 518 | 47,9 | −0,0012 ± 0,0024 | 19 | graphmask | 0,505 ± 0,133 | 8,0 / 33,0 / 52,1 | −0,0003 / −0,0001 / 0,0001 |  |
| wrong | 518 | 47,9 | −0,0012 ± 0,0024 | 19 | attention | 0,460 ± 0,258 | 9,8 / 26,6 / 40,0 | −0,0009 / −0,0010 / −0,0012 | 1,2e−04 |
| wrong | 518 | 47,9 | −0,0012 ± 0,0024 | 19 | random | 0,344 ± 0,009 | 8,8 / 26,6 / 33,0 | −0,0001 / 0,0002 / 0,0003 | 1,0e−02 |

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

**So sánh các cách giải thích** (mẫu cố định trong val; GNNExplainer, GraphSVX, SubgraphX và Shapley trên tập con).

| Nhóm | Cách | Câu | Giây / câu (CPU) | Deletion AUC (thấp tốt) | Insertion AUC (cao tốt) | k_flip đạt (%) | Bỏ top-1 / 3 / 5: mất hạng 1 (%) | Chỉ giữ top-1 / 3 / 5: hạng | Fid_α+ / Fid_α− (k = 5, α = 0,1) | Fid_α+ / Fid_α− (k = 5, α = 0,5) | Khớp Shapley: Spearman | Top-5 trùng Shapley (%) | p Wilcoxon deletion so với model (lớn nhất qua seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | model | 977 | 0 | 0,237 ± 0,066 | 0,764 ± 0,102 | 57,7 ± 2,5 | 19,5 / 33,7 / 41,4 | 2,82 / 2,12 / 2,14 | 0,050 ± 0,004 / 0,016 ± 0,011 | 0,289 ± 0,035 / 0,211 ± 0,034 | 0,460 ± 0,007 | 31,8 ± 2,6 | n/a |
| all | random | 977 | 0 | 0,838 ± 0,005 | −0,621 ± 0,105 | 25,5 ± 1,4 | 3,7 / 8,6 / 12,1 | 12,88 / 12,83 / 10,18 | 0,013 ± 0,001 / 0,053 ± 0,002 | 0,070 ± 0,004 / 0,409 ± 0,002 | −0,012 ± 0,009 | 14,3 ± 0,7 | 1,4e−105 |
| all | lowest | 977 | 0 | 1,009 ± 0,008 | −2,300 ± 0,279 | 10,9 ± 1,0 | 1,1 / 2,5 / 4,2 | 28,46 / 25,29 / 23,36 | 0,002 ± 0,002 / 0,068 ± 0,003 | 0,005 ± 0,005 / 0,535 ± 0,010 | −0,460 ± 0,007 | 2,1 ± 0,4 | 4,7e−157 |
| all | clip | 977 | 0,000 | 0,283 ± 0,030 | 0,234 ± 0,137 | 61,8 ± 1,0 | 10,9 / 29,6 / 42,8 | 5,17 / 5,34 / 4,83 | 0,047 ± 0,002 / 0,021 ± 0,004 | 0,277 ± 0,019 / 0,222 ± 0,099 | 0,352 ± 0,016 | 35,4 ± 1,5 | 9,6e−01 |
| all | gradient | 977 | 0,003 | 0,826 ± 0,043 | −1,038 ± 0,275 | 27,9 ± 2,9 | 8,8 / 14,1 / 17,8 | 13,30 / 12,17 / 10,94 | 0,012 ± 0,002 / 0,057 ± 0,004 | 0,071 ± 0,018 / 0,415 ± 0,018 | −0,133 ± 0,007 | 17,8 ± 1,4 | 7,6e−84 |
| all | integrated_gradients | 977 | 0,017 | −0,445 ± 0,019 | 0,646 ± 0,167 | 81,9 ± 0,4 | 39,3 / 64,6 / 74,0 | 7,23 / 2,46 / 1,78 | 0,090 ± 0,001 / −0,024 ± 0,005 | 0,555 ± 0,014 / −0,089 ± 0,012 | 0,288 ± 0,010 | 62,9 ± 1,6 | 8,1e−118 |
| all | occlusion | 977 | 0,017 | −0,244 ± 0,027 | 1,234 ± 0,141 | 74,8 ± 0,1 | 44,9 / 63,0 / 68,8 | 7,68 / 1,99 / 1,39 | 0,106 ± 0,004 / −0,046 ± 0,008 | 0,585 ± 0,027 / −0,264 ± 0,053 | 0,476 ± 0,007 | 58,5 ± 1,0 | 9,5e−77 |
| all | gnnexplainer | 195 | 0,289 | −0,169 ± 0,072 | −0,336 ± 0,951 | 81,8 ± 2,4 | 34,6 / 63,7 / 73,5 | 10,01 / 3,59 / 2,42 | 0,091 ± 0,017 / −0,000 ± 0,010 | 0,491 ± 0,060 / 0,105 ± 0,127 | 0,255 ± 0,010 | 56,7 ± 1,6 | 1,3e−12 |
| all | pgexplainer | 977 | 0,002 | 0,081 ± 0,071 | 0,196 ± 0,056 | 70,8 ± 2,1 | 16,8 / 37,6 / 50,9 | 10,79 / 6,32 / 4,54 | 0,050 ± 0,005 / 0,018 ± 0,002 | 0,318 ± 0,024 / 0,163 ± 0,029 | 0,069 ± 0,013 | 43,4 ± 4,3 | 8,0e−08 |
| all | gnn_lrp | 977 | 0,008 | 0,038 ± 0,048 | 0,391 ± 0,086 | 69,7 ± 2,2 | 23,5 / 43,4 / 52,8 | 12,14 / 5,57 / 3,71 | 0,068 ± 0,002 / 0,001 ± 0,006 | 0,394 ± 0,011 / 0,029 ± 0,019 | 0,070 ± 0,018 | 39,5 ± 0,7 | 4,9e−25 |
| all | graphsvx | 195 | 0,156 | −0,708 ± 0,208 | 1,502 ± 0,558 | 86,1 ± 2,5 | 45,2 / 69,4 / 79,5 | 6,15 / 1,91 / 1,40 | 0,114 ± 0,011 / −0,047 ± 0,027 | 0,696 ± 0,100 / −0,193 ± 0,090 | 0,719 ± 0,014 | 71,9 ± 1,4 | 2,4e−22 |
| all | subgraphx | 195 | 0,131 | 0,276 ± 0,069 | 0,617 ± 0,192 | 49,5 ± 1,7 | 32,4 / 38,7 / 42,5 | 8,18 / 4,26 / 3,80 | 0,083 ± 0,006 / −0,009 ± 0,018 | 0,396 ± 0,076 / −0,010 ± 0,076 | 0,110 ± 0,003 | 33,9 ± 0,7 | 5,1e−01 |
| all | shapley | 195 | 0,155 | −0,683 ± 0,226 | 1,532 ± 0,541 | 85,3 ± 1,7 | 43,1 / 67,1 / 77,4 | 6,14 / 1,72 / 1,14 | 0,118 ± 0,016 / −0,046 ± 0,022 | 0,749 ± 0,192 / −0,199 ± 0,096 | n/a | n/a | 8,1e−21 |
| all | cf_gnnexplainer | 195 | 0,580 | 0,406 ± 0,040 | −0,726 ± 0,592 | 50,8 ± 2,4 | 29,1 / 40,4 / 44,1 | 12,92 / 9,71 / 7,99 | 0,050 ± 0,014 / 0,033 ± 0,018 | 0,274 ± 0,027 / 0,240 ± 0,045 | 0,009 ± 0,020 | 28,6 ± 1,0 | 4,8e−02 |
| all | graphmask | 977 | 0,002 | 0,767 ± 0,014 | −0,860 ± 0,158 | 30,9 ± 0,3 | 9,0 / 15,1 / 19,5 | 18,48 / 13,58 / 10,80 | 0,021 ± 0,005 / 0,041 ± 0,003 | 0,108 ± 0,010 / 0,385 ± 0,042 | 0,007 ± 0,019 | 16,8 ± 1,9 | 7,6e−68 |
| all | gstarx | 195 | 0,024 | −0,131 ± 0,084 | 1,257 ± 0,603 | 76,9 ± 0,8 | 26,9 / 50,5 / 63,0 | 5,21 / 2,47 / 2,15 | 0,089 ± 0,008 / −0,009 ± 0,032 | 0,493 ± 0,066 / −0,018 ± 0,083 | 0,656 ± 0,004 | 54,7 ± 0,9 | 7,7e−04 |
| correct | model | 456 | 0 | 0,348 ± 0,037 | 0,654 ± 0,060 | 42,0 ± 1,6 | 11,5 / 23,6 / 29,9 | 2,22 / 1,86 / 1,84 | 0,043 ± 0,003 / 0,019 ± 0,008 | 0,250 ± 0,023 / 0,183 ± 0,035 | 0,448 ± 0,014 | 31,3 ± 2,5 | n/a |
| correct | random | 456 | 0 | 0,850 ± 0,027 | −0,373 ± 0,071 | 17,2 ± 0,3 | 1,8 / 5,5 / 7,8 | 9,03 / 8,68 / 6,74 | 0,014 ± 0,004 / 0,048 ± 0,002 | 0,069 ± 0,017 / 0,358 ± 0,028 | 0,002 ± 0,027 | 14,8 ± 1,5 | 2,0e−46 |
| correct | lowest | 456 | 0 | 1,002 ± 0,012 | −1,675 ± 0,022 | 6,5 ± 0,7 | 0,3 / 1,5 / 2,9 | 22,70 / 19,33 / 17,28 | 0,006 ± 0,002 / 0,062 ± 0,001 | 0,007 ± 0,007 / 0,460 ± 0,008 | −0,448 ± 0,014 | 2,3 ± 0,2 | 4,7e−73 |
| correct | clip | 456 | 0,000 | 0,411 ± 0,032 | 0,321 ± 0,060 | 45,7 ± 0,9 | 6,1 / 18,7 / 27,7 | 3,32 / 3,55 / 3,07 | 0,043 ± 0,002 / 0,020 ± 0,001 | 0,233 ± 0,006 / 0,159 ± 0,011 | 0,326 ± 0,032 | 33,1 ± 1,4 | 4,4e−01 |
| correct | gradient | 456 | 0,003 | 0,801 ± 0,037 | −0,613 ± 0,070 | 18,9 ± 3,4 | 5,3 / 8,8 / 11,7 | 8,92 / 7,81 / 7,01 | 0,013 ± 0,005 / 0,053 ± 0,008 | 0,079 ± 0,021 / 0,350 ± 0,020 | −0,134 ± 0,013 | 18,1 ± 1,4 | 2,7e−40 |
| correct | integrated_gradients | 456 | 0,017 | −0,218 ± 0,025 | 0,665 ± 0,079 | 69,5 ± 0,6 | 24,8 / 49,1 / 60,1 | 4,72 / 1,75 / 1,31 | 0,077 ± 0,001 / −0,017 ± 0,001 | 0,477 ± 0,004 / −0,065 ± 0,022 | 0,278 ± 0,009 | 63,5 ± 2,0 | 1,7e−58 |
| correct | occlusion | 456 | 0,016 | −0,060 ± 0,008 | 1,101 ± 0,040 | 58,4 ± 0,7 | 27,7 / 46,5 / 51,6 | 4,93 / 1,45 / 1,18 | 0,092 ± 0,003 / −0,034 ± 0,001 | 0,495 ± 0,007 / −0,191 ± 0,012 | 0,455 ± 0,016 | 58,4 ± 1,3 | 1,2e−39 |
| correct | gnnexplainer | 80 | 0,291 | −0,102 ± 0,060 | 0,458 ± 0,257 | 67,5 ± 2,1 | 20,4 / 50,0 / 59,4 | 5,09 / 1,71 / 1,39 | 0,080 ± 0,024 / −0,006 ± 0,011 | 0,446 ± 0,070 / 0,001 ± 0,069 | 0,272 ± 0,022 | 57,5 ± 1,8 | 3,9e−06 |
| correct | pgexplainer | 456 | 0,002 | 0,199 ± 0,056 | 0,256 ± 0,058 | 57,6 ± 1,7 | 9,0 / 25,5 / 37,4 | 7,45 / 4,05 / 2,88 | 0,045 ± 0,002 / 0,014 ± 0,001 | 0,292 ± 0,016 / 0,149 ± 0,016 | 0,080 ± 0,008 | 43,4 ± 3,6 | 9,3e−05 |
| correct | gnn_lrp | 456 | 0,007 | 0,133 ± 0,011 | 0,488 ± 0,079 | 55,6 ± 2,7 | 14,0 / 32,9 / 40,6 | 7,98 / 3,60 / 2,54 | 0,062 ± 0,005 / −0,000 ± 0,002 | 0,356 ± 0,010 / 0,030 ± 0,020 | 0,068 ± 0,029 | 41,2 ± 1,1 | 2,8e−17 |
| correct | graphsvx | 80 | 0,157 | −0,418 ± 0,049 | 1,189 ± 0,066 | 73,7 ± 3,9 | 30,8 / 54,1 / 66,2 | 3,39 / 1,20 / 1,07 | 0,089 ± 0,009 / −0,027 ± 0,006 | 0,563 ± 0,023 / −0,120 ± 0,009 | 0,734 ± 0,017 | 73,1 ± 1,3 | 1,2e−10 |
| correct | subgraphx | 80 | 0,135 | 0,297 ± 0,037 | 0,617 ± 0,142 | 33,3 ± 2,9 | 17,4 / 26,2 / 28,2 | 5,19 / 3,19 / 3,06 | 0,078 ± 0,018 / 0,004 ± 0,007 | 0,375 ± 0,017 / 0,011 ± 0,038 | 0,106 ± 0,033 | 34,7 ± 1,7 | 6,2e−01 |
| correct | shapley | 80 | 0,155 | −0,409 ± 0,034 | 1,201 ± 0,051 | 74,1 ± 2,4 | 29,6 / 52,4 / 64,5 | 3,46 / 1,20 / 1,03 | 0,087 ± 0,007 / −0,027 ± 0,006 | 0,568 ± 0,014 / −0,122 ± 0,010 | n/a | n/a | 2,2e−10 |
| correct | cf_gnnexplainer | 80 | 0,585 | 0,497 ± 0,060 | −0,193 ± 0,114 | 34,2 ± 0,8 | 16,3 / 25,0 / 28,7 | 7,75 / 5,95 / 4,64 | 0,041 ± 0,023 / 0,040 ± 0,012 | 0,202 ± 0,045 / 0,274 ± 0,020 | 0,008 ± 0,026 | 27,7 ± 1,0 | 2,4e−01 |
| correct | graphmask | 456 | 0,002 | 0,770 ± 0,016 | −0,499 ± 0,038 | 22,1 ± 0,4 | 5,5 / 9,9 / 12,7 | 12,95 / 9,30 / 7,06 | 0,018 ± 0,002 / 0,041 ± 0,001 | 0,106 ± 0,004 / 0,315 ± 0,006 | 0,003 ± 0,011 | 17,7 ± 1,8 | 3,2e−27 |
| correct | gstarx | 80 | 0,024 | 0,003 ± 0,073 | 0,919 ± 0,048 | 63,3 ± 1,2 | 17,1 / 35,8 / 50,2 | 3,11 / 1,62 / 1,42 | 0,074 ± 0,008 / −0,005 ± 0,012 | 0,394 ± 0,016 / 0,029 ± 0,025 | 0,672 ± 0,008 | 56,1 ± 0,9 | 3,0e−02 |
| wrong | model | 520 | 0 | 0,139 ± 0,094 | 0,861 ± 0,138 | 71,6 ± 3,3 | 26,6 / 42,5 / 51,5 | 3,34 / 2,35 / 2,40 | 0,055 ± 0,006 / 0,013 ± 0,013 | 0,324 ± 0,045 / 0,236 ± 0,033 | 0,468 ± 0,003 | 32,1 ± 2,9 | n/a |
| wrong | random | 520 | 0 | 0,829 ± 0,022 | −0,838 ± 0,227 | 32,7 ± 2,4 | 5,4 / 11,2 / 15,8 | 16,26 / 16,48 / 13,19 | 0,013 ± 0,002 / 0,057 ± 0,005 | 0,071 ± 0,010 / 0,454 ± 0,026 | −0,022 ± 0,015 | 13,9 ± 0,6 | 9,4e−61 |
| wrong | lowest | 520 | 0 | 1,015 ± 0,005 | −2,847 ± 0,535 | 14,7 ± 1,6 | 1,9 / 3,3 / 5,3 | 33,51 / 30,52 / 28,68 | −0,000 ± 0,002 / 0,072 ± 0,005 | 0,003 ± 0,003 / 0,600 ± 0,015 | −0,468 ± 0,003 | 1,9 ± 0,7 | 1,0e−85 |
| wrong | clip | 520 | 0,000 | 0,171 ± 0,029 | 0,159 ± 0,274 | 75,8 ± 1,0 | 15,1 / 39,2 / 55,9 | 6,79 / 6,91 / 6,36 | 0,051 ± 0,003 / 0,022 ± 0,007 | 0,316 ± 0,033 / 0,277 ± 0,186 | 0,371 ± 0,019 | 37,1 ± 1,6 | 9,1e−01 |
| wrong | gradient | 520 | 0,003 | 0,849 ± 0,058 | −1,410 ± 0,524 | 35,8 ± 2,6 | 11,8 / 18,8 / 23,1 | 17,15 / 16,01 / 14,39 | 0,011 ± 0,002 / 0,060 ± 0,001 | 0,063 ± 0,026 / 0,473 ± 0,043 | −0,132 ± 0,005 | 17,6 ± 1,7 | 2,8e−45 |
| wrong | integrated_gradients | 520 | 0,017 | −0,643 ± 0,052 | 0,630 ± 0,266 | 92,8 ± 1,0 | 52,1 / 78,3 / 86,1 | 9,42 / 3,08 / 2,19 | 0,102 ± 0,003 / −0,030 ± 0,009 | 0,624 ± 0,025 / −0,110 ± 0,008 | 0,295 ± 0,010 | 62,5 ± 2,0 | 6,5e−60 |
| wrong | occlusion | 520 | 0,017 | −0,405 ± 0,056 | 1,350 ± 0,232 | 89,2 ± 0,6 | 60,1 / 77,6 / 83,8 | 10,09 / 2,47 / 1,56 | 0,119 ± 0,005 / −0,057 ± 0,014 | 0,664 ± 0,045 / −0,328 ± 0,089 | 0,490 ± 0,003 | 58,6 ± 1,4 | 1,2e−39 |
| wrong | gnnexplainer | 115 | 0,287 | −0,216 ± 0,099 | −0,906 ± 1,717 | 91,8 ± 3,1 | 44,4 / 73,3 / 83,2 | 13,43 / 4,90 / 3,12 | 0,098 ± 0,012 / 0,004 ± 0,013 | 0,522 ± 0,071 / 0,180 ± 0,228 | 0,244 ± 0,015 | 56,1 ± 1,6 | 4,9e−07 |
| wrong | pgexplainer | 520 | 0,002 | −0,022 ± 0,087 | 0,144 ± 0,087 | 82,4 ± 3,2 | 23,7 / 48,2 / 62,7 | 13,71 / 8,32 / 5,99 | 0,055 ± 0,010 / 0,021 ± 0,005 | 0,341 ± 0,032 / 0,176 ± 0,043 | 0,061 ± 0,017 | 43,4 ± 4,7 | 1,9e−04 |
| wrong | gnn_lrp | 520 | 0,008 | −0,045 ± 0,085 | 0,306 ± 0,102 | 82,1 ± 1,7 | 31,8 / 52,7 / 63,6 | 15,78 / 7,31 / 4,74 | 0,073 ± 0,005 / 0,002 ± 0,009 | 0,428 ± 0,017 / 0,028 ± 0,023 | 0,071 ± 0,011 | 38,2 ± 0,4 | 1,8e−10 |
| wrong | graphsvx | 115 | 0,156 | −0,915 ± 0,398 | 1,731 ± 0,969 | 94,8 ± 2,3 | 55,2 / 80,0 / 88,7 | 8,07 / 2,40 / 1,62 | 0,131 ± 0,015 / −0,062 ± 0,045 | 0,791 ± 0,161 / −0,245 ± 0,155 | 0,709 ± 0,013 | 71,1 ± 1,5 | 5,4e−13 |
| wrong | subgraphx | 115 | 0,128 | 0,259 ± 0,140 | 0,622 ± 0,387 | 60,8 ± 1,7 | 42,7 / 47,4 / 52,3 | 10,26 / 5,00 / 4,31 | 0,087 ± 0,003 / −0,019 ± 0,036 | 0,412 ± 0,128 / −0,027 ± 0,150 | 0,112 ± 0,018 | 33,3 ± 0,4 | 7,3e−01 |
| wrong | shapley | 115 | 0,154 | −0,878 ± 0,415 | 1,773 ± 0,936 | 93,0 ± 2,7 | 52,6 / 77,3 / 86,3 | 8,00 / 2,08 / 1,22 | 0,140 ± 0,026 / −0,060 ± 0,038 | 0,879 ± 0,326 / −0,255 ± 0,160 | n/a | n/a | 5,0e−12 |
| wrong | cf_gnnexplainer | 115 | 0,576 | 0,341 ± 0,056 | −1,109 ± 1,090 | 62,5 ± 4,3 | 38,1 / 51,1 / 54,7 | 16,51 / 12,33 / 10,30 | 0,057 ± 0,008 / 0,029 ± 0,028 | 0,324 ± 0,028 / 0,217 ± 0,064 | 0,010 ± 0,040 | 29,2 ± 1,5 | 1,1e−01 |
| wrong | graphmask | 520 | 0,002 | 0,764 ± 0,014 | −1,176 ± 0,261 | 38,6 ± 1,0 | 12,1 / 19,6 / 25,5 | 23,34 / 17,34 / 14,08 | 0,023 ± 0,008 / 0,042 ± 0,005 | 0,109 ± 0,019 / 0,446 ± 0,080 | 0,009 ± 0,028 | 16,2 ± 2,1 | 4,0e−42 |
| wrong | gstarx | 115 | 0,024 | −0,226 ± 0,183 | 1,503 ± 1,055 | 86,3 ± 1,5 | 33,7 / 60,7 / 71,8 | 6,66 / 3,05 / 2,65 | 0,100 ± 0,012 / −0,013 ± 0,062 | 0,564 ± 0,127 / −0,053 ± 0,160 | 0,645 ± 0,007 | 53,7 ± 1,0 | 1,5e−02 |

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

**Kiểm tra:** IG completeness, |Σ attribution − (f(ảnh) − f(trung tính))| / hiệu ứng: 1,1 ± 0,1%. Shapley efficiency, sai lệch tối đa: 1,3e−07. GraphSVX efficiency, sai lệch tối đa: 4,4e−16; TB 734 lượt chấm mỗi câu. GNN-LRP: dựng lại đầu ra hai lớp GAT sai tối đa 1,4e−06, tổng relevance các walk lệch relevance đầu ra tối đa 9,2e−02. SubgraphX: TB 838 lượt chấm mỗi câu. GStarX: TB 318 lượt chấm mỗi câu.

## Phụ lục: 13 câu bằng chứng

### countonly_factual_r3 (seed 0)

| Câu (caption_id) | Ảnh hạng 1 | E top-3 (term) | Bỏ top-3: Δ điểm, hạng (ngẫu nhiên) | Chỉ giữ top-3: Δ điểm, hạng (ngẫu nhiên) | k_flip | k giữ 80% | Phần cho 80% term | Tính cộng k = 3: dự đoán / đo | Vật thể khớp strict | Bộ ba khớp đủ | Cắt truyền tin: Δ top (ngẫu nhiên) | Phản thực tế nhỏ nhất (≤ 2 phần) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | 2377477 (đúng) | table (0,028); room (0,025); stove (0,016) | −0,016, 4 (0,000, 1,0) | 0,016, 1 (−0,020, 6,7) | 1 | 2 | 22 | 0,069 / 0,016 | 1/3 | 0/2 | 0,0044 (0,0040) | table (1 tập cỡ 1) |
| A cow standing along side of a large river. (422698_1) | 2351201 (đúng) | cow standing ear (0,058); plant along water side (0,035); cow standing nose (0,034) | −0,008, 1 (0,001, 1,0) | 0,004, 1 (0,001, 1,0) | 4 | 1 | 11 | 0,127 / 0,008 | 2/2 | 0/1 | −0,0028 (−0,0000) | cow; cow (1 tập cỡ 2) |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | 2329851 (đúng) | pen (0,052); trainer observing dog (0,032); dog (0,031) | −0,020, 2 (−0,004, 1,3) | −0,015, 2 (−0,055, 6,0) | 1 | 3 | 15 | 0,115 / 0,020 | 3/5 | 0/4 | −0,0051 (−0,0014) | pen (2 tập cỡ 1) |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | 2402283 (đúng) | donut (0,030); bite (0,026); person holding donut (0,023) | −0,026, 4 (−0,001, 1,4) | 0,007, 1 (−0,023, 4,7) | 1 | 3 | 25 | 0,079 / 0,026 | 2/3 | 0/2 | 0,0038 (0,0023) | donut (3 tập cỡ 1) |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | 1159610 (đúng) | dog (0,047); umbrella (0,040); umbrella (0,026) | −0,069, 22 (−0,013, 1,8) | 0,000, 1 (−0,049, 14,1) | 1 | 2 | 9 | 0,113 / 0,069 | 4/5 | 0/3 | 0,0011 (0,0016) | dog (1 tập cỡ 1) |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | 2366682 (sai) | banana in plate (0,069); banana in paper plates (0,061); banana in paper plates (0,059) | −0,006, 1 (0,000, 1,0) | −0,008, 1 (−0,008, 1,0) | 9 | 1 | 8 | 0,189 / 0,006 | 2/2 | 0/1 | 0,0053 (0,0024) | plate; paper plates (1 tập cỡ 2) |
| A train riding along the tracks along a bank of dirt. (279933_1) | 2357910 (sai) | windshield wiper on train (0,024); gravel underneath track (0,020); logo on train (0,017) | −0,002, 2 (0,000, 1,3) | −0,007, 3 (−0,007, 3,1) | 1 | 7 | 23 | 0,061 / 0,002 | 2/3 | 0/2 | 0,0006 (0,0004) | gravel (8 tập cỡ 1) |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | 2353248 (sai) | bus (0,035); street (0,021); bus has bumper (0,014) | −0,020, 9 (−0,001, 1,5) | 0,008, 1 (−0,015, 7,4) | 1 | 2 | 12 | 0,070 / 0,020 | 2/2 | 0/1 | 0,0002 (0,0013) | bus (2 tập cỡ 1) |
| Two men standing in a living room playing video game with remotes. (34471_0) | 2395602 (sai) | wii controller (0,060); man wearing wrist band (0,048); man wearing short (0,042) | −0,022, 2 (−0,003, 1,1) | −0,005, 1 (−0,027, 6,0) | 1 | 2 | 13 | 0,149 / 0,022 | 1/4 | 0/3 | −0,0046 (−0,0024) | wii controller (1 tập cỡ 1) |
| A batter is preparing to hit the ball at a game. (526446_0) | 2417935 (sai) | helmet has black/face mask (0,068); woman swinging bat (0,066); black bar on helmet (0,063) | −0,007, 1 (−0,001, 1,0) | −0,043, 44 (−0,044, 44,4) | 4 | 10 | 13 | 0,197 / 0,007 | 2/2 | 0/1 | −0,0004 (0,0001) | bat; helmet (6 tập cỡ 2) |
| The two people are walking in the snow with their ski equipment. (472772_2) | 2384371 (đúng) | person carrying ski (0,083); woman (0,017); person (0,016) | −0,017, 3 (−0,002, 1,0) | 0,025, 1 (−0,023, 14,1) | 3 | 1 | 26 | 0,116 / 0,017 | 1/3 | 0/2 | 0,0041 (0,0026) | woman; person (6 tập cỡ 2) |
| A snow board sitting upright with a view of a mountain. (237041_4) | 2320074 (đúng) | logo on snowboard (0,050); design on snowboard (0,039); snowboard (0,033) | −0,024, 2 (−0,004, 1,2) | −0,026, 3 (−0,024, 4,1) | 3 | 1 | 6 | 0,122 / 0,024 | 1/2 | 0/1 | −0,0046 (−0,0036) | snowboard (1 tập cỡ 1) |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | 2385674 (đúng) | umbrella (0,037); rain (0,023); table (0,014) | −0,025, 8 (−0,005, 2,5) | −0,002, 1 (−0,027, 8,9) | 1 | 3 | 21 | 0,074 / 0,025 | 2/4 | 1/3 | −0,0011 (0,0008) | umbrella (2 tập cỡ 1) |

### main_r3 (seed 0)

| Câu (caption_id) | Ảnh hạng 1 | E top-3 (term) | Bỏ top-3: Δ điểm, hạng (ngẫu nhiên) | Chỉ giữ top-3: Δ điểm, hạng (ngẫu nhiên) | k_flip | k giữ 80% | Phần cho 80% term | Tính cộng k = 3: dự đoán / đo | Vật thể khớp strict | Bộ ba khớp đủ | Cắt truyền tin: Δ top (ngẫu nhiên) | Phản thực tế nhỏ nhất (≤ 2 phần) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dining table for six in a wood paneled room with a fireplace. (302038_1) | 2377477 (đúng) | room with stove (1,227); wood under stove (1,203); stove burning wood (1,195) | −0,067, 1 (−0,004, 1,0) | −0,125, 1 (−0,437, 3,2) | 7 | 1 | 19 | 3,624 / 0,067 | 1/3 | 0/1 | −0,0272 (−0,0135) | stove (1 tập cỡ 1) |
| A cow standing along side of a large river. (422698_1) | 2351201 (đúng) | moss on water (2,541); river surrounded by plant (2,527); shade area along edge of water (2,477) | −0,104, 1 (0,011, 1,0) | −1,286, 11 (−1,574, 12,2) | > 10 | > 10 | 12 | 7,544 / 0,104 | 2/3 | 0/2 | 0,0000 (0,0000) | cow; cow (1 tập cỡ 2) |
| A man sitting in front of the pen with a dog herding sheep in the background. (357943_3) | 2329851 (đúng) | pen of sheep (2,336); pen of sheep (2,336); pen of sheep (2,336) | −0,221, 1 (0,004, 1,0) | −0,830, 2 (−1,411, 11,7) | 5 | > 10 | 11 | 7,009 / 0,221 | 2/4 | 0/2 | 0,0006 (0,0006) | pen (1 tập cỡ 1) |
| Someone holding a doughnut that has a bite taken out of it. (445695_1) | 2402283 (đúng) | donut (1,241); bite (0,954); chocolate cream (0,064) | −1,410, 11 (−0,226, 2,4) | 0,143, 1 (−1,095, 8,0) | 1 | 2 | 2 | 2,259 / 1,410 | 1/2 | 0/0 | 0,0112 (0,0112) | donut (2 tập cỡ 1) |
| People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. (574350_4) | 1159610 (đúng) | frame has frame (0,942); man has pant (0,912); woman wearing pant (0,899) | −0,026, 1 (−0,001, 1,0) | −0,613, 2 (−0,648, 2,0) | 7 | 9 | 9 | 2,754 / 0,026 | 5/6 | 0/4 | 0,0000 (0,0000) | dog (1 tập cỡ 1) |
| A bunch of bananas are sitting on a wooden plate. (1153_3) | 2366682 (sai) | banana in plate (4,007); banana in plate (3,984); banana in plate (3,984) | −0,181, 1 (0,012, 1,0) | −0,611, 2 (−0,794, 2,8) | 9 | 10 | 7 | 11,974 / 0,181 | 2/3 | 0/2 | 0,0210 (0,0210) | plate; plate (2 tập cỡ 2) |
| A train riding along the tracks along a bank of dirt. (279933_1) | 2378096 (sai) | cfv under headlight (2,736); headlight on train (2,657); grass on ground (2,629) | −0,165, 1 (−0,014, 1,0) | −0,716, 4 (−1,253, 10,7) | 8 | > 10 | 14 | 8,022 / 0,165 | 2/3 | 0/1 | 0,0000 (0,0000) | dirt; 413 (10 tập cỡ 2) |
| A bus that is reading "89 Portola Plaza" is traveling on. (159627_2) | 2355061 (đúng) | bus (0,393); bus (0,393); bus (0,393) | −0,399, 3 (−0,005, 1,0) | −0,315, 3 (−0,572, 5,8) | 3 | 4 | 6 | 1,180 / 0,399 | 1/2 | 0/0 | 0,0001 (0,0001) | không có |
| Two men standing in a living room playing video game with remotes. (34471_0) | 2395602 (sai) | man wearing shoe (2,049); man has leg (2,040); writing on jean (2,028) | −0,051, 1 (−0,011, 1,0) | −0,414, 1 (−0,567, 2,2) | > 10 | > 10 | 13 | 6,117 / 0,051 | 1/4 | 0/3 | 0,0011 (0,0011) | wii controller (1 tập cỡ 1) |
| A batter is preparing to hit the ball at a game. (526446_0) | 2397612 (sai) | catcher (4,147); dirt (0,145); guard (0,113) | −2,260, 45 (−0,843, 21,9) | 0,127, 1 (−0,822, 22,0) | 1 | 1 | 1 | 4,405 / 2,260 | 0/3 | 0/0 | 0,0014 (0,0014) | catcher (1 tập cỡ 1) |
| The two people are walking in the snow with their ski equipment. (472772_2) | 2352632 (sai) | people in snow (3,051); people has ski (2,783); tree near building (2,671) | −0,180, 1 (0,006, 1,0) | 0,116, 1 (−0,420, 2,1) | 4 | 1 | 14 | 8,505 / 0,180 | 2/3 | 1/2 | −0,0272 (−0,0272) | snow (2 tập cỡ 1) |
| A snow board sitting upright with a view of a mountain. (237041_4) | 2412723 (sai) | mountain range in distance (2,811); snowboard has writing (2,802); mountain on horizon (2,795) | −0,117, 1 (−0,002, 1,0) | 0,100, 1 (−0,689, 5,5) | > 10 | 1 | 22 | 8,407 / 0,117 | 1/3 | 0/2 | 0,0165 (0,0131) | snowboard; mountain (1 tập cỡ 2) |
| A lone woman huddles under her umbrella at a picnic table in the rain. (151938_1) | 2360307 (sai) | lady sitting on outside bench (0,676); lady in coat (0,655); trim on umbrella (0,639) | −0,044, 1 (−0,000, 1,0) | −0,110, 1 (−0,246, 2,5) | 7 | > 10 | 13 | 1,970 / 0,044 | 2/4 | 0/3 | −0,0027 (−0,0027) | lady (1 tập cỡ 1) |

