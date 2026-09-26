# Nháp báo cáo: Cấp độ 1, Visual Genome ∩ COCO

Nháp để Lộc duyệt rồi chuyển sang bản báo cáo chính. Mọi con số lấy từ file kết quả trong repo; đường dẫn ghi ở cuối từng
mục. Thuật ngữ tiếng Anh giữ nguyên, kèm giải thích ở lần đầu. Người soạn nháp: Claude (theo yêu cầu của Lộc, 2026-09-26).

## 1. Dữ liệu và cách chia tập

Tập dữ liệu công khai thứ nhất là phần giao giữa **Visual Genome** (VG, bộ ảnh có scene graph gán tay: vật thể, thuộc tính,
quan hệ) và **MS-COCO** (ảnh có 5 caption mỗi ảnh). Hai bộ được nối theo `coco_id`. Truy vấn là caption COCO; đáp án đúng
là ảnh chứa caption đó. Chia theo **split Karpathy** (cách chia chuẩn của bài toán truy vấn ảnh–văn bản trên COCO), gộp
`restval` vào train. Lấy toàn bộ phần giao, không lấy mẫu; bỏ 290 ảnh VG trùng `coco_id` với ảnh khác.

| Split | Ảnh | Caption (truy vấn) |
|---|---:|---:|
| train | 46 944 | 234 854 |
| val | 2 126 | 10 633 |
| test | 2 138 | 10 696 |

Khi đánh giá một split, pool ứng viên là toàn bộ ảnh của chính split đó. Pool test (2 138 ảnh) nhỏ hơn MSCOCO 5K, nên số
của nhóm không so trực tiếp được với số trong các bài báo. Để kiểm tra pipeline, nhóm đã chạy lại CLIP trên đúng MSCOCO 5K
và so với số đã công bố (`experiments/checks/README.md`, mục 1).

**Đồ thị ảnh.** Mỗi ảnh có một scene graph riêng lấy từ `objects.json` + `relationships.json` của VG. Nhãn là nhãn mở,
đã chuẩn hoá alias, không lọc về VG150. Mỗi instance là một node; quan hệ có hướng, trùng cả ba thành phần thì gộp. Không
có cạnh giữa các ảnh, không dùng ConceptNet. Trung bình 26,7 node và 18,0 cạnh mỗi ảnh (train + val); 3,0% ảnh không có
cạnh.

**Đồ thị câu.** Caption được tách bằng `sng_parser` (SceneGraphParser, dựa trên spaCy) thành vật thể (`lemma_head`) và
quan hệ; bỏ đại từ. Trung bình 3,06 vật thể và 1,65 quan hệ mỗi câu; 14,2% câu val không tách được quan hệ nào.

Nguồn: `experiments/data_stats.md`, `docs/interfaces.md`, `docs/level1-model.md` mục 1.

## 2. Chống rò rỉ dữ liệu

- File split là nguồn duy nhất; mọi mô hình so sánh dùng cùng một bộ file split và cùng danh sách ứng viên top-50 của CLIP.
- Ứng viên lúc huấn luyện: train được chia cố định (seed 0) thành 22 nhóm khoảng 2 134 ảnh. Top-50 của mỗi caption train
  chỉ lấy trong nhóm của ảnh đúng, nên độ khó giống lúc đánh giá mà không chạm val hay test.
- CLIP đóng băng. Bảng vector nhãn / quan hệ / bộ ba được mã hoá bằng CLIP text đóng băng cho chữ của cả ba split. Không
  có tham số nào được khớp trên các vector này.
- Val chỉ dùng để dừng sớm, chọn checkpoint và so các ablation. Không có tập dev riêng.
- Test được chấm **một lần**, sau khi Lộc chốt phương pháp chính:
  - `scripts/evaluate_test.py` ở commit `f738de3` cho mô hình chính, và commit `8b36243` cho các ablation;
  - script kiểm tra ứng viên test tái tạo đúng số CLIP thuần trước khi chấm;
  - mỗi file kết quả chỉ ghi một lần;
  - `GraphStore` từ chối nạp test, trừ khi được gọi với `allow_test=True` và chỉ riêng split test.
- Bộ 20 truy vấn đánh giá của cấp độ 2 không dùng ở đây.

## 3. Mô hình

Chi tiết đầy đủ, kèm số tham số và dòng code: `docs/level1-model.md`. Tóm tắt:

1. **Kênh CLIP.** Cosine giữa vector CLIP ViT-B/32 (OpenAI, bản QuickGELU, đóng băng) của câu và của ảnh; lấy top-50.
2. **Mã hoá đồ thị**, dùng chung cho đồ thị câu và đồ thị ảnh. Node là vector CLIP text của nhãn (prompt
   "a photo of a {nhãn}"), chiếu 512 → 256; cạnh mang vector nhãn quan hệ. Hai lớp **GATv2** (Graph Attention Network
   bản 2: mỗi node gom thông tin từ hàng xóm theo trọng số attention học được), 4 head. Bộ ba (chủ thể, quan hệ, đối tượng)
   qua một MLP. Vector cuối của mỗi phần là `L2(f + U·g)`, với `U` khởi tạo bằng 0.
3. **Hai kênh đồ thị** (vật thể và bộ ba). Mỗi phần i của câu hỏi các phần cùng loại của ảnh: `sim_ij` là cosine,
   `a_ij = softmax_j(sim_ij / τ)`, `s_i = Σ_j a_ij·sim_ij`. Trọng số phần trong câu là `w_i = softmax_i(u·p_i)`, và điểm
   kênh là `S = Σ_i w_i·s_i`.
4. **Trộn.** `[a, b, c] = softmax(W·q_câu + bias)` theo từng câu;
   `Graph = z((b·z(S_obj) + c·z(S_tri)) / (b + c))`; điểm cuối = `a·z(CLIP) + (1 − a)·Graph`, với z-score tính trong
   top-50 của từng câu. Đây đúng công thức `α·z(CLIP) + (1−α)·z(Graph)` của đề, với α = a thay đổi theo câu.
5. **Khởi tạo** `U = 0`, `u = 0`, `W = 0`, `bias = log(0,60; 0,28; 0,12)`, `τ = 0,05`. Khi đó mô hình trùng đúng phép thử
   không huấn luyện đã đo trước (R@1 val 43,50). Có kiểm tra tự động cho việc này (`scripts/checks/step0_reference_check.py`).

Tổng cộng 1 054 214 tham số huấn luyện được; CLIP không nằm trong số này.

## 4. Huấn luyện

- Loss softmax cross-entropy trên 50 ứng viên của điểm đã trộn, chia cho nhiệt độ học được `T`.
- Tối ưu: AdamW, lr 1e-3, weight decay 1e-4, batch hiệu dụng 256 (micro-batch 128, cộng dồn gradient), dropout 0,1.
- Epoch 1 khoá `W` và `bias`, để phần đồ thị học trước khi mở cổng trộn.
- Dừng sớm khi R@1 val không tăng sau 2 epoch, tối đa 10 epoch; giữ checkpoint tốt nhất theo val. 3 seed (0, 1, 2).
- Trên RTX 4090, mỗi epoch khoảng 285 giây, mỗi run 25–50 phút. Toàn bộ 24 run chạy qua đêm 23–24/09/2026, không run nào
  lỗi.

Để đạt tốc độ này, đồ thị của mọi ảnh và câu được đổi sẵn thành chỉ số nguyên một lần, và vector được lấy bằng torch
indexing trên GPU. Trước đó mỗi epoch mất khoảng 45 phút (`experiments/level1/README.md`).

## 5. Kết quả

R@k là tỉ lệ câu có ảnh đúng nằm trong k kết quả đầu; MRR là trung bình nghịch đảo hạng của ảnh đúng (tính trong
top-50). Số dạng mean ± std qua 3 seed, lấy ở checkpoint tốt nhất theo val.

**Test** (10 696 truy vấn, pool 2 138 ảnh; `experiments/level1/test/`):

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần | 39,32 | 66,39 | 77,32 | 52,00 |
| Ba kênh, không huấn luyện | 42,54 | 70,04 | 80,35 | 55,05 |
| **CLIP + Đồ thị (mô hình chính)** | **45,77 ± 0,23** | **72,87 ± 0,31** | **82,69 ± 0,31** | **58,14 ± 0,18** |

**Val** (10 633 truy vấn): CLIP thuần 39,13; ba kênh không huấn luyện 43,50; mô hình chính 46,48 ± 0,24 R@1
(MRR 58,85 ± 0,18).

So với CLIP thuần trên test, mô hình chính tăng 6,45 điểm R@1 và 6,14 điểm MRR. Phần do học (so với bản không huấn
luyện) là 3,23 điểm R@1. Giới hạn trên của cả cách làm là recall@50 của CLIP: 96,66% trên test, tức 3,3% câu có ảnh đúng
nằm ngoài top-50 và không thể cứu bằng xếp lại.

## 6. Ablation và đối chứng

Mỗi dòng ablation là một mô hình **huấn luyện lại từ đầu**, cùng công thức, cùng 3 seed. Bảng dưới là R@1 (mean ± std);
bảng đủ 4 chỉ số ở `experiments/level1/README.md`.

| Cấu hình | Val | Test |
|---|---|---|
| Mô hình chính | 46,48 ± 0,24 | 45,77 ± 0,23 |
| a, b, c cố định (dùng chung cho mọi câu, học trên train) | 46,13 ± 0,15 | 45,13 ± 0,05 |
| Bỏ GAT (mỗi nhãn được biến đổi riêng, không lan truyền) | 46,71 ± 0,07 | 45,95 ± 0,20 |
| Tắt kênh bộ ba (GAT vẫn dùng cạnh) | 45,99 ± 0,05 | 45,07 ± 0,22 |
| Bỏ hẳn quan hệ (không kênh bộ ba, GAT không cạnh) | 45,98 ± 0,10 | 45,26 ± 0,14 |
| Phía câu không qua bộ mã hoá đồ thị | 46,92 ± 0,07 | 45,61 ± 0,29 |
| Huấn luyện và chấm trên đồ thị nối lại cạnh ngẫu nhiên (giữ bậc, nhãn đi theo cạnh) | 46,24 ± 0,24 | 45,50 ± 0,40 |
| Loss LambdaRank theo MRR thay cross-entropy | 45,18 ± 0,34 | 44,28 ± 0,30 |

**Đối chứng chỉ đánh giá**, trên 3 checkpoint mô hình chính × 3 seed đối chứng, val:
- Mỗi ảnh nhận scene graph của một ảnh khác: R@1 rơi xuống 26,33 ± 0,39, thấp hơn CLIP thuần 12,8 điểm.
- Nối lại cạnh ngẫu nhiên: R@1 46,29 ± 0,19.

**Quét α trên val** cho dòng a, b, c cố định: α học được (0,48–0,50) cách α tốt nhất trên lưới không quá 0,19 điểm R@1.

**Nhận xét:**
1. Đồ thị có ích, và mô hình thật sự dựa vào đồ thị **của đúng ảnh**. Thay đồ thị của ảnh khác làm kết quả sụp dưới cả
   CLIP, vì mô hình đặt khoảng một nửa trọng số (a ≈ 0,48) vào phần đồ thị.
2. Trọng số trộn theo từng câu có ích: +0,64 R@1 trên test so với trọng số cố định. Với câu mà parser tách hỏng, mô hình
   tự tăng trọng số CLIP (ví dụ trong kịch bản demo).
3. Quan hệ có ích qua kênh bộ ba tường minh: +0,5 đến +0,7 R@1 trên test. Tắt kênh bộ ba và bỏ hẳn quan hệ cho kết quả
   như nhau, nên cạnh trong GAT không thêm gì khi đã có kênh bộ ba.
4. **GAT không giúp.** Bỏ GAT không thua mô hình chính trên cả val và test (+0,18 R@1 trên test, cỡ một độ lệch chuẩn).
   Việc cạnh nối với vật thể nào cũng gần như không quan trọng: nối lại ngẫu nhiên chỉ giảm 0,27 R@1 trên test.
5. Dòng "phía câu không qua bộ mã hoá đồ thị" tốt nhất trên val nhưng thua mô hình chính trên test (−0,16 R@1). Các chênh
   lệch dưới khoảng 0,3 điểm giữa mô hình chính, bỏ GAT, dòng này và dòng nối lại cạnh nằm trong nhiễu giữa các seed.
6. Loss LambdaRank kém nhất (−1,49 R@1 trên test). Chưa tinh chỉnh riêng cho loss này.

## 7. Giải thích và phân tích 10 truy vấn

Giải thích sinh từ chính lần xếp hạng (`src/explain/graph_explainer.py`). Với mỗi phần của câu, nó cho biết:
- phần ảnh có độ tương đồng lớn nhất;
- trọng số `w_i` của phần câu, attention `a_ij`, độ tương đồng `sim_ij`;
- đóng góp = trọng số kênh × `w_i` × `a_ij` × `sim_ij`, cùng a, b, c của câu.

Demo hiển thị đúng các đại lượng này.

10 truy vấn val được bốc theo quy tắc cố định (seed 2026, câu có ít nhất một quan hệ) từ ba nhóm. Trên toàn bộ val, với
checkpoint seed 0:
- 1 235 câu được đồ thị sửa đúng (CLIP xếp ảnh đúng dưới hạng 1, mô hình xếp hạng 1);
- 434 câu bị làm hỏng (CLIP hạng 1, mô hình dưới hạng 1);
- 5 237 câu cả hai đều sai; 3 727 câu cả hai đều đúng.

Bảng đầy đủ, ảnh và phân tích từng ca: `experiments/level1/explanations/README.md`.

**Phép thử xoá.** Trên ảnh mô hình xếp hạng 1 của mỗi ca, xoá phần có đóng góp lớn nhất rồi so với xoá ngẫu nhiên
10 phần cùng loại.
- Ở 10/10 ca, xoá phần lớn nhất làm điểm giảm nhiều hơn xoá ngẫu nhiên.
- Ở 5/10 ca, riêng việc xoá phần đó đã làm ảnh rơi khỏi hạng 1. Ví dụ: `park bench` 1 → 14 (ngẫu nhiên: trung bình
  hạng 3,0); `church` 1 → 4 (ngẫu nhiên: 1,0); `racket` 1 → 25 (ngẫu nhiên: 2,5).
- Ở hai ca thất bại, xoá phần quyết định của ảnh sai là đủ để ảnh đúng trở lại hạng 1.

Đây là mức fidelity cơ bản; đo fidelity / validity / sparsity đầy đủ thuộc cấp độ 3.

**Nguyên nhân thất bại** quan sát được khi xem ảnh thật:
- Caption có nhiều ảnh đúng. Ví dụ "a child is standing outside in the grass": ảnh mô hình chọn cũng khớp câu.
- Khác biệt từ vựng giữa caption COCO và nhãn VG: `child` / `boy`; không có nhãn `tennis player` trong VG.
- Chú thích VG thiếu nhãn: ảnh tháp đá không có nhãn `tower`.
- Lỗi parser: "a couple of" bị tách thành vật thể `couple`.
- Ảnh đúng nằm ngoài top-50 của CLIP; "bike" trong caption thực ra là xe máy.

## 8. Hạn chế

- **GAT không đóng góp** so với biến đổi riêng từng nhãn, dù đề yêu cầu GNN. Nhóm giữ mô hình có GAT làm phương pháp chính,
  vì cách xử lý câu đối xứng với cách xử lý scene graph của ảnh (quyết định của Lộc trước khi chạy test), và báo cáo thẳng
  kết quả ablation.
- **Vector bộ ba bị ép sát nhau sau huấn luyện.** Cosine trung bình giữa hai bộ ba bất kỳ tăng từ 0,68 (CLIP đóng băng) lên
  khoảng 0,97; vector vật thể thì tách xa nhau hơn (0,79 → khoảng 0,25) (`experiments/checks/part_similarity.json`). Vì thế
  kênh bộ ba chỉ xếp hạng nhờ khác biệt rất nhỏ, và giải thích ở kênh bộ ba thường ghép các cặp không liên quan. Một nguyên
  nhân có thể là lớp `U` dùng chung cho vật thể và bộ ba. Chưa sửa vì thiết kế đã chốt trước khi chạy test.
- Chênh lệch giữa các cấu hình mạnh nhất chỉ cỡ 1–2 độ lệch chuẩn với 3 seed; chưa làm kiểm định theo cặp trên từng câu.
- Dừng sớm với patience 2 nhạy với nhiễu của val. Hai run đạt tốt nhất đúng ở epoch 10, là trần số epoch.
- Kết quả phụ thuộc chất lượng parser và độ phủ nhãn VG. Không dùng thuộc tính (tính từ, màu): phép thử không huấn luyện
  chỉ tăng +0,36 R@1, dưới ngưỡng 0,5 điểm đặt trước (`experiments/checks/attribute-probe.REPORT.md`).
- Xếp lại không cứu được 3,3% câu có ảnh đúng nằm ngoài top-50 của CLIP.
