# Nháp báo cáo: Cấp độ 1 và phase cải tiến, Visual Genome ∩ COCO

Nháp để Lộc duyệt rồi chuyển sang bản báo cáo chính. Mọi con số lấy từ file kết quả trong repo; đường dẫn ghi ở cuối từng
mục và ở mục 12. Thuật ngữ tiếng Anh giữ nguyên, kèm giải thích ở lần đầu. Người soạn nháp: Claude (theo yêu cầu của
Lộc). Bản đầu ngày 2026-09-26 (cấp độ 1); bản này cập nhật ngày 2026-10-25, thêm phase cải tiến và mô hình chính mới.

## Tóm tắt

Bài toán là truy vấn văn bản → ảnh (nhánh A của đề). CLIP đóng băng lấy top-50 ứng viên; một mô hình đồ thị (GATv2 trên
scene graph của ảnh và đồ thị tách từ câu) chấm điểm lại và xếp hạng lại. Mỗi kết quả kèm một bảng giải thích lấy thẳng
từ các đại lượng mô hình dùng để xếp hạng.

Kết quả trên **test** (10 696 truy vấn, pool 2 138 ảnh, chấm một lần cho mỗi mô hình chính):

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần | 39,32 | 66,39 | 77,32 | 52,00 |
| Ba kênh, không huấn luyện | 42,54 | 70,04 | 80,35 | 55,05 |
| Mô hình cấp 1 (`main_r3`) | 45,77 ± 0,23 | 72,87 ± 0,31 | 82,69 ± 0,31 | 58,14 ± 0,18 |
| **Mô hình chính (`countonly_factual_r3`)** | **47,95 ± 0,15** | **74,79 ± 0,03** | **84,38 ± 0,08** | **60,15 ± 0,15** |

Mô hình chính hơn CLIP thuần +8,63 R@1 và hơn mô hình cấp 1 +2,18 R@1 (Welch, p = 0,0005). Ablation train lại bỏ cạnh
làm R@1 val giảm 2,39 điểm; phần mất này tập trung ở câu có nhiều vật thể và có quan hệ.

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
đã chuẩn hoá alias, không lọc về VG150. Quan hệ có hướng, trùng cả ba thành phần thì gộp. Không có cạnh giữa các ảnh, không
dùng ConceptNet. Ở cấp 1, mỗi instance là một node: trung bình 26,7 node và 18,0 cạnh mỗi ảnh (train + val); 3,0% ảnh không
có cạnh. Mô hình chính dùng đồ thị đã làm sạch và nối liền (mục 3.2).

**Đồ thị câu.** Cấp 1 tách caption bằng `sng_parser` (SceneGraphParser, dựa trên spaCy): trung bình 3,06 vật thể và 1,65
quan hệ mỗi câu; 14,2% câu val không tách được quan hệ nào. Mô hình chính dùng **FACTUAL-T5 large** (Li và cộng sự, ACL
Findings 2023; model `lizhuang144/flan-t5-large-VG-factual-sg`, beam 5), một parser seq2seq được huấn luyện sẵn để sinh
scene graph từ câu. Trên val, FACTUAL giảm node đại từ từ 3,9% xuống 0,1%, và tăng tỉ lệ câu có ≥1 quan hệ từ 85,8% lên
97,2% (`experiments/improve/parser-compare/REPORT.md`). Parser chỉ chạy suy luận, không tinh chỉnh.

Nguồn: `experiments/data_stats.md`, `docs/interfaces.md`, `docs/level1-model.md` mục 1,
`experiments/improve/factual-retrain/REPORT.md`.

## 2. Chống rò rỉ dữ liệu

- File split là nguồn duy nhất; mọi mô hình so sánh dùng cùng một bộ file split và cùng danh sách ứng viên top-50 của CLIP.
- Ứng viên lúc huấn luyện: train được chia cố định (seed 0) thành 22 nhóm khoảng 2 134 ảnh. Top-50 của mỗi caption train
  chỉ lấy trong nhóm của ảnh đúng, nên độ khó giống lúc đánh giá mà không chạm val hay test.
- CLIP và FACTUAL-T5 đóng băng, chỉ chạy suy luận. Bảng vector nhãn / quan hệ / bộ ba được mã hoá bằng CLIP text đóng băng
  cho chữ của cả ba split; không có tham số nào được khớp trên các vector này.
- Luật đếm số lượng phía câu được viết tay một lần (mục 3.2), không chỉnh theo kết quả. Các hằng số khởi tạo của phép trộn
  điểm thô lấy từ 20 000 câu **train**.
- Val dùng để dừng sớm, chọn checkpoint, so các ablation và chọn bước cải tiến. Không có tập dev riêng.
- Test được chấm đúng **hai lần trong cả dự án, mỗi lần cho một mô hình chính đã chốt**:
  - 25/09: mô hình cấp 1 `main_r3` và các ablation của nó (`scripts/evaluate_test.py`, commit `f738de3` và `8b36243`);
  - 25/10: mô hình chính `countonly_factual_r3` (`scripts/improve/evaluate_test_countonly.py`, commit `8806b52`). Phase
    cải tiến chọn mọi bước trên val; test chỉ chấm sau khi Lộc chốt mô hình.
  - Mỗi lần chấm, script kiểm tra ứng viên test tái tạo đúng số CLIP thuần (R@1 39,32) trước khi chấm; mỗi file kết quả
    chỉ ghi một lần. `GraphStore` từ chối nạp test, trừ khi được gọi với `allow_test=True` và chỉ riêng split test.
- Các ablation và phân tích của phase cải tiến (mục 6–8) chỉ chạy trên val.
- Bộ 20 truy vấn đánh giá của cấp độ 2 không dùng ở đây.

## 3. Mô hình

### 3.1. Khung chung (mô hình cấp 1, `main_r3`)

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

### 3.2. Mô hình chính (`countonly_factual_r3`)

Giữ nguyên khung ở mục 3.1 và thay bốn chỗ. Mỗi chỗ là một bước của phase cải tiến, được giữ lại vì tăng R@1 val (mục 5.2).

1. **Đồ thị câu từ FACTUAL-T5 large** thay cho `sng_parser` (dữ liệu `factual_v1`).
2. **Trộn điểm thô, bỏ z-score:** điểm cuối = `a·cosine + b·S_obj + c·S_tri`, với `[a, b, c]` vẫn là softmax theo câu.
   - Ứng viên không có phần nào để so ở một kênh nhận điểm trung bình của các ứng viên có phần đó.
   - Khởi tạo tương đương bản z-score: bias tỉ lệ với (0,60; 0,28; 0,12) chia cho độ lệch chuẩn của từng kênh, đo trên
     20 000 câu train ở bước 0. Sàn của nhiệt độ loss T hạ từ 0,05 xuống 0,005.
   - Vì không có bước chuẩn hoá, mọi phần của điểm cuối tách được chính xác (mục 8).
3. **Đồ thị ảnh làm sạch và nối liền** (dữ liệu `merge_v1`):
   - **Gộp khung trùng:** hai node cùng loại (synset hoặc hypernym của WordNet; mọi nhãn chỉ người tính là
     `person.n.01`) có IoU ≥ 0,7 được gộp, khép bắc cầu bằng union-find. Node gộp lấy nhãn của khung lớn nhất và hộp bao
     các khung. Số node train giảm 7,0%.
   - **Hình học trên cạnh:** mỗi cạnh có hộp ở hai đầu được gán 6 số (độ lệch tâm theo kích thước chủ thể, log tỉ lệ
     rộng và cao, IoU, tỉ lệ diện tích), qua MLP 6 → 64 → 256 rồi cộng vào vector cạnh. Lớp cuối khởi tạo bằng 0.
   - **Cạnh kNN:** mỗi node nhận cạnh từ 5 node gần nhất (khoảng cách tâm khung chia đường chéo ảnh); nếu đồ thị còn rời
     thì thêm cầu nối giữa các mảnh (Kruskal). Các cạnh này mang vị từ học được `[none]`, chỉ truyền tin trong GAT và
     **không** được tính là bộ ba.
   - **Node tổng:** mỗi đồ thị ảnh và câu có thêm một node nối hai chiều tới mọi node, khởi tạo bằng trung bình các vector
     node; node này không tham gia so khớp.
   - Sau bước này mỗi ảnh val có trung bình 24,9 node, 17,4 cạnh có nhãn và 114,3 cạnh tự thêm; 100% đồ thị ảnh liền
     (cấp 1: 4,7%).
4. **Số lượng và bảng g** (dữ liệu `count_v1`):
   - **Phía ảnh:** số lượng của một node là số vật cùng loại trong ảnh sau khi gộp trùng, tính cả chính nó, chia nhóm
     1 / 2 / 3–4 / ≥5. Mỗi node vẫn là một vật riêng, giữ nhãn gốc.
   - **Phía câu:** luật viết tay trên câu và đồ thị FACTUAL gán mỗi vật thể một lớp: số nêu rõ (2–10, "a pair of"),
     "couple / both" → 2, "lone / single" → 1, "group / herd / several / many" → ≥3, danh từ số nhiều → ≥2, còn lại
     "không rõ".
   - **Bảng g[lớp câu, nhóm ảnh]** (7 × 4, khởi tạo 0, hàng "không rõ" cố định bằng 0) được cộng vào `sim_ij` của từng cặp
     node trong kênh vật thể, nên tác động cả attention lẫn điểm. Bảng học được có đường chéo dương (ví dụ câu "2" gặp
     node nhóm 2: +0,074) và các ô lệch số lượng âm (câu "1" gặp node nhóm ≥5: −0,165).

Thử nghiệm gom các vật cùng loại thành một node nhóm có số lượng (thí nghiệm 4) làm R@1 val giảm 0,63, nên mô hình chính
chỉ thêm số lượng, không gom nhóm.

Nguồn: `experiments/improve/fusion/raw/REPORT.md`, `experiments/improve/full-graph/REPORT.md`,
`experiments/improve/count-only/REPORT.md`.

## 4. Huấn luyện

- Loss softmax cross-entropy trên 50 ứng viên của điểm đã trộn, chia cho nhiệt độ học được `T`.
- Tối ưu: AdamW, lr 1e-3, weight decay 1e-4, batch hiệu dụng 256, dropout 0,1.
- Epoch 1 khoá `W` và `bias`, để phần đồ thị học trước khi mở cổng trộn.
- Dừng sớm khi R@1 val không tăng sau 2 epoch, tối đa 10 epoch; giữ checkpoint tốt nhất theo val. 3 seed (0, 1, 2).
- **Cấp 1:** micro-batch 128, khoảng 285 giây mỗi epoch trên RTX 4090, mỗi run 25–50 phút; 24 run chạy qua đêm 23–24/09.
  Đồ thị được đổi sẵn thành chỉ số nguyên và vector lấy bằng torch indexing trên GPU; trước đó mỗi epoch mất khoảng 45 phút.
- **Mô hình chính:** micro-batch 32 (vẫn cộng dồn gradient thành batch 256) vì bộ nhớ đệm GPU bị phân mảnh sau một lần cập
  nhật driver. Đối chứng trên công thức cấp 1 (`microbatch-control/`) cho R@1 và MRR lệch dưới 0,1 điểm so với
  micro-batch 128. Mỗi epoch 931 giây, 3 seed mất 7,03 giờ, bộ nhớ
  GPU tối đa 7,6 GB. Đồ thị liền làm số cạnh mỗi batch tăng khoảng 10 lần, nên chậm hơn cấp 1 khoảng 3 lần.

## 5. Kết quả

R@k là tỉ lệ câu có ảnh đúng nằm trong k kết quả đầu; MRR là trung bình nghịch đảo hạng của ảnh đúng (tính trong
top-50). Số dạng mean ± std qua 3 seed, lấy ở checkpoint tốt nhất theo val.

### 5.1. Test

Bảng ở phần Tóm tắt (`experiments/level1/test/`, `experiments/improve/test/README.md`). Từng seed của mô hình chính:
R@1 48,06 / 47,78 / 47,99.

So với mô hình cấp 1 (Welch, 3 seed mỗi bên): R@1 +2,18 (p = 0,0005), R@5 +1,91 (p = 0,008), R@10 +1,70 (p = 0,008),
MRR +2,00 (p = 0,0002). So với CLIP thuần: R@1 +8,63, MRR +8,15.

Chênh val → test: mô hình chính giảm 0,96 R@1 (48,91 → 47,95), mô hình cấp 1 giảm 0,71. Khoảng chênh lớn hơn một chút là
hợp lý, vì mô hình chính được chọn sau nhiều thí nghiệm đều dựa trên val.

Giới hạn trên của cả cách làm là recall@50 của CLIP: 96,66% trên test, tức 3,3% câu có ảnh đúng nằm ngoài top-50 và không
thể cứu bằng xếp lại.

### 5.2. Chuỗi cải tiến trên val

Mỗi bước đổi một thứ so với bước trước, trừ bước 3 gộp bốn thay đổi theo quyết định của Lộc. Mọi bước 3 seed.

| Bước | Mô hình | Thay đổi | R@1 | MRR | Chênh R@1 (p) |
|---|---|---|---|---|---|
| Cấp 1 | `main_r3` | — | 46,48 ± 0,24 | 58,85 ± 0,18 | |
| 1 | `main_factual_r3` | Đồ thị câu từ FACTUAL-T5 large | 47,23 ± 0,36 | 59,49 ± 0,29 | +0,76 (≈ 0,05) |
| 2 | `raw_factual_r3` | Trộn điểm thô, bỏ z-score | 47,48 ± 0,10 | 59,72 ± 0,00 | +0,25 (không có ý nghĩa) |
| 3 | `fullgraph_factual_r3` | Gộp trùng + hình học + cạnh kNN + node tổng | 48,38 ± 0,17 | 60,51 ± 0,17 | +0,90 (0,003) |
| 4 | **`countonly_factual_r3`** | Số lượng + bảng g | **48,91 ± 0,21** | **60,86 ± 0,22** | +0,53 (0,032) |

Bước 2 không có ý nghĩa thống kê, nhưng được giữ vì độ lệch giữa seed nhỏ nhất và R@1 còn đang tăng ở epoch 10.

Ở bước 3, GAT nhận gradient gấp khoảng 2,5 lần bước 2, và vector bộ ba bớt bị ép sát nhau (cosine trung bình 0,96 → 0,87).

### 5.3. Các hướng đã thử nhưng không giữ

| Thí nghiệm | Mốc lúc đó (R@1 val) | Kết quả val | Lý do không giữ |
|---|---|---|---|
| So khớp bộ ba tách thành phần chủ thể / vị từ / đối tượng | 46,48 | 46,00 ± 0,02 | Kênh bộ ba thành bản sao của kênh vật thể (tương quan 0,59) |
| Kênh mức cả đồ thị thay kênh bộ ba | 46,48 | 46,23 ± 0,13 | Không phân biệt được vị từ và chiều quan hệ |
| Kênh số lượng và vị trí theo luật, không huấn luyện | 46,48 | +0,10 / +0,05 | Kênh vị trí không có bằng chứng giúp |
| Đặt sàn cho z-score | 47,23 | 47,31 ± 0,17 | Bớt ép sát vector bộ ba nhưng không tăng thứ hạng |
| Ghép cả đồ thị (FGW, nhất quán node chung), không huấn luyện | 47,23 | tối đa +0,12 | Không đủ |
| Gom vật cùng loại thành node nhóm + số lượng | 48,38 | 47,75 ± 0,22 | Gom nhóm làm mất thông tin (−0,63, p = 0,019) |

Nguồn: `experiments/improve/README.md` và thư mục con của từng thí nghiệm.

## 6. Ablation và đối chứng

### 6.1. Mô hình cấp 1

Mỗi dòng là một mô hình **huấn luyện lại từ đầu**, cùng công thức, cùng 3 seed. R@1 (mean ± std); bảng đủ 4 chỉ số ở
`experiments/level1/README.md`.

| Cấu hình | Val | Test |
|---|---|---|
| Mô hình cấp 1 | 46,48 ± 0,24 | 45,77 ± 0,23 |
| a, b, c cố định (học trên train qua bias, dùng chung cho mọi câu) | 46,13 ± 0,15 | 45,13 ± 0,05 |
| Bỏ GAT (mỗi nhãn được biến đổi riêng, không lan truyền) | 46,71 ± 0,07 | 45,95 ± 0,20 |
| Tắt kênh bộ ba (GAT vẫn dùng cạnh) | 45,99 ± 0,05 | 45,07 ± 0,22 |
| Bỏ hẳn quan hệ (không kênh bộ ba, GAT không cạnh) | 45,98 ± 0,10 | 45,26 ± 0,14 |
| Phía câu không qua bộ mã hoá đồ thị | 46,92 ± 0,07 | 45,61 ± 0,29 |
| Huấn luyện và chấm trên đồ thị nối lại cạnh ngẫu nhiên (giữ bậc, nhãn đi theo cạnh) | 46,24 ± 0,24 | 45,50 ± 0,40 |
| Loss LambdaRank theo MRR thay cross-entropy | 45,18 ± 0,34 | 44,28 ± 0,30 |

Ở cấp 1, đồ thị có ích nhưng cạnh đóng góp ít: bỏ hẳn quan hệ chỉ làm giảm khoảng 0,5 R@1, và bỏ GAT không thua mô hình
có GAT.

### 6.2. Mô hình chính

**Ablation huấn luyện lại bỏ cạnh** (`experiments/improve/no-edges/REPORT.md`). Giữ y công thức, chỉ bỏ mọi cạnh: quan hệ
có nhãn, cạnh kNN, cạnh nối node tổng. GAT vì vậy không còn truyền tin giữa các node, và kênh bộ ba tắt theo. Số lượng và
bảng g vẫn giữ. 3 seed, chỉ chấm val.

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Mô hình chính | 48,91 ± 0,21 | 75,22 | 84,96 | 60,86 |
| Bỏ cạnh | 46,52 ± 0,02 | 73,38 | 83,23 | 58,71 |
| Chênh | −2,39 (p = 0,0024) | −1,84 (p = 0,0005) | −1,73 (p = 0,001) | −2,14 (p = 0,0015) |
| Cấp 1: bỏ cạnh | −0,50 (p = 0,055) | −0,66 | −0,54 | −0,56 |

- Cạnh chiếm 24% phần mô hình chính hơn CLIP trên val (2,39 / 9,78). Ở cấp 1 con số này là 7%.
- Bỏ cạnh, mô hình chính chỉ còn ngang mô hình cấp 1 đủ cạnh (46,52 so với 46,48). Phần mô hình chính hơn cấp 1 (+2,43 trên
  val) gần bằng đúng phần mất khi bỏ cạnh.

**Đối chứng chỉ đánh giá**, không huấn luyện lại, val, 3 checkpoint × 3 seed đối chứng (`experiments/improve/controls/`):

| Đối chứng | Mô hình chính: R@1 (chênh) | Cấp 1: R@1 (chênh) |
|---|---|---|
| Mỗi ảnh nhận scene graph của một ảnh khác | 21,91 ± 1,11 (−27,00) | 26,33 ± 0,26 (−20,15) |
| Nối lại các quan hệ có nhãn ngẫu nhiên | 48,32 ± 0,16 (−0,59) | 46,29 ± 0,22 (−0,19) |
| Bỏ bảng g lúc chấm | −0,35 ± 0,03 (câu có số nêu rõ: −1,62 ± 0,69) | — |
| Gỡ gộp trùng lúc chấm | −0,02 ± 0,05 | — |

- Mô hình dựa vào đồ thị **của đúng ảnh**: thay đồ thị của ảnh khác thì kết quả tụt xuống dưới cả CLIP thuần, mạnh hơn cấp
  1, vì phần đồ thị chiếm trung bình 68% điểm cuối.
- Xáo các quan hệ có nhãn chỉ làm giảm 0,59, trong khi bỏ mọi cạnh làm giảm 2,39. Phần lớn đóng góp của cạnh vì vậy nhiều
  khả năng nằm ở cạnh kNN và node tổng; muốn tách chính xác cần thêm ablation chỉ bỏ cạnh kNN (chưa chạy).

## 7. Phân tích mở rộng (val)

### 7.1. Độ nhạy theo trọng số kết hợp α

Ép trọng số CLIP a = α cho mọi câu; b, c giữ tỉ lệ đã học của từng câu, co theo (1 − α). Hình:
`experiments/improve/alpha-groups/alpha_sweep.png`.

| α | 0 | 0,2 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 1 |
|---|---|---|---|---|---|---|---|---|
| R@1 val | 32,81 | 40,73 | 46,52 | 47,82 | **48,14 ± 0,30** | 47,37 | 45,49 | 39,13 |

- α = 1 là CLIP thuần, α = 0 là chỉ dùng đồ thị. Đỉnh ở α = 0,6.
- a học theo từng câu cho 48,91, hơn α cố định tốt nhất 0,77 điểm, dù α tốt nhất được chọn trên chính val (nên còn lạc
  quan). Trung bình a học được là 0,57 ± 0,06, gần 0,6; phần lời thêm đến từ việc a đổi theo câu.
- Ở cấp 1, α cố định tốt nhất là 0,45 (46,17), và trọng số theo câu chỉ hơn trọng số cố định 0,35 điểm. Thang điểm hai bản
  khác nhau (điểm thô và z-score), nên chỉ so hình dạng đường cong.

### 7.2. Đồ thị giúp ở đâu: kết quả theo nhóm câu

Nhóm được định nghĩa một lần từ đồ thị FACTUAL của câu và dùng chung cho mọi mô hình. R@1 val, mean qua 3 seed; khoảng tin
cậy bootstrap ghép cặp theo câu, p là Welch giữa các seed (`experiments/improve/alpha-groups/`,
`experiments/improve/no-edges/`).

| Nhóm | Số câu | CLIP | Cấp 1 | Mô hình chính | Bỏ cạnh | Chính − CLIP | Chính − cấp 1 [KTC 95%] | Chính − bỏ cạnh |
|---|---|---|---|---|---|---|---|---|
| Tất cả | 10 633 | 39,13 | 46,48 | 48,91 | 46,52 | +9,78 | +2,43 [1,92; 2,93] | +2,39 |
| Có số lượng nêu rõ | 1 235 | 44,21 | 50,69 | 54,06 | 50,39 | +9,85 | +3,37 [1,75; 4,91] | +3,67 |
| Không có số nêu rõ | 9 398 | 38,47 | 45,92 | 48,23 | 46,02 | +9,77 | +2,31 [1,77; 2,85] | +2,22 |
| Có ≥1 quan hệ | 10 342 | 39,31 | 46,83 | 49,38 | 46,92 | +10,08 | +2,55 | +2,46 |
| Chỉ vật thể, không quan hệ | 291 | 32,99 | 33,91 | 32,19 | 32,42 | −0,80 | −1,72 [−4,7; 1,4] | −0,23 |
| 2 vật thể | 3 757 | 35,67 | 40,87 | 42,41 | 40,45 | +6,74 | +1,54 [0,72; 2,40] | +1,96 |
| ≥3 vật thể | 6 607 | 41,40 | 50,20 | 53,32 | 50,58 | +11,92 | +3,11 [2,45; 3,77] | +2,73 |

Trả lời câu hỏi của đề "tri thức quan hệ giúp truy vấn tốt hơn ở đâu":
- Phần lời của đồ thị nằm ở câu có nhiều vật thể và có quan hệ, và tăng theo số vật thể: +6,7 điểm ở câu 2 vật thể,
  +11,9 điểm ở câu từ 3 vật thể trở lên.
- Câu chỉ nêu một vật thể (2,7% số câu) thì đồ thị không giúp; CLIP đã đủ.
- Câu nêu số lượng được lợi thêm so với cấp 1 (+3,4 so với +2,3 ở các câu còn lại).
- Phần mất khi bỏ cạnh cũng tập trung ở câu có quan hệ và câu nhiều vật thể; câu chỉ có vật thể gần như không đổi.

## 8. Giải thích và phân tích 10 truy vấn

### 8.1. Giải thích sinh từ chính cơ chế xếp hạng

Mọi số trong bảng giải thích lấy từ đúng lần xếp hạng (`src/explain/graph_explainer.py`), không tìm đường đi sau khi đã có
kết quả. Với mỗi phần i của câu, bảng ghi `w_i`, phần j của ảnh có attention lớn nhất, và độ khớp `Σ_j a_ij·sim_ij`; với
vật thể có số lượng, ghi thêm lớp số lượng của câu, nhóm của node ảnh và giá trị g đã cộng.

Vì mô hình chính trộn điểm thô, điểm cuối tách **chính xác** thành:
`a·cosine + Σ_i b·w_i·Σ_j a_ij·sim_ij (vật thể) + Σ_i c·w_i·Σ_j a_ij·sim_ij (bộ ba)`.
Demo hiển thị hai cột cho mỗi dòng:
- **% điểm cuối:** phần của dòng đó trong điểm; các % cộng đúng 100%.
- **So với trung bình top-50:** giá trị dòng ở ảnh này trừ trung bình của đúng dòng đó trên 50 ứng viên. Tổng các cột này
  bằng đúng (điểm cuối − trung bình điểm cuối). Cột này cho biết dòng nào thật sự đẩy ảnh lên so với các ảnh khác. Ví dụ với
  câu "Two dogs playing with a frisbee on the grass", kênh bộ ba chiếm 59% điểm của ảnh hạng 1 nhưng chỉ góp +0,004 trong
  mức hơn trung bình +0,074; phần còn lại do CLIP (+0,030) và vật thể (+0,039).

Kiểm tra demo trên val: 200 caption gõ vào giao diện cho cùng 50 ứng viên, cùng đồ thị câu và cùng top-1 với pipeline đánh
giá; điểm cuối lệch tối đa 4·10⁻⁶ (`scripts/checks/demo_parity.py`, `experiments/improve/demo/README.md`).

### 8.2. Cách chọn 10 truy vấn

Chỉ xét câu val có ít nhất một quan hệ (89,7%), bốc ngẫu nhiên với seed 2026: 5 câu **thành công** (CLIP xếp ảnh đúng dưới
hạng 1, mô hình xếp hạng 1), 3 câu **mô hình làm tệ đi** (CLIP hạng 1, mô hình dưới hạng 1), 2 câu **cả hai đều sai**.
Cùng quy tắc với bản cấp 1 (`experiments/level1/explanations/`). Số câu mỗi loại trên toàn val:

| Checkpoint | Thành công | Làm tệ đi | Cả hai sai | Cả hai đúng |
|---|---|---|---|---|
| Mô hình chính seed 0 / 1 / 2 | 1 528 / 1 548 / 1 483 | 463 / 515 / 462 | 4 944 / 4 924 / 4 989 | 3 698 / 3 646 / 3 699 |
| Mô hình cấp 1 seed 0 | 1 235 | 434 | 5 237 | 3 727 |

So với mô hình cấp 1 (cặp seed tương ứng), mô hình chính sửa đúng 717–750 câu và làm hỏng 448–507 câu.

### 8.3. Mười truy vấn (mô hình chính, seed 0)

| Ca | Truy vấn | Hạng ảnh đúng: CLIP → mô hình | Nhận xét chính |
|---|---|---|---|
| 1 | Dining table for six in a wood paneled room with a fireplace. | 2 → 1 | Thành công |
| 2 | A cow standing along side of a large river. | 8 → 1 | Thành công |
| 3 | A man sitting in front of the pen with a dog herding sheep in the background. | 3 → 1 | Thành công |
| 4 | Someone holding a doughnut that has a bite taken out of it. | 4 → 1 | Thành công |
| 5 | People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella. | 2 → 1 | Thành công; g dùng số lượng của umbrella và people |
| 6 | A bunch of bananas are sitting on a wooden plate. | 1 → 2 | VG gọi đĩa gỗ là `tray`; thuộc tính "wooden" chưa được dùng |
| 7 | A train riding along the tracks along a bank of dirt. | 1 → 5 | Đồ thị VG của ảnh đúng rỗng (0 node), hai kênh lấy điểm trung bình |
| 8 | A bus that is reading "89 Portola Plaza" is traveling on. | 1 → 3 | Chi tiết quyết định là chữ trên biển, đồ thị không biểu diễn được |
| 9 | Two men standing in a living room playing video game with remotes. | 13 → 17 | Số lượng đúng (g +0,071 cho "two men") nhưng ảnh đúng thiếu nhãn tay cầm |
| 10 | A batter is preparing to hit the ball at a game. | 17 → 28 | Ảnh đúng thiếu nhãn `batter` / `ball` |

Ảnh, bảng giải thích đầy đủ và lời phân tích từng ca (viết sau khi xem ảnh thật): `experiments/improve/explanations/README.md`.
Ba ví dụ so với cấp 1 cũng ở đó: một câu số lượng được sửa đúng ("The two people are walking in the snow…", cấp 1 hạng 5 →
hạng 1), một câu được sửa nhờ node gộp ("A snow board sitting upright…", hạng 4 → 1), và một ca số lượng làm khớp sai
("A lone woman…" → `hair`).

### 8.4. Phép thử xoá (fidelity mức cơ bản)

Trên ảnh mô hình xếp hạng 1 của mỗi ca, xoá phần ảnh có đóng góp lớn nhất (term_j = trọng số kênh × Σ_i w_i·a_ij·sim_ij),
rồi so với trung bình 10 lần xoá ngẫu nhiên một phần cùng loại.
- Ở 10/10 ca, xoá phần lớn nhất làm điểm giảm nhiều hơn xoá ngẫu nhiên.
- Ở 7/10 ca, riêng việc xoá phần đó đã làm ảnh rơi khỏi hạng 1 (mô hình cấp 1: 5/10). Khi phần bị xoá là vật thể: 6/6 ca rơi
  hạng; khi là bộ ba: 1/4.
- Attention trung bình của cặp khớp nhất: kênh vật thể 0,51, kênh bộ ba 0,17. Kênh bộ ba có trọng số lớn nhưng khớp trải
  đều, nên thứ hạng chủ yếu do CLIP và kênh vật thể quyết định.

Đây là mức fidelity cơ bản; đo fidelity / validity / sparsity đầy đủ thuộc cấp độ 3.

**Nguyên nhân thất bại** quan sát được khi xem ảnh thật, ở cả hai mô hình:
- Chú thích VG thiếu nhãn, có ảnh có đồ thị rỗng.
- Khác biệt từ vựng giữa caption COCO và nhãn VG (`plate` / `tray`, `child` / `boy`, không có `tennis player`).
- Thuộc tính (màu, chất liệu) chưa được dùng.
- Chi tiết mà scene graph không biểu diễn được (chữ trên biển hiệu).
- Caption có nhiều ảnh đúng; ảnh đúng nằm ngoài top-50 của CLIP.

## 9. Demo

`app/demo_app.py` (Gradio, chạy được trên RTX 3070 8 GB): nhập truy vấn tiếng Anh, chọn CLIP thuần hoặc CLIP + Đồ thị, xem
top-k và bảng giải thích của từng ảnh như mục 8.1. Mặc định dùng `countonly_factual_r3_seed0` (seed chọn theo R@1 val);
`--run main_r3_seed0` chạy mô hình cấp 1. Câu gõ vào được tách bằng FACTUAL-T5 large và cùng luật số lượng như lúc đóng gói
dữ liệu; mỗi truy vấn khoảng 0,4 giây trên RTX 4090. Kịch bản video dự phòng: `docs/demo-video-script.md`.

## 10. Hạn chế

- **Bước 3 gộp bốn thay đổi.** Ablation bỏ cạnh cho thấy cạnh đóng góp 2,39 điểm, nhưng chưa tách được phần của cạnh kNN,
  node tổng và quan hệ có nhãn, và chưa có ablation bỏ GAT cho mô hình chính. Ở cấp 1, bỏ GAT không làm giảm kết quả.
- **Đếm số lượng gộp mọi người thành một loại.** "A lone woman" trong ảnh nhiều người bị coi là sai số lượng, và g có thể
  kéo node câu sang node ảnh khác nhãn (woman → hair). g đổi node được ghép ở 8,4% node câu có số lượng; khi đổi, khớp số
  lượng tăng 7,3 điểm nhưng khớp nhãn giảm 2,2 điểm.
- **Kênh bộ ba vẫn yếu khi đứng riêng.** Vector bộ ba bớt ép sát (cosine 0,97 ở cấp 1 → 0,88) nhưng attention vẫn trải đều,
  nên giải thích ở kênh bộ ba thường ghép các cặp ít liên quan. Cột "so với trung bình top-50" trong demo giúp không đọc nhầm
  kênh này là kênh quyết định.
- **Parser FACTUAL cắt token cuối.** Thư viện FactualSceneGraph 0.7.3 cắt token cuối của câu dài nhất trong mỗi batch: khoảng
  1,8% caption bị ảnh hưởng, trong đó 0,26% (train), 0,24% (val), 0,31% (test) mất một phần chữ (ví dụ "UMBRELLAS" →
  "UMBRELL"). Lỗi xảy ra như nhau ở ba split; dữ liệu không parse lại và các số đã báo giữ nguyên. Demo đã sửa lỗi này cho
  câu gõ vào.
- **Val bị dùng nhiều lần để chọn bước cải tiến,** nên val hơi lạc quan (chênh val → test −0,96 R@1).
- **Chi phí:** mỗi epoch chậm gấp khoảng 3 lần cấp 1 do cạnh kNN và bảng g.
- **Thuộc tính chưa dùng.** Phép thử không huấn luyện chỉ tăng +0,36 R@1, dưới ngưỡng 0,5 điểm đặt trước
  (`experiments/checks/attribute-probe.REPORT.md`); nhiều ca thất bại liên quan đến thuộc tính.
- Mới thực nghiệm trên một tập công khai; chênh lệch giữa các cấu hình chỉ kiểm định với 3 seed.
- Xếp lại không cứu được 3,3% câu có ảnh đúng nằm ngoài top-50 của CLIP.

## 11. Đóng góp

Phần cấp 1 và phase cải tiến do Nguyễn Đình Lộc làm cùng công cụ AI; Lộc chốt thiết kế, chọn từng bước giữ lại và chọn mô
hình chính trước khi chấm test. Chi tiết theo phần việc: `report/declaration-ai-oss.md`.

## 12. Nguồn số liệu

| Mục | File |
|---|---|
| Dữ liệu, đồ thị cấp 1 | `experiments/data_stats.md`, `docs/level1-model.md` |
| Test | `experiments/level1/test/`, `experiments/improve/test/README.md` |
| Ablation cấp 1 | `experiments/level1/README.md` |
| Chuỗi cải tiến | `experiments/improve/README.md`, `factual-retrain/`, `fusion/raw/`, `full-graph/`, `count-only/` |
| Ablation bỏ cạnh, theo nhóm | `experiments/improve/no-edges/REPORT.md` |
| Đối chứng lúc chấm | `experiments/improve/controls/README.md` |
| Quét α, nhóm câu | `experiments/improve/alpha-groups/REPORT.md` |
| 10 truy vấn | `experiments/improve/explanations/README.md`, `experiments/level1/explanations/README.md` |
| Demo | `experiments/improve/demo/README.md` |
