# Kế hoạch hoàn tất cấp độ 2 (MKG-W + NativE)

Lập ngày 16/11/2026.
- **Người làm phần còn lại: Khoa.**
- Khang giữ hàng 32 (chuyển giao sang MMKG tiếng Việt).
- Lộc review, merge và điều phối máy GPU.

Hạn nộp và vấn đáp: 22/11. **Đóng băng thực nghiệm: 19/11, 23:59.** Sau mốc này chỉ viết báo cáo, không đổi số.

Nguồn chuẩn cho mọi yêu cầu vẫn là đề: `assignments/Project03_Multimodal_XAI.pdf`, trang 3 và mục 6.

## 1. Đề yêu cầu gì ở cấp 2

1. Thực nghiệm trên **ít nhất 2 tập công khai**. Nhóm dùng VG∩COCO (cấp 1, đã xong) và **MKG-W**.
2. Tái lập **một phương pháp mới từ 2024 trở lại**, đánh giá trên cùng pipeline với cấp 1. Nhóm chọn **NativE (SIGIR 2024)**.
3. Có **ít nhất 20 truy vấn tự viết**, trong đó có nhóm gián tiếp. Báo cáo **tách riêng nhóm trực tiếp và nhóm gián tiếp**.

Mục "Nhiệm vụ 2" của đề còn yêu cầu ablation: bỏ cạnh, xáo cạnh hoặc thay bằng đồ thị ngẫu nhiên.

## 2. Hiện trạng (nhánh `khang/mkgw-level2`, chưa có PR)

| Phần | Ai làm | Trạng thái |
|---|---|---|
| Dữ liệu MKG-W: tải ảnh từ kho MMRNS, chia tập, che tên thực thể trong truy vấn | Đạt (24–25/09) | Xong |
| Bộ khớp tên thực thể và hàm điểm đồ thị hai kênh (`src/retrieval/mkgw_graph_score.py`) | Đạt | Xong |
| CLIP thuần trên val: R@1 24,33, MRR 34,24 (`experiments/level2/clip_baseline/`) | Đạt | Xong, 1 lần chạy |
| Bộ 25 truy vấn tự viết: 10 trực tiếp, 15 gián tiếp (`data/queries/mkgw_queries_hand.json`) | Khang (16/11) | Đã chốt |
| Script đánh giá đủ quy trình (`scripts/eval_mkgw_level2.py`) | Khang | Viết xong, chưa chạy |
| Notebook molab để huấn luyện NativE trên GPU (`notebooks/mkgw_molab.py`) | Khang | Viết xong |
| Script chuyển giao sang MMKG tiếng Việt (`scripts/eval_vnmmkg_transfer.py`) | Khang | Viết xong, chưa chạy |
| **Huấn luyện NativE thật** (bước a và bước b) | — | **Chưa chạy.** Mới chạy thử 2 epoch trên CPU |

Kết quả `rerank_sweep` hiện có (R@1 32,17 trên val) dùng embedding của bản chạy thử 2 epoch. Kênh embedding vì thế không giúp gì: khi dò trên val, trọng số tốt nhất của kênh này là 0. Số này **không dùng trong báo cáo**.

## 3. Chốt dữ liệu (không đổi nữa)

1. **Tập chia MKG-W bản v2 (25/09):**
   - 8.920 thực thể, mỗi thực thể có mô tả và ảnh lấy từ kho của tác giả MMRNS;
   - train 7.136, val 892, test 892, seed 0;
   - SHA-256 của mọi file nằm trong `data/MANIFEST.md`.
   
   **Không dựng lại tập chia.** Các file không commit thì lấy trên Drive nhóm, đúng cấu trúc ghi trong MANIFEST.
2. **Truy vấn:** mô tả tiếng Anh của thực thể, đã che tên. **Ứng viên:** mỗi thực thể một ảnh, lấy tất cả ảnh của tập đang đánh giá (892 ảnh cho val, 892 cho test).
3. **Đồ thị tri thức:** MKG-W của NativE, gồm 15.000 thực thể, 169 quan hệ, và 34.196 / 4.276 / 4.274 triple. **Kênh đồ thị chỉ dùng triple thuộc phần train của NativE.**
4. **Dữ liệu huấn luyện NativE:**
   - bước a dùng MKG-W gốc;
   - bước b dùng bản `MKG-W-rmask` (`scripts/mask_native_modalities.py`), trong đó vector ảnh và vector mô tả của mọi thực thể thuộc val và test bị đặt về 0.
   
   Lý do: bảng đặc trưng văn bản của NativE mã hoá chính mô tả, tức chính câu truy vấn, còn bảng đặc trưng ảnh mã hoá chính ảnh ứng viên. Dùng nguyên các bảng này là rò rỉ dữ liệu theo mục 6 của đề.
5. **Bộ 25 truy vấn tự viết:**
   - đã khoá, không sửa nữa;
   - ứng viên là ảnh của tập test;
   - chỉ dùng để đánh giá, không chọn bất cứ siêu tham số nào trên bộ này.
   
   Quy tắc xây và kiểm tra nằm ở đầu file `scripts/build_mkgw_hand_queries.py`.
6. **Bộ mã hoá:** CLIP ViT-B/32 (quickgelu, trọng số OpenAI), đóng băng, giống cấp 1. Lấy top-K = 50.
7. **MMKG tiếng Việt:** giữ nguyên bản đã merge ở PR #31.

## 4. Chốt cách làm

**Điểm xếp hạng** trên top-50 của CLIP, z-score tính trong danh sách của từng truy vấn:

`score = α·z(CLIP) + (1−α)·z(β·z(emb) + (1−β)·z(triple))`

- `triple`: số cạnh trực tiếp (chỉ thuộc phần train) giữa ứng viên và các thực thể được nhắc trong truy vấn.
- `emb`: cosine lớn nhất giữa embedding NativE của ứng viên và embedding của các thực thể được nhắc.
- Lưới tìm: α từ 0 đến 1, bước 0,05; β thuộc {0; 0,25; 0,5; 0,75; 1}.
- **Chọn trên val, riêng cho từng seed**, theo R@1, hoà thì xét MRR. Lựa chọn ghi vào `experiments/level2/val/choice.json`.
- **Test chỉ chạy một lần**, đọc lại `choice.json` và không chỉnh gì thêm.

**NativE:**
- code ở commit `c72291c` của `zjukg/NativE`;
- siêu tham số giữ đúng `scripts/run_mkgw.sh` của tác giả: 1.000 epoch, dim 250, margin 4, 128 mẫu âm, lr 1e-4, batch 1024;
- giữa các lần chạy chỉ đổi bộ dữ liệu và seed.

| Lần chạy | Dữ liệu | Seed | Dùng cho |
|---|---|---|---|
| `repro` | MKG-W gốc | 42 | Bước a: so MRR, Hits@1/3/10 với bảng MKG-W trong bài báo. Chênh khoảng 1 điểm trở xuống thì coi là tái lập được |
| `rmask` | MKG-W-rmask | 0, 1, 2 | Bước b: xuất embedding cho kênh `emb`, mỗi seed một file |

**Các dòng trong bảng kết quả MKG-W** (val và test, mean ± std qua 3 seed):

| Dòng | Ý nghĩa |
|---|---|
| `clip` | CLIP thuần |
| `kg_triple` | CLIP + kênh đếm cạnh (β = 0). Không cần huấn luyện |
| `native` | **Dòng chính của cấp 2**: CLIP + embedding NativE + đếm cạnh |
| `native_emb` | Ablation bỏ kênh cạnh, chỉ giữ embedding (β = 1) |
| `shuffled_edges` | Ablation xáo cạnh: trong mỗi quan hệ, đổi chỗ đuôi các triple, giữ bậc của đầu |
| `random_emb` | Ablation thay embedding NativE bằng nhiễu Gauss |

Báo cáo thêm:
- độ nhạy theo α;
- tách nhóm truy vấn có hoặc không khớp được thực thể nào;
- **bộ 25 câu tách trực tiếp và gián tiếp** (chạy `--hand`).

**Giải thích:** là danh sách cạnh và thực thể khớp mà hàm điểm trả về cùng với điểm. Danh sách này sinh ra từ chính cơ chế xếp hạng, không tìm thêm sau khi đã có kết quả.

**Chuyển giao sang MMKG tiếng Việt (hàng 32, Khang):**
- dùng kênh `kg_triple`;
- A0 (chính): zero-shot, lấy α đã chọn trên val MKG-W và bộ mã hoá do phép thử CLIP đa ngữ chọn;
- A1 (phụ): chọn lại α trên val tiếng Việt, rồi chạy test một lần.

Nếu dòng `native` tốt hơn `kg_triple` trên MKG-W, báo cáo phải ghi rõ vì sao chuyển giao kênh cạnh mà không chuyển NativE: NativE phải huấn luyện lại trên một đồ thị chỉ có 325 triple.

**Không làm (đề không bắt buộc, thiếu thời gian):**
- **GAT của cấp 1 trên MKG-W** (phần "CLIP + GNN" ở hàng 21). MKG-W không có scene graph cho từng ảnh. Ở bước b, embedding NativE thay vai trò của GNN, đúng như kế hoạch ban đầu. Hàng 21 coi như hoàn thành bằng hai dòng `clip` và `kg_triple`.
- **Bộ 20 truy vấn tự viết cho VG∩COCO** (hàng 24 ghi "mỗi tập"). Đề chỉ yêu cầu một bộ. Phía VG∩COCO đã có phần tách nhóm caption: có số lượng, có quan hệ, số vật thể.
- Chạy NativE trên Visual Genome (hàng 23, "nếu kịp").

## 5. Phân công

### Khoa: toàn bộ phần cấp 2 còn lại (hàng 21, 22, 23, 25, 26)

Làm trên nhánh `khoa/mkgw-level2-runs`, tách ra từ `origin/khang/mkgw-level2`. Không cần chờ PR của Khang merge; khi PR đó vào `main` thì rebase lên `main`.

| # | Việc | Hạn | Sản phẩm |
|---|---|---|---|
| 1 | Đọc file này, `experiments/level2/native-repro/README.md` (nhật ký của Đạt, có lỗi `Base.so` và cách sửa), và phần đầu ba script `eval_mkgw_level2.py`, `mask_native_modalities.py`, `export_native_embeddings.py`. Lấy dữ liệu theo `data/MANIFEST.md`, hoặc chạy notebook molab bước 1–3 | **17/11 sáng** | Môi trường chạy được |
| 2 | Chạy đo 10 epoch trên GPU molab: bao nhiêu giây một epoch, phiên molab sống được bao lâu. Nếu 1.000 epoch dài hơn giới hạn phiên, hoặc không có GPU, **báo Lộc ngay để chuyển sang máy 4090** | **17/11, 12:00** | Một dòng báo trong nhóm chat |
| 3 | Chạy 4 lần `rmask:0`, `rmask:1`, `rmask:2`, `repro:42` (notebook bước 4, hoặc máy 4090). Ưu tiên `rmask` vì đề bắt buộc 3 seed | Bắt đầu 17/11, xong **18/11 tối** | Log ở `experiments/level2/native-train/` |
| 4 | Bước a: bảng MRR, Hits@1/3/10 của `repro` đặt cạnh số trong bài báo, kèm nhận xét chênh lệch | 18/11 | Bổ sung `experiments/level2/native-repro/README.md`, thêm `metrics.json` |
| 5 | Xuất embedding: `scripts/export_native_embeddings.py --what ent --suffix _seed{0,1,2}` ra `data/features/mkgw_native_ent_seed{s}.npy` và `.ids.json`. Ghi SHA-256 vào `data/MANIFEST.md` và đưa file lên Drive nhóm | Theo từng seed | File embedding và MANIFEST |
| 6 | Có seed 0 thì chạy thử `eval_mkgw_level2.py --split val`. Đủ 3 seed thì chạy val bản cuối, ra `choice.json`, **nhắn Lộc xem**, sau đó chạy `--split test` một lần, rồi chạy `--hand` | 19/11 | `experiments/level2/{val,test}/` |
| 7 | Có `choice.json` rồi thì báo Khang để chạy chuyển giao A0 | 19/11 | — |
| 8 | Ví dụ giải thích trên MKG-W: ít nhất 3 câu tự viết (cả trực tiếp lẫn gián tiếp), mỗi câu chỉ ra thực thể khớp và cạnh đã cộng điểm | 20/11 | Bảng hoặc hình cho báo cáo |
| 9 | Hàng 26: phân tích độ nhạy theo α và so các nguồn cạnh (chỉ cạnh, chỉ embedding, cả hai, cạnh bị xáo) | 20/11 | Đoạn phân tích và hình |
| 10 | Phần báo cáo về cấp 2 gồm: NativE và tái lập bước a; bản `rmask` chống rò rỉ; bảng kết quả MKG-W (val và test, mean ± std); trực tiếp so với gián tiếp; ablation; phần mở rộng | 20/11 | Mục trong báo cáo |

### Khang: việc trên nhánh của Khang và hàng 32

| # | Việc | Hạn |
|---|---|---|
| 1 | Mở PR `khang/mkgw-level2` → `main` để Lộc review code trong lúc NativE đang chạy | 17/11 |
| 2 | Sửa `experiments/data_stats.md` trên nhánh: bỏ mục MKG-W cũ (5.752 thực thể), chỉ giữ bản v2 | 17/11 |
| 3 | Commit `experiments/checks/multilingual_clip_probe.json` (kết quả phép thử CLIP đa ngữ), vì chuyển giao A0 cần file này | 17/11 |
| 4 | Hàng 32: chạy A1 ngay sau việc 3, vì A1 không cần `choice.json`. A0 chạy khi Khoa có `choice.json`. Viết phần phân tích chuyển giao trong báo cáo | 19–20/11 |

### Đạt

Hỗ trợ khi Khoa hỏi về dữ liệu MKG-W và hàm điểm, là phần Đạt đã viết.

### Lộc

- Review và merge PR. PR của thành viên chỉ Lộc merge.
- Nếu molab không chạy được thì giao NativE cho máy 4090. Máy rảnh từ khoảng sáng 17/11, sau khi chạy xong phần chỉ số xAI, và chạy từng lần một.
- Ghép báo cáo, cập nhật trạng thái trong `task-assignment.xlsx`.

## 6. Chỗ nối

- **Embedding:**
  - file `data/features/mkgw_native_ent_seed{s}.npy`, kích thước 15.000 × 500, float32, đã chuẩn hoá L2, theo thứ tự `entity2id`;
  - file `.ids.json` đi kèm chứa QID;
  - script đánh giá đọc qua `--native-prefix mkgw_native_ent_seed`.
- **Lựa chọn trên val:** `experiments/level2/val/choice.json`. File này được ghi một lần ở bước 6 của Khoa, và test cùng chuyển giao A0 của Khang đọc lại nó.

## 7. Nếu trễ hoặc kết quả xấu

- **Đến 18/11 tối mà chưa đủ 3 seed `rmask`:** báo cáo với số seed đang có và ghi rõ là thiếu. Không giảm số epoch để chạy cho kịp.
- **Kênh embedding không giúp trên val** (chọn ra β = 0): vẫn báo cáo đầy đủ. Kết quả âm có phân tích (`native_emb`, `random_emb`, tách nhóm) vẫn hợp lệ, và việc tái lập ở bước a vẫn chứng minh nhóm đã chạy được NativE.
- **NativE không huấn luyện được trước 19/11:** cấp 2 thiếu phần "phương pháp mới". Báo Lộc ngay khi thấy nguy cơ, đừng đợi đến hạn.

## 8. Quy tắc nhắc lại

- Test chạy một lần cho mỗi mô hình đã chốt.
- Bộ 25 câu không dùng để chỉnh mô hình.
- Mỗi con số trong báo cáo phải có log hoặc json trong `experiments/`, kèm seed.
- Commit message bằng tiếng Anh, nội dung bằng tiếng Việt.
- Code mở hoặc phần làm có AI hỗ trợ phải khai báo trong `report/declaration-ai-oss.md`.
- Mỗi người phải tự giải thích được phần của mình khi vấn đáp.
