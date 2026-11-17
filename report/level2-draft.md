# Nháp báo cáo: Cấp độ 2, MKG-W và tái lập NativE

Nháp để Lộc duyệt rồi ghép vào bản báo cáo chính. Cùng cấu trúc và quy ước với `report/level1-draft.md`:
thuật ngữ tiếng Anh giữ nguyên, kèm giải thích lần đầu; mọi con số lấy từ file kết quả trong `experiments/`,
đường dẫn ghi ở cuối từng mục và ở mục 11. Người soạn nháp: Khoa cùng công cụ AI (Claude), theo phần việc
cấp 2 trong `docs/level2-plan.md` §5. Bản nháp ngày **2026-11-17**.

> ⚠️ **BẢN NHÁP — CHƯA ĐỦ SỐ.** Các thí nghiệm GPU (train NativE 4 lần, export embedding, eval val/test)
> chạy trên molab/4090/Kaggle, **đóng băng 23:59 ngày 19/11**. Trong văn bản này:
> - ô bảng ghi `__,__` và `± __,__` là **chưa có số**, chỉ điền sau khi chạy, **tuyệt đối không điền số bịa**
>   (CLAUDE.md và đề mục 6: mỗi số phải có log/json kèm seed);
> - `⟨… nguồn …⟩` là chỗ điền trong câu, kèm file sẽ lấy số;
> - số **đã có thật** (CLIP thuần trên val, các con số dữ liệu cố định, coverage) được điền sẵn và ghi nguồn.
>
> Nếu đến 19/11 vẫn thiếu seed hoặc một lần chạy, **ghi rõ là thiếu** ngay tại chỗ đó; không giảm epoch để
> chạy cho kịp (`docs/level2-plan.md` §7).

## Tóm tắt

Bài toán vẫn là truy vấn văn bản → ảnh (nhánh A của đề), nhưng trên tập công khai thứ hai **MKG-W** (một
multi-modal knowledge graph: đồ thị tri thức mà mỗi thực thể có kèm mô tả và ảnh). Truy vấn là **mô tả tiếng
Anh của một thực thể đã che tên**; đáp án đúng là ảnh của chính thực thể đó; ứng viên là toàn bộ ảnh của tập
đang đánh giá. CLIP đóng băng lấy top-50, rồi một **điểm đồ thị hai kênh** chấm lại và xếp hạng lại; mỗi kết
quả kèm danh sách thực thể/cạnh khớp lấy thẳng từ đại lượng dùng để xếp hạng (giải thích).

Phương pháp mới tái lập theo đề (một MMKGC từ 2024) là **NativE (SIGIR 2024)**. Nhóm dùng NativE theo hai
bước: **bước a** tái lập link prediction gốc trên MKG-W để so với bài báo; **bước b** lấy **embedding thực thể
của NativE** làm một trong hai kênh đồ thị thay cho GNN của cấp 1.

Kết quả chính trên **test** (892 truy vấn, pool 892 ảnh, chấm **một lần** cho mỗi mô hình đã chốt), mean ± std
qua 3 seed:

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần (`clip`) | __,__ | __,__ | __,__ | __,__ |
| CLIP + đếm cạnh (`kg_triple`) | __,__ | __,__ | __,__ | __,__ |
| **CLIP + NativE + đếm cạnh (`native`)** | **__,__ ± __,__** | **__,__ ± __,__** | **__,__ ± __,__** | **__,__ ± __,__** |

*(nguồn: `experiments/level2/test/summary.json`; chi tiết 6 dòng ở mục 5.3)*

Kết luận (điền sau khi có số): ⟨`native` hơn/không hơn `clip` và `kg_triple` bao nhiêu điểm R@1, có ý nghĩa
thống kê không⟩. Kết quả tái lập bước a: ⟨chênh MRR/Hits với bài báo NativE⟩. Lưu ý trước: **kết quả âm vẫn hợp
lệ** — nếu kênh embedding không giúp (val chọn ra β = 0), việc tái lập bước a vẫn chứng minh nhóm chạy được
NativE, và các đối chứng (`native_emb`, `random_emb`, tách nhóm) vẫn có giá trị phân tích (`docs/level2-plan.md` §7).

## 1. Dữ liệu và cách chia tập

Tập công khai thứ hai là **MKG-W** (MKG-Wikidata), một MMKGC do nhóm MMRNS phát hành; mỗi thực thể là một URI
Wikidata, có nhãn/mô tả/alias và một hoặc nhiều ảnh. Nhóm dùng MKG-W cho hai việc khác nhau nhưng cùng một bộ
thực thể:

- **Đồ thị tri thức (cho NativE và cho kênh đếm cạnh):** bản MKG-W của NativE — **15.000 thực thể, 169 quan
  hệ, 34.196 / 4.276 / 4.274 triple** cho train / valid / test của bài toán link prediction (KGC). **Chỉ triple
  thuộc phần KGC-train (34.196) mới được đưa vào đồ thị** của kênh đếm cạnh (chống rò rỉ, mục 2).
- **Bài toán retrieval (truy vấn → ảnh):** chỉ xét các thực thể vừa có mô tả vừa có ảnh lấy từ kho của chính
  nhóm tác giả MMRNS. Chia tập bản v2 (25/09, Đạt), seed 0:

| Split | Thực thể = ảnh = truy vấn |
|---|---:|
| train | 7 136 |
| val | 892 |
| test | 892 |
| **Tổng** | **8 920** |

Truy vấn là **mô tả tiếng Anh của thực thể, đã che tên** (ví dụ "a commune in Pyrénées-Atlantiques, France").
Khi đánh giá một split, **pool ứng viên là toàn bộ ảnh của chính split đó** (892 ảnh cho val, 892 cho test),
mỗi thực thể một ảnh (lấy ảnh đầu theo thứ tự tên, thu về JPEG 640px). **Không dựng lại tập chia**: mọi mô hình
so sánh dùng chung bộ file split này; file không commit thì lấy trên Drive nhóm đúng cấu trúc và SHA-256 trong
`data/MANIFEST.md`.

**Nguồn ảnh.** Ban đầu tải ảnh P18 từ Wikimedia Commons nhưng bị chặn rate-limit (HTTP 429). Thay bằng kho
phát hành của nhóm MMRNS (`MKG-W_img.zip`, 6,4 GB + `ent_links` ánh xạ DBpedia ↔ QID): 37.192 ảnh của 8.961
thực thể. **Nhất quán với embedding chính thức của NativE** — bảng `MKG-W-visual.pth` của NativE được trích
từ đúng bộ ảnh này. (Bản split cũ theo ảnh P18 có 5.752 thực thể; bản v2 dùng nguồn MMRNS có 8.920.)

**Độ phủ của kênh đồ thị** (đo trên train+val, **không đọc test**; `experiments/checks/mkgw_link_coverage.json`):

| Split | Truy vấn link được ≥1 thực thể | Đáp án có cạnh KGC-train trực tiếp |
|---|---:|---:|
| train | 81,5 % | 42,8 % |
| val | 78,9 % | 45,2 % |

Nghĩa là khoảng 4/5 truy vấn nhắc tới ít nhất một thực thể khác mà `EntityLinker` khớp được, và khoảng 45 %
truy vấn val có sẵn một cạnh KGC-train trực tiếp nối đáp án với thực thể được nhắc — đủ điều kiện cho kênh đồ
thị có ý nghĩa.

Nguồn: `docs/level2-plan.md` §3, `experiments/level2/native-repro/README.md`, `data/MANIFEST.md`,
`experiments/checks/mkgw_link_coverage.json`.

## 2. Chống rò rỉ dữ liệu

Đây là tiêu chí chấm (đề mục 6). Ở cấp 2 có một nguy cơ rò rỉ **đặc thù của NativE** cần xử lý riêng.

- **File split là nguồn duy nhất**; mọi mô hình (6 dòng ở mục 5) dùng cùng bộ split và cùng danh sách top-50
  của CLIP.
- **Kênh đếm cạnh chỉ dùng triple KGC-train** (34.196). Hàm điểm `MkgwGraphScorer` có `assert kgc_split ==
  "train"` khi dựng đồ thị, nên cạnh thuộc KGC-valid/test không bao giờ lọt vào (`src/retrieval/mkgw_graph_score.py`).
- **Rò rỉ đặc thù của NativE và bản `MKG-W-rmask`.** NativE có hai bảng đặc trưng đóng băng: `MKG-W-textual.pth`
  mã hoá **chính mô tả thực thể** (tức chính câu truy vấn), và `MKG-W-visual.pth` mã hoá **chính ảnh ứng viên**.
  Nếu huấn luyện NativE trên cả 15.000 hàng, các lớp chiếu và embedding cấu trúc sẽ "nhìn thấy" trực tiếp truy
  vấn và ứng viên của val/test. Vì vậy, cho **bước b**, nhóm tạo bản sao **`MKG-W-rmask`**
  (`scripts/mask_native_modalities.py`): **đặt về 0 hàng visual và textual của mọi thực thể thuộc val hoặc test**
  của retrieval (đúng tình huống "thiếu modality" mà NativE vốn được thiết kế để xử lý). Triple KGC giữ nguyên.
  Bản MKG-W gốc chỉ dùng cho **bước a** (tái lập link prediction, không đụng tới retrieval).
- **α và β chọn trên val, riêng từng seed**, theo R@1 (hoà thì xét MRR); test đọc lại `choice.json` và không
  chỉnh gì thêm (`experiments/level2/val/choice.json`).
- **Bộ 25 truy vấn tự viết** (mục 4.3) đã khoá, chỉ để **đo**; không chọn bất kỳ siêu tham số nào trên bộ này,
  nó tái dùng α/β đã chọn trên val.
- **Test chấm đúng một lần** cho mỗi mô hình đã chốt (`eval_mkgw_level2.py --split test`).
- Mọi phân tích mở rộng và ablation (mục 6–7) chọn/chạy trên **val**.

## 3. Phương pháp

### 3.1. NativE (SIGIR 2024) là gì

**NativE** (*NativE: Multi-modal Knowledge Graph Completion in the Wild*, SIGIR 2024, arXiv 2406.17605) là một
mô hình **MMKGC** (Multi-Modal Knowledge Graph Completion — hoàn thành đồ thị tri thức đa phương thức: dự đoán
cạnh còn thiếu bằng cả cấu trúc đồ thị lẫn ảnh/chữ của thực thể). Hai điểm chính:

1. **Lõi cấu trúc là AdvRelRotatE** — một biến thể của RotatE (mỗi quan hệ là một phép quay trong không gian
   phức), huấn luyện kiểu đối kháng (adversarial, WGAN-GP) để sinh mẫu âm khó.
2. **Hợp nhất modality theo quan hệ (relation-gated)**: đặc trưng ảnh và chữ được chiếu rồi trộn vào embedding
   thực thể với trọng số phụ thuộc quan hệ; mô hình được thiết kế để **chịu được thực thể thiếu ảnh hoặc thiếu
   chữ** ("in the wild"), chính là tính chất nhóm tận dụng cho bản `MKG-W-rmask`.

Mã dùng để tái lập: `zjukg/NativE`, commit `c72291c` (2024-08-12), clone ngoài repo nhóm (không commit).
**Sự cố đã biết:** `mmkgc/release/Base.so` kèm sẵn bị segfault khi đọc `train2id.txt` trên một số máy Linux;
sửa bằng rebuild từ source (`cd mmkgc && bash make.sh`, cần g++) — ai chạy máy mới cũng nên rebuild trước
(`experiments/level2/native-repro/README.md`).

### 3.2. Bước a — tái lập link prediction gốc

Chạy `repro` (MKG-W gốc, seed 42) đúng siêu tham số của tác giả (mục 4), rồi so **MRR và Hits@1/3/10** với bảng
MKG-W trong bài báo. Tiêu chí: **chênh ≤ ~1 điểm thì coi là tái lập được**.

| Chỉ số | Bài báo NativE (MKG-W) | Tái lập `repro` (seed 42) | Chênh |
|---|---|---|---|
| MRR | ⟨điền từ bảng MKG-W bài báo⟩ | __,__ | __,__ |
| Hits@1 | ⟨điền từ bài báo⟩ | __,__ | __,__ |
| Hits@3 | ⟨điền từ bài báo⟩ | __,__ | __,__ |
| Hits@10 | ⟨điền từ bài báo⟩ | __,__ | __,__ |

*(cột "Tái lập": `experiments/level2/native-repro/` và `metrics.json`; cột "Bài báo": điền từ Bảng kết quả
MKG-W của arXiv 2406.17605 — trích đúng, ghi rõ số bảng/trang.)* Nhận xét: ⟨đạt/không đạt ngưỡng 1 điểm, lý do⟩.

### 3.3. Bước b — điểm đồ thị cho retrieval

Truy vấn là mô tả đã che tên, nhưng **vẫn thường nhắc tên thực thể khác** ("a rock guitarist who was a member
of *Pink Floyd*"). Kênh đồ thị khai thác điều đó qua ba bước (`src/retrieval/mkgw_graph_score.py`):

1. **`EntityLinker`** quét truy vấn, khớp mọi label/alias của thực thể KG xuất hiện trong câu, **ưu tiên khớp
   dài nhất** (n-gram tối đa 5 token; "New York City" không khớp lại thành "York").
2. Với mỗi ảnh ứng viên (= một thực thể), tính **hai kênh điểm** với các thực thể được link:
   - **`triple`**: **số cạnh KGC-train trực tiếp** giữa ứng viên và thực thể được nhắc (không cần huấn luyện,
     xác định hoàn toàn);
   - **`emb`**: **cosine lớn nhất** giữa embedding NativE của ứng viên và embedding của một thực thể được nhắc
     (vector đã chuẩn hoá L2; `None` khi không link được gì, coi là trung tính).
3. **Các match dùng để tính điểm được trả về cùng điểm — chính là giải thích** (đề: giải thích phải sinh từ cơ
   chế xếp hạng, không hậu kiểm).

**Công thức trộn** (z-score tính trong top-50 của từng truy vấn), đúng khung cấp 1:

```
score = α · z(CLIP) + (1 − α) · z( β · z(emb) + (1 − β) · z(triple) )
```

- Lưới tìm: **α từ 0 đến 1, bước 0,05**; **β ∈ {0; 0,25; 0,5; 0,75; 1}**.
- Chọn trên **val**, riêng từng seed, theo R@1 (hoà thì xét MRR).

Nguồn code: `src/retrieval/mkgw_graph_score.py`, `scripts/eval_mkgw_level2.py`.

## 4. Huấn luyện NativE

Siêu tham số giữ **đúng `scripts/run_mkgw.sh` của tác giả**; giữa các lần chạy chỉ đổi bộ dữ liệu và seed:

- **1.000 epoch**, dim 250, margin 4, 128 mẫu âm (neg_num), lr 1e-4, batch 1024 (`mu`/`lrg` = 1e-4).
- Mã ở commit `c72291c`.

Bốn lần chạy:

| Lần chạy | Dữ liệu | Seed | Dùng cho |
|---|---|---|---|
| `repro` | MKG-W gốc | 42 | Bước a: so bài báo (mục 3.2) |
| `rmask` | MKG-W-rmask | 0 | Bước b: xuất embedding kênh `emb` |
| `rmask` | MKG-W-rmask | 1 | Bước b |
| `rmask` | MKG-W-rmask | 2 | Bước b |

Ưu tiên 3 seed `rmask` vì đề bắt buộc 3 seed cho thực nghiệm chính. Thời gian: CPU ~6,5 phút/epoch
(→ 4–5 ngày, không khả thi), nên **bắt buộc chạy GPU**; ước lượng thật đo ở bước 10-epoch trước
(`experiments/level2/native-train/`). Embedding xuất bằng `scripts/export_native_embeddings.py --what ent`
ra `data/features/mkgw_native_ent_seed{0,1,2}.npy` (15.000 × 500, float32, chuẩn hoá L2, theo thứ tự
`entity2id`) + `.ids.json`; SHA-256 ghi vào `data/MANIFEST.md`, **không commit file `.npy`**.

Thời gian thật (điền sau khi chạy): mỗi epoch ⟨… giây⟩ trên ⟨GPU⟩; mỗi lần chạy 1.000 epoch ⟨… giờ⟩; tổng 4
lần ⟨… giờ⟩. Nguồn: `experiments/level2/native-train/*.log`.

## 5. Kết quả

R@k là tỉ lệ truy vấn có ảnh đúng nằm trong k kết quả đầu; MRR là trung bình nghịch đảo hạng của ảnh đúng (tính
trong top-50). Số dạng **mean ± std qua 3 seed** (`rmask` 0/1/2), trừ `clip` và `kg_triple` không phụ thuộc
embedding.

**Sáu dòng kết quả:**

| Dòng | Ý nghĩa |
|---|---|
| `clip` | CLIP thuần (không đồ thị) |
| `kg_triple` | CLIP + kênh đếm cạnh (β = 0); không cần huấn luyện; α từ val |
| `native` | **Dòng chính cấp 2**: CLIP + embedding NativE + đếm cạnh; α, β từ val |
| `native_emb` | Ablation bỏ kênh cạnh, chỉ giữ embedding (β = 1) |
| `shuffled_edges` | Ablation xáo cạnh: trong mỗi quan hệ đổi chỗ đuôi các triple, giữ bậc của đầu |
| `random_emb` | Ablation thay embedding NativE bằng nhiễu Gauss (đã chuẩn hoá) |

### 5.1. CLIP thuần (mốc đã có)

Trên **val** (1 lần chạy, seed 0 — CLIP đóng băng nên không phụ thuộc seed huấn luyện; `experiments/level2/clip_baseline/metrics_val.json`):

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần, val | 24,33 | 44,51 | 53,70 | 34,24 |

Bộ mã hoá: **CLIP ViT-B/32 (QuickGELU, trọng số OpenAI), đóng băng**, giống cấp 1; top-K = 50.

### 5.2. Bảng kết quả trên **val** (chọn α/β)

| Dòng | α, β (val) | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| `clip` | — | 24,33 | 44,51 | 53,70 | 34,24 |
| `kg_triple` | __ , 0 | __,__ | __,__ | __,__ | __,__ |
| **`native`** | __ , __ | **__,__ ± __,__** | **__,__ ± __,__** | **__,__ ± __,__** | **__,__ ± __,__** |
| `native_emb` | __ , 1 | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ |
| `shuffled_edges` | = `native` | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ |
| `random_emb` | = `native` | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ |

*(nguồn: `experiments/level2/val/summary.json`, α/β trong `choice.json`)*

### 5.3. Bảng kết quả trên **test** (chấm một lần)

Đọc lại `choice.json` từ val, không tinh chỉnh gì.

| Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `clip` | __,__ | __,__ | __,__ | __,__ |
| `kg_triple` | __,__ | __,__ | __,__ | __,__ |
| **`native`** | **__,__ ± __,__** | **__,__ ± __,__** | **__,__ ± __,__** | **__,__ ± __,__** |
| `native_emb` | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ |
| `shuffled_edges` | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ |
| `random_emb` | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ | __,__ ± __,__ |

*(nguồn: `experiments/level2/test/summary.json`; từng seed của `native`: R@1 __ / __ / __)*

Nhận xét (điền sau khi có số): ⟨`native` so với `clip`: +/− … R@1; so với `kg_triple`: … (embedding có thêm
giá trị ngoài đếm cạnh không); chênh val → test⟩.

## 6. Ablation và đối chứng

Bốn dòng đối chứng ở mục 5 trả lời yêu cầu ablation của đề ("bỏ cạnh, xáo cạnh, thay đồ thị ngẫu nhiên"):

| Đối chứng | Thao tác | Câu hỏi nó trả lời |
|---|---|---|
| `kg_triple` (β = 0) | bỏ kênh embedding, chỉ đếm cạnh | Đếm cạnh KGC-train một mình giúp bao nhiêu? |
| `native_emb` (β = 1) | bỏ kênh đếm cạnh, chỉ embedding | Embedding NativE một mình giúp bao nhiêu? |
| `shuffled_edges` | xáo đuôi triple trong từng quan hệ (giữ bậc đầu), seed = run seed | Điểm của cạnh là do **đúng cạnh** hay chỉ do thống kê bậc? |
| `random_emb` | thay vector NativE bằng nhiễu Gauss chuẩn hoá, seed = run seed | Điểm của kênh embedding là do **NativE** hay chỉ do có thêm một kênh? |

Kết quả (điền từ mục 5):
- Đếm cạnh (`kg_triple` − `clip`): ⟨… R@1⟩.
- Embedding (`native` − `kg_triple`, và `native_emb` − `clip`): ⟨…⟩.
- `shuffled_edges` so `native`: ⟨nếu tụt gần về `kg_triple`/`clip` → điểm đến từ đúng cạnh, không phải bậc⟩.
- `random_emb` so `native`: ⟨nếu tụt → kênh embedding thật sự dùng tín hiệu NativE⟩.

**Lưu ý kịch bản kết quả âm** (`docs/level2-plan.md` §7): nếu val chọn ra **β = 0** (kênh embedding không giúp),
báo cáo vẫn đầy đủ; đó là một kết luận có bằng chứng (NativE embedding không thêm giá trị ngoài đếm cạnh trên
MKG-W), và bước a vẫn chứng minh nhóm tái lập được NativE.

## 7. Phân tích mở rộng (val)

### 7.1. Độ nhạy theo trọng số kết hợp α

Quét α của **dòng chính `native`** trên val (β giữ ở giá trị đã chọn), xem đường cong R@1.
Nguồn: `experiments/level2/val/alpha_beta_sweep.json`.

| α | 0,0 | 0,2 | 0,4 | 0,5 | 0,6 | 0,8 | 1,0 |
|---|---|---|---|---|---|---|---|
| R@1 val (`native`) | __,__ | __,__ | __,__ | __,__ | __,__ | __,__ | 24,33 |

*(α = 1 là CLIP thuần = 24,33; α = 0 là chỉ dùng đồ thị)*. Nhận xét: ⟨đỉnh ở α ≈ …; đồ thị kéo R@1 lên/không⟩.

### 7.2. Đồ thị giúp ở đâu — tách theo nhóm truy vấn

Tách truy vấn thành nhóm **`linked`** (link được ≥1 thực thể) và **`no_link`** (không link được gì); nhóm định
nghĩa một lần, dùng chung mọi dòng. Tỉ lệ `linked` trên val ≈ 78,9 % (mục 1).
Nguồn: `experiments/level2/val/summary.json` (khoá `groups`).

| Nhóm | Số câu | `clip` R@1 | `native` R@1 | Chênh |
|---|---:|---|---|---|
| `linked` | __ | __,__ | __,__ | __,__ |
| `no_link` | __ | __,__ | __,__ | __,__ |

Kỳ vọng: đồ thị chỉ giúp ở nhóm `linked`; nhóm `no_link` gần như không đổi (kênh đồ thị trung tính).

### 7.3. Bộ 25 truy vấn tự viết — trực tiếp so với gián tiếp

Bộ **25 câu tự viết** (`data/queries/mkgw_queries_hand.json`, Khang chốt 16/11): **10 trực tiếp** (direct) +
**15 gián tiếp** (indirect). Ứng viên là ảnh của **tập test**; chạy bằng `eval_mkgw_level2.py --split test --hand`,
**chỉ để đo**, tái dùng α/β của val. Báo cáo tách riêng hai nhóm (đề yêu cầu):

- **Trực tiếp:** mô tả nhận ra thực thể mà không cần suy luận qua cạnh ("the largest island in the Mediterranean,
  a region with a volcano" → Sicily/Q1460).
- **Gián tiếp:** phải đi qua một quan hệ, mô tả dạng `pattern` (quan hệ, thực thể trung gian) — ví dụ "a rock
  guitarist who was a member of Pink Floyd" có `pattern = [P463 (member of), Q2306 (Pink Floyd)]`.

| Nhóm | Số câu | `clip` R@1 | `kg_triple` R@1 | `native` R@1 |
|---|---:|---|---|---|
| direct | 10 | __,__ | __,__ | __,__ |
| indirect | 15 | __,__ | __,__ | __,__ |

*(nguồn: `experiments/level2/test/summary.json`, khoá `groups` `hand_*`)*. Lưu ý: 5/15 câu gián tiếp có cạnh
vàng **chỉ nằm trong KGC-test** (ghi chú `"edge only in KGC test"` trong file truy vấn), nên **kênh đếm cạnh
không thấy được** — đây là phép thử cố ý cho tính chống rò rỉ và cho vai trò của kênh embedding.

## 8. Giải thích — ví dụ

Giải thích lấy thẳng từ `matches` mà `MkgwGraphScorer.score` trả về (thực thể được link, cạnh KGC-train đã cộng
điểm, hoặc cosine embedding), **không hậu kiểm**. Dưới đây là ≥3 ví dụ (cả trực tiếp lẫn gián tiếp); hạng và
điểm cụ thể điền từ lần chạy eval (`--split test --hand`, seed 0).

**Ví dụ 1 — gián tiếp, giải thích bằng cạnh (triple).**
Truy vấn: *"the main railway station of Bordeaux"*. `EntityLinker` khớp **"Bordeaux"** (Q1479). Ảnh đúng là
**Gare de Bordeaux-Saint-Jean** (Q800564); giữa hai thực thể có cạnh KGC-train **P131** ("located in the
administrative territorial entity"). Giải thích: *ảnh được cộng điểm vì ga này nằm trong Bordeaux (cạnh P131)*.
Hạng CLIP → `native`: ⟨__ → __⟩; đóng góp kênh `triple`: ⟨…⟩.

**Ví dụ 2 — gián tiếp, giải thích bằng cạnh (member of).**
Truy vấn: *"a rock guitarist who was a member of Pink Floyd"*. Link **"Pink Floyd"** (Q2306). Ảnh đúng:
**David Gilmour** (Q178517); cạnh KGC-train **P463** ("member of"). Giải thích: *được cộng điểm vì là thành viên
của Pink Floyd (cạnh P463)*. Hạng CLIP → `native`: ⟨__ → __⟩.

**Ví dụ 3 — trực tiếp, giải thích bằng embedding.**
Truy vấn: *"the largest island in the Mediterranean, a region with a volcano and a long coastline"* → **Sicily**
(Q1460). Nếu câu không link được thực thể trung gian nào có cạnh trực tiếp, điểm đồ thị đến từ **kênh embedding**
(cosine NativE giữa ứng viên và thực thể được nhắc). Giải thích: ⟨thực thể được link và cosine cao nhất⟩;
hạng CLIP → `native`: ⟨__ → __⟩.

**Ví dụ 4 (đối chứng âm) — gián tiếp, cạnh chỉ ở KGC-test.**
Truy vấn: *"the rock band whose lead singer is Hayley Williams"* → **Paramore** (Q473466), `pattern = [P527
(has part), Q201562 (Hayley Williams)]`. Cạnh vàng **chỉ nằm trong KGC-test**, nên kênh đếm cạnh **không thấy**;
đây là ví dụ cho thấy giải thích trung thực (không "mượn" cạnh test) và cho thấy giới hạn của kênh đồ thị khi
cạnh bị giữ ngoài train. Kết quả: ⟨`native` có/không cải thiện so với CLIP⟩.

*(Nhãn quan hệ theo Wikidata; hạng/điểm/`matches` đầy đủ: lần chạy `eval_mkgw_level2.py --split test --hand`,
seed 0.)*

## 9. Hạn chế

- **Khác bộ mã hoá ảnh.** Kênh CLIP dùng CLIP ViT-B/32 của nhóm; kênh embedding dùng NativE, mà bảng visual của
  NativE là đặc trưng 383 chiều **đã trích sẵn** bởi tác giả (không phải ảnh gốc). Hai nguồn đặc trưng khác nhau;
  báo cáo so bằng chỉ số retrieval, không so trực tiếp vector.
- **Kênh embedding có thể không giúp.** Trên dữ liệu `MKG-W-rmask` (đã che modality val/test), embedding NativE
  có thể cho β = 0 trên val; nếu vậy, ghi rõ và giữ phân tích đối chứng.
- **Thiếu seed / thiếu lần chạy.** Nếu đến 19/11 chưa đủ 3 seed `rmask`, báo cáo với số đang có và **ghi rõ là
  thiếu**; không giảm epoch.
- **Không làm GAT trên MKG-W.** MKG-W không có scene graph cho từng ảnh, nên không tái dùng GNN của cấp 1;
  ở bước b embedding NativE đóng vai trò của GNN. Hàng "CLIP + GNN" coi như hoàn thành bằng `clip` và `kg_triple`
  (`docs/level2-plan.md` §4).
- **Không chuyển NativE sang MMKG tiếng Việt.** Chuyển giao (hàng 32, Khang) chỉ dùng kênh đếm cạnh, vì NativE
  phải huấn luyện lại trên đồ thị chỉ ~325 triple — không đủ dữ liệu.
- Mới thực nghiệm trên hai tập công khai (VG∩COCO ở cấp 1, MKG-W ở cấp 2); chênh giữa cấu hình chỉ kiểm với 3 seed.

## 10. Đóng góp

Phần cấp 2 (dữ liệu MKG-W, hàm điểm đồ thị: Đạt; bộ 25 câu, script đánh giá, notebook molab: Khang;
**huấn luyện NativE 4 lần, export embedding, đánh giá val/test, tái lập bước a, phân tích và nháp báo cáo cấp 2:
Trần Đắc Khoa**, cùng công cụ AI). Lộc review, merge và điều phối GPU. Chi tiết theo phần việc và khai báo công
cụ AI/mã nguồn mở: `report/declaration-ai-oss.md`.

## 11. Nguồn số liệu

| Mục | File |
|---|---|
| Dữ liệu, chia tập, nguồn ảnh | `docs/level2-plan.md` §3, `experiments/level2/native-repro/README.md`, `data/MANIFEST.md` |
| Độ phủ kênh đồ thị | `experiments/checks/mkgw_link_coverage.json` |
| CLIP thuần (val) | `experiments/level2/clip_baseline/metrics_val.json` |
| Tái lập bước a | `experiments/level2/native-repro/`, `metrics.json`, bài báo arXiv 2406.17605 |
| Log huấn luyện, thời gian | `experiments/level2/native-train/*.log` |
| Embedding xuất ra | `data/features/mkgw_native_ent_seed{0,1,2}.*`, SHA-256 trong `data/MANIFEST.md` |
| Kết quả val (6 dòng), α/β | `experiments/level2/val/summary.json`, `choice.json`, `alpha_beta_sweep.json` |
| Kết quả test (6 dòng), bộ 25 câu | `experiments/level2/test/summary.json` |
| Bộ 25 truy vấn tự viết | `data/queries/mkgw_queries_hand.json` |
| Hàm điểm, EntityLinker, eval | `src/retrieval/mkgw_graph_score.py`, `scripts/eval_mkgw_level2.py` |
