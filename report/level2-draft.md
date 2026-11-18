# Nháp báo cáo: Cấp độ 2, MKG-W và tái lập NativE

Nháp để Lộc duyệt rồi ghép vào bản báo cáo chính. Cùng cấu trúc và quy ước với `report/level1-draft.md`:
thuật ngữ tiếng Anh giữ nguyên, kèm giải thích lần đầu; mọi con số lấy từ file kết quả trong `experiments/`,
đường dẫn ghi ở cuối từng mục và ở mục 11. Người soạn nháp: Khoa cùng công cụ AI (Claude), theo phần việc
cấp 2 trong `docs/level2-plan.md` §5.

Bản đầu 2026-11-17 (để chỗ trống chờ chạy). **Bản này cập nhật 2026-11-18: đã điền số thật** từ các lần chạy
ngày 17/11 (train NativE trên máy 4090 của Lộc, đánh giá retrieval trên máy Lộc), đã đóng băng theo mốc 19/11.
Mỗi số đều có log/json kèm seed trong `experiments/level2/`; không có số bịa (CLAUDE.md, đề mục 6).

## Tóm tắt

Bài toán vẫn là truy vấn văn bản → ảnh (nhánh A của đề), trên tập công khai thứ hai **MKG-W** (một
multi-modal knowledge graph: đồ thị tri thức mà mỗi thực thể có kèm mô tả và ảnh). Truy vấn là **mô tả tiếng
Anh của một thực thể đã che tên**; đáp án đúng là ảnh của chính thực thể đó; ứng viên là toàn bộ ảnh của tập
đang đánh giá. CLIP đóng băng lấy top-50, rồi một **điểm đồ thị** chấm lại và xếp hạng lại; mỗi kết quả kèm
danh sách thực thể/cạnh khớp lấy thẳng từ đại lượng dùng để xếp hạng (giải thích).

Phương pháp mới tái lập theo đề (một MMKGC từ 2024) là **NativE (SIGIR 2024)**, dùng theo hai bước: **bước a**
tái lập link prediction gốc trên MKG-W để so với bài báo; **bước b** đưa tín hiệu của NativE vào điểm đồ thị
retrieval thay cho GNN của cấp 1.

Kết quả chính trên **test** (892 truy vấn, pool 892 ảnh, chấm **một lần**, mean ± std qua 3 seed NativE):

| Phương pháp | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần (`clip`) | 24,89 | 44,73 | 54,48 | 34,61 |
| **CLIP + đếm cạnh KG (`kg_triple`, cũng là dòng chính `native`/`native_kgc`)** | **29,82** | **51,46** | **59,42** | **39,89** |

Hai kết luận chính, cả hai đều có bằng chứng:

1. **Đếm cạnh tri thức trực tiếp là thứ giúp ích.** Xếp lại top-50 của CLIP theo **số cạnh KGC-train** giữa
   ứng viên và thực thể được nhắc trong truy vấn nâng R@1 test **24,89 → 29,82** (+4,93), val **24,33 → 32,40**
   (+8,07). Ở nhóm truy vấn liên kết được thực thể, mức tăng lớn hơn (val 20,91 → 31,15). Trên bộ 25 câu tự
   viết, **nhóm gián tiếp tăng R@1 từ 40,0 lên 80,0** (MRR 49,28 → 83,33).
2. **Embedding và điểm KGC của NativE không giúp thêm khi đã có kênh đếm cạnh** — đây là **kết quả âm, có giải
   thích**: khi dò trên val, cả `native` và `native_kgc` đều chọn trọng số 0 cho kênh NativE (β = 0), nên dòng
   chính trùng `kg_triple`. Đứng một mình, điểm NativE có ích hơn CLIP một chút (mục 5–6), nhưng yếu hơn đếm
   cạnh. Việc tái lập ở bước a (dưới đây) vẫn chứng minh nhóm chạy được NativE đúng như bài báo.

*(nguồn: `experiments/level2/test/summary.json`, `val/summary.json`, `val/README.md`)*

## 1. Dữ liệu và cách chia tập

Tập công khai thứ hai là **MKG-W** (MKG-Wikidata), một MMKGC do nhóm MMRNS phát hành; mỗi thực thể là một URI
Wikidata, có nhãn/mô tả/alias và ảnh. Dùng cho hai việc trên cùng một bộ thực thể:

- **Đồ thị tri thức (cho NativE và kênh đếm cạnh):** bản MKG-W của NativE — **15.000 thực thể, 169 quan hệ,
  34.196 / 4.276 / 4.274 triple** cho train / valid / test của bài toán link prediction (KGC). **Chỉ triple
  thuộc phần KGC-train (34.196) mới được đưa vào đồ thị** của kênh đếm cạnh (chống rò rỉ, mục 2).
- **Bài toán retrieval (truy vấn → ảnh):** chỉ xét thực thể vừa có mô tả vừa có ảnh (nguồn MMRNS). Chia tập bản
  v2 (25/09, Đạt), seed 0:

| Split | Thực thể = ảnh = truy vấn |
|---|---:|
| train | 7 136 |
| val | 892 |
| test | 892 |
| **Tổng** | **8 920** |

Truy vấn là **mô tả tiếng Anh của thực thể, đã che tên** (ví dụ "a commune in Pyrénées-Atlantiques, France").
Khi đánh giá một split, **pool ứng viên là toàn bộ ảnh của chính split đó** (892 cho val, 892 cho test), mỗi
thực thể một ảnh (JPEG 640px). **Không dựng lại tập chia**: mọi mô hình dùng chung bộ file split; SHA-256 trong
`data/MANIFEST.md`.

**Nguồn ảnh.** Kho phát hành của nhóm MMRNS (`MKG-W_img.zip`, 6,4 GB + `ent_links` ánh xạ DBpedia ↔ QID):
37.192 ảnh của 8.961 thực thể. **Nhất quán với embedding chính thức của NativE** — bảng `MKG-W-visual.pth`
được trích từ đúng bộ ảnh này.

**Dựng lại đặc trưng cho lần chạy này.** Bản đặc trưng gốc chỉ nằm trong repo Hugging Face riêng của Khang, nên
Lộc dựng lại bằng đúng các script trong repo (`extract_mkgw_images.py`, `extract_features.py`); SHA-256 của ảnh
nguồn và `ent_links` khớp MANIFEST. Đặc trưng CLIP có SHA khác bản gốc do **sai số số học giữa các GPU**, nhưng
kiểm lại trên val **R@1 = 24,33, đúng bằng `clip_baseline`** (R@5/R@10 lệch đúng 1 truy vấn). Cache Wikidata
lấy lại ngày 17/11 (bản tháng 9 của nhóm không có sẵn): 441/15.000 thực thể mất nhãn tiếng Anh (58 đã xoá, 383
chuyển sang mã ngôn ngữ `mul`), nhưng độ phủ liên kết gần như không đổi (704 → 703 truy vấn val link được).

**Độ phủ của kênh đồ thị.** Trên val, **78,81 %** truy vấn liên kết được ≥ 1 thực thể (test 80,94 %); khoảng
**45,6 %** truy vấn val có đáp án nối trực tiếp với một thực thể được nhắc bằng một cạnh KGC-train.

Nguồn: `docs/level2-plan.md` §3, `experiments/level2/native-repro/README.md`, `experiments/level2/val/README.md`,
`data/MANIFEST.md`.

## 2. Chống rò rỉ dữ liệu

Đây là tiêu chí chấm (đề mục 6). Cấp 2 có một nguy cơ rò rỉ **đặc thù của NativE**.

- **File split là nguồn duy nhất**; mọi dòng (mục 5) dùng chung bộ split và cùng danh sách top-50 của CLIP.
- **Kênh đếm cạnh chỉ dùng triple KGC-train** (34.196). Hàm điểm có `assert kgc_split == "train"` khi dựng đồ
  thị, nên cạnh KGC-valid/test không bao giờ lọt vào (`src/retrieval/mkgw_graph_score.py`).
- **Rò rỉ đặc thù của NativE và bản `MKG-W-rmask`.** `MKG-W-textual.pth` mã hoá **chính mô tả thực thể** (tức
  câu truy vấn) và `MKG-W-visual.pth` mã hoá **chính ảnh ứng viên**. Nếu huấn luyện NativE trên cả 15.000 hàng,
  lớp chiếu và embedding cấu trúc sẽ nhìn thấy trực tiếp truy vấn và ứng viên của val/test. Vì vậy, cho **bước
  b**, nhóm tạo bản `MKG-W-rmask` (`scripts/mask_native_modalities.py`): **đặt về 0 hàng visual và textual của
  mọi thực thể val/test retrieval** (đúng tình huống "thiếu modality" mà NativE vốn xử lý). Lần chạy thật che
  **1.784 thực thể** (892 val + 892 test). Bản MKG-W gốc chỉ dùng cho **bước a**. *(Việc rmask vẫn tái lập được
  link prediction — mục 4 — cho thấy việc che modality của ~12 % thực thể không phá mô hình.)*
- **α và β chọn trên val, riêng từng seed**, theo R@1 (hoà thì xét MRR); test đọc lại `choice.json`, không chỉnh
  (`experiments/level2/val/choice.json`).
- **Kênh `kgc` (mục 3.3) được chốt TRƯỚC khi chạy val** (Lộc duyệt), không phải thêm sau khi thấy kết quả.
- **Bộ 25 truy vấn tự viết** (mục 7.3) chỉ để **đo**; tái dùng α/β của val, không tinh chỉnh gì.
- **Test chấm đúng một lần** cho mỗi dòng đã chốt.

## 3. Phương pháp

### 3.1. NativE (SIGIR 2024) là gì

**NativE** (*NativE: Multi-modal Knowledge Graph Completion in the Wild*, SIGIR 2024, arXiv 2406.17605) là một
mô hình **MMKGC** (Multi-Modal Knowledge Graph Completion — dự đoán cạnh còn thiếu bằng cả cấu trúc đồ thị lẫn
ảnh/chữ của thực thể). Hai điểm chính: (1) lõi cấu trúc **AdvRelRotatE** — biến thể RotatE (mỗi quan hệ là một
phép quay trong không gian phức), huấn luyện đối kháng (adversarial, WGAN-GP) để sinh mẫu âm khó; (2) **hợp
nhất modality theo quan hệ (relation-gated)**: đặc trưng ảnh/chữ được chiếu rồi trộn vào embedding thực thể với
trọng số phụ thuộc quan hệ, chịu được thực thể thiếu ảnh hoặc thiếu chữ — chính tính chất nhóm dùng cho rmask.

Mã tái lập: `zjukg/NativE`, commit `c72291c` (2024-08-12). **Sự cố đã biết:** `mmkgc/release/Base.so` kèm sẵn
bị segfault khi đọc `train2id.txt`; sửa bằng rebuild từ source (`cd mmkgc && bash make.sh`, cần g++).

### 3.2. Bước a — tái lập link prediction gốc

Chạy `repro` (MKG-W gốc, seed 42) đúng siêu tham số tác giả (mục 4), so **MRR và Hits@1/3/10** với bài báo trên
tập test KGC của chính NativE (filtered, không ràng buộc kiểu). Tiêu chí: **chênh ≤ ~1 điểm là tái lập được.**

| Chỉ số | Bài báo NativE (MKG-W, S+I+T) | Tái lập `repro` (seed 42) | Chênh |
|---|---|---|---|
| MRR | 36,58 | **36,99** | +0,41 |
| Hits@1 | 29,56 | **30,17** | +0,61 |
| Hits@3 | (bài báo không ghi) | 39,95 | — |
| Hits@10 | (bài báo không ghi) | 49,32 | — |

**Kết luận: tái lập thành công** — chênh ≤ ~0,6 điểm trên hai chỉ số bài báo có công bố, thậm chí nhỉnh hơn một
chút. *(Bài báo chỉ công bố MRR và Hits@1 cho MKG-W ở Bảng 2; MR và Hits@3/10 để trống.)*

*(nguồn: `experiments/level2/native-repro/metrics.json`, log `native-train/repro_seed42_e1000.log`)*

### 3.3. Bước b — điểm đồ thị cho retrieval

Truy vấn là mô tả đã che tên nhưng vẫn thường nhắc tên thực thể khác ("a rock guitarist who was a member of
*Pink Floyd*"). Kênh đồ thị khai thác điều đó:

1. **`EntityLinker`** khớp mọi label/alias của thực thể KG xuất hiện trong câu, **ưu tiên khớp dài nhất**
   (n-gram tối đa 5 token).
2. Với mỗi ảnh ứng viên (= một thực thể), tính các **kênh điểm** với thực thể được link:
   - **`triple`**: **số cạnh KGC-train trực tiếp** giữa ứng viên và thực thể được nhắc (không huấn luyện).
   - **`emb`** (cosine): cosine lớn nhất giữa embedding cấu trúc NativE của ứng viên và của thực thể được nhắc.
   - **`kgc`** (điểm bộ ba của chính NativE, thêm 17/11, Lộc duyệt, chốt trước khi chạy val): giá trị lớn nhất
     của `AdvRelRotatE.forward` (checkpoint rmask cùng seed) qua mọi thực thể được link, 169 quan hệ, cả hai
     chiều `(c,r,e)` và `(e,r,c)`. Khác `emb` ở chỗ **có dùng quan hệ**. Code: `src/retrieval/native_kgc.py`,
     tính lại đúng `forward()` của NativE c72291c, có test đối chiếu (`tests/test_native_kgc.py`).
3. **Các match dùng để tính điểm được trả về cùng điểm — chính là giải thích** (đề: giải thích sinh từ cơ chế
   xếp hạng, không hậu kiểm).

**Công thức trộn** (z-score trong top-50 của từng truy vấn), đúng khung cấp 1:

```
score = α · z(CLIP) + (1 − α) · z( β · z(kênh NativE) + (1 − β) · z(triple) )
```

với "kênh NativE" là `emb` (dòng `native`) hoặc `kgc` (dòng `native_kgc`). Lưới: **α ∈ {0; 0,05; …; 1}**,
**β ∈ {0; 0,25; 0,5; 0,75; 1}**; chọn trên **val**, riêng từng seed, theo R@1 (hoà thì MRR).

Nguồn code: `src/retrieval/mkgw_graph_score.py`, `src/retrieval/native_kgc.py`, `scripts/eval_mkgw_level2.py`.

## 4. Huấn luyện NativE

Chạy ngày 17/11 trên **RTX 4090 24 GB** của Lộc (phương án dự phòng trong `docs/level2-plan.md` §5 khi molab
chưa sẵn lúc 12:00). Siêu tham số giữ **đúng `scripts/run_mkgw.sh` của tác giả**, giữa các lần chỉ đổi dữ liệu
và seed: **1.000 epoch**, dim 250, margin 4, 128 mẫu âm, lr 1e-4, batch 1024; mã commit `c72291c`.

| Lần chạy | Dữ liệu | Seed | Giây/epoch | Thời gian | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---:|---:|---|---|---|---|
| `rmask` | MKG-W-rmask | 0 | 3,93 | 1,10 h | 0,3714 | 0,3009 | 0,4041 | 0,4996 |
| `rmask` | MKG-W-rmask | 1 | 3,95 | 1,10 h | 0,3713 | 0,3000 | 0,4057 | 0,5011 |
| `rmask` | MKG-W-rmask | 2 | 3,97 | 1,11 h | 0,3671 | 0,2987 | 0,3993 | 0,4911 |
| **rmask, mean ± std** | | 0/1/2 | | | **0,3699 ± 0,0024** | **0,2998 ± 0,0011** | **0,4030 ± 0,0033** | **0,4973 ± 0,0054** |
| `repro` | MKG-W gốc | 42 | 3,95 | 1,10 h | 0,3699 | 0,3017 | 0,3995 | 0,4932 |

Bốn lần nối tiếp trong một phiên `tmux`, mỗi lúc một job GPU, tổng **4 giờ 25 phút**. Cột link prediction là
kết quả trên tập test KGC của NativE (filtered), **không đụng tập test retrieval**. Embedding cấu trúc xuất
bằng `scripts/export_native_embeddings.py --what ent` ra `data/features/mkgw_native_ent_seed{0,1,2}.npy`
(15.000 × 500, chuẩn hoá L2); SHA-256 trong `data/MANIFEST.md`, không commit file `.npy`.

**Lưu ý về seed (quan trọng khi đọc std).** `-seed` chỉ đặt `torch.manual_seed` (khởi tạo tham số + nhiễu GAN);
bộ lấy mẫu âm viết bằng C không gọi `srand` nên **chuỗi mẫu âm giống hệt ở mọi seed**. Do đó độ lệch giữa 3
seed rmask chỉ đo ảnh hưởng của khởi tạo và nhiễu GAN, **không đo ảnh hưởng của cách lấy mẫu âm** — std thật
của mô hình có thể lớn hơn con số báo cáo. Ghi rõ để không đọc nhầm.

*(nguồn: `experiments/level2/native-train/README.md`, `metrics_rmask.json`, `driver.log`, các `*.log`)*

## 5. Kết quả retrieval

R@k là tỉ lệ truy vấn có ảnh đúng trong k kết quả đầu; MRR là trung bình nghịch đảo hạng (trong top-50). Số
dạng **mean ± std qua 3 seed NativE** (std = 0 với các dòng không phụ thuộc embedding).

**Các dòng** (α, β chọn trên val, `choice.json`):

| Dòng | Ý nghĩa | α, β (val) |
|---|---|---|
| `clip` | CLIP thuần | — |
| `kg_triple` | CLIP + đếm cạnh (β = 0), không huấn luyện | 0 , 0 |
| `native` | CLIP + emb(cosine) + đếm cạnh | 0 , 0 |
| `native_kgc` | **Dòng chính cấp 2**: CLIP + kgc(điểm NativE) + đếm cạnh | 0 , 0 |
| `native_emb` | Chỉ emb cosine (β = 1) — bỏ cạnh | 0,6–0,7 , 1 |
| `kgc_only` | Chỉ điểm kgc (β = 1) — bỏ cạnh | 0,45–0,5 , 1 |
| `shuffled_edges` | Xáo đuôi cạnh trong từng quan hệ (giữ bậc đầu) | = `native` |
| `random_emb` | Thay embedding NativE bằng nhiễu Gauss | = `native` |
| `random_kgc` | Thay embedding cấu trúc (trong kgc) bằng nhiễu Gauss | = `native_kgc` |

Vì cả `native` và `native_kgc` đều chọn **β = 0** ở cả 3 seed, kênh NativE có trọng số 0, nên **năm dòng
`kg_triple` = `native` = `native_kgc` = `random_emb` = `random_kgc` trùng nhau từng số**. Đây là phát hiện,
không phải lỗi (mục 6).

### 5.1. Trên **val** (892 truy vấn, chọn α/β)

| Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `clip` | 24,33 | 44,39 | 53,81 | 34,22 |
| **`kg_triple` = `native` = `native_kgc`** | **32,40** | **50,78** | **57,62** | **41,18** |
| `native_emb` | 26,64 ± 0,26 | 45,96 ± 0,39 | 54,30 ± 0,62 | 35,98 ± 0,15 |
| `kgc_only` | 28,25 ± 0,11 | 45,85 ± 0,51 | 54,37 ± 0,70 | 36,97 ± 0,08 |
| `shuffled_edges` | 21,82 ± 0,39 | 42,56 ± 0,06 | 51,72 ± 0,26 | 32,03 ± 0,23 |
| `random_emb` = `random_kgc` | 32,40 | 50,78 | 57,62 | 41,18 |

### 5.2. Trên **test** (892 truy vấn, chấm một lần)

| Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `clip` | 24,89 | 44,73 | 54,48 | 34,61 |
| **`kg_triple` = `native` = `native_kgc`** | **29,82** | **51,46** | **59,42** | **39,89** |
| `native_emb` | 26,08 ± 0,45 | 44,92 ± 0,53 | 54,22 ± 0,95 | 35,46 ± 0,58 |
| `kgc_only` | 27,24 ± 0,19 | 46,52 ± 0,11 | 56,28 ± 0,11 | 36,67 ± 0,14 |
| `shuffled_edges` | 21,45 ± 0,06 | 43,31 ± 0,51 | 53,29 ± 0,13 | 32,02 ± 0,16 |

Dòng chính hơn CLIP thuần **+4,93 R@1** và **+5,28 MRR** trên test; chênh val → test −2,58 R@1 (32,40 → 29,82),
hợp lý vì α/β chọn trên val.

*(nguồn: `experiments/level2/{val,test}/summary.json`, `val/choice.json`)*

## 6. Ablation và đối chứng

Các dòng đối chứng trả lời yêu cầu ablation của đề ("bỏ cạnh, xáo cạnh, thay đồ thị ngẫu nhiên"):

- **Đếm cạnh một mình đã đủ mạnh** (`kg_triple`): +8,07 R@1 val, +4,93 R@1 test so với CLIP. Đây là nguồn lợi
  chính.
- **Embedding/điểm NativE đứng riêng: có ích hơn CLIP nhưng yếu hơn đếm cạnh.**
  - `native_emb` (cosine): val R@1 26,64 (+2,3 so CLIP).
  - `kgc_only` (điểm bộ ba NativE, có dùng quan hệ): val R@1 28,25 (+3,9 so CLIP), **tốt hơn cosine** — dùng
    đúng hàm chấm có quan hệ của mô hình KGC có ích hơn. Nhưng vẫn kém `kg_triple` (32,40).
- **Khi đã có kênh đếm cạnh, NativE không thêm giá trị:** val chọn β = 0 cho cả `native` và `native_kgc` ở mọi
  seed; mọi β > 0 đều làm R@1 giảm (ví dụ α = 0, β = 0,25 → chỉ còn 24,0). Cách hiểu: với ~45,6 % truy vấn có
  cạnh train trực tiếp, đếm cạnh đã chứa đúng thông tin; phần còn lại điểm NativE yếu hơn CLIP.
- **Xáo cạnh (`shuffled_edges`): tụt dưới cả CLIP** (val 21,82 < 24,33). Đổi chỗ đuôi triple trong từng quan hệ
  (giữ bậc) phá hết thông tin cạnh thật → chứng tỏ phần tăng đến từ **đúng cạnh**, không phải thống kê bậc.
- **Nhiễu Gauss (`random_emb`, `random_kgc`) = dòng chính:** vì β = 0 nên kênh ngẫu nhiên có trọng số 0; trùng
  số với `kg_triple`, xác nhận kênh NativE không đóng góp trong dòng chính.

**Kết quả âm có giá trị** (`docs/level2-plan.md` §7): việc val chọn β = 0 là một kết luận có bằng chứng (embedding
NativE không thêm giá trị ngoài đếm cạnh trên MKG-W), và bước a vẫn chứng minh nhóm tái lập được NativE.

## 7. Phân tích mở rộng (val)

### 7.1. Độ nhạy theo trọng số kết hợp α (dòng chính, β = 0)

| α | 0–0,35 | 0,4 | 0,5 | 0,6 | 0,8 | 1,0 |
|---|---|---|---|---|---|---|
| R@1 val | **32,40** | 32,06 | 31,61 | 30,61 | 28,36 | 24,33 |

- α = 1 là CLIP thuần (24,33); α = 0 là chỉ dùng đồ thị. Đường cong **đạt đỉnh ở cả đoạn phẳng α ∈ [0; 0,35]**
  rồi giảm dần — đây là một vùng phẳng, không phải điểm nhọn, nên chọn α = 0 ổn định.
- Điều đáng chú ý: ở đây **đồ thị (đếm cạnh) tốt hơn CLIP để xếp top-50** — ngược với cấp 1, nơi đỉnh ở α ≈ 0,6.
  Lý do: trên MKG-W, truy vấn là mô tả đã che tên (CLIP yếu hơn), còn số cạnh train là tín hiệu rất mạnh cho
  nhóm truy vấn liên kết được.

*(nguồn: `experiments/level2/val/alpha_beta_sweep.json`, `alpha_beta_sweep_kgc.json`)*

### 7.2. Đồ thị giúp ở đâu — tách theo nhóm truy vấn

Nhóm `linked` (liên kết được ≥ 1 thực thể) và `no_link`, định nghĩa một lần, dùng chung mọi dòng. R@1 val:

| Nhóm | Số câu | `clip` | `kg_triple`/`native` | `native_emb` | `kgc_only` | `shuffled_edges` |
|---|---:|---|---|---|---|---|
| `linked` | 703 | 20,91 | **31,15** | 23,85 | 25,89 | 17,73 |
| `no_link` | 189 | 37,04 | 37,04 | 37,04 | 37,04 | 37,04 |

- **Toàn bộ phần tăng nằm ở nhóm `linked`**: +10,24 R@1 (20,91 → 31,15). Nhóm `no_link` giữ nguyên như CLIP vì
  kênh đồ thị bằng 0 → thứ hạng không đổi (phép sắp xếp ổn định).
- Nhóm `no_link` lại có R@1 cao hơn `linked` (37,04 so 20,91): các truy vấn không nhắc thực thể nào thường ngắn,
  trực quan, CLIP vốn làm tốt. Trên test cũng vậy (`linked` 22,71 → `native` 28,81; `no_link` 34,12 không đổi).

### 7.3. Bộ 25 truy vấn tự viết — trực tiếp so với gián tiếp

Bộ **25 câu** (`data/queries/mkgw_queries_hand.json`, Khang chốt 16/11): **10 trực tiếp** + **15 gián tiếp**.
Ứng viên là ảnh tập test; chạy `--split test --hand`, **chỉ để đo**, tái dùng α/β của val. R@1 / R@5 / R@10 / MRR:

| Nhóm | Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| direct (10) | `clip` | 80,00 | 90,00 | 90,00 | 85,00 |
| direct (10) | `kg_triple`/`native`/`native_kgc` | 80,00 | 90,00 | 90,00 | 85,00 |
| indirect (15) | `clip` | 40,00 | 60,00 | 60,00 | 49,28 |
| indirect (15) | **`kg_triple`/`native`/`native_kgc`** | **80,00** | **86,67** | **86,67** | **83,33** |

- **Đây là bằng chứng rõ nhất cho giá trị của kênh đồ thị:** với câu **gián tiếp** (phải đi qua một quan hệ),
  đếm cạnh nâng R@1 **40,00 → 80,00** (6/15 → 12/15) và MRR **49,28 → 83,33**.
- Câu **trực tiếp** không đổi (CLIP đã đạt 8/10; mô tả trực tiếp đủ cho CLIP).
- Lưu ý: 5/15 câu gián tiếp có cạnh vàng **chỉ nằm trong KGC-test** (ghi chú trong file truy vấn), nên kênh đếm
  cạnh **không thấy được**; dù vậy nhóm gián tiếp vẫn đạt 12/15 — phần còn lại nhờ các cạnh train khác và CLIP.

*(nguồn: `experiments/level2/test/summary.json`, khoá `groups` `hand_*`)*

## 8. Giải thích — ví dụ

Giải thích lấy thẳng từ `matches` mà hàm điểm trả về (thực thể được link + cạnh KGC-train đã cộng điểm), **không
hậu kiểm**. Vì dòng chính chọn α = 0, β = 0, điểm xếp hạng **chính là số cạnh train** giữa ứng viên và thực thể
được nhắc — nên giải thích rất trực tiếp. Dưới đây là các ví dụ (hạng cụ thể của từng câu sinh lại được từ
`eval_mkgw_level2.py --split test --hand`, seed 0; số theo nhóm ở mục 7.3).

**Ví dụ 1 — gián tiếp, cạnh "located in".** *"the main railway station of Bordeaux"*. `EntityLinker` khớp
**Bordeaux** (Q1479); ảnh đúng **Gare de Bordeaux-Saint-Jean** (Q800564) có cạnh KGC-train **P131** ("located
in the administrative territorial entity") tới Bordeaux. Giải thích: *được cộng điểm vì ga nằm trong Bordeaux*.

**Ví dụ 2 — gián tiếp, cạnh "member of".** *"a rock guitarist who was a member of Pink Floyd"*. Khớp **Pink
Floyd** (Q2306); ảnh đúng **David Gilmour** (Q178517) có cạnh **P463** ("member of"). Hai ví dụ này thuộc nhóm
gián tiếp mà CLIP xếp sai còn kênh cạnh kéo lên hạng 1 (mục 7.3: nhóm gián tiếp 40 → 80).

**Ví dụ 3 — trực tiếp.** *"the largest island in the Mediterranean, a region with a volcano and a long
coastline"* → **Sicily** (Q1460). CLIP đã đủ (nhóm trực tiếp 8/10), kênh cạnh không làm hỏng (giữ 80).

**Ví dụ 4 (đối chứng âm) — cạnh chỉ ở KGC-test.** *"the rock band whose lead singer is Hayley Williams"* →
**Paramore** (Q473466), cạnh vàng **chỉ nằm trong KGC-test** nên kênh đếm cạnh **không thấy** (giải thích trung
thực, không "mượn" cạnh test). Đây là một trong vài câu gián tiếp mà kênh cạnh không giúp được.

*(Nhãn quan hệ theo Wikidata. `matches`/hạng đầy đủ sinh lại từ eval script.)*

## 9. Hạn chế

- **Embedding NativE không giúp trên MKG-W (kết quả âm).** Val chọn β = 0 cho dòng chính; đóng góp của phương
  pháp mới ở bước b là 0 khi đã có đếm cạnh. Giá trị của NativE trong báo cáo nằm ở **bước a** (tái lập) và ở
  phân tích đứng-riêng (`kgc_only` > `native_emb` > CLIP, đều < `kg_triple`).
- **Std 3 seed chưa đo biến thiên do lấy mẫu âm** (bộ sampler C không được seed) — std thật có thể lớn hơn.
- **Khác bộ mã hoá ảnh:** kênh CLIP dùng ViT-B/32 của nhóm; NativE dùng đặc trưng visual 383 chiều đã trích sẵn
  của tác giả. So bằng chỉ số retrieval, không so trực tiếp vector.
- **Cache Wikidata lấy lại 17/11** (441 thực thể mất nhãn `en`); độ phủ liên kết gần như không đổi nhưng không
  trùng byte với bản gốc tháng 9.
- **Không làm GAT trên MKG-W** (không có scene graph từng ảnh); hàng "CLIP + GNN" coi như hoàn thành bằng `clip`
  và `kg_triple` (`docs/level2-plan.md` §4).
- **Không chuyển NativE sang MMKG tiếng Việt** (chỉ ~325 triple, không đủ huấn luyện lại); chuyển giao chỉ dùng
  kênh đếm cạnh (hàng 32, Khang, báo cáo ở `report/level3-vnmmkg-draft.md`).

## 10. Đóng góp

Phần cấp 2: dữ liệu MKG-W và hàm điểm đồ thị (**Đạt**); bộ 25 câu, script đánh giá, notebook molab và kênh
`kgc` (**Khang**); huấn luyện NativE 4 lần trên 4090, export embedding, chạy đánh giá val/test và tái lập bước a
(**Lộc**, với công cụ AI — khai báo trong `report/declaration-ai-oss.md`); **nháp báo cáo cấp 2 và phân tích kết
quả (Trần Đắc Khoa**, với công cụ AI). Lộc review, merge và điều phối GPU.

## 11. Nguồn số liệu

| Mục | File |
|---|---|
| Dữ liệu, chia tập, dựng lại đặc trưng | `experiments/level2/val/README.md`, `docs/level2-plan.md` §3, `data/MANIFEST.md` |
| CLIP thuần (val) | `experiments/level2/clip_baseline/metrics_val.json` |
| Tái lập bước a, so bài báo | `experiments/level2/native-repro/metrics.json`, bài báo arXiv 2406.17605 (Bảng 2) |
| Log huấn luyện, thời gian, link prediction | `experiments/level2/native-train/README.md`, `metrics_rmask.json`, `*.log` |
| Kết quả val/test (9 dòng), nhóm, bộ 25 câu | `experiments/level2/{val,test}/summary.json` |
| α/β chọn, quét độ nhạy | `experiments/level2/val/choice.json`, `alpha_beta_sweep.json`, `alpha_beta_sweep_kgc.json` |
| Bộ 25 truy vấn tự viết | `data/queries/mkgw_queries_hand.json` |
| Hàm điểm, kênh kgc, eval | `src/retrieval/mkgw_graph_score.py`, `src/retrieval/native_kgc.py`, `scripts/eval_mkgw_level2.py` |
