# Cấp độ 2 — đánh giá trên tập val (MKG-W, 3 seed NativE)

Chạy ngày 2026-11-17 trên máy Lộc (RTX 3070, CPU cho phần xếp hạng lại), branch `feat/mkgw-level2-eval` (code của PR #39).

```bash
python scripts/eval_mkgw_level2.py --split val      # log: logs/run.log
```

Chưa đọc tập test. α, β của từng seed chọn trên val theo R@1, hoà thì theo MRR; kết quả chọn nằm ở `choice.json`, dùng lại nguyên vẹn khi chạy test.

## Đầu vào

| File | Bytes | SHA-256 | Nguồn |
|---|---|---|---|
| `data/splits/mkgw_val.json` | 251533 | `4762f9c0…` | đã commit, khớp MANIFEST |
| `data/features/mkgw_image.npy` | 18268288 | `f4de9e4ee51a6e2557e3fd1c92fcea0c5f62cd307794075e4aeb718db60b7a90` | tính lại, xem dưới |
| `data/features/mkgw_image.ids.json` | 55893 | `1c893c30618da232ab69562f0e353d660809b49cc690f4b7eb1e7c56e26e4095` | tính lại |
| `data/features/mkgw_caption.npy` | 18268288 | `ea8679cbd5402e68abeadc15164fab020513e6507a57458c3d9cee6c279e2667` | tính lại |
| `data/features/mkgw_caption.ids.json` | 120869 | `8cc753a267a938a07610fba144e148e09067439cc1d5a58ffb90c83276c9d722` | tính lại |
| `data/features/mkgw_native_ent_seed0.npy` | 30000128 | `2e64d98d442a041f6a7a64dff8b4dbd7d1e40800410bb77026841e0fdd0155b5` | NativE `MKG-W-rmask`, seed 0, 1000 epoch (4090) |
| `data/features/mkgw_native_ent_seed1.npy` | 30000128 | `46b22cac34db88173a0ae313b56ba1d5736189a0eb3385a6e3ef091bdb5a093a` | như trên, seed 1 |
| `data/features/mkgw_native_ent_seed2.npy` | 30000128 | `5f8730e609c6d64f9bcc0fae4a408da1614692e440c41393e117dcb65013295a` | như trên, seed 2 |
| `data/processed/mkgw_triples.jsonl` | 2262289 | `40ccfa72ce54d9d2c6a23e9ed50ea571255425e20cf1b0cf6c2c5e2e3cecbedb` | khớp MANIFEST |
| `data/raw/mkgw/wikidata_entities.jsonl` | 3532544 | `dd47e44dbb068c0f209e89bee562efd8c07f7d932aefd6706196585f6718425c` | lấy lại từ Wikidata ngày 17/11, xem dưới |
| `data/raw/mkgw/wikidata_relations.json` | 9452 | `f0760499881f04319a32ea936f6e3f8fe65d7860857d93000716fd96755344f8` | lấy lại từ Wikidata ngày 17/11 |

**Vì sao tính lại đặc trưng CLIP và cache Wikidata.** Bản gốc (MANIFEST, mục MKG-W) chỉ nằm trong repo Hugging Face riêng của Khang, chưa lên Drive nhóm. Nên dựng lại bằng đúng các script trong repo:

- `MKG-W_img.zip` tải từ Drive của MMRNS, SHA khớp MANIFEST (`0178f7bb…`).
- `ent_links.tsv` cũng khớp MANIFEST (`4aa7e9f5…`).
- `scripts/extract_mkgw_images.py` cho 8 961 thực thể có ảnh (340 folder không khớp, như README). `author_images_index.json` khớp MANIFEST (`0727dafe…`) sau khi đổi CRLF → LF, tức là chọn đúng các ảnh như bản gốc.
- `scripts/extract_features.py --dataset mkgw` trên RTX 3070.
  - SHA của `.npy` khác bản gốc do sai số số học giữa các GPU.
  - Kiểm tra lại CLIP trên val: R@1 = 24,33, đúng bằng `experiments/level2/clip_baseline/metrics_val.json`.
  - R@5 44,39 so với 44,51 và R@10 53,81 so với 53,70: mỗi chỉ số lệch 1 truy vấn.
- Cache Wikidata lấy lại bằng `scripts/fetch_mkgw_wikidata.py` và `scripts/fetch_mkgw_relations.py` (worker trên 4090).
  - Nhỏ hơn bản gốc 32 147 byte vì Wikidata đã sửa từ tháng 9.
  - 441/15 000 thực thể không còn nhãn tiếng Anh: 58 mục đã xoá; 383 mục chuyển nhãn sang mã ngôn ngữ `mul`.
  - Độ phủ liên kết thực thể trên val gần như không đổi so với `experiments/checks/mkgw_link_coverage.json`:

    | | Bản gốc | Bản lấy lại |
    |---|---|---|
    | Truy vấn có ≥ 1 thực thể liên kết | 704 | 703 |
    | Đáp án có cạnh train trực tiếp | 403 | 407 |

Lần chạy test dùng đúng các file trên, không đổi sang bản gốc nếu bản gốc tới sau, để α, β chọn trên val khớp với đặc trưng dùng ở test.

## Kết quả (892 truy vấn, mean ± std qua 3 seed NativE)

| Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `clip` (CLIP đóng băng) | 24,33 ± 0,00 | 44,39 ± 0,00 | 53,81 ± 0,00 | 34,22 ± 0,00 |
| `kg_triple` (CLIP + đếm cạnh train) | 32,40 ± 0,00 | 50,78 ± 0,00 | 57,62 ± 0,00 | 41,18 ± 0,00 |
| `native` (CLIP + emb + triple, dòng chính) | 32,40 ± 0,00 | 50,78 ± 0,00 | 57,62 ± 0,00 | 41,18 ± 0,00 |
| `native_emb` (CLIP + emb NativE, bỏ cạnh) | 26,64 ± 0,26 | 45,96 ± 0,39 | 54,30 ± 0,62 | 35,98 ± 0,15 |
| `shuffled_edges` (cạnh xáo trộn) | 21,82 ± 0,39 | 42,56 ± 0,06 | 51,72 ± 0,26 | 32,03 ± 0,23 |
| `random_emb` (emb ngẫu nhiên) | 32,40 ± 0,00 | 50,78 ± 0,00 | 57,62 ± 0,00 | 41,18 ± 0,00 |

78,81 % truy vấn val liên kết được ít nhất một thực thể. R@1 theo nhóm:

| Nhóm | `clip` | `kg_triple` / `native` | `native_emb` | `shuffled_edges` |
|---|---|---|---|---|
| Có liên kết | 20,91 | 31,15 | 23,85 | 17,73 |
| Không liên kết | 37,04 | 37,04 | 37,04 | 37,04 |

Ở nhóm không liên kết, kênh đồ thị bằng 0 nên thứ hạng giữ nguyên như CLIP.

## Nhận xét

- **α, β chọn được.** Cả 3 seed đều chọn α = 0, β = 0 cho `native`.
  - Với α = 0, điểm cuối chỉ còn số cạnh train giữa ứng viên và thực thể trong truy vấn. Các ứng viên cùng số cạnh giữ thứ tự CLIP, vì phép sắp xếp ổn định và danh sách vào đã theo CLIP.
  - Với β = 0 thì `native` trùng `kg_triple`, và `random_emb` cũng trùng vì kênh emb có trọng số 0.
  - R@1 bằng nhau trên cả đoạn α ∈ [0; 0,35] (β = 0) theo `alpha_beta_sweep.json`. Vậy đây là một vùng phẳng, không phải một điểm nhọn.
- **Embedding NativE chỉ có tác dụng khi đứng một mình.**
  - `native_emb` (cosine giữa embedding thực thể NativE của ứng viên và của thực thể trong truy vấn) tăng R@1 2,3 điểm so với CLIP, ổn định qua 3 seed.
  - Khi đã có kênh đếm cạnh, val không chọn trọng số dương cho kênh này ở bất kỳ β nào.
- **Phần tăng là do cạnh thật.** Xáo đuôi cạnh trong từng quan hệ làm R@1 tụt xuống 21,82, dưới cả CLIP.

## Thêm kênh `kgc`: điểm bộ ba của chính NativE (17/11, sau lần chạy trên)

Kênh cosine ở trên không dùng tới quan hệ. Vì vậy nhóm (Lộc duyệt) thêm kênh `kgc`, chốt **trước** khi chạy val:

- **Định nghĩa.** kgc(c, L) là giá trị lớn nhất của điểm `AdvRelRotatE.forward` (checkpoint `rmask` cùng seed), lấy max qua:
  - mọi thực thể liên kết e ∈ L, với e ≠ ứng viên c;
  - 169 quan hệ;
  - cả hai chiều, (c, r, e) và (e, r, c).
- **Code.** `src/retrieval/native_kgc.py` tính lại đúng hàm `forward()` của NativE c72291c:
  - gồm embedding cấu trúc, ảnh và văn bản, ghép bằng attention có nhiệt độ theo quan hệ;
  - các bảng ảnh/văn bản là bản `rmask` lưu trong checkpoint;
  - test `tests/test_native_kgc.py` so với `forward()` gốc trên checkpoint thật.
- **Điền giá trị.** Truy vấn không liên kết được thực thể nào: 0 cho mọi ứng viên. Ứng viên chỉ liên kết với chính nó: lấy trung bình các giá trị của truy vấn.
- **Ba dòng mới:**
  - `native_kgc` = CLIP + kgc + đếm cạnh, là **dòng chính cấp 2**, chốt trước khi xem val;
  - `kgc_only` (β = 1), bỏ kênh đếm cạnh;
  - `random_kgc`: dùng α, β của `native_kgc`, nhưng embedding cấu trúc thay bằng nhiễu Gauss, giữ nguyên chuẩn của từng dòng.

α, β chọn trên val bằng sweep riêng (`alpha_beta_sweep_kgc.json`), cùng quy tắc: R@1 trước, hoà thì MRR.

Chạy lại `python scripts/eval_mkgw_level2.py --split val` trên máy Lộc (RTX 3070), commit `0a77935`. Kiểm tra sau khi chạy:
- 6 file `metrics_*.json` cũ và `alpha_beta_sweep.json` giống từng byte bản trước;
- các khoá cũ trong `choice.json` không đổi.

| Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `native_kgc` (dòng chính) | 32,40 ± 0,00 | 50,78 ± 0,00 | 57,62 ± 0,00 | 41,18 ± 0,00 |
| `kgc_only` | 28,25 ± 0,11 | 45,85 ± 0,51 | 54,37 ± 0,70 | 36,97 ± 0,08 |
| `random_kgc` | 32,40 ± 0,00 | 50,78 ± 0,00 | 57,62 ± 0,00 | 41,18 ± 0,00 |

R@1 nhóm có liên kết: `kgc_only` 25,89 (CLIP 20,91, `native_emb` 23,85, `kg_triple` 31,15).

**Nhận xét:**
- **Điểm của NativE tốt hơn cosine.** Đứng một mình, `kgc_only` tăng R@1 3,9 điểm so với CLIP, còn `native_emb` (cosine) chỉ tăng 2,3. Kết quả ổn định qua 3 seed. Vậy dùng đúng hàm chấm điểm có quan hệ của mô hình KGC thì có ích hơn.
- **Val vẫn chọn β = 0** cho `native_kgc` ở cả 3 seed, nên dòng chính trùng `kg_triple`, và `random_kgc` trùng theo.
- **Mọi β > 0 đều làm R@1 giảm.**
  - Tốt nhất ở β = 0,25 là khoảng 30,0–30,4.
  - Khi α = 0 và β = 0,25, R@1 chỉ còn 24,0, tức là để kgc thay CLIP làm tiêu chí phá hoà giữa các ứng viên cùng số cạnh thì tệ hơn hẳn.
  - Cách hiểu: khi đáp án có cạnh train trực tiếp tới thực thể trong truy vấn (45,6 % truy vấn val), đếm cạnh đã chứa đúng thông tin đó. Với phần còn lại, điểm NativE yếu hơn CLIP.
- Lần chạy test dùng nguyên `choice.json` này, không chỉnh thêm.
