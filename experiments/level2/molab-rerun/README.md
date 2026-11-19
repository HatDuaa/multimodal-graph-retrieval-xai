# Cấp độ 2 trên MKG-W: lần chạy lại độc lập trên molab (Khang, 16–18/11/2026)

Thư mục này lưu một lần chạy lại toàn bộ cấp độ 2, độc lập với kết quả chính ở `../val/` và `../test/`. Báo cáo dẫn tới nó ở mục kết quả cấp độ 2 (`report/latex/sections/05-level2-results.tex`). **Số chính thức của cấp độ 2 vẫn là số ở `../val/` và `../test/`**; thư mục này chỉ để đối chiếu.

Các file được chép nguyên từ nhánh `khang/mkgw-level2` (commit `5ed30cf` cho `val/`, `test/`, hai log; `native-train/` lấy từ cùng nhánh), không chạy lại và không sửa số. Đường dẫn `/home/marimo/...` trong log là đường dẫn trên molab.

## Khác gì so với lần chạy chính

| | Lần chạy chính (`../val`, `../test`) | Lần chạy này |
|---|---|---|
| Máy | RTX 4090 (NativE), RTX 3070 (đặc trưng CLIP) | molab (`notebooks/mkgw_molab.py`, bước 4–7) |
| NativE `rmask`, 3 seed | checkpoint trên máy 4090 | Khang tự huấn luyện, cùng siêu tham số của tác giả (`native-train/README.md`) |
| Đặc trưng CLIP MKG-W | bản dựng lại 17/11, `mkgw_image.npy` SHA `f4de9e4e…` (`data/MANIFEST.md`) | bản trích trên molab, `mkgw_image.npy` SHA `2a12b583…`, `mkgw_caption.npy` SHA `ea86c984…` |
| Cache Wikidata | lấy lại 17/11, SHA `dd47e44d…` | lấy trên molab, SHA `e52ed48c…`; `wikidata_relations.json` giống hệt |
| Kênh `kgc` của NativE | có | chưa có (kênh này thêm sau) |
| Split, triple KGC, bộ 25 câu soạn tay | giống hệt | giống hệt |

Quy trình giống lần chạy chính: α, β chọn trên val riêng cho từng seed; test chạy một lần, đọc lại `val/choice.json`; bộ câu soạn tay chỉ dùng để chấm. Lệnh: `python scripts/eval_mkgw_level2.py --split val`, rồi `--split test --hand`.

## Kết quả (mean ± std qua 3 seed; std = 0 khi hàng không phụ thuộc seed)

NativE `rmask` (link prediction, filter, `native-train/metrics.json`): MRR **36,89 ± 0,27**, Hit@1 29,93 ± 0,16. Bản MKG-W gốc, seed 42: MRR 37,06 (bài báo 36,58).

Val chọn α = 0, β = 0 ở cả 3 seed, giống lần chạy chính, nên dòng chính trùng `kg_triple`.

| Test, 892 câu | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 25,00 | 44,73 | 54,60 | 34,66 |
| CLIP + đồ thị (`native` = `kg_triple`) | **29,82** | **51,46** | **59,64** | **39,91** |
| CLIP + embedding NativE (`native_emb`, β = 1) | 26,05 ± 0,51 | 45,03 ± 0,28 | 54,33 ± 0,84 | 35,54 ± 0,40 |
| Đối chứng xáo cạnh (`shuffled_edges`) | 21,41 ± 0,11 | 43,35 ± 0,56 | 53,44 ± 0,06 | 31,99 ± 0,13 |

| Val, 892 câu | R@1 | MRR |
|---|---|---|
| CLIP | 24,33 | 34,22 |
| CLIP + đồ thị (`native` = `kg_triple`) | 32,40 | 41,18 |
| `native_emb` (β = 1) | 26,53 ± 0,36 | 35,96 ± 0,26 |
| `shuffled_edges` | 21,82 ± 0,39 | 32,03 ± 0,23 |

Bộ câu soạn tay (pool test): trực tiếp 10 câu, R@1 80,00 ở cả CLIP và CLIP + đồ thị; gián tiếp 15 câu, R@1 40,00 → 80,00.

## So với lần chạy chính

- Dòng chính cho đúng cùng số: R@1 val 32,40, test 29,82.
- CLIP thuần trên val là 24,33 ở cả hai lần. Trên test lần này được 25,00 thay vì 24,89, tức lệch 1 câu, do khác bản đặc trưng CLIP.
- `native_emb` và `shuffled_edges` gần với lần chạy chính, chênh lệch cỡ độ lệch chuẩn giữa các seed.
