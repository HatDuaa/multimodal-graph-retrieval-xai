# Cấp độ 2 — kết quả trên tập test (MKG-W, 3 seed NativE), chạy một lần duy nhất

Chạy ngày 2026-11-17 trên máy Lộc (RTX 3070), sau khi đã commit `choice.json` của val (commit `ddc68bc`, xem `../val/README.md`):

```bash
python scripts/eval_mkgw_level2.py --split test --hand      # log: logs/run.log
```

- α, β đọc nguyên từ `../val/choice.json`; ở đây không chỉnh gì.
- Đầu vào giống hệt lần chạy val: cùng đặc trưng CLIP tính lại, cùng cache Wikidata lấy lại, cùng 3 file embedding và 3 checkpoint `rmask`. Mã SHA xem `../val/README.md` và `data/MANIFEST.md`.
- Đây là lần **duy nhất** đọc tập test. Không chạy lại để chọn phương pháp.

## Kết quả (892 truy vấn, mean ± std qua 3 seed NativE)

| Dòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `clip` (CLIP đóng băng) | 24,89 ± 0,00 | 44,73 ± 0,00 | 54,48 ± 0,00 | 34,61 ± 0,00 |
| **`native_kgc` (dòng chính: CLIP + điểm NativE + đếm cạnh)** | **29,82 ± 0,00** | **51,46 ± 0,00** | **59,42 ± 0,00** | **39,89 ± 0,00** |
| `kg_triple` (CLIP + đếm cạnh train) | 29,82 ± 0,00 | 51,46 ± 0,00 | 59,42 ± 0,00 | 39,89 ± 0,00 |
| `kgc_only` (CLIP + điểm NativE, bỏ cạnh) | 27,24 ± 0,19 | 46,52 ± 0,11 | 56,28 ± 0,11 | 36,67 ± 0,14 |
| `native_emb` (CLIP + cosine embedding NativE) | 26,08 ± 0,45 | 44,92 ± 0,53 | 54,22 ± 0,95 | 35,46 ± 0,58 |
| `shuffled_edges` (cạnh xáo trộn) | 21,45 ± 0,06 | 43,31 ± 0,51 | 53,29 ± 0,13 | 32,02 ± 0,16 |
| `native` (CLIP + cosine + đếm cạnh) | 29,82 ± 0,00 | 51,46 ± 0,00 | 59,42 ± 0,00 | 39,89 ± 0,00 |
| `random_emb` | 29,82 ± 0,00 | 51,46 ± 0,00 | 59,42 ± 0,00 | 39,89 ± 0,00 |
| `random_kgc` | 29,82 ± 0,00 | 51,46 ± 0,00 | 59,42 ± 0,00 | 39,89 ± 0,00 |

Ba dòng `native`, `random_emb` và `random_kgc` trùng với dòng chính. Lý do: trên val, cả 3 seed đều chọn β = 0, nên kênh embedding có trọng số 0 và thay bằng vector ngẫu nhiên cũng không đổi kết quả.

### Theo nhóm truy vấn (R@1 / MRR)

80,94 % truy vấn test liên kết được ít nhất một thực thể (722 / 892).

| Nhóm | `clip` | `native_kgc` | `kgc_only` | `native_emb` | `shuffled_edges` |
|---|---|---|---|---|---|
| Có liên kết (722) | 22,71 / 32,36 | 28,81 / 38,89 | 25,62 / 34,91 | 24,19 / 33,41 | 18,47 / 29,15 |
| Không liên kết (170) | 34,12 / 44,17 | 34,12 / 44,17 | 34,12 / 44,17 | 34,12 / 44,17 | 34,12 / 44,17 |

### Truy vấn viết tay (`data/queries/mkgw_queries_hand.json`, kho ảnh test, R@1 / MRR)

| Nhóm | `clip` | `kg_triple` | `native_kgc` |
|---|---|---|---|
| Trực tiếp (10 câu) | 80,0 / 85,00 | 80,0 / 85,00 | 80,0 / 85,00 |
| Gián tiếp (15 câu) | 40,0 / 49,28 | 80,0 / 83,33 | 80,0 / 83,33 |

## Nhận xét

- **Đồ thị giúp ở test, nhưng ít hơn ở val.** Dòng chính tăng R@1 4,93 điểm và MRR 5,28 điểm so với CLIP; ở val mức tăng R@1 là 8,07. Toàn bộ phần tăng nằm ở nhóm có liên kết (+6,10 R@1). Nhóm không liên kết giữ nguyên CLIP, đúng theo thiết kế.
- **Cạnh thật là nguồn của phần tăng.** Xáo đuôi cạnh làm R@1 còn 21,45, thấp hơn CLIP 3,44 điểm, cùng chiều với val.
- **NativE (SIGIR 2024) có đóng góp riêng, nhưng không cộng thêm được vào phép đếm cạnh.**
  - Đứng một mình, điểm bộ ba của NativE (`kgc_only`) tăng R@1 2,35 điểm so với CLIP. Mức này nhiều hơn cosine embedding (+1,19).
  - Thứ tự `kgc_only` > `native_emb` > `clip` giống với val, và ổn định qua 3 seed.
  - Khi đã có phép đếm cạnh, val chọn trọng số 0 cho NativE, nên dòng chính trùng `kg_triple`.
- **Truy vấn gián tiếp là nơi đồ thị có ích rõ nhất.** Câu gián tiếp chỉ nhắc tới thực thể láng giềng, không nhắc tới đáp án. Trên 15 câu này, R@1 tăng từ 40 lên 80. Trên 10 câu trực tiếp, kết quả không đổi (80). Bộ câu nhỏ, nên con số chỉ mang tính minh hoạ.
