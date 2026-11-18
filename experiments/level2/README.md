# Cấp độ 2 trên MKG-W: CLIP và CLIP + đồ thị (dòng 21, 23, 24, 25, 26)

Người làm:
- Đạt dựng dữ liệu, split, linker và bộ chấm điểm đồ thị (`src/retrieval/mkgw_graph_score.py`), rồi chạy thử quét α trên một checkpoint (`rerank_sweep/`).
- Khang train NativE 3 seed trên bản che modality (`native-train/README.md`), viết `scripts/eval_mkgw_level2.py` cùng bộ truy vấn soạn tay, và chạy đánh giá ngày 2026-11-18 trên molab (`notebooks/mkgw_molab.py`, bước 5 và 6).

Log của từng lần chạy: `val_run.log`, `test_run.log`. Số liệu từng seed: `<split>/metrics_<hàng>.json`; bảng tổng: `<split>/summary.json`.

## Thiết lập

- **Bài toán:** truy vấn ở mức thực thể. Truy vấn là mô tả Wikidata đã che tên thực thể, đáp án là ảnh của thực thể đó. Pool val và pool test đều có 892 ảnh (split của Đạt, seed 0).
- **CLIP:** OpenAI ViT-B/32, đóng băng, giống cấp độ 1. Lấy top-50 theo cosine.
- **Hai kênh đồ thị**, chấm cho từng ảnh ứng viên trong top-50:
  - `triple`: số cạnh KG trực tiếp giữa thực thể của ảnh ứng viên và các thực thể mà linker tìm thấy trong câu. Chỉ dùng cạnh thuộc phần train của KGC.
  - `emb`: cosine lớn nhất giữa embedding cấu trúc NativE của ứng viên và của một thực thể được nhắc. Đây là bước b của dòng 23: embedding thực thể thay cho điểm GNN.
- **Điểm cuối** = α·z(CLIP) + (1−α)·z(β·z(emb) + (1−β)·z(triple)). Z-score tính trong top-50 của từng câu.
- **NativE an toàn rò rỉ:** train trên bản `MKG-W-rmask`. Ở bản này, vector ảnh và vector văn bản của 1 784 thực thể val/test bị đặt về 0, vì ảnh của chúng chính là đáp án cần tìm. Có 3 seed (0, 1, 2), mỗi seed cho một bộ embedding.
- **Chống rò rỉ:**
  - α và β được chọn trên val riêng cho từng seed (lưới α 0–1 bước 0,05, β ∈ {0; 0,25; 0,5; 0,75; 1}). Lựa chọn lưu ở `val/choice.json`.
  - Test chạy một lần, đọc lại lựa chọn đó.
  - Bộ 25 truy vấn soạn tay chỉ dùng để đánh giá.
- **Một điểm cần ghi rõ về thứ tự chạy:** trước lần chạy test chính thức, Khang (cùng Claude) chạy thử `--split test --hand` với embedding giả để kiểm tra code. Hàng `kg_triple` không dùng embedding, nên số test của hàng này, cả trên bộ soạn tay, đã hiện ra ở lần chạy thử. Lựa chọn α của hàng này lúc đó đã chốt trên val, và sau khi thấy số thì không có gì được chỉnh.

## Kết quả chính

Lựa chọn trên val giống nhau ở cả 3 seed: **α = 0, β = 0**. Nghĩa là val không dùng kênh embedding NativE. Vì vậy, mô hình chính (hàng `native`) trùng hệ với hàng `kg_triple`.

Với α = 0, thứ hạng đi theo số cạnh KG. CLIP chỉ phân định các ảnh có cùng số cạnh, vì phép sắp xếp ổn định giữ nguyên thứ tự top-50 của CLIP.

**Test** (892 câu, một lần; mean ± std qua 3 seed; std = 0 khi hàng đó không phụ thuộc seed):

| Hàng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 25.00 | 44.73 | 54.60 | 34.66 |
| CLIP + đồ thị (`native`, α, β chọn trên val, = `kg_triple`) | **29.82** | **51.46** | **59.64** | **39.91** |
| CLIP + embedding NativE (`native_emb`, ép β = 1, α chọn trên val) | 26.05 ± 0.51 | 45.03 ± 0.28 | 54.33 ± 0.84 | 35.54 ± 0.40 |
| Đối chứng: xáo cạnh KG (`shuffled_edges`) | 21.41 ± 0.11 | 43.35 ± 0.56 | 53.44 ± 0.06 | 31.99 ± 0.13 |

**Val** (892 câu, dùng để chọn α và β):

| Hàng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 24.33 | 44.28 | 53.92 | 34.22 |
| CLIP + đồ thị (`native` = `kg_triple`) | **32.40** | **50.78** | **57.96** | **41.18** |
| `native_emb` (β = 1) | 26.53 ± 0.36 | 46.26 ± 0.34 | 55.27 ± 0.30 | 35.96 ± 0.26 |
| `shuffled_edges` | 21.82 ± 0.39 | 42.49 ± 0.00 | 52.13 ± 0.19 | 32.03 ± 0.23 |

Hàng `random_emb` (thay embedding NativE bằng vector ngẫu nhiên) cho kết quả trùng hàng chính, vì với β = 0 kênh embedding không tham gia. Hàng này vẫn có trong `*/metrics_random_emb.json` nhưng không mang thêm thông tin.

## Đọc kết quả

1. **Cấu trúc quan hệ là phần giúp.** Kênh đếm cạnh tăng R@1 test thêm 4,82 điểm so với CLIP. Khi giữ nguyên bậc của mỗi đầu cạnh nhưng xáo đuôi cạnh trong cùng quan hệ, R@1 tụt xuống 21,41, thấp hơn cả CLIP. Như vậy phần cải thiện đến từ việc cạnh nối đúng thực thể, không phải từ việc một thực thể có nhiều cạnh.
2. **Embedding NativE thay cho điểm GNN chỉ giúp ít** (dòng 23, bước b). Khi chỉ dùng embedding, R@1 tăng 1,05 điểm trên test và 2,20 điểm trên val. Khi trộn với kênh đếm cạnh, mọi β > 0 trên val đều thấp hơn β = 0. R@1 tốt nhất theo từng β, gộp 3 seed:

   | β | 0 | 0,25 | 0,5 | 0,75 | 1 |
   |---|---|---|---|---|---|
   | R@1 val tốt nhất | 32,40 | 31,05–31,17 | 30,16–30,83 | 28,25–29,37 | 26,12–26,79 |

   - Khi dùng riêng embedding (α = 0, β = 1), R@1 val chỉ còn 15,5–16,3, thấp xa CLIP.
   - Giả thuyết (chưa kiểm chứng riêng): embedding đo độ "gần" của hai thực thể trong không gian KGC. Nhiều thực thể cùng loại, cùng lân cận đều gần thực thể được nhắc, nên kênh này phân biệt kém. Đếm cạnh thì chỉ cộng cho ứng viên có cạnh thật.

   Nhóm báo cáo đúng như vậy: ở bài toán này, phần đóng góp của NativE nằm ở dòng `native_emb`, không nằm ở mô hình chính.
3. **Độ nhạy theo α** (dòng 26, hướng mở rộng "độ nhạy α"), với β = 0, seed 0, R@1 val:

   | α | 0–0,35 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 | 1 (CLIP) |
   |---|---|---|---|---|---|---|---|---|
   | R@1 | 32,4 | 32,1 | 31,6 | 30,6 | 29,7 | 28,4 | 26,6 | 24,3 |

   Đường cong phẳng từ 0 đến 0,35, rồi giảm dần. Lý do: truy vấn tự động là mô tả Wikidata, gần như toàn thông tin quan hệ (ví dụ "1935 film directed by …"), nên kênh đồ thị gánh gần hết. Trên MMKG tiếng Việt, câu có nhiều chi tiết hình ảnh hơn nên val chọn α = 0,7 (`../level3/vnmmkg/README.md`).
4. **Theo nhóm câu, tức câu có hay không có thực thể được linker tìm thấy** (test):

   | Nhóm | Số câu | CLIP R@1 | CLIP + đồ thị R@1 | CLIP MRR | CLIP + đồ thị MRR |
   |---|---|---|---|---|---|
   | Có thực thể được nhắc | 722 (80,94%) | 22.85 | **28.81** | 32.40 | **38.89** |
   | Không có | 170 | 34.12 | 34.12 | 44.26 | 44.26 |

   Ở nhóm không có thực thể, hai kênh đồ thị đều bằng 0, nên thứ hạng giữ nguyên như CLIP. Toàn bộ phần tăng nằm ở nhóm có thực thể.

## Truy vấn soạn tay: trực tiếp và gián tiếp (dòng 24, 25)

Bộ truy vấn `data/queries/mkgw_queries_hand.json` có 25 câu, đã khoá trước khi chạy test. Pool là ảnh test. Cách soạn và kiểm tra nằm ở `scripts/build_mkgw_hand_queries.py`.
- **10 câu trực tiếp:** mô tả thứ nhìn thấy được trong ảnh. Mỗi câu có đúng một đáp án trong pool, và không có thực thể nào được nhắc trong câu là hàng xóm KG của đáp án. Điều kiện này để kênh đồ thị không đi tắt được.
- **15 câu gián tiếp:** đáp án sinh từ một mẫu cạnh KG, ví dụ "phim do X đạo diễn". Đáp án là mọi thực thể trong pool thoả mẫu đó (tối đa 5), và tên đáp án không xuất hiện trong câu.

| Nhóm | Số câu | CLIP R@1 | CLIP + đồ thị R@1 | CLIP MRR | CLIP + đồ thị MRR |
|---|---|---|---|---|---|
| Trực tiếp | 10 | 80.00 | 80.00 | 85.00 | 85.00 |
| Gián tiếp | 15 | 40.00 | **80.00** | 49.28 | **83.33** |

Kênh đồ thị không làm hỏng câu trực tiếp, vì những câu này không nhắc thực thể nào nối với đáp án. Với câu gián tiếp, số câu đúng ở hạng 1 tăng từ 6 lên 12 trên 15 câu. Số câu ít nên các con số này chỉ có tính minh hoạ.

Giải thích của mỗi ảnh là danh sách các cạnh KG đã được đếm (`MkgwGraphScorer.score()["matches"]`). Danh sách này sinh ra từ chính phép xếp hạng, không phải tìm đường đi sau khi đã có kết quả.

## Về dòng 21: "CLIP + GNN trên MKG-W"

Mô hình cấp độ 1 (GATv2 trong từng scene graph) cần scene graph cho mỗi ảnh. MKG-W không có scene graph, mà có một KG chung cho các thực thể. Vì vậy, ở cấp độ 2, điểm đồ thị là hai kênh KG ở trên, thay cho GNN cấp độ 1. Dòng 21 được hiểu là chạy lại CLIP thuần và CLIP + điểm đồ thị trên MKG-W.

## Chạy lại

```bash
# sau khi có 3 checkpoint rmask (native-train/README.md)
python scripts/export_native_embeddings.py --checkpoint <ckpt seed s> --what ent --suffix _seed<s>   # s = 0, 1, 2
python scripts/eval_mkgw_level2.py --split val
python scripts/eval_mkgw_level2.py --split test --hand
```

Hoặc trên molab: `notebooks/mkgw_molab.py`, bấm bước 5 rồi bước 6. Embedding đã export được lưu ở repo HF riêng tư của nhóm, trong `data/features/mkgw_native_ent_seed*`.
