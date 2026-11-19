# NativE trên MKG-W: tái lập link prediction (dòng 22) và bản an toàn rò rỉ cho retrieval (dòng 23)

Người làm: Đạt dựng môi trường và chạy smoke test (`../../native-repro/README.md`). Khang chạy bản đầy đủ trên molab bằng `notebooks/mkgw_molab.py` (bước 4), ngày 2026-11-16/17.

Siêu tham số giữ nguyên `scripts/run_mkgw.sh` của tác giả: dim 250, batch 1024, margin 4, 128 mẫu âm, lr 1e-4, lrg 1e-4, mu 1e-4, 1000 epoch. NativE ở commit `c72291c`, `Base.so` biên dịch lại trên máy chạy. Đánh giá dùng setting filter, trung bình hai chiều head và tail, không dùng ràng buộc kiểu (type constraint). Số liệu đầy đủ, gồm từng chiều head và tail, ở `metrics.json`. Số được chép từ khối kết quả cuối của từng log train.

## Bước a: tái lập số của bài báo

| | MRR | Hit@1 | Hit@3 | Hit@10 |
|---|---|---|---|---|
| Bài báo, Bảng 2 (NativE, MKG-W) | 36.58 | 29.56 | — | — |
| Nhóm, MKG-W gốc, seed 42 | 37.06 | 30.29 | 40.02 | 49.45 |

- Chênh lệch so với bài báo: MRR +0,48, Hit@1 +0,73. Cả hai đều dưới ngưỡng ±1 điểm đặt ra trong `../../native-repro/README.md`, nên **coi là tái lập được**.
- Bài báo chỉ báo MRR và Hit@1 cho MKG-W, nên Hit@3 và Hit@10 không có số để so.
- Đây là một lần chạy với seed gốc của tác giả, đúng như cách bài báo báo cáo.

## Bản an toàn rò rỉ `MKG-W-rmask` (3 seed)

`scripts/mask_native_modalities.py` đặt về 0 các hàng đặc trưng ảnh và đặc trưng text của 1 784 thực thể thuộc val và test của bài toán retrieval. Lý do: bảng text của NativE được trích từ chính mô tả thực thể, mà mô tả này chính là truy vấn của nhóm; bảng ảnh trích từ chính ảnh ứng viên (đề bài, Mục 6). Sau khi che, có 2 321 hàng ảnh và 2 555 hàng text bằng 0, vì bộ dữ liệu gốc vốn đã có sẵn 537 và 771 hàng bằng 0. Triple KGC giữ nguyên, và chỉ phần KGC-train được dùng để train.

| Seed | MRR | Hit@1 | Hit@3 | Hit@10 |
|---|---|---|---|---|
| 0 | 36.78 | 29.88 | 40.11 | 49.13 |
| 1 | 37.20 | 30.11 | 40.54 | 49.98 |
| 2 | 36.70 | 29.81 | 40.00 | 49.08 |
| **mean ± std** | **36.89 ± 0.27** | **29.93 ± 0.16** | **40.22 ± 0.28** | **49.40 ± 0.50** |

Việc che modality của val/test hầu như không làm giảm link prediction: MRR chỉ thấp hơn bản gốc 0,17, nhỏ hơn cả độ lệch giữa các seed. Embedding cấu trúc của ba seed này (`mkgw_native_ent_seed{0,1,2}`) là đầu vào cho kênh embedding ở bước b (`scripts/eval_mkgw_level2.py`).
