# Kiểm tra tính đúng đắn của pipeline

Hai phép kiểm tra này **không phải thực nghiệm của đề tài** (đề không yêu cầu, không có đồ thị, không tính là tập dữ liệu của nhóm). Chúng trả lời hai câu hỏi về baseline CLIP: pipeline có chạy đúng không, và cách tiền xử lý ảnh có nên đổi không. Mọi số dưới đây lấy từ file JSON cùng thư mục, do script sinh ra, không sửa tay. Chạy ngày 2026-09-19 trên RTX 4090, encoder `ViT-B-32/openai` (open_clip 3.3.0, torch 2.14.0).

## 1. Đối chiếu với số đã công bố trên MSCOCO 5K

**Vì sao cần.** Tập test của nhóm (Visual Genome ∩ COCO, 2 138 ảnh) là tập con của tập test chuẩn MSCOCO 5K (split Karpathy, 5 000 ảnh); chỉ 2 138 ảnh trong đó có scene graph. Pool nhỏ hơn thì ít ảnh gây nhiễu hơn nên Recall cao hơn, vì vậy số của nhóm (R@1 = 38,50) **không so trực tiếp được** với số trong các bài báo (pool 5 000). Để biết pipeline có đúng không, chạy chính code của nhóm (mã hoá ảnh, mã hoá văn bản, FAISS, module đánh giá) trên đúng pool 5 000 ảnh rồi so với số đã công bố cho cùng mô hình.

**Cách chạy.** `python scripts/checks/coco5k_zero_shot.py` — tải 5 000 ảnh test gốc từ `images.cocodataset.org` (~800 MB, giữ lại ở `data/raw/coco5k_images/`), mã hoá 5 000 ảnh và 25 010 caption, chấm cả hai chiều. Kết quả: `coco5k_zero_shot.json`.

**Số tham chiếu.** Bảng benchmark chính thức của open_clip, file `docs/openclip_retrieval_results.csv`, dòng `ViT-B-32,openai`, các cột "MSCOCO image retr." (text→ảnh) và "MSCOCO text retr." (ảnh→text).

| Chiều | Nguồn | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Text → ảnh (chiều của đề tài) | Đã công bố | 30,44 | 55,94 | 66,87 |
| | Pipeline của nhóm | 30,26 | 54,97 | 65,71 |
| | Chênh lệch | −0,18 | −0,97 | −1,16 |
| Ảnh → text | Đã công bố | 50,12 | 75,00 | 83,52 |
| | Pipeline của nhóm | 49,16 | 73,34 | 82,34 |
| | Chênh lệch | −0,96 | −1,66 | −1,18 |

**Kết luận.** Tái lập sát, lệch tối đa 1,66 điểm, ở chiều của đề tài lệch tối đa 1,16 điểm và R@1 chỉ lệch 0,18. Lỗi hệ thống (sai biến thể mô hình, sai tiền xử lý, chấm điểm sai) thường gây lệch vài điểm đến hàng chục điểm, nên pipeline được coi là đúng. Không ghi là "khớp hoàn toàn".

**Về khoảng lệch còn lại.**
- Đã loại trừ: số caption mỗi ảnh. Một số ảnh có 6–7 caption; khi chỉ lấy 5 caption đầu (25 000 câu) kết quả gần như không đổi (text→ảnh 30,26 / 54,98 / 65,71).
- Đã kiểm tra: tiền xử lý ảnh giống nhau. Mặc định của open_clip (`PreprocessCfg.resize_mode = "shortest"`) là co cạnh ngắn về 224 px rồi **cắt hình vuông ở giữa**, đúng cách nhóm đang dùng; không phải thêm viền đen.
- Chưa kiểm tra: bộ benchmark chuẩn có thể chạy ở độ chính xác số khác (nửa chính xác), phiên bản thư viện khác, hoặc xử lý văn bản caption hơi khác.

**Lưu ý khi trình bày.** Bài báo gốc của CLIP (Radford và cộng sự, 2021) chỉ báo cáo zero-shot retrieval cho mô hình lớn nhất (ViT-L/14, 336 px); số của ViT-B/32 lấy từ bảng open_clip ở trên. Cách làm giống hệt baseline của nhóm: mô hình đóng băng, xếp hạng theo cosine, không huấn luyện, không đồ thị.

## 2. Cách đưa ảnh về hình vuông trước khi vào CLIP

**Vì sao cần.** CLIP cắt hình vuông ở giữa nên không thấy hai đầu của ảnh ngang hoặc ảnh dọc. Trên split val: 73% ảnh ngang, 23% ảnh dọc, tỉ lệ khung hình trung vị 1,33; trung bình chỉ 71% chiều dài ảnh còn lại sau khi cắt, ảnh tệ nhất còn 25%. Câu hỏi: thêm viền cho vuông (thấy đủ ảnh) có tốt hơn không?

**Cách chạy.** `python scripts/checks/resize_mode_probe.py` — mã hoá lại 2 126 ảnh **val** theo ba chế độ của open_clip, dùng lại đặc trưng caption có sẵn, 10 633 truy vấn. Chỉ dùng val nên lựa chọn không đụng tới test (Mục 6 đề bài). Kết quả: `resize_mode_probe.json`.

| Chế độ | Ý nghĩa | R@1 | R@5 | R@10 | MRR@50 | R@50 |
|---|---|---|---|---|---|---|
| `shortest` | Cắt giữa (mặc định, **đang dùng**) | 38,00 | 66,20 | 77,19 | 51,06 | 96,59 |
| `longest` | Co cạnh dài, thêm viền đen cho vuông | 37,87 | 65,34 | 77,09 | 50,67 | 96,46 |
| `squash` | Ép méo về hình vuông | 37,99 | 65,05 | 76,75 | 50,72 | 96,50 |

**Kết luận.** Ba cách gần như ngang nhau (chênh 0,1–1,2 điểm), cắt giữa không thua ở chỉ số nào. Giữ nguyên mặc định, **không trích lại đặc trưng**. Giải thích hợp lý: CLIP được huấn luyện với ảnh đã cắt; thêm viền làm vật thể nhỏ đi và đưa vào dải viền mô hình ít gặp; ép méo làm sai hình dạng; còn caption COCO thường tả nội dung chính nằm giữa ảnh.

**Hạn chế vẫn còn, ghi vào báo cáo.** Vật thể nằm ở rìa ảnh có thể không được CLIP nhìn thấy, trong khi scene graph của Visual Genome được gán trên toàn bộ ảnh. Đây là một chỗ bước xếp hạng lại bằng đồ thị có thể bù cho CLIP; khi có kết quả CLIP + Đồ thị, nên xem riêng nhóm ảnh có tỉ lệ khung hình lớn.

## File giữ lại

| Thứ | Ở đâu | Ghi chú |
|---|---|---|
| Script | `scripts/checks/` | Trong git |
| Kết quả | `experiments/checks/*.json` | Trong git |
| 5 000 ảnh test MSCOCO | `data/raw/coco5k_images/` (793 MB) | Chỉ trên máy chạy; script chạy lại sẽ bỏ qua ảnh đã có |
| Đặc trưng MSCOCO 5K | `data/features/coco5k_{image,caption}.npy` + `.ids.json` (~60 MB) | Chỉ trên máy chạy; checksum ở `data/MANIFEST.md`. Không đưa lên Drive vì các gói khác không dùng |
