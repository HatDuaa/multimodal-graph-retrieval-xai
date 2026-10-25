# Phần 2 — Kênh số lượng + vị trí từ khung VG (val, không train)

Lệnh chạy lại: `scripts/improve/build_boxes.py --split val`, rồi `scripts/improve/geometry_channel.py`; cả hai có
trong `../parser-compare/run_eval.sh`. Kết quả: `geometry_channel.json`. Bảng: `summary.md`, sinh bằng
`scripts/improve/summarize_parsers.py`. Ví dụ giải thích: `examples.md`. Luật: `src/graph/geometry.py`
(test: `tests/test_geometry.py`).

## 2a. Toạ độ khung

`data/graphs/vg_coco_scene_graphs.jsonl` chỉ giữ nhãn, không có toạ độ. Toạ độ được lấy từ file VG gốc đã có sẵn
trên server (`data/raw/visual_genome/objects.json`, `relationships.json`), không tải lại. Đồ thị ảnh được dựng lại
bằng đúng code đóng gói (`build_scene_graph`, open vocabulary) kèm id object gốc của VG. Script kiểm tra id, tên và
quan hệ đều trùng với bản đóng gói, rồi gắn khung (x, y, w, h). Kết quả: 2 126 ảnh val, 56 181 object, không object
nào thiếu khung. File: `data/processed/geometry/boxes_val.json` (không commit), chạy mất 29 giây.

## 2b. Luật (không train)

- **Số lượng.** Lấy số đếm tường minh 2–10 trong câu bằng spaCy: chữ số hoặc chữ ("two dogs", "3 cats"), cộng với
  "a pair of X" (= 2). Đếm instance khớp nhãn trong ảnh, sau khi gộp các khung trùng cùng nhãn (IoU ≥ 0,5, vì VG hay
  lặp object). Điểm = 1 − min(|n − c| / c, 1).
- **Vị trí.** Lấy các bộ ba của đồ thị câu cấp 1 (sng) có vị từ left of / right of / above / below / under /
  underneath / beneath / on top of / next to / beside, kiểm trên mọi cặp (khung chủ ngữ, khung tân ngữ):
  - left/right/above/below: so tâm hai khung;
  - on top of: cần thêm chồng lấn theo chiều ngang;
  - next to/beside: cần chồng lấn theo chiều dọc và khoảng hở ngang không quá bề rộng khung lớn hơn.

  Điểm = tỉ lệ cặp thoả. Bỏ "in front of" và "behind" vì cần độ sâu.
- **Khớp nhãn.** Hai nhãn khớp khi bằng nhau sau chuẩn hoá alias VG, hoặc khi cosine CLIP đóng băng của
  "a photo of a {}" ≥ 0,955. Ngưỡng này chọn trước khi nhìn val, từ vài cặp nhãn phổ thông: man/men 0,985,
  man/guy 0,983, dog/puppy 0,959, boy/child 0,958, so với dog/cat 0,931, man/woman 0,941, cow/bull 0,945.
- **Thiếu bằng chứng.** Nếu ảnh không có instance nào khớp thì mục đó không được tính, vì đó là việc của kênh vật
  thể. Kênh của một ứng viên là trung bình các mục kiểm được, hoặc NaN nếu không kiểm được mục nào.

## 2c. Kênh thứ 4 cộng vào main_r3

`điểm cuối = fused_main_r3 + w · z(kênh)`. z-score tính trên top-50 của từng câu, giống các kênh khác; NaN nhận giá
trị trung bình. w được quét trên val. Thêm một kiểm tra 2-fold: chia val theo ảnh đúng, chọn w trên một nửa và chấm
trên nửa kia, để thấy việc quét làm số đẹp lên bao nhiêu. Ba biến thể: `geometry` (số lượng + vị trí), `count`,
`spatial`.

## 2c. Kênh số lượng + vị trí cộng vào main_r3 (val, 3 seed)

Câu có số đếm: 1121 (1150 mục). Câu có vị từ không gian: 756 (799 bộ ba; theo loại {'left': 0, 'right': 0, 'above': 33, 'below': 166, 'on_top': 30, 'beside': 570}). Tỉ lệ cặp (câu, ứng viên) kiểm được: geometry 5.1%, count 4.5%, spatial 0.7%.

### Kênh `geometry`

| w | R@1 toàn val | MRR toàn val | R@1 câu có số đếm | MRR | R@1 câu có vị từ không gian | MRR |
|---|---|---|---|---|---|---|
| 0.0 | 46.48 ± 0.24 | 58.85 ± 0.18 | 53.20 ± 0.14 | 65.14 ± 0.06 | 54.37 ± 0.95 | 65.67 ± 0.45 |
| 0.1 (tốt nhất trên val) | 46.58 ± 0.27 | 58.95 ± 0.20 | 53.94 ± 0.14 | 65.98 ± 0.12 | 54.72 ± 0.65 | 65.93 ± 0.36 |
| chọn w trên nửa kia (2-fold) | 46.53 ± 0.28 | 58.92 ± 0.22 | 53.64 ± 0.14 | 65.79 ± 0.32 | 54.63 ± 0.69 | 65.84 ± 0.33 |

R@1 toàn val theo w: 0.0: 46.48, 0.05: 46.52, 0.1: 46.58, 0.15: 46.57, 0.2: 46.55, 0.3: 46.43, 0.4: 46.31, 0.5: 46.11, 0.75: 45.73, 1.0: 45.39, 1.5: 44.73, 2.0: 44.31.

### Kênh `count`

| w | R@1 toàn val | MRR toàn val | R@1 câu có số đếm | MRR | R@1 câu có vị từ không gian | MRR |
|---|---|---|---|---|---|---|
| 0.0 | 46.48 ± 0.24 | 58.85 ± 0.18 | 53.20 ± 0.14 | 65.14 ± 0.06 | 54.37 ± 0.95 | 65.67 ± 0.45 |
| 0.1 (tốt nhất trên val) | 46.56 ± 0.26 | 58.94 ± 0.20 | 53.94 ± 0.14 | 65.96 ± 0.11 | 54.45 ± 0.81 | 65.76 ± 0.39 |
| chọn w trên nửa kia (2-fold) | 46.52 ± 0.27 | 58.91 ± 0.22 | 53.55 ± 0.22 | 65.71 ± 0.37 | 54.45 ± 0.90 | 65.74 ± 0.45 |

R@1 toàn val theo w: 0.0: 46.48, 0.05: 46.50, 0.1: 46.56, 0.15: 46.56, 0.2: 46.52, 0.3: 46.46, 0.4: 46.34, 0.5: 46.18, 0.75: 45.91, 1.0: 45.67, 1.5: 45.10, 2.0: 44.74.

### Kênh `spatial`

| w | R@1 toàn val | MRR toàn val | R@1 câu có số đếm | MRR | R@1 câu có vị từ không gian | MRR |
|---|---|---|---|---|---|---|
| 0.0 | 46.48 ± 0.24 | 58.85 ± 0.18 | 53.20 ± 0.14 | 65.14 ± 0.06 | 54.37 ± 0.95 | 65.67 ± 0.45 |
| 0.2 (tốt nhất trên val) | 46.51 ± 0.26 | 58.87 ± 0.19 | 53.23 ± 0.10 | 65.17 ± 0.04 | 54.76 ± 0.80 | 65.87 ± 0.40 |
| chọn w trên nửa kia (2-fold) | 46.47 ± 0.25 | 58.85 ± 0.18 | 53.20 ± 0.14 | 65.15 ± 0.07 | 54.32 ± 1.00 | 65.64 ± 0.52 |

R@1 toàn val theo w: 0.0: 46.48, 0.05: 46.49, 0.1: 46.50, 0.15: 46.49, 0.2: 46.51, 0.3: 46.44, 0.4: 46.43, 0.5: 46.39, 0.75: 46.30, 1.0: 46.21, 1.5: 46.13, 2.0: 46.07.


## Nhận xét

1. **Kênh số lượng giúp nhẹ nhưng đều trên câu có số đếm.** Trên 1 121 câu có số đếm (10,5% val), R@1 tăng từ
   53,20 ± 0,14 lên 53,94 ± 0,14 (+0,74) với w = 0,1 chọn trên val. Khi chọn w theo 2-fold, R@1 còn 53,64 ± 0,14
   (+0,44); MRR tăng +0,84 và +0,65. Trên toàn val chỉ tăng +0,10 và +0,05, vì kênh chỉ kiểm được 4,5% số cặp
   (câu, ứng viên).
2. **Kênh vị trí không có bằng chứng giúp.** Trên 756 câu có vị từ không gian, R@1 tăng +0,35 với w tốt nhất trên
   val, nhưng còn −0,05 khi chọn theo 2-fold, trong khi std giữa các seed trên tập này là 0,7–1,0. Val gần như không
   có left/right: 570 trong 799 bộ ba là "next to/beside", một quan hệ mà khung VG kiểm rất lỏng. Kênh chỉ kiểm được
   0,7% số cặp.
3. **Giới hạn của luật.** VG hay gán một khung cho cả nhóm ("people") và thường thiếu nhãn, nên câu "five people"
   có thể chỉ đếm được 2. Xem các dòng ✗ trong `examples.md`. Một số trường hợp đúng: "Three young men are standing
   next to four bicycles" thì `câu: 3 man — ảnh: 3 man ✓`, `câu: 4 bicycle — ảnh: 4 bicycle ✓`.
4. **Có nên đưa vào mô hình train?** Theo luật đã đặt, chỉ tính tới khi kênh thắng rõ trên tập con. Kênh vị trí:
   không. Kênh số lượng: mức tăng 0,4–0,7 R@1 trên tập con, lớn hơn std giữa các seed (0,14), nhưng nhỏ về tuyệt đối
   (khoảng 5–8 câu trên 1 121). Đề xuất: giữ kênh số lượng như một kênh luật **không train**, cộng sau main_r3 với
   w = 0,1, vì nó có dòng giải thích đọc được ngay. Việc đưa vào mô hình train để nhóm quyết; chưa làm gì.
