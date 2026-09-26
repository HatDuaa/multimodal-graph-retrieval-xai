# Kết quả re-ranker đồ thị

## Kiểm tra bước 0

Mô hình mới khởi tạo được kiểm tra ở chế độ đánh giá trên đúng các ứng viên top-50 của tập val.
Các vector phần tử, điểm thô, z-score, điểm `Graph`, điểm trộn và thứ hạng của 200 truy vấn đầu
được đối chiếu với `experiments/checks/three_channel_probe_reference_scores.npz`, sai số tuyệt đối
tối đa cho phép là `1e-4`. R@1 toàn bộ val phải khớp `0.43496661337346` của phép thử ba kênh
không huấn luyện, với trọng số ban đầu `(a,b,c) = (0,60; 0,28; 0,12)` và nhiệt độ `0,05`.

Lệnh chạy:

```bash
~/venvs/mgrx/bin/python scripts/checks/step0_reference_check.py
```

Kết quả lần chạy kiểm tra được ghi trong log của script; kết quả huấn luyện sẽ bổ sung sau.

## Tăng tốc đường dữ liệu và đo bộ nhớ (2026-09-23)

`GraphStore` tính sẵn một lần, cho mọi ảnh và câu truy vấn của train/val, chỉ số dòng trong bảng vector phần tử
và `edge_index` cục bộ (mảng số nguyên). Bảng vector nằm hẳn trong RAM (không memory-map), được chép một lần lên GPU,
và mỗi batch chỉ lấy vector bằng torch indexing. Val được chấm theo batch 256 câu. Kiểm tra bước 0 chạy trên
đúng đường mới: R@1 val = 0,43496661337346, 200 truy vấn đầu lệch tối đa 1,96e-5 ở `z_objects` và 5,9e-6 ở điểm trộn (ngưỡng 1e-4), thứ tự giống hệt.

Giây mỗi epoch (gồm cả chấm val), batch hiệu dụng 256:

| Đường dữ liệu | Micro-batch | Giây/epoch |
|---|---|---|
| Cũ, dựng đồ thị từ JSON mỗi bước (`main_r2_seed2`) | 32 | 2 779 – 2 794 |
| Cũ (`main_seed0`, chẩn đoán) | 256 | 486 – 494 |
| Mới (`check_fastpath_main_seed2_mb128`) | 128 | 285 – 287 |

Run mới lặp lại cấu hình của `main_r2_seed2` (seed 2). Epoch 1: R@1 0,4524 so với 0,4515 cũ, R@5 0,7273 so với 0,7271,
MRR 0,5772 so với 0,5766. Epoch 2: R@1 0,4589 so với 0,4585. Mọi chênh lệch ≤ 1e-3.

Bộ nhớ theo micro-batch, 30 bước tối ưu của epoch 1 (`scripts/checks/memory_probe.py`, kết quả trong
`experiments/checks/memory_probe*.jsonl`). Lúc đo, GPU không có tiến trình nào của người khác; RAM máy còn trống ~111 GB / 125 GB.

| Micro-batch | RAM đỉnh | GPU cấp phát đỉnh | GPU giữ đỉnh (mặc định) | GPU giữ đỉnh (`expandable_segments`) | Giây/bước |
|---|---|---|---|---|---|
| 32 | 4,95 GB | 4,79 GB | 10,23 GB | — | 0,360 |
| 64 | 4,95 GB | 8,22 GB | 18,27 GB | 9,58 GB | 0,350 |
| 128 | 4,95 GB | 14,92 GB | 22,91 GB | 17,70 GB | 0,338 |
| 256 | — | tràn bộ nhớ GPU (OOM) | — | — | — |

Script huấn luyện giờ mặc định dùng `expandable_segments` và micro-batch 128, mức lớn nhất không tràn bộ nhớ
(~17,7 GB trên 24 GB; GPU không có người khác dùng nên không cần chừa). Batch hiệu dụng vẫn là 256.

## Kết quả r3 trên val (2026-09-24)

Toàn bộ số dưới đây là của **val**, lấy ở epoch tốt nhất theo R@1 val, mean ± std (ddof=1) qua 3 seed (0, 1, 2), đi qua
`src/eval/metrics.py`. Test chưa được chạm. Mọi dòng dùng cùng một công thức huấn luyện: code ở commit `268b5b6`,
đường dữ liệu tính sẵn, batch hiệu dụng 256, micro-batch 128, xáo mỗi epoch, khoá cổng trong epoch 1, AdamW lr 1e-3,
dừng sớm sau 2 epoch không tăng, tối đa 10 epoch. Mô tả mô hình: `docs/level1-model.md`. Tạo lại bảng bằng
`python scripts/summarize_level1.py r3` (ghi `experiments/level1/summary_r3.json`). Cả 24 run xong ngay lần đầu
(exit 0, không phải resume lần nào).

| Cấu hình | Seed xong | R@1 | R@5 | R@10 | MRR | a / b / c | Epoch tốt nhất |
|---|---|---|---|---|---|---|---|
| CLIP thuần | — | 39,13 | 67,88 | 78,97 | 52,34 | 1 / 0 / 0 | — |
| Ba kênh, không huấn luyện (bước 0) | — | 43,50 | 70,95 | 81,59 | 56,10 | 0,60 / 0,28 / 0,12 | — |
| **Đầy đủ** (GAT, 3 kênh, a/b/c theo từng câu) | 3/3 | 46,48 ± 0,24 | 73,78 ± 0,16 | 83,44 ± 0,24 | 58,85 ± 0,18 | 0,485 / 0,355 / 0,160 | 8, 10, 6 |
| a, b, c cố định (a, b, c học trên train qua bias, dùng chung cho mọi câu) | 3/3 | 46,13 ± 0,15 | 73,20 ± 0,13 | 83,20 ± 0,06 | 58,38 ± 0,10 | 0,488 / 0,368 / 0,144 | 6, 3, 7 |
| Bỏ GAT | 3/3 | 46,71 ± 0,07 | 73,81 ± 0,29 | 83,56 ± 0,24 | 59,04 ± 0,09 | 0,473 / 0,353 / 0,174 | 5, 5, 3 |
| Tắt kênh bộ ba (GAT vẫn dùng cạnh) | 3/3 | 45,99 ± 0,05 | 73,33 ± 0,13 | 82,84 ± 0,10 | 58,34 ± 0,05 | 0,504 / 0,496 / 0 | 4, 4, 6 |
| Bỏ hẳn quan hệ | 3/3 | 45,98 ± 0,10 | 73,12 ± 0,05 | 82,90 ± 0,04 | 58,29 ± 0,04 | 0,499 / 0,501 / 0 | 5, 8, 6 |
| Câu là text thuần (phía câu không qua GAT/MLP/U) | 3/3 | **46,92 ± 0,07** | **74,01 ± 0,04** | **83,74 ± 0,07** | **59,25 ± 0,03** | 0,486 / 0,348 / 0,166 | 5, 4, 6 |
| Huấn luyện lại trên cạnh nối ngẫu nhiên (`rewire_r3`) | 3/3 | 46,24 ± 0,24 | 73,63 ± 0,37 | 83,30 ± 0,16 | 58,64 ± 0,25 | 0,492 / 0,349 / 0,159 | 2, 8, 9 |
| Loss LambdaRank theo MRR thay cross-entropy (`lambda_r3`) | 3/3 | 45,18 ± 0,34 | 72,74 ± 0,23 | 82,81 ± 0,19 | 57,70 ± 0,23 | 0,524 / 0,345 / 0,132 | 1, 2, 3 |

Hai dòng CLIP thuần và ba kênh không huấn luyện không có seed. R@5, R@10 và MRR của dòng ba kênh lấy từ phép thử 1b;
chạy lại kiểm tra bước 0 cho R@5 70,95, R@10 81,59, MRR 56,10.

**Đối chứng phá đồ thị, chỉ đánh giá** (`scripts/checks/graph_controls.py`; 3 checkpoint `main_r3` × 3 seed đối chứng
= 9 lần chấm; kết quả trong `experiments/level1/controls/`):

| Đối chứng | R@1 | MRR |
|---|---|---|
| Đồ thị đúng (cùng 3 checkpoint) | 46,48 | 58,85 |
| Mỗi ảnh nhận scene graph của một ảnh val khác (hoán vị không điểm bất động) | 26,33 ± 0,39 | 39,71 ± 0,31 |
| Nối lại cạnh giữ bậc vào/ra, nhãn quan hệ đi theo cạnh, cụm bộ ba mã hoá lại | 46,29 ± 0,19 | 58,72 ± 0,13 |

**Quét α trên val cho dòng a, b, c cố định** (`scripts/checks/alpha_sweep.py`, chỉ đánh giá, giữ β = b/(b+c) đã học,
lưới 0,05–0,95). α học được là 0,488 / 0,499 / 0,477, cho R@1 46,01 / 46,07 / 46,30. α tốt nhất trên lưới là
0,45 / 0,50 / 0,45, cho 46,05 / 46,08 / 46,49. Số ở α chọn trên val bị lạc quan vì chính val đã chọn nó. Dòng này dùng a, b, c học trên train, **không** dùng α chọn trên val. Plan v7 định chọn α trên val cho dòng này, nhưng bước quét α chỉ được báo trên val, và test không chấm lại với α chọn trên val (như vậy là chạm test thêm một lần sau khi đã thấy số). Trên val, α học được kém α tốt nhất trên lưới tối đa 0,19 điểm R@1. Mô hình học α (tức a) thay vì chọn tay, đúng tinh thần "điều chỉnh cách kết hợp điểm nếu giải thích được cơ chế" của đề.

### Số nói gì

1. **Học có ích.** So với bước 0 không huấn luyện, mô hình đầy đủ tăng 2,98 điểm R@1 (43,50 → 46,48). So với CLIP thuần
   là tăng 7,35 điểm.
2. **GAT không giúp.** Bỏ GAT (46,71 ± 0,07) không thua mô hình đầy đủ (46,48 ± 0,24). Dòng tốt nhất là dòng câu không
   qua bộ mã hoá đồ thị (46,92 ± 0,07). Lan truyền trên đồ thị câu còn làm kém đi khoảng 0,4 điểm.
3. **Quan hệ giúp một chút, qua kênh bộ ba tường minh, không qua lan truyền.** Tắt kênh bộ ba (45,99) hay bỏ hẳn quan hệ
   (45,98) đều thấp hơn mô hình đầy đủ khoảng 0,5 điểm, và hai dòng này bằng nhau. Nghĩa là cạnh quan hệ trong GAT không
   thêm gì khi đã có kênh bộ ba.
4. **Cấu trúc nối cạnh gần như không quan trọng.** Nối lại cạnh ngẫu nhiên lúc đánh giá chỉ làm giảm 0,19 điểm. Huấn
   luyện lại trên đồ thị đã nối ngẫu nhiên cho 46,24 ± 0,24, kém 0,24 điểm, cỡ một độ lệch chuẩn. Sau khi nối lại, bộ ba
   vẫn giữ đúng chủ thể và nhãn quan hệ, chỉ đổi đối tượng sang một vật thể khác có thật trong ảnh.
5. **Mô hình thật sự dựa vào đồ thị của đúng ảnh.** Thay bằng đồ thị của ảnh khác thì R@1 rơi xuống 26,33, **thấp hơn
   CLIP thuần 12,8 điểm**, chứ không phải rơi về gần 39,13 như dự đoán trước. Lý do: a ≈ 0,49, tức mô hình đặt khoảng
   một nửa trọng số vào kênh đồ thị, và lúc đánh giá không tự hạ được trọng số này khi đồ thị sai.
6. **a, b, c theo từng câu giúp ít.** Trọng số theo từng câu hơn trọng số cố định 0,35 điểm R@1 và 0,47 điểm MRR. α học
   trên train đã gần điểm tốt nhất trên val (chọn lại α trên val chỉ thêm tối đa 0,19 điểm).
7. **Loss LambdaRank theo MRR kém cross-entropy** 1,3 điểm R@1 và 1,15 điểm MRR, dù được thiết kế để tối ưu MRR. Nó
   dừng sớm ở epoch 1–3. Chưa tinh chỉnh (dùng cùng lr và nhiệt độ), nên chỉ ghi nhận là kết quả âm.

### Lưu ý khi đọc

- Mọi so sánh là trên val, và val cũng dùng để dừng sớm, nên số hơi lạc quan như nhau cho mọi dòng. Số chính thức phải
  lấy trên test, một lần.
- Các chênh lệch dưới khoảng 0,3 điểm (đầy đủ so với bỏ GAT, đầy đủ so với rewire) chỉ cỡ 1–2 độ lệch chuẩn với 3 seed.
  Chưa làm kiểm định theo cặp trên từng câu.
- Dừng sớm sau 2 epoch không tăng khá nhạy với nhiễu của val. Ví dụ `rewire_r3_seed0` dừng ở epoch 4 với tốt nhất 45,96,
  trong khi hai seed còn lại lên 46,37–46,38. Hai run (`main_r3_seed1`, `rewire_r3_seed1`) đạt tốt nhất đúng ở epoch 10,
  tức trần 10 epoch có thể đã chặn chúng.

## Kết quả trên test (2026-09-25, chạy một lần)

Lộc chốt phương pháp chính là **mô hình đầy đủ** (có lan truyền GAT ở cả phía câu và phía ảnh), vì cách xử lý câu
khớp với cách xử lý scene graph của ảnh. Quyết định đưa ra trước khi chạy test. Test được chấm đúng một lần bằng
`scripts/evaluate_test.py --training-free` (commit `f738de3`), dùng `checkpoint_best.pt` (chọn theo val) của
`main_r3_seed0..2`. Trước khi chấm, script kiểm tra ứng viên top-50 của test tái tạo đúng số CLIP thuần đã có.
Kết quả ở `experiments/level1/test/`. Mỗi file chỉ ghi một lần, script từ chối chạy lại. Các ablation chưa được
chấm trên test.

| Phương pháp | R@1 | R@5 | R@10 | MRR | a / b / c |
|---|---|---|---|---|---|
| CLIP thuần | 39,32 | 66,39 | 77,32 | 52,00 | — |
| Ba kênh, không huấn luyện | 42,54 | 70,04 | 80,35 | 55,05 | 0,60 / 0,28 / 0,12 |
| **Đầy đủ (3 seed)** | **45,77 ± 0,23** | **72,87 ± 0,31** | **82,69 ± 0,31** | **58,14 ± 0,18** | 0,483 / 0,358 / 0,159 |

10 696 truy vấn trên pool 2 138 ảnh test. Theo từng seed, R@1 là 45,93 / 45,88 / 45,50.

So với CLIP thuần, mô hình đầy đủ tăng **+6,45 R@1**, +6,48 R@5, +5,37 R@10 và +6,14 MRR. So với bản ba kênh không
huấn luyện, phần học thêm được +3,23 R@1. Số trên test thấp hơn trên val khoảng 0,7 điểm R@1 (46,48 → 45,77), đúng như
dự đoán vì val đã được dùng để dừng sớm và chọn checkpoint. Mức tăng so với CLIP trên test (+6,45) gần bằng trên val
(+7,35).

### Các ablation trên test (chạy một lần, sau khi đã chốt phương pháp chính)

Chấm bằng `scripts/evaluate_test.py --method <dòng>_r3` (commit `8b36243`), dùng checkpoint tốt nhất theo val của
từng seed. Riêng dòng `rewire_r3` được chấm trên đồ thị test đã nối lại cạnh với cùng seed nối như lúc huấn luyện.
Cụm bộ ba mới sinh ra khi nối lại được mã hoá vào `vg_coco_graph_parts_rewire_seed<k>_test`.

| Cấu hình | R@1 | R@5 | R@10 | MRR | a / b / c |
|---|---|---|---|---|---|
| CLIP thuần | 39,32 | 66,39 | 77,32 | 52,00 | — |
| Ba kênh, không huấn luyện | 42,54 | 70,04 | 80,35 | 55,05 | 0,60 / 0,28 / 0,12 |
| **Đầy đủ (phương pháp chính)** | **45,77 ± 0,23** | 72,87 ± 0,31 | 82,69 ± 0,31 | 58,14 ± 0,18 | 0,483 / 0,358 / 0,159 |
| a, b, c cố định (a, b, c học trên train qua bias, dùng chung cho mọi câu) | 45,13 ± 0,05 | 72,25 ± 0,19 | 82,15 ± 0,25 | 57,47 ± 0,11 | 0,488 / 0,368 / 0,144 |
| Bỏ GAT | 45,95 ± 0,20 | 73,01 ± 0,07 | 82,78 ± 0,07 | 58,25 ± 0,09 | 0,471 / 0,356 / 0,173 |
| Tắt kênh bộ ba | 45,07 ± 0,22 | 71,99 ± 0,06 | 82,00 ± 0,08 | 57,40 ± 0,13 | 0,502 / 0,498 / 0 |
| Bỏ hẳn quan hệ | 45,26 ± 0,14 | 72,10 ± 0,08 | 82,05 ± 0,15 | 57,55 ± 0,11 | 0,497 / 0,503 / 0 |
| Câu là text thuần (phía câu không qua GAT/MLP/U) | 45,61 ± 0,29 | 73,21 ± 0,03 | 82,93 ± 0,11 | 58,16 ± 0,24 | 0,484 / 0,351 / 0,165 |
| Huấn luyện và chấm trên cạnh nối ngẫu nhiên | 45,50 ± 0,40 | 72,65 ± 0,43 | 82,67 ± 0,14 | 57,86 ± 0,36 | 0,490 / 0,352 / 0,158 |
| Loss LambdaRank (MRR) | 44,28 ± 0,30 | 71,51 ± 0,23 | 81,81 ± 0,19 | 56,78 ± 0,26 | 0,523 / 0,346 / 0,132 |

Dòng "a, b, c cố định" trên test dùng a, b, c học trên train, không dùng α chọn trên val (xem mục quét α ở trên).

Test xác nhận phần lớn kết luận trên val:
- Trọng số a, b, c theo từng câu có ích: +0,64 R@1 so với trọng số cố định. Mức này trên test rõ hơn trên val (+0,35).
- Quan hệ có ích qua kênh bộ ba: +0,5 đến +0,7 R@1.
- GAT không giúp: bỏ GAT cho 45,95, cao hơn bản đầy đủ 0,18, cỡ một độ lệch chuẩn.
- Nối cạnh ngẫu nhiên chỉ làm giảm 0,27, dưới một độ lệch chuẩn.
- Loss LambdaRank kém nhất trong các bản đã huấn luyện, thua bản đầy đủ 1,5 R@1.

Có một chỗ **đổi thứ tự so với val**: dòng câu là text thuần tốt nhất trên val (46,92) nhưng trên test chỉ đạt 45,61,
thấp hơn bản đầy đủ 0,16 R@1, trong khi R@5, R@10 và MRR vẫn ngang hoặc nhỉnh hơn. Các chênh lệch dưới khoảng 0,3 điểm
giữa bản đầy đủ, bỏ GAT, câu là text thuần và rewire nằm trong nhiễu giữa các seed; chưa làm kiểm định theo cặp.
