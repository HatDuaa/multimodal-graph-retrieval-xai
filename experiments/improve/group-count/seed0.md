# Thí nghiệm 4 — báo cáo nhanh seed 0 (23/10/2026, 23:10)

Báo cáo này viết khi seed 0 vừa xong và chưa gửi được cho KG (project): lúc gửi, mã liên lạc của phiên KG đã hết hạn.
Kết quả đủ 3 seed sẽ nằm ở `REPORT.md`.

`groupcount_factual_r3_seed0` đạt R@1 val tốt nhất 47,93 và MRR 60,06 ở epoch 9. Mức này thấp hơn fullgraph
(48,38 ± 0,17) 0,45, và thấp hơn cả seed thấp nhất của fullgraph (48,21). Vẫn cao hơn mốc 1a (47,48).

| Epoch | R@1 | R@5 | R@10 | MRR | Giây | a / b / c |
|---|---|---|---|---|---|---|
| 1 | 46,42 | 73,66 | 83,59 | 58,67 | 711 | 0,726 / 0,212 / 0,062 |
| 2 | 47,04 | 74,10 | 83,88 | 59,34 | 685 | 0,744 / 0,157 / 0,100 |
| 3 | 46,96 | 74,38 | 84,35 | 59,39 | 686 | 0,704 / 0,140 / 0,157 |
| 4 | 47,26 | 74,42 | 83,97 | 59,59 | 686 | 0,645 / 0,130 / 0,226 |
| 5 | 47,01 | 74,84 | 84,32 | 59,52 | 686 | 0,604 / 0,116 / 0,280 |
| 6 | 47,55 | 74,96 | 84,52 | 59,90 | 687 | 0,573 / 0,107 / 0,319 |
| 7 | 47,63 | 75,04 | 84,32 | 59,88 | 686 | 0,544 / 0,103 / 0,353 |
| 8 | 47,70 | 74,59 | 84,27 | 59,89 | 686 | 0,525 / 0,098 / 0,378 |
| **9** | **47,93** | 74,74 | 84,51 | **60,06** | 686 | 0,499 / 0,097 / 0,404 |
| 10 | 46,96 | 74,84 | 84,60 | 59,43 | 686 | 0,492 / 0,093 / 0,416 |

## R@1 theo nhóm câu, seed 0 của cả hai bản (`count_subset_seed0_cpu.json`)

Tính trên CPU để không chen GPU đang train seed 1. fullgraph seed 0 tính lại trên CPU ra đúng 48,21 như trên GPU.

| Nhóm câu val | Số câu | groupcount | fullgraph | Chênh |
|---|---|---|---|---|
| Tất cả | 10 633 | 47,93 | 48,21 | −0,28 |
| Có số lượng đã biết | 4 543 | 50,41 | 50,25 | +0,16 |
| — có số nêu rõ (1 / 2 / 3–4 / ≥5) | 1 235 | 55,38 | 52,31 | +3,07 |
| — chỉ có số nhiều hoặc group/herd… | 3 308 | 48,55 | 49,49 | −0,94 |
| Không rõ số lượng | 6 090 | 46,08 | 46,68 | −0,60 |

## Bảng g đã học (seed 0, checkpoint tốt nhất)

Hàng là số lượng phía câu, cột là nhóm số lượng phía ảnh.

| Câu \ Ảnh | 1 | 2 | 3–4 | ≥5 |
|---|---|---|---|---|
| không rõ (cố định) | 0 | 0 | 0 | 0 |
| 1 | +0,102 | +0,004 | −0,126 | −0,180 |
| 2 | −0,095 | +0,073 | +0,057 | −0,021 |
| 3–4 | −0,117 | −0,076 | +0,111 | +0,032 |
| ≥5 | −0,117 | −0,083 | −0,025 | +0,138 |
| ≥2 | −0,052 | −0,000 | +0,039 | +0,053 |
| ≥3 | −0,104 | −0,056 | +0,040 | +0,092 |

## Cách đọc (mới 1 seed)

- **g học đúng cấu trúc:** mọi ô trên đường chéo dương, các ô lệch số lượng âm. Nó giúp rõ ở câu có số nêu rõ
  (+3,1 R@1 trên 1 235 câu).
- **Nhưng các câu mà g không chạm tới vẫn tụt:** câu không rõ số lượng (hàng g cố định bằng 0) −0,6; câu chỉ có số
  nhiều −0,9. Nhiều khả năng do chính graph đã gom: số node giảm 24% và một phần nhãn đổi sang hypernym.
- **Đối chiếu:** checkpoint fullgraph seed 0 chưa train lại, chấm thẳng trên graph đã gom, cũng tụt từ 48,21 xuống
  47,50.
- **Muốn tách hai phần** thì cần thêm một bản: bảng g trên merge_v1, không gom nhóm, node đơn đếm số vật cùng loại.
  Bản này chưa chạy vì nằm ngoài kế hoạch.

Đính chính: tin gửi KG trước đó ghi 129 test pass; số đúng là 128/128.
