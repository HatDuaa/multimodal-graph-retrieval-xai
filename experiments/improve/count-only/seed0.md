# Thí nghiệm 5 — báo cáo nhanh seed 0 (24/10/2026, 13:59)

`countonly_factual_r3_seed0` đạt R@1 val tốt nhất 49,15 và MRR 61,10, ở epoch 8.

- So với fullgraph_factual_r3 (48,38 ± 0,17): cao hơn trung bình 0,77, và cao hơn seed tốt nhất của fullgraph (48,56).
- So với groupcount (47,75 ± 0,22): cao hơn 1,40.

| Epoch | R@1 | R@5 | R@10 | MRR | Giây | a / b / c |
|---|---|---|---|---|---|---|
| 1 | 47,00 | 73,96 | 83,87 | 59,24 | 964 | 0,726 / 0,212 / 0,062 |
| 2 | 47,55 | 74,45 | 84,37 | 59,86 | 931 | 0,737 / 0,162 / 0,100 |
| 3 | 48,01 | 74,95 | 84,48 | 60,16 | 930 | 0,708 / 0,155 / 0,137 |
| 4 | 48,37 | 75,43 | 84,65 | 60,54 | 932 | 0,662 / 0,148 / 0,190 |
| 5 | 48,30 | 75,70 | 84,95 | 60,62 | 931 | 0,630 / 0,136 / 0,234 |
| 6 | 48,51 | 75,54 | 84,81 | 60,65 | 931 | 0,599 / 0,129 / 0,272 |
| 7 | 48,49 | 75,36 | 85,23 | 60,65 | 931 | 0,568 / 0,124 / 0,308 |
| **8** | **49,15** | 75,28 | 85,23 | **61,10** | 931 | 0,548 / 0,119 / 0,333 |
| 9 | 48,72 | 75,60 | 84,91 | 60,87 | 932 | 0,528 / 0,112 / 0,360 |
| 10 | 48,58 | 75,83 | 85,32 | 60,84 | 932 | 0,517 / 0,107 / 0,376 |

Mỗi epoch mất 931 giây, so với 681 giây của fullgraph. Phần chậm thêm nằm ở phép tra bảng g cho từng cặp node: backward
của nó cộng dồn vào một bảng chỉ có 28 ô. Kết quả không bị ảnh hưởng. Mình không sửa giữa chừng, để 3 seed chạy cùng
một code.

## R@1 theo tập con câu, seed 0 của hai bản (`count_subset_seed0_cpu.json`, tính trên CPU)

| Tập con val | Số câu | countonly | fullgraph | Chênh |
|---|---|---|---|---|
| Tất cả | 10 633 | 49,15 | 48,21 | +0,94 |
| Có số lượng | 4 543 | 51,38 | 50,25 | +1,13 |
| — có số nêu rõ | 1 235 | 55,14 | 52,31 | +2,83 |
| — chỉ số nhiều / group | 3 308 | 49,97 | 49,49 | +0,48 |
| Không rõ số lượng | 6 090 | 47,49 | 46,68 | +0,81 |

## Bảng g, seed 0

Hàng là số lượng phía câu, cột là nhóm số lượng phía ảnh.

| Câu \ Ảnh | 1 | 2 | 3–4 | ≥5 |
|---|---|---|---|---|
| không rõ (cố định) | 0 | 0 | 0 | 0 |
| 1 | +0,087 | +0,005 | −0,088 | −0,149 |
| 2 | −0,081 | +0,071 | +0,035 | −0,030 |
| 3–4 | −0,094 | −0,088 | +0,105 | +0,015 |
| ≥5 | −0,125 | −0,092 | −0,015 | +0,136 |
| ≥2 | −0,036 | −0,005 | +0,035 | +0,037 |
| ≥3 | −0,073 | −0,046 | +0,031 | +0,064 |

Cấu trúc giống thí nghiệm 4: đường chéo dương, các ô lệch số lượng âm.

Mới có 1 seed. Ở các câu không rõ số lượng, g không tác động trực tiếp, nên mức +0,81 ở đó có thể một phần là dao động
giữa các seed: R@1 của fullgraph trên tập này ở 3 seed là 46,68 / 47,39 / 46,83.
