# Thí nghiệm 5 — Chỉ thêm số lượng và bảng g lên fullgraph, không gom nhóm (`countonly_factual_r3`)

> **Bản nháp (24/10/2026, 18:00).** Mới có seed 0 và seed 1. Lúc 17:41 KG đã dừng chuỗi vì Lộc cần GPU; seed 2 dừng
> sau epoch 4 (`checkpoint_last.pt` lưu lúc 17:39). Khi được phép chạy lại GPU, mình sẽ: (1) chạy tiếp seed 2 từ
> checkpoint đó với `--resume`, (2) chạy bộ chẩn đoán (`diagnostics.sh`), (3) điền các mục đang ghi "chờ".

Mọi số ở đây là của **val**. Test chưa chấm.

Mốc so sánh là `fullgraph_factual_r3` (`../full-graph/`), R@1 48,38 ± 0,17. So thêm với thí nghiệm 4
`groupcount_factual_r3` (`../group-count/`), R@1 47,75 ± 0,22.

## Thay đổi so với fullgraph

Công thức train giữ nguyên như fullgraph:

- dữ liệu merge_v1;
- `--fusion raw`, cùng hằng số;
- hình học trên cạnh;
- kNN-5 cùng cầu nối;
- node tổng;
- micro-batch 32, `expandable_segments:False`;
- tối đa 10 epoch.

Chỉ thêm số lượng và bảng g (`--count-table`), **không gom nhóm**.

**Phía ảnh (`scripts/improve/package_counts.py`, dữ liệu `count_v1`)**

- Mỗi node của merge_v1 vẫn là một vật riêng và giữ nhãn gốc.
- Số lượng của node là số vật cùng loại trong ảnh sau khi gộp trùng, tính cả chính nó. "Cùng loại" theo `same_type` của
  audit: synset/hypernym, và mọi người (man, woman, skier…) tính chung một loại.
- Số lượng được xếp vào 4 nhóm: 1 / 2 / 3–4 / ≥5.
- Để tiết kiệm đĩa, graph không được chép lại. Chỉ ghi `data/graphs/count_v1/count_buckets.json` (9 MB); GraphStore
  gắn số lượng vào node qua `paths.count_buckets` (`configs/count_v1.yaml`).

Tỉ lệ node theo nhóm số lượng:

| | 1 | 2 | 3–4 | ≥5 |
|---|---|---|---|---|
| val | 50,6% | 19,3% | 14,6% | 15,5% |
| train | 50,2% | 19,4% | 15,0% | 15,3% |

Thí nghiệm 4 (đã gom nhóm) có 67% node ở nhóm "1"; ở đây ít hơn nhiều vì mỗi node đếm cả các vật cùng loại khác
trong ảnh.

**Phía câu và bảng g:** giữ y như thí nghiệm 4 (`../group-count/REPORT.md`). Cùng luật số lượng, dùng lại query graph
của group_v1 (chỉ đọc). Bảng g khởi tạo bằng 0, hàng "không rõ" cố định bằng 0, và chỉ cộng vào kênh vật thể.

## Kiểm tra trước khi train

- **g = 0** (`check_zero.json`, `logs/check_zero_repeat.log`):
  - Checkpoint `fullgraph_factual_r3_seed0` với bảng g bằng 0, chạy trên count_v1, cho R@1 0,48208408 và MRR 0,60380839,
    trùng `metrics.json`. Lặp lại 5 lần đều như vậy.
  - Cùng trọng số nhưng không có bảng g: 2 trong 14 lần chạy cho MRR lệch 2,7e-7 (một câu đổi hạng 18↔19); R@1/5/10 không
    đổi. Lệch này xuất hiện cả trên merge_v1 khi không có g. Nguyên nhân là các phép cộng dồn trên GPU trong GAT không
    tất định, không phải do g.
- **Khớp số lượng ở ảnh đúng** (`check_match.json`): 5 871 node câu có số lượng đã biết. Có node ảnh khớp nhãn 66,1%;
  khớp cả số lượng 40,7%. Trùng hàng "merge_v1, chưa gom" của audit. Thí nghiệm 4 (đã gom nhóm) chỉ đạt 62,3% / 37,3%.

## Hạ tầng

- Một epoch mất 931 giây; fullgraph mất 681 giây. Phần chậm thêm là phép tra bảng g cho từng cặp (câu, ảnh, node,
  node): backward của nó cộng dồn vào một bảng chỉ có 28 ô. Kết quả không bị ảnh hưởng. Mình không sửa giữa chừng để
  các seed chạy cùng một code (commit `c9102e9`).
- Peak GPU khoảng 7,6 GB. Seed 0 và seed 2 mỗi seed có 1 cảnh báo phân mảnh bộ nhớ lúc chấm val ở epoch 1, đều tự hồi
  phục.
- **Ổ đĩa** còn 21 GB trong suốt quá trình chạy (sàn là 20 GB). `run.sh` kiểm tra trước mỗi seed. Phần đĩa hao đi là của
  người dùng khác; mình không xoá gì.
- **Dòng thời gian:**
  - seed 0: 11:23–13:59;
  - seed 1: 13:59–16:36;
  - seed 2: bắt đầu 16:36, dừng 17:41 theo yêu cầu (GPU cần cho việc khác). Đã xong 4 epoch, phần dở của epoch 5 bị
    bỏ.

## Val theo epoch

| Epoch | seed 0 R@1 | seed 0 MRR | seed 1 R@1 | seed 1 MRR | seed 2 R@1 | seed 2 MRR |
|---|---|---|---|---|---|---|
| 1 | 47,00 | 59,24 | 46,79 | 59,12 | 46,95 | 59,10 |
| 2 | 47,55 | 59,86 | 47,46 | 59,80 | 47,22 | 59,68 |
| 3 | 48,01 | 60,16 | 47,91 | 60,16 | 48,22 | 60,28 |
| 4 | 48,37 | 60,54 | 48,33 | 60,49 | 47,98 | 60,29 |
| 5 | 48,30 | 60,62 | 47,80 | 60,23 | — | — |
| 6 | 48,51 | 60,65 | 48,60 | 60,76 | — | — |
| 7 | 48,49 | 60,65 | 48,56 | 60,66 | — | — |
| 8 | **49,15** | **61,10** | 48,69 | 60,88 | — | — |
| 9 | 48,72 | 60,87 | **48,85** | 60,80 | — | — |
| 10 | 48,58 | 60,84 | 48,78 | 60,89 | — | — |

Checkpoint tốt nhất theo val R@1 (R@1 / R@5 / R@10 / MRR):

- seed 0, epoch 8: 49,15 / 75,28 / 85,23 / 61,10;
- seed 1, epoch 9: 48,85 / 75,21 / 84,76 / 60,80;
- seed 2: chờ (tốt nhất tới epoch 4 là 48,22 ở epoch 3).

Hai seed đầu có R@1 trung bình 49,00, cao hơn fullgraph (48,38 ± 0,17) và cao hơn mọi seed của fullgraph (48,21 /
48,56 / 48,39). Chưa đủ 3 seed nên chưa báo mean ± std hay kiểm định.

Trọng số kênh a/b/c ở epoch tốt nhất: seed 0 là 0,548 / 0,119 / 0,333; seed 1 là 0,527 / 0,121 / 0,352. Cả hai đều chạy
đủ 10 epoch, không dừng sớm.

## R@1 theo tập con câu

Chờ `count_subset.json` (3 seed của countonly, fullgraph và groupcount). Riêng seed 0 đã có bản tính sớm trên CPU
(`count_subset_seed0_cpu.json`, `seed0.md`):

| Tập con val | Số câu | countonly s0 | fullgraph s0 | Chênh |
|---|---|---|---|---|
| Tất cả | 10 633 | 49,15 | 48,21 | +0,94 |
| Có số lượng | 4 543 | 51,38 | 50,25 | +1,13 |
| — có số nêu rõ | 1 235 | 55,14 | 52,31 | +2,83 |
| — chỉ số nhiều / group | 3 308 | 49,97 | 49,49 | +0,48 |
| Không rõ số lượng | 6 090 | 47,49 | 46,68 | +0,81 |

## Bảng g đã học (checkpoint tốt nhất)

Hàng là số lượng phía câu, cột là nhóm số lượng phía ảnh (1 / 2 / 3–4 / ≥5); hàng "không rõ" cố định bằng 0.

| Câu \ Ảnh | seed 0 | seed 1 |
|---|---|---|
| 1 | +0,087 / +0,005 / −0,088 / −0,149 | +0,108 / −0,022 / −0,104 / −0,184 |
| 2 | −0,081 / +0,071 / +0,035 / −0,030 | −0,069 / +0,075 / +0,044 / −0,041 |
| 3–4 | −0,094 / −0,088 / +0,105 / +0,015 | −0,100 / −0,065 / +0,106 / +0,016 |
| ≥5 | −0,125 / −0,092 / −0,015 / +0,136 | −0,106 / −0,090 / −0,017 / +0,116 |
| ≥2 | −0,036 / −0,005 / +0,035 / +0,037 | −0,035 / +0,001 / +0,023 / +0,024 |
| ≥3 | −0,073 / −0,046 / +0,031 / +0,064 | −0,068 / −0,047 / +0,032 / +0,067 |

Cấu trúc giống thí nghiệm 4 và gần như giống nhau giữa hai seed: đường chéo dương, lệch số lượng âm. Hàng "≥2" nhỏ hơn
thí nghiệm 4 (khoảng ±0,04 so với ±0,05).

## Chờ bộ chẩn đoán

Các mục sau sẽ được điền sau khi chạy `diagnostics.sh`:

- **Ca g làm đổi node được ghép** (`count_match_changes.json/.md`): số ca, phân loại tốt/xấu, ví dụ ghi cả node khi
  có g và khi không có g. Có so sánh với 3 seed của thí nghiệm 4 (`exp4-reference/`).
  - Bản chạy thử trên thí nghiệm 4 seed 0: g đổi node ghép ở 14,0% node câu có số lượng (783 / 5 582); 26 ca tốt,
    197 ca xấu. Tỉ lệ khớp nhãn giảm từ 61,6% xuống 58,5%, tỉ lệ khớp số lượng tăng từ 52,5% lên 63,6%.
- **Ví dụ giải thích có dòng số lượng** (`count_examples.md`).
- **Đổi động từ / chủ–tân, độ dồn, tương quan giữa các kênh, gradient** (`summary.md`).
