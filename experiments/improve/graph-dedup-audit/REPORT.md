# Audit làm sạch đồ thị ảnh: gộp trùng và gom nhóm theo vai trò (CPU, không train)

Lệnh chạy lại: `run.sh`. Code: `src/graph/dedup.py` (test: `tests/test_dedup.py`), `scripts/improve/graph_dedup_{audit,val,draw}.py`.
Kết quả: `structure.json` (train và val), `val_checks.json`. Kiểm tra bằng mắt: `visual_check.md`, ảnh trong `img/`.
Toạ độ khung và synset lấy từ file VG gốc trên server: `scripts/improve/build_boxes.py --synsets`, cho 46 944 ảnh
train (1 255 685 object) và 2 126 ảnh val, không object nào thiếu khung. Ngưỡng và tiêu chí chỉ chọn theo thống kê
train và việc xem ảnh; không chấm xếp hạng nào. Chưa train gì.

## Định nghĩa

- **Cùng loại.**
  - Synset của hai node cùng quy về person.n.01 nếu đều là người. VG gọi một người theo vai trò (man, skier,
    player, boy...), nên nhiều nhãn khác nhau có thể cùng chỉ một người.
  - Với loại khác, cùng loại khi một synset bằng hoặc là hypernym WordNet của synset kia (animal ~ dog, nhưng không
    dog ~ cat). Mình không cắt ở một độ sâu cố định trong cây, vì như vậy sẽ gộp dog với cat (cùng là carnivore).
  - Node không có synset thì so theo tên đã chuẩn hoá alias.
- **Bước 1, gộp trùng.**
  - Cùng loại và IoU ≥ ngưỡng, lấy bao đóng bắc cầu (union-find). Cùng tên mà khung khác thì giữ riêng.
  - Cạnh được trỏ lại vào node đã gộp; cạnh trùng và cạnh tự nối bị bỏ.
  - Node gộp vẫn đếm là **một** vật; số khung đã gộp ghi trong trường riêng.
- **Bước 2, gom nhóm** (không dùng IoU, làm trên đồ thị đã gộp). Hai node cùng loại được nối khi, với một **node
  neo** chung (đầu kia của một bộ ba), chúng có:
  - **(a)** cùng vị từ đã gán, cùng chiều, cả hai đều có cạnh; hoặc
  - **(b)** cùng lớp hình học tính từ khung so với node neo: trái / phải / trên / dưới / chồng lên hoặc chứa trong
    (giao ≥ ½ khung nhỏ) / bên cạnh (chồng lên theo chiều dọc và khoảng hở ngang ≤ bề rộng khung lớn hơn). Phải có
    ít nhất một trong hai node có cạnh đã gán tới node neo. Không xét trước/sau vì ảnh 2D không có độ sâu.
  - Biến thể **"gần nhau"** để so: cùng loại, tỉ lệ diện tích ≤ 2, khoảng hở giữa hai khung ≤ nửa cạnh khung trung bình.
  - Mỗi thành phần liên thông có từ 2 vật trở lên thành một node nhóm: số lượng = số vật, khung = khung bao chung,
    cạnh = hợp các cạnh của thành viên.
- **Tên biến thể:** `merge<t>` là gộp ở IoU ≥ t; `merge<t>+a/b/ab/near` là gộp rồi gom nhóm theo tiêu chí tương ứng.

## Cấu trúc đồ thị trước và sau khi làm sạch

### train (46944 ảnh)

Tỉ lệ node có một node cùng loại chồng lên với IoU ≥ 0,5 / 0,7 / 0,9 (trước khi gộp): 16.2% / 11.8% / 3.0%.

| Biến thể | node/ảnh | cạnh/ảnh | thành phần liên thông/ảnh | thành phần lớn nhất (node) | thành phần lớn nhất (% node) | node bậc 1 | node cô lập | node nhóm/ảnh | số vật TB mỗi nhóm | ảnh có nhóm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 26.75 | 18.05 | 10.96 | 10.49 | 43.6% | 55.9% | 22.1% | — | — | — |
| merge0.5 | 24.11 | 17.32 | 9.38 | 11.01 | 48.8% | 51.8% | 23.2% | — | — | — |
| merge0.7 | 24.88 | 17.52 | 9.80 | 10.95 | 47.4% | 53.1% | 22.9% | — | — | — |
| merge0.9 | 26.33 | 17.86 | 10.66 | 10.65 | 44.6% | 55.5% | 22.2% | — | — | — |
| merge0.7+a | 22.46 | 14.77 | 9.80 | 9.16 | 45.0% | 51.9% | 25.3% | 1.34 | 2.80 | 61.3% |
| merge0.7+b | 18.98 | 14.45 | 7.33 | 10.01 | 55.3% | 47.1% | 24.4% | 2.86 | 3.06 | 83.7% |
| merge0.7+ab | 18.82 | 14.27 | 7.33 | 9.89 | 55.1% | 47.0% | 24.6% | 2.90 | 3.09 | 84.3% |
| merge0.7+near | 21.82 | 16.16 | 8.24 | 10.72 | 51.6% | 50.9% | 22.5% | 1.97 | 2.56 | 71.5% |
| merge0.9+ab | 18.84 | 14.28 | 7.35 | 9.89 | 55.1% | 47.0% | 24.7% | 3.33 | 3.25 | 85.6% |

Cặp node cùng loại được nối theo từng tiêu chí, trên đồ thị đã gộp. Hai cột mâu thuẫn đếm theo node neo mà cả hai node cùng có cạnh tới.

| Gộp ở IoU | cặp theo (a) | cặp theo (b) | cặp theo cả hai | (a) đúng, (b) sai | (b) đúng, (a) sai |
|---|---|---|---|---|---|
| 0.5 | 212827 | 466128 | 188744 | 37207 | 24532 |
| 0.7 | 221428 | 538366 | 195240 | 39592 | 30033 |
| 0.9 | 234978 | 718988 | 205460 | 42111 | 39227 |

Biến thể hai tầng (chỉ đếm, chưa dựng), trên merge0.7 + (a) hoặc (b): thêm 2.90 node nhóm và 8.97 cạnh "member of" mỗi ảnh.

### val (2126 ảnh)

Tỉ lệ node có một node cùng loại chồng lên với IoU ≥ 0,5 / 0,7 / 0,9 (trước khi gộp): 16.3% / 11.8% / 3.1%.

| Biến thể | node/ảnh | cạnh/ảnh | thành phần liên thông/ảnh | thành phần lớn nhất (node) | thành phần lớn nhất (% node) | node bậc 1 | node cô lập | node nhóm/ảnh | số vật TB mỗi nhóm | ảnh có nhóm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 26.43 | 17.90 | 10.85 | 10.45 | 44.6% | 55.9% | 22.1% | — | — | — |
| merge0.5 | 23.79 | 17.17 | 9.25 | 10.98 | 49.8% | 51.9% | 23.0% | — | — | — |
| merge0.7 | 24.57 | 17.38 | 9.67 | 10.92 | 48.4% | 53.3% | 22.8% | — | — | — |
| merge0.9 | 26.00 | 17.71 | 10.54 | 10.58 | 45.5% | 55.6% | 22.2% | — | — | — |
| merge0.7+a | 22.21 | 14.67 | 9.67 | 9.14 | 46.0% | 52.1% | 25.2% | 1.30 | 2.81 | 60.7% |
| merge0.7+b | 18.74 | 14.34 | 7.22 | 9.95 | 56.0% | 47.0% | 24.3% | 2.83 | 3.06 | 82.5% |
| merge0.7+ab | 18.58 | 14.16 | 7.22 | 9.83 | 55.8% | 46.9% | 24.5% | 2.87 | 3.08 | 83.0% |
| merge0.7+near | 21.55 | 16.03 | 8.13 | 10.63 | 52.2% | 50.9% | 22.4% | 1.92 | 2.57 | 69.1% |
| merge0.9+ab | 18.61 | 14.17 | 7.24 | 9.83 | 55.8% | 46.9% | 24.6% | 3.29 | 3.25 | 84.3% |

Cặp node cùng loại được nối theo từng tiêu chí, trên đồ thị đã gộp. Hai cột mâu thuẫn đếm theo node neo mà cả hai node cùng có cạnh tới.

| Gộp ở IoU | cặp theo (a) | cặp theo (b) | cặp theo cả hai | (a) đúng, (b) sai | (b) đúng, (a) sai |
|---|---|---|---|---|---|
| 0.5 | 9397 | 20140 | 8337 | 1654 | 1002 |
| 0.7 | 9869 | 23509 | 8737 | 1782 | 1244 |
| 0.9 | 10550 | 32159 | 9238 | 1926 | 1634 |

Biến thể hai tầng (chỉ đếm, chưa dựng), trên merge0.7 + (a) hoặc (b): thêm 2.87 node nhóm và 8.87 cạnh "member of" mỗi ảnh.

## Val: nhất quán node chung và khớp số lượng

**Nhất quán node chung.** Đây là phép đo của thí nghiệm 2, làm lại trên đồ thị đã làm sạch: câu FACTUAL, vector cụm
CLIP đóng băng, mỗi cạnh mới giữ vector của quan hệ gốc.

Câu val có ≥1 cặp bộ ba chung node và có ảnh đúng trong top-50: 5957.

| Biến thể | nhất quán, ảnh đúng | nhất quán, 49 ảnh khác | chênh lệch | % câu ảnh đúng > TB ảnh khác | nhất quán hoàn toàn, ảnh đúng | nhất quán hoàn toàn, ảnh khác |
|---|---|---|---|---|---|---|
| before | 0.3696 | 0.3784 | -0.0088 | 38.2% | 31.9% | 31.7% |
| merge0.5 | 0.3883 | 0.3874 | +0.0009 | 40.1% | 33.7% | 32.5% |
| merge0.7 | 0.3826 | 0.3842 | -0.0016 | 39.5% | 33.1% | 32.2% |
| merge0.9 | 0.3729 | 0.3795 | -0.0066 | 38.6% | 32.2% | 31.8% |
| merge0.7+a | 0.3956 | 0.3937 | +0.0019 | 40.8% | 34.5% | 33.0% |
| merge0.7+b | 0.4208 | 0.4152 | +0.0057 | 43.2% | 36.8% | 35.0% |
| merge0.7+ab | 0.4214 | 0.4158 | +0.0056 | 43.2% | 36.9% | 35.1% |
| merge0.7+near | 0.4023 | 0.3956 | +0.0067 | 41.5% | 35.0% | 33.2% |
| merge0.9+ab | 0.4216 | 0.4156 | +0.0060 | 43.3% | 36.9% | 35.0% |

Node câu có nêu số lượng (≥ 2): 1152. Node số nhiều không nêu số: 4430.

| Biến thể | nêu số: ảnh đúng có đúng số vật khớp nhãn | nêu số: có node nhóm khớp nhãn mang đúng số lượng | số nhiều: ảnh đúng có ≥ 2 vật khớp | số nhiều: có node nhóm khớp (≥ 2) |
|---|---|---|---|---|
| before | 24.9% | — | 38.1% | — |
| merge0.5 | 31.2% | — | 35.3% | — |
| merge0.7 | 30.2% | — | 36.3% | — |
| merge0.9 | 26.2% | — | 37.6% | — |
| merge0.7+a | 29.1% | 17.3% | 36.4% | 15.7% |
| merge0.7+b | 25.7% | 21.4% | 34.8% | 27.8% |
| merge0.7+ab | 25.7% | 22.0% | 34.8% | 28.4% |
| merge0.7+near | 28.7% | 20.3% | 35.7% | 24.7% |
| merge0.9+ab | 22.6% | 19.4% | 35.8% | 30.0% |

**Khớp số lượng** chỉ xét ảnh đúng, với node của câu FACTUAL có nêu số lượng hoặc ở dạng số nhiều. Nhãn khớp khi
bằng nhau sau alias VG, hoặc cosine CLIP ≥ 0,955 như kênh hình học. Cột "node nhóm" chỉ có nghĩa với các biến thể có
gom nhóm.

## Nhận xét

1. **Khung trùng có thật và khá nhiều.**
   - Trên train, 16,2% node có một node cùng loại chồng lên với IoU ≥ 0,5; 11,8% ở 0,7; 3,0% ở 0,9.
   - Gộp ở 0,7 làm số node mỗi ảnh giảm từ 26,7 xuống 24,9, số thành phần liên thông giảm từ 11,0 xuống 9,8, và
     thành phần lớn nhất tăng từ 43,6% lên 47,4% số node. Các chuỗi bị đứt do khoanh lặp đã nối lại một phần.
2. **Đề xuất ngưỡng IoU 0,7.**
   - Trên ảnh: 0,7 gộp đúng các khung trùng rõ ràng (người bị khoanh 2 lần, biển STOP khoanh lặp).
   - 0,5 bắt đầu gộp những vùng khác nhau nằm sát nhau, ví dụ hai mảng cỏ.
   - 0,9 bỏ sót các khung trùng có IoU khoảng 0,8.
3. **Gộp trùng phải làm trước gom nhóm.** Nếu chỉ gộp ở 0,9 rồi gom (a) hoặc (b), được 3,33 nhóm mỗi ảnh với
   3,25 vật mỗi nhóm, so với 2,90 và 3,09 khi gộp ở 0,7. Phần chênh là khung trùng bị đếm thành vật riêng, đúng như
   lo ngại "horse ×2".
4. **(a) bảo thủ, (b) gom nhiều hơn gấp đôi.**
   - Trên train sau khi gộp 0,7: (a) cho 1,34 nhóm mỗi ảnh (61% ảnh có nhóm), (b) cho 2,86 (84% ảnh), (a) hoặc (b)
     cho 2,90.
   - Cặp được nối: theo (a) 221 nghìn, theo (b) 538 nghìn, theo cả hai 195 nghìn.
   - Mâu thuẫn tính theo node neo: 39,6 nghìn trường hợp cùng vị từ nhưng khác lớp hình học, thường vì lớp chia quá
     mịn (hai cây nến cùng "on" bậu cửa nhưng một cây "chồng lên", cây kia "phía trên"). 30,0 nghìn trường hợp cùng
     lớp hình học nhưng khác vị từ, phần lớn do nhãn VG nhiễu ("brushing h" so với "brushing") hoặc quan hệ ngược
     chiều ("snow on mountain" so với "mountain has snow"). Đây đúng là chỗ (b) bắt được còn (a) bỏ sót. Ví dụ trong
     `visual_check.md`.
   - Biến thể "gần nhau" cho 1,97 nhóm mỗi ảnh, nằm giữa (a) và (b).
5. **Nhất quán node chung chỉ nhích rất ít.**
   - Chênh lệch ảnh đúng − ảnh khác tăng từ −0,009 (trước) lên +0,006 (gộp 0,7 rồi gom (a) hoặc (b)).
   - Tỉ lệ câu có ảnh đúng cao hơn trung bình ảnh khác tăng từ 38,2% lên 43,2%.
   - Cả ảnh đúng lẫn ảnh khác đều nhất quán hơn, vì đồ thị nối lại hơn. Ảnh đúng vẫn chưa tách khỏi ảnh sai rõ ràng,
     nên lợi ích của việc làm sạch (nếu có) phải thể hiện qua train (thí nghiệm 3), không qua probe này.
6. **Khớp số lượng.**
   - Với 1 152 node câu có nêu số, tỉ lệ ảnh đúng có đúng số vật khớp nhãn tăng từ 24,9% lên 30,2% sau khi gộp ở 0,7.
   - Sau khi gom (a) hoặc (b), 22,0% có một node nhóm mang đúng số lượng đó. Với 4 430 node số nhiều không nêu số,
     28,4% có node nhóm khớp.
   - Tỉ lệ khớp số vật lại giảm sau khi gom (25,7%), vì node nhóm lấy tên của thành viên lớn nhất, và nhóm "cùng
     loại" có thể trộn man/skier. Cách đặt tên node nhóm cần xem lại nếu làm thí nghiệm 3.
7. **Biến thể hai tầng** (giữ node từng vật cùng node nhóm, nối bằng "member of"), chỉ đếm: trên merge0.7 + (a)
   hoặc (b), mỗi ảnh thêm 2,90 node nhóm và 8,97 cạnh "member of".
8. **Ghi chú quy trình.** Lần chạy đầu có lỗi đếm: node sau khi gộp trùng mang số khung như thể nhiều vật, làm thổi
   phồng cỡ nhóm. Kết quả lỗi được giữ lại ở `structure.count-bug.json` và `val_checks.count-bug.json`. Mọi số trong
   báo cáo này là sau khi sửa; test `tests/test_dedup.py` kiểm tra đúng trường hợp này.
