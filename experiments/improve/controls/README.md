# Kiểm soát đồ thị lúc eval cho `countonly_factual_r3` (val, không train lại)

Làm giống `experiments/level1/controls` của mô hình cấp 1. Mục đích là xem phần tăng của mô hình cuối có thật sự đến từ
đồ thị của ảnh hay không. Không train gì, không mở test.

- **Đồ thị của ảnh khác (`other_image`):** mỗi ảnh val được chấm bằng đồ thị của một ảnh val khác. Phép hoán vị không có
  điểm cố định, cùng hàm `graph_derangement` và cùng seed với cấp 1. Đồ thị mang theo box và nhóm số lượng của nó.
- **Nối lại cạnh ngẫu nhiên (`rewire`):** trong mỗi đồ thị, đích của các quan hệ có nhãn được hoán vị, giữ bậc vào và bậc
  ra, nhãn vị từ đi theo cạnh (cùng `rewire_graphs` và cùng seed với cấp 1).
  - Cạnh hoàn thiện (`[none]` kNN và cầu nối) và hình học cạnh được dựng lại từ box theo cấu trúc của run, như với mọi
    đồ thị. Cạnh kNN vì vậy vẫn là cạnh không gian thật, chỉ các quan hệ có nhãn bị xáo.
  - Các cụm bộ ba mới mà bảng vector chưa có (khoảng 18 400 cụm mỗi seed) được mã hoá bằng cùng CLIP đông cứng, giữ trong
    RAM, không ghi ra đĩa.
- **Số lần chạy:** 3 seed kiểm soát (0, 1, 2) trên mỗi seed của run.
  - Mỗi seed run lấy trung bình qua 3 seed kiểm soát.
  - Sau đó tính trung bình ± độ lệch chuẩn qua 3 seed run.
  - R@1 khi chưa xáo trùng với `metrics.json` của từng run.

Script: `scripts/improve/graph_controls_countonly.py` (chạy) và `scripts/improve/summarize_controls_countonly.py` (tổng
hợp, `summary.json`). Mỗi run mất khoảng 16 giây cho cả 4 lần chấm trên RTX 4090. Log: `graph_controls.log`.

## Kết quả (val, %)

| Mô hình | Kiểm soát | R@1 gốc | R@1 sau khi xáo | Chênh R@1 | MRR sau khi xáo | Chênh MRR |
|---|---|---|---|---|---|---|
| `countonly_factual_r3` | đồ thị ảnh khác | 48,91 | 21,91 ± 1,11 | −27,00 ± 1,08 | 34,62 ± 1,13 | −26,24 ± 1,10 |
| `countonly_factual_r3` | nối lại cạnh | 48,91 | 48,32 ± 0,16 | −0,59 ± 0,06 | 60,47 ± 0,17 | −0,38 ± 0,04 |
| `main_r3` (cấp 1) | đồ thị ảnh khác | 46,48 | 26,33 ± 0,26 | −20,15 ± 0,32 | 39,71 ± 0,26 | −19,15 ± 0,35 |
| `main_r3` (cấp 1) | nối lại cạnh | 46,48 | 46,29 ± 0,22 | −0,19 ± 0,02 | 58,72 ± 0,15 | −0,14 ± 0,04 |
| CLIP thuần | — | 39,13 | | | 52,34 | |

R@5 và R@10 nằm trong `summary.json`. Với `countonly`, đồ thị ảnh khác làm R@5 giảm 27,97 điểm và R@10 giảm 23,43 điểm;
nối lại cạnh làm R@5 giảm 0,06 và R@10 giảm 0,12.

## Đọc kết quả

- **Đồ thị đúng của ảnh là thứ quyết định.** Chấm bằng đồ thị của ảnh khác, R@1 tụt 27 điểm, xuống 21,9%, thấp hơn hẳn
  CLIP thuần (39,1%).
  - Đồ thị sai không chỉ mất tác dụng mà còn kéo sai thứ hạng: dưới fusion raw, phần đồ thị chiếm trung bình 68% điểm cuối
    (44–87%) ở 15 ảnh được giải thích trong `experiments/improve/explanations/`.
  - Mức tụt lớn hơn cấp 1 (−20,2), nghĩa là mô hình mới dựa vào nội dung đồ thị của từng ảnh nhiều hơn.
- **Cách nối các quan hệ có nhãn chỉ đóng góp một phần nhỏ lúc eval.** Xáo đích các quan hệ làm R@1 giảm 0,59 điểm.
  - Mức này lớn gấp khoảng 3 lần cấp 1 (−0,19), và ổn định qua các seed (độ lệch chuẩn 0,06).
  - Nhưng so với phần mô hình mới hơn cấp 1 (+2,43 R@1 val), nó nhỏ.
  - Cạnh kNN không gian không bị xáo, nên phép thử này không trả lời được câu hỏi "GAT có cần cạnh hay không".
- **Kết hợp với `experiments/improve/explanations/`:** gỡ gộp đồ thị lúc eval không đổi R@1 (−0,02); bỏ g làm R@1 giảm 0,35.
- **Kết luận cho báo cáo:**
  - Phần tăng của mô hình cuối phụ thuộc chặt vào đồ thị đúng của từng ảnh, tức nhãn node, số lượng và box.
  - Riêng việc quan hệ nào nối với node nào thì chỉ đóng góp khoảng 0,6 điểm lúc eval.
  - Câu hỏi cạnh có cần cho mô hình hay không sẽ do ablation train lại bỏ toàn bộ cạnh (`countonly_noedges_r3`) trả lời.
