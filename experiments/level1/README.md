# Kết quả cấp độ 1 — Visual Genome ∩ COCO

Mọi số lấy từ các file `metrics_*.json` trong thư mục con, không sửa tay. Đơn vị: %.

## Baseline CLIP thuần (`clip_baseline/`)

Chạy: `python -m src.retrieval.clip_baseline`. Encoder `ViT-B-32/openai` đóng băng, không huấn luyện gì nên kết quả xác định, std = 0. Pool ứng viên là toàn bộ ảnh của split; mỗi caption là một truy vấn.

| Split | Pool ảnh | Truy vấn | R@1 | R@5 | R@10 | MRR@50 | R@50 |
|---|---|---|---|---|---|---|---|
| val | 2 126 | 10 633 | 38.00 | 66.20 | 77.19 | 51.06 | 96.59 |
| test | 2 138 | 10 696 | 38.50 | 65.77 | 76.91 | 51.22 | 96.69 |

- **R@50 là trần của bước xếp hạng lại**: đồ thị chỉ sắp lại top-50 của CLIP, nên khoảng 3,3% truy vấn (đáp án nằm ngoài top-50) không thể cứu được.
- Đã chạy lại trên máy thứ hai (RTX 3070, WSL) với cùng file đặc trưng: ra đúng các số trên.
- `top50_<split>.json` (danh sách top-50 từng truy vấn, đầu vào của bước xếp hạng lại) không commit vì nặng ~16 MB mỗi file; sinh lại bằng lệnh ở trên trong vài giây.
