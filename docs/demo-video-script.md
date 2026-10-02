# Kịch bản video demo dự phòng (≤ 5 phút)

Dùng khi demo trực tiếp lúc vấn đáp gặp sự cố. Lộc quay. Chạy trên pool **val** để phát lại đúng các ca đã phân tích
trong `experiments/level1/explanations/README.md` (val chỉ dùng để minh hoạ ở đây; số báo cáo chính lấy trên test).

```bash
python app/demo_app.py --split val        # mở http://localhost:7860, k = 12
```

Mọi kết quả dưới đây đã chạy thử trên máy RTX 4090 ngày 2026-09-26 với `main_r3_seed0`. Hạng của ảnh đúng lấy từ nút
"Caption ngẫu nhiên" hoặc từ file phân tích.

| # | Thời lượng | Thao tác | Lời dẫn (ý chính) |
|---|---|---|---|
| 0 | 0:00–0:30 | Mở trang, chỉ ô truy vấn, chế độ xếp hạng, số kết quả k. | Bài toán: nhập câu, tìm ảnh. CLIP lấy top-50, sau đó mô hình đồ thị xếp lại top-50 này bằng scene graph của từng ảnh. Kết quả kèm giải thích lấy từ chính lần xếp hạng. |
| 1 | 0:30–1:30 | Gõ `A man sitting on a bench alongside the curb of a busy street.`. Chạy **CLIP thuần**, rồi chuyển sang **CLIP + Đồ thị**. Bấm ảnh hạng 1 (`1160012`). | CLIP đưa ảnh đúng xuống hạng 6; mô hình đồ thị đưa nó lên hạng 1. Trong bảng giải thích, `bench` khớp `park bench` với a = 0,97 là cặp đóng góp lớn nhất. Nếu xoá phần này khỏi đồ thị ảnh (phép thử trong báo cáo), ảnh rơi xuống hạng 14. |
| 2 | 1:30–2:15 | Gõ `An old classic church is in front a big blue sky.`, chế độ CLIP + Đồ thị, bấm ảnh hạng 1 (`2401726`). | CLIP xếp ảnh đúng hạng 3; ảnh CLIP xếp nhất là một tháp gạch không có nhãn `church`. `church` chiếm w = 0,72 trong câu. Chỉ ra đồ thị câu do parser tách ra: `front` là lỗi parser nhưng trọng số thấp. |
| 3 | 2:15–3:05 | Gõ `a couple of airplanes lifting off the ground`, chế độ CLIP + Đồ thị. So ảnh hạng 1 và hạng 2. | Ca thất bại: CLIP đúng (hạng 1), mô hình đẩy ảnh đúng xuống hạng 2. Giải thích cho thấy `ground` quyết định (a = 0,70), và parser tách sai `couple` thành vật thể. Ảnh mô hình chọn cũng có hai máy bay, một chiếc đang cất cánh, nên câu có nhiều ảnh đúng. |
| 4 | 3:05–3:50 | Gõ câu tự do `two dogs playing with a frisbee on the beach`. Bấm ảnh hạng 1, chỉ đồ thị câu và trọng số. | Câu tự gõ cũng chạy: parser tách `dog`, `frisbee`, `beach` và hai quan hệ. Trọng số của câu này là CLIP 54%, vật thể 35%, bộ ba 11%. |
| 5 | 3:50–4:30 | Gõ `a zebroid grazing beside a flamingo statue`. | Từ lạ được mã hoá ngay bằng CLIP. Parser tách sai (`grazing` thành vật thể), và mô hình tự tăng trọng số CLIP lên 73%, nên thứ hạng giữ gần như CLIP thuần. Đây là tác dụng của trọng số a, b, c theo từng câu. |
| 6 | 4:30–5:00 | Quay lại màn hình kết quả. | Tóm tắt số test: CLIP thuần R@1 39,32; mô hình đồ thị 45,77 ± 0,23 (3 seed). Giải thích lấy từ đúng các cặp mà mô hình dùng để chấm điểm. |

Mẹo khi quay: chạy một truy vấn bất kỳ trước khi bấm ghi, để mô hình và bảng vector đã nạp xong (lần đầu mất khoảng 2 giây).
