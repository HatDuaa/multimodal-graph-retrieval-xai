# Kịch bản video demo dự phòng (≤ 5 phút)

Dùng khi demo trực tiếp lúc vấn đáp gặp sự cố. Lộc quay. Chạy trên pool **val** để phát lại đúng các ca đã phân tích
trong `experiments/improve/explanations/README.md`. Val chỉ dùng để minh hoạ ở đây; số báo cáo chính lấy trên test.
Mô hình là mô hình chính `countonly_factual_r3_seed0`, mặc định của `--run`.

```bash
python app/demo_app.py --split val        # mở http://localhost:7860, k = 12
```

Mọi hạng và số dưới đây đã chạy thử qua đường demo ngày 2026-10-25 (server, pool val). Hạng của ảnh đúng lấy từ chế độ
**CLIP thuần** / **CLIP + Đồ thị**, hoặc từ file phân tích. "%" là cột "% điểm cuối" của bảng giải thích.

| # | Thời lượng | Thao tác | Lời dẫn (ý chính) |
|---|---|---|---|
| 0 | 0:00–0:30 | Mở trang, chỉ ô truy vấn, chế độ xếp hạng, số kết quả k. | Bài toán: nhập câu, tìm ảnh. CLIP lấy top-50, rồi mô hình đồ thị xếp lại top-50 này bằng scene graph của từng ảnh. Câu được tách thành đồ thị bằng FACTUAL-T5, kèm số lượng. Kết quả có giải thích lấy từ chính lần xếp hạng, và các % cộng đúng bằng 100%. |
| 1 | 0:30–1:20 | Gõ `People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella.`. Chạy **CLIP thuần**, rồi **CLIP + Đồ thị**. Bấm ảnh hạng 1 (`1159610`). | CLIP xếp ảnh đúng hạng 2; ảnh CLIP xếp nhất là ô màu chụp từ trên cao, không có chó. Mô hình đưa ảnh đúng lên hạng 1. Dòng `dog` → `dog` có a = 1,00 (10% điểm cuối); xoá node này khỏi đồ thị ảnh thì ảnh rơi xuống hạng 3 (phép thử trong báo cáo). Hai dòng có số lượng: "several umbrellas" (≥3) gặp node `umbrella` nhóm 2 nên g = −0,046; "people" (≥2) gặp `man` nhóm 3–4 nên g = +0,035. |
| 2 | 1:20–2:05 | Gõ `The two people are walking in the snow with their ski equipment.`, chế độ CLIP + Đồ thị, bấm ảnh hạng 1 (`2384371`). | Ví dụ về số lượng. Mô hình cấp 1 từng đẩy ảnh này xuống hạng 5; mô hình chính giữ nó ở hạng 1. Dòng `people` (số lượng 2) khớp `woman` (số lượng 2), g = +0,071. Bỏ g thì ảnh vẫn hạng 1, nhưng cách biệt với ảnh kế tiếp giảm từ 0,0136 xuống 0,0081. |
| 3 | 2:05–2:40 | Gõ `A snow board sitting upright with a view of a mountain.`, bấm ảnh hạng 1 (`2320074`). | Ví dụ về đồ thị gộp. Node `snowboard` mà `snow board` khớp vào (a = 0,85) được gộp từ 2 box trùng nhau của Visual Genome; trên màn hình chỉ thấy tên node. Mô hình cấp 1 xếp ảnh này hạng 4, dưới một ảnh có chiếc ván trượt mà VG cũng gắn nhãn snowboard. |
| 4 | 2:40–3:25 | Gõ `A train riding along the tracks along a bank of dirt.`. Kéo xuống ảnh đúng `2391578` (hạng 5), bấm vào. | Ca thất bại có dòng **"cả kênh"**. CLIP xếp ảnh đúng hạng 1, mô hình đẩy xuống hạng 5. Đồ thị VG của ảnh này rỗng (0 vật thể, 0 quan hệ), nên bảng chỉ có hai dòng "cả kênh": vật thể 14% và bộ ba 53%. Fusion raw lấy điểm kênh trung bình của các ứng viên cho ảnh không có phần nào để khớp. Đây là cách mô hình chấm thật, không phải lỗi hiển thị. Các ảnh tàu có đồ thị tốt vượt lên trên mức trung bình đó. |
| 5 | 3:25–4:05 | Gõ câu tự do `two dogs playing with a frisbee on the grass`. Bấm ảnh hạng 1 (`2319590`), rồi tìm ảnh `2399623` (hạng 34, cần k = 50) và bấm. | Câu tự gõ cũng chạy. Đồ thị câu: `dog` (số lượng 2), `frisbee`, `grass` và hai quan hệ. Ở ảnh hạng 1, `dog` khớp `dog` nhóm 3–4, g = +0,035. Ảnh `2399623` không có bộ ba nào, nên dòng "cả kênh" của bộ ba chiếm 67% điểm cuối. Nếu đã có cột "So với TB top-50" (nhánh `feat/demo-delta-column`), chỉ vào cột đó: bộ ba ±0, còn vật thể −0,024 mới là thứ kéo ảnh xuống. |
| 6 | 4:05–4:35 | Gõ `A lone woman huddles under her umbrella at a picnic table in the rain.`, bấm ảnh hạng 1 (`2385674`). | Hạn chế. Ảnh đúng lên hạng 1 (CLIP hạng 10). Nhưng dòng `woman` → `hair` trông sai: mọi người trong ảnh được tính chung một loại, nên node `woman` ở nhóm 3–4 bị g phạt (câu nói "lone"), còn `hair` ở nhóm 1 được thưởng +0,087. Giải thích vẫn trung thực với cách mô hình chấm, chỉ là cách chấm có lỗi. |
| 7 | 4:35–5:00 | Quay lại màn hình kết quả. | Tóm tắt số test (chấm đúng một lần): CLIP thuần R@1 39,32; mô hình cấp 1 45,77 ± 0,23; mô hình chính 47,95 ± 0,15 (3 seed). Giải thích lấy từ đúng các cặp mà mô hình dùng để chấm điểm. |

Mẹo khi quay:
- Chạy một truy vấn bất kỳ trước khi bấm ghi, để FACTUAL, mô hình và bảng vector đã nạp xong. Lần gõ đầu chậm hơn vì phải
  dựng phần đồ thị của ảnh.
- Cảnh 4 và 5 cần kéo xuống hoặc đặt k = 50 để thấy ảnh ở hạng 5 và hạng 34.
- Cần dữ liệu đồ thị val của count_v1 (`data/graphs/merge_v1/vg_coco_scene_graphs.jsonl`, `data/graphs/count_v1/count_buckets.json`,
  `data/features/merge_v1/vg_coco_graph_parts_extra.*`); xem danh sách file ở `experiments/improve/demo/README.md`.

Bản kịch bản cũ quay với `main_r3_seed0` (sng_parser, fusion z-score) nằm trong lịch sử git của file này.
