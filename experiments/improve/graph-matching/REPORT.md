# Thí nghiệm 2 — Ghép cả đồ thị câu với đồ thị ảnh (probe không train, val)

Lệnh chạy lại: `run.sh` (chỉ dùng CPU, `nice`, 12 tiến trình, chạy song song với job train trên GPU). Code:
`src/graph/graph_matching.py` (test: `tests/test_graph_matching.py`), `scripts/improve/graph_matching_{probe,eval,examples}.py`.
Kết quả: `graph_matching.json`, `gold_vs_others.json`. Ví dụ giải thích: `examples.md`. Dữ liệu `factual_v1`; mốc so
sánh là main_factual_r3 (3 seed). Điểm và vector đã train của mốc được lưu sẵn bằng
`scripts/improve/dump_val_scores.py` (một job GPU 14 giây ở đầu chuỗi `../fusion/run.sh`).

## Cách làm

Đồ thị câu lấy đúng như mô hình xếp hạng nhìn thấy: các node chấm được (không đại từ, không nhãn rỗng) và các quan hệ
giữa chúng. Hai họ kênh, mỗi kênh thử với vector CLIP đóng băng và với vector node/bộ ba đã train của main_factual_r3
(kênh từ seed k đi với điểm của seed k):

- **2a, FGW** (POT 0.9.7; cài kèm scipy 1.18.1, torch không đổi).
  - Chi phí node: 1 − cosine. Cấu trúc: số bước đường đi ngắn nhất trên đồ thị quan hệ vô hướng, chặn ở 3 (cặp không
    nối nhau cũng nhận 3), chia cho 3. α = 0,5. α và mức chặn chọn trước, không dò trên val.
  - "Cân bằng": FGW chuẩn, hai phân phối đều đều.
  - "Nửa nới lỏng": semi-relaxed FGW (Vincent-Cuaz và cs., ICLR 2022), chỉ ràng buộc phía câu. Đồ thị câu khoảng 3
    node phải ghép vào một phần của đồ thị ảnh khoảng 26 node; FGW cân bằng lại buộc trải khối lượng lên mọi node ảnh.
  - Điểm = −khoảng cách. Phương án vận chuyển cho biết node câu nào ghép với node ảnh nào.
- **2b, nhất quán node chung.** Với mỗi cặp bộ ba trong câu có chung một node, lấy bộ ba ảnh khớp nhất (cosine lớn
  nhất) cho từng bộ ba, rồi kiểm tra node chung có rơi vào cùng một node ảnh không. Điểm = tỉ lệ cặp nhất quán.
- **Tập con:** câu có ít nhất 2 bộ ba chung node, gồm 6 319/10 633 câu.
- **Kênh 4:** điểm main_factual_r3 + w·z(kênh). w được chọn trên val; bản 2-fold chọn w trên một nửa val (chia theo
  ảnh đúng) rồi chấm nửa kia.
- **"Xếp riêng":** xếp top-50 chỉ theo kênh. Câu không có điểm xếp cuối; hoà điểm thì giữ thứ tự CLIP, hoặc tách hoà
  ngẫu nhiên (seed 0). Bản ngẫu nhiên loại được tín hiệu CLIP ẩn trong các kênh ít giá trị như kênh nhất quán. Mức
  ngẫu nhiên là 2%.

## Kết quả

Mốc main_factual_r3 (3 seed): toàn val R@1 47.23 ± 0.36, MRR 59.49 ± 0.29; tập con (n = 6319) R@1 50.72 ± 0.43, MRR 62.49 ± 0.34.

| Kênh | xếp riêng, R@1 tập con (hoà theo CLIP / hoà ngẫu nhiên) | tương quan với kênh vật thể | w tốt nhất | + kênh 4, R@1 toàn val | + kênh 4, R@1 tập con | 2-fold, R@1 toàn val | 2-fold, R@1 tập con |
|---|---|---|---|---|---|---|---|
| FGW cân bằng, CLIP đóng băng | 2.29 / 2.29 | -0.041 | 0.0 | 47.23 ± 0.36 | 50.72 ± 0.43 | 47.23 ± 0.36 | 50.72 ± 0.43 |
| FGW cân bằng, vector đã train (seed k với seed k) | 4.26 ± 0.55 / 4.26 ± 0.55 | 0.233 ± 0.047 | 0.0, 0.05, 0.0 | 47.27 ± 0.33 | 50.73 ± 0.42 | 47.20 ± 0.40 | 50.70 ± 0.44 |
| FGW nửa nới lỏng, CLIP đóng băng | 15.90 / 15.86 | 0.288 | 0.05 | 47.29 ± 0.24 | 50.77 ± 0.35 | 47.26 ± 0.24 | 50.71 ± 0.40 |
| FGW nửa nới lỏng, vector đã train (seed k với seed k) | 25.55 ± 1.91 / 25.55 ± 1.91 | 0.721 ± 0.051 | 0.0, 0.1, 0.05 | 47.35 ± 0.34 | 50.82 ± 0.52 | 47.32 ± 0.39 | 50.78 ± 0.56 |
| nhất quán node chung, CLIP đóng băng | 17.80 / 2.23 | -0.061 | 0.0 | 47.23 ± 0.36 | 50.72 ± 0.43 | 47.23 ± 0.36 | 50.72 ± 0.43 |
| nhất quán node chung, vector đã train (seed k với seed k) | 17.81 ± 1.03 / 2.72 ± 0.30 | -0.050 ± 0.003 | 0.0, 0.0, 0.0 | 47.23 ± 0.36 | 50.72 ± 0.43 | 47.21 ± 0.32 | 50.68 ± 0.36 |

Với các hàng vector đã train, mỗi ô là mean ± std qua 3 seed, mỗi seed dùng w tốt nhất của chính nó. Các hàng CLIP
đóng băng dùng chung một w cho cả 3 seed.

Điểm của ảnh đúng so với 49 ảnh còn lại trên tập con (`gold_vs_others.json`, CLIP đóng băng):

- **FGW nửa nới lỏng:** ảnh đúng −0,073, ảnh khác −0,092. Ở 73% số câu, ảnh đúng cao hơn trung bình các ảnh khác.
- **Nhất quán node chung:** ảnh đúng 0,369, ảnh khác 0,379. Tỉ lệ ảnh nhất quán hoàn toàn là 31,9% ở ảnh đúng và
  31,2% ở ảnh khác.

## Nhận xét

1. **Không biến thể nào giúp điểm cuối.**
   - Cộng làm kênh 4, mức tăng lớn nhất là +0,12 R@1 toàn val (FGW nửa nới lỏng, vector đã train, w chọn trên val) và
     +0,09 khi chọn 2-fold, so với std giữa các seed là 0,36.
   - Trên tập con có bộ ba chung node, mức tăng tối đa +0,10, 2-fold +0,06.
2. **FGW cân bằng không dùng được.** Xếp riêng chỉ đạt 2,3–4,3%, gần mức ngẫu nhiên. Ép khối lượng phủ mọi node
   ảnh làm điểm phản ánh cả những vật không liên quan tới câu.
3. **FGW nửa nới lỏng có thông tin nhưng phần lớn trùng kênh vật thể.**
   - Với CLIP đóng băng: xếp riêng 15,9%, tương quan với kênh vật thể 0,29.
   - Với vector đã train: xếp riêng 25,6%, nhưng tương quan lên 0,72. Phần thông tin mới so với kênh vật thể quá ít
     để cải thiện xếp hạng.
4. **Nhất quán node chung độc lập với kênh vật thể (tương quan −0,05) nhưng không có tín hiệu.**
   - Tách hoà ngẫu nhiên, xếp riêng chỉ còn 2,2–2,7%. Con số 17,8% khi giữ thứ tự CLIP là tín hiệu của CLIP.
   - Ảnh đúng không nhất quán hơn ảnh sai (0,369 so với 0,379).
   - Lý do có thể: đồ thị VG thường không nối các quan hệ qua cùng một instance; người gán nhãn hay tạo object mới cho
     mỗi quan hệ, như các "horse" lặp ở ảnh 61513. Vì vậy cấu trúc nối của câu không tìm thấy tương ứng trong ảnh.
5. **Giá trị giải thích.** Phương án vận chuyển của FGW nửa nới lỏng cho lời giải thích ghép node dễ đọc, gồm cả các
   cặp đồng nghĩa như "lady → woman", "person → boy", "package → box" (`examples.md`). Đây là lời giải thích sinh từ
   chính phép ghép. Nếu dùng nó để giải thích thì nên gắn với kênh có trong điểm xếp hạng; hiện kênh này không được
   đưa vào điểm.
