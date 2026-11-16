# Tương quan giữa các kênh theo từng câu (val, không train)

Script: `scripts/improve/channel_correlation.py`. Lệnh chạy lại: `run.sh` (khoảng 20 giây trên RTX 4090).
Kết quả mỗi run: `channel_correlation.json`.

Với mỗi câu val, script tính hệ số tương quan Pearson giữa điểm thô của hai kênh trên top-50 ứng viên. Chỉ tính
những ứng viên mà cả hai kênh đều có điểm, chỉ những câu mà cả hai kênh cùng hoạt động, và cần ít nhất 3 ứng viên.
Z-score là phép biến đổi affine theo từng câu, nên đây cũng là tương quan giữa các z-score mà `fuse_tensor` trộn
với nhau. Kênh 3 là kênh bộ ba (main_r3, no_gat_r3, factorized) hoặc kênh mức đồ thị (graph_channel). Mỗi ô là
mean ± std qua 3 seed của tương quan trung bình theo câu.

| Mô hình | vật thể ~ kênh 3 | CLIP ~ kênh 3 | CLIP ~ vật thể | tỉ lệ câu có kênh 3 hoạt động |
|---|---|---|---|---|
| main_r3 | 0.197 ± 0.009 | 0.140 ± 0.005 | 0.315 ± 0.004 | 85.8% |
| no_gat_r3 | 0.209 ± 0.033 | 0.153 ± 0.013 | 0.323 ± 0.004 | 85.8% |
| factorized | 0.587 ± 0.004 | 0.247 ± 0.003 | 0.315 ± 0.007 | 85.8% |
| graph_channel | 0.518 ± 0.023 | 0.229 ± 0.010 | 0.311 ± 0.001 | 100.0% |

**Kết luận.** Ở main_r3, kênh bộ ba dạng cụm tuy yếu khi đứng riêng (R@1 6,24) nhưng chỉ tương quan 0,20 với kênh
vật thể, nên nó mang tín hiệu khác và vẫn giúp điểm cuối thêm 0,49 điểm R@1. Kênh factorized (0,59) và kênh mức đồ thị
(0,52) mạnh hơn nhiều khi đứng riêng (R@1 17,6 và 15,4), nhưng tương quan với kênh vật thể gấp 2,5–3 lần. Lý do:
cả hai đều dựng từ chính các vector node mà kênh vật thể dùng. Sau z-score, phần lớn thông tin của chúng đã có sẵn
trong kênh vật thể. Kết quả này xác nhận giả thuyết ở `../triple-factorized/REPORT.md`, mục 4.
