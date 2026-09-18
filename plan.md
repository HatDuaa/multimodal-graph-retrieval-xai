# Kế hoạch thực hiện — Project 3: Truy vấn đa phương thức có giải thích bằng đồ thị

Nhóm 7 · GVHD: TS. Lê Ngọc Thành · Lập ngày 2026-09-18.
Tài liệu gốc: [assignments/Project03_Multimodal_XAI.pdf](assignments/Project03_Multimodal_XAI.pdf) · Tổng quan: [README.md](README.md) · Bảng nhận việc: [task-assignment.xlsx](task-assignment.xlsx).

## 1. Mục tiêu

- **Hạn nộp + vấn đáp: 22/11/2026** → còn **9 tuần** kể từ 21/09.
- **Mốc cam kết: cấp độ 2 (trần 8.5).** Mục tiêu: **cấp độ 3 (trần 10)**. Cấp độ 4 (điểm cộng) chỉ làm khi cấp 1–3 đã chắc.
- Ba bộ dữ liệu, chung một pipeline: **Visual Genome ∩ COCO** (cấp 1), **MKG-W** (cấp 2), **MMKG tiếng Việt tự xây** (cấp 3).
- Nhánh bắt buộc: **A — text-to-image**. Metric: Recall@1/5/10, MRR; ≥3 seed, mean ± std.

## 2. Chia việc: 5 gói song song

Nguyên tắc chia: mỗi người sở hữu trọn một gói mạch lạc (giảng viên vấn đáp từng người, điểm cá nhân theo commit + bảng phân công); khối lượng xấp xỉ nhau; **tối đa việc chạy song song, tối thiểu việc chờ nhau**. Xử lý dữ liệu không tách thành gói riêng — ai tiêu thụ bộ dữ liệu nào thì tự chuẩn bị bộ đó, để ba bộ dữ liệu được chuẩn bị đồng thời.

Mỗi người tự nhận gói trong `task-assignment.xlsx` (điền "Chính" vào các dòng của gói mình). Hai dòng cấp độ 4 để trống, nhận sau khi cấp 1–3 ổn định. Báo cáo: mỗi người viết phần thuộc gói mình, gói 5 ráp và điều phối.

### Gói 1 — Hạ tầng, baseline & demo

Phần demo đã có Lộc nhận "Chính" nên gói này hợp với Lộc nhất.

| Việc | Giai đoạn |
|---|---|
| Khung repo, config theo seed, requirements, script tải dữ liệu | 0 |
| Trích đặc trưng CLIP (ảnh + text), lưu `.npy`, index FAISS | 0 |
| Join `coco_id`, tập con cố định, **split Karpathy, phát hành file split** | 1 |
| Baseline CLIP thuần: caption → top-k | 1 |
| Platform demo web app: bản CLIP thuần trước, cắm đồ thị của gói 2 sau | 1 |
| Demo trực tiếp khi vấn đáp + video dự phòng ≤5 phút | 5 |

### Gói 2 — Đồ thị & GNN re-ranking

Nặng kỹ thuật nhất, không gánh thêm việc khác.

| Việc | Giai đoạn |
|---|---|
| Dựng đồ thị: lọc VG150, node ảnh + khái niệm, cạnh chứa/quan hệ + cạnh ConceptNet | 1 |
| Nối truy vấn vào đồ thị: tách danh/động từ, khớp node khái niệm | 1 |
| GAT + điểm đồ thị cho từng ứng viên top-k | 1 |
| Re-rank α·cosine + (1−α)·điểm đồ thị, chọn α trên validation | 1 |
| Ablation: bỏ cạnh quan hệ, xáo cạnh, đồ thị ngẫu nhiên | 1 |
| Chạy lại CLIP thuần + CLIP + GNN trên MKG-W | 2 |

### Gói 3 — Tái lập NativE / AdaMF-MAT

Độc lập gần như hoàn toàn; ít dòng nhưng rủi ro tái lập cao nhất.

| Việc | Giai đoạn |
|---|---|
| Dữ liệu MKG-W: kiểm tra ảnh + mô tả, sinh truy vấn từ mô tả che tên, chia theo split | 2 |
| Tái lập bước a: link prediction gốc, so với số trong bài báo | 2 |
| Tái lập bước b: embedding thực thể thay GNN trong điểm đồ thị | 2 |
| Mở rộng: độ nhạy theo α, so các cách tạo cạnh | 2 |

### Gói 4 — Đánh giá & xAI

Người "giữ chuẩn" đo lường của cả nhóm.

| Việc | Giai đoạn |
|---|---|
| Module đánh giá: Recall@1/5/10, MRR, mean ± std qua ≥3 seed | 0 |
| Giải thích cơ bản: trích cạnh/đường đi đóng góp vào điểm đồ thị (thiết kế chung với gói 2) | 1 |
| Phân tích 5 truy vấn thành công + 5 thất bại | 1 |
| Soạn bộ ≥20 truy vấn mỗi tập (trực tiếp, gián tiếp, đảo quan hệ) + cách xây, kiểm tra | 2 |
| Báo cáo kết quả tách nhóm trực tiếp / gián tiếp | 2 |
| Chốt định nghĩa fidelity / validity / sparsity | 3 |
| Cài đặt fidelity (bỏ cạnh giải thích so với bỏ cạnh ngẫu nhiên) và sparsity | 3 |
| Chấm validity trên ≥10 truy vấn | 3 |

### Gói 5 — MMKG tiếng Việt & sản phẩm nộp

Độc lập, nặng công thủ công trải đều theo thời gian.

| Việc | Giai đoạn |
|---|---|
| Thử CLIP đa ngữ với truy vấn tiếng Việt, chốt encoder cho cả project | 0 |
| MMKG tiếng Việt: chọn miền, thu thập thực thể / triple / ảnh, liên kết ảnh–thực thể | 3 |
| MMKG tiếng Việt: soạn truy vấn tiếng Việt, viết data card | 3 |
| Chạy phương pháp tốt nhất trên MMKG tiếng Việt, phân tích chuyển giao | 3 |
| Điều phối báo cáo kỹ thuật 12–18 trang | 5 |
| Slide trình bày ≤20 phút | 5 |
| README chạy lại, mô tả chống rò rỉ, khai báo dùng AI / mã nguồn mở, bảng phân công cuối | 5 |

## 3. Phụ thuộc và song song

Tuần đầu, cả 5 gói đều có việc bắt đầu ngay, không ai chờ ai:

```
Gói 1: tải VG+COCO → join, split ──► trích CLIP ──► baseline ──► demo
Gói 2: tải VG thô → lọc VG150 ────► dựng đồ thị ──► GNN ──► re-rank
Gói 3: tải MKG-W → sinh truy vấn ─► NativE bước a ──► bước b
Gói 4: module đánh giá + nháp 20 truy vấn ──► metric xAI
Gói 5: chốt encoder ──► thu thập MMKG tiếng Việt ──► data card
```

Các điểm chờ bất khả kháng — đều có việc đệm trong lúc chờ:

| Ai chờ | Chờ cái gì | Việc đệm |
|---|---|---|
| Gói 2 (huấn luyện GNN) | đặc trưng CLIP + file split của gói 1 | dựng đồ thị, nối truy vấn |
| Gói 3 (bước b) | interface re-rank của gói 2 | bước a (link prediction) |
| Gói 4 (fidelity) | cơ chế giải thích của gói 2 | định nghĩa metric, bộ 20 truy vấn |
| Gói 1 (demo phần graph) | điểm đồ thị của gói 2 | demo bản CLIP thuần |
| Gói 5 (chuyển giao) | phương pháp tốt nhất từ cấp 1/2 | thu thập dữ liệu, data card |

### Hợp đồng interface — chốt trong buổi họp tuần 1

Bốn format thống nhất trước khi tách ra làm, để gói 3, 4 mock đầu vào mà không đợi gói 1, 2:

1. **File split** train/validation/test (gói 1 phát hành, các gói khác chỉ đọc).
2. **Đặc trưng CLIP**: quy ước tên file `.npy` + index FAISS.
3. **Đồ thị**: format danh sách node / cạnh (loại node, loại cạnh, trọng số).
4. **API điểm đồ thị**: `score(query, candidate) → (điểm, đường đi giải thích)`.

### Ràng buộc phối hợp

- **Gói 2 + gói 4 thiết kế chung cơ chế giải thích ngay từ đầu**: giải thích phải sinh từ chính cơ chế xếp hạng (attention trên cạnh hoặc đường đi trong điểm đồ thị). Đề cấm tìm đường đi hậu kiểm sau khi có kết quả.
- **File split là nguồn chân lý duy nhất** (gói 1 giữ): mọi mô hình so sánh dùng cùng một file; cạnh thống kê chỉ xây từ train; α và siêu tham số chọn trên validation; bộ 20 truy vấn chỉ để đánh giá, không tinh chỉnh (Mục 6 đề bài — tiêu chí chấm).
- Kết quả thực nghiệm phải có log + file kết quả trong `experiments/` kèm seed. Không bịa số liệu.
- Dữ liệu lớn không commit; link tải + script ghi vào `data/README.md`.

## 4. Lộ trình 9 tuần (21/09 → hạn nộp 22/11/2026)

| Tuần | Ngày | Mốc | Điều kiện đạt |
|---|---|---|---|
| 1 | 21–27/09 | Họp nhận gói + chốt 4 hợp đồng interface; tải xong VG∩COCO và MKG-W; **gói 1 phát hành file split**; gói 2+4 chốt thiết kế cơ chế giải thích trên giấy | Cả 5 gói bắt đầu được việc chính |
| 2 | 28/09–04/10 | Baseline CLIP chạy được (Recall/MRR trên VG∩COCO); đồ thị VG dựng xong; NativE/AdaMF bước a chạy được bản đầu; chốt miền MMKG Việt; nháp bộ 20 truy vấn | Số baseline đầu tiên vào `experiments/` |
| 3–4 | 05–18/10 | **Hoàn thành cấp độ 1 (18/10)**: GNN + re-rank + ablation + giải thích cơ bản, 3 seed; demo bản CLIP thuần chạy được | Bảng CLIP vs CLIP + Graph, mean ± std |
| 5–6 | 19/10–01/11 | **Hoàn thành cấp độ 2 (01/11)**: NativE bước b; CLIP + GNN trên MKG-W; bộ ≥20 truy vấn cả 2 tập; kết quả tách trực tiếp / gián tiếp; demo cắm đồ thị. Song song: gói 4 cài fidelity/sparsity, gói 5 hoàn tất thu thập MMKG Việt | Đủ 3 hàng so sánh trên MKG-W |
| 7–8 | 02–15/11 | **Hoàn thành cấp độ 3 (12/11)**: fidelity / validity / sparsity trên ≥10 truy vấn; MMKG Việt + data card + chạy chuyển giao. **Đóng băng thực nghiệm 15/11** — sau ngày này chỉ viết, không đổi số | Bảng metric xAI; MMKG đạt quy mô đề yêu cầu |
| 8–9 | 09–22/11 | Báo cáo 12–18 trang (viết song song từ 09/11, mỗi người phần gói mình), slide, video dự phòng, tổng duyệt demo + vấn đáp thử (~19–20/11). **Nộp 22/11** | Mỗi người trả lời được phần của gói mình |

Với 9 tuần, **cấp độ 4 không nằm trong lộ trình chính** — chỉ nhận nếu trước 08/11 mà cấp 1–3 đã xong sớm (thực tế khó xảy ra); ưu tiên dồn sức cho fidelity/validity/sparsity và MMKG Việt.

Kiểm soát tiến độ: cập nhật cột Trạng thái trong `task-assignment.xlsx` mỗi tuần, họp ngắn đầu tuần soát điểm chờ. **Mốc quyết định là 18/10 (cấp độ 1)**: nếu trễ quá 1 tuần, cắt phần mở rộng của gói 3 (độ nhạy α, so cách tạo cạnh) và ô "phương pháp mới trên VG", giữ nguyên cấp độ 3; nếu trễ quá 2 tuần, hạ mục tiêu MMKG Việt xuống quy mô tối thiểu của đề (50 thực thể, 300 triple, 200 ảnh).

## 5. Rủi ro chính và phương án

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| CLIP vốn mạnh trên COCO → đồ thị không cải thiện truy vấn trực tiếp | Kết quả cấp 1 "âm" | Soạn sớm nhóm truy vấn gián tiếp (gói 4, ngay tuần 2) để chứng minh giá trị đồ thị ở đúng chỗ đề nhấn mạnh; kết quả âm có phân tích vẫn chấp nhận được |
| Code NativE / AdaMF không chạy lại được | Mất điều kiện cấp 2 | Gói 3 thử cả hai từ tuần 1, chốt theo cái chạy được trước; phương án cuối: Mixed-Curvature MMKGC |
| Cơ chế giải thích thiết kế sai (hậu kiểm) | Vi phạm yêu cầu xuyên suốt của đề | Gói 2 + 4 chốt thiết kế trên giấy tuần 1–2, hỏi giảng viên nếu chưa chắc |
| MMKG tiếng Việt tốn công hơn dự kiến | Trễ cấp 3 | Bắt đầu thu thập từ tuần 1; chốt miền hẹp, ưu tiên nguồn có sẵn giấy phép (Wikidata, Wikimedia Commons) |
| Thành viên nghẽn việc riêng | Gói bị treo | Mỗi gói ghi rõ "Hỗ trợ" trong xlsx; họp ngắn hằng tuần soát điểm chờ |
