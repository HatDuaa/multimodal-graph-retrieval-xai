# Quy tắc làm việc — multimodal-graph-retrieval-xai

Project cuối kì môn Học máy với dữ liệu đồ thị. Đề đã chọn: **Project 3 — Truy vấn đa phương thức có giải thích bằng đồ thị**. Đọc `README.md` trước khi làm bất cứ việc gì; đề gốc ở `assignments/Project03_Multimodal_XAI.pdf` là nguồn chuẩn cho mọi yêu cầu.

## Nhóm

- 25C11050 Nguyễn Đình Lộc (người đang làm việc với Claude).
- 25C15003 Ngô Trương Minh Đạt.
- 25C15015 Trần Đắc Khoa.
- 25C15055 Mẫn Nhi — chưa có họ tên đầy đủ.
- 25C15045 Dương Khang — chưa có họ tên đầy đủ.

Nhóm số 7 trong danh sách đăng ký. Giảng viên hướng dẫn: TS. Lê Ngọc Thành.

Phân công chưa chốt: mỗi người tự chọn việc trong `task-assignment.xlsx`. Không tự gán tên ai vào việc nào khi chưa được nói rõ. Khi tạo code hoặc tài liệu, ghi rõ phần nào thuộc ai để làm bảng phân công.

## Quy tắc chung

- **Tên folder, tên file, code, comment, tên biến, commit message: tiếng Anh** (kebab-case cho folder và file markdown/script, snake_case cho Python).
- **Nội dung** README, báo cáo, slide, ghi chú: **tiếng Việt**, giữ thuật ngữ tiếng Anh kèm giải thích lần đầu.
- Không sửa `assignments/`.
- Không bịa số liệu. Mọi kết quả phải có log hoặc file kết quả trong `experiments/` kèm seed.
- Chống rò rỉ dữ liệu là tiêu chí chấm (Mục 6 đề bài): cạnh thống kê, lớp chiếu, siêu tham số, α chỉ từ train/validation. Bộ 20 truy vấn đánh giá không dùng để tinh chỉnh. Mọi mô hình so sánh dùng cùng một tệp chia.
- Thực nghiệm chính chạy ≥3 seed, báo cáo mean ± std.
- Giải thích phải sinh từ chính cơ chế xếp hạng. Không chấp nhận cách tìm đường đi sau khi đã có kết quả.
- Dùng code mở hoặc AI phải khai báo trong báo cáo, ghi rõ phần đã sửa.
- Dữ liệu lớn không commit; ghi link tải và script tải vào `data/README.md`.

## Liên hệ với seminar

Repo này tách từ `HatDuaa/DLDT` (chứa seminar MMKGC của Lộc và Đạt: hai bài báo HFR-MKGC, MDBGF). Được dùng lại kiến thức nền và cùng dataset MKG-W, nhưng project này là bài toán **retrieval**, không phải KGC. Không copy kết quả từ seminar sang. Repo này không phụ thuộc đường dẫn nào ngoài chính nó.
