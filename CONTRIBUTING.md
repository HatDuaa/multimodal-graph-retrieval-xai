# Quy ước làm việc nhóm

Giảng viên vấn đáp từng thành viên và xem lịch sử commit để chấm đóng góp cá nhân. Vì vậy:

- **Ai làm phần nào tự commit phần đó**, bằng tài khoản GitHub của mình. Không commit hộ.
- Nhận việc trong `task-assignment.xlsx` trước khi bắt đầu; cập nhật cột Trạng thái mỗi tuần.

## Nhánh và merge

- `main` luôn chạy được. Không push thẳng code lên `main`; tài liệu nhỏ (README, bảng phân công) thì được.
- Mỗi việc một nhánh: `<tên>/<việc>`, ví dụ `dat/gat-ranker`, `nhi/query-set`.
- Merge qua pull request, cần ít nhất một người khác xem. Trước khi mở PR: `pytest` phải xanh.

## Commit

- Tiếng Anh, dạng conventional commit: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `exp:` (kết quả thực nghiệm).
- Không commit dữ liệu lớn, checkpoint, khoá API, file `.env`.

## Code

- Tên folder, file, biến, comment: tiếng Anh. README, báo cáo, ghi chú: tiếng Việt.
- Đường dẫn và siêu tham số đọc từ `configs/default.yaml`, không viết cứng trong script.
- Mọi thực nghiệm gọi `set_seed()` và ghi seed vào kết quả. Thực nghiệm chính chạy đủ các seed trong `configs/default.yaml`.
- Kết quả ghi vào `experiments/<cấp độ>/<tên>/` kèm config đã dùng. Không sửa tay số liệu.

## Chống rò rỉ dữ liệu (Mục 6 đề bài, có chấm điểm)

- File split trong `data/splits/` là nguồn chân lý duy nhất; mọi mô hình so sánh đọc cùng file đó.
- Cạnh đồ thị suy từ thống kê chỉ xây từ tập train.
- α và mọi siêu tham số chọn trên validation. Test chỉ chạy một lần cho mỗi cấu hình đã chốt.
- Bộ ≥20 truy vấn của cấp độ 2 chỉ dùng để đánh giá.

## Khai báo công cụ

Dùng mã nguồn mở hoặc công cụ AI thì ghi lại (phần nào, đã sửa gì) để đưa vào báo cáo; mỗi người phải tự giải thích được phần mình commit.
