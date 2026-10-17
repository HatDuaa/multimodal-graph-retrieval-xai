# Phase cải tiến — kênh bộ ba

Mọi số trong thư mục này là của **val**. Test chưa được chạm. Mỗi thí nghiệm chỉ đổi đúng một thứ so với `main_r3`
(`experiments/level1/gat/main_r3_seed*`) và chạy 3 seed (0, 1, 2). Số được tổng hợp bằng `src/eval/metrics.py`
(mean ± std, ddof = 1).

| Thư mục | Nội dung |
|---|---|
| `triple-perturbation/` | Bước 1: đổi động từ hoặc đổi chủ ngữ/tân ngữ phía câu, trên các checkpoint cấp 1 (không train) |
| `microbatch-control/` | Đối chứng cho việc giảm micro-batch 128 → 32 (xem mục Hạ tầng) |
| `triple-factorized/` | Bước 2, thí nghiệm A: so khớp bộ ba tách theo thành phần, `sim = c(s)·c(p)·c(o)` |
| `graph-channel/` | Bước 3, thí nghiệm B: thay kênh bộ ba bằng kênh mức đồ thị (attention pooling + cosine) |

Mỗi thư mục có `run.sh` (lệnh chạy lại), `REPORT.md` (kết quả và nhận xét), log và file kết quả. Checkpoint `*.pt`
không commit (`.gitignore`). Ở mỗi run, `config.json` lưu đủ cờ, seed và commit.

## Hạ tầng (máy RTX 4090, 18/10/2026)

- **NVML.** Driver NVIDIA tự cập nhật ngày 15/10 mà chưa reboot: kernel module 595.84, thư viện 595.99. Vì vậy
  NVML không khởi tạo được. Chế độ `expandable_segments:True` mà script train đặt mặc định lại gọi NVML, nên mọi
  job CUDA đều chết. Mọi lệnh ở đây chạy với `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Biến này chỉ đổi
  cách cấp phát bộ nhớ, không đổi phép tính. Kiểm tra: chấm lại `main_r3_seed0` ra R@1 val 0,46666039687764505,
  trùng từng chữ số với `metrics.json` cấp 1.
- **Micro-batch.** Không có `expandable_segments` thì bộ nhớ cache phân mảnh. Ngay công thức main_r3 ở
  micro-batch 128 đã giữ tới 22,9/24 GB (bảng bộ nhớ trong `experiments/level1/README.md`), và bản factorized tràn
  bộ nhớ ở batch đầu tiên (`triple-factorized/archive-oom-mb128/`). Vì thế mọi run cải tiến dùng micro-batch 32,
  batch hiệu dụng vẫn 256 (gradient cộng dồn đúng tỉ lệ). Mô hình không có batch-norm, z-score và loss tính theo
  từng câu, nên gradient mỗi bước tối ưu chỉ khác ở thứ tự cộng số thực và ở cách rút mặt nạ dropout.
  `microbatch-control/` kiểm tra lại điều này trên công thức main_r3.
