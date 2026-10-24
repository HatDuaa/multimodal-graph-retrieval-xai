# Chuẩn bị chấm test một lần cho mô hình chính `countonly_factual_r3`

**Chưa chấm test.** Thư mục này chỉ chứa dữ liệu test đã đóng gói, các phép kiểm tra và lệnh chấm. Lệnh chấm chỉ được
chạy một lần, khi Lộc cho phép.

## Dữ liệu test (chuỗi giống hệt train/val)

Script: `scripts/improve/package_test_count_v1.py --split test`. Cấu hình đọc: `configs/count_v1_test.yaml`.

- **Hộp và synset VG của ảnh test:** `scripts/improve/build_boxes.py --split test --synsets` →
  `data/processed/geometry/objects_test.json`. Script tự đối chiếu graph dựng lại với graph cấp 1 đã đóng gói.
  2 138 ảnh, 56 381 vật, không vật nào thiếu hộp (`logs/build_boxes_test.log`).
- **Graph ảnh:** gộp trùng như merge_v1, gắn số lượng như count_v1 → `data/graphs/count_v1_test/`.
  Số node từ 56 381 xuống còn 52 510.
- **Graph câu:** FACTUAL-T5 large (đã parse ở factual_v1), gắn số lượng như group_v1 →
  `data/processed/count_v1_test/query_graphs_test.json`. Có 10 696 câu, 44,8% có ít nhất một node đã biết số lượng.
- **Ứng viên:** symlink tới top-50 CLIP cấp 1 (`candidates_test.npz`).
- **Vector part:** thêm 388 cụm bộ ba mà bảng factual_v1 chưa có, mã hoá bằng đúng CLIP đóng băng
  → `data/features/count_v1_test/`.
- Thông tin đóng gói ở `manifest_count_v1_test.json`. Tổng dung lượng khoảng 15 MB.

## Kiểm tra (không đụng tới test)

- **Chuỗi đóng gói chạy trên val** (`--split val` vào thư mục nháp; `manifest_val_chain_check.json`). Kết quả trùng
  hoàn toàn với dữ liệu mà các run đã dùng để train và chọn checkpoint:
  - graph ảnh = merge_v1;
  - nhóm số lượng = count_v1;
  - graph câu kèm số lượng = group_v1;
  - các khoá part mới nằm trọn trong phần bổ sung của merge_v1.
- **Chạy thử script chấm trên val** (`logs/dry_run_val.log`). Code giống hệt lúc chấm test:
  - CLIP thuần ra đúng baseline val cấp 1: R@1 39,13, MRR 52,34.
  - Cả 3 checkpoint cho lại đúng số trong `metrics.json`: R@1 49,15 / 48,85 / 48,74; trung bình 48,91 ± 0,21.
  - Không ghi file kết quả test nào.

## Lệnh chấm test (chạy đúng một lần, khi Lộc cho phép)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False ~/venvs/mgrx/bin/python scripts/improve/evaluate_test_countonly.py --split test
```

Script làm theo thứ tự:

1. Dừng ngay nếu đã có file kết quả.
2. Kiểm tra ứng viên test cho lại đúng baseline CLIP test cấp 1; lệch thì dừng.
3. Chấm `checkpoint_best.pt` của 3 seed. Mỗi seed dùng đúng cờ và cấu trúc graph của nó.
4. Ghi `experiments/improve/test/metrics_countonly_factual_r3.json` (từng seed và mean ± std, kèm sha256 checkpoint
   và commit) cùng `metrics_clip_only.json`.

Thời gian dự kiến khoảng 5 phút trên GPU.
