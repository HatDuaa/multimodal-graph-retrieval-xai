# Test — chấm một lần cho mô hình chính của phase cải tiến

Ngày 25/10/2026, lúc 00:09, Lộc duyệt (qua KG) cho chấm test **đúng một lần**. Mô hình được chấm là
`countonly_factual_r3` (`../count-only/`). Lệnh đã chạy, không lỗi, không chạy lại:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False ~/venvs/mgrx/bin/python scripts/improve/evaluate_test_countonly.py --split test
```

- Code ở commit `8806b52`. Dữ liệu test đóng gói theo chuỗi count_v1 (`../test-prep/README.md`, `configs/count_v1_test.yaml`).
- Trước khi chấm, script kiểm tra danh sách ứng viên test: CLIP thuần cho lại đúng baseline test cấp 1
  (`experiments/level1/clip_baseline/metrics_test.json`, R@1 39,32).
- Mỗi seed dùng `checkpoint_best.pt` đã chọn theo val R@1: seed 0 ở epoch 8, seed 1 ở epoch 9, seed 2 ở epoch 5.
  sha256 của từng checkpoint ghi trong file kết quả.

## Kết quả (10 696 câu test, top-50 của CLIP)

| Mô hình | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP thuần (`metrics_clip_only.json`) | 39.32 | 66.39 | 77.32 | 52.00 |
| main_r3, cấp 1 (`experiments/level1/test/metrics_main_r3.json`) | 45.77 ± 0.23 | 72.87 ± 0.31 | 82.69 ± 0.31 | 58.14 ± 0.18 |
| **countonly_factual_r3** (`metrics_countonly_factual_r3.json`) | **47.95 ± 0.15** | **74.79 ± 0.03** | **84.38 ± 0.08** | **60.15 ± 0.15** |

Từng seed của countonly:

| Seed | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| 0 | 48.06 | 74.76 | 84.46 | 60.32 |
| 1 | 47.78 | 74.80 | 84.29 | 60.05 |
| 2 | 47.99 | 74.80 | 84.39 | 60.07 |

So với main_r3 cấp 1 (Welch, 3 seed mỗi bên):

| Chỉ số | Chênh | p |
|---|---|---|
| R@1 | +2,18 | 0,0005 |
| R@5 | +1,91 | 0,008 |
| R@10 | +1,70 | 0,008 |
| MRR | +2,00 | 0,0002 |

So với CLIP thuần, R@1 cao hơn +8,63.

- **Chênh val → test:** R@1 của countonly giảm từ 48,91 (val) xuống 47,95 (test), tức −0,96. main_r3 giảm từ 46,48
  xuống 45,77, tức −0,71.
- Khoảng chênh lớn hơn một chút là hợp lý: countonly được chọn sau nhiều thí nghiệm, đều dựa trên val, nên val hơi lạc
  quan.
