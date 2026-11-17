# Kết quả scene ranker từ repo cá nhân

File chép nguyên từ `artifacts/runs/` của repo cá nhân `nhinbm/multimodal-graph-retrieval-xai` (commit `cdb4b5b`), không sửa tay.
Mô hình, train và test đều chạy ở repo đó; báo cáo ở `report/scene-ranker.md`.

| File | Run | Nội dung |
|---|---|---|
| `rank_train_0056_metrics.json`, `_0057_`, `_0063_` | rank-train seed 0, 1, 2 | Metric val theo epoch, epoch chọn, kiểm tra giải thích và phép thử xoá |
| `rank_train_0065_metrics.json` | rank-train seed 0, không cạnh | Train lại không có cạnh concept và cạnh scene graph |
| `rank_ablation_0058_metrics.json`, `_0062_`, `_0064_` | rank-ablation trên 0056, 0057, 0063 | Ablation lúc đánh giá trên val |
| `test_eval_0069_val_metrics.json`, `test_eval_0069_val_ranks.npz` | test-eval trên val | Chạy lại toàn chuỗi trên val; mọi kiểm tra tái lập đạt trước khi mở test |
| `test_eval_0070_metrics.json`, `test_eval_0070_test_ranks.npz`, `test_eval_0070_config.json` | test-eval trên test | Chấm test đúng một lần: CLIP, B1, B2 và 3 seed, cùng cấu hình |
| `comparison_test.json` | | So với `countonly_factual_r3` trên test, sinh bằng `python scripts/improve/compare_scene_ranker.py` |

Hạng trong các file `.npz` là hạng trên toàn gallery (1 là tốt nhất) của từng caption, theo thứ tự `caption_ids`.
