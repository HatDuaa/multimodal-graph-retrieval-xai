# Attribute probe (validation only)

Visual Genome `attributes.json.zip` đã được tải bằng `scripts/download_data.py` (83 280 561 bytes; SHA-256 `7f71c80fb5396c67a3022e0748de16a3253b7fe5759e151215e0f1a3de3f38c7`). Các graph thuộc tính được tạo cho train và val; test không được đọc hay chấm điểm.

Coverage trên 51 208 ảnh (1 368 247 object instances; thuộc tính được dựng từ annotation, không chấm điểm test): 43,56% instance có ít nhất một thuộc tính; trung bình 0,626 thuộc tính/object. Các thuộc tính thường gặp nhất: white, black, blue, green, red, brown, yellow, small, large, gray, wooden, metal, silver, grey, tall, long, dark, pink, clear, standing, here, round, tan, wood, purple, open, short, glass, big. Một số nhãn nhiễu như `here` và `standing` vẫn còn trong nguồn.

Trên query train (234 854): 19,32% object có amod và 49,57% query có ít nhất một amod. Trên val (10 633): 19,67% object có amod và 50,58% query có ít nhất một amod. Query adjectives được lấy từ modifier `amod`, bỏ determiner và số lượng.

| Variant | fixed R@1 | best R@1 (alpha,beta) | fixed R@5 | best R@5 | fixed R@10 | best R@10 | fixed MRR | best MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_none | 0.434967 | 0.434967 (0.6,0.7) | 0.709489 | 0.709489 | 0.815856 | 0.815856 | 0.561039 | 0.561039 |
| B_concat | 0.425092 | 0.426032 (0.6,0.6) | 0.698016 | 0.699708 | 0.802784 | 0.805323 | 0.550657 | 0.552292 |
| C_parts | 0.438540 | 0.443055 (0.5,0.8) | 0.716167 | 0.713533 | 0.821123 | 0.816703 | 0.565763 | 0.566132 |
| D_parts_triples | 0.425186 | 0.429418 (0.6,0.4) | 0.698956 | 0.702624 | 0.803630 | 0.809743 | 0.550995 | 0.555399 |

C_parts tăng R@1 ở best grid nhưng fixed setting chỉ là 0.438540, gain +0.003574, dưới ngưỡng adoption +0.005. Theo spec không adopt attributes; chạy các ablation `_r2` trên dữ liệu hiện tại. Giữ A_none làm đường dẫn tương thích. B_concat và D_parts_triples làm giảm R@1. Probe chưa chạy shuffled control riêng sau khi sửa cuối; đây là concern cần ghi rõ trước khi dùng kết quả trong báo cáo cuối.

Các file chính: `data/graphs/vg_coco_scene_graphs_attr.jsonl`, `data/processed/query_graphs_train_attr.json`, `data/processed/query_graphs_val_attr.json`, `scripts/build_attribute_data.py`, `scripts/checks/attribute_probe.py`, `experiments/checks/attribute_probe.json`, `experiments/checks/attribute_probe_reference_scores.npz`, `tests/test_attributes.py`.

Status: DONE_WITH_CONCERNS
