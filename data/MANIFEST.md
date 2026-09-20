# Manifest dữ liệu

Mỗi file dùng chung ghi một dòng: tên, kích thước, SHA-256, ngày, script sinh ra. Trước khi chạy thực nghiệm, đối chiếu checksum để chắc cả nhóm dùng cùng một bản.

## File tải về (`scripts/download_data.py`)

Tải ngày 2026-09-18.

| File | Bytes | SHA-256 |
|---|---|---|
| `data/raw/visual_genome/image_data.json.zip` | 1780854 | `b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a` |
| `data/raw/visual_genome/objects.json.zip` | 55323929 | `efa456069aedb420b7dabf54f4608c89c78a00de550f1433066e9b6f84fb629c` |
| `data/raw/visual_genome/relationships.json.zip` | 77904473 | `e648867b8087e4aeb10019a959f914e2580909f1c3e50a21b25de8741679182f` |
| `data/raw/visual_genome/object_alias.txt` | 60166 | `0c8e059fc31eeebfd98231f5789892da8ae33bfa00434c70aee969dc6eaa853b` |
| `data/raw/visual_genome/relationship_alias.txt` | 122102 | `15f7f64802c95c5bf5b5690457566b1b19ee64eac7a94d8b8cbb3a4378f2ec7c` |
| `data/raw/vg150/object_list.txt` | 885 | `7f0824cff05b721ff3dd52114c88b683082a4628b3152fb7c222853a9d262404` |
| `data/raw/vg150/predicate_list.txt` | 393 | `d455d4dd841b94e42bfc96bcf7c040ec0efd905e85f51cd7649ab4791e8c9aeb` |
| `data/raw/vg150/object_alias.txt` | 60166 | `0c8e059fc31eeebfd98231f5789892da8ae33bfa00434c70aee969dc6eaa853b` |
| `data/raw/vg150/predicate_alias.txt` | 122102 | `15f7f64802c95c5bf5b5690457566b1b19ee64eac7a94d8b8cbb3a4378f2ec7c` |
| `data/raw/coco/caption_datasets.zip` | 36745453 | `4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817` |

Bốn file `vg150/` là danh sách 150 lớp đối tượng và 50 quan hệ (VG150, Xu và cộng sự 2017) lấy từ repo `danfeiX/scene-graph-TF-release`, tải ngày 2026-09-19. Hai file alias ở đó trùng từng byte với hai file alias của Visual Genome (cùng SHA-256).

Sau khi giải nén, `data/raw/` chiếm khoảng 1,4 GB.

Số đếm thật từ các file này (in ra bởi `scripts/smoke_test.py`): Visual Genome có 108 077 ảnh; 51 498 ảnh có `coco_id` nằm trong file split Karpathy, gồm train 34 027, restval 13 183, val 2 146, test 2 142.

## File nhóm sinh ra

### Đặc trưng CLIP của vg_coco

Trích ngày 2026-09-19 bằng `scripts/extract_features.py` trên RTX 4090, encoder `ViT-B-32-quickgelu/openai` (open_clip 3.3.0, torch 2.14.0), từ đủ 51 208 ảnh (0 ảnh lỗi khi tải).

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/vg_coco_image.npy` | 51 208 × 512 | 104874112 | `091acde31fa1b3e7a24314dab889205c24d4ff0bdf405fb2128d923c87e2a1a9` |
| `data/features/vg_coco_image.ids.json` | — | 459664 | `b82397bb868044a9e7a8dd351b978c73b33cbfbb84d178fb49f0a072c990e25b` |
| `data/features/vg_coco_caption.npy` | 256 183 × 512 | 524662912 | `c116bfe099bdcb33dc5810f829de9080647b972eb53104fa647b2e2e0cb286d3` |
| `data/features/vg_coco_caption.ids.json` | — | 3025464 | `7eb6a7a05dff4007bdd6aa7c2f4b08570f79d095336a932ce5843da92df6cf69` |

Kiểm tra sau khi tải về: `sha256sum data/features/vg_coco_*` (Windows: `certutil -hashfile <file> SHA256`).

### Đặc trưng MSCOCO 5K (chỉ để kiểm tra pipeline, không phải dữ liệu của đề tài)

Sinh ngày 2026-09-19 bằng `scripts/checks/coco5k_zero_shot.py` trên RTX 4090, encoder `ViT-B-32-quickgelu/openai`. Chỉ nằm trên máy chạy, không có trên Drive. Xem `experiments/checks/README.md`.

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/coco5k_image.npy` | 5 000 × 512 | 10240128 | `14e7908d8151ef25128596586efc2652e88eeac7f2c076b99c3176bbd6916a93` |
| `data/features/coco5k_image.ids.json` | — | 39089 | `ed26e1361a67dc470249f4b6885585f9123c6ce991ac1ddac12d76876be455f2` |
| `data/features/coco5k_caption.npy` | 25 010 × 512 | 51220608 | `2b6a18218cbcbf05bb1e1d8fdfcc4ce8cb330b7bc23df99c39ca59ad4aa1a8f5` |
| `data/features/coco5k_caption.ids.json` | — | 295319 | `a563a520c6ae78fd7c793ca61eb6a174afe98ac1440fd5b599424bd905b1a56b` |

### Vector các phần đồ thị của split val (phép thử ở `experiments/checks/`, mục 4)

Sinh bởi `scripts/checks/soft_graph_rerank_probe.py` ngày 2026-09-19; khoá trong `ids` có dạng `node:<nhãn>` hoặc `triple:<cụm bộ ba>`. Chỉ lưu trên máy chạy.

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/vg_coco_val_graph_parts.npy` | 42 736 × 512 | 87523456 | `8c41676c5f228eafbfbf9957768daaf552bf4cfd2f56d1bf819d856cd461adca` |
| `data/features/vg_coco_val_graph_parts.ids.json` | — | 1176825 | `5dc2be646f18a0d11025842e171f56315197e6871bfe5d68a13610e1f31e5543` |

Ghi chú: các file trước đó (encoder `ViT-B-32/openai`) được lưu trữ trên máy GPU tại `data/features/archive-vit-b-32-gelu/`. Ai đã tải feature từ Drive trước thay đổi này cần tải lại.

### Gói dữ liệu re-ranker đồ thị

Sinh bởi `src.data.train_groups` và `scripts/build_graph_data.py`; các giá trị dưới đây chờ reviewer chạy đủ dữ liệu (`TODO(run)`).

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/splits/vg_coco_train_groups.json` | TODO(run) | TODO(run) |
| `data/graphs/vg_coco_scene_graphs.jsonl` | TODO(run) | TODO(run) |
| `data/processed/query_graphs_{train,val,test}.json` | TODO(run) | TODO(run) |
| `data/features/vg_coco_graph_parts.npy` + `.ids.json` | TODO(run) | TODO(run) |
| `data/processed/candidates_train.npz` | TODO(run) | TODO(run) |
| `data/processed/candidates_val.npz` | TODO(run) | TODO(run) |
