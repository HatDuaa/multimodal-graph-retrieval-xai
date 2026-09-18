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
| `data/raw/coco/caption_datasets.zip` | 36745453 | `4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817` |

Sau khi giải nén, `data/raw/` chiếm khoảng 1,4 GB.

Số đếm thật từ các file này (in ra bởi `scripts/smoke_test.py`): Visual Genome có 108 077 ảnh; 51 498 ảnh có `coco_id` nằm trong file split Karpathy, gồm train 34 027, restval 13 183, val 2 146, test 2 142.

## File nhóm sinh ra

### Đặc trưng CLIP của vg_coco

Trích ngày 2026-09-19 bằng `scripts/extract_features.py` trên RTX 4090, encoder `ViT-B-32/openai` (open_clip 3.3.0, torch 2.14.0), từ đủ 51 208 ảnh (0 ảnh lỗi khi tải).

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/vg_coco_image.npy` | 51 208 × 512 | 104874112 | `76f873e5f423a0e6575a6765cf322888f1267851d5411456a78dab91c37c8a6a` |
| `data/features/vg_coco_image.ids.json` | — | 459654 | `6afabaf0e7801db02a8d167e378995d8cf520feb2b35503ef33458e164701f51` |
| `data/features/vg_coco_caption.npy` | 256 183 × 512 | 524662912 | `c61a39290c4d0eb3e2958467f6fc7098b147305585a4e6b8a4323f312fa392d5` |
| `data/features/vg_coco_caption.ids.json` | — | 3025454 | `ff1824f6c87cfae46ba2cef4ff1fc47691b3ebbf8ec2ad20f0ec24977504a370` |

Kiểm tra sau khi tải về: `sha256sum data/features/vg_coco_*` (Windows: `certutil -hashfile <file> SHA256`).
