# Thống kê dữ liệu — Visual Genome ∩ COCO

Sinh bởi `python -m src.data.build_split`. Không sửa tay.

- Ảnh trong Visual Genome: 108077
- Ảnh có `coco_id` nằm trong file split Karpathy: 51498
- Bỏ vì trùng `coco_id` (hai ảnh Visual Genome cho cùng một ảnh COCO): 290
- Theo split Karpathy gốc: {'restval': 13096, 'test': 2138, 'train': 33848, 'val': 2126} (`restval` gộp vào train)

| Split | Số ảnh | Số caption (truy vấn) |
|---|---|---|
| train | 46944 | 234854 |
| val | 2126 | 10633 |
| test | 2138 | 10696 |

<!-- vnmmkg:begin -->
## MMKG tiếng Việt (`vnmmkg`)

Sinh bởi `python -m src.data.build_vnmmkg` (seed 0). Không sửa tay.

| Thành phần | Số lượng | Khung của đề |
|---|---|---|
| Thực thể | 150 (di tích 75, ngữ cảnh 75) | 50–150 |
| Triple | 325 | 300–1 000 |
| Ảnh đã duyệt | 331 | 200–500 |

- `valid_candidates`: 110
- `categories`: {"class": 17, "deity": 5, "designation": 5, "landmark": 75, "other": 17, "period": 4, "person": 7, "province": 13, "religion": 5, "style": 2}
- `triples_by_source`: {"derived": 42, "infobox": 26, "wikidata": 257}
- `triples_by_relation`: {"P112": 7, "P131": 51, "P138": 4, "P140": 57, "P1435": 36, "P149": 9, "P186": 1, "P2348": 21, "P276": 4, "P31": 86, "P361": 20, "P527": 22, "P825": 7}
- `triple_proposal_counters`: {"infobox_unlinked": 9, "no_province": 15, "year_without_period": 10}
- `triple_review`: {}
- `dropped_by_entity_cap`: {"entities": 51, "triples": 51}
- `image_selection`: {"images_over_max": 144, "images_rejected": 541, "images_undecided": 1, "landmarks_too_few_images": 35}
- `licenses`: {"CC BY 2.0": 51, "CC BY 3.0": 14, "CC BY 4.0": 18, "CC BY-SA 2.0": 25, "CC BY-SA 2.5": 4, "CC BY-SA 3.0": 79, "CC BY-SA 4.0": 55, "CC0": 28, "Public Domain": 57}
- `queries_auto`: 197
- `query_drops`: {}
- `split`: {"val": {"landmarks": 38, "images": 170, "queries": 94}, "test": {"landmarks": 37, "images": 161, "queries": 103}}
<!-- vnmmkg:end -->
