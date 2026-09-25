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

## MKG-W retrieval split (built {"seed": 0, "lang": "en"})

- Cache: 15000 entities; usable (description + image + non-empty masked query): 5752
- Dropped: {'missing': 0, 'no_desc': 62, 'no_image': 9186, 'empty_query': 0}
- Split sizes: train 4602, val 575, test 575
- KGC triples copied with their NativE split: 42746

## MKG-W retrieval split (built {"seed": 0, "lang": "en"})

- Cache: 15000 entities; usable (description + author image + non-empty masked query): 8920
- Dropped: {'missing': 0, 'no_desc': 62, 'no_author_image': 6016, 'empty_query': 2}
- Split sizes: train 7136, val 892, test 892
- KGC triples copied with their NativE split: 42746
