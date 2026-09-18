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
