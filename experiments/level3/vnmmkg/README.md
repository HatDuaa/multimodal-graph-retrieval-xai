# MMKG tiếng Việt: chạy phương pháp cấp độ 2 và phân tích chuyển giao (dòng 32)

Người làm: Khang. Script: `scripts/eval_vnmmkg_transfer.py`. Chạy ngày 2026-11-18 trên CPU.

## Thiết lập

- **Encoder:** XLM-R large ViT-H/14 (LAION-5B), đóng băng. Encoder này được chọn ở dòng 9 trên MKG-W val (`experiments/checks/README.md`, mục 7), không chọn trên bộ dữ liệu này.
- **Phương pháp chuyển sang:** kênh cạnh KG của cấp độ 2 (hàng `kg_triple` của MKG-W).
  - Linker tìm các thực thể được nhắc trong câu tiếng Việt, khớp theo nhãn và bí danh tiếng Việt, độ dài tối đa 12 âm tiết.
  - Điểm đồ thị của một ảnh ứng viên là số cạnh KG trực tiếp giữa di tích của ảnh đó và các thực thể được nhắc.
  - Điểm cuối = α·z(CLIP) + (1−α)·z(đồ thị), tính trong top-50 của mỗi câu.
  - Giải thích của mỗi ảnh là danh sách chính các cạnh đã được đếm (`a0_explanations.json`).
- **Không chuyển kênh embedding NativE:** kênh này cần train NativE trên đồ thị 150 thực thể này. Phần chuyển giao được là kênh cấu trúc.
- **Giao thức** (`vietnamese-mmkg/DATA_CARD.md`, mục 8):
  - **A0 zero-shot:** pool là toàn bộ 331 ảnh. Truy vấn gồm 197 câu tự động và 30 câu soạn tay. α = 0,0 lấy từ MKG-W val (hàng `kg_triple`). Không chọn gì trên bộ này.
  - **A1:** chọn α trên vnmmkg val (pool là ảnh val), rồi chạy test đúng một lần (pool là ảnh test). Câu soạn tay không tham gia A1.
- Encoder đóng băng và không có huấn luyện, nên chỉ chạy một lần, không có độ lệch giữa các seed.

## A0: zero-shot (kết quả chính)

| Nhóm truy vấn | Số câu | CLIP R@1 | CLIP + KG R@1 | CLIP MRR | CLIP + KG MRR |
|---|---|---|---|---|---|
| Tất cả | 227 | 23.35 | **33.92** | 37.18 | **45.18** |
| Tự động (câu đầu bài viwiki, đã che tên) | 197 | 21.32 | **29.95** | 35.21 | **41.61** |
| Soạn tay, trực tiếp | 12 | **75.00** | 33.33 | **81.25** | 48.90 |
| Soạn tay, gián tiếp | 18 | 11.11 | **77.78** | 29.41 | **81.81** |

Có 84,14% câu linker tìm được ít nhất một thực thể.

## A1: α chọn trên val, chạy test một lần

| Test (103 câu tự động, pool 161 ảnh) | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 37.86 | 72.82 | 79.61 | 52.39 |
| CLIP + KG, α = 0,7 (chọn trên val) | **47.57** | **75.73** | **84.47** | **60.41** |
| CLIP + KG, α = 0,0 (α của A0, chỉ để đối chiếu, không phải lựa chọn) | 50.49 | 70.87 | 81.55 | 60.46 |

Trên val, R@1 gần như không đổi trong khoảng α từ 0,4 đến 0,75 (37,2). Với α từ 0 đến 0,35 thì được 36,2. R@1 giảm dần khi α vượt 0,75, còn 29,8 khi α = 1 (chỉ CLIP).

## Phân tích chuyển giao

1. **Kênh cạnh KG chuyển được sang tiếng Việt.** R@1 tăng ở cả A0 (+10,6) và A1 (+9,7), dù linker chỉ khớp chuỗi đơn giản và encoder khác với encoder của nguồn.
2. **Đồ thị giúp đúng ở chỗ đề bài đặt ra, là truy vấn gián tiếp.** R@1 của nhóm gián tiếp tăng từ 11,11 lên 77,78, tức 14 trên 18 câu.
   - Ví dụ "Ngôi chùa ở Hà Nội được xây dựng dưới thời nhà Lý": ảnh xếp đầu (đúng) được cộng điểm nhờ ba cạnh: P131 (ở) → Hà Nội, P2348 (thời kỳ) → nhà Lý, P31 (là một) → chùa. CLIP không thể thấy thời kỳ hay vị trí hành chính trong ảnh.
   - 4 câu gián tiếp còn sai đều chung một nguyên nhân: **kênh cạnh 1-hop khớp đúng thực thể, không hiểu quan hệ lớp cha – lớp con.**
     - Ba câu hỏi "Nhà thờ … ở Ninh Bình / Khánh Hòa / Thành phố Hồ Chí Minh" có đáp án là "nhà thờ chính tòa". Từ "nhà thờ" được nối vào lớp cha Q16970, nên đáp án không có cạnh P31 tới lớp đó. Trong khi đó các nhà thờ thường lại có đủ P31 và P140 (Công giáo), nên vượt lên trên đáp án.
     - Câu 17 nhắc "di sản thế giới" (Q9259), nhưng danh hiệu thật của đáp án là một thực thể khác (Q43113623).
     - Hướng sửa (chưa làm): đi thêm cạnh P279 (lớp con của) khi đếm cạnh, hoặc đếm đường đi 2-hop.
3. **Đồ thị làm hỏng truy vấn trực tiếp khi α quá thấp.** R@1 nhóm trực tiếp giảm từ 75,00 xuống 33,33. Nguyên nhân thấy rõ trong file giải thích:
   - Câu trực tiếp chứa các từ chỉ loại như "nhà thờ", "tháp", "chùa". Linker nối các từ này vào thực thể lớp, nên mọi nhà thờ trong pool đều có cạnh P31 và đều được cộng điểm như nhau.
   - α = 0 lấy từ MKG-W lại để điểm đồ thị quyết định gần như toàn bộ thứ hạng; CLIP chỉ còn dùng để phá thế hoà. Chi tiết hình ảnh mà CLIP bắt được, như "sơn màu hồng" hay "hai tháp chuông", vì thế bị lấn át.
   - Bài học chuyển giao: α tối ưu phụ thuộc vào tỉ lệ truy vấn gián tiếp của từng bộ dữ liệu. Truy vấn tự động của MKG-W là mô tả Wikidata gần như toàn quan hệ ("1935 film by …"), nên α ở đó về 0. Truy vấn tiếng Việt có nhiều mô tả hình ảnh hơn, nên val chọn α = 0,7.
   - Đây là lý do cho hướng mở rộng (a) của đề: kết hợp điểm thích nghi theo từng câu.
4. **Ở A1 có một kết quả ngược chiều: trên test, α của A0 cho R@1 cao hơn α chọn trên val** (50,49 so với 47,57). Nhóm báo cáo đúng như vậy và không chọn lại α trên test. Đường quét trên val rất phẳng, và 103 câu test chỉ là mẫu nhỏ.

## Chạy lại

```bash
python scripts/eval_vnmmkg_transfer.py --encoder xlmr_h14
```

Script đọc α của A0 từ `experiments/level2/val/choice.json` (hàng `kg_triple`, seed 0). Hiện giá trị này bằng 0,0. Lần chạy ghi ở trên diễn ra trước khi có file này, nên đã truyền tay `--alpha-a0 0.0`. Giá trị chọn trên val sau đó (PR #42) cũng bằng 0,0, nên kết quả không đổi.
