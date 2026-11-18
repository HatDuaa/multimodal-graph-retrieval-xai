# Nháp báo cáo: Cấp độ 3, MMKG tiếng Việt và phân tích chuyển giao

Nháp để Lộc duyệt rồi ghép vào báo cáo chính. Cấu trúc và quy ước giống `report/level1-draft.md` và
`report/level2-draft.md`:
- giữ thuật ngữ tiếng Anh, giải thích ở lần xuất hiện đầu;
- mọi con số lấy từ file kết quả trong `experiments/`, nguồn ghi cuối từng mục và ở mục 10.

Người soạn nháp: Âu Dương Khang cùng Claude, ngày 2026-11-18. Phạm vi: dòng 9 (chọn CLIP đa ngữ), dòng 30–31
(dữ liệu, đã có trong `vietnamese-mmkg/DATA_CARD.md`) và dòng 32 (chạy phương pháp cấp 2 và phân tích chuyển giao)
của bảng phân công.

## Tóm tắt

Đề yêu cầu ở cấp 3 tự xây một MMKG (multimodal knowledge graph, tức đồ thị tri thức mà thực thể có kèm ảnh)
bằng tiếng Việt, rồi chạy phương pháp tốt nhất của nhóm trên đó và phân tích khả năng chuyển giao. Nhóm xây
một MMKG nhỏ về di tích và công trình ở Việt Nam, gồm 150 thực thể, 325 triple và 331 ảnh đã duyệt tay. Truy vấn
là câu tiếng Việt, đáp án là ảnh của di tích.

Phần được chuyển sang là **kênh đếm cạnh KG** của cấp 2 (hàng `kg_triple` trên MKG-W). Encoder ảnh và văn bản
là **XLM-R large ViT-H/14**, chọn trên MKG-W val vì CLIP của OpenAI gần như không hiểu tiếng Việt. Kết quả chính
là **zero-shot** (A0): không chọn bất cứ thứ gì trên bộ tiếng Việt.

| A0, pool 331 ảnh | Số câu | CLIP R@1 | CLIP + KG R@1 | CLIP MRR | CLIP + KG MRR |
|---|---|---|---|---|---|
| Tất cả | 227 | 23,35 | **33,92** | 37,18 | **45,18** |
| Soạn tay, gián tiếp | 18 | 11,11 | **77,78** | 29,41 | **81,81** |
| Soạn tay, trực tiếp | 12 | **75,00** | 33,33 | **81,25** | 48,90 |

Kết luận:
- Kênh cạnh KG **chuyển giao được** sang tiếng Việt. R@1 tăng 10,6 điểm ở A0 và 9,7 điểm ở A1, dù linker chỉ
  khớp chuỗi và encoder khác với encoder trên tập nguồn.
- Phần tăng nằm đúng ở chỗ đề bài nhắm tới, là **truy vấn gián tiếp** (cần một quan hệ không thấy được trong ảnh).
- Cái giá phải trả là **truy vấn trực tiếp bị kéo xuống**, vì α chọn trên MKG-W (α = 0) để đồ thị quyết định
  gần hết thứ hạng. Điều này cho thấy α tối ưu phụ thuộc vào tỉ lệ câu gián tiếp của từng bộ dữ liệu.

## 1. Dữ liệu

Chi tiết đầy đủ ở `vietnamese-mmkg/DATA_CARD.md`; dưới đây chỉ là phần cần cho việc đọc kết quả.

- **Miền:** di tích và công trình ở Việt Nam, gồm chùa, đền, đình, tháp, nhà thờ, lăng, thành, bảo tàng và nơi
  có danh hiệu di sản. Chọn miền này vì dữ liệu có giấy phép mở và nhiều quan hệ không thấy được trong ảnh:
  ở tỉnh nào, xây thời nào, ai sáng lập, thờ ai.
- **Nguồn:**
  - Wikidata (CC0) cho thực thể và triple;
  - phần mở đầu và infobox của Wikipedia tiếng Việt (CC BY-SA 4.0);
  - ảnh trên Wikimedia Commons, chỉ nhận giấy phép CC0, PD, CC BY, CC BY-SA, ghi tác giả cho từng ảnh.
- **Quy mô** (khung của đề trong ngoặc):
  - 150 thực thể (50–150): 75 di tích, 75 thực thể ngữ cảnh;
  - 325 triple (300–1 000), thuộc 13 quan hệ;
  - 331 ảnh (200–500), đều do Khang duyệt tay.
- **Triple:** 257 lấy từ Wikidata, 26 từ infobox, 42 từ hai quy tắc suy diễn cố định (tỉnh theo chuỗi P131,
  thời kỳ theo năm dựng). Khang duyệt mọi triple từ infobox và suy diễn.
- **Truy vấn:**
  - 197 câu tự động: tối đa 3 câu đầu bài viwiki, đã che tên di tích;
  - 30 câu soạn tay: 12 trực tiếp (tả cái thấy được trong ảnh, một đáp án) và 18 gián tiếp (cần ít nhất một
    quan hệ). Đáp án câu gián tiếp tính tự động từ triple.
  - Bộ soạn tay **khoá ở commit `ecfca98`** trước mọi lần chạy.

*(nguồn: `vietnamese-mmkg/DATA_CARD.md` mục 3–7, `experiments/data_stats.md` khối `vnmmkg`)*

## 2. Chống rò rỉ dữ liệu

Bộ tiếng Việt quá nhỏ để vừa chọn tham số vừa chấm điểm, nên nhóm dùng hai giao thức, chốt trước khi chạy
(`vietnamese-mmkg/DATA_CARD.md` mục 8):

- **A0, zero-shot (kết quả chính):**
  - pool là toàn bộ 331 ảnh, truy vấn là 197 câu tự động và 30 câu soạn tay;
  - **không chọn gì trên bộ này**: encoder chọn trên MKG-W val (mục 3), α lấy từ MKG-W val
    (`experiments/level2/val/choice.json`, hàng `kg_triple`, α = 0).
- **A1, chọn lại α (phân tích phụ):**
  - chia di tích 50/50 bằng seed 0: val có 38 di tích, 170 ảnh, 94 câu; test có 37 di tích, 161 ảnh, 103 câu;
  - chỉ chọn α trên val (pool là ảnh val), rồi chạy test **một lần** (pool là ảnh test);
  - câu soạn tay không tham gia A1.
- **Triple** là tri thức có sẵn, không phải cạnh thống kê đồng xuất hiện, nên được dùng hết. Quy tắc "cạnh thống
  kê chỉ lấy từ train" của đề không áp dụng ở đây.
- **Không có huấn luyện**, encoder đóng băng, nên mỗi giao thức chỉ chạy một lần; không có độ lệch giữa các seed.
  Đây là lý do bảng ở mục 5 không có ± std.

## 3. Chọn encoder cho truy vấn tiếng Việt (dòng 9)

CLIP ViT-B/32 của OpenAI, encoder chung của cấp 1 và cấp 2, chỉ được huấn luyện với văn bản tiếng Anh. Để không
chọn gì trên bộ tiếng Việt, nhóm so 4 encoder trên MKG-W val:
- 84 thực thể val có mô tả tiếng Việt trên Wikidata, mỗi thực thể hỏi hai lần (tiếng Việt và tiếng Anh);
- cả hai câu đều che tên bằng cùng một hàm;
- pool là 892 ảnh val.

| Encoder | R@1 vi | R@10 vi | MRR vi | R@1 en | MRR en |
|---|---|---|---|---|---|
| OpenAI ViT-B/32 (encoder hiện tại) | 10,71 | 17,86 | 14,10 | 35,71 | 46,03 |
| XLM-R base + ViT-B/32 (LAION-5B) | 25,00 | 61,90 | 37,83 | 28,57 | 40,50 |
| M-CLIP (text đa ngữ chưng cất vào không gian OpenAI B/32) | 28,57 | 55,95 | 40,05 | 33,33 | 44,62 |
| **XLM-R large + ViT-H/14 (LAION-5B)** | **40,48** | **78,57** | **51,80** | 40,48 | 51,31 |

- Với câu tiếng Việt, R@1 của OpenAI B/32 tụt từ 35,71 xuống 10,71: encoder này không dùng được.
- XLM-R large ViT-H/14 cao nhất với câu tiếng Việt và không chênh giữa hai ngôn ngữ. Nhóm chốt encoder này cho
  mọi truy vấn tiếng Việt.
- Cấp 1 và cấp 2 vẫn giữ OpenAI B/32 để không phải chạy lại các kết quả tiếng Anh.
- Hạn chế: chỉ có 84 câu. Khoảng cách giữa hai encoder B/32 đa ngữ (3 câu) nằm trong mức nhiễu. Khoảng cách
  lên H/14 (từ 12 điểm trở lên) thì rõ ràng.

*(nguồn: `experiments/checks/multilingual_clip_probe.json`, `experiments/checks/README.md` mục 7)*

## 4. Phương pháp chuyển giao

Phương pháp giữ nguyên cấp 2 (hàng `kg_triple`), chỉ thay đồ thị, ngôn ngữ và encoder:

1. Encoder lấy top-50 ảnh theo cosine giữa câu và ảnh.
2. **Linker** (`EntityLinker` của Đạt) tìm các thực thể được nhắc trong câu. Linker so khớp chuỗi với nhãn và
   bí danh tiếng Việt, độ dài tối đa 12 âm tiết; cấp 2 dùng 5 vì tên tiếng Việt dài hơn.
3. **Điểm đồ thị** của một ảnh = số cạnh KG trực tiếp giữa di tích trong ảnh và các thực thể được nhắc.
4. **Điểm cuối** = α·z(CLIP) + (1−α)·z(đồ thị), với z-score tính trong top-50 của từng câu.
5. **Giải thích** của mỗi ảnh là danh sách chính các cạnh đã được đếm ở bước 3. Danh sách này sinh ra cùng
   lúc với điểm, không tìm đường đi sau khi xếp hạng.

**Không chuyển kênh embedding NativE.** Kênh này cần train lại NativE trên đồ thị mới, mà 325 triple là quá ít
để học embedding. Trên MKG-W, val cũng chọn trọng số 0 cho kênh embedding (β = 0), nên mô hình chính ở đó đã
trùng với `kg_triple` (`report/level2-draft.md`). Phần chuyển giao ở đây vì vậy chính là phương pháp tốt nhất
của cấp 2.

Code: `scripts/eval_vnmmkg_transfer.py --encoder xlmr_h14`.

## 5. Kết quả

### 5.1. A0, zero-shot (kết quả chính)

| Nhóm câu | Số câu | CLIP R@1 | CLIP + KG R@1 | CLIP R@10 | CLIP + KG R@10 | CLIP MRR | CLIP + KG MRR |
|---|---|---|---|---|---|---|---|
| Tất cả | 227 | 23,35 | **33,92** | 72,69 | **74,01** | 37,18 | **45,18** |
| Tự động | 197 | 21,32 | **29,95** | 70,05 | **72,59** | 35,21 | **41,61** |
| Soạn tay, trực tiếp | 12 | **75,00** | 33,33 | **100,00** | 75,00 | **81,25** | 48,90 |
| Soạn tay, gián tiếp | 18 | 11,11 | **77,78** | 83,33 | **88,89** | 29,41 | **81,81** |

Linker tìm được ít nhất một thực thể ở 84,14% số câu.

### 5.2. A1, α chọn trên val tiếng Việt, test chạy một lần

| Test (103 câu tự động, pool 161 ảnh) | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 37,86 | 72,82 | 79,61 | 52,39 |
| CLIP + KG, α = 0,7 (chọn trên val) | **47,57** | **75,73** | **84,47** | **60,41** |
| CLIP + KG, α = 0 (α của A0, chỉ để đối chiếu) | 50,49 | 70,87 | 81,55 | 60,46 |

R@1 trên val gần như không đổi khi α đi từ 0,4 đến 0,75 (37,2). Với α từ 0 đến 0,35, R@1 là 36,2. Khi α vượt
0,75, R@1 giảm dần, còn 29,8 ở α = 1 (chỉ CLIP).

*(nguồn: `experiments/level3/vnmmkg/xlmr_h14/a0_metrics.json`, `a1_metrics.json`)*

## 6. Giải thích: ví dụ

Mỗi dòng là ảnh xếp hạng 1 sau khi trộn điểm, kèm các cạnh mà hàm điểm đã đếm cho ảnh đó.

| Câu (soạn tay) | Ảnh hạng 1 | Đúng? | Cạnh được đếm |
|---|---|---|---|
| Ngôi chùa ở Hà Nội được xây dựng dưới thời nhà Lý | Chùa Một Cột | Đúng | P131 (ở) → Hà Nội; P2348 (thời kỳ) → nhà Lý; P31 (là một) → chùa |
| Ngôi đền ở Hà Nội thờ theo Đạo giáo | Đền Quán Thánh | Đúng | P131 → Hà Nội; P825 (thờ) → Đạo giáo; P31 → đền |
| Nhà thờ ở Hà Nội được đặt theo tên thánh Antôn thành Padova | Nhà thờ Hàm Long | Đúng | P138 (đặt tên theo) → thánh Antôn; P31 → nhà thờ; P131 → Hà Nội |
| Nhà thờ Công giáo ở Ninh Bình | Vương cung thánh đường Phú Nhai | Sai | P31 → nhà thờ; P140 (tôn giáo) → Công giáo |
| Công trình bằng đá xám có nhiều tầng mái cong như đình chùa, trên đỉnh có cây thánh giá | Chùa Bút Tháp | Sai | P31 → chùa |

Ba câu đầu cho thấy kiểu giải thích mà CLIP không có: vị trí hành chính, thời kỳ và đối tượng thờ đều không nhìn
thấy được trong ảnh.

Hai câu sai cũng đọc được nguyên nhân ngay từ danh sách cạnh:
- Câu thứ tư: ảnh sai có hai cạnh khớp. Đáp án (một nhà thờ chính tòa ở Ninh Bình) lại thiếu cạnh P31 tới lớp
  "nhà thờ", vì trên Wikidata nó thuộc lớp con "nhà thờ chính tòa".
- Câu thứ năm: từ "chùa" trong câu chỉ là một so sánh, nhưng linker vẫn nối nó vào lớp "chùa". Kết quả là một
  ngôi chùa thật vượt lên trên đáp án, vốn là một nhà thờ.

*(nguồn: `experiments/level3/vnmmkg/xlmr_h14/a0_explanations.json`)*

## 7. Phân tích chuyển giao

1. **Kênh cạnh KG chuyển được.** Trên MKG-W test, kênh này nâng R@1 từ 24,89 lên 29,82
   (`experiments/level2/test/summary.json`). Trên bộ tiếng Việt, R@1 tăng 10,6 điểm ở A0 và 9,7 điểm ở A1.
   Kết quả này có được dù khác ngôn ngữ, khác encoder, đồ thị nhỏ hơn khoảng 130 lần (325 so với 42 746 triple) và không train lại gì.
   Lý do là kênh này không học tham số nào. Nó chỉ cần ba thứ: một linker theo tên, một KG có quan hệ có nghĩa,
   và một α.
2. **Đồ thị giúp đúng ở truy vấn gián tiếp.** R@1 nhóm gián tiếp tăng từ 11,11 lên 77,78 (14/18 câu). Bốn câu
   còn sai có chung một nguyên nhân: **kênh 1-hop chỉ khớp đúng thực thể, không hiểu quan hệ lớp cha – lớp con.**
   - Ba câu "Nhà thờ … ở Ninh Bình / Khánh Hòa / Thành phố Hồ Chí Minh" có đáp án là nhà thờ chính tòa. Từ "nhà
     thờ" được nối vào lớp cha Q16970, nên đáp án thiếu cạnh P31 tới lớp đó. Trong khi đó, các nhà thờ thường có
     đủ P31 và P140 (Công giáo) nên vượt lên.
   - Câu 17 nhắc "di sản thế giới" (Q9259), nhưng danh hiệu thật của đáp án là một thực thể khác (Q43113623).
   - Hướng sửa (chưa làm): khi đếm cạnh thì đi thêm cạnh P279 (lớp con của), hoặc đếm cả đường đi 2-hop.
3. **Đồ thị làm hỏng truy vấn trực tiếp khi α quá thấp.** R@1 nhóm trực tiếp giảm từ 75,00 xuống 33,33.
   - Câu trực tiếp chứa các từ chỉ loại như "nhà thờ", "tháp", "chùa". Linker nối các từ này vào thực thể lớp,
     nên mọi nhà thờ trong pool đều có cạnh P31 và được cộng điểm như nhau.
   - Với α = 0, điểm đồ thị quyết định gần như toàn bộ thứ hạng, CLIP chỉ còn phá thế hoà. Những chi tiết hình
     ảnh CLIP bắt được, như "sơn màu hồng" hay "hai tháp chuông", vì thế bị lấn át.
   - Trên MKG-W, đường cong α phẳng từ 0 đến 0,35 rồi giảm dần: truy vấn tự động là mô tả Wikidata, gần như toàn
     quan hệ ("1935 film by …"), nên α về 0 không mất gì. Câu tiếng Việt có nhiều mô tả hình ảnh hơn, nên val
     tiếng Việt chọn α = 0,7.
   - **Bài học:** α tối ưu phụ thuộc vào tỉ lệ câu gián tiếp của từng bộ dữ liệu, nên một α cố định khó chuyển
     giao. Đây là động lực cho hướng mở rộng (a) của đề: trọng số kết hợp thích nghi theo từng câu. Mô hình chính
     cấp 1 đã làm theo hướng này (a, b, c theo từng câu, `report/level1-draft.md` mục 3).
4. **Ở A1 có một kết quả ngược chiều.** Trên test, α của A0 cho R@1 cao hơn α chọn trên val (50,49 so với 47,57),
   còn MRR gần như bằng nhau (60,46 so với 60,41). Nhóm báo cáo đúng như vậy và không chọn lại α trên test.
   Đường quét trên val rất phẳng, và 103 câu test là mẫu nhỏ: 3 điểm R@1 tương ứng 3 câu.
5. **Linker là nút thắt khi chuyển ngôn ngữ.** Linker hiện chỉ so khớp chuỗi trên nhãn và bí danh. Nó tìm được
   thực thể ở 84,14% số câu, nhưng có hai kiểu lỗi đã thấy ở mục 6: nối từ chỉ loại vào thực thể lớp, và nối cả
   những từ dùng để so sánh ("như đình chùa"). Cả hai đều là lỗi của bước linking, không phải của đồ thị.

## 8. Hạn chế

- Bộ dữ liệu nhỏ: 30 câu soạn tay, 103 câu test ở A1. Chênh lệch vài điểm chỉ ứng với vài câu, nên chỉ nên đọc
  các xu hướng lớn (gián tiếp tăng mạnh, trực tiếp giảm).
- Encoder đóng băng, không có huấn luyện, nên không có độ lệch giữa seed. Độ bất định ở đây đến từ cỡ mẫu,
  không phải từ seed, và chưa được đo bằng bootstrap.
- Thiên lệch dữ liệu: ưu tiên di tích nổi tiếng (theo số sitelink), tỉnh lẫn trước và sau đợt sáp nhập năm 2025,
  thời kỳ chỉ suy theo năm (`vietnamese-mmkg/DATA_CARD.md` mục 9).
- Không chuyển NativE (mục 4), và chưa thử các hướng sửa ở mục 7: thêm P279, đi 2-hop, α theo từng câu.
- KG ở mức thực thể, không có scene graph cho từng ảnh. Vì vậy mô hình cấp 1 (khớp scene graph) không áp dụng
  được cho bộ này.

## 9. Đóng góp

Âu Dương Khang làm dòng 9, 30, 31, 32 cùng Claude (Claude Code):
- **Khang** chốt miền và nguồn dữ liệu, duyệt toàn bộ ảnh và mọi triple infobox/suy diễn, duyệt và khoá bộ truy
  vấn soạn tay, chạy phép thử encoder trên molab và duyệt kết quả.
- **Claude** viết code, chạy pipeline và soạn nháp tài liệu.

Dòng 9 ban đầu do Nhi nhận; Khang làm thay theo thoả thuận nhóm ngày 16/11. EntityLinker và hàm điểm đồ thị là
của Đạt. Khai báo đầy đủ ở `report/declaration-ai-oss.md`.

## 10. Nguồn số liệu

| Mục | File |
|---|---|
| Dữ liệu, thống kê, giao thức A0/A1 | `vietnamese-mmkg/DATA_CARD.md`, `experiments/data_stats.md` |
| Chọn encoder (dòng 9) | `experiments/checks/multilingual_clip_probe.json`, `experiments/checks/README.md` mục 7 |
| α của A0 | `experiments/level2/val/choice.json` (hàng `kg_triple`) |
| A0, A1 | `experiments/level3/vnmmkg/xlmr_h14/a0_metrics.json`, `a1_metrics.json` |
| Giải thích | `experiments/level3/vnmmkg/xlmr_h14/a0_explanations.json` |
| Phân tích chi tiết | `experiments/level3/vnmmkg/README.md` |
| MKG-W để đối chiếu | `experiments/level2/test/summary.json` |
| Code | `scripts/eval_vnmmkg_transfer.py`, `scripts/checks/multilingual_clip_probe.py`, `src/retrieval/mkgw_graph_score.py` |
