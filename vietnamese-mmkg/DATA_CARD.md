# Data card — MMKG tiếng Việt về di tích và công trình Việt Nam (`vnmmkg`)

Người thực hiện: Âu Dương Khang (25C15045), nhóm 7, môn Học máy với dữ liệu đồ thị. Phiên bản: build ngày 2026-10-26, seed 0.

## 1. Mục đích và phạm vi

Bộ dữ liệu phục vụ cấp độ 3 của Project 3 (truy vấn đa phương thức có giải thích bằng đồ thị). Nó kiểm tra xem phương pháp tốt nhất của cấp 1/2 có chuyển giao được sang một miền hẹp bằng tiếng Việt mà không cần huấn luyện lại hay không. Miền là di tích và công trình kiến trúc ở Việt Nam: chùa, đền, đình, tháp, nhà thờ, lăng, thành, bảo tàng và di tích có danh hiệu di sản. Truy vấn viết bằng tiếng Việt, đáp án là ảnh của di tích.

## 2. Nguồn dữ liệu và giấy phép

| Nguồn | Dùng để | Ngày tải | Giấy phép |
|---|---|---|---|
| Wikidata (SPARQL, `wbgetentities`) | danh sách di tích, nhãn, alias, triple | 2026-10-24 | CC0 |
| Wikipedia tiếng Việt (MediaWiki API, ghim `revid` từng bài) | phần mở đầu (truy vấn tự động), infobox | 2026-10-24 | CC BY-SA 4.0 |
| Wikimedia Commons (category P373 và ảnh P18) | ảnh | 2026-10-24 → 2026-10-25 | theo từng file: chỉ CC0, Public Domain, CC BY, CC BY-SA; tác giả, giấy phép, URL nguồn ghi trong `images.jsonl` |

Phân bố giấy phép của ảnh đã giữ: {"CC BY 2.0": 51, "CC BY 3.0": 14, "CC BY 4.0": 18, "CC BY-SA 2.0": 25, "CC BY-SA 2.5": 4, "CC BY-SA 3.0": 79, "CC BY-SA 4.0": 55, "CC0": 28, "Public Domain": 57}.

## 3. Chọn thực thể

1. Ứng viên là thực thể có `P17 = Việt Nam` và thoả một trong hai điều kiện:
   - có `P1435` (danh hiệu di sản);
   - có `P31` thuộc một trong các lớp: đền (Q44539), chùa (Q5393308), đan viện (Q160742), nhà thờ chính tòa (Q2977), nhà thờ Kitô giáo (Q16970), thần xã (Q845945), nơi thờ phụng (Q1370598), lâu đài (Q23413), tượng đài (Q4989906), bảo tàng (Q33506).

   Ứng viên phải có nhãn tiếng Việt, bài Wikipedia tiếng Việt và Commons category. Hiện vật di động (có `P195` hoặc `P189`) bị loại.
2. Ứng viên được xếp theo số sitelink giảm dần, nếu hoà thì theo QID. Truy vấn SPARQL trả về 339 ứng viên, trong đó 133 hợp lệ (`experiments/checks/vnmmkg_triple_budget_after.json`, trường `valid_candidates`). 110 ứng viên đầu được đưa vào duyệt ảnh (`stats: valid_candidates` = 110).
3. Chỉ giữ di tích có ít nhất 3 ảnh được duyệt, mỗi di tích tối đa 5 ảnh (ảnh P18 trước, rồi theo tên file).
4. Thực thể ngữ cảnh (tỉnh, người, thần, tôn giáo, danh hiệu, thời kỳ, phong cách, lớp, khác) được thêm theo bậc giảm dần, nếu hoà thì theo QID, cho đến tổng 150 thực thể. Số bị bỏ do giới hạn: {"entities": 51, "triples": 51}.

Lưu ý về thiên lệch:
- Xếp theo sitelink ưu tiên di tích nổi tiếng, nên di tích ít được biết đến bị thiếu.
- Quy tắc `P1435` nhận mọi thứ có danh hiệu di sản, nên bộ dữ liệu có cả vườn quốc gia, hồ, sông, vịnh, đảo và phố cổ (ví dụ Vườn quốc gia Phong Nha - Kẻ Bàng, Hồ Ba Bể, Mê Kông, Vịnh Hạ Long, Phố cổ Hội An). Nhóm giữ nguyên theo thiết kế.

## 4. Cách tạo triple

Triple đến từ ba nguồn. Mỗi bộ ba (đầu, quan hệ, đuôi) chỉ giữ một nguồn, theo thứ tự ưu tiên Wikidata > infobox > suy diễn.

- **Wikidata**: giá trị truthy (hạng tốt nhất) của các quan hệ P31, P1435, P140, P149, P138, P825, P2348, P361, P186, P112, P276, P527, đi từ di tích sang thực thể ngữ cảnh. Thêm triple một bước giữa các thực thể ngữ cảnh, dùng cùng danh sách quan hệ cộng P131. P276 (vị trí) và P527 (gồm có) được thêm ở cổng G2 (xem cuối mục này).
- **Infobox Wikipedia tiếng Việt**: dùng 10 mẫu infobox: "Thông tin chùa", "Thông tin đền Việt Nam", "Thông tin tháp Chăm", "Thông tin di tích lịch sử", "Infobox di tích lịch sử", "Thông tin công trình xây dựng", "Infobox Mandir", "Infobox church", "Infobox religious building", "Infobox ancient site". Chỉ đọc các trường cùng nghĩa với ba quan hệ:
  - người sáng lập (P112): `founder`, `người sáng lập`, `creator`;
  - thờ (P825): `thờ phụng`, `đối tượng thờ`, `tên thần`, `thờ`, `primary_deity`, `deity`;
  - tôn giáo (P140): `denomination`, `religious_affiliation`.

  Mẫu infobox bảo tàng không được dùng, vì trường "thành lập" ở đó là năm lập bảo tàng, không phải năm dựng công trình. Chỉ nhận giá trị là wikilink, rồi ánh xạ sang QID qua `pageprops` của viwiki. Có 9 giá trị chữ không có link, bị bỏ (`infobox_unlinked`).
- **Suy diễn, hai quy tắc cố định**:
  - *Tỉnh*: đi ngược chuỗi P131 truthy tới thực thể thuộc một trong ba lớp: tỉnh của Việt Nam (Q2824648), thành phố trực thuộc trung ương (Q1381899), tỉnh cũ của Việt Nam (Q137325529).
    - Nếu mọi đường đi cùng tới đúng một tỉnh thì tạo `(di tích, P131, tỉnh)`. Nếu tới hai tỉnh (ví dụ do sáp nhập tỉnh năm 2025) thì không tạo.
    - Tỉnh nằm ngay ở P131 của di tích được ghi là nguồn Wikidata.
    - Lớp "tỉnh cũ" được thêm ở cổng G1/G2: sau đợt sáp nhập năm 2025, các tỉnh cũ trên Wikidata mất lớp "tỉnh của Việt Nam", làm 17/90 di tích không tìm ra tỉnh. Hệ quả là một số triple tỉnh trỏ tới tỉnh trước sáp nhập (ví dụ Quảng Bình, Quảng Nam, Thừa Thiên - Huế).
    - 15 di tích vẫn không có triple tỉnh (`no_province`).
  - *Thời kỳ*: năm dựng lấy từ P571 (độ chính xác năm), hoặc từ trường infobox `founded`, `thành lập`, `xây dựng`, `date_built`, `built`, `year_completed`, `completed date`. Chỉ dùng khi có đúng một năm và không có chữ ước lượng ("khoảng", "thế kỷ"...). Năm được đối chiếu với bảng mốc không chồng nhau dưới đây. Năm rơi vào khoảng trống hoặc trước 1009 thì không tạo triple (10 trường hợp, `year_without_period`).

| Thời kỳ | Từ | Đến | QID |
|---|---|---|---|
| Nhà Lý | 1009 | 1225 | Q878309 |
| Nhà Trần | 1226 | 1399 | Q3111454 |
| Nhà Lê sơ | 1428 | 1526 | Q878276 |
| Nhà Mạc | 1527 | 1532 | Q1317991 |
| Nhà Lê trung hưng | 1593 | 1777 | Q10800791 |
| Nhà Tây Sơn | 1789 | 1801 | Q1034173 |
| Nhà Nguyễn | 1802 | 1883 | Q6500483 |
| Pháp thuộc | 1884 | 1945 | Q2523849 |

Nguồn bảng mốc là bài "Lịch sử Việt Nam" trên Wikipedia tiếng Việt, revid 75635404. Các khoảng trống là cố ý:
- 1400–1427: nhà Hồ, Minh thuộc;
- 1533–1592: Nam – Bắc triều;
- 1778–1788: Lê trung hưng và Tây Sơn chồng nhau.

Vì mốc chỉ tính theo năm, công trình xây ở Nam Kỳ trong khoảng 1862–1883 vẫn được xếp vào "Nhà Nguyễn".

**Duyệt tay**: Khang duyệt mọi triple từ infobox và suy diễn. Triple Wikidata mặc định được giữ, chỉ bỏ khi sai rõ. Quyết định lưu trong `review_triples.json`.
- Trên trang duyệt, Khang quyết định cả 107 triple cần duyệt và chỉ bỏ 1 triple (`Q391406|P825|Q11378|infobox`).
- Di tích của triple đó (Mỹ Sơn) không đủ ảnh nên không vào bộ dữ liệu. Vì vậy bộ đếm `triple_review` của bản build là {}: không còn triple nào chưa duyệt hay bị bỏ trong phạm vi đã dựng.

Thống kê cuối: 325 triple.
- Theo nguồn: {"derived": 42, "infobox": 26, "wikidata": 257}.
- Theo quan hệ: {"P112": 7, "P131": 51, "P138": 4, "P140": 57, "P1435": 36, "P149": 9, "P186": 1, "P2348": 21, "P276": 4, "P31": 86, "P361": 20, "P527": 22, "P825": 7}.

Lượng triple đo trước khi duyệt (cổng G2), với 90 di tích, sau khi giới hạn 150 thực thể:
- 307 với danh sách quan hệ ban đầu (`experiments/checks/vnmmkg_triple_budget_before.json`);
- 332 sau khi thêm P276 và P527 (`experiments/checks/vnmmkg_triple_budget_after.json`).

Điểm nghẽn là giới hạn 150 thực thể, không phải thiếu dữ liệu.

## 5. Liên kết ảnh – thực thể

Ảnh ứng viên của một di tích gồm ảnh P18 cộng các file nằm trực tiếp trong Commons category của nó (không đi vào category con). Chỉ nhận JPEG/PNG rộng từ 400 px, tối đa 12 ứng viên. Ảnh được tải ở cỡ 960 px của Commons rồi thu về rộng 640 px; ảnh gốc hẹp hơn 640 px giữ nguyên cỡ. Mỗi ảnh ghi SHA-256. Khang duyệt từng ảnh trên trang `review.html`:

- **giữ** ảnh cho thấy rõ chính di tích (mặt ngoài, hoặc nội thất / chi tiết đặc trưng);
- **bỏ** bản đồ, sơ đồ, tài liệu, ảnh gần trùng, ảnh có người nhận diện được là chủ thể chính, ảnh không nhận ra là di tích đó.

Ảnh dùng chung cho hai di tích chỉ được giữ ở di tích xếp trên. Kết quả duyệt: {"images_over_max": 144, "images_rejected": 541, "images_undecided": 1, "landmarks_too_few_images": 35}, giữ được 331 ảnh. Quyết định lưu trong `review_images.json`.

Mọi ảnh đều có giấy phép và URL nguồn. Có 2 ảnh không có tác giả vì metadata Commons không ghi (`image_id` 6, CC0; `image_id` 76, Public Domain, bưu thiếp khoảng năm 1906). Hai giấy phép này không bắt buộc ghi công.

35/110 di tích bị loại vì còn dưới 3 ảnh được giữ. Trong đó có những di tích nổi tiếng mà bộ ảnh ứng viên chủ yếu là lối vào, cảnh rừng hoặc người, ví dụ Thánh địa Mỹ Sơn chỉ giữ được 1/12 ảnh.

## 6. Truy vấn

- **Tự động** (`queries_auto.jsonl`): tối đa 3 câu đầu phần mở đầu bài viwiki (ghim `revid`).
  - Bỏ nội dung trong ngoặc, rồi che nhãn, alias, tiêu đề bài, nhãn tiếng Anh và các tên đã bỏ tiền tố loại ("Nhà thờ chính tòa", "Nhà thờ", "Chùa", "Đền", "Đình", "Tháp", "Lăng", "Thành", "Bảo tàng", "Miếu", "Phủ", "Văn miếu").
  - So khớp theo ranh giới từ, không phân biệt hoa thường, trên văn bản Unicode NFC.
  - Bỏ câu còn dưới 6 từ hoặc còn sót tên.
  - Kết quả: 197 truy vấn; số câu bị bỏ: {}.
- **Soạn tay** (`queries_hand.json`): 12 câu *trực tiếp* (tả điều thấy trong ảnh, một đáp án) và 18 câu *gián tiếp* (cần ít nhất một quan hệ không thấy được trong ảnh).
  - Đáp án câu gián tiếp được tính tự động từ `pattern` trên `triples.jsonl`, mỗi câu 1–5 đáp án. Phân bố số đáp án: {"1": 15, "2": 1, "3": 1, "4": 1}.
  - Claude soạn nháp từ triple và ảnh đã duyệt; Khang duyệt rồi khoá.
  - Câu gián tiếp tránh những pattern biết là thiếu đáp án. Ví dụ, Lăng Gia Long trên Wikidata là "ngôi mộ" chứ không phải "lăng", và Nhà thờ chính tòa Đà Lạt không có triple thời kỳ.
  - **Bộ câu được khoá ở commit `ecfca9883e259ec033caa3015badb4b267a783c6`**, trước mọi lần chạy truy vấn trên bộ này, và không được sửa theo kết quả.

Hạn chế:
- Tên khác của di tích không có trên Wikidata và không nằm trong ngoặc vẫn có thể lọt vào truy vấn tự động.
- Việc che tên đã bỏ tiền tố cũng có thể xoá các từ thường trùng với tên (ví dụ "một cột").

## 7. Thống kê

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
- `hand_queries`: {"direct": 12, "indirect": 18, "indirect_gold_sizes": {"1": 15, "2": 1, "3": 1, "4": 1}}

## 8. Chia dữ liệu và chống rò rỉ

- **A0 – zero-shot (kết quả chính)**: pool là toàn bộ ảnh đã duyệt (val ∪ test), truy vấn là toàn bộ truy vấn tự động và bộ soạn tay. Không học hay chọn gì trên bộ này; α và mọi siêu tham số giữ nguyên từ tập nguồn.
- **A1 – thích nghi α (phân tích phụ)**: di tích được chia 50/50 thành val/test bằng seed 0 (`data/splits/vnmmkg_{val,test}.json`).
  - Val: 38 di tích, 170 ảnh, 94 truy vấn. Test: 37 di tích, 161 ảnh, 103 truy vấn.
  - Chỉ α (và β nếu có) được chọn trên val, với pool là val. Test chạy một lần.
- Triple là tri thức có sẵn (Wikidata, infobox, hai quy tắc cố định), không phải cạnh thống kê đồng xuất hiện. Vì vậy dùng toàn bộ triple không vi phạm quy tắc "cạnh thống kê chỉ lấy từ train".

## 9. Hạn chế

- Thiên lệch theo độ phổ biến (sitelink) và theo những gì Wikidata/Wikipedia tiếng Việt có ghi.
- Wikidata thưa ở các quan hệ thời kỳ, thờ ai, người dựng. Phần lớn các triple này đến từ infobox và quy tắc suy diễn.
- Đơn vị hành chính lấy theo dữ liệu Wikidata tại ngày tải, có thể lẫn trước và sau đợt sáp nhập tỉnh năm 2025 (xem mục 4).
- Bảng mốc thời kỳ chỉ dựa theo năm, không theo vùng.
- Một số di tích nổi tiếng không vào được bộ dữ liệu vì thiếu ảnh dùng được (mục 5).
- Chùa Long Sơn có 4 ảnh được giữ gần như trùng nhau (cùng một góc chụp).
- KG ở mức thực thể, không có scene graph theo từng ảnh.

## 10. Điều kiện sử dụng

Chỉ dùng cho mục đích học thuật trong môn học.
- Ảnh thuộc quyền tác giả theo giấy phép của từng file. Khi hiển thị phải ghi công tác giả và giấy phép như trong `images.jsonl`.
- Văn bản lấy từ Wikipedia theo CC BY-SA 4.0.
- Không thu thập dữ liệu cá nhân. Ảnh có người là chủ thể chính đã bị loại khi duyệt.

## 11. Tái lập

- Thứ tự lệnh ở `data/README.md`, mục MMKG tiếng Việt.
- Cache thô (`data/raw/vnmmkg/`) nằm trên Drive và Hugging Face (link trong `data/README.md`); checksum ở `data/MANIFEST.md`.
- Chạy lại `python -m src.data.build_vnmmkg` từ cache cho ra đúng các file đã commit. Đã kiểm tra bằng `git diff` sau khi chạy lại.

## 12. Người thực hiện và phần AI hỗ trợ

- **Khang** chốt miền và nguồn dữ liệu, duyệt toàn bộ ảnh và triple, duyệt và khoá bộ truy vấn soạn tay.
- **Claude** (Claude Code) hỗ trợ viết thiết kế, kế hoạch, mã nguồn và kiểm thử. Claude cũng chạy pipeline theo yêu cầu của Khang, đề xuất các quyết định ở cổng G1/G2 (QID thời kỳ và lớp tỉnh, ánh xạ infobox, thêm P276/P527), soạn nháp bộ truy vấn soạn tay và data card.

Chi tiết ở `report/declaration-ai-oss.md`.
