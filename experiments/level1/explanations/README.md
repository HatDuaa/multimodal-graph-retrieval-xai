# Giải thích và 10 truy vấn phân tích (cấp độ 1, val)

Sinh bởi `scripts/explain_cases.py` (chọn ca, tính giải thích, phép thử xoá) và `scripts/render_explanations.py` (trang này). Dữ liệu gốc: `cases.json`. Lời phân tích: `analysis_vi.json`, viết sau khi xem ảnh thật. Mô hình: checkpoint tốt nhất theo val của `main_r3_seed0`, là mô hình chính đã chốt. Test không được dùng ở đây.

## Cách chọn truy vấn

- Chỉ xét câu val mà đồ thị câu có ít nhất một quan hệ (để hiện được cả kênh vật thể lẫn kênh bộ ba); 88,4% số câu val thoả điều kiện này.
- Bốc ngẫu nhiên với `numpy.random.default_rng(2026)`: 5 câu **thành công** (CLIP xếp ảnh đúng dưới hạng 1, mô hình xếp hạng 1), 3 câu **mô hình làm tệ đi** (CLIP hạng 1, mô hình dưới hạng 1), 2 câu **cả hai đều sai**.

Số câu mỗi loại trên toàn bộ 10 633 câu val (hạng tính trên top-50 của CLIP; ảnh đúng ngoài top-50 tính là sai):

| Checkpoint | Thành công | Mô hình làm tệ đi | Cả hai sai | Cả hai đúng | Mô hình xếp ảnh đúng cao hơn CLIP | Thấp hơn CLIP |
|---|---|---|---|---|---|---|
| `main_r3_seed0` | 1235 | 434 | 5237 | 3727 | 3987 | 1735 |
| `main_r3_seed1` | 1241 | 451 | 5231 | 3710 | 3978 | 1751 |
| `main_r3_seed2` | 1186 | 434 | 5286 | 3727 | 3975 | 1716 |

Với `main_r3_seed0`, số câu được sửa đúng gấp khoảng 2,8 lần số câu bị làm hỏng (1 235 so với 434), và R@1 bằng (1 235 + 3 727) / 10 633 = 46,67%, khớp số đã báo.

## Cách đọc bảng giải thích

Mọi số lấy thẳng từ các đại lượng mô hình dùng để xếp hạng (`src/explain/graph_explainer.py`), không tìm đường đi sau khi có kết quả. Với mỗi phần i của câu (vật thể hoặc bộ ba), bảng ghi phần j của ảnh có độ tương đồng lớn nhất, `w_i` (trọng số phần i trong câu), `a_ij` (attention của i lên j), `sim_ij` (cosine sau khi mã hoá), và đóng góp = trọng số kênh (b hoặc c) × w_i × a_ij × sim_ij. Mỗi dòng chỉ là cặp có attention lớn nhất, nên các đóng góp không cộng lại thành điểm cuối (điểm cuối còn qua z-score trong top-50).

## Một phát hiện từ giải thích: vector bộ ba bị ép sát nhau

Trong mọi ca, `sim_ij` của kênh bộ ba đều quanh 0,95–0,99, kể cả với cặp không liên quan (`woman on table` → `man wearing tie`). Đo trên đồ thị của 300 ảnh val (`scripts/checks/part_similarity.py`), cosine trung bình giữa hai phần bất kỳ:

| Mô hình | Vật thể | Bộ ba |
|---|---|---|
| `step0_frozen_clip` | 0,789 | 0,684 |
| `main_r3_seed0` | 0,271 | 0,967 |
| `main_r3_seed1` | 0,219 | 0,958 |
| `main_r3_seed2` | 0,274 | 0,974 |
| `no_gat_r3_seed0` | 0,611 | 0,972 |

Huấn luyện làm vector vật thể tách xa nhau hơn (0,79 → khoảng 0,25), nhưng lại dồn vector bộ ba về gần một hướng (0,68 → khoảng 0,97). Như vậy kênh bộ ba chỉ còn xếp hạng nhờ những khác biệt rất nhỏ được z-score phóng lên. Điều này khớp với các ablation: nối lại cạnh ngẫu nhiên gần như không làm giảm điểm, và phần lớn giải thích có ý nghĩa nằm ở kênh vật thể. Một nguyên nhân có thể là lớp `U` dùng chung cho vật thể và bộ ba (xem `docs/level1-model.md`). Chưa sửa, vì thiết kế đã chốt; ghi vào phần hạn chế.

## Thành công: đồ thị kéo ảnh đúng lên hạng 1

### Ca 1. "A den with a chandelier, chairs, couch and a television."

`546569_2` · ảnh đúng `2375034` · hạng theo CLIP: 4 · hạng theo mô hình: **1** · a / b / c = 0,44 / 0,45 / 0,11

Đồ thị câu: vật thể `den`, `chandelier`, `chair`, `couch`, `television`; bộ ba `den with chandelier`, `den with chair`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2375034.jpg) | ![](img/2375034.jpg) | ![](img/2366702.jpg) | ![](img/2346424.jpg) |
| `2375034` | `2375034` (CLIP hạng 4) — ảnh đúng | `2366702` (CLIP hạng 1) | `2346424` (CLIP hạng 8) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `den` | `ceiling` | 0,20 | 0,10 | 0,648 | 0,0059 |
| vật thể | `chandelier` | `chandlier` | 0,32 | 0,17 | 0,627 | 0,0150 |
| vật thể | `chair` | `chair` | 0,17 | 0,23 | 1,000 | 0,0182 |
| vật thể | `couch` | `living room` | 0,12 | 0,43 | 0,932 | 0,0225 |
| vật thể | `television` | `television` | 0,19 | 0,31 | 1,000 | 0,0267 |
| bộ ba | `den with chandelier` | `chandlier hanging from ceiling` | 0,60 | 0,07 | 0,996 | 0,0047 |
| bộ ba | `den with chair` | `chair in room` | 0,40 | 0,07 | 0,992 | 0,0029 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`television`, vật thể): hạng 1 → **1**, điểm 1,680 → 1,659. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 1,685.

**Phân tích.** Ảnh đúng là một phòng khách có đèn chùm, ghế bành, ghế sofa và TV. Ảnh CLIP xếp nhất có TV và ghế bành nhưng không có đèn chùm. Kênh vật thể khớp đủ `chandelier`, `chair`, `television` và `couch` (qua nhãn `living room`) nên kéo ảnh đúng từ hạng 4 lên hạng 1. Nhãn Visual Genome ghi sai chính tả là `chandlier`, nhưng vẫn được khớp với độ tương đồng 0,63. Bằng chứng trải trên nhiều vật thể, nên bỏ riêng `television` (phần đóng góp lớn nhất) ảnh đúng vẫn giữ hạng 1.

### Ca 2. "A man riding his bike near the beach."

`171351_3` · ảnh đúng `2349944` · hạng theo CLIP: 2 · hạng theo mô hình: **1** · a / b / c = 0,36 / 0,38 / 0,26

Đồ thị câu: vật thể `man`, `bike`, `beach`; bộ ba `man riding bike`, `man near beach`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2349944.jpg) | ![](img/2349944.jpg) | ![](img/2357349.jpg) | ![](img/2328865.jpg) |
| `2349944` | `2349944` (CLIP hạng 2) — ảnh đúng | `2357349` (CLIP hạng 1) | `2328865` (CLIP hạng 3) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `man` | `man` | 0,27 | 0,62 | 1,000 | 0,0643 |
| vật thể | `bike` | `bike` | 0,48 | 0,34 | 0,995 | 0,0622 |
| vật thể | `beach` | `beach` | 0,25 | 0,51 | 1,000 | 0,0482 |
| bộ ba | `man riding bike` | `bald man riding bike` | 0,65 | 0,13 | 0,996 | 0,0220 |
| bộ ba | `man near beach` | `bald man wearing shirt` | 0,35 | 0,12 | 0,949 | 0,0106 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`man`, vật thể): hạng 1 → **1**, điểm 2,588 → 2,411. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 2,610.

**Phân tích.** Ảnh đúng có người đàn ông đứng cạnh xe đạp trên kè biển, bãi cát và nước ở phía sau. Ảnh CLIP xếp nhất là một người đạp xe trên đường phố, không có biển. Cả ba vật thể của câu (`man`, `bike`, `beach`) đều có đúng nhãn trong đồ thị ảnh đúng (độ tương đồng khoảng 1,0), và `beach` là thứ ảnh kia không có. Bỏ `man` làm điểm giảm (2,59 → 2,41) nhưng ảnh vẫn hạng 1, vì `bike` và `beach` vẫn còn. Phần bộ ba đóng góp ít: `man near beach` được ghép với `bald man wearing shirt`.

### Ca 3. "Men standing around a sitting woman writing on table"

`382557_4` · ảnh đúng `2328541` · hạng theo CLIP: 6 · hạng theo mô hình: **1** · a / b / c = 0,39 / 0,38 / 0,23

Đồ thị câu: vật thể `man`, `woman`, `table`; bộ ba `man standing around woman`, `woman on table`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2328541.jpg) | ![](img/2328541.jpg) | ![](img/2327352.jpg) | ![](img/1159280.jpg) |
| `2328541` | `2328541` (CLIP hạng 6) — ảnh đúng | `2327352` (CLIP hạng 2) | `1159280` (CLIP hạng 4) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `man` | `man` | 0,33 | 0,09 | 1,000 | 0,0110 |
| vật thể | `woman` | `woman` | 0,45 | 0,38 | 1,000 | 0,0652 |
| vật thể | `table` | `table` | 0,22 | 0,30 | 1,000 | 0,0240 |
| bộ ba | `man standing around woman` | `man wearing name tag` | 0,52 | 0,04 | 0,990 | 0,0045 |
| bộ ba | `woman on table` | `man wearing tie` | 0,48 | 0,04 | 0,992 | 0,0042 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`woman`, vật thể): hạng 1 → **1**, điểm 1,589 → 1,503. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 1,585.

**Phân tích.** Ảnh đúng là cảnh một người phụ nữ ngồi ký giấy ở bàn, nhiều người đàn ông đứng quanh. Câu có đúng ba vật thể là `man`, `woman`, `table` và đồ thị ảnh đúng có đủ cả ba (`woman` có w = 0,45). Ảnh CLIP xếp nhất (hạng 4 sau khi xếp lại) là hai người đàn ông ngồi bên bàn có laptop, không có phụ nữ. Ảnh hạng 2 là một nhóm ngồi quanh bàn nếm rượu, trong đó người phụ nữ ngồi lẫn giữa mọi người và không có ai đứng quanh. Kênh bộ ba ở đây vô nghĩa: `woman on table` bị ghép với `man wearing tie`. Đây là ví dụ cho việc vector bộ ba đã bị ép sát nhau sau huấn luyện (mục đầu trang).

### Ca 4. "An old classic church is in front a big blue sky."

`513497_4` · ảnh đúng `2401726` · hạng theo CLIP: 3 · hạng theo mô hình: **1** · a / b / c = 0,51 / 0,32 / 0,17

Đồ thị câu: vật thể `church`, `front`, `sky`; bộ ba `church is sky`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2401726.jpg) | ![](img/2401726.jpg) | ![](img/2393194.jpg) | ![](img/2367339.jpg) |
| `2401726` | `2401726` (CLIP hạng 3) — ảnh đúng | `2393194` (CLIP hạng 4) | `2367339` (CLIP hạng 1) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `church` | `church` | 0,72 | 0,60 | 0,999 | 0,1380 |
| vật thể | `front` | `side` | 0,13 | 0,09 | 0,541 | 0,0020 |
| vật thể | `sky` | `sky` | 0,15 | 0,45 | 0,987 | 0,0222 |
| bộ ba | `church is sky` | `steeple on building` | 1,00 | 0,03 | 0,991 | 0,0058 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`church`, vật thể): hạng 1 → **4**, điểm 2,127 → 1,673. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 2,130.

**Phân tích.** Ảnh đúng là nhà thờ có tháp chuông trên nền trời xanh. Ảnh CLIP xếp nhất là một tháp gạch (hạng 3 sau khi xếp lại) và đồ thị của nó không có nhãn `church`. Vật thể `church` chiếm trọng số w = 0,72 trong câu và khớp đúng nhãn trong ảnh. **Bỏ `church` khỏi đồ thị ảnh đúng thì ảnh rơi từ hạng 1 xuống 4, còn bỏ ngẫu nhiên 10 phần khác thì trung bình vẫn hạng 1.** Parser tách sai `in front a` thành vật thể `front` và bộ ba `church is sky`, nhưng hai phần này có trọng số thấp nên không ảnh hưởng.

### Ca 5. "A man sitting on a bench alongside the curb of a busy street."

`206684_2` · ảnh đúng `1160012` · hạng theo CLIP: 6 · hạng theo mô hình: **1** · a / b / c = 0,38 / 0,45 / 0,17

Đồ thị câu: vật thể `man`, `bench`, `curb`, `street`; bộ ba `man sitting on bench`, `man alongside curb`, `curb of street`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/1160012.jpg) | ![](img/1160012.jpg) | ![](img/2357478.jpg) | ![](img/1159280.jpg) |
| `1160012` | `1160012` (CLIP hạng 6) — ảnh đúng | `2357478` (CLIP hạng 4) | `1159280` (CLIP hạng 2) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `man` | `man` | 0,27 | 0,69 | 1,000 | 0,0835 |
| vật thể | `bench` | `park bench` | 0,28 | 0,97 | 0,966 | 0,1186 |
| vật thể | `curb` | `park bench` | 0,24 | 0,41 | 0,543 | 0,0239 |
| vật thể | `street` | `man` | 0,21 | 0,37 | 0,451 | 0,0154 |
| bộ ba | `man sitting on bench` | `man with pant` | 0,29 | 0,50 | 0,955 | 0,0231 |
| bộ ba | `man alongside curb` | `man with pant` | 0,38 | 0,51 | 0,993 | 0,0324 |
| bộ ba | `curb of street` | `man with shirt` | 0,33 | 0,51 | 0,959 | 0,0273 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`park bench`, vật thể): hạng 1 → **14**, điểm 1,545 → 0,543. Xoá ngẫu nhiên 3 phần cùng loại: hạng trung bình 3,0, điểm trung bình 1,332.

**Phân tích.** Ảnh đúng có người đàn ông áo trắng ngồi trên ghế băng cạnh vỉa hè của một phố đông. Hai ảnh cạnh tranh (người đội mũ trùm ngồi trên bệ đá, và người ngồi trên khối vuông ở quảng trường) không có ghế băng cạnh lề đường. `bench` khớp `park bench` với a = 0,97. **Bỏ `park bench` làm ảnh đúng rơi từ hạng 1 xuống 14, còn bỏ ngẫu nhiên thì trung bình xuống hạng 3,0.** Đây là bằng chứng rõ nhất trong 10 ca rằng điểm đến từ đúng cặp khớp được hiển thị. `curb` và `street` không có nhãn tương ứng trong đồ thị ảnh nên bị ghép gượng với `park bench` và `man`.

## Thất bại: CLIP đúng, mô hình làm tệ đi

### Ca 6. "a child is standing outside in the grass"

`290979_2` · ảnh đúng `2331362` · hạng theo CLIP: 1 · hạng theo mô hình: **2** · a / b / c = 0,45 / 0,34 / 0,21

Đồ thị câu: vật thể `child`, `grass`; bộ ba `child in grass`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2331362.jpg) | ![](img/2380632.jpg) | ![](img/2331362.jpg) | ![](img/2374266.jpg) |
| `2331362` | `2380632` (CLIP hạng 2) | `2331362` (CLIP hạng 1) — ảnh đúng | `2374266` (CLIP hạng 5) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `child` | `boy` | 0,72 | 0,79 | 0,889 | 0,1736 |
| vật thể | `grass` | `grass` | 0,28 | 0,47 | 0,999 | 0,0441 |
| bộ ba | `child in grass` | `boy standing on grass` | 1,00 | 0,09 | 0,984 | 0,0176 |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2380632`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `child` | `child` | 0,72 | 0,44 | 1,000 | 0,1092 |
| vật thể | `grass` | `grass` | 0,28 | 0,55 | 0,998 | 0,0518 |
| bộ ba | `child in grass` | `grass in field` | 1,00 | 0,10 | 0,963 | 0,0195 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`child`, vật thể): hạng 1 → **1**, điểm 2,181 → 2,105. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 2,186.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`boy`, vật thể): hạng 2 → **3**, điểm 1,718 → 1,097. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 2,0, điểm trung bình 1,737.

**Phân tích.** Câu mơ hồ: 'a child is standing outside in the grass'. Ảnh đúng (cậu bé cởi trần đứng trên cỏ cạnh hồ) và ảnh mô hình xếp nhất (em bé ôm gấu bông đứng trong bãi cỏ) đều khớp câu. Khác biệt đến từ nhãn: ảnh mô hình chọn có đúng nhãn `child` (tương đồng 1,0), còn ảnh đúng gắn nhãn `boy` (0,89). Mô hình không sai theo nghĩa của câu. Đây là lỗi 'nhiều ảnh đúng' cộng với khác biệt từ vựng giữa caption COCO và nhãn Visual Genome.

### Ca 7. "a couple of airplanes lifting off the ground"

`503292_2` · ảnh đúng `2366949` · hạng theo CLIP: 1 · hạng theo mô hình: **2** · a / b / c = 0,37 / 0,37 / 0,25

Đồ thị câu: vật thể `couple`, `airplane`, `ground`; bộ ba `couple of airplane`, `airplane lifting ground`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2366949.jpg) | ![](img/2319085.jpg) | ![](img/2366949.jpg) | ![](img/2355290.jpg) |
| `2366949` | `2319085` (CLIP hạng 2) | `2366949` (CLIP hạng 1) — ảnh đúng | `2355290` (CLIP hạng 8) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `couple` | `runway` | 0,41 | 0,12 | 0,430 | 0,0079 |
| vật thể | `airplane` | `airplane` | 0,39 | 0,14 | 1,000 | 0,0196 |
| vật thể | `ground` | `ground` | 0,20 | 0,56 | 0,994 | 0,0411 |
| bộ ba | `couple of airplane` | `runway lined with plane` | 0,49 | 0,13 | 0,988 | 0,0162 |
| bộ ba | `airplane lifting ground` | `plane at gate` | 0,51 | 0,13 | 0,977 | 0,0167 |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2319085`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `couple` | `sign` | 0,41 | 0,11 | 0,532 | 0,0089 |
| vật thể | `airplane` | `airplane` | 0,39 | 0,08 | 1,000 | 0,0114 |
| vật thể | `ground` | `ground` | 0,20 | 0,70 | 0,998 | 0,0515 |
| bộ ba | `couple of airplane` | `sign near plane` | 0,49 | 0,05 | 0,990 | 0,0058 |
| bộ ba | `airplane lifting ground` | `plane on ground` | 0,51 | 0,05 | 0,986 | 0,0066 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`ground`, vật thể): hạng 1 → **2**, điểm 1,489 → 1,297. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 1,491.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`ground`, vật thể): hạng 2 → **2**, điểm 1,431 → 1,332. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 2,0, điểm trung bình 1,435.

**Phân tích.** Ảnh đúng là máy bay Delta đang cất cánh cạnh sân bay có nhiều máy bay khác. Ảnh mô hình xếp nhất cũng là một máy bay đang cất cánh, phía trước có một máy bay đỗ, tức cũng có 'a couple of airplanes'. Parser coi 'couple' là một vật thể và tạo bộ ba `couple of airplane`, nên vật thể `couple` được ghép gượng với `runway` hoặc `sign`. Phần quyết định là `ground` (a = 0,70 ở ảnh mô hình chọn, 0,56 ở ảnh đúng). **Bỏ `ground` khỏi ảnh mô hình chọn thì nó rơi xuống hạng 2 và ảnh đúng trở lại hạng 1**, tức đúng phần được hiển thị đã quyết định thứ hạng.

### Ca 8. "High stone tower with windows in an old village."

`245173_1` · ảnh đúng `2372939` · hạng theo CLIP: 1 · hạng theo mô hình: **2** · a / b / c = 0,54 / 0,32 / 0,14

Đồ thị câu: vật thể `stone tower`, `window`, `village`; bộ ba `stone tower with window`, `stone tower in village`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2372939.jpg) | ![](img/2360229.jpg) | ![](img/2372939.jpg) | ![](img/2380649.jpg) |
| `2372939` | `2360229` (CLIP hạng 2) | `2372939` (CLIP hạng 1) — ảnh đúng | `2380649` (CLIP hạng 3) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `stone tower` | `building` | 0,44 | 0,19 | 0,762 | 0,0202 |
| vật thể | `window` | `window` | 0,25 | 0,24 | 1,000 | 0,0188 |
| vật thể | `village` | `building` | 0,31 | 0,21 | 0,817 | 0,0165 |
| bộ ba | `stone tower with window` | `brick wall on brick building` | 0,51 | 0,07 | 0,995 | 0,0047 |
| bộ ba | `stone tower in village` | `sun shining on top half of brick building` | 0,49 | 0,07 | 0,993 | 0,0048 |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2360229`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `stone tower` | `roof tower` | 0,44 | 0,19 | 0,759 | 0,0200 |
| vật thể | `window` | `window` | 0,25 | 0,87 | 1,000 | 0,0691 |
| vật thể | `village` | `building` | 0,31 | 0,16 | 0,814 | 0,0130 |
| bộ ba | `stone tower with window` | `building with window` | 0,51 | 0,05 | 0,997 | 0,0038 |
| bộ ba | `stone tower in village` | `corner of building` | 0,49 | 0,05 | 0,995 | 0,0037 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`window`, vật thể): hạng 1 → **2**, điểm 2,400 → 1,995. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 2,399.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`building`, vật thể): hạng 2 → **2**, điểm 2,294 → 2,179. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 2,0, điểm trung bình 2,297.

**Phân tích.** Ảnh đúng là tháp đá cổ giữa phố cổ. Đồ thị Visual Genome của nó nghèo: không có nhãn `tower` hay `village`, chỉ có `building`, `window` và các bộ ba dài như 'sun shining on top half of brick building'. Ảnh mô hình xếp nhất là cận cảnh tháp chuông có đồng hồ, có nhãn `roof tower` và nhiều `window` (a = 0,87). **Bỏ `window` khỏi ảnh đó thì nó rơi xuống hạng 2 và ảnh đúng lên lại hạng 1.** Lỗi đến từ chú thích Visual Genome thiếu nhãn ở ảnh đúng, không phải từ câu truy vấn.

## Thất bại: cả CLIP và mô hình đều sai

### Ca 9. "there is a man riding a bike up the road"

`401068_1` · ảnh đúng `2392790` · hạng theo CLIP: ngoài top-50 · hạng theo mô hình: **ngoài top-50** · a / b / c = 0,35 / 0,44 / 0,21

Đồ thị câu: vật thể `man`, `bike`, `road`; bộ ba `man riding bike`, `man up road`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2392790.jpg) | ![](img/2357349.jpg) | ![](img/2407430.jpg) | ![](img/2355050.jpg) |
| `2392790` | `2357349` (CLIP hạng 2) | `2407430` (CLIP hạng 12) | `2355050` (CLIP hạng 16) |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2357349`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `man` | `man` | 0,28 | 0,56 | 1,000 | 0,0684 |
| vật thể | `bike` | `bike` | 0,49 | 0,31 | 1,000 | 0,0665 |
| vật thể | `road` | `road` | 0,23 | 0,15 | 0,978 | 0,0146 |
| bộ ba | `man riding bike` | `man above bicycle` | 0,51 | 0,05 | 0,992 | 0,0054 |
| bộ ba | `man up road` | `man has leg` | 0,49 | 0,05 | 0,987 | 0,0049 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`man`, vật thể): hạng 1 → **1**, điểm 2,088 → 1,650. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 2,072.

**Phân tích.** Ảnh đúng là người lái mô tô (caption COCO dùng 'bike' cho xe máy), và nó nằm ngoài top-50 của CLIP nên bộ xếp lại không thể cứu. Ba ảnh mô hình xếp đầu (người đạp xe trên đường, người lái xe máy nhỏ trên phố, người trượt ván trên đường dốc) đều hợp nghĩa 'man riding a bike up the road' theo cách hiểu thông thường. Đây là giới hạn của khâu lấy ứng viên (CLIP top-50) cộng với từ 'bike' đa nghĩa.

### Ca 10. "a tennis player with a ball and a racket"

`398534_1` · ảnh đúng `2325149` · hạng theo CLIP: 17 · hạng theo mô hình: **14** · a / b / c = 0,31 / 0,53 / 0,16

Đồ thị câu: vật thể `tennis player`, `ball`, `racket`; bộ ba `tennis player with ball`, `racket with ball`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2325149.jpg) | ![](img/2389999.jpg) | ![](img/2368474.jpg) | ![](img/2415580.jpg) |
| `2325149` | `2389999` (CLIP hạng 5) | `2368474` (CLIP hạng 1) | `2415580` (CLIP hạng 3) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `tennis player` | `racket` | 0,30 | 0,33 | 0,959 | 0,0511 |
| vật thể | `ball` | `ball` | 0,34 | 0,47 | 1,000 | 0,0836 |
| vật thể | `racket` | `racket` | 0,35 | 0,48 | 1,000 | 0,0897 |
| bộ ba | `tennis player with ball` | `man has shoulder` | 0,52 | 0,07 | 0,974 | 0,0062 |
| bộ ba | `racket with ball` | `man has shoulder` | 0,48 | 0,07 | 0,974 | 0,0057 |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2389999`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `tennis player` | `racket` | 0,30 | 0,89 | 0,959 | 0,1370 |
| vật thể | `ball` | `ball` | 0,34 | 0,95 | 1,000 | 0,1695 |
| vật thể | `racket` | `racket` | 0,35 | 0,94 | 1,000 | 0,1757 |
| bộ ba | `tennis player with ball` | `tattoo on wrist` | 0,52 | 0,14 | 0,973 | 0,0115 |
| bộ ba | `racket with ball` | `tattoo on wrist` | 0,48 | 0,14 | 0,973 | 0,0107 |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`racket`, vật thể): hạng 1 → **25**, điểm 0,961 → 0,053. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 2,5, điểm trung bình 0,928.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`racket`, vật thể): hạng 14 → **19**, điểm 0,384 → 0,260. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 15,3, điểm trung bình 0,367.

**Phân tích.** Câu rất chung: 'a tennis player with a ball and a racket' khớp hầu hết ảnh tennis trong pool. Ảnh đúng lên từ hạng 17 (CLIP) lên 14. Visual Genome không có nhãn `tennis player` (chỉ có `man`, `woman`), nên vật thể này bị ghép với `racket`. Ảnh mô hình xếp nhất có quả bóng và vợt rất rõ ở tiền cảnh (a ≈ 0,9 cho cả hai), vì thế thắng. **Bỏ `racket` khỏi ảnh đó làm nó rơi từ hạng 1 xuống 25, còn bỏ ngẫu nhiên thì trung bình ở hạng 2,5.** Kênh bộ ba ghép `tennis player with ball` với `tattoo on wrist`, một ví dụ nữa về vector bộ ba bị ép sát.

## Tổng hợp phép thử xoá

Xoá trên ảnh mô hình xếp hạng 1 của mỗi ca: phần có đóng góp lớn nhất so với trung bình 10 phần ngẫu nhiên cùng loại.

| Ca | Phần bị xoá | Hạng trước | Hạng sau (phần lớn nhất) | Hạng sau (ngẫu nhiên, trung bình) | Giảm điểm (lớn nhất) | Giảm điểm (ngẫu nhiên) |
|---|---|---|---|---|---|---|
| 1 | `television` | 1 | 1 | 1,0 | 0,021 | -0,005 |
| 2 | `man` | 1 | 1 | 1,0 | 0,177 | -0,023 |
| 3 | `woman` | 1 | 1 | 1,0 | 0,087 | 0,004 |
| 4 | `church` | 1 | 4 | 1,0 | 0,454 | -0,003 |
| 5 | `park bench` | 1 | 14 | 3,0 | 1,002 | 0,213 |
| 6 | `child` | 1 | 1 | 1,0 | 0,076 | -0,005 |
| 7 | `ground` | 1 | 2 | 1,0 | 0,191 | -0,002 |
| 8 | `window` | 1 | 2 | 1,0 | 0,405 | 0,000 |
| 9 | `man` | 1 | 1 | 1,0 | 0,437 | 0,015 |
| 10 | `racket` | 1 | 25 | 2,5 | 0,908 | 0,034 |

Ở 10/10 ca, xoá phần có đóng góp lớn nhất làm điểm giảm nhiều hơn xoá ngẫu nhiên. Ở 5/10 ca, riêng việc xoá một phần này đã làm ảnh rơi khỏi hạng 1. Các ca còn lại giữ hạng vì bằng chứng trải trên nhiều vật thể. Đây là phép thử fidelity mức cơ bản; đo đầy đủ trên ≥10 truy vấn với các chỉ số fidelity / validity / sparsity thuộc cấp độ 3.
