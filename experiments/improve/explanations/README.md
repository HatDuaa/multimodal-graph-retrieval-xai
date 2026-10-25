# Giải thích và 10 truy vấn phân tích — mô hình chính `countonly_factual_r3` (val)

Sinh bởi `scripts/improve/explain_cases_countonly.py` (chọn ca, giải thích, phép thử xoá, so với cấp 1), `scripts/improve/explain_counterfactuals.py` (bỏ g / gỡ gộp lúc eval) và `scripts/improve/render_explanations_countonly.py` (trang này). Dữ liệu gốc: `cases.json`, `counterfactuals.json`. Lời phân tích: `analysis_vi.json`, viết sau khi xem ảnh thật. Mô hình: checkpoint tốt nhất theo val của `countonly_factual_r3_seed0` (seed có R@1 val cao nhất trong 3 seed). Test không được dùng ở đây. Bản tương ứng của mô hình cấp 1: `experiments/level1/explanations/`.

## Cách chọn truy vấn

Giống hệt cấp 1 (`scripts/explain_cases.py`, dùng lại các hàm của nó):

- Chỉ xét câu val mà đồ thị câu FACTUAL có ít nhất một quan hệ; 97,3% số câu val thoả điều kiện này.
- Bốc ngẫu nhiên với `numpy.random.default_rng(2026)`: 5 câu **thành công** (CLIP xếp ảnh đúng dưới hạng 1, mô hình xếp hạng 1), 3 câu **mô hình làm tệ đi** (CLIP hạng 1, mô hình dưới hạng 1), 2 câu **cả hai đều sai**.

Số câu mỗi loại trên toàn bộ 10 633 câu val (hạng tính trên top-50 của CLIP; ảnh đúng ngoài top-50 tính là sai). R@1 tính lại ở đây trùng với `metrics.json` của từng run:

| Checkpoint | Thành công | Mô hình làm tệ đi | Cả hai sai | Cả hai đúng | Mô hình xếp ảnh đúng cao hơn CLIP | Thấp hơn CLIP | R@1 |
|---|---|---|---|---|---|---|---|
| `countonly_factual_r3_seed0` | 1528 | 463 | 4944 | 3698 | 4211 | 1689 | 49,15 |
| `countonly_factual_r3_seed1` | 1548 | 515 | 4924 | 3646 | 4164 | 1801 | 48,85 |
| `countonly_factual_r3_seed2` | 1483 | 462 | 4989 | 3699 | 4155 | 1706 | 48,74 |
| `main_r3_seed0` | 1235 | 434 | 5237 | 3727 | 3987 | 1735 | 46,67 |
| `main_r3_seed1` | 1241 | 451 | 5231 | 3710 | 3978 | 1751 | 46,56 |
| `main_r3_seed2` | 1186 | 434 | 5286 | 3727 | 3975 | 1716 | 46,21 |

Với seed 0, mô hình mới sửa đúng 1528 câu CLIP sai và làm hỏng 463 câu CLIP đúng, tỉ lệ 3,3 lần. `main_r3_seed0` là 1235 / 434 (2,8 lần). R@1 = (1528 + 3698) / 10 633 = 49,15%.

## Cách đọc bảng giải thích

Mọi số lấy thẳng từ lần xếp hạng (`src/explain/graph_explainer.py`). Mô hình dùng fusion raw:

điểm cuối = a × cosine CLIP + b × S(vật thể) + c × S(bộ ba), với S = Σ_i w_i · Σ_j a_ij · sim_ij.

Vì không có z-score trong top-50, mỗi dòng là một phần **chính xác** của điểm cuối, khác với bảng cấp 1:

- Một dòng = trọng số kênh × w_i × (Σ_j a_ij · sim_ij) (cột "Σ a·sim"). Các phần cộng lại đúng bằng điểm cuối.
- Cột "Phần của ảnh" ghi phần j có attention lớn nhất, kèm a_ij và sim_ij của nó.
- Với vật thể có số lượng nêu trong câu, sim_ij đã gồm g[lớp số lượng của câu, nhóm số lượng của node ảnh]. Lớp và nhóm ghi trong ngoặc, g ở cột riêng.
- Ảnh không có phần nào cho một kênh: fusion raw lấy điểm kênh trung bình của các ứng viên, ghi thành dòng "cả kênh".
- Cạnh kNN `[none]` và node tổng chỉ dùng lúc truyền tin trong GAT, không thành dòng.

Phép thử xoá chọn **phần của ảnh** có đóng góp chính xác lớn nhất, term_j = trọng số kênh × Σ_i w_i · a_ij · sim_ij. Đây là cùng phần điểm đó, nhưng chia theo phần của ảnh thay vì phần của câu. Sau khi xoá một vật thể, nhóm số lượng của ảnh được tính lại theo đúng luật count_v1.

## Phát hiện: kênh bộ ba nặng ký nhưng khớp trải đều

Trong 10 ca, attention của cặp khớp nhất trung bình là 0,51 ở kênh vật thể nhưng chỉ 0,17 ở kênh bộ ba (41 và 26 dòng). Một dòng bộ ba vì vậy là trung bình của gần như mọi bộ ba trong ảnh, nên cặp "khớp nhất" thường vô nghĩa (`cow standing ear`, `man wearing pant`). Phần điểm của kênh này lớn, nhưng gần như bằng nhau giữa ảnh đúng và ảnh mô hình xếp hạng 1:

| Ca | Ảnh | CLIP | Vật thể | Bộ ba | Điểm cuối |
|---|---|---|---|---|---|
| 6 | đúng | 0,1667 | 0,0767 | 0,3077 | 0,5511 |
| 6 | hạng 1 | 0,1622 | 0,0905 | 0,3120 | 0,5648 |
| 7 | đúng | 0,1618 | 0,0722 | 0,2689 | 0,5029 |
| 7 | hạng 1 | 0,1573 | 0,0835 | 0,2721 | 0,5129 |
| 8 | đúng | 0,2414 | 0,0528 | 0,1365 | 0,4307 |
| 8 | hạng 1 | 0,2359 | 0,0615 | 0,1372 | 0,4345 |
| 9 | đúng | 0,1251 | 0,0934 | 0,3372 | 0,5558 |
| 9 | hạng 1 | 0,1510 | 0,1168 | 0,3429 | 0,6107 |
| 10 | đúng | 0,0834 | 0,0405 | 0,5215 | 0,6454 |
| 10 | hạng 1 | 0,0883 | 0,0515 | 0,5328 | 0,6725 |

Chênh lệch giữa hai ảnh chủ yếu nằm ở CLIP và kênh vật thể. Phép thử xoá cho thấy cùng điều đó: xoá một vật thể thường đổi hạng, còn xoá bộ ba "đóng góp lớn nhất" thì gần như không. Điều này khớp với phép đo trước (`experiments/improve/count-only/part_similarity.json`): vector bộ ba của countonly vẫn bị ép sát nhau, cosine trung bình giữa hai bộ ba bất kỳ 0,875 / 0,899 / 0,867 ở 3 seed, so với 0,684 của CLIP đông cứng và khoảng 0,97 ở `main_r3`.

## Thành công: đồ thị kéo ảnh đúng lên hạng 1

### Ca 1. "Dining table for six in a wood paneled room with a fireplace."

`302038_1` · ảnh đúng `2377477` · hạng theo CLIP: 2 · hạng theo mô hình: **1** · a / b / c = 0,60 / 0,16 / 0,24

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `dining table`, `room`, `fireplace`; bộ ba `dining table in room`, `room with fireplace`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2377477.jpg) | ![](img/2377477.jpg) | ![](img/2336567.jpg) | ![](img/2379623.jpg) |
| `2377477` | `2377477` (CLIP hạng 2) — ảnh đúng | `2336567` (CLIP hạng 23) | `2379623` (CLIP hạng 13) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,60 | cả câu | cả ảnh |  |  | cosine 0,2805 |  |  | 0,1684 | 38% |
| vật thể 0,16 | `fireplace` | `stove` | 0,46 | 0,30 | 0,745 |  | 0,671 | 0,0484 | 11% |
| vật thể 0,16 | `dining table` | `table` | 0,30 | 0,66 | 0,850 |  | 0,774 | 0,0368 | 8% |
| vật thể 0,16 | `room` | `room` | 0,24 | 0,63 | 0,915 |  | 0,837 | 0,0312 | 7% |
| bộ ba 0,24 | `room with fireplace` | `wood under fire` | 0,65 | 0,09 | 0,640 |  | 0,622 | 0,0976 | 22% |
| bộ ba 0,24 | `dining table in room` | `wood under fire` | 0,35 | 0,10 | 0,690 |  | 0,668 | 0,0573 | 13% |
| **Tổng** |  |  |  |  |  |  |  | **0,4397** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`table`, vật thể, phần điểm 0,0284): hạng 1 → **2**, điểm 0,4397 → 0,4327. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,4394.

**Phân tích.** Ảnh đúng là một phòng ăn: bàn gỗ có ghế quanh, xà gỗ trên trần, và một lò sưởi đốt củi kiểu bếp lò đang cháy. Hai ảnh xếp sau là phòng khách có lò sưởi nhưng không có bàn ăn. Đồ thị chọn đúng các phần phân biệt. `dining table` khớp với `table` (sim 0,850). `fireplace` khớp với `stove` (0,745): nhãn VG gọi chiếc lò là stove, nhưng đó đúng là thứ câu gọi là fireplace. Xoá node `table` làm ảnh rơi xuống hạng 2, trong khi xoá ngẫu nhiên 10 vật thể gần như không đổi điểm. Hai dòng bộ ba đều trỏ vào `wood under fire`, tức đống củi trong lò, với attention chỉ khoảng 0,1. Đây là kiểu khớp bộ ba "trải đều" lặp lại ở hầu hết các ca (xem mục phát hiện).

### Ca 2. "A cow standing along side of a large river."

`422698_1` · ảnh đúng `2351201` · hạng theo CLIP: 8 · hạng theo mô hình: **1** · a / b / c = 0,50 / 0,11 / 0,39

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `cow`, `river`; bộ ba `cow stand alongside river`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2351201.jpg) | ![](img/2351201.jpg) | ![](img/2369610.jpg) | ![](img/2400010.jpg) |
| `2351201` | `2351201` (CLIP hạng 8) — ảnh đúng | `2369610` (CLIP hạng 7) | `2400010` (CLIP hạng 1) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,50 | cả câu | cả ảnh |  |  | cosine 0,2962 |  |  | 0,1485 | 31% |
| vật thể 0,11 | `river` | `river` | 0,61 | 0,51 | 0,950 |  | 0,905 | 0,0594 | 12% |
| vật thể 0,11 | `cow` | `cow` | 0,39 | 0,48 | 0,945 |  | 0,934 | 0,0395 | 8% |
| bộ ba 0,39 | `cow stand alongside river` | `cow standing ear` | 1,00 | 0,23 | 0,637 |  | 0,609 | 0,2380 | 49% |
| **Tổng** |  |  |  |  |  |  |  | **0,4854** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`cow standing ear`, bộ ba, phần điểm 0,0584): hạng 1 → **1**, điểm 0,4854 → 0,4820. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,4856.

**Phân tích.** Ảnh đúng là một con bò đứng giữa đám cỏ dại sát mép một dòng sông rộng. Ảnh CLIP xếp hạng 1 (2400010) là con bò ở một phiên chợ, cạnh người dắt, không có sông; mô hình đẩy nó xuống hạng 3. Câu đặt 61% trọng số kênh vật thể vào `river`, và `river` khớp đúng `river` (0,950), nên phần phân biệt ở đây là dòng sông. Ảnh hạng 2 (bò uống nước dưới sông) cũng có sông nhưng có nhiều con bò; câu không nêu số lượng nên g không tác động. Phần ảnh có đóng góp lớn nhất lại là bộ ba `cow standing ear`, một nhãn VG vô nghĩa. Xoá nó chỉ làm điểm giảm 0,003 và ảnh vẫn hạng 1. Như vậy dòng bộ ba to nhất trong bảng không phải là bằng chứng mô hình thực sự dựa vào: attention của nó chỉ 0,23, phần còn lại trải trên 12 bộ ba khác.

### Ca 3. "A man sitting in front of the pen with a dog herding sheep in the background."

`357943_3` · ảnh đúng `2329851` · hạng theo CLIP: 3 · hạng theo mô hình: **1** · a / b / c = 0,47 / 0,26 / 0,27

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `dog`, `sheep` (>=2), `man`, `pen`, `background`; bộ ba `dog herd sheep`, `man sit in front of pen`, `man with dog`, `sheep in background`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2329851.jpg) | ![](img/2329851.jpg) | ![](img/2361529.jpg) | ![](img/2404746.jpg) |
| `2329851` | `2329851` (CLIP hạng 3) — ảnh đúng | `2361529` (CLIP hạng 2) | `2404746` (CLIP hạng 5) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,47 | cả câu | cả ảnh |  |  | cosine 0,3317 |  |  | 0,1569 | 27% |
| vật thể 0,26 | `dog` | `dog` | 0,28 | 0,45 | 0,928 |  | 0,908 | 0,0666 | 12% |
| vật thể 0,26 | `pen` | `pen` | 0,23 | 0,93 | 0,942 |  | 0,919 | 0,0542 | 9% |
| vật thể 0,26 | `sheep` (>=2) | `sheep` (3-4) | 0,19 | 0,28 | 0,958 | +0,035 | 0,943 | 0,0459 | 8% |
| vật thể 0,26 | `man` | `trainer` | 0,18 | 0,91 | 0,640 |  | 0,608 | 0,0278 | 5% |
| vật thể 0,26 | `background` | `sheep` | 0,12 | 0,23 | 0,769 |  | 0,718 | 0,0229 | 4% |
| bộ ba 0,27 | `dog herd sheep` | `trainer observing dog` | 0,34 | 0,13 | 0,826 |  | 0,812 | 0,0743 | 13% |
| bộ ba 0,27 | `man with dog` | `trainer observing dog` | 0,30 | 0,24 | 0,751 |  | 0,724 | 0,0587 | 10% |
| bộ ba 0,27 | `man sit in front of pen` | `trainer opening gate` | 0,19 | 0,21 | 0,653 |  | 0,629 | 0,0323 | 6% |
| bộ ba 0,27 | `sheep in background` | `trainer opening gate` | 0,17 | 0,12 | 0,727 |  | 0,717 | 0,0318 | 6% |
| **Tổng** |  |  |  |  |  |  |  | **0,5714** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`pen`, vật thể, phần điểm 0,0523): hạng 1 → **2**, điểm 0,5714 → 0,5530. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,1, điểm trung bình 0,5698.

**Phân tích.** Ảnh đúng là một bãi cỏ có hàng rào chuồng bằng kim loại trắng. Một người đàn ông đi bên cạnh (không ngồi như câu nói), một chú chó border collie đang lùa vài con cừu ở phía sau. Ảnh hạng 2 (người ngồi với cừu và chó trên núi) hợp với chữ "sitting" hơn nhưng không có chuồng. `pen` khớp `pen` với attention 0,93 và là phần ảnh đóng góp lớn nhất. Xoá nó làm ảnh rơi xuống hạng 2, điểm giảm 0,018, gấp khoảng 11 lần xoá ngẫu nhiên. Dòng `sheep` có số lượng: câu nói "sheep" (lớp ≥2), node `sheep` của ảnh thuộc nhóm 3–4, g cộng +0,035. Đây là một khớp số lượng đúng nhưng nhỏ. `man` khớp `trainer` (0,640), đúng người trong ảnh.

### Ca 4. "Someone holding a doughnut that has a bite taken out of it."

`445695_1` · ảnh đúng `2402283` · hạng theo CLIP: 4 · hạng theo mô hình: **1** · a / b / c = 0,55 / 0,11 / 0,34

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `bite`, `doughnut`, `person`; bộ ba `bite out of doughnut`, `person hold doughnut`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2402283.jpg) | ![](img/2402283.jpg) | ![](img/2320230.jpg) | ![](img/2383048.jpg) |
| `2402283` | `2402283` (CLIP hạng 4) — ảnh đúng | `2320230` (CLIP hạng 3) | `2383048` (CLIP hạng 1) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,55 | cả câu | cả ảnh |  |  | cosine 0,2789 |  |  | 0,1528 | 37% |
| vật thể 0,11 | `doughnut` | `donut` | 0,34 | 0,84 | 0,887 |  | 0,848 | 0,0317 | 8% |
| vật thể 0,11 | `person` | `person` | 0,35 | 0,62 | 0,905 |  | 0,834 | 0,0313 | 7% |
| vật thể 0,11 | `bite` | `bite` | 0,31 | 0,85 | 0,883 |  | 0,846 | 0,0285 | 7% |
| bộ ba 0,34 | `person hold doughnut` | `person holding donut` | 0,68 | 0,18 | 0,547 |  | 0,497 | 0,1163 | 28% |
| bộ ba 0,34 | `bite out of doughnut` | `bite taken out of donut` | 0,32 | 0,11 | 0,567 |  | 0,524 | 0,0574 | 14% |
| **Tổng** |  |  |  |  |  |  |  | **0,4181** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`donut`, vật thể, phần điểm 0,0298): hạng 1 → **3**, điểm 0,4181 → 0,4068. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,1, điểm trung bình 0,4180.

**Phân tích.** Ảnh đúng là một bàn tay cầm chiếc donut nhân sô-cô-la đã bị cắn dở. Ảnh CLIP xếp hạng 1 (2383048) là chiếc cupcake rắc cốm cầm trên tay, nhìn rất giống về màu và bố cục; mô hình đẩy nó xuống hạng 3. Ba vật thể của câu đều khớp đúng nhãn: `doughnut`→`donut`, `person`→`person`, và cả `bite`→`bite`, vì VG có ghi vết cắn. Bộ ba `person hold doughnut` khớp `person holding donut`. Xoá node `donut` làm ảnh rơi xuống hạng 3. Như vậy chính vật thể donut, thứ ảnh cupcake không có, kéo ảnh đúng lên.

### Ca 5. "People huddle under several umbrellas on a sidewalk area and a white dog follows people with a pink umbrella."

`574350_4` · ảnh đúng `1159610` · hạng theo CLIP: 2 · hạng theo mô hình: **1** · a / b / c = 0,67 / 0,19 / 0,13

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `dog`, `people` (>=2), `umbrella` (>=3), `sidewalk`, `umbrella`; bộ ba `dog follow people`, `people huddle under umbrella`, `people on sidewalk`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/1159610.jpg) | ![](img/1159610.jpg) | ![](img/2345853.jpg) | ![](img/2378292.jpg) |
| `1159610` | `1159610` (CLIP hạng 2) — ảnh đúng | `2345853` (CLIP hạng 1) | `2378292` (CLIP hạng 3) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,67 | cả câu | cả ảnh |  |  | cosine 0,3017 |  |  | 0,2027 | 44% |
| vật thể 0,19 | `dog` | `dog` | 0,27 | 1,00 | 0,895 |  | 0,893 | 0,0469 | 10% |
| vật thể 0,19 | `umbrella` | `umbrella` | 0,19 | 0,61 | 0,937 |  | 0,917 | 0,0346 | 8% |
| vật thể 0,19 | `umbrella` (>=3) | `umbrella` (2) | 0,19 | 0,54 | 0,884 | −0,046 | 0,855 | 0,0323 | 7% |
| vật thể 0,19 | `people` (>=2) | `man` (3-4) | 0,20 | 0,31 | 0,889 | +0,035 | 0,840 | 0,0320 | 7% |
| vật thể 0,19 | `sidewalk` | `jean` | 0,14 | 0,18 | 0,676 |  | 0,623 | 0,0174 | 4% |
| bộ ba 0,13 | `dog follow people` | `woman has pant` | 0,49 | 0,22 | 0,676 |  | 0,669 | 0,0444 | 10% |
| bộ ba 0,13 | `people on sidewalk` | `man has pant` | 0,25 | 0,24 | 0,791 |  | 0,782 | 0,0258 | 6% |
| bộ ba 0,13 | `people huddle under umbrella` | `man has pant` | 0,26 | 0,24 | 0,707 |  | 0,696 | 0,0243 | 5% |
| **Tổng** |  |  |  |  |  |  |  | **0,4603** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh đúng (`dog`, vật thể, phần điểm 0,0469): hạng 1 → **3**, điểm 0,4603 → 0,4257. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,4602.

**Phân tích.** Ảnh đúng là một quảng trường lát đá: vài người che ô xanh và ô hồng đi bộ, một chú chó trắng nhỏ đi theo sau. Ảnh CLIP xếp hạng 1 (2345853) là ô màu chụp từ trên cao trên vạch qua đường, không có chó. `dog`→`dog` có attention 1,00 và là phần ảnh đóng góp lớn nhất. Xoá nó làm ảnh rơi xuống hạng 3, còn xoá 10 vật thể ngẫu nhiên không đổi gì. Hai dòng có số lượng cho thấy g hoạt động theo cả hai chiều. "several umbrellas" (lớp ≥3) gặp node `umbrella` nhóm 2, nên g trừ 0,046. "People" (≥2) gặp `man` nhóm 3–4, nên g cộng 0,035. Ở đây g chỉ làm điểm dịch chút ít; con chó mới là phần quyết định. `sidewalk` khớp nhầm `jean` (0,676) vì đồ thị ảnh không có node vỉa hè.

## Thất bại: CLIP đúng, mô hình làm tệ đi

### Ca 6. "A bunch of bananas are sitting on a wooden plate."

`1153_3` · ảnh đúng `2347327` · hạng theo CLIP: 1 · hạng theo mô hình: **2** · a / b / c = 0,50 / 0,10 / 0,40

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `banana` (>=2), `plate`; bộ ba `banana sit on plate`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2347327.jpg) | ![](img/2366682.jpg) | ![](img/2347327.jpg) | ![](img/2368470.jpg) |
| `2347327` | `2366682` (CLIP hạng 2) | `2347327` (CLIP hạng 1) — ảnh đúng | `2368470` (CLIP hạng 10) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,50 | cả câu | cả ảnh |  |  | cosine 0,3351 |  |  | 0,1667 | 30% |
| vật thể 0,10 | `banana` (>=2) | `banana` (>=5) | 0,56 | 0,16 | 0,969 | +0,037 | 0,948 | 0,0539 | 10% |
| vật thể 0,10 | `plate` | `tray` | 0,44 | 0,40 | 0,611 |  | 0,520 | 0,0229 | 4% |
| bộ ba 0,40 | `banana sit on plate` | `bowl under banana` | 1,00 | 0,08 | 0,780 |  | 0,766 | 0,3077 | 56% |
| **Tổng** |  |  |  |  |  |  |  | **0,5511** | 100% |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2366682`):

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,50 | cả câu | cả ảnh |  |  | cosine 0,3261 |  |  | 0,1622 | 29% |
| vật thể 0,10 | `banana` (>=2) | `banana` (>=5) | 0,56 | 0,15 | 0,968 | +0,037 | 0,954 | 0,0542 | 10% |
| vật thể 0,10 | `plate` | `plate` | 0,44 | 0,59 | 0,861 |  | 0,826 | 0,0363 | 6% |
| bộ ba 0,40 | `banana sit on plate` | `banana in plate` | 1,00 | 0,22 | 0,789 |  | 0,777 | 0,3120 | 55% |
| **Tổng** |  |  |  |  |  |  |  | **0,5648** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`banana in plate`, bộ ba, phần điểm 0,0693): hạng 1 → **1**, điểm 0,5648 → 0,5632. Xoá ngẫu nhiên 6 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,5650.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`bowl under banana`, bộ ba, phần điểm 0,0260): hạng 2 → **2**, điểm 0,5511 → 0,5508. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 2,0, điểm trung bình 0,5512.

**Phân tích.** Ảnh đúng là một nải chuối đặt trên một chiếc đĩa gỗ tròn trong bếp. Ảnh mô hình xếp hạng 1 là chuối đã bóc vỏ chất trên đĩa sứ trắng. Hai ảnh cùng có `banana` nhóm ≥5, cùng được g +0,037, nên số lượng không phân biệt được. Chỗ lệch nằm ở chữ "plate". VG gọi chiếc đĩa gỗ của ảnh đúng là `tray`, nên `plate`→`tray` chỉ được sim 0,611; ảnh kia có node `plate` thật (0,861). Riêng dòng này chênh 0,013 cho ảnh sai, lớn hơn phần CLIP mà ảnh đúng hơn (0,0045). Chữ "wooden", thứ phân biệt hai ảnh, nằm trong thuộc tính của FACTUAL nhưng mô hình chỉ dùng tên node. Xoá bộ ba lớn nhất của ảnh sai hầu như không đổi gì: hai ảnh hơn nhau ở kênh vật thể chứ không ở bộ ba.

### Ca 7. "A train riding along the tracks along a bank of dirt."

`279933_1` · ảnh đúng `2391578` · hạng theo CLIP: 1 · hạng theo mô hình: **5** · a / b / c = 0,51 / 0,10 / 0,39

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `track` (>=2), `dirt`, `train`; bộ ba `track along dirt`, `train ride along track`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2391578.jpg) | ![](img/2357910.jpg) | ![](img/2341342.jpg) | ![](img/2378096.jpg) |
| `2391578` | `2357910` (CLIP hạng 3) | `2341342` (CLIP hạng 2) | `2378096` (CLIP hạng 7) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,51 | cả câu | cả ảnh |  |  | cosine 0,3184 |  |  | 0,1618 | 32% |
| vật thể 0,10 | cả kênh | — ảnh không có vật thể: trung bình của các ứng viên |  |  |  |  | 0,720 | 0,0722 | 14% |
| bộ ba 0,39 | cả kênh | — ảnh không có bộ ba: trung bình của các ứng viên |  |  |  |  | 0,687 | 0,2689 | 53% |
| **Tổng** |  |  |  |  |  |  |  | **0,5029** | 100% |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2357910`):

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,51 | cả câu | cả ảnh |  |  | cosine 0,3097 |  |  | 0,1573 | 31% |
| vật thể 0,10 | `track` (>=2) | `track` (3-4) | 0,31 | 0,34 | 0,971 | +0,035 | 0,960 | 0,0302 | 6% |
| vật thể 0,10 | `train` | `train` | 0,34 | 0,47 | 0,924 |  | 0,878 | 0,0298 | 6% |
| vật thể 0,10 | `dirt` | `gravel` | 0,35 | 0,52 | 0,748 |  | 0,674 | 0,0234 | 5% |
| bộ ba 0,39 | `train ride along track` | `windshield wiper on train` | 0,76 | 0,09 | 0,728 |  | 0,702 | 0,2103 | 41% |
| bộ ba 0,39 | `track along dirt` | `gravel underneath track` | 0,24 | 0,10 | 0,695 |  | 0,671 | 0,0617 | 12% |
| **Tổng** |  |  |  |  |  |  |  | **0,5129** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`windshield wiper on train`, bộ ba, phần điểm 0,0238): hạng 1 → **2**, điểm 0,5129 → 0,5122. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,5130.

**Phân tích.** Ảnh đúng là một tấm ảnh màu nâu đỏ: đoàn tàu hàng chạy trên đường ray giữa hai bờ đất đào, đúng như câu tả. CLIP xếp nó hạng 1. Nhưng đồ thị VG của ảnh này rỗng hoàn toàn, 0 vật thể và 0 quan hệ, cả ở bản cấp 1 lẫn merge_v1. Dưới fusion raw, cả hai kênh của ảnh đều lấy điểm trung bình của các ứng viên, nên bảng giải thích của ảnh đúng chỉ có hai dòng "cả kênh". Các ảnh tàu có đồ thị khớp tốt (`train`, `track` nhóm 3–4 với g +0,035, `gravel` cho "dirt") vượt lên trên trung bình, đẩy ảnh đúng xuống hạng 5. Đây là lỗi phủ dữ liệu, không phải lỗi so khớp: mô hình không có gì để dùng hay để giải thích. Riêng ca này, xoá một bộ ba của ảnh hạng 1 cũng đủ đổi hạng, vì hai ảnh đầu chỉ cách nhau 0,0003 điểm; ảnh đúng ở hạng 5 cũng chỉ kém ảnh hạng 1 khoảng 0,010.

### Ca 8. "A bus that is reading "89 Portola Plaza" is traveling on."

`159627_2` · ảnh đúng `2355061` · hạng theo CLIP: 1 · hạng theo mô hình: **3** · a / b / c = 0,74 / 0,07 / 0,19

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `bus`, `road`; bộ ba `bus travel on road`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2355061.jpg) | ![](img/2353248.jpg) | ![](img/2370017.jpg) | ![](img/2355061.jpg) |
| `2355061` | `2353248` (CLIP hạng 3) | `2370017` (CLIP hạng 2) | `2355061` (CLIP hạng 1) — ảnh đúng |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,74 | cả câu | cả ảnh |  |  | cosine 0,3258 |  |  | 0,2414 | 56% |
| vật thể 0,07 | `bus` | `bus` | 0,55 | 0,69 | 0,959 |  | 0,900 | 0,0350 | 8% |
| vật thể 0,07 | `road` | `car` | 0,45 | 0,17 | 0,640 |  | 0,553 | 0,0177 | 4% |
| bộ ba 0,19 | `bus travel on road` | `bus has side mirror` | 1,00 | 0,07 | 0,734 |  | 0,725 | 0,1365 | 32% |
| **Tổng** |  |  |  |  |  |  |  | **0,4307** | 100% |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2353248`):

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,74 | cả câu | cả ảnh |  |  | cosine 0,3184 |  |  | 0,2359 | 54% |
| vật thể 0,07 | `bus` | `bus` | 0,55 | 0,94 | 0,954 |  | 0,933 | 0,0363 | 8% |
| vật thể 0,07 | `road` | `street` | 0,45 | 0,79 | 0,832 |  | 0,785 | 0,0252 | 6% |
| bộ ba 0,19 | `bus travel on road` | `bus has bumper` | 1,00 | 0,10 | 0,740 |  | 0,729 | 0,1372 | 32% |
| **Tổng** |  |  |  |  |  |  |  | **0,4345** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`bus`, vật thể, phần điểm 0,0350): hạng 1 → **5**, điểm 0,4345 → 0,4205. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,1, điểm trung bình 0,4345.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`bus`, vật thể, phần điểm 0,0267): hạng 3 → **4**, điểm 0,4307 → 0,4241. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 3,0, điểm trung bình 0,4308.

**Phân tích.** Ảnh đúng là đầu một chiếc xe buýt OCTA có biển đèn "89 PORTOLA PLAZA". Câu bị cắt ở cuối ("is traveling on."), FACTUAL tự thêm `road` và bộ ba `bus travel on road`. Chi tiết quyết định là dòng chữ trên biển. CLIP đọc được phần nào (a = 0,74, ảnh đúng hạng 1 theo CLIP), nhưng đồ thị không có cách nào biểu diễn nó. Trên kênh đồ thị, ảnh sai (xe buýt xanh megabus trên phố có hàng cọ) có node `street` cho "road" (0,832), trong khi ảnh đúng chỉ có `car` (0,640). Khoảng chênh đó đủ đẩy ảnh đúng xuống hạng 3. Xoá node `bus` ở ảnh sai làm nó rơi xuống hạng 5, ở ảnh đúng làm nó rơi xuống hạng 4. Cả hai ảnh đều được xếp chủ yếu nhờ việc là xe buýt.

## Thất bại: cả CLIP và mô hình đều sai

### Ca 9. "Two men standing in a living room playing video game with remotes."

`34471_0` · ảnh đúng `2417133` · hạng theo CLIP: 13 · hạng theo mô hình: **17** · a / b / c = 0,42 / 0,14 / 0,44

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `man` (2), `video game`, `remote` (>=2), `living room`; bộ ba `man play video game`, `man play with remote`, `man stand in living room`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2417133.jpg) | ![](img/2395602.jpg) | ![](img/2366027.jpg) | ![](img/2413879.jpg) |
| `2417133` | `2395602` (CLIP hạng 1) | `2366027` (CLIP hạng 8) | `2413879` (CLIP hạng 9) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,42 | cả câu | cả ảnh |  |  | cosine 0,2996 |  |  | 0,1251 | 23% |
| vật thể 0,14 | `man` (2) | `man` (2) | 0,24 | 0,39 | 0,954 | +0,071 | 0,912 | 0,0307 | 6% |
| vật thể 0,14 | `video game` | `man` | 0,28 | 0,16 | 0,656 |  | 0,612 | 0,0238 | 4% |
| vật thể 0,14 | `remote` (>=2) | `man` (2) | 0,31 | 0,17 | 0,566 | −0,005 | 0,520 | 0,0227 | 4% |
| vật thể 0,14 | `living room` | `room` | 0,17 | 0,21 | 0,739 |  | 0,680 | 0,0163 | 3% |
| bộ ba 0,44 | `man play video game` | `man wearing pant` | 0,43 | 0,29 | 0,748 |  | 0,731 | 0,1403 | 25% |
| bộ ba 0,44 | `man play with remote` | `man wearing pant` | 0,35 | 0,26 | 0,817 |  | 0,803 | 0,1249 | 22% |
| bộ ba 0,44 | `man stand in living room` | `man wearing pant` | 0,21 | 0,33 | 0,780 |  | 0,760 | 0,0721 | 13% |
| **Tổng** |  |  |  |  |  |  |  | **0,5558** | 100% |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2395602`):

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,42 | cả câu | cả ảnh |  |  | cosine 0,3616 |  |  | 0,1510 | 25% |
| vật thể 0,14 | `remote` (>=2) | `wii controller` (3-4) | 0,31 | 0,87 | 0,928 | +0,035 | 0,890 | 0,0389 | 6% |
| vật thể 0,14 | `video game` | `wii controller` | 0,28 | 0,70 | 0,887 |  | 0,817 | 0,0317 | 5% |
| vật thể 0,14 | `man` (2) | `man` (3-4) | 0,24 | 0,36 | 0,958 | +0,035 | 0,902 | 0,0303 | 5% |
| vật thể 0,14 | `living room` | `couch` | 0,17 | 0,64 | 0,737 |  | 0,667 | 0,0160 | 3% |
| bộ ba 0,44 | `man play video game` | `man wearing wrist band` | 0,43 | 0,15 | 0,770 |  | 0,748 | 0,1437 | 24% |
| bộ ba 0,44 | `man play with remote` | `man wearing wrist band` | 0,35 | 0,16 | 0,841 |  | 0,816 | 0,1269 | 21% |
| bộ ba 0,44 | `man stand in living room` | `man wearing short` | 0,21 | 0,14 | 0,785 |  | 0,763 | 0,0724 | 12% |
| **Tổng** |  |  |  |  |  |  |  | **0,6107** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`wii controller`, vật thể, phần điểm 0,0598): hạng 1 → **2**, điểm 0,6107 → 0,5917. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,6107.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`man wearing pant`, bộ ba, phần điểm 0,1004): hạng 17 → **19**, điểm 0,5558 → 0,5527. Xoá ngẫu nhiên 5 phần cùng loại: hạng trung bình 17,0, điểm trung bình 0,5562.

**Phân tích.** Ảnh đúng là hai người đàn ông lớn tuổi đứng trong phòng khách, tay cầm tay cầm Wii nhỏ. Câu đúng từng chữ, kể cả "two men". Số lượng hoạt động đúng: `man` (lớp 2) gặp `man` nhóm 2 của ảnh đúng, g +0,071, so với nhóm 3–4 (+0,035) ở ảnh hạng 1. Nhưng đồ thị VG của ảnh đúng không có node tay cầm hay trò chơi. Vì vậy `remote` và `video game` đều phải khớp với `man` (0,566 và 0,656), trong khi ảnh hạng 1 (mấy thanh niên ngồi sofa chơi Wii) có `wii controller` khớp cả hai (0,928 và 0,887). Thiếu nhãn đó nặng hơn phần cộng của g. Ảnh hạng 2 (hai người mặc đồ ngủ đứng cầm tay cầm) cũng hợp với câu, nên đây là một truy vấn có vài ảnh gần đúng. Kênh bộ ba nặng (c = 0,44) nhưng gần như bằng nhau giữa hai ảnh (0,140 so với 0,144).

### Ca 10. "A batter is preparing to hit the ball at a game."

`526446_0` · ảnh đúng `2408950` · hạng theo CLIP: 17 · hạng theo mô hình: **28** · a / b / c = 0,27 / 0,06 / 0,67

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `batter`, `ball`; bộ ba `batter hit ball`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 |
|---|---|---|---|
| ![](img/2408950.jpg) | ![](img/2417935.jpg) | ![](img/2417287.jpg) | ![](img/2329839.jpg) |
| `2408950` | `2417935` (CLIP hạng 3) | `2417287` (CLIP hạng 15) | `2329839` (CLIP hạng 25) |

Giải thích cho **ảnh đúng**:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,27 | cả câu | cả ảnh |  |  | cosine 0,3118 |  |  | 0,0834 | 13% |
| vật thể 0,06 | `batter` | `bat` | 0,49 | 0,37 | 0,864 |  | 0,746 | 0,0221 | 3% |
| vật thể 0,06 | `ball` | `bat` | 0,51 | 0,22 | 0,717 |  | 0,609 | 0,0184 | 3% |
| bộ ba 0,67 | `batter hit ball` | `man wearing helmet` | 1,00 | 0,08 | 0,805 |  | 0,775 | 0,5215 | 81% |
| **Tổng** |  |  |  |  |  |  |  | **0,6454** | 100% |

Giải thích cho **ảnh mô hình xếp hạng 1** (`2417935`):

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,27 | cả câu | cả ảnh |  |  | cosine 0,3300 |  |  | 0,0883 | 13% |
| vật thể 0,06 | `ball` | `ball` | 0,51 | 0,73 | 0,930 |  | 0,866 | 0,0262 | 4% |
| vật thể 0,06 | `batter` | `batter` | 0,49 | 0,44 | 0,941 |  | 0,853 | 0,0252 | 4% |
| bộ ba 0,67 | `batter hit ball` | `helmet has black/face mask` | 1,00 | 0,12 | 0,811 |  | 0,792 | 0,5328 | 79% |
| **Tổng** |  |  |  |  |  |  |  | **0,6725** | 100% |

Phép thử xoá:
- Xoá phần đóng góp lớn nhất của ảnh mô hình xếp hạng 1 (`helmet has black/face mask`, bộ ba, phần điểm 0,0676): hạng 1 → **1**, điểm 0,6725 → 0,6707. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 1,0, điểm trung bình 0,6727.
- Xoá phần đóng góp lớn nhất của ảnh đúng (`man wearing helmet`, bộ ba, phần điểm 0,0424): hạng 28 → **31**, điểm 0,6454 → 0,6436. Xoá ngẫu nhiên 10 phần cùng loại: hạng trung bình 28,0, điểm trung bình 0,6455.

**Phân tích.** Ảnh đúng là một cầu thủ số 28 của giải nhà nghề đang đứng chờ bóng, với người bắt bóng, trọng tài và khán đài phía sau. Không thấy quả bóng. Ảnh hạng 1 là một cầu thủ softball đang vung gậy trúng bóng, quả bóng hiện rõ. Đồ thị của ảnh đúng không có node `batter` hay `ball`, nên cả hai phải khớp với `bat` (0,864 và 0,717). Ảnh sai có đúng `batter` và `ball`. Câu đặt 67% trọng số vào kênh bộ ba, nhưng bộ ba `batter hit ball` chỉ khớp trải đều (a ≈ 0,1) vào các bộ ba về mũ bảo hiểm ở cả hai ảnh, với phần gần bằng nhau (0,52 so với 0,53). Trong top-50 có nhiều ảnh bóng chày; "preparing to hit" và "at a game" không thành phần nào của đồ thị. Ảnh đúng rơi từ hạng 17 xuống 28.

## Tổng hợp phép thử xoá

Xoá trên ảnh mô hình xếp hạng 1 của mỗi ca: phần của ảnh có đóng góp lớn nhất, so với trung bình 10 phần ngẫu nhiên cùng loại (ít hơn nếu ảnh không đủ 11 phần loại đó).

| Ca | Phần bị xoá | Loại | Hạng trước | Hạng sau (phần lớn nhất) | Hạng sau (ngẫu nhiên, trung bình) | Giảm điểm (lớn nhất) | Giảm điểm (ngẫu nhiên) |
|---|---|---|---|---|---|---|---|
| 1 | `table` | vật thể | 1 | 2 | 1,0 | 0,0070 | 0,0003 |
| 2 | `cow standing ear` | bộ ba | 1 | 1 | 1,0 | 0,0034 | -0,0002 |
| 3 | `pen` | vật thể | 1 | 2 | 1,1 | 0,0184 | 0,0017 |
| 4 | `donut` | vật thể | 1 | 3 | 1,1 | 0,0113 | 0,0001 |
| 5 | `dog` | vật thể | 1 | 3 | 1,0 | 0,0346 | 0,0001 |
| 6 | `banana in plate` | bộ ba | 1 | 1 | 1,0 | 0,0016 | -0,0002 |
| 7 | `windshield wiper on train` | bộ ba | 1 | 2 | 1,0 | 0,0007 | -0,0001 |
| 8 | `bus` | vật thể | 1 | 5 | 1,1 | 0,0141 | 0,0000 |
| 9 | `wii controller` | vật thể | 1 | 2 | 1,0 | 0,0190 | 0,0001 |
| 10 | `helmet has black/face mask` | bộ ba | 1 | 1 | 1,0 | 0,0018 | -0,0002 |

- Ở 10/10 ca, xoá phần có đóng góp lớn nhất làm điểm giảm nhiều hơn xoá ngẫu nhiên.
- Ở 7/10 ca, riêng việc xoá một phần này đã làm ảnh rơi khỏi hạng 1.
- Chia theo loại phần bị xoá: vật thể 6/6 ca rơi hạng, bộ ba 1/4 ca. Ca bộ ba duy nhất rơi hạng là ca 7, nơi hai ảnh đầu chỉ cách nhau 0,0003.
- Cấp 1 (`main_r3`) dùng tiêu chí khác: cặp khớp nhất của dòng có đóng góp lớn nhất, nên hai bảng không so trực tiếp từng số được.

## So với mô hình cấp 1 (`main_r3`)

Mỗi seed của mô hình mới so với cùng seed của `main_r3`, trên 10 633 câu val. "Sửa đúng": `main_r3` xếp ảnh đúng dưới hạng 1, mô hình mới xếp hạng 1. "Làm hỏng": ngược lại. "Có số nêu rõ": câu có ít nhất một node thuộc lớp 1 / 2 / 3–4 / ≥5 (1235 câu, 11,6%).

| Cặp seed | Sửa đúng | Làm hỏng | Chênh | Cả hai đúng | Cả hai sai | Sửa đúng (có số nêu rõ) | Làm hỏng (có số nêu rõ) |
|---|---|---|---|---|---|---|---|
| seed0 | 747 | 483 | +264 | 4479 | 4924 | 109 | 54 |
| seed1 | 750 | 507 | +243 | 4444 | 4932 | 101 | 60 |
| seed2 | 717 | 448 | +269 | 4465 | 5003 | 89 | 60 |

Trung bình 3 cặp: sửa đúng 738 ± 18, làm hỏng 479 ± 30 câu. Mô hình mới không chỉ cộng thêm câu đúng: nó đổi khá nhiều câu theo cả hai chiều, và phần lời ròng khoảng 260 câu, tức khoảng 2,4 điểm R@1.

### Bỏ g hoặc gỡ gộp lúc eval (không train lại)

Đếm bằng `explain_counterfactuals.py`: 52230 node ảnh val khi gộp, 56181 node khi gỡ gộp.

| Run | Thay đổi lúc eval | R@1 | MRR | R@1 có số nêu rõ | R@1 không có số nêu rõ |
|---|---|---|---|---|---|
| `countonly_factual_r3_seed0` | như đã train | 49,15 | 61,10 | 55,14 | 48,36 |
| `countonly_factual_r3_seed0` | g = 0 | 48,76 | 60,79 | 53,44 | 48,15 |
| `countonly_factual_r3_seed0` | gỡ gộp | 49,07 | 61,00 | 55,06 | 48,29 |
| `countonly_factual_r3_seed1` | như đã train | 48,85 | 60,80 | 54,09 | 48,16 |
| `countonly_factual_r3_seed1` | g = 0 | 48,50 | 60,53 | 51,82 | 48,06 |
| `countonly_factual_r3_seed1` | gỡ gộp | 48,86 | 60,81 | 53,93 | 48,19 |
| `countonly_factual_r3_seed2` | như đã train | 48,74 | 60,68 | 52,96 | 48,18 |
| `countonly_factual_r3_seed2` | g = 0 | 48,42 | 60,41 | 52,06 | 47,94 |
| `countonly_factual_r3_seed2` | gỡ gộp | 48,75 | 60,74 | 52,87 | 48,21 |

- **Bỏ g:** R@1 giảm 0,35 ± 0,03 điểm. Trên câu có số nêu rõ, R@1 giảm 1,62 ± 0,69 điểm. Mô hình đã train thật sự dùng g lúc xếp hạng, và dùng nhiều nhất ở câu có số.
- **Gỡ gộp:** R@1 đổi -0,02 ± 0,05 điểm, tức gần như không đổi. Mô hình đã train trên đồ thị gộp không nhạy với việc gộp lúc eval. Phần tăng của thí nghiệm 3 vì vậy khó quy cho việc gộp. Thí nghiệm 3 gộp 4 thay đổi mà chưa có ablation, như đã ghi ở phần hạn chế.

### Ví dụ 1. Số lượng: một câu có số nêu rõ được sửa đúng

"The two people are walking in the snow with their ski equipment." · `472772_2` · ảnh đúng `2384371` · hạng theo CLIP: 1 · `main_r3_seed0`: 5 · `countonly_factual_r3_seed0`: **1**

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `people` (2), `snow`, `ski equipment`; bộ ba `people walk in snow`, `people with ski equipment`.

Đồ thị câu của cấp 1 (sng_parser): vật thể `people`, `snow`, `ski equipment`; bộ ba `people in snow`, `people with ski equipment`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 | main_r3 hạng 1 |
|---|---|---|---|---|
| ![](img/2384371.jpg) | ![](img/2384371.jpg) | ![](img/2320258.jpg) | ![](img/2352632.jpg) | ![](img/2352632.jpg) |
| `2384371` | `2384371` (CLIP hạng 1) — ảnh đúng | `2320258` (CLIP hạng 3) | `2352632` (CLIP hạng 2) | `2352632` |

Giải thích của **countonly** cho ảnh đúng:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,51 | cả câu | cả ảnh |  |  | cosine 0,3152 |  |  | 0,1607 | 33% |
| vật thể 0,12 | `people` (2) | `woman` (2) | 0,36 | 0,44 | 0,887 | +0,071 | 0,848 | 0,0363 | 7% |
| vật thể 0,12 | `ski equipment` | `snowboard` | 0,41 | 0,35 | 0,620 |  | 0,517 | 0,0254 | 5% |
| vật thể 0,12 | `snow` | `ski` | 0,23 | 0,45 | 0,833 |  | 0,750 | 0,0208 | 4% |
| bộ ba 0,37 | `people walk in snow` | `person carrying ski` | 0,72 | 0,39 | 0,700 |  | 0,637 | 0,1703 | 35% |
| bộ ba 0,37 | `people with ski equipment` | `person carrying ski` | 0,28 | 0,13 | 0,790 |  | 0,741 | 0,0766 | 16% |
| **Tổng** |  |  |  |  |  |  |  | **0,4900** | 100% |

Giải thích của **main_r3** cho ảnh đúng (bảng cấp 1: cặp khớp nhất, trước z-score):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `people` | `people` | 0,31 | 0,44 | 1,000 | 0,0463 |
| vật thể | `snow` | `parka` | 0,25 | 0,18 | 0,737 | 0,0111 |
| vật thể | `ski equipment` | `skiing` | 0,45 | 0,16 | 0,665 | 0,0167 |
| bộ ba | `people in snow` | `people on part` | 0,45 | 0,03 | 0,966 | 0,0025 |
| bộ ba | `people with ski equipment` | `pattern on shirt` | 0,55 | 0,02 | 0,992 | 0,0027 |

Giải thích của **main_r3** cho ảnh nó xếp hạng 1 (`2352632`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `people` | `people` | 0,31 | 0,63 | 1,000 | 0,0665 |
| vật thể | `snow` | `snow` | 0,25 | 0,78 | 1,000 | 0,0674 |
| vật thể | `ski equipment` | `ski` | 0,45 | 0,25 | 0,652 | 0,0256 |
| bộ ba | `people in snow` | `people in snow` | 0,45 | 0,11 | 1,000 | 0,0099 |
| bộ ba | `people with ski equipment` | `people has ski` | 0,55 | 0,07 | 0,990 | 0,0080 |

Bỏ g hoặc gỡ gộp lúc eval (countonly seed 0), hạng của ảnh đúng và cách biệt với ảnh tốt nhất còn lại:

| Thay đổi | Hạng | Điểm ảnh đúng | Cách biệt |
|---|---|---|---|
| như đã train | 1 | 0,4900 | 0,0136 |
| g = 0 | 1 | 0,4873 | 0,0081 |
| gỡ gộp | 1 | 0,4916 | 0,0149 |

**Phân tích.** Ảnh đúng là hai người đi trên con đường tuyết giữa rừng thông, một người vác snowboard, một người cầm gậy và ván trượt. CLIP xếp nó hạng 1, main_r3 làm nó tụt xuống hạng 5, countonly đưa nó về hạng 1. Ảnh main_r3 chọn (2352632) cũng có hai người cầm ván trượt băng đồng trên tuyết. Hai ảnh đều có "two people", nên số lượng không thể là phần phân biệt duy nhất. Trong countonly, `people` (lớp 2) khớp `woman` nhóm 2, g +0,071. Bỏ g lúc eval thì ảnh đúng vẫn hạng 1, nhưng cách biệt với ảnh kế tiếp giảm từ 0,0136 xuống 0,0081. Như vậy g đóng góp khoảng 40% cách biệt, không quyết định một mình. Phần còn lại đến từ bộ ba FACTUAL `people walk in snow` khớp `person carrying ski` với attention 0,39, cao bất thường cho kênh bộ ba. Ở main_r3, `snow` của câu khớp `parka` (đồ thị ảnh đúng không có node tuyết), còn ảnh kia có đúng `snow` và `people in snow`.

### Ví dụ 2. Đồ thị gộp: một câu được sửa đúng, phần khớp lớn nhất là node đã gộp

"A snow board sitting upright with a view of a mountain." · `237041_4` · ảnh đúng `2320074` · hạng theo CLIP: 1 · `main_r3_seed0`: 4 · `countonly_factual_r3_seed0`: **1**

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `snow board`, `mountain`; bộ ba `snow board sit upright with mountain`.

Đồ thị câu của cấp 1 (sng_parser): vật thể `snow board`, `view`, `mountain`; bộ ba `snow board with view`, `view of mountain`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 | main_r3 hạng 1 |
|---|---|---|---|---|
| ![](img/2320074.jpg) | ![](img/2320074.jpg) | ![](img/2412723.jpg) | ![](img/2403334.jpg) | ![](img/2412723.jpg) |
| `2320074` | `2320074` (CLIP hạng 1) — ảnh đúng | `2412723` (CLIP hạng 4) | `2403334` (CLIP hạng 8) | `2412723` |

Giải thích của **countonly** cho ảnh đúng:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,52 | cả câu | cả ảnh |  |  | cosine 0,3223 |  |  | 0,1671 | 40% |
| vật thể 0,08 | `snow board` | `snowboard` | 0,53 | 0,85 | 0,904 |  | 0,864 | 0,0370 | 9% |
| vật thể 0,08 | `mountain` | `mountain` | 0,47 | 0,64 | 0,923 |  | 0,883 | 0,0342 | 8% |
| bộ ba 0,40 | `snow board sit upright with mountain` | `logo on snowboard` | 1,00 | 0,27 | 0,455 |  | 0,441 | 0,1764 | 43% |
| **Tổng** |  |  |  |  |  |  |  | **0,4148** | 100% |

Giải thích của **main_r3** cho ảnh đúng (bảng cấp 1: cặp khớp nhất, trước z-score):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `snow board` | `snowboard` | 0,45 | 0,49 | 0,946 | 0,0615 |
| vật thể | `view` | `hillside` | 0,11 | 0,30 | 0,733 | 0,0071 |
| vật thể | `mountain` | `mountain` | 0,44 | 0,76 | 1,000 | 0,0974 |
| bộ ba | `snow board with view` | `snowboard in snow` | 0,57 | 0,17 | 0,983 | 0,0196 |
| bộ ba | `view of mountain` | `snowboard in snow` | 0,43 | 0,17 | 0,979 | 0,0148 |

Giải thích của **main_r3** cho ảnh nó xếp hạng 1 (`2412723`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `snow board` | `snowboard` | 0,45 | 0,92 | 0,947 | 0,1156 |
| vật thể | `view` | `slope` | 0,11 | 0,27 | 0,805 | 0,0069 |
| vật thể | `mountain` | `mountain` | 0,44 | 0,37 | 0,996 | 0,0470 |
| bộ ba | `snow board with view` | `snowboard has writing` | 0,57 | 0,04 | 0,992 | 0,0050 |
| bộ ba | `view of mountain` | `mountain on horizon` | 0,43 | 0,04 | 0,996 | 0,0038 |

Bỏ g hoặc gỡ gộp lúc eval (countonly seed 0), hạng của ảnh đúng và cách biệt với ảnh tốt nhất còn lại:

| Thay đổi | Hạng | Điểm ảnh đúng | Cách biệt |
|---|---|---|---|
| như đã train | 1 | 0,4148 | 0,0101 |
| g = 0 | 1 | 0,4148 | 0,0101 |
| gỡ gộp | 1 | 0,4100 | 0,0054 |

**Phân tích.** Ảnh đúng là một tấm snowboard Burton cắm đứng trên tuyết, phía sau là dãy núi. Ảnh main_r3 chọn (2412723) là một chiếc ván trượt cắm đứng cạnh một người nhìn ra núi; VG gắn nhãn `snowboard` cho chiếc ván đó, nên main_r3 thấy hai ảnh gần như ngang nhau và xếp ảnh đúng hạng 4. Trong countonly, `snow board` khớp node `snowboard` đã gộp từ 2 box trùng (IoU ≥ 0,7). Gỡ gộp toàn bộ đồ thị val lúc eval thì ảnh đúng vẫn hạng 1, nhưng cách biệt giảm từ 0,0101 còn 0,0054. Đồ thị gộp giúp khoảng một nửa cách biệt ở ca này, không quyết định một mình. FACTUAL cũng bỏ node `view` mà sng_parser tách ra, nên đồ thị câu gọn hơn, chỉ còn snowboard và núi.

### Ví dụ 3. Số lượng làm khớp sai: "lone woman" → `hair`

"A lone woman huddles under her umbrella at a picnic table in the rain." · `151938_1` · ảnh đúng `2385674` · hạng theo CLIP: 10 · `main_r3_seed0`: 6 · `countonly_factual_r3_seed0`: **1**

Đồ thị câu (FACTUAL, lớp số lượng trong ngoặc): vật thể `woman` (1), `picnic table`, `umbrella`, `rain`; bộ ba `woman at picnic table`, `woman huddle under umbrella`, `woman in rain`.

Đồ thị câu của cấp 1 (sng_parser): vật thể `woman`, `umbrella`, `picnic table`, `rain`; bộ ba `woman under umbrella`, `woman at picnic table`, `picnic table in rain`.

| Ảnh đúng | Mô hình hạng 1 | Mô hình hạng 2 | Mô hình hạng 3 | main_r3 hạng 1 |
|---|---|---|---|---|
| ![](img/2385674.jpg) | ![](img/2385674.jpg) | ![](img/2360307.jpg) | ![](img/2413726.jpg) | ![](img/2360307.jpg) |
| `2385674` | `2385674` (CLIP hạng 10) — ảnh đúng | `2360307` (CLIP hạng 1) | `2413726` (CLIP hạng 5) | `2360307` |

Giải thích của **countonly** cho ảnh đúng:

| Kênh | Phần của câu | Phần của ảnh (a lớn nhất) | w_i | a_ij | sim_ij (gồm g) | g | Σ a·sim | Phần điểm | % điểm cuối |
|---|---|---|---|---|---|---|---|---|---|
| CLIP 0,73 | cả câu | cả ảnh |  |  | cosine 0,2755 |  |  | 0,2012 | 49% |
| vật thể 0,15 | `umbrella` | `umbrella` | 0,27 | 0,91 | 0,960 |  | 0,931 | 0,0379 | 9% |
| vật thể 0,15 | `woman` (1) | `hair` (1) | 0,26 | 0,36 | 0,917 | +0,087 | 0,845 | 0,0323 | 8% |
| vật thể 0,15 | `picnic table` | `table` | 0,25 | 0,45 | 0,807 |  | 0,767 | 0,0289 | 7% |
| vật thể 0,15 | `rain` | `rain` | 0,21 | 0,79 | 0,863 |  | 0,807 | 0,0256 | 6% |
| bộ ba 0,12 | `woman in rain` | `woman in rain` | 0,35 | 0,17 | 0,781 |  | 0,737 | 0,0314 | 8% |
| bộ ba 0,12 | `woman huddle under umbrella` | `woman facing man` | 0,36 | 0,08 | 0,684 |  | 0,663 | 0,0287 | 7% |
| bộ ba 0,12 | `woman at picnic table` | `woman wearing a striped shirt` | 0,29 | 0,09 | 0,728 |  | 0,703 | 0,0250 | 6% |
| **Tổng** |  |  |  |  |  |  |  | **0,4109** | 100% |

Giải thích của **main_r3** cho ảnh đúng (bảng cấp 1: cặp khớp nhất, trước z-score):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `woman` | `woman` | 0,26 | 0,91 | 1,000 | 0,0459 |
| vật thể | `umbrella` | `umbrella` | 0,26 | 0,75 | 1,000 | 0,0372 |
| vật thể | `picnic table` | `table` | 0,25 | 0,41 | 0,920 | 0,0180 |
| vật thể | `rain` | `rain` | 0,23 | 0,75 | 0,992 | 0,0322 |
| bộ ba | `woman under umbrella` | `woman under umbrella` | 0,37 | 0,07 | 1,000 | 0,0015 |
| bộ ba | `woman at picnic table` | `woman sitting at table` | 0,35 | 0,05 | 0,997 | 0,0009 |
| bộ ba | `picnic table in rain` | `table in rain` | 0,28 | 0,04 | 0,997 | 0,0007 |

Giải thích của **main_r3** cho ảnh nó xếp hạng 1 (`2360307`):

| Kênh | Phần của câu | Phần của ảnh được chọn | w_i | a_ij | sim_ij | Đóng góp |
|---|---|---|---|---|---|---|
| vật thể | `woman` | `lady` | 0,26 | 0,61 | 0,988 | 0,0305 |
| vật thể | `umbrella` | `umbrella` | 0,26 | 0,43 | 1,000 | 0,0213 |
| vật thể | `picnic table` | `bench` | 0,25 | 0,21 | 0,602 | 0,0059 |
| vật thể | `rain` | `umbrella` | 0,23 | 0,29 | 0,800 | 0,0101 |
| bộ ba | `woman under umbrella` | `lady sitting on outside bench` | 0,37 | 0,10 | 0,950 | 0,0021 |
| bộ ba | `woman at picnic table` | `lady sitting on outside bench` | 0,35 | 0,09 | 0,976 | 0,0017 |
| bộ ba | `picnic table in rain` | `trim on umbrella` | 0,28 | 0,09 | 0,985 | 0,0014 |

Bỏ g hoặc gỡ gộp lúc eval (countonly seed 0), hạng của ảnh đúng và cách biệt với ảnh tốt nhất còn lại:

| Thay đổi | Hạng | Điểm ảnh đúng | Cách biệt |
|---|---|---|---|
| như đã train | 1 | 0,4109 | 0,0052 |
| g = 0 | 1 | 0,4113 | 0,0083 |
| gỡ gộp | 1 | 0,4109 | 0,0035 |

**Phân tích.** Ảnh đúng được chụp từ trong nhà tối ra ngoài trời mưa. Có một người phụ nữ tóc vàng áo sọc ngồi ở chiếc bàn dài ướt mưa, dưới một chiếc ô, và cạnh đó còn một người khác; VG có bộ ba `woman facing man`. Mọi người được tính chung một loại, nên node `woman` thuộc nhóm 3–4. "A lone woman" (lớp 1) vì vậy bị g phạt −0,088 ở node đó. Node `hair` (nhóm 1) lại được thưởng +0,087, nên dòng giải thích ghi `woman` ↔ `hair`. Người đọc sẽ thấy cặp khớp này sai, dù sim gồm g vẫn cao (0,917). Countonly vẫn xếp ảnh đúng hạng 1 (CLIP hạng 10, main_r3 hạng 6), nhờ `umbrella`, `rain`, `picnic table`→`table` và bộ ba `woman in rain` khớp đúng `woman in rain`. Bỏ g lúc eval thì ảnh đúng vẫn hạng 1, và cách biệt tăng từ 0,0052 lên 0,0083. Ở ca này g làm hại cả cách biệt lẫn độ tin của giải thích, nhưng không làm hỏng thứ hạng.

## Ghi chú về cách làm

- Ví dụ 1 và 2 bốc theo luật, tiếp tục cùng bộ sinh số sau 10 ca:
  - ví dụ 1: ngẫu nhiên trong các câu được sửa đúng có số nêu rõ;
  - ví dụ 2: câu đầu tiên, theo thứ tự ngẫu nhiên, trong các câu được sửa đúng mà dòng vật thể có phần điểm lớn nhất ở ảnh đúng khớp một node đã gộp (boxes > 1).
- Ví dụ 3 (151938_1) do nhóm chỉ định trước, vì `scripts/improve/count_match_changes.py` đã tìm thấy nó như một ca g kéo sang node sai.
- Lần chạy đầu, phép thử xoá chọn phần theo dòng có channel_term lớn nhất. Mình đổi sang term_j theo phần của ảnh trước khi viết phân tích, vì attention bộ ba trải đều nên cặp khớp nhất của dòng chỉ giữ một phần nhỏ của dòng. Cách chọn ca không đổi: chạy lại cho đúng 10 ca và 3 ví dụ như lần đầu.
