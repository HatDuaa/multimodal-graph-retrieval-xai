# Demo với mô hình chính `countonly_factual_r3` (25/10/2026)

Chế độ **CLIP + Đồ thị** của `app/demo_app.py` giờ mặc định chạy mô hình chính của phase cải tiến. Mô hình cấp 1 vẫn chạy
được bằng `--run main_r3_seed0`.

## Chọn checkpoint

Checkpoint được chọn theo R@1 val, không nhìn test. Mỗi seed dùng `checkpoint_best.pt`, tức epoch có R@1 val cao nhất
của seed đó.

| Run | R@1 val | MRR val |
|---|---|---|
| **`countonly_factual_r3_seed0`** (chọn) | **49,15** | 61,10 |
| `countonly_factual_r3_seed1` | 48,85 | 60,80 |
| `countonly_factual_r3_seed2` | 48,74 | 60,68 |

Checkpoint: `experiments/improve/count-only/countonly_factual_r3_seed0/checkpoint_best.pt` (epoch 8, SHA-256
`b065eb16f26c0bdd976970b6849f5c1fb6b62a062bc55e1ebcaf6e2fb9ae3631`).

## Đường demo

`src/service/graph_mode.py` đọc `config.json` của run để biết run được train trên phiên bản dữ liệu nào. Từ đó nó lấy dữ
liệu của split đang xem (`DEMO_DATA`):

- **Câu gõ vào** được tách thành đồ thị theo đúng chuỗi lúc đóng gói caption
  (`src/graph/caption_parsers.FactualQueryParser`):
  - FACTUAL-T5 large (`lizhuang144/flan-t5-large-VG-factual-sg`), beam 5, `filter_factual_chars`;
  - chuẩn hoá tên và vị từ bằng `convert()`;
  - lớp số lượng cho từng node theo luật count_v1 (`src/graph/counts.py`), từ cây spaCy của câu và head thô của FACTUAL.
- **Đồ thị ảnh:**
  - val: `data/graphs/merge_v1` cùng `data/graphs/count_v1/count_buckets.json`;
  - test: `data/graphs/count_v1_test`;
  - cấu trúc lấy theo `config.json`: cạnh hình học, 5 cạnh kNN `[none]`, cầu nối, node tổng;
  - phần đồ thị của mỗi ảnh được dựng một lần rồi giữ lại.
- **Điểm:** fusion raw và bảng g, đúng như lúc train.

### Bảng giải thích dưới fusion raw

Điểm cuối = a × cosine CLIP + b × S(vật thể) + c × S(bộ ba), với S = Σ_i w_i · Σ_j a_ij · sim_ij. Vì không có z-score
trong top-50, mọi phần đều tách đúng.

- **Mỗi dòng của một kênh** là trọng số kênh × w_i × Độ khớp, trong đó Độ khớp = Σ_j a_ij · sim_ij. Các dòng của một kênh
  cộng lại đúng bằng phần của kênh đó, và mọi % cộng lại bằng 100%.
- **Cột Phần khớp** ghi phần của ảnh có a lớn nhất. Rê chuột lên ô % để xem a và sim của phần đó.
  - Vì cột Độ khớp giờ là tổng trên mọi phần của ảnh, tiêu đề cột đổi thành "Độ khớp (Σ a × sim)".
  - Dưới fusion z-score (`main_r3`) giữ nguyên cột "a × sim" cũ của phần khớp nhất. Phần đồ thị khi đó vẫn chia xấp xỉ
    theo tỉ lệ, như trước.
- **Dòng vật thể có số lượng nêu trong câu** hiện lớp số lượng của câu, nhóm số lượng của node ảnh và g đã cộng vào sim.
  Số lượng "không rõ" thì g = 0 và không hiện gì.
- **Kênh mà ảnh không có phần nào để khớp** (thường là ảnh không có bộ ba): `fuse_raw` gán cho kênh đó điểm trung bình của
  các ứng viên có phần này. Bảng hiện riêng phần đó thành dòng "cả kênh", kèm % của nó.
  - Ví dụ, với câu "Two dogs playing with a frisbee on the grass", ảnh val 2399623 không có bộ ba nào. Dòng "cả kênh" của
    kênh bộ ba vẫn chiếm 67% điểm cuối.
  - Đây đúng là cách mô hình đang chấm điểm, không phải lỗi hiển thị. Nên nhắc khi demo.
- **Không thành dòng:** cạnh kNN `[none]` và node tổng. Cả hai chỉ dùng lúc truyền tin trong GAT, không được so khớp.
- **Lý do "không khớp"** vẫn lấy từ `unmatched_parts`, trên đồ thị FACTUAL:
  - đại từ, cùng các quan hệ dính vào nó;
  - nhãn rỗng;
  - ảnh không có vật thể hoặc không có bộ ba.

  FACTUAL thường tự thay đại từ: với "he is riding a horse", node ra là `man`, nên câu này không có dòng đại từ.

## Kiểm tra khớp điểm với đường đánh giá đã đóng gói (val)

`scripts/checks/demo_parity.py` chọn ngẫu nhiên caption val (seed 0) và cho mỗi caption đi qua hai đường:

- **đường demo:** gõ caption như người dùng: CLIP top-50 của câu, đồ thị câu tách ngay lúc chạy, đồ thị ảnh dựng trong
  service;
- **đường đóng gói:** `GraphStore` của run và `store.batch`, cùng checkpoint.

Không đọc test.

| Run | Caption | Cùng 50 ứng viên | Cùng đồ thị câu | Lệch điểm cuối tối đa | Các phần − điểm cuối | Top-1 trùng | Giây / truy vấn |
|---|---|---|---|---|---|---|---|
| `countonly_factual_r3_seed0` (fusion raw) | 200 | 200 | 200 | 4,0e-6 | 1,1e-7 | 200 | 0,40 |
| `main_r3_seed0` (fusion z-score) | 100 | 100 | 100 | 2,9e-4 | 2,5e-7 | 100 | 0,05 |

- **Độ lệch còn lại** đến từ việc mã hoá câu bằng CLIP lúc chạy, thay vì dùng vector caption đã lưu. Với `main_r3`, phép
  z-score chia cho độ lệch chuẩn của cosine trong top-50 (khoảng 0,02), nên độ lệch lớn hơn.
- **Cột "Các phần − điểm cuối"** là sai lệch lớn nhất giữa tổng các phần trong bảng và điểm cuối:
  - fusion raw: CLIP + mọi dòng + các kênh lấy trung bình;
  - z-score: CLIP + đồ thị.
- **Tổng các % trừ 1** (fusion raw): tối đa 2,1e-7.
- **Thời gian:** trên RTX 4090, gồm FACTUAL. Lần gõ đầu chậm hơn vì phải dựng phần đồ thị của ảnh.

Kết quả từng caption: `parity_<run>_val.json`, log: `parity_<run>_val.log`.

## Lỗi cắt token trong FactualSceneGraph 0.7.3

Lần chạy kiểm tra đầu, 7/30 caption cho đồ thị khác với bản đóng gói. Nguyên nhân nằm ở `SceneGraphParser.parse`:

- Hàm này lấy độ dài tối đa của batch bằng prompt dài nhất, đếm **không kèm** token kết thúc `</s>`.
- Sau đó nó tokenize **có** `</s>` với `truncation=True`, nên prompt dài nhất của mỗi batch mất token cuối.
- Một câu gõ vào được tách một mình, nên lúc nào cũng là câu dài nhất batch và lúc nào cũng mất token cuối. Ví dụ: "wok"
  thành "wo", "a snowy ledge" thành "snow".

Demo sửa bằng `FactualParser.generate_one`: tokenize một câu với giới hạn đã tính cả `</s>` (tối đa 64 token, như
`max_input_len`), còn phần sinh và hậu xử lý giữ nguyên của `parse()`. Kết quả giống các caption đóng gói không bị cắt.

Ảnh hưởng tới dữ liệu đã đóng gói (`scripts/checks/factual_truncation.py`, `factual_truncation.json`). Cách tính: dựng lại
các batch 64 theo đúng thứ tự lúc parse, rồi dùng tokenizer để biết caption nào bị cắt; không chạy mô hình nào trên test.

| Split | Caption | Bị cắt | Chỉ mất dấu câu | Mất một phần chữ |
|---|---|---|---|---|
| train | 234 854 | 4 283 (1,82%) | 3 680 | 603 (0,26%) |
| val | 10 633 | 186 (1,75%) | 161 | 25 (0,24%) |
| test | 10 696 | 205 (1,92%) | 172 | 33 (0,31%) |

- **Kiểm chứng trên val:** tách lại một mình 25 caption val bị mất chữ thì ra đúng output thô đã đóng gói, cả 25/25. Như
  vậy giải thích trên khớp với dữ liệu.
  - Ví dụ 314515_3: "... WITH UMBRELLAS" thành `( people , stand around , UMBRELL )`.
  - Ví dụ 324291_2: "... in a green field" thành `( clothes , is , green )`.
- **Mức ảnh hưởng:** khoảng 0,3% caption mỗi split, như nhau ở train, val và test, nên không lệch giữa train và đánh giá.
  Mình không parse lại và không train lại; mọi kết quả đã báo giữ nguyên.
- **Ở demo:** câu gõ vào không bị cắt. Một caption val nằm trong số 0,24% bị mất chữ sẽ ra điểm khác bản đóng gói. Trong
  200 caption kiểm tra có 3 caption bị cắt, nhưng cả 3 chỉ mất dấu chấm và đồ thị vẫn trùng.

## Test

- **`tests/test_demo_countonly.py`:**
  - các phần dưới fusion raw cộng đúng bằng điểm cuối (cả kênh lấy trung bình);
  - trên một mô hình nhỏ có bảng g, cạnh kNN, node tổng và cạnh hình học, các channel term của `matched_pairs` cộng với
    CLIP và các kênh lấy trung bình ra đúng điểm fused;
  - node tổng không được khớp, chỉ quan hệ có nhãn mới thành dòng bộ ba;
  - ghi chú số lượng chỉ hiện khi câu nêu số lượng;
  - `generate_one` không cắt token của một câu.
- Toàn bộ test: 133 passed.
- **Chạy thử app trên server:** `--split val`, gọi `/run` qua API của Gradio. Ra 5 kết quả và bảng giải thích như mô tả.

## Cột "So với TB top-50" (thêm 25/10/2026)

**Vì sao cần.** Dưới fusion raw, cột "% điểm cuối" cho kênh bộ ba chiếm phần lớn điểm, dù phần này gần như bằng nhau giữa
các ảnh ứng viên (xem `experiments/improve/explanations/`). Người xem dễ hiểu nhầm rằng bộ ba quyết định thứ hạng.

**Cột mới.** Với mỗi dòng (CLIP, từng dòng vật thể, từng dòng bộ ba, dòng "cả kênh"):
Δ = giá trị dòng ở ảnh này − trung bình của đúng dòng đó trên 50 ứng viên của câu
(`src/explain/score_breakdown.top50_deltas`).

- Trọng số kênh và w_i không đổi giữa các ứng viên, chỉ Σ a·sim đổi.
- Ứng viên không có phần nào cho một kênh lấy giá trị đã fill, đúng như `fuse_raw`. Mỗi dòng nhận trung bình của các ứng
  viên có phần, nên tổng các dòng đúng bằng điểm kênh đã fill. Một kênh đã fill vì vậy luôn có Δ = 0.
- Tổng các Δ của một ảnh đúng bằng fused − mean(fused). Test: `tests/test_demo_countonly.py`, có một ca thiếu kênh bộ ba.
- Dòng tóm tắt dạng "Hơn trung bình top-50: +X = CLIP +… + Vật thể +… + Bộ ba +…".
- Thanh xanh nghĩa là dòng đẩy ảnh lên, đỏ là kéo xuống. Mô hình cấp 1 (z-score) giữ nguyên bảng cũ.

**Ví dụ.** Câu "Two dogs playing with a frisbee on the grass", chạy qua API của app trên server (CPU, split val):
- `delta_example_api.html`: bảng của ảnh hạng 1 mà `/run` trả về.
- `delta_example.html`: thêm ảnh 2399623, ảnh không có bộ ba, kèm CSS của app.

| Ảnh | Điểm cuối | Bộ ba (% điểm cuối) | Hơn / kém trung bình top-50 |
|---|---|---|---|
| 2319590 (hạng 1) | 0,584 | 57% | +0,0669 = CLIP +0,0256 + Vật thể +0,0429 + Bộ ba −0,0016 |
| 2399623 (hạng 34, không có bộ ba) | 0,503 | 67% (dòng "cả kênh") | −0,0140 = CLIP +0,0098 + Vật thể −0,0237 + Bộ ba +0,0000 |

Ở cả hai ảnh, kênh bộ ba chiếm hơn nửa điểm cuối nhưng gần như không đổi thứ hạng. Chính kênh vật thể (dog, frisbee,
grass) và CLIP đẩy ảnh hạng 1 lên trên trung bình.

## Node nào của câu khớp node nào của ảnh (thêm 16/11/2026)

**Vì sao cần.** Bảng cho biết mỗi phần của câu đóng góp bao nhiêu, nhưng không rõ nó khớp với node nào của ảnh:

- cột Phần khớp chỉ ghi một nhãn (phần có sim lớn nhất) và không chỉ vị trí trong ảnh;
- với bộ ba, attention trải đều, nên một nhãn dễ gây hiểu nhầm.

**Top-3 phần khớp của mỗi dòng.** `matched_pairs` vốn đã tính a_ij và sim_ij cho mọi phần j của ảnh. Giờ nó trả thêm
`top_matches`: 3 phần có a_ij · sim_ij lớn nhất, tức 3 số hạng lớn nhất của Độ khớp = Σ_j a_ij · sim_ij. Mỗi phần kèm
a, sim và tỉ lệ a_ij · sim_ij / Độ khớp. Không tìm kiếm gì thêm ngoài các đại lượng của lần xếp hạng.

- Khi sim dương, a tăng theo sim, nên phần đầu tiên trùng với phần có sim lớn nhất mà cột cũ ghi.
- Ô Phần khớp hiện 3 dòng dạng `nhãn a … · sim … · %`; rê chuột lên một dòng để xem cả a × sim.
- Ghi chú số lượng (nhóm số lượng của node ảnh và g) đi theo phần đầu tiên; hai phần sau có trong phần rê chuột.
- Không ghi chỉ số node (dạng `dog #3`), để khỏi lẫn với số dòng trên hộp. Hai phần cùng nhãn phân biệt bằng a, và
  bằng hộp khi rê chuột.

**Hộp trên ảnh** (`src/explain/match_view.py`):

- **Nguồn hộp.** Hộp là `box: [x, y, w, h]` của node trong chính đồ thị ảnh mà mô hình chấm điểm: val là merge_v1, test
  là count_v1_test. Node gộp mang hộp bao các hộp thành viên, và hiện "gộp n" trong đồ thị con và khi rê chuột.
- **Kích thước ảnh.** File trong `data/raw/images` có đúng `width`, `height` của đồ thị trên cả 2 138 ảnh test (đã đo
  bằng PIL). Code vẫn tính tỉ lệ từ kích thước file, phòng khi có bản ảnh đã thu nhỏ.
- **Dòng vật thể.** Khoanh phần khớp nhất, màu cam, đánh số trùng với số dòng trong bảng.
- **Dòng bộ ba.** Khoanh chủ thể và đối tượng, nối bằng mũi tên, màu tím; số dòng nằm giữa mũi tên.
- **Làm nổi một dòng.** Rê chuột lên một dòng của bảng, một hộp hoặc một node của đồ thị con:
  - các dòng khác mờ đi;
  - phần khớp thứ 2 và 3 hiện thêm, nét đứt, kèm nhãn và a.

  Bấm để giữ trạng thái này, bấm lại để bỏ. Phần rê chuột chỉ dùng CSS `:has()` (Chrome/Edge 105+, Firefox 121+).
  Phần bấm dùng `js_on_load` của `gr.HTML`.
- **Khi cuộn.** Ảnh được giữ trên đầu khung chi tiết trong lúc cuộn bảng.
- **Không vẽ:**
  - node tổng và cạnh kNN `[none]`: không bao giờ được so khớp;
  - node không có hộp: bỏ qua.
- **Mô hình cấp 1** (`--run main_r3_seed0`): đồ thị `data/graphs/vg_coco_scene_graphs.jsonl` không có hộp. Vì vậy app
  ghi một dòng chú thích thay cho ảnh; ô top-3 và đồ thị con vẫn hiện.

**Đồ thị con câu ↔ ảnh.** Nằm dưới bảng:

- bên trái là các phần của câu (từ FACTUAL), số như trong bảng;
- bên phải là các phần của ảnh nằm trong top-3 của chúng, mỗi phần ảnh một node, kèm ô cắt từ ảnh;
- nét nối dày theo w_i · a_ij · sim_ij;
- rê chuột hay bấm cũng làm nổi dòng như ở bảng và ảnh.

**Bố cục.** Lưới kết quả và khung chi tiết giờ chia 1:1 (trước là 11:9). Nhờ vậy, ở màn 1440 px, bảng có thêm cột top-3
vẫn vừa khung, không phải cuộn ngang.

**Ví dụ** (split test, mô hình `countonly_factual_r3_seed0`, chạy trên RTX 3070). Ảnh chụp trong `node-matching/`, sinh
bằng `scripts/improve/capture_demo_detail.py` từ app đang chạy:

- **"Two dogs playing with a frisbee on the grass"**, ảnh hạng 1 là 2392212 (`frisbee_pin_grass.jpg`, đang giữ dòng
  `grass`):
  - dòng `dog` dồn 98% Độ khớp vào node `dog`;
  - dòng `grass` chia cho grass 62%, lawn 22%, yard 11%;
  - hai dòng bộ ba: mỗi phần trong top-3 chỉ có a từ 0,07 đến 0,09, nên cả 3 phần chỉ chiếm 23% và 25% Độ khớp.
- **"A man riding a horse on a beach"**, ảnh hạng 1 là 2316823 (`horse.jpg`):
  - phần khớp nhất của horse, man, beach lần lượt chiếm 69%, 75% và 80% Độ khớp;
  - dòng `man –ride→ horse` khớp nhất với `man –riding→ horse` của ảnh, nhưng phần này chỉ có a = 0,12 (13%), và cả
    top-3 chỉ chiếm 25%.

  Như vậy, kênh bộ ba so câu với gần như cả đồ thị ảnh, không với riêng một bộ ba. Đây đúng là điều cột Phần khớp cũ che
  mất.
- Cùng câu với mô hình cấp 1: `horse_level1_main_r3_seed0.jpg`, có dòng chú thích thay cho ảnh.
- Toàn khung app (Browser pane, 1440 × 900, thu về 800 × 500): `app_overview_frisbee.jpg`.

**Test** (`tests/test_demo_node_matching.py`):

- top-3 xếp đúng theo a × sim, tỉ lệ nằm trong [0, 1] và tổng ≤ 1, số phần không vượt số phần của ảnh;
- qua `GraphRerankMode` với mô hình nhỏ: các dòng đã đánh số vẫn lấy đúng phần điểm, nên CLIP + các dòng + kênh lấy trung
  bình = điểm cuối, và tổng các Δ = điểm cuối − trung bình top-50;
- hàm vẽ không lỗi với:
  - node không có hộp, hộp rỗng, node tổng, node gộp;
  - cạnh `[none]`;
  - ảnh không có bộ ba;
  - đồ thị cấp 1 không có hộp.
