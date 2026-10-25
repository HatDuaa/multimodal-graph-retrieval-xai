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
