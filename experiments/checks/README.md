# Kiểm tra pipeline và kiểm tra trước khi thiết kế

Mục 1 và 2 kiểm tra baseline CLIP; mục 3 và 4 (thêm 2026-09-19) đo dữ liệu và thử điểm đồ thị không huấn luyện trước khi viết GNN; mục 5 ghi lại lỗi tên cấu hình mô hình CLIP đã sửa tối 2026-09-19. **Mọi số ở mục 1, 2 và 4 là số sau khi sửa**; số cũ nằm ở `experiments/archive-vit-b-32-gelu/`. Hai phép kiểm tra đầu **không phải thực nghiệm của đề tài** (đề không yêu cầu, không có đồ thị, không tính là tập dữ liệu của nhóm). Chúng trả lời hai câu hỏi về baseline CLIP: pipeline có chạy đúng không, và cách tiền xử lý ảnh có nên đổi không. Mọi số dưới đây lấy từ file JSON cùng thư mục, do script sinh ra, không sửa tay. Chạy ngày 2026-09-19 trên RTX 4090, encoder `ViT-B-32-quickgelu/openai` (open_clip 3.3.0, torch 2.14.0).

## 1. Đối chiếu với số đã công bố trên MSCOCO 5K

**Vì sao cần.** Tập test của nhóm (Visual Genome ∩ COCO, 2 138 ảnh) là tập con của tập test chuẩn MSCOCO 5K (split Karpathy, 5 000 ảnh); chỉ 2 138 ảnh trong đó có scene graph. Pool nhỏ hơn thì ít ảnh gây nhiễu hơn nên Recall cao hơn, vì vậy số của nhóm (R@1 = 39,32) **không so trực tiếp được** với số trong các bài báo (pool 5 000). Để biết pipeline có đúng không, chạy chính code của nhóm (mã hoá ảnh, mã hoá văn bản, FAISS, module đánh giá) trên đúng pool 5 000 ảnh rồi so với số đã công bố cho cùng mô hình.

**Cách chạy.** `python scripts/checks/coco5k_zero_shot.py` — tải 5 000 ảnh test gốc từ `images.cocodataset.org` (~800 MB, giữ lại ở `data/raw/coco5k_images/`), mã hoá 5 000 ảnh và 25 010 caption, chấm cả hai chiều. Kết quả: `coco5k_zero_shot.json`.

**Số tham chiếu.** Bảng benchmark chính thức của open_clip, file `docs/openclip_retrieval_results.csv`, dòng `ViT-B-32,openai` (với open_clip 3.x, trọng số này phải nạp bằng tên `ViT-B-32-quickgelu`, xem mục 5), các cột "MSCOCO image retr." (text→ảnh) và "MSCOCO text retr." (ảnh→text).

| Chiều | Nguồn | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Text → ảnh (chiều của đề tài) | Đã công bố | 30,44 | 55,94 | 66,87 |
| | Pipeline của nhóm | 30,45 | 55,97 | 66,86 |
| | Chênh lệch | +0,01 | +0,03 | −0,01 |
| Ảnh → text | Đã công bố | 50,12 | 75,00 | 83,52 |
| | Pipeline của nhóm | 50,12 | 75,00 | 83,54 |
| | Chênh lệch | 0,00 | 0,00 | +0,02 |

**Kết luận.** Tái lập gần như tuyệt đối: lệch tối đa 0,03 điểm ở cả hai chiều. Pipeline (mã hoá ảnh, mã hoá văn bản, FAISS, module đánh giá) được coi là đúng.

**Về khoảng lệch ở lần chạy đầu.** Lần chạy đầu (sáng 2026-09-19) thấp hơn số công bố khoảng 1 điểm (text→ảnh 30,26 / 54,97 / 65,71). Nguyên nhân là nạp trọng số OpenAI bằng cấu hình mô hình sai, xem mục 5. Hai giả thuyết đã loại trừ trước đó vẫn đúng: số caption mỗi ảnh (lấy 5 caption đầu không đổi kết quả) và cách tiền xử lý ảnh (mặc định của open_clip là co cạnh ngắn về 224 px rồi cắt hình vuông ở giữa, đúng cách nhóm dùng).

**Lưu ý khi trình bày.** Bài báo gốc của CLIP (Radford và cộng sự, 2021) chỉ báo cáo zero-shot retrieval cho mô hình lớn nhất (ViT-L/14, 336 px); số của ViT-B/32 lấy từ bảng open_clip ở trên. Cách làm giống hệt baseline của nhóm: mô hình đóng băng, xếp hạng theo cosine, không huấn luyện, không đồ thị.

## 2. Cách đưa ảnh về hình vuông trước khi vào CLIP

**Vì sao cần.** CLIP cắt hình vuông ở giữa nên không thấy hai đầu của ảnh ngang hoặc ảnh dọc. Trên split val: 73% ảnh ngang, 23% ảnh dọc, tỉ lệ khung hình trung vị 1,33; trung bình chỉ 71% chiều dài ảnh còn lại sau khi cắt, ảnh tệ nhất còn 25%. Câu hỏi: thêm viền cho vuông (thấy đủ ảnh) có tốt hơn không?

**Cách chạy.** `python scripts/checks/resize_mode_probe.py` — mã hoá lại 2 126 ảnh **val** theo ba chế độ của open_clip, dùng lại đặc trưng caption có sẵn, 10 633 truy vấn. Chỉ dùng val nên lựa chọn không đụng tới test (Mục 6 đề bài). Kết quả: `resize_mode_probe.json`.

| Chế độ | Ý nghĩa | R@1 | R@5 | R@10 | MRR@50 | R@50 |
|---|---|---|---|---|---|---|
| `shortest` | Cắt giữa (mặc định, **đang dùng**) | 39,13 | 67,88 | 78,97 | 52,34 | 96,72 |
| `longest` | Co cạnh dài, thêm viền đen cho vuông | 39,48 | 67,77 | 78,61 | 52,45 | 96,83 |
| `squash` | Ép méo về hình vuông | 39,53 | 67,26 | 78,36 | 52,39 | 96,78 |

**Kết luận.** Ba cách gần như ngang nhau và không cách nào thắng ở mọi chỉ số: thêm viền và ép méo nhỉnh hơn ở R@1 (+0,35 và +0,40 điểm) và MRR (+0,11 và +0,05), cắt giữa nhỉnh hơn ở R@5 (+0,11 và +0,62) và R@10 (+0,36 và +0,61). Chênh lệch cỡ này không đủ để đổi cách tiền xử lý. Giữ cắt giữa vì đó là cách của số đã công bố (mục 1), nên baseline của nhóm so được với tài liệu. (Với đặc trưng cũ, trước khi sửa lỗi ở mục 5, cắt giữa không thua ở chỉ số nào; kết luận chung không đổi.) Giải thích hợp lý: CLIP được huấn luyện với ảnh đã cắt; thêm viền làm vật thể nhỏ đi và đưa vào dải viền mô hình ít gặp; ép méo làm sai hình dạng; còn caption COCO thường tả nội dung chính nằm giữa ảnh.

**Hạn chế vẫn còn, ghi vào báo cáo.** Vật thể nằm ở rìa ảnh có thể không được CLIP nhìn thấy, trong khi scene graph của Visual Genome được gán trên toàn bộ ảnh. Đây là một chỗ bước xếp hạng lại bằng đồ thị có thể bù cho CLIP; khi có kết quả CLIP + Đồ thị, nên xem riêng nhóm ảnh có tỉ lệ khung hình lớn.

## 3. Caption có khớp được với scene graph của chính ảnh nó không

**Vì sao cần.** Thiết kế đồ thị của cấp độ 1 (sheet "Cap do 1", đổi ngày 2026-09-19) tách truy vấn thành bộ ba (chủ thể, quan hệ, đối tượng) rồi so với scene graph của từng ảnh ứng viên. Hướng này chỉ có nghĩa nếu: bộ tách lấy được bộ ba từ caption; bộ ba đó tìm thấy trong scene graph của đúng ảnh gốc; và không tìm thấy ở phần lớn ảnh khác. Phép đo này kiểm tra ba điều đó **trước khi viết GNN**, chỉ bằng **khớp nhãn chính xác** (sau khi chuẩn hoá alias), không dùng embedding và không huấn luyện. Vì vậy các con số là **cận dưới** của cách khớp mềm.

**Cách chạy.** `python scripts/checks/scene_graph_coverage.py --workers 24` (51 giây trên máy 32 nhân). Dùng toàn bộ 10 633 caption của val và 20 000 caption train lấy ngẫu nhiên (seed 0); **không đọc split test**. Bộ tách: SceneGraphParser 0.1.0 trên spaCy 3.8.16 / `en_core_web_sm` 3.8.0. Hai cách lấy nhãn cho scene graph: `vg150` (chỉ giữ 150 lớp đối tượng và 50 quan hệ của bộ benchmark scene graph quen dùng) và `open_vocab` (giữ mọi nhãn, chỉ chuẩn hoá alias). Kết quả: `scene_graph_coverage.json`. Số trên train và val gần như trùng nhau; các bảng dưới là val.

**Scene graph mỗi ảnh (val, 2 126 ảnh).**

| Cách lấy nhãn | Đối tượng / ảnh (TB) | Quan hệ / ảnh (TB) | Bộ ba khác nhau / ảnh (TB) | Ảnh không còn quan hệ nào |
|---|---|---|---|---|
| `vg150` | 12,83 | 5,34 | 4,20 | 15,52% |
| `open_vocab` | 26,43 | 17,90 | 14,65 | 2,92% |

**Bộ tách caption (val).** 99,98% caption có ít nhất một thực thể, 88,38% có ít nhất một quan hệ; trung bình 3,18 thực thể và 1,75 quan hệ mỗi caption.

**Độ phủ trên ảnh gốc của caption (val, khớp chính xác).**

| Mức khớp | `vg150` | `open_vocab` |
|---|---|---|
| Thực thể của caption có trong ảnh (tính theo thực thể) | 31,03% | 44,62% |
| Caption có ≥1 thực thể trúng | 67,78% | 80,62% |
| Caption có **mọi** thực thể trúng | 3,71% | 10,97% |
| Quan hệ của caption mà hai đầu được nối trong ảnh, đúng chiều | 3,80% | 8,20% |
| Như trên, không kể chiều | 4,64% | 10,15% |
| Quan hệ khớp đủ bộ ba (chủ thể, quan hệ, đối tượng) | 1,62% | 3,40% |
| Caption có ≥1 cặp trúng (không kể chiều) | 7,66% | 15,44% |
| Caption có ≥1 bộ ba trúng | 2,73% | 5,58% |

**Độ chọn lọc trong pool val (`open_vocab`).** Với các caption trúng trên ảnh gốc, tỉ lệ ảnh trong pool cũng trúng (càng thấp càng phân biệt tốt):

| Tín hiệu | Số caption trúng ảnh gốc | Tỉ lệ pool cũng trúng (TB / trung vị) |
|---|---|---|
| ≥1 thực thể | 8 572 | 13,98% / 11,85% |
| Mọi thực thể | 1 166 | 0,59% / 0,19% |
| ≥1 cặp | 1 642 | 0,64% / 0,28% |
| ≥1 bộ ba | 593 | 0,30% / 0,14% |

**Vì sao tỉ lệ thực thể chỉ 45%.** Xem theo từng nhãn (mục `frequent_entities` trong JSON): danh từ cụ thể trúng rất cao, ví dụ `horse` 95,8%, `umbrella` 94,4%, `cat` 94,2%, `clock` 93,7%, `train` 90,9%, `man` 76,7%. Phần kéo tỉ lệ xuống gồm ba nhóm:

- Nhiễu của bộ tách: `that`, `it`, `group` được coi là thực thể nhưng không phải vật thể (trúng 0–4%); `side`, `picture`, `background`, `top`, `bunch` cũng nằm trong 30 nhãn trượt nhiều nhất.
- Khác mức khái quát: caption ghi `people` (trúng 39%) hoặc `person` (34%) trong khi ảnh được gán `man`, `woman`, `player`.
- Từ chỉ khung cảnh: `room` 28%, `street` 40%, `field` 55%; scene graph gán nhãn vật thể, ít gán khung cảnh.

Về quan hệ, các vị từ trượt nhiều nhất là giới từ chung chung: `of` (2 761 lần), `with` (2 745), `in` (2 392), `on` (2 229).

**Kết luận.**

1. **Khớp bộ ba chính xác không đủ làm tín hiệu xếp hạng**: chỉ 5,58% caption có bộ ba trúng trên ảnh gốc, 15,44% có cặp trúng. Khi trúng thì rất chọn lọc (0,3–0,6% pool), nhưng phủ quá ít truy vấn. Một phần nguyên nhân nằm ở dữ liệu chứ không chỉ ở từ vựng: scene graph của Visual Genome thưa (khoảng 18 quan hệ mỗi ảnh, nhiều quan hệ kiểu `man wearing shirt`, `window on building`), không nhất thiết gán đúng quan hệ mà caption nhắc tới.
2. **Mức thực thể là tín hiệu mạnh và sẵn có**: vật thể cụ thể trúng 75–95%, và "mọi thực thể đều có trong ảnh" chỉ đúng với 0,59% pool.
3. **Không lọc theo VG150**: bộ lọc này bỏ mất `water`, `grass`, `field` (trúng 0% so với 82%, 86%, 55% khi không lọc), làm 15,52% ảnh không còn quan hệ nào và giảm khoảng một nửa mọi độ phủ. Dùng `open_vocab`, có thể cắt theo tần suất trên train.
4. **Bắt buộc phải khớp mềm** (vector chữ của nhãn, hoặc embedding học bằng GNN) để nối `people`–`man`, `with`–`holding`. Phép đo này **chưa trả lời được** khớp mềm có đủ tốt hay không; nó chỉ cho thấy khớp ký hiệu thuần tuý thì không đủ.
5. Cần lọc thực thể nhiễu của bộ tách (đại từ, `group`, `side`, `picture`…) trước khi dựng đồ thị truy vấn.

**Bước kiểm tra tiếp theo (chưa làm).** Một điểm đồ thị không cần huấn luyện: mã hoá nhãn và cụm bộ ba bằng CLIP text, điểm = trung bình độ tương tự lớn nhất giữa từng thực thể / bộ ba của truy vấn với các thực thể / bộ ba của ảnh, rồi xếp hạng lại top-50 trên val với α quét từ 0 đến 1. Nếu cách này không nhích được R@1 so với CLIP thuần thì GNN học trên cùng dữ liệu cũng khó làm được, và cần xét lại thiết kế.

**Hạn chế của phép đo.** Chỉ một bộ tách, chưa so với bộ tách khác; mẫu train 20 000 caption (val dùng đủ); đối tượng có nhiều tên chỉ lấy tên đầu; chưa gộp số ít / số nhiều ngoài những gì file alias đã có.

## 4. Điểm đồ thị không huấn luyện có giúp được CLIP không

**Vì sao cần.** Mục 3 cho thấy khớp nhãn chính xác phủ quá ít truy vấn, nên phải khớp mềm bằng vector. Trước khi viết GNN, phép thử này hỏi: chỉ cần vector hoá các phần của đồ thị bằng CLIP text (đóng băng), **không huấn luyện gì, không GNN**, thì điểm đồ thị đã giúp được CLIP chưa? Nếu chưa thì GNN học trên cùng dữ liệu cũng khó làm được.

**Cách chạy.** `python scripts/checks/soft_graph_rerank_probe.py --workers 24` (60 giây, RTX 4090). Chỉ dùng split **val**: 10 633 truy vấn, pool 2 126 ảnh, xếp hạng lại top-50 của baseline CLIP. Scene graph dùng mọi nhãn (không lọc VG150). Mỗi nhãn đối tượng được mã hoá theo mẫu `a photo of a {tên}`, mỗi bộ ba được viết thành cụm (`boy holding ball`) rồi mã hoá; tổng cộng 8 241 nhãn và 34 495 cụm bộ ba khác nhau. Mỗi phần của truy vấn đi tìm phần giống nó nhất bên ảnh (cosine), điểm đồ thị = trung bình trên các phần của truy vấn. Điểm cuối = `α·z(CLIP) + (1−α)·z(đồ thị)`, trong đó z là chuẩn hoá z-score trong top-50 của từng truy vấn; truy vấn hoặc ảnh không có phần nào thì điểm đồ thị coi là 0 (trung tính). α quét từ 0 đến 1, bước 0,1. Đại từ do bộ tách sinh ra (`that`, `it`…) bị bỏ. Kết quả: `soft_graph_rerank_probe.json` (có toàn bộ đường quét α).

| Điểm đồ thị | α tốt nhất | R@1 | R@5 | R@10 | MRR@50 |
|---|---|---|---|---|---|
| Không có (CLIP thuần, α = 1) | — | 39,13 | 67,88 | 78,97 | 52,34 |
| Mức đối tượng (chỉ nhãn vật thể, không quan hệ) | 0,6 | **42,99** | 70,46 | 81,24 | 55,67 |
| Mức bộ ba, lấy bộ ba giống nhất | 0,7 | 40,93 | 69,02 | 80,32 | 53,99 |
| Mức bộ ba, softmax nhiệt độ 0,02 | 0,6 | 41,14 | 69,15 | 80,18 | 54,04 |
| Mức bộ ba, softmax nhiệt độ 0,1 | 0,6 | 41,16 | 69,03 | 80,01 | 54,06 |
| Đối tượng + bộ ba (trung bình hai z-score) | 0,5 | 42,93 | 70,06 | 80,49 | 55,42 |
| Đối chứng: gán cho mỗi ảnh scene graph của ảnh khác (cả ba loại điểm) | 1,0 | 39,13 | 67,88 | 78,97 | 52,34 |

Đường quét α của mức đối tượng trơn và có một đỉnh: R@1 = 26,34 (α = 0, chỉ đồ thị) → 39,78 (0,3) → 42,92 (0,5) → **42,99 (0,6)** → 41,84 (0,8) → 39,13 (1,0).

**Kết luận.**

1. **Scene graph có thông tin mà CLIP chưa dùng được**: chưa huấn luyện gì mà R@1 tăng 3,86 điểm (39,13 → 42,99), R@10 tăng 2,27 điểm. Đối chứng xáo đồ thị không tăng chút nào (α tốt nhất = 1,0, tức bỏ hẳn điểm đồ thị), nên phần tăng đến từ việc đồ thị **đúng là của ảnh đó**, không phải do cách chuẩn hoá hay do cộng thêm một điểm bất kỳ.
2. **Mức đối tượng mạnh hơn mức bộ ba** (+3,86 so với +1,80), và ghép hai mức **không hơn** dùng riêng mức đối tượng (42,93 so với 42,99). Khi chưa huấn luyện, quan hệ chưa đóng góp thêm gì ngoài danh sách vật thể. Hai cách giải thích, chưa phân biệt được: (a) vector CLIP text của cụm bộ ba cũng mang tính túi từ, `ball on table` vẫn gần `child playing with ball` vì chung chữ `ball`; (b) scene graph của Visual Genome ít khi gán đúng quan hệ mà caption nhắc tới (mục 3).
3. **Softmax so với lấy max ở phía ảnh gần như không khác** (41,14–41,16 so với 40,93) khi chưa có gì để học.
4. Đồ thị một mình kém CLIP xa (R@1 = 26,34 so với 39,13): nó là tín hiệu bổ sung, không thay được CLIP.

**Ý nghĩa cho thiết kế cấp độ 1.**
- Hướng "đồ thị riêng từng ảnh + khớp mềm" **khả thi**, tiếp tục.
- Bảng này là **baseline đồ thị không huấn luyện** để so với GAT trong báo cáo: GAT phải vượt 42,99 (trên val) thì mới chứng minh được việc học và lan truyền thông tin trong đồ thị có ích.
- Câu hỏi mở mà GAT phải trả lời: làm cho **quan hệ** đóng góp thêm ngoài danh sách vật thể. Ablation "bỏ cạnh quan hệ" của đề sẽ đo đúng điều này; nếu sau khi huấn luyện mà bỏ cạnh vẫn không đổi kết quả thì phải báo cáo đúng như vậy.
- Điểm đồ thị nên có cả hai mức (đối tượng và bộ ba); mức đối tượng không được bỏ.

**Hạn chế của phép thử.** α được chọn trên chính tập val dùng để báo cáo, nên con số hơi lạc quan (11 giá trị α trên 10 633 truy vấn, sai lệch nhỏ); kết quả chính thức phải chọn α trên val rồi đo trên test. Chưa chạy trên test. Scene graph là chú thích gán tay của ảnh ứng viên; ảnh mới ngoài thực tế không có sẵn. 1 510 truy vấn (14,2%) không có bộ ba nào sau khi bỏ đại từ, và 62 ảnh (2,9%) không có quan hệ nào; các trường hợp này nhận điểm bộ ba trung tính.

## 5. Tên cấu hình mô hình CLIP (QuickGELU)

**Vì sao cần.** Mọi lần nạp CLIP, open_clip đều in cảnh báo `QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)`. Trọng số `openai` được huấn luyện với hàm kích hoạt QuickGELU, còn cấu hình `ViT-B-32` của open_clip 3.x dùng GELU thường; tên đúng là `ViT-B-32-quickgelu`. Cảnh báo này đã bị bỏ sót từ lần trích đặc trưng đầu tiên, và lần đối chiếu MSCOCO 5K đầu tiên thấp hơn số công bố khoảng 1 điểm mà chưa rõ lý do.

**Cách chạy.** `python scripts/checks/quickgelu_probe.py` — nạp hai tên mô hình với cùng trọng số `openai`, mã hoá lại split val của nhóm và pool MSCOCO 5K, không ghi đè đặc trưng nào. Kết quả: `quickgelu_probe.json`.

| Tên mô hình | Có cảnh báo | Val của nhóm R@1 / R@5 / R@10 | 5K text→ảnh R@1 / R@5 / R@10 | 5K ảnh→text R@1 / R@5 / R@10 |
|---|---|---|---|---|
| `ViT-B-32` (dùng trước đây) | Có | 38,00 / 66,20 / 77,19 | 30,26 / 54,97 / 65,71 | 49,16 / 73,34 / 82,34 |
| `ViT-B-32-quickgelu` | Không | 39,13 / 67,88 / 78,97 | 30,45 / 55,97 / 66,86 | 50,12 / 75,00 / 83,54 |
| Đã công bố | — | — | 30,44 / 55,94 / 66,87 | 50,12 / 75,00 / 83,52 |

**Kết luận.** Khoảng lệch 1 điểm là do tên cấu hình sai. Đã sửa `configs/default.yaml` thành `ViT-B-32-quickgelu`, **trích lại toàn bộ đặc trưng** và chạy lại baseline, mục 1, mục 2 và mục 4 trong tối 2026-09-19. Baseline của nhóm tăng khoảng 1 điểm (test R@1 38,50 → 39,32; val 38,00 → 39,13). Mục 3 không dùng CLIP nên không đổi. Ai đã tải đặc trưng từ Drive trước thời điểm này phải tải lại (checksum mới ở `data/MANIFEST.md`).

**Bài học.** Đọc cảnh báo của bộ nạp mô hình trước khi tin một baseline; tái lập "gần đúng" (lệch 1 điểm) vẫn có thể che một lỗi cấu hình.

## File giữ lại

| Thứ | Ở đâu | Ghi chú |
|---|---|---|
| Script | `scripts/checks/` | Trong git |
| Kết quả | `experiments/checks/*.json` | Trong git |
| 5 000 ảnh test MSCOCO | `data/raw/coco5k_images/` (793 MB) | Chỉ trên máy chạy; script chạy lại sẽ bỏ qua ảnh đã có |
| Đặc trưng MSCOCO 5K | `data/features/coco5k_{image,caption}.npy` + `.ids.json` (~60 MB) | Chỉ trên máy chạy; checksum ở `data/MANIFEST.md`. Không đưa lên Drive vì các gói khác không dùng |
| Caption đã tách thành đồ thị | `data/processed/query_graphs_check_{train,val}.json` (5,4 MB và 2,9 MB) | Chỉ trên máy chạy; SHA-256 bắt đầu bằng `3f78bc2a` và `c8680264`; chạy lại script với seed 0 sẽ sinh lại |
| Danh sách VG150 | `data/raw/vg150/` | Tải bằng `scripts/download_data.py`; checksum ở `data/MANIFEST.md` |
| Vector CLIP text của các phần đồ thị (val) | `data/features/vg_coco_val_graph_parts.npy` + `.ids.json` (84 MB) | Chỉ trên máy chạy; checksum ở `data/MANIFEST.md`; script chạy lại sẽ sinh lại |
| Kết quả và đặc trưng trước khi sửa tên mô hình | `experiments/archive-vit-b-32-gelu/` (trong git); `data/features/archive-vit-b-32-gelu/` trên máy GPU, máy Windows của Lộc và Drive | Giữ để đối chiếu, không dùng cho thực nghiệm mới |

## 6. Tham chiếu ba kênh cho mô hình GAT

Phép kiểm tra này tạo đúng phép tính không huấn luyện mà mô hình re-ranker phải tái tạo ở bước 0. Chỉ dùng split **val**, không đọc test. Hai kênh đồ thị giữ riêng vật thể và bộ ba; điểm cuối dùng `Graph = z(beta * z(S_objects) + (1 - beta) * z(S_triples))` rồi trộn với CLIP.

| Cấu hình | R@1 | R@5 | R@10 | MRR@50 |
|---|---:|---:|---:|---:|
| Tham chiếu tốt nhất K=50 | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| Xáo scene graph | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| beta = 1 (vật thể) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| beta = 0 (bộ ba) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |

Chạy trên máy GPU sau khi các đặc trưng và split đã có:

```bash
~/venvs/mgrx/bin/python scripts/checks/three_channel_probe.py --workers 16 --top-k 50
~/venvs/mgrx/bin/python scripts/checks/three_channel_probe.py --workers 16 --top-k 50 --encoder sbert
```

Kết quả nằm ở `experiments/checks/three_channel_probe.json` (hoặc hậu tố `_sbert.json`). Điểm thô, z-score, điểm trộn và thứ hạng của 200 truy vấn đầu được lưu trong `experiments/checks/three_channel_probe_reference_scores.npz`.