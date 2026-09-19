# Kiểm tra pipeline và kiểm tra trước khi thiết kế

Mục 1 và 2 kiểm tra baseline CLIP; mục 3 và 4 (thêm 2026-09-19) đo dữ liệu và thử điểm đồ thị không huấn luyện trước khi viết GNN. Hai phép kiểm tra đầu **không phải thực nghiệm của đề tài** (đề không yêu cầu, không có đồ thị, không tính là tập dữ liệu của nhóm). Chúng trả lời hai câu hỏi về baseline CLIP: pipeline có chạy đúng không, và cách tiền xử lý ảnh có nên đổi không. Mọi số dưới đây lấy từ file JSON cùng thư mục, do script sinh ra, không sửa tay. Chạy ngày 2026-09-19 trên RTX 4090, encoder `ViT-B-32/openai` (open_clip 3.3.0, torch 2.14.0).

## 1. Đối chiếu với số đã công bố trên MSCOCO 5K

**Vì sao cần.** Tập test của nhóm (Visual Genome ∩ COCO, 2 138 ảnh) là tập con của tập test chuẩn MSCOCO 5K (split Karpathy, 5 000 ảnh); chỉ 2 138 ảnh trong đó có scene graph. Pool nhỏ hơn thì ít ảnh gây nhiễu hơn nên Recall cao hơn, vì vậy số của nhóm (R@1 = 38,50) **không so trực tiếp được** với số trong các bài báo (pool 5 000). Để biết pipeline có đúng không, chạy chính code của nhóm (mã hoá ảnh, mã hoá văn bản, FAISS, module đánh giá) trên đúng pool 5 000 ảnh rồi so với số đã công bố cho cùng mô hình.

**Cách chạy.** `python scripts/checks/coco5k_zero_shot.py` — tải 5 000 ảnh test gốc từ `images.cocodataset.org` (~800 MB, giữ lại ở `data/raw/coco5k_images/`), mã hoá 5 000 ảnh và 25 010 caption, chấm cả hai chiều. Kết quả: `coco5k_zero_shot.json`.

**Số tham chiếu.** Bảng benchmark chính thức của open_clip, file `docs/openclip_retrieval_results.csv`, dòng `ViT-B-32,openai`, các cột "MSCOCO image retr." (text→ảnh) và "MSCOCO text retr." (ảnh→text).

| Chiều | Nguồn | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Text → ảnh (chiều của đề tài) | Đã công bố | 30,44 | 55,94 | 66,87 |
| | Pipeline của nhóm | 30,26 | 54,97 | 65,71 |
| | Chênh lệch | −0,18 | −0,97 | −1,16 |
| Ảnh → text | Đã công bố | 50,12 | 75,00 | 83,52 |
| | Pipeline của nhóm | 49,16 | 73,34 | 82,34 |
| | Chênh lệch | −0,96 | −1,66 | −1,18 |

**Kết luận.** Tái lập sát, lệch tối đa 1,66 điểm, ở chiều của đề tài lệch tối đa 1,16 điểm và R@1 chỉ lệch 0,18. Lỗi hệ thống (sai biến thể mô hình, sai tiền xử lý, chấm điểm sai) thường gây lệch vài điểm đến hàng chục điểm, nên pipeline được coi là đúng. Không ghi là "khớp hoàn toàn".

**Về khoảng lệch còn lại.**
- Đã loại trừ: số caption mỗi ảnh. Một số ảnh có 6–7 caption; khi chỉ lấy 5 caption đầu (25 000 câu) kết quả gần như không đổi (text→ảnh 30,26 / 54,98 / 65,71).
- Đã kiểm tra: tiền xử lý ảnh giống nhau. Mặc định của open_clip (`PreprocessCfg.resize_mode = "shortest"`) là co cạnh ngắn về 224 px rồi **cắt hình vuông ở giữa**, đúng cách nhóm đang dùng; không phải thêm viền đen.
- Chưa kiểm tra: bộ benchmark chuẩn có thể chạy ở độ chính xác số khác (nửa chính xác), phiên bản thư viện khác, hoặc xử lý văn bản caption hơi khác.

**Lưu ý khi trình bày.** Bài báo gốc của CLIP (Radford và cộng sự, 2021) chỉ báo cáo zero-shot retrieval cho mô hình lớn nhất (ViT-L/14, 336 px); số của ViT-B/32 lấy từ bảng open_clip ở trên. Cách làm giống hệt baseline của nhóm: mô hình đóng băng, xếp hạng theo cosine, không huấn luyện, không đồ thị.

## 2. Cách đưa ảnh về hình vuông trước khi vào CLIP

**Vì sao cần.** CLIP cắt hình vuông ở giữa nên không thấy hai đầu của ảnh ngang hoặc ảnh dọc. Trên split val: 73% ảnh ngang, 23% ảnh dọc, tỉ lệ khung hình trung vị 1,33; trung bình chỉ 71% chiều dài ảnh còn lại sau khi cắt, ảnh tệ nhất còn 25%. Câu hỏi: thêm viền cho vuông (thấy đủ ảnh) có tốt hơn không?

**Cách chạy.** `python scripts/checks/resize_mode_probe.py` — mã hoá lại 2 126 ảnh **val** theo ba chế độ của open_clip, dùng lại đặc trưng caption có sẵn, 10 633 truy vấn. Chỉ dùng val nên lựa chọn không đụng tới test (Mục 6 đề bài). Kết quả: `resize_mode_probe.json`.

| Chế độ | Ý nghĩa | R@1 | R@5 | R@10 | MRR@50 | R@50 |
|---|---|---|---|---|---|---|
| `shortest` | Cắt giữa (mặc định, **đang dùng**) | 38,00 | 66,20 | 77,19 | 51,06 | 96,59 |
| `longest` | Co cạnh dài, thêm viền đen cho vuông | 37,87 | 65,34 | 77,09 | 50,67 | 96,46 |
| `squash` | Ép méo về hình vuông | 37,99 | 65,05 | 76,75 | 50,72 | 96,50 |

**Kết luận.** Ba cách gần như ngang nhau (chênh 0,1–1,2 điểm), cắt giữa không thua ở chỉ số nào. Giữ nguyên mặc định, **không trích lại đặc trưng**. Giải thích hợp lý: CLIP được huấn luyện với ảnh đã cắt; thêm viền làm vật thể nhỏ đi và đưa vào dải viền mô hình ít gặp; ép méo làm sai hình dạng; còn caption COCO thường tả nội dung chính nằm giữa ảnh.

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

**Cách chạy.** `python scripts/checks/soft_graph_rerank_probe.py --workers 24` (58 giây, RTX 4090). Chỉ dùng split **val**: 10 633 truy vấn, pool 2 126 ảnh, xếp hạng lại top-50 của baseline CLIP. Scene graph dùng mọi nhãn (không lọc VG150). Mỗi nhãn đối tượng được mã hoá theo mẫu `a photo of a {tên}`, mỗi bộ ba được viết thành cụm (`boy holding ball`) rồi mã hoá; tổng cộng 8 241 nhãn và 34 495 cụm bộ ba khác nhau. Mỗi phần của truy vấn đi tìm phần giống nó nhất bên ảnh (cosine), điểm đồ thị = trung bình trên các phần của truy vấn. Điểm cuối = `α·z(CLIP) + (1−α)·z(đồ thị)`, trong đó z là chuẩn hoá z-score trong top-50 của từng truy vấn; truy vấn hoặc ảnh không có phần nào thì điểm đồ thị coi là 0 (trung tính). α quét từ 0 đến 1, bước 0,1. Đại từ do bộ tách sinh ra (`that`, `it`…) bị bỏ. Kết quả: `soft_graph_rerank_probe.json` (có toàn bộ đường quét α).

| Điểm đồ thị | α tốt nhất | R@1 | R@5 | R@10 | MRR@50 |
|---|---|---|---|---|---|
| Không có (CLIP thuần, α = 1) | — | 38,00 | 66,20 | 77,19 | 51,06 |
| Mức đối tượng (chỉ nhãn vật thể, không quan hệ) | 0,6 | **41,95** | 69,75 | 80,05 | 54,63 |
| Mức bộ ba, lấy bộ ba giống nhất | 0,6 | 39,89 | 67,37 | 78,43 | 52,68 |
| Mức bộ ba, softmax nhiệt độ 0,02 | 0,6 | 40,05 | 67,38 | 78,64 | 52,81 |
| Mức bộ ba, softmax nhiệt độ 0,1 | 0,6 | 39,80 | 67,84 | 78,73 | 52,70 |
| Đối tượng + bộ ba (trung bình hai z-score) | 0,6 | 41,80 | 69,05 | 80,13 | 54,40 |
| Đối chứng: gán cho mỗi ảnh scene graph của ảnh khác (cả ba loại điểm) | 1,0 | 38,00 | 66,20 | 77,19 | 51,06 |

Đường quét α của mức đối tượng trơn và có một đỉnh: R@1 = 25,69 (α = 0, chỉ đồ thị) → 38,59 (0,3) → 41,75 (0,5) → **41,95 (0,6)** → 40,65 (0,8) → 38,00 (1,0).

**Kết luận.**

1. **Scene graph có thông tin mà CLIP chưa dùng được**: chưa huấn luyện gì mà R@1 tăng 3,95 điểm (38,00 → 41,95), R@10 tăng 2,86 điểm. Đối chứng xáo đồ thị không tăng chút nào (α tốt nhất = 1,0, tức bỏ hẳn điểm đồ thị), nên phần tăng đến từ việc đồ thị **đúng là của ảnh đó**, không phải do cách chuẩn hoá hay do cộng thêm một điểm bất kỳ.
2. **Mức đối tượng mạnh hơn mức bộ ba** (+3,95 so với +1,89), và ghép hai mức **không hơn** dùng riêng mức đối tượng (41,80 so với 41,95). Khi chưa huấn luyện, quan hệ chưa đóng góp thêm gì ngoài danh sách vật thể. Hai cách giải thích, chưa phân biệt được: (a) vector CLIP text của cụm bộ ba cũng mang tính túi từ, `ball on table` vẫn gần `child playing with ball` vì chung chữ `ball`; (b) scene graph của Visual Genome ít khi gán đúng quan hệ mà caption nhắc tới (mục 3).
3. **Softmax so với lấy max ở phía ảnh gần như không khác** (39,80–40,05 so với 39,89) khi chưa có gì để học.
4. Đồ thị một mình kém CLIP xa (R@1 = 25,69 so với 38,00): nó là tín hiệu bổ sung, không thay được CLIP.

**Ý nghĩa cho thiết kế cấp độ 1.**
- Hướng "đồ thị riêng từng ảnh + khớp mềm" **khả thi**, tiếp tục.
- Bảng này là **baseline đồ thị không huấn luyện** để so với GAT trong báo cáo: GAT phải vượt 41,95 thì mới chứng minh được việc học và lan truyền thông tin trong đồ thị có ích.
- Câu hỏi mở mà GAT phải trả lời: làm cho **quan hệ** đóng góp thêm ngoài danh sách vật thể. Ablation "bỏ cạnh quan hệ" của đề sẽ đo đúng điều này; nếu sau khi huấn luyện mà bỏ cạnh vẫn không đổi kết quả thì phải báo cáo đúng như vậy.
- Điểm đồ thị nên có cả hai mức (đối tượng và bộ ba); mức đối tượng không được bỏ.

**Hạn chế của phép thử.** α được chọn trên chính tập val dùng để báo cáo, nên con số hơi lạc quan (11 giá trị α trên 10 633 truy vấn, sai lệch nhỏ); kết quả chính thức phải chọn α trên val rồi đo trên test. Chưa chạy trên test. Scene graph là chú thích gán tay của ảnh ứng viên; ảnh mới ngoài thực tế không có sẵn. 1 510 truy vấn (14,2%) không có bộ ba nào sau khi bỏ đại từ, và 62 ảnh (2,9%) không có quan hệ nào; các trường hợp này nhận điểm bộ ba trung tính.

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
