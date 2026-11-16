# Thí nghiệm 3 — Train trên graph ảnh đã làm sạch và nối liền (bản gộp `fullgraph_factual_r3`)

Mọi số ở đây là của **val**. Test chưa chấm.

Mốc so sánh là 1a, `raw_factual_r3` (`../fusion/raw/`): `--fusion raw` trên dữ liệu `factual_v1`, R@1 47,48 ± 0,10.

## Kế hoạch đã đổi

Kế hoạch ban đầu (21/10) là 4 bản tuần tự, mỗi bản thêm một thứ: 3.1 gộp trùng, 3.2 hình học trên cạnh, 3.3 graph
liền, 3.4 node tổng. Sau đó có thêm đối chứng 3.3 tắt GAT.

Ngày 22/10, Lộc đổi sang **một bản duy nhất gộp cả 4 thay đổi**. Lý do: chỉ khi bản gộp tốt hơn mốc rõ ràng mới
tách ablation để biết phần nào có tác dụng, và việc đó Lộc quyết sau.

Vì vậy chưa có run nào của 3.1–3.4 và đối chứng tắt GAT. Trước khi đổi kế hoạch, mình mới chỉ đo bộ nhớ và tốc độ cho
các bản trung gian, kết quả lưu ở `memory_probe.jsonl`.

- Dòng cuối của file đó bị cắt dở, vì phiên làm việc bị ngắt giữa lúc ghi. Mình giữ nguyên, không xoá.
- Server reboot tối 21/10, lúc đó chưa có run nào chạy, nên không mất kết quả.

## Bản gộp gồm những gì

So với `raw_factual_r3`, bản gộp đổi 4 thứ cùng lúc. Phía câu giữ nguyên graph FACTUAL của `factual_v1`.

### 1. Gộp trùng: dữ liệu `merge_v1`

Script đóng gói là `scripts/improve/package_merged.py`, cấu hình là `configs/merge_v1.yaml`.

- **Định nghĩa trùng** giống hệt bản audit (`src/graph/dedup.py`): cùng loại theo synset/hypernym, trong đó mọi
  synset chỉ người đều tính là `person.n.01`; và IoU ≥ 0,7; khép bắc cầu bằng union-find.
- **Node sau khi gộp** lấy nhãn và synset của **khung lớn nhất**; nếu bằng diện tích thì lấy node có chỉ số nhỏ nhất.
  Hộp của node là hộp bao các khung. Chọn cách này để khớp với audit (`../graph-dedup-audit/`).
- **Cạnh** được nối lại vào node đã gộp; bộ ba (s, p, o) trùng và vòng tự thân bị bỏ.

Số liệu đóng gói (`data/graphs/merge_v1/MANIFEST.json`):

| Split | Node trước → sau | Cạnh trước → sau | Node là kết quả gộp |
|---|---|---|---|
| train | 1 255 685 → 1 168 021 (−7,0%) | 847 129 → 822 515 | 60 277 |
| val | 56 181 → 52 230 | 38 066 → 36 954 | 2 664 |

Nhãn của một số node đổi sau khi gộp, nên có 8 768 cụm bộ ba mới. Các cụm này được mã hoá bằng đúng CLIP đóng băng của
bảng part cũ (`data/features/merge_v1/vg_coco_graph_parts_extra.*`). Mình mã hoá lại 1 000 khoá cũ để kiểm tra: sai
lệch lớn nhất bằng 0,0.

Test chưa được đóng gói theo `merge_v1`. Việc này chỉ làm khi nhóm quyết định chấm test lần cuối; khi đó cần chạy
`build_boxes.py` cho split test trước.

### 2. Hình học trên cạnh (`--edge-geometry`)

Mỗi cạnh s → o có hộp ở cả hai đầu được gán 6 số (`src/graph/structure.py`):

- (cx_o − cx_s)/w_s và (cy_o − cy_s)/h_s, cắt vào khoảng [−10, 10];
- log(w_o/w_s) và log(h_o/h_s);
- IoU;
- tỉ lệ diện tích, viết ở dạng bị chặn diện tích_o / (diện tích_s + diện tích_o). Số này là hàm của hai số log, nên
  không thêm thông tin mới; giữ lại vì đề yêu cầu.

Cạnh rộng hoặc cao dưới 1 px được tính là 1 px.

6 số này qua một MLP 6 → 64 → 256, rồi cộng vào vector cạnh `p_edge(·)`. Vector cạnh này được dùng cả trong GAT lẫn
trong `triple_mlp`.

- Lớp cuối của MLP khởi tạo bằng 0, nên ở bước 0 mô hình y như khi chưa có hình học.
- Cạnh phía câu và cạnh của node tổng không có hộp, nên phần hình học bằng 0.

### 3. Graph liền (`--knn-edges 5`)

- **Cạnh kNN.** Mỗi node ảnh nhận một cạnh từ mỗi node trong 5 node gần nó nhất, tính theo khoảng cách tâm khung chia
  cho đường chéo ảnh. Chia cho đường chéo không đổi thứ tự láng giềng trong một ảnh. Hoà khoảng cách thì xếp theo chỉ
  số node.
- **Cầu nối.** Nếu sau bước trên graph vẫn còn nhiều mảnh, ta lặp lại việc nối cặp node gần nhất giữa hai mảnh khác
  nhau, cho tới khi liền. Đây là thuật toán Kruskal trên graph các mảnh. Mỗi cầu nối được thêm theo cả hai chiều.
- Nếu cặp có hướng đó đã có cạnh có nhãn thì không thêm.
- **Vị từ `[none]`.** Cạnh tự thêm dùng một vector 512 chiều học được, đi qua `p_edge`, thay cho vector CLIP của vị từ.
  Vector này khởi tạo ngẫu nhiên với norm 1, cùng cỡ với vector CLIP.
- **Không phải bộ ba.** Cạnh tự thêm chỉ truyền tin trong GAT. Kênh bộ ba vẫn chỉ gồm cạnh có nhãn; unit test kiểm tra
  điều này.

### 4. Node tổng (`--global-node`)

Mỗi graph ảnh và mỗi graph câu có thêm một node tổng, nối hai chiều tới mọi node bằng loại cạnh riêng. Loại cạnh này
có vector `[global]` học được, đi qua `p_edge`, không có hình học.

- **Khởi tạo:** vector vào của node tổng là trung bình chuẩn hoá của các vector node chấm điểm được trong graph đó. Lý
  do: nếu dùng một vector học được cố định thì ở lớp GAT thứ nhất node tổng chưa mang thông tin gì. Khởi tạo bằng
  trung bình thì ngay lớp 1 nó đã tóm tắt cả graph, và lớp 2 phát tóm tắt đó lại cho các node.
- **Không tham gia so khớp vật thể:** mặt nạ part của node tổng bằng False.

### Công thức train

Giữ như `raw_factual_r3`:

- cùng hằng số `--fusion-scale` (std train ở bước 0 của `factual_v1`, lấy từ `../fusion/train_channel_stats.json`);
- micro-batch 32, `expandable_segments:False`;
- tối đa 10 epoch, dừng khi 2 epoch liền không vượt R@1 val tốt nhất;
- seed 0, 1, 2.

Cờ mới mặc định giữ hành vi cũ. Mình đã kiểm tra bằng cách chấm lại `raw_factual_r3_seed0` với code mới: R@1 val ra
0,47361986269162043, trùng từng chữ số với `metrics.json`.

Lệnh chạy lại là `run.sh` (train), rồi `diagnostics.sh` (chẩn đoán). Test đơn vị ở `tests/test_full_graph.py`, 13
test; toàn bộ 109 test pass.

### Hạ tầng

- **Tốc độ.** Số cạnh ảnh tăng khoảng 10 lần: mỗi micro-batch 32 câu có 287 nghìn cạnh, so với 27 nghìn. Bản đầu
  chạy backward 133 ms mỗi micro-batch. Nguyên nhân là phép gather vector loại cạnh theo chỉ số: backward của nó phải
  sắp xếp khoảng 300 nghìn chỉ số trùng. Mình đổi sang `torch.where` và backward còn 51 ms. Phép tính giữ nguyên.
- **Thời gian thực tế:** 681–742 giây mỗi epoch, kể cả chấm val, so với khoảng 312 giây của 1a. Peak GPU 7,6 GB.
- **Tổng:** train 3 seed mất 4,40 giờ, chạy từ 20:35 ngày 22/10 đến 01:01 ngày 23/10; chẩn đoán mất 3 phút.
- **Cảnh báo bộ nhớ.** Lúc chấm val ở epoch 1 của seed 0 và seed 2, PyTorch in cảnh báo "allocation failed with OOM".
  Nguyên nhân là bộ nhớ đệm bị phân mảnh khi không có `expandable_segments`. Bộ cấp phát tự giải phóng đệm rồi cấp phát
  lại thành công; không run nào lỗi hay phải chạy lại.
- **Seed 0 chạy hai lần.** Lần đầu bị dừng sau vài giây, khi còn đang nạp dữ liệu, để commit code trước. Lần chạy lại
  ghi đè `config.json` của lần đầu và ghi đúng commit `9962f52`. Việc này có ghi trong `logs/timeline.log`.

## Val (checkpoint tốt nhất theo val R@1, 3 seed)

| Mô hình | R@1 | R@5 | R@10 | MRR | Epoch tốt nhất | Số epoch | Thời gian train |
|---|---|---|---|---|---|---|---|
| fullgraph_factual_r3 | **48.38 ± 0.17** | **75.25 ± 0.09** | **84.71 ± 0.09** | **60.51 ± 0.17** | 4, 8, 5 | 6, 10, 7 | 4.40 h (3 seed) |
| raw_factual_r3 (mốc 1a) | 47.48 ± 0.10 | 74.70 ± 0.18 | 84.10 ± 0.08 | 59.72 ± 0.00 | 9, 10, 9 | 10, 10, 10 | 2.59 h (3 seed) |

Từng seed:

| Seed | R@1 bản gộp | R@1 mốc | MRR bản gộp | MRR mốc |
|---|---|---|---|---|
| 0 | 48.21 | 47.36 | 60.38 | 59.71 |
| 1 | 48.56 | 47.56 | 60.69 | 59.72 |
| 2 | 48.39 | 47.52 | 60.44 | 59.72 |

Chênh lệch so với mốc, kiểm định Welch với 3 seed mỗi bên:

| Chỉ số | Chênh | t | p |
|---|---|---|---|
| R@1 | +0,90 | 7,70 | 0,003 |
| R@5 | +0,56 | 4,82 | 0,018 |
| R@10 | +0,61 | 8,37 | 0,001 |
| MRR | +0,79 | 8,26 | 0,014 |

Seed thấp nhất của bản gộp (48,21) cao hơn seed cao nhất của mốc (47,56).

Diễn biến theo epoch (R@1 val):

- seed 0: 46,99 → 47,17 → 47,91 → **48,21** → 47,91 → 47,86, dừng ở epoch 6;
- seed 1: 46,57 → 47,18 → 47,40 → 48,43 → 48,26 → 48,44 → 48,01 → **48,56** → 48,44 → 48,49;
- seed 2: 46,61 → 47,39 → 47,46 → 47,79 → **48,39** → 48,14 → 47,95, dừng ở epoch 7.

Ba seed của 1a không dừng sớm lần nào, R@1 vẫn còn tăng ở epoch 10. Bản gộp vượt mức cuối của 1a ngay từ epoch 3–5,
rồi chững lại.

## Độ liền của graph (`graph_structure.json`, `scripts/improve/graph_structure_stats.py`)

Cách đếm: bỏ qua chiều cạnh, node cô lập tính là một mảnh. Số đầu là val, số trong ngoặc là train. Bỏ qua 30 (643)
ảnh không có vật thể nào.

| Graph ảnh | Node/ảnh | Mảnh rời/ảnh (trung vị) | % node trong mảnh lớn nhất | % ảnh đã liền | % node cô lập | Cạnh có nhãn / tự thêm mỗi ảnh |
|---|---|---|---|---|---|---|
| Cấp 1 (`factual_v1`) | 26,8 (27,1) | 11,0 (11,1), trung vị 10 | 44,6 (43,6) | 4,7 (3,7) | 22,1 (22,1) | 17,9 / 0 |
| Gộp trùng (`merge_v1`) | 24,9 (25,2) | 9,8 (9,9), trung vị 9 | 48,4 (47,4) | 5,0 (3,9) | 22,8 (22,9) | 17,4 / 0 |
| + kNN-5 + cầu nối | 24,9 (25,2) | 1,0 (1,0) | 100 (100) | 100 (100) | 0,01 | 17,4 / 114,3 |
| + node tổng (bản gộp) | 25,6 (25,9) | 1,0 | 100 | 100 | 0,06 | 17,4 / 114,3, cộng 2 cạnh tổng mỗi node |

- **Cầu nối:** chỉ 1,0% ảnh cần (trung bình 0,02 cầu mỗi ảnh). kNN-5 tự nó đã nối liền gần như mọi ảnh.
- **Node cô lập còn lại:** là node duy nhất của ảnh chỉ có 1 vật thể. Ở dòng có node tổng, đó là node tổng của ảnh
  không có vật thể nào.
- **Phía câu:** 94,6% graph câu val đã liền sẵn; trung bình 2,8 node và 1,06 mảnh. Có node tổng thì 100% liền.

## Gradient của GAT

Đo theo hai cách:

- **Probe** (`gradient_probe.json`, `scripts/improve/gradient_probe.py`): checkpoint tốt nhất, chế độ eval (không
  dropout), cùng 10 batch train × 256 câu cho mọi run. Báo mean qua batch, rồi mean ± std qua 3 seed.
- **Batch cuối:** gradient của batch cuối ở epoch tốt nhất, lấy từ log train như các report trước.

| Module | Probe, bản gộp | Probe, mốc 1a | Batch cuối, bản gộp | Batch cuối, mốc 1a |
|---|---|---|---|---|
| gat1 | 0.335 ± 0.045 | 0.136 ± 0.011 | 0.402 ± 0.052 | 0.179 ± 0.033 |
| gat2 | 0.256 ± 0.019 | 0.094 ± 0.004 | 0.321 ± 0.043 | 0.135 ± 0.022 |
| p_node | 0.476 ± 0.069 | 0.315 ± 0.025 | 0.575 ± 0.054 | 0.387 ± 0.073 |
| p_edge | 0.096 ± 0.015 | 0.047 ± 0.004 | 0.081 ± 0.016 | 0.053 ± 0.015 |
| triple_mlp | 0.282 ± 0.029 | 0.143 ± 0.011 | 0.257 ± 0.067 | 0.186 ± 0.045 |
| u | 0.193 ± 0.003 | 0.063 ± 0.003 | 0.231 ± 0.024 | 0.078 ± 0.009 |
| geometry_mlp | 0.105 ± 0.018 | — | 0.108 ± 0.016 | — |
| none_predicate | 0.047 ± 0.015 | — | 0.050 ± 0.021 | — |
| global_predicate | 0.037 ± 0.018 | — | 0.042 ± 0.017 | — |
| u_objects | 0.020 ± 0.002 | 0.026 ± 0.002 | 0.020 ± 0.002 | 0.024 ± 0.008 |
| u_triples | 0.0030 ± 0.0003 | 0.0018 ± 0.0002 | 0.0036 ± 0.0004 | 0.0025 ± 0.0002 |

Trên cùng các batch, GAT nhận gradient gấp khoảng 2,5 lần mốc (gat1) và 2,7 lần (gat2); `u`, đường đưa đầu ra GAT vào
vector node, nhận gấp 3 lần. Các vector học được `[none]`, `[global]` và MLP hình học đều nhận gradient khác 0.

## Độ dồn của vector bộ ba (`part_similarity.json`)

Đo trên 300 ảnh val: cosine trung bình giữa 3 000 cặp ngẫu nhiên.

| | Bộ ba | Vật thể |
|---|---|---|
| bản gộp (từng seed) | 0,827 / 0,916 / 0,865 | 0,402 / 0,251 / 0,349 |
| bản gộp (mean ± std) | **0,869 ± 0,045** | 0,334 ± 0,077 |
| mốc 1a | 0,961 ± 0,001 | 0,202 ± 0,041 |
| CLIP đóng băng ở step 0 (graph `merge_v1`) | 0,683 | 0,790 |

Vector bộ ba bớt dồn rõ: cosine giảm từ 0,96 xuống 0,87. Mức này ngang bản sàn z-score 1b (0,876), nhưng ở 1b thứ hạng
không tăng. Vector vật thể dồn hơn mốc một chút.

## Tương quan vật thể–bộ ba (`channel_correlation.json`)

Hệ số Pearson theo từng câu trên top-50 của câu đó, rồi lấy trung bình trên các câu val.

| | vật thể ~ bộ ba | CLIP ~ bộ ba | CLIP ~ vật thể |
|---|---|---|---|
| bản gộp | **0,323 ± 0,015** | 0,202 ± 0,015 | 0,326 ± 0,006 |
| mốc 1a | 0,206 ± 0,004 | 0,134 ± 0,006 | 0,325 ± 0,002 |

Kênh bộ ba giờ giống kênh vật thể hơn: tương quan tăng từ 0,21 lên 0,32. Nhưng vẫn thấp xa các bản A factorized (0,59)
và B graph-channel (0,52), là những bản đã không giúp được.

## Chẩn đoán đổi động từ / đổi chủ–tân (`triple_perturbation.json`)

Hai phép nhiễu chỉ áp lên phía câu: a là đổi động từ, b là đổi chủ ngữ với tân ngữ. Δ là chênh so với điều kiện giữ
nguyên.

| Mô hình | Giữ nguyên | a. đổi động từ (Δ) | b. đổi chủ–tân (Δ) |
|---|---|---|---|
| bản gộp, điểm cuối, toàn val | 48.38 ± 0.17 | 47.36 ± 0.18 (−1,02) | 48.00 ± 0.40 (−0,38) |
| mốc 1a, điểm cuối, toàn val | 47.48 ± 0.10 | 46.56 ± 0.10 (−0,92) | 46.62 ± 0.16 (−0,86) |
| bản gộp, chỉ kênh bộ ba, câu có ≥1 bộ ba | **13.14 ± 1.25** | 9.02 ± 0.98 (−4,12) | 10.57 ± 0.76 (−2,57) |
| mốc 1a, chỉ kênh bộ ba, câu có ≥1 bộ ba | 7.22 ± 0.34 | 5.04 ± 0.26 (−2,18) | 5.00 ± 0.40 (−2,22) |

Bảng đầy đủ R@1/5/10/MRR cho cả 4 góc nhìn ở `summary.md`.

- **Kênh bộ ba đứng riêng** mạnh gần gấp đôi mốc (R@1 13,14 so với 7,22). Nó vẫn đọc vị từ và chiều: đổi động từ làm
  mất 4,1 điểm, đổi chủ–tân mất 2,6 điểm. Nhưng vẫn kém cụm CLIP đóng băng ở step 0 của đồ thị câu sng (18,84; xem
  `../README.md`).
- **Ở điểm cuối**, đổi động từ làm mất 1,02 R@1, ngang mốc (0,92). Đổi chủ–tân chỉ làm mất 0,38 ± 0,40, so với 0,86
  của mốc. Như vậy điểm cuối của bản gộp ít phụ thuộc vào chiều của bộ ba hơn mốc, dù kênh bộ ba đứng riêng nhạy chiều
  hơn.
  - Một khả năng là ở epoch tốt nhất, trọng số c của bản gộp thấp hơn mốc (a/b/c = 0,618 / 0,135 / 0,247, so với
    0,467 / 0,078 / 0,455).
  - Nhưng ở chế độ trộn điểm thô, trọng số còn phụ thuộc thang điểm của từng kênh, nên không so trực tiếp được. Mình
    chưa kiểm chứng giải thích này.

## Attention của GAT (`attention.json`, `attention_examples.md`, `scripts/improve/gat_attention.py`)

Cách đo: lấy toàn bộ 2 126 ảnh val, ghi alpha của GATv2, lấy trung bình 4 head. Alpha là softmax trên các cạnh đi vào
một node, kể cả self-loop. Chỉ xét node nhận là node thật, không xét node tổng. Báo mean ± std qua 3 seed.

- **Phần khối lượng:** tổng alpha mà một node nhận từ mỗi loại cạnh.
- **Lift:** alpha × bậc vào. Lift bằng 1 nghĩa là loại cạnh đó được chú ý đúng bằng mức chia đều.

Một node ảnh điển hình nhận 5 cạnh kNN, 1 cạnh từ node tổng, 1 self-loop, và trung bình 0,7 cạnh có nhãn.

| Lớp | Loại cạnh | Phần khối lượng | Lift |
|---|---|---|---|
| 1 | self-loop | 0.420 ± 0.046 | 3.10 ± 0.34 |
| 1 | node tổng | 0.331 ± 0.094 | 2.44 ± 0.69 |
| 1 | tự thêm (`[none]`) | 0.194 ± 0.038 | 0.30 ± 0.06 |
| 1 | có nhãn | 0.055 ± 0.016 | 0.62 ± 0.18 |
| 2 | self-loop | 0.474 ± 0.017 | 3.47 ± 0.13 |
| 2 | tự thêm (`[none]`) | 0.341 ± 0.043 | 0.53 ± 0.07 |
| 2 | node tổng | 0.152 ± 0.046 | 1.13 ± 0.34 |
| 2 | có nhãn | 0.033 ± 0.002 | 0.40 ± 0.03 |

Theo cặp nhãn nguồn → nhận, chỉ xét cặp gặp ≥ 30 lần (seed 0, 523 cặp):

- **Lớp 1, lift cao nhất:** bus → window 1,18; giraffe → spot 1,17; train → window 1,14; man → hand 1,12;
  train → door 1,03; man → head 1,01; man → arm 1,01.
- **Lớp 1, lift thấp nhất:** ear → cow 0,01; spot → giraffe 0,02; eye → whisker 0,02; head → cow 0,03;
  leg → horse 0,03.
- **Lớp 2, lift cao nhất:** street → road 1,13; road → street 1,10; beach → beach 1,03; ground → road 1,01;
  sand → beach 0,96; table → plate 0,91.

Cách đọc:

- **Lớp 1:** node "bộ phận" (cửa sổ, đốm, tay) nhận nhiều từ vật thể chứa nó (xe buýt, hươu, người). Chiều ngược lại
  gần như bị bỏ qua: con bò hầu như không nghe cái tai.
- **Lớp 2:** các node nền và bề mặt trao đổi với nhau (đường, cát, bãi biển).
- **Node tổng** được chú ý nhiều ở lớp 1 (lift 2,4), ít hơn ở lớp 2 (1,1).
- **Cạnh có nhãn** nhận ít hơn mức chia đều ở cả hai lớp. Riêng ở lớp 1, mỗi cạnh có nhãn vẫn được chú ý gấp đôi một
  cạnh kNN (lift 0,62 so với 0,30).

`attention_examples.md` có 12 ảnh val, chọn ngẫu nhiên với seed 0: 8 câu mô hình xếp ảnh đúng hạng 1, 4 câu xếp hạng
2–10. Mỗi ảnh vẽ 6 cạnh tự thêm có alpha lớp 1 cao nhất. Ví dụ:

- ảnh `455624_3`: man → head, man → person (alpha khoảng 0,22–0,24);
- ảnh `442809_2`: các cạnh cupcake → cupcake gần như bị bỏ qua (alpha khoảng 0,01).

Lưu ý khi dùng phần này cho mục giải thích: attention cho biết láng giềng nào tạo nên vector của một node. Nó không
cho biết cạnh đó làm điểm xếp hạng thay đổi bao nhiêu, vì điểm đến từ kênh vật thể và kênh bộ ba. Muốn dùng attention
làm lời giải thích "trung thành" với cơ chế, cần kiểm thêm bằng cách che cạnh: bỏ các cạnh tự thêm có alpha cao rồi
đo điểm của ảnh đúng thay đổi bao nhiêu. Việc này chưa làm.

## Nhận xét

1. **Bản gộp tốt hơn mốc 1a rõ ràng trên val.** R@1 +0,90 (48,38 ± 0,17 so với 47,48 ± 0,10, Welch p = 0,003), MRR
   +0,79, R@5 +0,56, R@10 +0,61. Mọi seed của bản gộp cao hơn mọi seed của mốc. Đây là kết quả val tốt nhất trong
   phase cải tiến, nhưng cái giá là thời gian train gấp khoảng 2,2 lần mỗi epoch.
2. **Chưa biết phần nào có tác dụng,** vì 4 thay đổi được gộp làm một. Các chẩn đoán chỉ cho thấy những gì đi kèm với
   kết quả tốt hơn:
   - GAT nhận gradient gấp 2,5 lần mốc;
   - vector bộ ba bớt dồn (0,96 → 0,87);
   - kênh bộ ba đứng riêng mạnh gần gấp đôi (7,2 → 13,1);
   - kênh bộ ba giống kênh vật thể hơn (tương quan 0,21 → 0,32).

   1b cũng giảm độ dồn xuống 0,88 mà không tăng thứ hạng, nên bớt dồn tự nó không đủ giải thích.
3. **Câu hỏi "GAT có tác dụng thật không" vẫn còn mở.** Đối chứng tắt GAT trên graph liền đã bị bỏ khi đổi kế hoạch.
   Gradient và attention cho thấy GAT được dùng nhiều hơn, nhưng không chứng minh được nó đóng góp vào thứ hạng.
4. **Dừng sớm:** hai trên ba seed dừng sớm (đỉnh ở epoch 4 và 5). Trọng số kênh bộ ba c vẫn tăng đều tới epoch cuối
   (seed 1: 0,06 → 0,37), nên quy tắc dừng sau 2 epoch không tiến có thể đã cắt hơi sớm. Đây là quy tắc chung của mọi
   run, mình không đổi.

## Việc tiếp theo (Lộc quyết)

- **Ablation**, vì bản gộp đã tốt hơn mốc rõ ràng. Thứ tự gợi ý:
  - (a) bản gộp tắt GAT (`--no-use-gat`): trả lời câu hỏi 3; nhanh vì không có GAT;
  - (b) bỏ từng phần một khỏi bản gộp: không node tổng, không kNN, không hình học, không gộp trùng; mỗi phần 3 seed.

  Bỏ kNN thì số cạnh về như cũ, nên những bản đó nhanh gần bằng 1a, khoảng 0,9 giờ mỗi seed. Các bản còn kNN tốn
  khoảng 1,2–1,9 giờ mỗi seed.
- **Chấm test lần cuối,** nếu chọn bản gộp: cần đóng gói split test theo `merge_v1` trước, gồm hộp, synset và gộp
  trùng. Việc này không động gì tới tham số.

## File

- Run: `fullgraph_factual_r3_seed{0,1,2}/`, gồm `config.json` (cờ, seed, commit `9962f52`), `log.jsonl`,
  `metrics.json`. Checkpoint `*.pt` không commit.
- Chẩn đoán: `summary.md`, `summary.json`, `triple_perturbation.json`, `part_similarity.json`,
  `channel_correlation.json`, `gradient_probe.json`, `graph_structure.json`, `attention.json`,
  `attention_examples.md`, `img/`.
- Log: `logs/` (`timeline.log`, stdout từng run, log từng bước chẩn đoán, `package_merged.log`);
  `memory_probe.jsonl`.
