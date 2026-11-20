# Giải thích cấp 3 so với đáp án biết trước: ba benchmark

Validity ở v2 và v3 (`experiments/xai-metrics-v3/`) chấm bằng WordNet: node ảnh được chú ý có cùng loại với từ trong câu không.
Phép chấm đó không biết phần nào của ảnh *thực sự* là lý do ảnh đứng hạng 1.
Ở đây dùng ba bộ dữ liệu mà đáp án (các phần đúng của ảnh) có sẵn trước khi chạy bất kỳ cách giải thích nào, không ai phải gán tay, và chấm cả 16 cách giải thích của v3.

- **Script:** `scripts/xai_ground_truth.py` (ba benchmark), `src/explain/synthetic_graphs.py` (dữ liệu tổng hợp), `scripts/summarize_xai_ground_truth.py` (mean ± std qua seed, `summary.json`, bảng ở `logs/tables.md`); `run.sh` chạy lại tất cả, log ở `logs/run.log`.
- **Mô hình thật:** `countonly_factual_r3` và `main_r3`, 3 seed, `checkpoint_best.pt`, như v3; chỉ dùng val.
- **Cách giải thích:** đúng 16 cách và các tham số của `scripts/xai_metrics.py` v3; PGExplainer và GraphMask học lại cho mỗi checkpoint trên 200 câu val nằm ngoài các câu được chấm.
- **Chỉ giải thích ảnh hạng 1, và chỉ giữ ca mà ảnh hạng 1 là ảnh có đáp án.**
  Mỗi mô hình, mỗi seed, 100 ca.

## Độ đo

Mỗi cách giải thích xếp các phần của ảnh (node vật thể và bộ ba có nhãn); g là số phần của đáp án.

- **GEA** (graph explanation accuracy, Agarwal và cộng sự, GraphXAI, Scientific Data 2023): Jaccard giữa g phần đầu của thứ tự và đáp án.
- **Precision@g:** tỉ lệ phần đúng trong g phần đầu.
- **AUROC:** xác suất một phần đúng đứng trước một phần không thuộc đáp án.
- **Vị trí phần đúng đầu tiên** (1 là tốt nhất).
- Riêng benchmark C: **phân vị của mồi nhử**, vị trí trung bình của các phần mồi nhử chia cho số phần (cao là tốt, mồi nhử bị xếp cuối).

Mốc: random (hoán vị đều) và lowest (E đảo ngược).

## Ba benchmark

### B. Dữ liệu tổng hợp kiểu GraphXAI (`synthetic/`): đo độ trung thực

GraphXAI cài một motif quyết định nhãn, nên motif là đáp án đúng; ở đây làm tương tự cho bài toán xếp hạng.

- Câu là một motif nhỏ: 2 vật và 1 quan hệ, hoặc chuỗi 3 vật và 2 quan hệ, tên vật và vị từ lấy từ 40 tên và 8 vị từ VG phổ biến (vector CLIP thật).
- Ảnh vàng là một cảnh ngẫu nhiên có motif, 5–11 vật nền, 3–8 quan hệ nền và 1–2 quan hệ nối motif với nền.
- 19 ảnh âm, mỗi ảnh thiếu **đúng một** phần của motif: đổi vị từ, đổi tên một vật (và nền không có tên nào của câu), hoặc bỏ một quan hệ.
  Mỗi vật và mỗi quan hệ của motif bị phá ở ít nhất một ảnh âm của mỗi câu, nên mô hình cần **mọi** phần của đáp án để thắng (tránh lỗi "mô hình không dùng đáp án", Faber và cộng sự, KDD 2021).
- Điểm CLIP là một hằng số, nên chỉ đồ thị xếp được hạng; thứ tự vật, thứ tự quan hệ và box là ngẫu nhiên.
- Đáp án: các node vật của motif và bộ ba có nhãn của motif (TB 4,0 phần trên 18,6 phần của ảnh).
- Mô hình: cùng kiến trúc và cờ với countonly (`count_table`, cạnh hoàn thiện kNN 5, node toàn cục, fusion raw), học lại trên 6 000 câu tổng hợp, 8 epoch, 3 seed (`synthetic/training.json`).
  R@1 trên 2 400 câu val tổng hợp: 100% ở cả 3 seed (epoch tốt nhất 2, 3, 2; khi nhiều epoch cùng R@1, lấy epoch sớm nhất).
  Câu được chấm là val 0..1999; PGExplainer và GraphMask học trên val 2000..2199.
- Tên và vị từ của dữ liệu tổng hợp có thể chưa có trong bảng vector; chúng được mã hoá bằng CLIP đóng băng của cấu hình, cache ở `data/features/synthetic_xai/` (không đưa vào git).

### C. Cài phần đúng vào ảnh thật (`planted/`): độ trung thực trên mô hình thật

- Với một câu val có quan hệ (chủ thể, vị từ, đối tượng) đủ tên, cài quan hệ đó vào một ứng viên đang ở hạng 2–5: hai node mới mang tên của câu và vị từ của câu giữa chúng.
- Cùng lúc cài một **mồi nhử** cùng dạng, với tên và vị từ không có trong câu (tên từ lamp, fence, cloud, bench, sign, pole, window, plant, curtain, rock; vị từ từ near, behind, next to, above, under).
- Node mới lấy box của một node ngẫu nhiên của ảnh và nhóm số lượng theo số node cùng tên; ảnh không có vật nào thì box là cả ảnh.
  Node và quan hệ mới được xáo vào vị trí ngẫu nhiên trong danh sách của ảnh: nhiều cách giải thích sắp xếp ổn định, nên phần luôn đứng cuối sẽ thua mọi lần hoà điểm.
- Bỏ quan hệ có đầu là đại từ (who, those, which, ...), vì ảnh không có vật mang tên đó.
  Văn bản phần mới chưa có trong bảng được mã hoá bằng CLIP đóng băng của cấu hình, như demo làm với truy vấn gõ tay.
- **Chỉ giữ ca mà ảnh sau khi cài lên hạng 1**, nên mô hình cần chính các phần đã cài (bỏ chúng thì ảnh về đồ thị gốc, hạng cũ ≥ 2).
  Để có 100 ca, cần thử 233–248 câu (countonly) và 212–238 câu (`main_r3`).
- Đáp án: 3 phần cài vào (2 node, 1 bộ ba); 3 phần mồi nhử chấm riêng.

### A. Region graph của Visual Genome (`vg-regions/`): độ hợp lý

- Visual Genome có mô tả vùng ("region description") kèm region graph: các object và quan hệ của ảnh mà người gán nhãn nói tới trong mô tả đó.
  Tải bằng `python scripts/download_data.py --regions` (`region_graphs.json.zip`, 332 MB).
- Object id của VG được nối tới node của ảnh qua đồ thị cấp 1 dựng lại với id nguồn (`build_scene_graph(..., include_source_ids=True)`), rồi qua `members` của merge_v1.
- Truy vấn: chính region graph (tên và vị từ chuẩn, không qua parser, nên lời giải thích sai không thể do parse sai); vector câu là CLIP của mô tả; ứng viên là CLIP top-50 trong 2 126 ảnh val.
- `vg-regions/prepare.json`: 105 434 vùng; giữ 26 249 truy vấn sau khi bỏ vùng có object không nối được (10 133), không có bộ ba (67 268) hoặc trùng (1 784); 14 951 truy vấn có ảnh đúng trong top-50.
  Ảnh có đáp án ở hạng 1 ở 39,5 ± 1,1% truy vấn (countonly) và 32,4 ± 0,6% (`main_r3`) trong số đã thử.
- Hai đáp án: **chính** (đúng các node được gán, và bộ ba giữa chúng với vị từ của vùng; TB 3,4 phần) và **theo nhãn** (mọi node trùng tên với một node được gán, vì mô hình khớp theo nhãn và không phân biệt được hai vật cùng tên).
- **Đây là độ hợp lý (plausibility), không phải độ trung thực:** đáp án là thứ người gán nhãn nói tới, mô hình có thể dùng phần khác của ảnh mà vẫn xếp đúng.

## Kết quả (mean qua 3 seed; std và các độ đo khác ở phụ lục)

**GEA** (cao là tốt; A: đáp án chính):

| Cách | B. tổng hợp | C. countonly | C. `main_r3` | A. countonly | A. `main_r3` |
|---|---|---|---|---|---|
| occlusion | **1,000** | **0,654** | **0,606** | **0,771** | **0,758** |
| GraphSVX | 0,333 | 0,466 | 0,396 | 0,620 | 0,512 |
| Shapley | 0,514 | 0,445 | 0,389 | 0,592 | 0,505 |
| SubgraphX | 0,839 | 0,465 | 0,020 | 0,453 | 0,594 |
| IG | 0,802 | 0,388 | 0,396 | 0,539 | 0,433 |
| GNNExplainer | 0,772 | 0,434 | 0,386 | 0,543 | 0,427 |
| **E** (`model`) | 0,665 | 0,416 | 0,250 | 0,595 | 0,281 |
| GStarX | 0,283 | 0,372 | 0,288 | 0,393 | 0,422 |
| CF-GNNExplainer | 0,699 | 0,356 | 0,244 | 0,396 | 0,225 |
| clip | 0,878 | 0,376 | 0,356 | 0,386 | 0,378 |
| GNN-LRP | 0,550 | 0,255 | 0,116 | 0,357 | 0,263 |
| PGExplainer | 0,511 | 0,143 | 0,150 | 0,193 | 0,183 |
| gradient | 0,423 | 0,062 | 0,051 | 0,136 | 0,082 |
| GraphMask | 0,171 | 0,060 | 0,101 | 0,070 | 0,108 |
| random | 0,143 | 0,049 | 0,052 | 0,050 | 0,057 |
| lowest | 0,000 | 0,001 | 0,003 | 0,000 | 0,002 |

**AUROC của E:** B 0,956; C 0,896 (countonly) và 0,766 (`main_r3`); A 0,938 và 0,746.

Đọc bảng:

- **Occlusion đứng đầu ở cả năm cột**, và đạt tuyệt đối trên dữ liệu tổng hợp (GEA 1,000 ở cả 3 seed).
  Đây cũng là cách có phản thực tế tham lam tốt nhất ở v3: với mô hình này, bỏ từng phần rồi chấm lại là lời giải thích đúng nhất.
- **E tốt với countonly, kém với `main_r3`, nhất quán qua ba benchmark.**
  Với countonly, E đứng ngang Shapley trên region graph (0,595 so với 0,592) và trên dữ liệu tổng hợp đặt phần đúng lên đầu ở mọi ca (vị trí phần đúng đầu tiên 1,00, AUROC 0,956).
  Với `main_r3`, E chỉ đạt 0,25–0,28 ở A và C, thấp hơn IG, GNNExplainer và cả clip; đúng như v2, các bộ ba chen lên đầu E mà không mang điểm.
- **E thay đổi nhiều giữa ba mô hình tổng hợp** (0,665 ± 0,135), trong khi occlusion đạt 1,000 ở cả ba.
- **Shapley và GraphSVX tốt trên ảnh thật, kém trên dữ liệu tổng hợp** (0,51 và 0,33).
  Hàm giá trị của chúng là biên hạng 1 khi chỉ giữ một liên minh, trong khi 19 ảnh âm giữ nguyên đồ thị; một lý do có thể, chưa kiểm riêng, là khi motif chưa đủ thì biên âm như nhau với mọi liên minh, nên giá trị dồn vào các tương tác bậc cao.
- **SubgraphX phụ thuộc vào tính liên thông.**
  Nó giữ thành phần liên thông lớn nhất khi tỉa, nên với `main_r3`, nơi đồ thị không có cạnh hoàn thiện và cụm phần cài vào tách rời khỏi phần còn lại của ảnh (đã kiểm), nó thường tỉa cụm đó sớm: GEA 0,020, AUROC 0,278, phần đúng đầu tiên ở vị trí 28,8 trên khoảng 50.
  Với countonly và mô hình tổng hợp, cạnh hoàn thiện nối các phần lại; SubgraphX ngang GraphSVX ở C (countonly, 0,465 và 0,466) và đứng thứ ba ở B.
- **clip rất mạnh trên dữ liệu tổng hợp** (0,878) vì ở đó phần đúng chính là phần mang tên trong câu, và xếp mồi nhử về cuối tốt hơn mọi cách khác ở C (phân vị 0,81 ở cả hai mô hình), nhưng chỉ ở mức trung bình trên ảnh thật: clip không biết mô hình dùng gì.
- **GraphMask gần mức ngẫu nhiên ở cả năm cột**, khớp với v3.
  Trên dữ liệu tổng hợp và với `main_r3`, nó đóng gần hết cổng mà vẫn thoả ràng buộc (điểm đổi 0,014–0,017 hiệu ứng trên dữ liệu tổng hợp, 0,018–0,044 với `main_r3`), tức mô hình không cần message; với countonly nó giữ 83–99% cổng mở.
  Cả hai cách đều để các xác suất cổng gần bằng nhau, nên không có gì để GraphMask phân biệt các phần.
- **GStarX và CF-GNNExplainer ở mức giữa**, không lần nào vượt occlusion hay IG.
- **Với đáp án theo nhãn (A)**, IG, GraphSVX và Shapley tăng nhiều nhất (+0,09 đến +0,14 GEA) và tới gần occlusion, còn occlusion giảm nhẹ (−0,02 đến −0,04): các cách này hay xếp cao cả những node cùng tên mà người gán nhãn không chọn.

## Kết luận

1. **Bỏ từng phần rồi chấm lại (occlusion) là cách giải thích đúng nhất** trên cả ba benchmark, kể cả dữ liệu tổng hợp nơi đáp án là nguyên nhân thật.
2. **E của countonly đáng tin:** gần mức Shapley ở ảnh thật và luôn đặt một phần đúng lên đầu ở dữ liệu tổng hợp.
   E của `main_r3` kém hơn rõ, như v2 và v3 đã thấy bằng các độ đo khác.
3. **Ngoài occlusion, không cách nào ổn định trên mọi bộ:** SubgraphX hỏng khi đáp án không liên thông với phần còn lại, Shapley và GraphSVX kém trên dữ liệu tổng hợp, clip chỉ tốt khi đáp án trùng tên trong câu.
   Vì vậy nhóm chấm trên cả ba bộ thay vì một.

## Hạn chế

- Mỗi mô hình 100 ca mỗi seed; ca bị chọn theo điều kiện ảnh có đáp án ở hạng 1, nên là các ca dễ hơn trung bình.
- C cài phần mới với box sao chép từ node có sẵn; cụm cài vào có thể dễ nhận ra hơn một phần tự nhiên.
- A là độ hợp lý: một cách giải thích trung thực vẫn có thể lệch khỏi đáp án của người.
- B dùng từ vựng nhỏ (40 tên, 8 vị từ) và mô hình học lại trên dữ liệu đó, không phải chính checkpoint của nhóm.
- Mọi kết quả ở đây chạy lại từ đầu bằng `run.sh` sau khi sửa cách học GraphMask (commit 792c851, xem `experiments/xai-metrics-v3/REPORT.md`); so với lần chạy trước, chỉ số của GraphMask và thời gian đổi; dữ liệu tổng hợp sinh lại được từ seed 2031, còn các checkpoint tổng hợp (`synthetic/models/*.pt`, 12 MB) không đưa vào git, `synthetic train` dựng lại chúng.

Phụ lục dưới là bảng đầy đủ của `scripts/summarize_xai_ground_truth.py` (cũng ở `logs/tables.md`).

## C. Cài phần đúng vào ảnh thật (`planted`)

### countonly_factual_r3

ca theo seed: 100, 100, 100; cỡ đáp án 3,0 ± 0,0; số phần 45,6 ± 0,9.

**đáp án chính.**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên | Phân vị của mồi nhử (cao tốt) |
| --- | --- | --- | --- | --- | --- |
| model | 0,416 ± 0,057 | 0,543 ± 0,058 | 0,896 ± 0,033 | 1,57 ± 0,37 | 0,679 ± 0,003 |
| random | 0,049 ± 0,003 | 0,080 ± 0,006 | 0,511 ± 0,004 | 10,96 ± 0,66 | 0,492 ± 0,020 |
| lowest | 0,001 ± 0,001 | 0,002 ± 0,002 | 0,104 ± 0,033 | 34,34 ± 2,37 | 0,321 ± 0,003 |
| clip | 0,376 ± 0,026 | 0,498 ± 0,027 | 0,897 ± 0,006 | 2,08 ± 0,11 | 0,815 ± 0,006 |
| gradient | 0,062 ± 0,040 | 0,100 ± 0,061 | 0,306 ± 0,117 | 17,81 ± 6,80 | 0,348 ± 0,009 |
| integrated_gradients | 0,388 ± 0,032 | 0,520 ± 0,039 | 0,861 ± 0,033 | 1,59 ± 0,17 | 0,531 ± 0,017 |
| occlusion | 0,654 ± 0,055 | 0,729 ± 0,047 | 0,912 ± 0,026 | 1,50 ± 0,30 | 0,556 ± 0,037 |
| gnnexplainer | 0,434 ± 0,034 | 0,560 ± 0,033 | 0,825 ± 0,014 | 1,66 ± 0,12 | 0,599 ± 0,019 |
| pgexplainer | 0,143 ± 0,013 | 0,227 ± 0,017 | 0,661 ± 0,014 | 4,17 ± 0,55 | 0,547 ± 0,013 |
| gnn_lrp | 0,255 ± 0,019 | 0,368 ± 0,025 | 0,696 ± 0,015 | 3,13 ± 0,24 | 0,543 ± 0,016 |
| graphsvx | 0,466 ± 0,020 | 0,597 ± 0,018 | 0,941 ± 0,002 | 1,38 ± 0,09 | 0,681 ± 0,012 |
| subgraphx | 0,465 ± 0,008 | 0,520 ± 0,009 | 0,725 ± 0,010 | 5,06 ± 0,22 | 0,519 ± 0,019 |
| shapley | 0,445 ± 0,025 | 0,577 ± 0,030 | 0,937 ± 0,002 | 1,40 ± 0,11 | 0,664 ± 0,007 |
| cf_gnnexplainer | 0,356 ± 0,034 | 0,487 ± 0,038 | 0,784 ± 0,019 | 1,98 ± 0,30 | 0,572 ± 0,025 |
| graphmask | 0,060 ± 0,048 | 0,088 ± 0,064 | 0,495 ± 0,049 | 12,60 ± 2,12 | 0,468 ± 0,040 |
| gstarx | 0,372 ± 0,029 | 0,493 ± 0,032 | 0,880 ± 0,008 | 2,09 ± 0,11 | 0,663 ± 0,006 |

### main_r3

ca theo seed: 100, 100, 100; cỡ đáp án 3,0 ± 0,0; số phần 49,7 ± 2,4.

**đáp án chính.**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên | Phân vị của mồi nhử (cao tốt) |
| --- | --- | --- | --- | --- | --- |
| model | 0,250 ± 0,010 | 0,376 ± 0,008 | 0,766 ± 0,006 | 1,83 ± 0,10 | 0,623 ± 0,016 |
| random | 0,052 ± 0,008 | 0,084 ± 0,013 | 0,504 ± 0,004 | 11,85 ± 1,08 | 0,510 ± 0,024 |
| lowest | 0,003 ± 0,001 | 0,004 ± 0,002 | 0,234 ± 0,006 | 29,09 ± 0,91 | 0,377 ± 0,016 |
| clip | 0,356 ± 0,011 | 0,483 ± 0,012 | 0,882 ± 0,004 | 2,48 ± 0,10 | 0,812 ± 0,008 |
| gradient | 0,051 ± 0,011 | 0,079 ± 0,013 | 0,540 ± 0,019 | 12,44 ± 0,30 | 0,542 ± 0,014 |
| integrated_gradients | 0,396 ± 0,021 | 0,546 ± 0,018 | 0,857 ± 0,012 | 1,39 ± 0,02 | 0,595 ± 0,013 |
| occlusion | 0,606 ± 0,002 | 0,711 ± 0,004 | 0,970 ± 0,003 | 1,26 ± 0,01 | 0,649 ± 0,040 |
| gnnexplainer | 0,386 ± 0,005 | 0,531 ± 0,007 | 0,834 ± 0,007 | 1,55 ± 0,08 | 0,574 ± 0,004 |
| pgexplainer | 0,150 ± 0,004 | 0,234 ± 0,004 | 0,649 ± 0,011 | 4,79 ± 0,27 | 0,563 ± 0,021 |
| gnn_lrp | 0,116 ± 0,016 | 0,183 ± 0,023 | 0,649 ± 0,013 | 4,43 ± 0,35 | 0,499 ± 0,014 |
| graphsvx | 0,396 ± 0,032 | 0,529 ± 0,028 | 0,928 ± 0,008 | 1,51 ± 0,09 | 0,650 ± 0,035 |
| subgraphx | 0,020 ± 0,010 | 0,020 ± 0,010 | 0,278 ± 0,004 | 28,77 ± 1,52 | 0,711 ± 0,012 |
| shapley | 0,389 ± 0,020 | 0,520 ± 0,017 | 0,930 ± 0,006 | 1,54 ± 0,09 | 0,643 ± 0,031 |
| cf_gnnexplainer | 0,244 ± 0,020 | 0,357 ± 0,026 | 0,591 ± 0,028 | 5,13 ± 0,94 | 0,501 ± 0,032 |
| graphmask | 0,101 ± 0,024 | 0,141 ± 0,034 | 0,544 ± 0,036 | 14,34 ± 3,43 | 0,606 ± 0,093 |
| gstarx | 0,288 ± 0,004 | 0,404 ± 0,004 | 0,842 ± 0,008 | 2,77 ± 0,23 | 0,610 ± 0,023 |

## A. Region graph Visual Genome (`vg-regions`)

### countonly_factual_r3

ca theo seed: 100, 100, 100; cỡ đáp án 3,4 ± 0,0; số phần 45,4 ± 0,9; ảnh có đáp án ở hạng 1: 39,5 ± 1,1% truy vấn.

**đáp án chính.**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên |
| --- | --- | --- | --- | --- |
| model | 0,595 ± 0,102 | 0,698 ± 0,078 | 0,938 ± 0,018 | 1,39 ± 0,05 |
| random | 0,050 ± 0,001 | 0,081 ± 0,003 | 0,490 ± 0,014 | 11,33 ± 0,50 |
| lowest | 0,000 ± 0,000 | 0,000 ± 0,000 | 0,062 ± 0,018 | 37,14 ± 1,40 |
| clip | 0,386 ± 0,011 | 0,506 ± 0,014 | 0,880 ± 0,005 | 1,66 ± 0,14 |
| gradient | 0,136 ± 0,068 | 0,206 ± 0,089 | 0,398 ± 0,112 | 11,18 ± 4,34 |
| integrated_gradients | 0,539 ± 0,042 | 0,659 ± 0,031 | 0,905 ± 0,028 | 1,31 ± 0,07 |
| occlusion | 0,771 ± 0,038 | 0,822 ± 0,028 | 0,941 ± 0,008 | 1,74 ± 0,25 |
| gnnexplainer | 0,543 ± 0,028 | 0,653 ± 0,020 | 0,839 ± 0,005 | 1,83 ± 0,11 |
| pgexplainer | 0,193 ± 0,001 | 0,293 ± 0,002 | 0,666 ± 0,011 | 3,76 ± 0,16 |
| gnn_lrp | 0,357 ± 0,036 | 0,489 ± 0,038 | 0,714 ± 0,036 | 2,07 ± 0,19 |
| graphsvx | 0,620 ± 0,032 | 0,717 ± 0,023 | 0,949 ± 0,002 | 1,49 ± 0,29 |
| subgraphx | 0,453 ± 0,027 | 0,526 ± 0,023 | 0,707 ± 0,018 | 4,15 ± 0,47 |
| shapley | 0,592 ± 0,017 | 0,693 ± 0,011 | 0,952 ± 0,007 | 1,30 ± 0,18 |
| cf_gnnexplainer | 0,396 ± 0,048 | 0,518 ± 0,046 | 0,785 ± 0,036 | 2,72 ± 0,75 |
| graphmask | 0,070 ± 0,042 | 0,110 ± 0,064 | 0,551 ± 0,027 | 10,66 ± 1,57 |
| gstarx | 0,393 ± 0,056 | 0,510 ± 0,053 | 0,848 ± 0,033 | 2,06 ± 0,31 |

**đáp án theo nhãn (mọi node cùng tên).**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên |
| --- | --- | --- | --- | --- |
| model | 0,603 ± 0,133 | 0,703 ± 0,108 | 0,920 ± 0,029 | 1,32 ± 0,03 |
| random | 0,076 ± 0,005 | 0,124 ± 0,010 | 0,488 ± 0,005 | 9,01 ± 0,29 |
| lowest | 0,008 ± 0,002 | 0,011 ± 0,003 | 0,080 ± 0,029 | 34,45 ± 1,64 |
| clip | 0,431 ± 0,012 | 0,552 ± 0,010 | 0,880 ± 0,002 | 1,61 ± 0,09 |
| gradient | 0,188 ± 0,062 | 0,275 ± 0,079 | 0,433 ± 0,096 | 8,92 ± 3,56 |
| integrated_gradients | 0,671 ± 0,029 | 0,777 ± 0,020 | 0,931 ± 0,022 | 1,03 ± 0,01 |
| occlusion | 0,730 ± 0,046 | 0,810 ± 0,035 | 0,907 ± 0,017 | 1,03 ± 0,01 |
| gnnexplainer | 0,585 ± 0,024 | 0,702 ± 0,018 | 0,837 ± 0,001 | 1,16 ± 0,08 |
| pgexplainer | 0,274 ± 0,001 | 0,389 ± 0,005 | 0,718 ± 0,008 | 2,92 ± 0,06 |
| gnn_lrp | 0,450 ± 0,051 | 0,589 ± 0,052 | 0,759 ± 0,022 | 1,64 ± 0,19 |
| graphsvx | 0,726 ± 0,039 | 0,816 ± 0,028 | 0,960 ± 0,005 | 1,06 ± 0,01 |
| subgraphx | 0,440 ± 0,036 | 0,537 ± 0,031 | 0,716 ± 0,019 | 2,66 ± 0,44 |
| shapley | 0,684 ± 0,042 | 0,785 ± 0,033 | 0,961 ± 0,010 | 1,05 ± 0,01 |
| cf_gnnexplainer | 0,431 ± 0,040 | 0,557 ± 0,037 | 0,770 ± 0,029 | 2,07 ± 0,60 |
| graphmask | 0,090 ± 0,028 | 0,142 ± 0,048 | 0,538 ± 0,018 | 8,58 ± 1,77 |
| gstarx | 0,463 ± 0,062 | 0,581 ± 0,058 | 0,854 ± 0,035 | 1,53 ± 0,22 |

### main_r3

ca theo seed: 100, 100, 100; cỡ đáp án 3,4 ± 0,0; số phần 48,1 ± 0,8; ảnh có đáp án ở hạng 1: 32,4 ± 0,6% truy vấn.

**đáp án chính.**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên |
| --- | --- | --- | --- | --- |
| model | 0,281 ± 0,025 | 0,414 ± 0,025 | 0,746 ± 0,013 | 1,17 ± 0,03 |
| random | 0,057 ± 0,008 | 0,092 ± 0,011 | 0,509 ± 0,006 | 11,84 ± 0,76 |
| lowest | 0,002 ± 0,000 | 0,003 ± 0,000 | 0,254 ± 0,013 | 25,35 ± 0,38 |
| clip | 0,378 ± 0,004 | 0,501 ± 0,003 | 0,875 ± 0,002 | 1,82 ± 0,10 |
| gradient | 0,082 ± 0,015 | 0,131 ± 0,022 | 0,535 ± 0,024 | 10,21 ± 1,01 |
| integrated_gradients | 0,433 ± 0,013 | 0,587 ± 0,013 | 0,856 ± 0,009 | 1,20 ± 0,02 |
| occlusion | 0,758 ± 0,011 | 0,824 ± 0,009 | 0,978 ± 0,005 | 1,14 ± 0,03 |
| gnnexplainer | 0,427 ± 0,007 | 0,581 ± 0,008 | 0,818 ± 0,006 | 1,48 ± 0,02 |
| pgexplainer | 0,183 ± 0,030 | 0,280 ± 0,042 | 0,645 ± 0,010 | 4,37 ± 0,59 |
| gnn_lrp | 0,263 ± 0,037 | 0,388 ± 0,047 | 0,697 ± 0,008 | 2,25 ± 0,21 |
| graphsvx | 0,512 ± 0,018 | 0,635 ± 0,015 | 0,941 ± 0,002 | 1,58 ± 0,20 |
| subgraphx | 0,594 ± 0,010 | 0,625 ± 0,009 | 0,806 ± 0,005 | 7,51 ± 0,45 |
| shapley | 0,505 ± 0,046 | 0,626 ± 0,042 | 0,939 ± 0,011 | 1,63 ± 0,39 |
| cf_gnnexplainer | 0,225 ± 0,025 | 0,329 ± 0,034 | 0,545 ± 0,039 | 5,88 ± 2,61 |
| graphmask | 0,108 ± 0,019 | 0,163 ± 0,023 | 0,581 ± 0,022 | 10,46 ± 0,64 |
| gstarx | 0,422 ± 0,009 | 0,537 ± 0,012 | 0,884 ± 0,006 | 2,08 ± 0,09 |

**đáp án theo nhãn (mọi node cùng tên).**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên |
| --- | --- | --- | --- | --- |
| model | 0,242 ± 0,024 | 0,358 ± 0,026 | 0,700 ± 0,014 | 1,15 ± 0,03 |
| random | 0,088 ± 0,004 | 0,144 ± 0,005 | 0,502 ± 0,010 | 8,28 ± 0,46 |
| lowest | 0,015 ± 0,000 | 0,020 ± 0,000 | 0,300 ± 0,014 | 23,91 ± 0,24 |
| clip | 0,436 ± 0,001 | 0,555 ± 0,001 | 0,873 ± 0,002 | 1,80 ± 0,10 |
| gradient | 0,131 ± 0,020 | 0,201 ± 0,029 | 0,535 ± 0,030 | 7,57 ± 1,11 |
| integrated_gradients | 0,570 ± 0,028 | 0,714 ± 0,023 | 0,886 ± 0,009 | 1,00 ± 0,00 |
| occlusion | 0,738 ± 0,012 | 0,816 ± 0,009 | 0,924 ± 0,003 | 1,00 ± 0,01 |
| gnnexplainer | 0,517 ± 0,015 | 0,667 ± 0,014 | 0,833 ± 0,008 | 1,04 ± 0,03 |
| pgexplainer | 0,283 ± 0,042 | 0,397 ± 0,049 | 0,711 ± 0,011 | 3,55 ± 0,37 |
| gnn_lrp | 0,300 ± 0,028 | 0,437 ± 0,034 | 0,728 ± 0,012 | 1,67 ± 0,13 |
| graphsvx | 0,645 ± 0,019 | 0,755 ± 0,013 | 0,951 ± 0,002 | 1,05 ± 0,02 |
| subgraphx | 0,525 ± 0,007 | 0,598 ± 0,007 | 0,778 ± 0,003 | 4,03 ± 0,35 |
| shapley | 0,620 ± 0,037 | 0,739 ± 0,026 | 0,945 ± 0,004 | 1,04 ± 0,02 |
| cf_gnnexplainer | 0,241 ± 0,029 | 0,351 ± 0,039 | 0,529 ± 0,042 | 4,31 ± 1,97 |
| graphmask | 0,117 ± 0,022 | 0,184 ± 0,028 | 0,516 ± 0,017 | 9,24 ± 0,77 |
| gstarx | 0,520 ± 0,015 | 0,646 ± 0,013 | 0,897 ± 0,008 | 1,07 ± 0,02 |

## B. Dữ liệu tổng hợp (`synthetic`)

### synthetic

ca theo seed: 100, 100, 100; cỡ đáp án 4,0 ± 0,0; số phần 18,6 ± 0,0; R@1 val tổng hợp 100,0 ± 0,0%.

**đáp án chính.**

| Cách | GEA | Precision@g | AUROC | Vị trí phần đúng đầu tiên |
| --- | --- | --- | --- | --- |
| model | 0,665 ± 0,135 | 0,767 ± 0,108 | 0,956 ± 0,026 | 1,00 ± 0,00 |
| random | 0,143 ± 0,014 | 0,227 ± 0,024 | 0,499 ± 0,017 | 3,96 ± 0,45 |
| lowest | 0,000 ± 0,000 | 0,000 ± 0,000 | 0,044 ± 0,026 | 14,43 ± 0,64 |
| clip | 0,878 ± 0,000 | 0,924 ± 0,000 | 0,989 ± 0,000 | 1,00 ± 0,00 |
| gradient | 0,423 ± 0,052 | 0,548 ± 0,061 | 0,624 ± 0,024 | 1,70 ± 0,59 |
| integrated_gradients | 0,802 ± 0,031 | 0,867 ± 0,022 | 0,933 ± 0,013 | 1,00 ± 0,00 |
| occlusion | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,000 ± 0,000 | 1,00 ± 0,00 |
| gnnexplainer | 0,772 ± 0,046 | 0,840 ± 0,037 | 0,887 ± 0,025 | 1,00 ± 0,00 |
| pgexplainer | 0,511 ± 0,191 | 0,639 ± 0,174 | 0,873 ± 0,085 | 1,02 ± 0,02 |
| gnn_lrp | 0,550 ± 0,073 | 0,680 ± 0,064 | 0,732 ± 0,012 | 1,04 ± 0,07 |
| graphsvx | 0,333 ± 0,010 | 0,481 ± 0,010 | 0,788 ± 0,003 | 1,00 ± 0,00 |
| subgraphx | 0,839 ± 0,011 | 0,879 ± 0,009 | 0,939 ± 0,002 | 1,00 ± 0,01 |
| shapley | 0,514 ± 0,119 | 0,641 ± 0,102 | 0,913 ± 0,055 | 1,00 ± 0,00 |
| cf_gnnexplainer | 0,699 ± 0,046 | 0,770 ± 0,039 | 0,808 ± 0,047 | 1,79 ± 0,23 |
| graphmask | 0,171 ± 0,013 | 0,254 ± 0,013 | 0,546 ± 0,005 | 4,61 ± 0,16 |
| gstarx | 0,283 ± 0,030 | 0,422 ± 0,034 | 0,779 ± 0,025 | 1,81 ± 0,13 |

