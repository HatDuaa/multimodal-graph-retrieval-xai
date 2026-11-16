# Ablation bỏ cạnh trên mô hình chính: `countonly_noedges_r3` (val)

Lộc duyệt ablation này ngày 25/10/2026. Đề bài yêu cầu thử "bỏ cạnh hoặc thay bằng cạnh ngẫu nhiên". Phần cạnh ngẫu
nhiên lúc eval đã có ở `experiments/improve/controls/`; train lại với cạnh ngẫu nhiên thì chưa chạy. Chỉ dùng val,
không chấm test, giống các ablation của cấp 1.

## Cách làm

- **Công thức:** giữ y `countonly_factual_r3` (`experiments/improve/count-only/run.sh`). Dữ liệu count_v1, fusion raw
  với cùng `--fusion-scale`, bảng g, node đã gộp, cùng cờ cấu trúc, micro-batch 32, `expandable_segments:False`,
  tối đa 10 epoch với cùng luật dừng sớm, seed 0, 1, 2.
- **Thay đổi duy nhất:** thêm `--no-use-edges`, đúng nghĩa `no_edges_r3` của cấp 1.
  - GAT không nhận cạnh nào: không quan hệ, không cạnh kNN `[none]`, không cạnh nối node tổng.
  - Kênh bộ ba tắt, nên trọng số c = 0.
  - Chạy thử 1 epoch trước khi train: gradient của `p_edge`, `geometry_mlp`, `none_predicate`, `global_predicate` đều
    bằng 0, và c = 0.
- **Script:** `run.sh` (chuỗi tmux, có `--resume`, chặn khi đĩa dưới 20 GB), `scripts/improve/summarize_noedges.py`
  (`summary.json`), `scripts/improve/noedges_groups.py` (`groups.json`). Log: `logs/`, `groups.log`.
- **Mỗi run chiếm 17 MB** (checkpoint và log). Đĩa còn 40 GB suốt quá trình.

## Từng seed

| Run | Epoch tốt nhất / số epoch đã chạy | R@1 | R@5 | R@10 | MRR | Thời gian train | Giây / epoch | Bộ nhớ GPU đỉnh |
|---|---|---|---|---|---|---|---|---|
| `countonly_noedges_r3_seed0` | 7 / 9 | 46,53 | 73,37 | 83,30 | 58,78 | 87 phút | 582 | 4,8 GB |
| `countonly_noedges_r3_seed1` | 6 / 8 | 46,54 | 73,24 | 83,03 | 58,64 | 78 phút | 582 | 4,8 GB |
| `countonly_noedges_r3_seed2` | 8 / 10 | 46,50 | 73,52 | 83,34 | 58,72 | 97 phút | 584 | 4,8 GB |
| `countonly_factual_r3` (đủ cạnh, để so) | 8, 9, 5 / 10, 10, 7 | 49,15 / 48,85 / 48,74 | | | | 156 / 156 / 110 phút | 934–941 | 7,1 GB |

Bỏ cạnh làm mỗi epoch nhanh hơn khoảng 1,6 lần và dùng ít bộ nhớ hơn. Cả chuỗi 3 seed chạy từ 13:45 tới 18:09.

## So với mô hình đủ cạnh (val, mean ± std qua 3 seed, %)

| Mô hình | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| `countonly_factual_r3` | 48,91 ± 0,21 | 75,22 ± 0,05 | 84,96 ± 0,25 | 60,86 ± 0,22 |
| `countonly_noedges_r3` | 46,52 ± 0,02 | 73,38 ± 0,14 | 83,23 ± 0,17 | 58,71 ± 0,07 |
| Chênh (bỏ cạnh − đủ cạnh) | **−2,39** (p = 0,0024) | −1,84 (p = 0,0005) | −1,73 (p = 0,001) | −2,14 (p = 0,0015) |
| `main_r3` (cấp 1) | 46,48 ± 0,24 | 73,78 ± 0,16 | 83,44 ± 0,24 | 58,85 ± 0,18 |
| `no_edges_r3` (cấp 1) | 45,98 ± 0,10 | 73,12 ± 0,05 | 82,90 ± 0,04 | 58,29 ± 0,04 |
| Chênh ở cấp 1 | −0,50 (p = 0,055) | −0,66 (p = 0,014) | −0,54 (p = 0,055) | −0,56 (p = 0,027) |
| CLIP thuần | 39,13 | 67,88 | 78,97 | 52,34 |

p là Welch giữa 3 seed của hai bên.

## Theo nhóm câu (cùng các nhóm của Việc 7, `experiments/improve/alpha-groups/`)

- Δ = R@1 của mô hình đủ cạnh − R@1 của bản bỏ cạnh. KTC 95% lấy bằng bootstrap ghép cặp theo câu, trên tỉ lệ đúng đã
  trung bình qua seed (2 000 lần lấy mẫu, rng 2026). p là Welch giữa 3 + 3 seed.
- "Phần của cạnh" = Δ / (mô hình đủ cạnh − CLIP), tức cạnh chiếm bao nhiêu trong toàn bộ phần lời so với CLIP.
- R@1 của từng run tính lại ở đây trùng `metrics.json`.

| Nhóm | N | CLIP | countonly | noedges | main_r3 | no_edges_r3 | Δ countonly [KTC 95%], p | Phần của cạnh (countonly) | Δ cấp 1 [KTC 95%], p | Phần của cạnh (cấp 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| Tất cả | 10 633 | 39,13 | 48,91 | 46,52 | 46,48 | 45,98 | +2,39 [1,90; 2,86], 0,002 | 24% | +0,50 [0,13; 0,85], 0,055 | 7% |
| Có số nêu rõ | 1 235 | 44,21 | 54,06 | 50,39 | 50,69 | 50,23 | +3,67 [2,29; 5,02], 0,026 | 37% | +0,46 [−0,59; 1,51], 0,045 | 7% |
| Không có số nêu rõ | 9 398 | 38,47 | 48,23 | 46,02 | 45,92 | 45,42 | +2,22 [1,71; 2,74], 0,0002 | 23% | +0,50 [0,14; 0,88], 0,062 | 7% |
| Có ≥1 quan hệ | 10 342 | 39,31 | 49,38 | 46,92 | 46,83 | 46,32 | +2,46 [1,98; 2,94], 0,001 | 24% | +0,52 [0,16; 0,90], 0,054 | 7% |
| Chỉ vật thể | 291 | 32,99 | 32,19 | 32,42 | 33,91 | 34,02 | −0,23 [−2,52; 2,18], 0,80 | — (không hơn CLIP) | −0,11 [−2,29; 1,95], 0,86 | không có nghĩa |
| 2 vật thể | 3 757 | 35,67 | 42,41 | 40,45 | 40,87 | 40,35 | +1,96 [1,20; 2,77], 0,009 | 29% | +0,51 [−0,08; 1,05], 0,048 | 10% |
| ≥3 vật thể | 6 607 | 41,40 | 53,32 | 50,58 | 50,20 | 49,68 | +2,73 [2,13; 3,32], 0,003 | 23% | +0,52 [0,05; 0,99], 0,10 | 6% |

MRR theo nhóm nằm trong `groups.json`. Chiều thay đổi giống R@1: Δ MRR của countonly là +2,14 trên tất cả, +2,56 ở câu có
số nêu rõ, +2,39 ở câu ≥3 vật thể và −0,13 ở câu chỉ có vật thể. Nhóm "1 vật thể" (269 câu) gần như trùng nhóm "chỉ vật
thể" nên không ghi riêng.

## Kết luận

1. **Cạnh quan trọng với mô hình mới hơn nhiều so với cấp 1.**
   - Bỏ toàn bộ cạnh làm R@1 val giảm 2,39 điểm (p = 0,002), so với 0,50 ở cấp 1 (p = 0,055).
   - Tính theo phần lời so với CLIP, cạnh chiếm 24% ở mô hình mới và 7% ở cấp 1.
2. **Bản bỏ cạnh của mô hình mới chỉ ngang mô hình cấp 1 đủ cạnh** (46,52 so với 46,48).
   - Phần mô hình mới hơn cấp 1 (+2,43) gần bằng đúng phần mất khi bỏ cạnh (2,39).
   - Các thay đổi chỉ chạm vào node (FACTUAL, fusion raw, gộp trùng, số lượng và g), khi không có cạnh, chỉ đưa mô hình
     về đúng mức cấp 1.
   - Những thay đổi cần cạnh (cạnh hoàn thiện kèm hình học, node tổng, kênh bộ ba trên đồ thị đó) mang phần lời còn lại.
   - Cần nói rõ hai giới hạn: bỏ cạnh cũng tắt luôn kênh bộ ba, và hai mô hình này khác nhau ở nhiều chỗ, nên đây là so
     sánh giữa hai cấu hình, không phải ablation từng phần.
3. **Phần mất tập trung ở câu có quan hệ và nhiều vật thể.**
   - Câu có ≥1 quan hệ: −2,46. Câu 2 vật thể: −1,96. Câu ≥3 vật thể: −2,73.
   - Câu chỉ có vật thể: −0,23, không có ý nghĩa thống kê; ở nhóm này mô hình đủ cạnh vốn cũng không hơn CLIP.
   - Như vậy cạnh giúp đúng ở chỗ câu tả nhiều thành phần liên kết với nhau, khớp với kết luận của Việc 7.
4. **Câu có số nêu rõ mất nhiều nhất** (−3,67, 37% phần lời).
   - Một cách giải thích có thể: bảng g cộng vào độ giống giữa các node đã qua GAT, nên khi các node đó được truyền tin từ
     hàng xóm (vị trí, node tổng), mô hình phân biệt các instance tốt hơn.
   - Đây là giả thuyết, chưa kiểm. Nhóm này nhỏ (1 235 câu), std giữa các seed của countonly cũng lớn (± 1,09).
5. **Ghép với phép xáo cạnh lúc eval** (`experiments/improve/controls/`):
   - Xáo đích các quan hệ có nhãn lúc eval chỉ làm R@1 giảm 0,59, trong khi train lại không có cạnh nào làm giảm 2,39.
   - Mô hình cần có cạnh lúc train và lúc chấm, nhưng nhãn cụ thể của quan hệ nào nối node nào chỉ đóng góp một phần
     nhỏ; cạnh kNN không gian vẫn còn nguyên khi xáo.
   - Muốn tách phần của cạnh kNN với phần của quan hệ có nhãn thì cần thêm một ablation chỉ bỏ cạnh kNN. Ablation này
     chưa chạy.
