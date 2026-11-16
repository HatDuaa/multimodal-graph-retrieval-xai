# Phần 1 — So sánh parser câu (val, không train)

Lệnh chạy lại: `run_parse.sh` (1a), rồi `run_eval.sh` (1b, 1c, và phần 2). Bảng sinh bằng
`python scripts/improve/summarize_parsers.py` (ghi `summary.md`). File kết quả: `intrinsic.json`, `probe_<parser>.json`,
`rerank_main_r3_<parser>.json`, `timing_val_<parser>.json`. 20 câu đặt cạnh nhau: `side_by_side.md`. Đồ thị câu của
từng parser: `data/processed/parsers/query_graphs_val_<parser>.json` (không commit, giống mọi file trong `data/processed`).

## 1a. Các parser

| Parser | Nguồn | Cách chạy |
|---|---|---|
| sng | `sng_parser` (SceneGraphParser 0.1.0, spaCy `en_core_web_sm`), parser cấp 1 | 16 process CPU; kết quả trùng từng câu với `data/processed/query_graphs_val.json` |
| factual-base | FACTUAL-T5 (Li et al., ACL Findings 2023), `lizhuang144/flan-t5-base-VG-factual-sg` | gói `FactualSceneGraph` 0.7.3, beam 5, batch 64, `filter_factual_chars=True` |
| factual-large | `lizhuang144/flan-t5-large-VG-factual-sg` | như trên |
| qwen2.5-3b | `Qwen/Qwen2.5-3B-Instruct` (repo chính chủ, bf16, 6,2 GB) | few-shot (3 ví dụ tự viết, không lấy từ val), trả JSON, greedy, batch 64; prompt trong `src/graph/caption_parsers.py` |

Mọi parser được đổi về cùng schema đồ thị câu (`src/graph/caption_parsers.py`). Với factual và qwen, tên vật thể được
viết thường và lemma hoá từ cuối bằng spaCy, giống `lemma_head` của sng. Cụm danh từ ghép được giữ nguyên, ví dụ
"tennis racket". Phía câu vẫn đi qua `GraphStore._graph_parts(query=True)`: bỏ đại từ, ẩn nhãn rỗng.

Ghi chú cài đặt:

- Cài `FactualSceneGraph` theo cách thông thường sẽ hạ torch 2.14 (bản cu130, chạy được với driver mới) xuống 2.6.0.
  Vì vậy gói được cài với `--no-deps`, cộng thêm `nltk` 3.10.3. Phần parser của gói chỉ cần torch, transformers, nltk.
- transformers 5 đã bỏ `as_target_tokenizer`. Gói chỉ dùng hàm này để đếm token của prompt, nên được thay bằng một
  context manager rỗng.
- Qwen có 18/10 633 câu (0,17%) không trả được JSON hợp lệ; các câu này được tính là đồ thị rỗng.
- Đĩa: ba model chiếm 9,7 GB, đĩa còn trống 35 GB.

## 1b. Đo nội tại (val, 10 633 câu)

| Parser | câu có ≥1 bộ ba | vật thể/câu | bộ ba/câu | quan hệ bị bỏ | node đại từ | node rỗng | node OOV | vật thể có trong ảnh đúng | câu có ≥1 bộ ba khớp ảnh đúng | bộ ba khớp ảnh đúng | giây / 1 000 câu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sng | 85.80% | 3.06 | 1.65 | 5.89% | 3.92% | 0.00% | 4.92% | 46.44% | 5.58% | 3.61% | 0.4 (cpu, 16 processes) |
| factual-base | 97.17% | 2.77 | 1.73 | 0.17% | 0.10% | 0.00% | 1.75% | 53.91% | 4.74% | 3.00% | 14.3 (cuda (RTX 4090)) |
| factual-large | 97.26% | 2.80 | 1.76 | 0.17% | 0.08% | 0.00% | 1.67% | 54.03% | 4.88% | 3.04% | 36.0 (cuda (RTX 4090)) |
| qwen2.5-3b | 84.60% | 2.64 | 1.21 | 0.11% | 0.06% | 0.00% | 5.33% | 53.84% | 4.66% | 4.01% | 137.4 (cuda (RTX 4090)) |

## 1c. Probe 3 kênh không train (val, top-50 CLIP)

| Parser | R@1 tốt nhất | R@5 | R@10 | MRR | (τ, α, β) tốt nhất | R@1 tại (0,05; 0,6; 0,7) | MRR tại điểm đó | chỉ vật thể R@1 | chỉ bộ ba R@1 | Δ R@1 tốt nhất so với sng |
|---|---|---|---|---|---|---|---|---|---|---|
| sng | 43.50 | 70.95 | 81.59 | 56.10 | (0.05, 0.6, 0.7) | 43.50 | 56.10 | 43.21 | 41.45 | +0.00 |
| factual-base | 43.52 | 70.82 | 81.70 | 56.01 | (0.05, 0.6, 0.5) | 43.27 | 56.00 | 43.14 | 41.94 | +0.02 |
| factual-large | 43.62 | 70.39 | 81.08 | 55.96 | (0.02, 0.5, 0.6) | 43.46 | 56.13 | 43.13 | 41.74 | +0.12 |
| qwen2.5-3b | 43.66 | 71.28 | 81.64 | 56.27 | (0.02, 0.6, 0.8) | 43.46 | 56.07 | 43.24 | 41.67 | +0.16 |

Chỉ CLIP: R@1 39.13, MRR 52.34.

## 1c. main_r3 đã train (3 seed), chỉ thay đồ thị câu

| Parser | R@1 toàn val | MRR toàn val | R@1 câu có bộ ba (parser đó) | n | R@1 câu có bộ ba theo sng | n |
|---|---|---|---|---|---|---|
| sng | 46.48 ± 0.24 | 58.85 ± 0.18 | 47.37 ± 0.29 | 9123 | 47.37 ± 0.29 | 9123 |
| factual-base | 46.36 ± 0.14 | 58.85 ± 0.09 | 46.84 ± 0.14 | 10332 | 47.06 ± 0.16 | 9123 |
| factual-large | 46.61 ± 0.12 | 59.06 ± 0.03 | 47.02 ± 0.12 | 10342 | 47.29 ± 0.06 | 9123 |
| qwen2.5-3b | 46.04 ± 0.12 | 58.58 ± 0.03 | 47.53 ± 0.21 | 8995 | 46.67 ± 0.14 | 9123 |

Định nghĩa các cột:

- "Quan hệ bị bỏ": tỉ lệ quan hệ của parser bị loại ở phía câu vì chạm vào một đại từ.
- "Node OOV": nhãn không xuất hiện ở bất kỳ đồ thị ảnh train nào, sau khi chuẩn hoá alias VG.
- Ba cột "khớp ảnh đúng": khớp nhãn chính xác sau alias, so với đồ thị VG của ảnh đúng
  (`src/graph/coverage.py`, open vocabulary). Đây là cận dưới, vì không dùng embedding.
- Probe: cùng bộ xếp hạng và lưới như `scripts/checks/three_channel_probe.py`, K = 50. Với sng, probe tái hiện đúng
  mốc cấp 1 (R@1 43,4967 tại τ 0,05, α 0,6, β 0,7). "Tốt nhất" là chọn trên chính val nên lạc quan; cột tại điểm cố
  định (0,05; 0,6; 0,7) so các parser ở cùng siêu tham số.
- main_r3 được train trên đồ thị sng, nên các parser khác bị chấm trong điều kiện **lệch phân phối**. Phần chữ mới
  mã hoá bằng CLIP đóng băng, giống `graph_mode`.

## Nhận xét

1. **FACTUAL sửa đúng các lỗi đã thấy ở sng.**
   - Node đại từ giảm từ 3,92% xuống 0,10%. Quan hệ bị bỏ vì đại từ giảm từ 5,89% xuống 0,17%.
   - Câu có ≥1 bộ ba tăng từ 85,8% lên 97,2%. Node OOV giảm từ 4,9% xuống 1,7%.
   - Tỉ lệ vật thể có trong ảnh đúng tăng từ 46,4% lên 53,9%.
   - Ví dụ ở `side_by_side.md`: "A man and woman walking huddled under an umbrella" thì sng ra "walking —under→ umbrella",
     còn FACTUAL ra "man/woman —walk under→ umbrella".
   - FACTUAL gộp các entity trùng tên, nên số vật thể mỗi câu thấp hơn (2,77 so với 3,06). Vị từ của nó ở dạng nguyên
     mẫu ("hand", "fly"), nên khớp chính xác bộ ba với VG lại thấp hơn (3,00% so với 3,61%), vì VG ghi "holding",
     "flying"...
2. **Qwen2.5-3B kém hơn FACTUAL và chậm nhất.** Nó ra ít bộ ba (1,21 mỗi câu, 84,6% câu có ≥1 bộ ba), chậm gấp 10 lần
   FACTUAL-base (137 giây cho 1 000 câu), và có lỗi quan hệ, ví dụ "man —laying next to→ sand". Nó cũng hay điền
   count = 1 dù câu không nêu số.
3. **Probe: không parser nào đạt ngưỡng +0,3 R@1.** Mức tăng R@1 tốt nhất so với sng: factual-base +0,02, factual-large
   +0,12, qwen2.5-3b +0,16, đều chọn trên val. Ở điểm cố định của cấp 1, cả ba đều ≤ sng (43,27–43,46 so với 43,50).
   Kênh vật thể đứng riêng gần như không đổi (43,13–43,24 so với 43,21). Kênh bộ ba đứng riêng nhích lên một chút
   (41,67–41,94 so với 41,45).
4. **main_r3 với đồ thị mới:** factual-large đạt 46,61 ± 0,12, hơn sng 0,13 nhưng vẫn trong nhiễu (std của sng là
   0,24), dù mô hình train trên sng. Factual-base đạt 46,36; qwen 46,04.
5. **Quyết định:** theo luật đã đặt, dừng ở 1c, **không làm 1d** (không parse train, không đóng gói lại, không train).
   Parser sạch hơn rõ rệt ở chỉ số nội tại nhưng không chuyển thành điểm xếp hạng. Lý do có thể: phần lợi lớn nhất
   (bỏ ít quan hệ hơn, ít node đại từ hơn) chỉ chạm khoảng 6% số quan hệ, và kênh CLIP đã chiếm phần lớn điểm.
