# Tái lập NativE (SIGIR 2024) trên MKG-W — nhật ký thiết lập

Người làm: Đạt. Việc: dòng 22 bảng phân công (tái lập bước a: link prediction gốc, so số bài báo).

## Nguồn

- Code: `zjukg/NativE`, commit `c72291c3d571b36c754897935d21718aa4568fe7` (2024-08-12), clone về `~/work/NativE` (ngoài repo nhóm, không commit).
- Bài báo: NativE: Multi-modal Knowledge Graph Completion in the Wild, SIGIR 2024 (arXiv 2406.17605).
- Embeddings đa phương thức: Google Drive trong README của tác giả (folder `191u4WhT...`), file `MKG.zip`.

## Môi trường (2026-09-24)

- Conda env `native`: Python 3.10, torch 2.14.0+cpu, numpy/scikit-learn/tqdm mới nhất.
- Không cần đúng phiên bản cổ trong requirements của tác giả (Python 3.8, torch 1.9.1): toàn bộ import và vòng train chạy được với torch 2.14.
- **Sự cố + cách sửa**: `mmkgc/release/Base.so` kèm sẵn trong repo bị segfault ngay khi đọc `train2id.txt` trên máy Linux của Đạt. Sửa bằng cách biên dịch lại từ source: `cd mmkgc && bash make.sh` (cần g++). Sau khi biên dịch lại thì chạy bình thường — ai chạy trên máy khác cũng nên rebuild trước.

## Dữ liệu đã kiểm kê

| Thứ | Giá trị | Khớp bài báo? |
|---|---|---|
| Thực thể | 15 000 (URI Wikidata trong `entity2id.txt`) | ✅ |
| Quan hệ | 169 | ✅ |
| Triple train / valid / test | 34 196 / 4 276 / 4 274 | ✅ |
| `MKG-W-visual.pth` | 15000 × 383, float32, sha256 đầu `9cf9346fb7bcd1d7` | — |
| `MKG-W-textual.pth` | 15000 × 384, float32, sha256 đầu `13b765b5bddceaef` | — |

**Lưu ý quan trọng cho gói dữ liệu retrieval (dòng 20)**: NativE **không phát ảnh gốc**, chỉ có visual features 383 chiều đã trích + giảm chiều sẵn. Muốn trích CLIP theo pipeline nhóm phải tìm ảnh gốc từ nguồn khác (kho MMRNS / pengfei-luo/multimodal-knowledge-graph), hoặc dùng features này và ghi rõ khác biệt encoder trong báo cáo.

## Smoke test (2026-09-24, chưa phải kết quả)

- Lệnh: `python run_cpu_smoke.py` (wrapper vá `.cuda()` thành no-op để chạy máy không GPU, epoch=2, còn lại đúng siêu tham số `scripts/run_mkgw.sh`).
- Kết quả: chạy trọn 2 epoch + phase test, exit 0. D loss 238.4 → 21.8, G loss 340.1 → 123.0.
- Tốc độ CPU ~6.5 phút/epoch → 1000 epoch ≈ 4–5 ngày CPU. **Kết luận: chạy thật phải trên GPU** (ước lượng vài giờ trên RTX 3070/4090).
- Số metric của smoke test vô nghĩa (2 epoch), không dùng.

## Bước tiếp theo

1. Chạy đủ config gốc `scripts/run_mkgw.sh` (epoch=1000, margin=4, dim=250, neg_num=128, lr=1e-4) trên máy GPU, seed gốc 42.
2. So MRR / Hits@1/3/10 với bảng MKG-W trong bài báo; chênh ≤ ~1 điểm coi là tái lập được.
3. Ghi log + metrics vào folder này (`metrics.json` theo schema `write_metrics()` khi vào pipeline nhóm; log train giữ nguyên của tác giả).

## Kiểm tra trước khi train — kênh đồ thị cho bước b (2026-09-25)

Thiết kế bước b: truy vấn (mô tả đã che tên) thường nhắc tên các thực thể KHÁC → `EntityLinker`
khớp label/alias trong truy vấn, rồi hai kênh điểm trên thực thể ứng viên: (1) cosine embedding
NativE với thực thể được nhắc, (2) đếm cạnh KGC-train trực tiếp — các match dùng để tính điểm
chính là giải thích (không hậu kiểm). Trộn theo đúng khung cấp 1: `α·z(CLIP) + (1−α)·z(β·z(emb) + (1−β)·z(triple))`.

Số đo (`scripts/checks/mkgw_link_coverage.py`, chỉ train+val, không đọc test; kết quả
`experiments/checks/mkgw_link_coverage.json`):

| Split | Truy vấn | Link ≥1 thực thể | Đáp án có cạnh train trực tiếp |
|---|---|---|---|
| train | 4 602 | 81,16% | 30,12% |
| val | 575 | 81,39% | 30,09% |

Kết luận: kênh đồ thị phủ tốt (81% truy vấn), và gần 1/3 truy vấn có giải thích dạng triple
(vd "commune in Pyrénées-Atlantiques, France" → Bayonne —country→ France). Đủ điều kiện
tiếp tục bước b sau khi có embedding từ checkpoint train thật. Kênh embedding hiện đo bằng
checkpoint smoke (2 epoch) nên chưa nói lên gì; chỉ kênh cấu trúc (link + cạnh) là kết luận được.
