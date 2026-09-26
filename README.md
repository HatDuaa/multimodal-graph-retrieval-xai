# Project cuối kì — Học máy với dữ liệu đồ thị

## Đề đã chọn

**Project 3 — Truy vấn đa phương thức có giải thích bằng đồ thị**
(Multimodal retrieval + graph re-ranking + explainable AI). Đề gốc: [assignments/Project03_Multimodal_XAI.pdf](assignments/Project03_Multimodal_XAI.pdf).

Hai đề còn lại để tham khảo, không làm:
- [assignments/Project01_Graph_Editing.pdf](assignments/Project01_Graph_Editing.pdf) — scene graph cho chỉnh sửa ảnh. Loại vì phụ thuộc nhiều mô hình vision nặng (RelTR, InstructPix2Pix), gán nhãn tay lớn.
- [assignments/Project02_Knowledge_Tracing.pdf](assignments/Project02_Knowledge_Tracing.pdf) — knowledge tracing. An toàn nhưng không tận dụng được kiến thức MMKG từ seminar.

Lý do chọn Project 3: cấp độ 2 yêu cầu tái lập một phương pháp multimodal KG từ 2024 (NativE, AdaMF-MAT, Mixed-Curvature MMKGC), trùng với các baseline trong hai bài seminar HFR-MKGC và MDBGF; nhóm đã quen dataset MKG-W / MKG-Y / DB15K từ seminar (tải lại bằng script trong `data/`, nguồn: kho mã của NativE / MMRNS).

## Nhóm 7 (5 thành viên)

Giảng viên hướng dẫn: **TS. Lê Ngọc Thành**.

| MSHV | Họ tên | Vai trò ở project |
|---|---|---|
| 25C11050 | Nguyễn Đình Lộc | — |
| 25C15003 | Ngô Trương Minh Đạt | — |
| 25C15015 | Trần Đắc Khoa | — |
| 25C15055 | Nguyễn Bùi Mẫn Nhi | — |
| 25C15045 | Âu Dương Khang | — |

**Phân công:** chưa chốt, mỗi người tự chọn việc trong [task-assignment.xlsx](task-assignment.xlsx) (hàng = đầu việc theo giai đoạn, cột = thành viên, điền "Chính" / "Hỗ trợ"). Bảng phân công cuối cùng phải ghi vào báo cáo (đề yêu cầu) kèm lịch sử commit; giảng viên vấn đáp từng người nên ai cũng cần có commit ở phần của mình.

## Kế hoạch theo cấp độ

**Hạn nộp + vấn đáp: 22/11/2026.** Kế hoạch chi tiết (5 gói công việc, phụ thuộc, lộ trình 9 tuần, rủi ro): [plan.md](plan.md).

| Giai đoạn | Bộ dữ liệu | Việc chính |
|---|---|---|
| 0. Nền chung | — | Khung repo + config theo seed; trích đặc trưng CLIP + FAISS; module đánh giá (Recall@1/5/10, MRR, ≥3 seed); thử CLIP đa ngữ cho truy vấn tiếng Việt để chốt encoder. |
| 1. Cấp độ 1 | Visual Genome ∩ COCO | Join `coco_id`, tập con cố định, split Karpathy → baseline CLIP → dựng scene graph từng ảnh (nhãn mở của Visual Genome, không lọc VG150, không ConceptNet) → tách truy vấn thành đồ thị → GAT + hai kênh khớp vật thể / bộ ba → re-rank với trọng số a, b, c học theo từng câu (quét α trên validation cho dòng trọng số cố định) → ablation và đối chứng bỏ/xáo cạnh → hiển thị đường đi → platform demo (web app nhỏ: nhập truy vấn → top-k, so CLIP với CLIP + Graph, đường đi giải thích; Lộc phụ trách) → phân tích 5 thành công / 5 thất bại. |
| 2. Cấp độ 2 | + MKG-W | Sinh truy vấn từ mô tả thực thể, chia truy vấn và triple theo split → chạy lại CLIP và CLIP + GNN → tái lập NativE/AdaMF-MAT (a: link prediction gốc, so với bài báo; b: embedding thay GNN trong điểm đồ thị) → bộ ≥20 truy vấn mỗi tập, báo cáo tách trực tiếp / gián tiếp. |
| 3. Cấp độ 3 | + MMKG tiếng Việt | Chốt định nghĩa và cài đặt fidelity / validity / sparsity trên ≥10 truy vấn → xây MMKG tiếng Việt (50–150 thực thể, 300–1.000 triple, 200–500 ảnh) + data card → chạy phương pháp tốt nhất, phân tích chuyển giao. |
| 4. Mở rộng | 2 tập công khai | Ưu tiên hướng (a): α thích nghi theo truy vấn hoặc cách chọn đường đi giải thích. Hướng (c) khảo sát người dùng nếu còn thời gian. Bỏ hướng video. |
| 5. Sản phẩm nộp | — | Báo cáo 12–18 trang, slide, demo + video dự phòng, README chạy lại, bảng phân công cuối. |

Làm song song được ngay từ đầu: tái lập NativE/AdaMF-MAT (bước a) và thu thập MMKG tiếng Việt, cả hai không phụ thuộc cấp độ 1.

Rủi ro chính: CLIP vốn mạnh trên COCO nên đồ thị có thể không cải thiện truy vấn trực tiếp. Bộ truy vấn gián tiếp cần soạn sớm để kiểm tra hướng đi.

## Các cấp độ và điểm trần (theo đề)

| Cấp | Nội dung tóm tắt | Điểm trần |
|---|---|---|
| 1 | ≥1 tập công khai. Baseline CLIP đóng băng; đồ thị từ dữ liệu có sẵn; GCN/GAT/GraphSAGE; re-rank CLIP + Graph; ablation bỏ/xáo cạnh; hiển thị đường đi giải thích; 5 thành công / 5 thất bại. | 7.0 |
| 2 | ≥2 tập công khai. Tái lập 1 phương pháp ≥2024 (multimodal KG / graph-enhanced retrieval / explainable graph retrieval). Bộ ≥20 truy vấn có nhóm gián tiếp, báo cáo tách riêng. | 8.5 |
| 3 | Fidelity, validity, sparsity định lượng trên ≥10 truy vấn. Tự xây MMKG nhỏ tiếng Việt (50–150 thực thể, 300–1.000 triple, 200–500 ảnh), chạy phương pháp tốt nhất, phân tích chuyển giao. | 10 |
| 4 | Cải tiến riêng (graph-aware contrastive loss, re-ranking thích nghi, chọn đường đi giải thích…), video, hoặc khảo sát người dùng. Cần giả thuyết + ablation + ≥2 tập. | +1.0 – 2.0 |

Metric bắt buộc: Recall@1/5/10, MRR; so CLIP thuần với CLIP + Graph trên cùng cách chia; ≥3 seed, báo cáo mean ± std.

## Quyết định kỹ thuật ban đầu

- **Nhánh A (text-to-image)** là bắt buộc. Image-to-text làm sau nếu còn thời gian.
- **Ba bộ dữ liệu (chốt 2026-09-17), mỗi bộ chạy riêng, chung một pipeline:**
  1. **Visual Genome, phần giao với MS-COCO** (join theo `coco_id`): truy vấn = caption COCO (split Karpathy), đồ thị = scene graph Visual Genome của từng ảnh, nhãn mở đã chuẩn hoá alias, không lọc VG150, không dùng ConceptNet ở cấp độ 1 (thiết kế cuối, xem `docs/level1-model.md`). Tính là **một** tập. Dùng cho cấp độ 1. Thay cho FB15k-237-IMG vì bộ này có sẵn cả truy vấn lẫn đồ thị gán tay; FB15k-237-IMG không có caption, phải tự sinh truy vấn.
  2. **MKG-W** (tải từ kho NativE / MMRNS): truy vấn sinh từ mô tả thực thể + bộ ≥20 câu soạn tay; đồ thị = triple. Dùng cho cấp độ 2.
  3. **MMKG tiếng Việt tự xây**: cấp độ 3.
- **Phương pháp tái lập cấp độ 2:** ưu tiên **NativE (SIGIR 2024)** hoặc **AdaMF-MAT (LREC-COLING 2024)**, chọn theo cái nào chạy lại được trước. Embedding thực thể của nó thay embedding GNN trong điểm đồ thị.
- **Khung so sánh:** trên mỗi tập công khai có 3 hàng (CLIP thuần, CLIP + GNN, CLIP + phương pháp mới). Tối thiểu phải đủ 3 hàng trên MKG-W; ô "phương pháp mới trên Visual Genome" là phần làm thêm.
- **Điểm đồ thị phải mang giải thích ngay từ đầu**: thiết kế theo đường đi truy vấn → khái niệm → ảnh hoặc attention trên cạnh (GAT), để cấp độ 3 đo fidelity bằng cách bỏ đúng cạnh đã dùng. Không dùng cách vẽ đường đi hậu kiểm (đề cấm).
- Đặc trưng CLIP trích một lần, lưu `.npy`; FAISS cho top-k; PyTorch Geometric cho GNN.

## Sản phẩm phải nộp

- Mã nguồn (tiền xử lý, trích đặc trưng, tạo đồ thị, huấn luyện, đánh giá) + `requirements.txt` + hướng dẫn chạy lại.
- Tệp chia dữ liệu hoặc seed + mã tạo split, mô tả chống rò rỉ.
- Báo cáo kỹ thuật 12–18 trang, slide, bảng phân công.
- Notebook hoặc giao diện nhỏ: nhập truy vấn → top-k → hiển thị đường đi giải thích.
- Phân tích ≥5 truy vấn thành công, ≥5 thất bại.
- Cấp độ 2: bộ ≥20 truy vấn + cách xây. Cấp độ 3: MMKG tự xây + data card.
- Demo chạy trực tiếp khi vấn đáp, video dự phòng ≤5 phút. Trình bày ≤20 phút.

## Cài đặt và chạy thử

Cần Python 3.11 hoặc 3.12 (bộ phiên bản trong `requirements.txt` đã kiểm tra trên Ubuntu 24.04 + RTX 4090; máy không có GPU NVIDIA vẫn cài được, chạy bằng CPU).

```bash
python3.12 -m venv ~/venvs/mgrx && source ~/venvs/mgrx/bin/activate
pip install -r requirements.txt
pytest                              # kiểm tra các module dùng chung
python scripts/download_data.py     # tải chú thích Visual Genome + split Karpathy (~170 MB, chưa có ảnh)
python scripts/smoke_test.py        # chạy thử đầu-cuối trên 100 ảnh: join → tải ảnh → CLIP → FAISS → top-k
```

### Chạy platform demo

Cần các file sau (checksum ở `data/MANIFEST.md`). Tất cả nằm trên Drive nhóm, thư mục `DLDT/multimodal-graph-retrieval-xai-data/`, cùng cấu trúc `data/` và `experiments/` như repo; tải về đặt đúng đường dẫn (xem thêm [data/README.md](data/README.md)):

- `data/features/vg_coco_image.npy` và `vg_coco_caption.npy`, mỗi file kèm `.ids.json`: cho chế độ CLIP thuần.
- Thêm cho chế độ **CLIP + Đồ thị**:
  - `data/features/vg_coco_graph_parts.npy` và `.ids.json` (1,2 GB);
  - `data/graphs/vg_coco_scene_graphs.jsonl`;
  - checkpoint `experiments/level1/gat/main_r3_seed0/checkpoint_best.pt` (4,2 MB, SHA-256 `7575951f…991e6d`) cùng file `config.json` đã có sẵn trong git.
- Mô hình spaCy cho parser: `python -m spacy download en_core_web_sm`.
- Ảnh của split muốn xem.

```bash
python scripts/download_images.py --splits test    # 2 138 ảnh, một lần
python app/demo_app.py --split test                 # mở http://localhost:7860
python app/demo_app.py --split test --no-graph      # chỉ CLIP thuần, không cần checkpoint
```

Nhập một câu tiếng Anh rồi bấm **Tìm**, chọn chế độ **CLIP thuần** hoặc **CLIP + Đồ thị**. Bấm vào một ảnh trong lưới để xem:
điểm cuối, điểm CLIP, điểm đồ thị, trọng số a / b / c của câu, đồ thị câu do parser tách ra, và bảng các cặp khớp giữa câu
và ảnh (w, a, sim, đóng góp). Bảng này lấy từ chính lần xếp hạng (`src/explain/graph_explainer.matched_pairs`), không
phải tìm đường đi sau khi đã có kết quả. Nút **Caption ngẫu nhiên của pool** lấy một caption có sẵn đáp án và báo ảnh đúng
đứng hạng mấy.

Chế độ đồ thị (`src/service/graph_mode.py`) lấy top-50 của CLIP, tách câu bằng `sng_parser`, mã hoá ngay bằng CLIP những
nhãn chưa có trong bảng vector, rồi xếp lại bằng mô hình chính `main_r3_seed0`. Đã kiểm tra trên máy không GPU
(`CUDA_VISIBLE_DEVICES=""`): khởi động khoảng 2 giây, mỗi truy vấn dưới 1 giây. Trên val, chế độ này cho top-10 giống hệt
đường đánh giá đã đóng gói (điểm lệch khoảng 1e-4 do mã hoá câu lại lúc chạy). Kịch bản video dự phòng:
[docs/demo-video-script.md](docs/demo-video-script.md).

Quy ước làm việc nhóm: [CONTRIBUTING.md](CONTRIBUTING.md). Format dùng chung giữa các gói: [docs/interfaces.md](docs/interfaces.md). Dữ liệu: [data/README.md](data/README.md).

## Cấu trúc thư mục dự kiến

```
multimodal-graph-retrieval-xai/
├── README.md
├── CLAUDE.md                # quy tắc làm việc trong folder này
├── assignments/                  # 3 đề gốc (PDF), không sửa
├── data/                    # dữ liệu tải về + tập chia (không commit dữ liệu lớn)
├── src/                     # mã nguồn
├── notebooks/               # demo, phân tích
├── experiments/             # log, kết quả theo seed
├── report/                  # báo cáo LaTeX, slide (nội dung tiếng Việt)
└── vietnamese-mmkg/         # MMKG tiếng Việt tự xây (cấp độ 3) + data card
```

## Trạng thái

- 2026-09-09: chốt đề Project 3, chưa bắt đầu code.
- 2026-09-17: nhóm 7 đủ 5 người; chốt 3 bộ dữ liệu (Visual Genome ∩ COCO, MKG-W, MMKG tiếng Việt); lập khung kế hoạch và file phân công. Việc tiếp theo: cả nhóm chọn việc trong `task-assignment.xlsx`, bắt đầu giai đoạn 0.
