# Khai báo mã nguồn mở, dữ liệu và công cụ AI (nháp, chờ Lộc duyệt)

Theo mục 9 của đề: khai báo mã nguồn mở và công cụ AI đã dùng, ghi rõ phần đã chỉnh sửa. Bản nháp do Claude soạn ngày
2026-09-26, dựa trên lịch sử git, `~/mgrx-night/SUPERVISOR.md` và các spec trong `local-docs/specs/`. Những chỗ ghi
**[Lộc xác nhận]** là thông tin Claude không kiểm chứng được từ repo.

## 1. Thư viện mã nguồn mở

Dùng qua API, không chép hay sửa mã nguồn của thư viện. Phiên bản ghim trong `requirements.txt`.

| Thư viện | Dùng để | Giấy phép |
|---|---|---|
| PyTorch 2.14, torchvision 0.29 | Mô hình, huấn luyện | BSD-3-Clause |
| PyTorch Geometric 2.8 (`GATv2Conv`, `to_dense_batch`) | Lớp GATv2, gom node theo đồ thị | MIT |
| open_clip 3.3 với trọng số OpenAI CLIP ViT-B/32 (`ViT-B-32-quickgelu`, `openai`) | Mã hoá ảnh và văn bản, đóng băng | MIT (mã); trọng số theo giấy phép của OpenAI CLIP |
| FAISS (faiss-cpu 1.15) | Tìm top-k theo cosine cho baseline và demo | MIT |
| spaCy 3.8 + `en_core_web_sm` | Phân tích cú pháp cho parser | MIT |
| SceneGraphParser 0.1 (`sng_parser`) | Tách caption thành vật thể và quan hệ | MIT **[Lộc xác nhận]** |
| NumPy, pandas, Pillow, PyYAML, requests, tqdm | Xử lý dữ liệu, ảnh, tải file | BSD / MIT / HPND / Apache-2.0 |
| Gradio 6.28 | Giao diện demo | Apache-2.0 |
| pytest | Kiểm thử | MIT |

Nhóm không dùng mã nguồn của bài báo nào ở cấp độ 1. Mô hình xếp lại (`src/models/graph_reranker.py`) do nhóm viết theo
thiết kế riêng (`local-docs/plans/level1-graph-reranker-plan.md`, bản 7).

## 2. Dữ liệu

| Dữ liệu | Dùng để | Giấy phép / nguồn |
|---|---|---|
| Visual Genome 1.4: `objects.json`, `relationships.json`, `attributes.json`, ảnh | Scene graph của ảnh; phép thử thuộc tính | CC BY 4.0 |
| MS-COCO 2014: caption | Truy vấn | CC BY 4.0 (chú thích) |
| Split Karpathy (`dataset_coco.json`) | Chia train / val / test | Công bố cùng bài báo của Karpathy & Fei-Fei |

Link tải và checksum: `data/README.md`, `data/MANIFEST.md`.

## 3. Công cụ AI

Hai công cụ đã dùng: **OpenAI Codex** (qua các session Codex CLI trên máy RTX 4090 của Lộc) và **Anthropic Claude**
(Claude Code, gồm session review "KG (project)" trên máy Windows của Lộc và session làm việc trên máy RTX 4090). Commit
do AI tạo vẫn mang tên tác giả Git "Hạt Bí" (tài khoản của Lộc); từ commit không phân biệt được người hay AI viết, nên
bảng dưới lấy theo nhật ký làm việc.

| Phần việc | Ai / công cụ nào | Người duyệt, chỉnh sửa |
|---|---|---|
| Chọn đề, kế hoạch 9 tuần, chia gói việc (`plan.md`, `README.md`) | Lộc, có AI hỗ trợ soạn **[Lộc xác nhận mức độ]** | Lộc |
| Khung repo, split VG ∩ COCO, trích đặc trưng CLIP, module metric, baseline CLIP, demo CLIP thuần (09/18–19) | **[Lộc xác nhận]**: Lộc tự viết hay có AI | Lộc |
| Kiểm tra pipeline: MSCOCO 5K, cách resize ảnh, lỗi QuickGELU; độ phủ scene graph; phép thử xếp lại không huấn luyện | **[Lộc xác nhận]** | Lộc |
| Thiết kế mô hình cấp độ 1 (plan bản 1–7) | Lộc quyết định từng điểm; một session Codex riêng phản biện bản 1 và bản 4 | Lộc |
| Đóng gói dữ liệu đồ thị, mô hình, vòng huấn luyện, kiểm tra bước 0, explainer (commit `81ed30c`–`d051e3d`) | Codex, theo spec Lộc duyệt (`local-docs/specs/step1-*`, `step2-*`) | Lộc duyệt spec; Claude review |
| Gỡ lỗi crash, micro-batch, phép thử thuộc tính (các commit fix trong dải trên và `2aad92b`) | Codex (session giám sát qua đêm) | Claude review |
| Đường dữ liệu tính sẵn (tăng tốc khoảng 10 lần), checkpoint và resume, driver chạy chuỗi, đo bộ nhớ (`67448ba`–`268b5b6`) | Claude | Lộc duyệt qua session review |
| Báo cáo mô hình theo code (`docs/level1-model.md`) | Claude | Lộc |
| Hai đối chứng phá đồ thị, quét α, huấn luyện trên cạnh nối lại, loss LambdaRank, bảng tổng hợp, chạy 24 run r3 (`7b81561`–`6ada7a3`) | Claude, theo danh sách việc Lộc duyệt | Lộc |
| Chọn mô hình chính (có GAT) trước khi chạy test | **Lộc** | — |
| Đánh giá test một lần (`f738de3`, `8b36243`) | Claude, sau khi Lộc chốt | Lộc |
| Giải thích và 10 ca phân tích, phép thử xoá, đo độ trải vector (`d9eddc3`) | Claude. Ca được chọn bằng quy tắc cố định; lời phân tích viết sau khi xem ảnh thật | Lộc duyệt |
| Chế độ CLIP + Đồ thị trong demo, kịch bản video (`657fea7`) | Claude | Lộc |
| Nháp báo cáo cấp độ 1 (`report/level1-draft.md`) và bản khai báo này | Claude | Lộc duyệt, sửa |

Nguyên tắc áp dụng cho mọi phần do AI làm:
- Số liệu chỉ lấy từ file kết quả có log và seed.
- Không nới lỏng phép kiểm tra nào: kiểm tra bước 0 và các file tham chiếu giữ nguyên.
- Test chỉ chạm một lần.
- Lịch sử việc làm được ghi trong nhật ký (`~/mgrx-night/SUPERVISOR.md`, ngoài repo) và trong commit.

**Việc Lộc cần làm trước khi nộp:**
- Điền các ô **[Lộc xác nhận]**.
- Ghi rõ những đoạn Lộc đã tự sửa lại sau AI.
- Nếu muốn, chép `SUPERVISOR.md` vào phụ lục hoặc vào repo.
