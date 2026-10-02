# Khai báo mã nguồn mở, dữ liệu và công cụ AI

Theo mục 9 của đề: khai báo mã nguồn mở và công cụ AI đã dùng, ghi rõ phần đã chỉnh sửa. Claude soạn ngày 2026-09-26 dựa
trên lịch sử git, nhật ký làm việc và các spec trong `local-docs/specs/`. Phần phân công giữa người và AI do Lộc xác nhận
cùng ngày.

## 1. Thư viện mã nguồn mở

Dùng qua API, không chép hay sửa mã nguồn của thư viện. Phiên bản ghim trong `requirements.txt`.

| Thư viện | Dùng để | Giấy phép |
|---|---|---|
| PyTorch 2.14, torchvision 0.29 | Mô hình, huấn luyện | BSD-3-Clause |
| PyTorch Geometric 2.8 (`GATv2Conv`, `to_dense_batch`) | Lớp GATv2, gom node theo đồ thị | MIT |
| open_clip 3.3 với trọng số OpenAI CLIP ViT-B/32 (`ViT-B-32-quickgelu`, `openai`) | Mã hoá ảnh và văn bản, đóng băng | MIT (mã); trọng số theo giấy phép của OpenAI CLIP |
| FAISS (faiss-cpu 1.15) | Tìm top-k theo cosine cho baseline và demo | MIT |
| spaCy 3.8 + `en_core_web_sm` | Phân tích cú pháp cho parser | MIT |
| SceneGraphParser 0.1 (`sng_parser`) | Tách caption thành vật thể và quan hệ | MIT |
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

Toàn bộ phần cấp độ 1 trong repo do **Nguyễn Đình Lộc** làm cùng hai công cụ AI. Phần này gồm: chia tập, trích đặc trưng
CLIP, baseline, demo CLIP thuần (18–19/09), dựng đồ thị, mô hình, huấn luyện, đánh giá, giải thích, demo đồ thị và báo cáo
nháp. Các thành viên khác chưa đóng góp vào phần cấp độ 1.

- **Anthropic Claude** (Claude Code): lên thiết kế cùng Lộc, viết spec cho Codex, review code và kết quả, chạy thực nghiệm,
  và viết một phần code (đường dữ liệu tính sẵn, checkpoint và resume, driver, đối chứng phá đồ thị, quét α, đánh giá test,
  giải thích và 10 ca, chế độ đồ thị trong demo) cùng tài liệu và báo cáo nháp.
- **OpenAI Codex** (model gpt-6-astra, qua Codex CLI): viết phần lớn code theo spec đã duyệt. Phần này gồm đóng gói dữ
  liệu đồ thị, mô hình, vòng huấn luyện, kiểm tra bước 0, gỡ lỗi crash và phép thử thuộc tính.
- **Lộc** chốt thiết kế (plan bản 1–7 và mọi lựa chọn trong quá trình làm, trong đó có chọn mô hình chính trước khi chạy
  test), duyệt spec, code và kết quả, và chịu trách nhiệm mọi quyết định.

Commit do AI tạo vẫn mang tên tác giả Git "Hạt Bí" (tài khoản của Lộc). Bảng dưới ghi công cụ chính của từng phần theo
nhật ký làm việc.

| Phần việc | Công cụ AI chính | Lộc |
|---|---|---|
| Chọn đề, kế hoạch 9 tuần, chia gói việc (`plan.md`, `README.md`) | Claude hỗ trợ soạn | Quyết định, duyệt |
| Khung repo, split VG ∩ COCO, trích đặc trưng CLIP, module metric, baseline CLIP, demo CLIP thuần (18–19/09) | Claude và Codex | Chốt cách làm, duyệt |
| Kiểm tra pipeline (MSCOCO 5K, cách resize ảnh, lỗi QuickGELU), độ phủ scene graph, phép thử xếp lại không huấn luyện | Claude và Codex | Chốt, duyệt |
| Thiết kế mô hình cấp độ 1 (plan bản 1–7) | Claude lên phương án; một session Codex riêng phản biện bản 1 và bản 4 | Chốt từng điểm |
| Đóng gói dữ liệu đồ thị, mô hình, vòng huấn luyện, kiểm tra bước 0, explainer (commit `81ed30c`–`d051e3d`) | Codex theo spec | Duyệt spec; Claude review |
| Gỡ lỗi crash, micro-batch, phép thử thuộc tính (các commit fix trong dải trên và `2aad92b`) | Codex (session giám sát qua đêm) | Duyệt; Claude review |
| Đường dữ liệu tính sẵn (tăng tốc khoảng 10 lần), checkpoint và resume, driver chạy chuỗi, đo bộ nhớ (`67448ba`–`268b5b6`) | Claude | Duyệt |
| Báo cáo mô hình theo code (`docs/level1-model.md`) | Claude | Duyệt |
| Đối chứng phá đồ thị, quét α, huấn luyện trên cạnh nối lại, loss LambdaRank, bảng tổng hợp, chạy 24 run r3 (`7b81561`–`6ada7a3`) | Claude | Duyệt danh sách việc và kết quả |
| Chọn mô hình chính (có GAT) trước khi chạy test | — | **Lộc quyết định** |
| Đánh giá test một lần (`f738de3`, `8b36243`) | Claude, sau khi Lộc chốt | Duyệt |
| Giải thích và 10 ca phân tích, phép thử xoá, đo độ trải vector (`d9eddc3`) | Claude; ca chọn bằng quy tắc cố định, lời phân tích viết sau khi xem ảnh thật | Duyệt |
| Chế độ CLIP + Đồ thị trong demo, kịch bản video (`657fea7`) | Claude | Duyệt; tự quay video |
| Nháp báo cáo cấp độ 1 và bản khai báo này | Claude | Duyệt, sửa |

Nguyên tắc áp dụng cho mọi phần do AI làm:
- Số liệu chỉ lấy từ file kết quả có log và seed.
- Không nới lỏng phép kiểm tra nào: kiểm tra bước 0 và các file tham chiếu giữ nguyên.
- Test chỉ chạm một lần.
- Mọi thay đổi đi qua commit và pull request có người duyệt.
