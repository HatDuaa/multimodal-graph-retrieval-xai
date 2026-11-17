# Huấn luyện NativE trên máy RTX 4090 (cấp độ 2)

Thực hiện ngày 17/11/2026. Lộc phụ trách máy 4090, theo phương án dự phòng trong `docs/level2-plan.md` (mục 5, việc 2 và 3 của Khoa): đến 12:00 ngày 17/11 vẫn chưa có báo cáo chạy trên molab.

## Chạy gì

| Lần chạy | Dữ liệu | Seed | Dùng cho |
|---|---|---|---|
| `rmask_seed0_e1000`, `rmask_seed1_e1000`, `rmask_seed2_e1000` | `MKG-W-rmask` | 0, 1, 2 | Bước b: xuất embedding cấu trúc (`ent`) cho kênh `emb` |
| `repro_seed42_e1000` | `MKG-W` gốc | 42 | Bước a: so link prediction với bài báo (`../native-repro/README.md`) |

Bốn lần chạy nối tiếp nhau, mỗi lúc chỉ một job GPU, do `run.sh` điều khiển trong một phiên `tmux`. Phiên này vẫn sống khi mất kết nối.

## Môi trường

- **Code:** `zjukg/NativE` commit `c72291c3d571b36c754897935d21718aa4568fe7`, clone vào `~/work/NativE`.
  - Đường dẫn lấy từ `configs/default.yaml` (`mkgw.native_repo`).
  - `mmkgc/release/Base.so` đã được biên dịch lại bằng `cd mmkgc && bash make.sh` vì bản đi kèm repo bị segfault (xem `../native-repro/README.md`). Bản gốc được giữ lại ở `release/Base.so.shipped`.
- **Bảng đặc trưng ảnh và văn bản của tác giả:**
  - lấy từ `MKG.zip` trên Google Drive của tác giả (id `16qQg1NCyHSRaAWjRPpPEFajW2wrhC4aX`), 78.685.701 byte, SHA-256 `6508d4a2fbf263d5…`;
  - chỉ giải nén `MKG-W-visual.pth` (15000 × 383, SHA-256 `9cf9346fb7bcd1d7…`) và `MKG-W-textual.pth` (15000 × 384, SHA-256 `13b765b5bddceaef…`), khớp nhật ký của Đạt. Zip đã xoá sau khi giải nén.
- **`MKG-W-rmask`:** dựng bằng `python scripts/mask_native_modalities.py --name MKG-W-rmask`.
  - 1.784 thực thể của tập val và test retrieval (892 + 892) bị đặt vector về 0.
  - Bảng ảnh: 537 dòng vốn đã bằng 0 (khớp 15000 − 14463 ở Bảng 1 của bài báo), sau khi che có 2.321 dòng bằng 0; đổi 1.784 dòng.
  - Bảng văn bản: 851 dòng vốn đã bằng 0, sau khi che có 2.555 dòng bằng 0; đổi 1.704 dòng, vì 80 thực thể bị che vốn không có văn bản.
- **Môi trường Python:** venv `~/venvs/mgrx`, Python 3.12.3, torch 2.14.0+cu130 (CUDA 13.0). Đã thêm scikit-learn 1.9.1 và gdown 6.5.1.
- **GPU:** NVIDIA GeForce RTX 4090 24 GB, driver 595.99.02. Mỗi lần chạy dùng khoảng 10 GB, tải GPU 87–94%.

## Lệnh

Toàn bộ chuỗi chạy bằng lệnh:

```bash
tmux new -d -s native "bash experiments/level2/native-train/run.sh"
```

Mỗi lần chạy gọi lệnh dưới đây trong `~/work/NativE`, với `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Siêu tham số giữ đúng `scripts/run_mkgw.sh` của tác giả, giống cách `notebooks/mkgw_molab.py` dựng lệnh; giữa các lần chạy chỉ đổi `-dataset` và `-seed`:

```
python -u run_adv_wgan_gp.py -dataset=<MKG-W-rmask|MKG-W> -batch_size=1024 -margin=4 -epoch=1000 -dim=250 \
  -save=./checkpoint/<tag>_seed<s>_e1000.ckpt -neg_num=128 -mu=0.0001 -learning_rate=1e-4 -lrg=1e-4 -seed=<s>
```

Dòng đầu của mỗi log ghi đúng lệnh đã chạy. Sau mỗi lần `rmask`, driver xuất embedding bằng:

```
python scripts/export_native_embeddings.py --checkpoint ~/work/NativE/checkpoint/rmask_seed<s>_e1000.ckpt --what ent --suffix _seed<s>
```

File ra là `data/features/mkgw_native_ent_seed<s>.npy` và `.ids.json`. SHA-256 của chúng ghi trong `data/MANIFEST.md`. File `.npy` và `.ckpt` không commit.

## Kết quả và thời gian

Thời gian lấy từ `driver.log`. Số link prediction lấy từ dòng `averaged(filter)` cuối mỗi log. Số trong `metrics_rmask.json` và `../native-repro/metrics.json` đọc bằng `scripts/parse_native_log.py`.

| Lần chạy | Bắt đầu – kết thúc (17/11) | Thời gian thực | Huấn luyện 1.000 epoch (tqdm) | Giây/epoch | Loss D / G ở epoch cuối | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `rmask_seed0_e1000` | 17:21:12 – 18:27:01 | 3.949 s (1,10 h) | 1:05:26 | 3,93 | 4,78 / 200,05 | 0,3714 | 831,8 | 0,3009 | 0,4041 | 0,4996 |
| `rmask_seed1_e1000` | 18:27:02 – 19:33:11 | 3.969 s (1,10 h) | 1:05:46 | 3,95 | 4,79 / 183,82 | 0,3713 | 859,1 | 0,3000 | 0,4057 | 0,5011 |
| `rmask_seed2_e1000` | 19:33:12 – 20:39:45 | 3.993 s (1,11 h) | 1:06:09 | 3,97 | 4,72 / 97,54 | 0,3671 | 946,7 | 0,2987 | 0,3993 | 0,4911 |
| **rmask, mean ± std (3 seed)** | | | | | | 0,3699 ± 0,0024 | 879,2 ± 60,0 | 0,2998 ± 0,0011 | 0,4030 ± 0,0033 | 0,4973 ± 0,0054 |
| `repro_seed42_e1000` | 20:39:46 – 21:45:59 | 3.973 s (1,10 h) | 1:05:50 | 3,95 | 4,66 / 107,88 | 0,3699 | 862,2 | 0,3017 | 0,3995 | 0,4932 |

Tổng thời gian của chuỗi là 4 giờ 25 phút trên một GPU, từ 17:21 đến 21:46. Std tính với ddof = 1.

Checkpoint lưu tại `~/work/NativE/checkpoint/` (không commit). SHA-256 của chúng ghi trong `data/MANIFEST.md`. Bảng ảnh và bảng văn bản nằm trong checkpoint `rmask` (`img_embeddings.weight`, `text_embeddings.weight`). Hai bảng này trùng từng phần tử với `MKG-W-rmask-{visual,textual}.pth`, đã kiểm tra bằng `torch.equal` trên seed 0. Như vậy mô hình dùng đúng bản đã che val và test.

Cột link prediction là kết quả trên **tập test KGC của chính NativE** (`test2id.txt`, 4.274 triple), thiết lập filtered, không ràng buộc kiểu, đúng như tester của NativE in ra. Đây là một phần của bước a và không đụng tới tập test retrieval.

Các lần `rmask` không so được trực tiếp với bài báo, vì bảng đặc trưng của 1.784 thực thể đã bị đặt về 0.

## Ghi chú

1. **Seed chỉ đổi phần ngẫu nhiên của torch.**
   - `-seed` đặt `torch.manual_seed`, nên chỉ đổi khởi tạo tham số và nhiễu của generator.
   - Bộ lấy mẫu âm viết bằng C (`mmkgc/base/Random.h`, hàm `randReset`) khởi tạo seed cho từng luồng bằng `rand()` mà không gọi `srand`. Vì vậy chuỗi mẫu âm giống hệt nhau ở mọi lần chạy.
   - Do đó độ lệch giữa 3 seed `rmask` chỉ đo ảnh hưởng của khởi tạo và nhiễu GAN, không đo ảnh hưởng của cách lấy mẫu âm.
2. **Tester chỉ in số filtered.** Các dòng in số raw (unfiltered) trong `mmkgc/base/Test.h` đã bị tác giả comment. Bài báo cũng dùng thiết lập filter (mục 5.1.2). `scripts/parse_native_log.py` đọc các dòng `l(filter)` (đoán đầu), `r(filter)` (đoán đuôi) và `averaged(filter)`.
3. **Trong log, thanh tiến độ tqdm dùng ký tự `\r`.** Muốn xem từng epoch thì chạy `tr '\r' '\n' < <log>`. Các dòng `Test Head` / `Test Tail` là của tester tác giả, mỗi triple test một dòng.
4. **Lần đo tốc độ 10 epoch** (rmask seed 0, 3,93 giây/epoch, 62 giây cả nạp dữ liệu và test) chỉ để ước lượng thời gian. Log của lần này không commit.
5. **Việc phụ chạy bằng CPU trong lúc GPU đang huấn luyện** (không đưa vào git vì nằm trong `data/raw` và `data/processed`):
   - `data/processed/mkgw_triples.jsonl` ghi bằng `copy_kgc_triples`, giống notebook molab. File có 42.746 triple, 2.262.289 byte, SHA-256 khớp `data/MANIFEST.md`.
   - **Cache Wikidata lấy lại ngày 17/11/2026, chỉ dùng dự phòng** khi không có bản gốc tháng 9 của nhóm. Bản này **không** ghi vào MANIFEST và không khớp bản tháng 9 (3.564.691 byte, `8e1aa638…`), vì Wikidata đã thay đổi từ đó.
     - `data/raw/mkgw/wikidata_entities.jsonl`: 15.000 dòng, 3.532.544 byte, SHA-256 `dd47e44dbb068c0f209e89bee562efd8c07f7d932aefd6706196585f6718425c`.
     - `data/raw/mkgw/wikidata_relations.json`: 169 quan hệ, 9.452 byte, SHA-256 `f0760499881f04319a32ea936f6e3f8fe65d7860857d93000716fd96755344f8`.
     - Lần lấy đầu bị đứt sau 4.000 thực thể (`RemoteDisconnected`), và lần chạy lại đã chạy tiếp từ chỗ dừng. Script giữ nguyên.
     - 441 thực thể không có `label_en`. Trong đó, 58 dòng rỗng hoàn toàn. Với 383 dòng còn lại, nhiều khả năng nhãn đã chuyển sang mã ngôn ngữ `mul` của Wikidata: đã kiểm tra 3 thực thể và cả 3 đều vậy, nhưng script hiện chỉ đọc `en` và `vi`. Nhóm sẽ quyết định có dùng nhãn `mul` hay không nếu cuối cùng không có bản gốc.

## Phân công

Lộc (máy 4090) chạy với sự hỗ trợ của AI (Claude Code). Phần này được khai báo trong `report/declaration-ai-oss.md`. Gồm:
- `run.sh`, `scripts/parse_native_log.py` và test của nó;
- các log, file này, và phần bổ sung trong `../native-repro/`.
