# Manifest dữ liệu

Mỗi file dùng chung ghi một dòng: tên, kích thước, SHA-256, ngày, script sinh ra. Trước khi chạy thực nghiệm, đối chiếu checksum để chắc cả nhóm dùng cùng một bản.

## File tải về (`scripts/download_data.py`)

Tải ngày 2026-09-18.

| File | Bytes | SHA-256 |
|---|---|---|
| `data/raw/visual_genome/image_data.json.zip` | 1780854 | `b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a` |
| `data/raw/visual_genome/objects.json.zip` | 55323929 | `efa456069aedb420b7dabf54f4608c89c78a00de550f1433066e9b6f84fb629c` |
| `data/raw/visual_genome/relationships.json.zip` | 77904473 | `e648867b8087e4aeb10019a959f914e2580909f1c3e50a21b25de8741679182f` |
| `data/raw/visual_genome/object_alias.txt` | 60166 | `0c8e059fc31eeebfd98231f5789892da8ae33bfa00434c70aee969dc6eaa853b` |
| `data/raw/visual_genome/relationship_alias.txt` | 122102 | `15f7f64802c95c5bf5b5690457566b1b19ee64eac7a94d8b8cbb3a4378f2ec7c` |
| `data/raw/vg150/object_list.txt` | 885 | `7f0824cff05b721ff3dd52114c88b683082a4628b3152fb7c222853a9d262404` |
| `data/raw/vg150/predicate_list.txt` | 393 | `d455d4dd841b94e42bfc96bcf7c040ec0efd905e85f51cd7649ab4791e8c9aeb` |
| `data/raw/vg150/object_alias.txt` | 60166 | `0c8e059fc31eeebfd98231f5789892da8ae33bfa00434c70aee969dc6eaa853b` |
| `data/raw/vg150/predicate_alias.txt` | 122102 | `15f7f64802c95c5bf5b5690457566b1b19ee64eac7a94d8b8cbb3a4378f2ec7c` |
| `data/raw/coco/caption_datasets.zip` | 36745453 | `4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817` |

Bốn file `vg150/` là danh sách 150 lớp đối tượng và 50 quan hệ (VG150, Xu và cộng sự 2017) lấy từ repo `danfeiX/scene-graph-TF-release`, tải ngày 2026-09-19. Hai file alias ở đó trùng từng byte với hai file alias của Visual Genome (cùng SHA-256).

Sau khi giải nén, `data/raw/` chiếm khoảng 1,4 GB.

Số đếm thật từ các file này (in ra bởi `scripts/smoke_test.py`): Visual Genome có 108 077 ảnh; 51 498 ảnh có `coco_id` nằm trong file split Karpathy, gồm train 34 027, restval 13 183, val 2 146, test 2 142.

## File nhóm sinh ra

### Đặc trưng CLIP của vg_coco

Trích ngày 2026-09-19 bằng `scripts/extract_features.py` trên RTX 4090, encoder `ViT-B-32-quickgelu/openai` (open_clip 3.3.0, torch 2.14.0), từ đủ 51 208 ảnh (0 ảnh lỗi khi tải).

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/vg_coco_image.npy` | 51 208 × 512 | 104874112 | `091acde31fa1b3e7a24314dab889205c24d4ff0bdf405fb2128d923c87e2a1a9` |
| `data/features/vg_coco_image.ids.json` | — | 459664 | `b82397bb868044a9e7a8dd351b978c73b33cbfbb84d178fb49f0a072c990e25b` |
| `data/features/vg_coco_caption.npy` | 256 183 × 512 | 524662912 | `c116bfe099bdcb33dc5810f829de9080647b972eb53104fa647b2e2e0cb286d3` |
| `data/features/vg_coco_caption.ids.json` | — | 3025464 | `7eb6a7a05dff4007bdd6aa7c2f4b08570f79d095336a932ce5843da92df6cf69` |

Kiểm tra sau khi tải về: `sha256sum data/features/vg_coco_*` (Windows: `certutil -hashfile <file> SHA256`).

### Đặc trưng MSCOCO 5K (chỉ để kiểm tra pipeline, không phải dữ liệu của đề tài)

Sinh ngày 2026-09-19 bằng `scripts/checks/coco5k_zero_shot.py` trên RTX 4090, encoder `ViT-B-32-quickgelu/openai`. Chỉ nằm trên máy chạy, không có trên Drive. Xem `experiments/checks/README.md`.

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/coco5k_image.npy` | 5 000 × 512 | 10240128 | `14e7908d8151ef25128596586efc2652e88eeac7f2c076b99c3176bbd6916a93` |
| `data/features/coco5k_image.ids.json` | — | 39089 | `ed26e1361a67dc470249f4b6885585f9123c6ce991ac1ddac12d76876be455f2` |
| `data/features/coco5k_caption.npy` | 25 010 × 512 | 51220608 | `2b6a18218cbcbf05bb1e1d8fdfcc4ce8cb330b7bc23df99c39ca59ad4aa1a8f5` |
| `data/features/coco5k_caption.ids.json` | — | 295319 | `a563a520c6ae78fd7c793ca61eb6a174afe98ac1440fd5b599424bd905b1a56b` |

### Vector các phần đồ thị của split val (phép thử ở `experiments/checks/`, mục 4)

Sinh bởi `scripts/checks/soft_graph_rerank_probe.py` ngày 2026-09-19; khoá trong `ids` có dạng `node:<nhãn>` hoặc `triple:<cụm bộ ba>`. Chỉ lưu trên máy chạy.

| File | Dòng × chiều | Bytes | SHA-256 |
|---|---|---|---|
| `data/features/vg_coco_val_graph_parts.npy` | 42 736 × 512 | 87523456 | `8c41676c5f228eafbfbf9957768daaf552bf4cfd2f56d1bf819d856cd461adca` |
| `data/features/vg_coco_val_graph_parts.ids.json` | — | 1176825 | `5dc2be646f18a0d11025842e171f56315197e6871bfe5d68a13610e1f31e5543` |

Ghi chú: các file trước đó (encoder `ViT-B-32/openai`) được lưu trữ trên máy GPU tại `data/features/archive-vit-b-32-gelu/`. Ai đã tải feature từ Drive trước thay đổi này cần tải lại.

### Gói dữ liệu re-ranker đồ thị

Sinh bởi `src.data.train_groups` và `scripts/build_graph_data.py`; kích thước và SHA-256 đo trên máy GPU ngày 2026-09-23 (nhiều file trong một dòng thì theo đúng thứ tự tên).

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/splits/vg_coco_train_groups.json` | 703407 | `235ddfdde7c65e711a1e1f34232743cef071b4fca8b977602c41cfc64944547e` |
| `data/graphs/vg_coco_scene_graphs.jsonl` | 91246921 | `1f65f3e24e4f2bcb9e390e1092aa12f764154a398d3e163fffc237b837f41788` |
| `data/processed/query_graphs_{train,val,test}.json` | 110566228<br>4999446<br>5026707 | `fb79548ba4565fa6a7d3f141bebed3edd62f6891e0de1d38c6da14929107bb04`<br>`f2f18a871e9cf001ad8b8ec2712b911cf64cfb5ea148525e08ab35ec83c32412`<br>`e5972020e319891c05253c39bf91ca35b103740bbad7c3e505414745e8b4f84e` |
| `data/features/vg_coco_graph_parts.npy` + `.ids.json` | 1192110208<br>17039232 | `555eebcfef67d57bdd56d4813abfa109186de910e12beb4b7266244ed1adc5d3`<br>`bec0332e783afd5566b5326af2625fe68de3b4e1e246b48361304593c884e929` |
| `data/processed/candidates_train.npz` | 155954210 | `549083b1756df5a467887cc8d860a2d056f6f93878f088c819e0bffb6053f958` |
| `data/processed/candidates_val.npz` | 7064252 | `ec4bc3432594d9923b1de7ac4ff33b1511d334690a8b681768fd9637ede91d79` |
| `data/processed/candidates_test.npz` (`scripts/evaluate_test.py`, từ `top50_test.json` của baseline) | 7018626 | `75448b186b67e6cd98f36fcb418e15aabf7c3e587a6630a131cb9251eb759a86` |
| `data/raw/visual_genome/attributes.json.zip` | 83280561 | `7f71c80fb5396c67a3022e0748de16a3253b7fe5759e151215e0f1a3de3f38c7` |

### Vector bộ ba của đồ thị nối lại cạnh (đối chứng)

Sinh bởi `scripts/build_rewired_parts.py --seed k` (chỉ đồ thị train/val; chỉ gồm cụm bộ ba chưa có trong `vg_coco_graph_parts`), dùng cho `scripts/checks/graph_controls.py --control rewire` và các run `rewire_r3_seed<k>`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/features/vg_coco_graph_parts_rewire_seed0.npy` | 719802496 | `c59e92ccb0313ce19f71166d67f249c36fc17acfc7191e2e3f887ff1700e226a` |
| `data/features/vg_coco_graph_parts_rewire_seed0.ids.json` | 10606326 | `38e5b591d5367d0384128b74b91ecae3dd008ac67a699ec45d36d6da0082d497` |
| `data/features/vg_coco_graph_parts_rewire_seed1.npy` | 721426560 | `7e4b3ca4da67f95440420f3f38473bdb709229c42693d61661b34d08a14c5875` |
| `data/features/vg_coco_graph_parts_rewire_seed1.ids.json` | 10631284 | `1dc918602f801f58c011bdc8feae348886aa026184c670dd0157311812407a7e` |
| `data/features/vg_coco_graph_parts_rewire_seed2.npy` | 721016960 | `fc1c076d410bc1b7ac8a67bfe3f0941459515ea2b18dcd78c767c5540da39e56` |
| `data/features/vg_coco_graph_parts_rewire_seed2.ids.json` | 10622224 | `a3c2ed7fdaf63742a9d013047bac007aee3224828d8a9a9d9a6ab1c53de01a90` |

## MKG-W (gói cấp độ 2, Đạt)

Bản 2026-09-25 (v2 — ảnh lấy từ kho tác giả MMRNS thay vì Wikimedia Commons; bản v1 ngày 24/09 với 5 752 thực thể theo P18 đã bỏ). Cache Wikidata: `scripts/fetch_mkgw_wikidata.py` + `scripts/fetch_mkgw_relations.py`; ảnh: `scripts/extract_mkgw_images.py`; split + triple: `python -m src.data.build_mkgw_split` (seed 0); đặc trưng CLIP: `scripts/extract_features.py --dataset mkgw`. File không commit chia sẻ qua Drive nhóm, cấu trúc thư mục như dưới (riêng `mkgw_img.zip` tải từ Drive của MMRNS, link trong README repo `quqxui/MMRNS`).

| File | Bytes | SHA-256 |
|---|---|---|
| `data/raw/mkgw/wikidata_entities.jsonl` | 3564691 | `8e1aa6387e9a21a5067b577b1d2deae1651232cba54d30f8232685908c689daf` |
| `data/raw/mkgw/ent_links.tsv` (MMRNS) | 1299610 | `4aa7e9f5ff3bbeea7dee3124669edba7903e65b22844529999325c6bec624511` |
| `data/raw/mkgw/mkgw_img.zip` (MMRNS `MKG-W_img.zip`) | 6422519900 | `0178f7bb3b6199309a56fad23c595c252f79506deb5dbecece58d814d5f0aede` |
| `data/raw/mkgw/author_images_index.json` | 624758 | `0727dafebdf3d9c3fd988778e34f574d324eb3bf9dd9e144ce15c19631dd45bd` |
| `data/splits/mkgw_train.json` | 2020637 | `d731488f6618bf24937457f6deee75d849ab79bf20ec074f4ca56795264ba89a` |
| `data/splits/mkgw_val.json` | 251533 | `4762f9c0731ffcc164baf8642a130b0e30d478f5998f812be4c3414646a2c0b2` |
| `data/splits/mkgw_test.json` | 254576 | `6416ff37d6ab1e0e921fb21a74655e3a2654a72d497eaf07151bd6ab3c198886` |
| `data/processed/mkgw_triples.jsonl` | 2262289 | `40ccfa72ce54d9d2c6a23e9ed50ea571255425e20cf1b0cf6c2c5e2e3cecbedb` |
| `data/features/mkgw_image.npy` (8 920 × 512) | 18268288 | `fe9739e40478cf58e87e6a720ceba9bf3ec4deee9bef9e92107a99f7ba2da827` |
| `data/features/mkgw_caption.npy` (8 920 × 512) | 18268288 | `59de0cf337a14ae3c468131bb9ca6952d85d3f4be577f971303a2c042de17d2d` |

Ảnh chuẩn hoá `data/raw/images/<qid>.jpg` (8 961 file, 476 MB, JPEG cạnh dài 640) sinh lại được từ zip + index nên không cần checksum từng file. Embeddings gốc của NativE (tải từ Drive tác giả, đặt trong clone NativE): `MKG-W-visual.pth` sha256 đầu `9cf9346fb7bcd1d7`, `MKG-W-textual.pth` sha256 đầu `13b765b5bddceaef`.

### NativE trên máy 4090 (cấp độ 2, 17/11/2026)

Lộc chạy các bước này; lệnh và log ở `experiments/level2/native-train/`.
- Hai bảng đặc trưng của tác giả lấy từ `MKG.zip` (78 685 701 byte, SHA-256 `6508d4a2fbf263d5bd9ff2eb73f6d5ea90767516e0fa6e6d247feaf68b9812c4`), chỉ giải nén phần MKG-W.
- Hai bảng `MKG-W-rmask-*.pth` sinh bằng `python scripts/mask_native_modalities.py --name MKG-W-rmask`.
- Embedding cấu trúc xuất bằng `scripts/export_native_embeddings.py --what ent --suffix _seed<s>` từ checkpoint `rmask_seed<s>_e1000.ckpt`.
  - Mỗi file gồm 15 000 × 500 số float32, đã chuẩn hoá L2, xếp theo thứ tự `entity2id`.
  - Ba file `.ids.json` có cùng danh sách QID. Chúng chỉ khác nhau ở trường `encoder`, nơi ghi tên checkpoint.

| File | Bytes | SHA-256 |
|---|---:|---|
| `~/work/NativE/embeddings/MKG-W-visual.pth` (15 000 × 383) | 22980728 | `9cf9346fb7bcd1d7027a35a8e8bc637ca42f38b72e48f11495ef8c9c9d880afc` |
| `~/work/NativE/embeddings/MKG-W-textual.pth` (15 000 × 384) | 23040760 | `13b765b5bddceaefeebe1ce5b2e99e4b6e0cffe2539a3d2490adcfefde012a2e` |
| `~/work/NativE/embeddings/MKG-W-rmask-visual.pth` | 22981622 | `39d7ef7d2b93c486996465f2cfb6f4d353cb097fcd40aa9488c02ee9725f5a45` |
| `~/work/NativE/embeddings/MKG-W-rmask-textual.pth` | 23041661 | `5d16ccb390db8a6acd0519954345342aaa00e124877ae44127409ff1f9ac8b1a` |
| `data/features/mkgw_native_ent_seed0.npy` | 30000128 | `2e64d98d442a041f6a7a64dff8b4dbd7d1e40800410bb77026841e0fdd0155b5` |
| `data/features/mkgw_native_ent_seed0.ids.json` | 173451 | `a882daf25a59de5cef0cb1f4597db152037ac7f64312e50e7ee50a4e8e69b622` |
| `data/features/mkgw_native_ent_seed1.npy` | 30000128 | `46b22cac34db88173a0ae313b56ba1d5736189a0eb3385a6e3ef091bdb5a093a` |
| `data/features/mkgw_native_ent_seed1.ids.json` | 173451 | `6266f0815d0b3a9c9e75c3ae1ba5c98d3a328ec83d618871e588a31cb8b01dcf` |
| `data/features/mkgw_native_ent_seed2.npy` | 30000128 | `5f8730e609c6d64f9bcc0fae4a408da1614692e440c41393e117dcb65013295a` |
| `data/features/mkgw_native_ent_seed2.ids.json` | 173451 | `0ae64f19e9e18a23c66fba6cafa3dd1c882e212f7ac4f4abc099aac0e6cd1378` |
| `~/work/NativE/checkpoint/rmask_seed0_e1000.ckpt` | 80745984 | `562be25df84e2986a6007351d781aa09241835a2c47739a75c6cca139d60965b` |
| `~/work/NativE/checkpoint/rmask_seed1_e1000.ckpt` | 80745984 | `d83f2909b1c79f511020d142812e5da31dfae68600f689f6b439a531fa8db3df` |
| `~/work/NativE/checkpoint/rmask_seed2_e1000.ckpt` | 80745984 | `07095b5ca98a4b764e83df078e22c6604d8e5efa49e7f826f3d4708b427504bb` |
| `~/work/NativE/checkpoint/repro_seed42_e1000.ckpt` (MKG-W gốc, bước a) | 80746013 | `43f2aa8da52da18e8a5cdfa16f0793373b42db90a6f7a48008ca64fa074ed62e` |

Checkpoint nằm trên máy 4090. Ba checkpoint `rmask` có thêm một bản trên máy Lộc, vì kênh `kgc` cần nạp chúng khi đánh giá. Embedding của lần chạy `repro` không được xuất, vì bước b chỉ dùng các lần chạy `rmask`.

### Bản dựng lại, dùng cho mọi kết quả cấp 2 đã báo cáo (17/11/2026)

Bản gốc của đặc trưng CLIP và cache Wikidata (bảng đầu mục này) không có trên Drive nhóm, nên được dựng lại bằng chính các script ở trên. Kết quả val/test trong `experiments/level2/` dùng **các file dưới đây**, không dùng bản gốc. Chi tiết kiểm tra ở `experiments/level2/val/README.md`.

- `mkgw_img.zip` và `ent_links.tsv` tải lại, khớp SHA bảng trên.
- `author_images_index.json` khớp SHA bảng trên sau khi đổi CRLF thành LF, nghĩa là các ảnh được chọn giống hệt bản gốc.
- Đặc trưng CLIP tính lại trên RTX 3070. SHA khác bản gốc do sai số số học giữa các GPU. Trên val, R@1 của CLIP vẫn là 24,33, đúng bằng số đã commit.
- Cache Wikidata lấy lại ngày 17/11 (Wikidata đã sửa từ tháng 9). Độ phủ liên kết thực thể trên val:
  - truy vấn có ít nhất một thực thể được liên kết: 703, bản gốc 704;
  - đáp án có cạnh trực tiếp tới thực thể đó: 407, bản gốc 403.

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/features/mkgw_image.npy` (8 920 × 512) | 18268288 | `f4de9e4ee51a6e2557e3fd1c92fcea0c5f62cd307794075e4aeb718db60b7a90` |
| `data/features/mkgw_image.ids.json` | 55893 | `1c893c30618da232ab69562f0e353d660809b49cc690f4b7eb1e7c56e26e4095` |
| `data/features/mkgw_caption.npy` (8 920 × 512) | 18268288 | `ea8679cbd5402e68abeadc15164fab020513e6507a57458c3d9cee6c279e2667` |
| `data/features/mkgw_caption.ids.json` | 120869 | `8cc753a267a938a07610fba144e148e09067439cc1d5a58ffb90c83276c9d722` |
| `data/raw/mkgw/wikidata_entities.jsonl` | 3532544 | `dd47e44dbb068c0f209e89bee562efd8c07f7d932aefd6706196585f6718425c` |
| `data/raw/mkgw/wikidata_relations.json` | 9452 | `f0760499881f04319a32ea936f6e3f8fe65d7860857d93000716fd96755344f8` |

## MMKG tiếng Việt (cấp độ 3, Khang)

Bản 2026-10-26, cache thô tải ngày 2026-10-24.
- Đóng gói bằng `scripts/archive_vnmmkg_raw.py`, chia sẻ qua Drive (link trong `data/README.md`, mục MMKG tiếng Việt) và Hugging Face (`duongkhang2100/graphml-vietnamese-mmkg`).
- Dựng bằng `python -m src.data.build_vnmmkg` (seed 0).

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/raw/vnmmkg_raw_2026-10-24.zip` | 137913533 | `7215a609f957fa1eaa6f350aa784898b6f72872a214933e5df61ce2c38c9f506` |
| `vietnamese-mmkg/entities.jsonl` | 54729 | `743d99fec526d33c7365b0a0aaa6a68d41ff32948fc2547c353116f6ef7850fb` |
| `vietnamese-mmkg/triples.jsonl` | 40181 | `06391fb60b3a56cc88749608d7b803e3b005999aba2c00cb4434a69fe950c1ba` |
| `vietnamese-mmkg/images.jsonl` | 223663 | `48174233fd9067300238229458b649589709b40295f190d3fc0b4804ab9cd776` |
| `vietnamese-mmkg/queries_auto.jsonl` | 55229 | `0f5e239e84c57ccedad73025f40430c696476358fb5649ab8eb12f637b3d9fc8` |
| `vietnamese-mmkg/queries_hand.json` | 7930 | `fe311fa0bc44aa7f8e237fe33b33d01e7a8b82b0d434c7c73fe04ee116bb8c1e` |
| `data/splits/vnmmkg_val.json` | 86978 | `910847b4000eb65a05223a9a7194131bfa0d8f40a695650edc9720caf0963508` |
| `data/splits/vnmmkg_test.json` | 88504 | `56d939c13461187204b860a2ed99bb1fbd430bde2be9f37be8c11364dd654574` |

Ảnh `data/raw/vnmmkg/images/<image_id>.jpg` sinh lại từ zip bằng lệnh build; SHA-256 từng ảnh nằm trong `images.jsonl`.
