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

Sinh ngày 2026-09-24. Cache Wikidata: `scripts/fetch_mkgw_wikidata.py`; split + triple: `python -m src.data.build_mkgw_split` (seed 0). Hai file không commit (`wikidata_entities.jsonl`, `mkgw_triples.jsonl`) chia sẻ qua Drive, cấu trúc thư mục như dưới.

| File | Bytes | SHA-256 |
|---|---|---|
| `data/raw/mkgw/wikidata_entities.jsonl` | 3564691 | `8e1aa6387e9a21a5067b577b1d2deae1651232cba54d30f8232685908c689daf` |
| `data/splits/mkgw_train.json` | 1576400 | `60df300d9c2253626a70d9459683aa0f096d9243d58c27589cdd12a5fbaa9ee0` |
| `data/splits/mkgw_val.json` | 196433 | `14558c9d5212ddeb23d851f82f1815144ecfb75d79254270096c7e79d7108b50` |
| `data/splits/mkgw_test.json` | 198249 | `0b171d52587730bb1993d1248a283b2f470a526ba32d5fe3a9baab7d9ea82b11` |
| `data/processed/mkgw_triples.jsonl` | 2262289 | `40ccfa72ce54d9d2c6a23e9ed50ea571255425e20cf1b0cf6c2c5e2e3cecbedb` |

Embeddings gốc của NativE (tải từ Drive tác giả, đặt trong clone NativE): `MKG-W-visual.pth` sha256 đầu `9cf9346fb7bcd1d7`, `MKG-W-textual.pth` sha256 đầu `13b765b5bddceaef`.
