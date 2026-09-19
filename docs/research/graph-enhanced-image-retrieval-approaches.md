# Khảo sát: dùng đồ thị để cải thiện và giải thích truy vấn văn bản → ảnh

> **Nguồn gốc và mức tin cậy.** Bản khảo sát này do một agent AI tra cứu web và viết ngày 2026-09-19, sau đó được soát lại một lượt. Dùng để định hướng thiết kế, **không trích thẳng vào báo cáo**: trước khi trích bài nào phải mở bài gốc kiểm tra tên, tác giả, nơi đăng và con số.
>
> **Đã sửa khi soát lại** (theo hiểu biết của người soát, vẫn nên mở bài gốc xác nhận):
> - SGRAF là *Similarity Reasoning and Filtration for Image-Text Matching* (Diao và cộng sự, AAAI 2021), không phải "Scene Graph Assisted Learning". Bài này dựng đồ thị trên các vector độ tương đồng vùng ảnh–từ, không dùng scene graph gán nhãn.
> - VSRN dựng đồ thị trên các **vùng ảnh phát hiện được** rồi chạy GCN; phía văn bản mã hoá bằng GRU, không phân tích caption thành đồ thị.
> - VQA-GNN là của Yanan Wang và cộng sự (ICCV 2023), không phải "Lei".
>
> **Điểm yếu còn lại của bản khảo sát.** (1) Lập luận bác phương án A chủ yếu dựa trên các bài re-ranking bằng đồ thị k-NN, vốn là một thứ khác với đồ thị khái niệm chung mà nhóm cân nhắc, nên chưa thuyết phục. (2) Một số link trỏ tới trang tổng hợp (emergentmind, researchgate) thay vì nguồn gốc. (3) Các con số cải thiện ghi trong mục 3 chưa được đối chiếu với bài gốc. (4) Mục 5.2 là phác thảo của agent, chưa phải thiết kế đã chốt của nhóm.

## 1. Tóm tắt

- **Xu hướng chính**: Hai hướng tiếp cận chính là (B) **per-image scene graphs** từ ảnh, được match với truy vấn được parse thành đồ thị; và (A) **global heterogeneous graphs** kết hợp ảnh, concept nodes và ConceptNet. Ít hơn là (C) hybrid với triple nodes.
- **Phổ biến nhất**: Per-image scene graphs (B) được sử dụng bởi VSRN, SGRAF, SGM, GSMN, SPAN, SEMScene. Đây là lựa chọn được chứng minh nhất.
- **Gần với đề tài**: (C) hybrid có tiềm năng nhưng chưa được nghiên cứu kỹ. Hầu hết công bố chọn (B) vì tính clarity. (A) global được dùng trong knowledge-enhanced retrieval nhưng ít khi giải thích từ cơ chế xếp hạng.
- **Giải thích từ cơ chế**: Chỉ một số công bố (SGM, VSRN, Structure-CLIP) tập trung vào việc giải thích đến từ chính cơ chế matching graph. Hầu hết dùng matched sub-graphs hay paths như explanations.
- **Ablation qua edges**: K-reciprocal matching, diffusion-based re-ranking và edge importance (fidelity/sparsity metrics) được dùng nhưng chủ yếu cho re-ranking, không phải core scoring.
- **Bag-of-words weakness**: NegCLIP/ARO (ICLR 2023) và Structure-CLIP (AAAI 2024) đều chứng minh CLIP bị bag-of-words, và giải pháp của họ dùng scene graph hoặc hard negatives để fix.
- **ConceptNet + VG**: Một số công bố (Knowledge Aware Semantic Concept Expansion, VQA-GNN) kết hợp ConceptNet với scene graphs nhưng chủ yếu cho VQA, không text-to-image retrieval.
- **Text parser**: Stanford Scene Graph Parser là tiêu chuẩn, nhưng SceneGraphParser (Python, GitHub) và LLM-based parsers mới hơn cũng tồn tại.

---

## 2. Bảng tổng hợp

| Bài báo | Năm/Nơi | Loại Đồ thị | Cách tính điểm | Giải thích | Code | Link |
|--------|---------|-----------|---------------|----------|------|------|
| Visual Semantic Reasoning for Image-Text Matching (VSRN) | ICCV 2019 | B | GCN aggregation + cosine sim trên embeddings | Sub-graph từ matching | Public | https://github.com/KunpengLi1994/VSRN |
| Cross-modal Scene Graph Matching (SGM) | WACV 2020 | B | Multi-modal GCN encoder, dual-graph match | Node/relation-level sim | Public | https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_Cross-modal_Scene_Graph_Matching_for_Relationship-aware_Image-Text_Retrieval_WACV_2020_paper.pdf |
| Graph Structured Network (GSMN) | CVPR 2020 | B | Graph encoder trên scene graphs + similarity | Implicit từ matching | Public | https://github.com/CrossmodalGroup/GSMN |
| Consensus-Aware Visual-Semantic Embedding (CVSE) | ECCV 2020 | A | Concept co-occurrence graph + consensus CAC | Concept correlations | Public | https://github.com/BruceW91/CVSE |
| ERNIE-ViL | AAAI 2021 | B+A | Scene graph + entity prediction tasks | Entity predictions | Partial | https://www.researchgate.net/publication/363399592_ERNIE-ViL_Knowledge_Enhanced_Vision-Language_Representations_through_Scene_Graphs |
| Similarity Reasoning and Filtration for Image-Text Matching (SGRAF) | AAAI 2021 | B (đồ thị trên vector tương đồng, không phải scene graph) | Suy luận đồ thị trên các vector tương đồng vùng–từ | Không | chưa kiểm chứng | https://ojs.aaai.org/index.php/AAAI/article/view/17281 |
| Structure-CLIP | AAAI 2024 | B | Scene graph embeddings + Knowledge-Enhanced Encoder | Learned representations | Partial | https://arxiv.org/abs/2305.06152 |
| When and why vision-language models behave like bags-of-words (NegCLIP/ARO) | ICLR 2023 | Benchmark | Hard negative mining + CLIP fine-tuning | ARO test suite | Public | https://arxiv.org/pdf/2210.01936 |
| Winoground: Visio-Linguistic Compositionality | 2022 | Benchmark | Cross-modal matching on 400 carefully curated pairs | Human composition test | Public | https://www.emergentmind.com/papers/2204.03162 |
| Learning Similarity between Scene Graphs and Images (SPAN) | 2023 | B | Transformer-based graph-image similarity | Transformer attention weights | Public | https://arxiv.org/pdf/2304.00590 |
| Semantic-Consistency Enhanced Multi-Level Scene Graph Matching (SEMScene) | 2024 | B | Three-level matching (node/triplet/graph) | Multi-level structures | Partial | https://dl.acm.org/doi/10.1145/3664816 |
| Composing Object Relations and Attributes (CORA) | CVPR 2024 | B | Dual GAT for object-attribute + object-object | Graph attention mechanisms | Public | https://mlanthology.org/cvpr/2024/pham2024cvpr-composing/ |
| Knowledge Aware Semantic Concept Expansion | IJCAI 2019 | A | Concept co-occurrence + ConceptNet | ConceptNet paths | Unclear | https://www.ijcai.org/proceedings/2019/0720.pdf |
| Understanding Image Retrieval Re-Ranking: GNN Perspective | 2020 | A | GCN message passing over k-NN graph | k-reciprocal edges | Partial | https://arxiv.org/pdf/2012.07620 |
| Graph Convolution Based Efficient Re-Ranking | 2023 | A | GCN feature propagation over k-NN | Neighbor affinity | Public | https://arxiv.org/abs/2306.08792 |
| VQA-GNN: Reasoning with Multimodal Knowledge via GNN | - | A+C | ConceptNet + scene graphs + GNN | Knowledge paths | Partial | https://arxiv.org/pdf/2205.11501 |
| Visual Genome | 2016 | Dataset | Benchmark scene graph dataset | Objects, attributes, relations | Public | https://arxiv.org/pdf/1602.07332 |
| Stanford Scene Graph Parser | - | Tool | Dependency parsing + rule-based extraction | N/A | Public | https://nlp.stanford.edu/software/scenegraph-parser.shtml |
| SceneGraphParser | - | Tool | Python toolkit, spaCy-based | N/A | Public | https://github.com/vacancy/SceneGraphParser |
| Image Synthesis with Graph Conditioning (2024) | 2024 | B | CLIP guidance + diffusion + graph embedding | Graph-to-image alignment | Public | https://arxiv.org/pdf/2401.14111 |
| Knowledge Graphs Meet Multi-Modal Learning: Survey | 2024 | Survey | Multi-modal KG embedding + retrieval | Multiple approaches | Survey | https://arxiv.org/pdf/2402.05391 |
| Heterogeneous Graph Fusion Network | 2024 | A | Heterogeneous GNN embedding visual-textual | Fusion network scores | Partial | https://www.sciencedirect.com/science/article/abs/pii/S0957417424007085 |
| Cluster-Aware Similarity Diffusion | 2024 | A | Diffusion-based similarity propagation | Cluster memberships | Public | https://arxiv.org/pdf/2406.02343 |

---

## 3. Chi tiết từng nhóm hướng đi

### 3.1. Per-image scene graphs matching (Option B)

**VSRN (Visual Semantic Reasoning for Image-Text Matching), Kunpeng Li et al., ICCV 2019**

Dựng đồ thị quan hệ giữa các vùng ảnh phát hiện được (bounding boxes); phía văn bản mã hoá bằng GRU, không phân tích caption thành đồ thị. Dùng Graph Convolutional Network để tính toán feature với semantic relationships. GCN aggregates information từ neighbors, tạo ra visual representation với global semantic concepts. Sau đó đơn giản dùng inner product để tính similarity. Cải thiện SCAN 6.8% trên retrieval trên COCO, 12.6% trên Flickr30k. Code public. **Option B**.

**Cross-modal Scene Graph Matching (SGM), Wang et al., WACV 2020**

Chuyển đổi images và captions thành scene graphs (VSG, TSG). Dùng Multi-modal Graph Convolutional Network (MGCN) encoder cho mỗi graph. MGCN refines node representations bằng cách aggregate neighborhood information, update object và relationship features khác nhau. Matching ở object-level và relationship-level. SOTA on Flickr30k, COCO. **Option B**.

**Graph Structured Network (GSMN), Liu et al., CVPR 2020**

Graph-based encoder cho image-text matching. Xây dựng graphs từ visual region proposals và text tokens. Dùng graph encoder (GCN-like) để tính toán similarities. Kết quả SOTA. Implementation public. **Option B**.

**ERNIE-ViL, Yu et al., AAAI 2021**

Knowledge-enhanced vision-language model dùng scene graphs. Hai-stream architecture: một stream cho visual scene graphs, một cho text. Incorporate scene graph prediction tasks vào pretraining. Hybrid approach (B+A). **Option B + external knowledge**.

**SGRAF (Similarity Reasoning and Filtration for Image-Text Matching), Diao et al., AAAI 2021**

Dựng đồ thị trên các vector độ tương đồng giữa vùng ảnh và từ, suy luận trên đồ thị đó rồi lọc bớt các cặp kém liên quan. Không dùng scene graph gán nhãn. Chi tiết chưa kiểm chứng từ bài gốc. **Gần Option B**.

**Structure-CLIP, Huang et al., AAAI 2024**

Extends CLIP với scene graph knowledge. Dùng scene graphs để guide negative example construction (hard negatives), và Knowledge-Enhanced Encoder (KEE) để leverage scene graphs như input. Achieves SOTA on VG-Attribution (12.5% ahead) và VG-Relation datasets (4.1% ahead). Directly addresses bag-of-words weakness. **Option B with structured representations**.

**SPAN (Learning Similarity between Scene Graphs and Images with Transformers), 2023**

Transformer-based approach để match scene graphs với images. Converts both into structured representations và dùng transformer attention để compute similarity. Public code. **Option B**.

**SEMScene (Semantic-Consistency Enhanced Multi-Level Scene Graph Matching), 2024**

Three-level matching strategy: node-level (objects), semantic triplet level (subject-predicate-object), graph-level (holistic). Dùng semantic consistency để address heterogeneity. Published in ACM TMM, 2024. **Option B with multi-granularity**.

**CORA (Composing Object Relations and Attributes for Image-Text Matching), CVPR 2024**

Replaces sequence models với Graph Attention Networks (GAT) để produce scene graph embeddings. Hai-step: GAT cho object-attribute composition + GAT cho object-object relational modeling. Public code. **Option B**.

---

### 3.2. Global / Knowledge-graph approaches (Option A)

**Consensus-Aware Visual-Semantic Embedding (CVSE), Wang et al., ECCV 2020**

Dùng concept co-occurrence correlation graph từ image captioning corpus. Constructs consensus information shared between visual và textual modalities (CAC - consensus-aware concept representations). Graph itself là concept correlations. Public code. **Option A: shared concept layer**.

**Knowledge Aware Semantic Concept Expansion, IJCAI 2019**

Leverages ConceptNet để expand semantic concepts cho image-text matching. Dùng knowledge graph để improve concept understanding. ConceptNet contains over 30 relation types, visual relations như LocatedNear, AtLocation, UsedFor. **Option A: external KG**.

**Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective, 2020**

Reformulates k-NN based re-ranking as GCN problem. K-reciprocal nearest neighbors defined as gallery images that are both in k-nearest neighbors of query AND have query in their k-nearest neighbors. Can be viewed as highly relevant candidates. Graph diffusion methods propagate similarity through k-NN graph. **Option A: post-retrieval re-ranking graph**.

**Graph Convolution Based Efficient Re-Ranking (GCR), 2023**

GCN-based feature propagation trên k-NN graph. Efficiently refines affinities so items in same manifold are similar. SOTA on image retrieval, person Re-ID, video Re-ID. **Option A: post-retrieval re-ranking**.

**VQA-GNN, Yanan Wang et al., ICCV 2023**

Combines per-image scene graphs với ConceptNet. Fuses triplets từ ConceptNet và Comet vào contextual knowledge graph. Multi-hop graph reasoning selects top-K knowledge items. **Option C: per-image + external KG**.

**Heterogeneous Graph Fusion Network, 2024**

Jointly models visual và textual graphs into unified heterogeneous representation graph. Heterogeneous GNN dùng để embed both modalities. SOTA on Flickr30K, MSCOCO. **Option A: heterogeneous**.

**Cluster-Aware Similarity Diffusion, 2024**

Diffusion-based similarity propagation on instance graph. Mentions multiple re-ranking approaches: Query Expansion, Diffusion, Context-based, Learning-based. **Option A: post-retrieval diffusion**.

---

### 3.3. Structural fixes cho bag-of-words behavior (CLIP improvements)

**When and why vision-language models behave like bags-of-words, Yuksekgonul et al., ICLR 2023**

Introduces ARO benchmark (Attribution, Relation, Order) with 50,000+ test cases từ Visual Genome, COCO. Demonstrates CLIP acts as bag-of-words: ví dụ "grass eating horse" ranked higher than correct caption in 81% cases. NegCLIP fine-tunes CLIP với POS-tag-based hard negatives để fix. **Does NOT use graphs, uses negatives instead, but identifies exact CLIP weakness team observed**.

**Winoground: Probing Vision and Language Models, Thrush et al., 2022**

Visio-Linguistic Compositionality benchmark. 400 carefully curated pairs: same words, different order. CLIP, UNITER, FLAVA achieve 30-35%, far below human 85-90%. Demonstrates compositional reasoning failure. **Documents the weakness, no graph solution**.

**Structure-CLIP, Huang et al., AAAI 2024**

Explicitly fixes bag-of-words problem using scene graphs. Generates hard negatives từ scene graph modifications. Knowledge-Enhanced Encoder incorporates SGK. Results: 12.5% on VG-Attribution, 4.1% on VG-Relation ahead of prior SOTA. **Option B: scene graph-based structured fix**.

---

### 3.4. Explainability with graphs

**Understanding Image Retrieval Re-Ranking: Graph perspective (2020)**

Explains re-ranking through k-reciprocal graph: explanations are the edge sets (which k-nearest neighbors enabled re-ranking). **Explanations come from graph structure itself**.

**Graph Neural Network Explainability Survey (multiple papers 2022-2024)**

Fidelity: whether explanations are faithfully important (remove important structures, check prediction drop).
Sparsity: fraction of structures identified as important.
Trade-off: lower sparsity → higher fidelity (more noise = prediction worse).
Methods: GNNExplainer, PGExplainer, GISExplainer, HSIC-based explanations, integrated gradients. **Standard metrics but mostly for node/graph classification, not image retrieval**.

**SEMScene, SGM, SPAN**

Explanations are matched sub-graphs, triplets, or attention weights. Not formalized as fidelity/sparsity deletion tests, but matched structures are interpretable. **Option B explanations from matching mechanism**.

---

### 3.5. Per-image scene graphs + external knowledge (Option C, closest to team lean)

**VQA-GNN: Reasoning with Multimodal Knowledge, Yanan Wang et al., ICCV 2023**

Per-image scene graphs + ConceptNet/Comet triplets → contextual knowledge graph → multi-hop GNN reasoning → top-K knowledge for QA. Demonstrates the integration pattern. **Option C concept**. But for VQA, not retrieval.

**Knowledge Graphs Meet Multi-Modal Learning: Comprehensive Survey (2024)**

Extensive review of multimodal KG approaches. Covers entity alignment, relation prediction, image-KG matching. No specific per-image scene graph + external KG method dominates for text-to-image retrieval yet. **C is underexplored for retrieval compared to B and A**.

**A Multi-Hop Graph Reasoning Network for Knowledge-Based VQA**

Scene graphs + external KB → contextual graph → message passing GNN → answer. Demonstrates mechanism. **Option C pattern**.

---

### 3.6. Text scene-graph parsers

**Stanford Scene Graph Parser (NLP toolkit)**

Java, CoreNLP 3.6.0+. Rule-based + classifier-based. Standard tool, widely cited. Public. https://nlp.stanford.edu/software/scenegraph-parser.shtml

**SceneGraphParser (Python, GitHub, vacancy/SceneGraphParser)**

Python toolkit. spaCy-based syntactic parsing. Converts text to graph (words as nodes, syntactic relations as edges). Lighter, more modern than Stanford. Public. https://github.com/vacancy/SceneGraphParser

**Scene Graph Parsing as Dependency Parsing, Yu-Siang Wang et al.**

Treats scene graph parsing as dependency parsing problem. Structured approach tới extracting objects, attributes, relations từ text. **chưa kiểm chứng venue/year**.

**LLM-based parsing (2023+)**

Recent work mentions FACTUAL parser (2023) và LLM-based approaches. Details limited. **Emerging approach, less standard**.

---

## 4. So sánh ba phương án A, B, C cho đề tài này

| Tiêu chí | A (Global Heterogeneous) | B (Per-image Scene Graphs) | C (Hybrid Triple Nodes) |
|----------|--------------------------|--------------------------|------------------------|
| **Ưu điểm** | • Dùng external knowledge (ConceptNet)<br/>• Cấu trúc global, concept sharing<br/>• Tái sử dụng concept nodes<br/>• Tiềm năng cho cross-image reasoning | • Được chứng minh: 8 bài báo major<br/>• Clear per-image representation<br/>• Dễ parse query thành graph<br/>• Matching mechanism rõ ràng<br/>• Giải thích từ sub-graph trực tiếp<br/>• SOTA results consistent | • Kết hợp per-image clarity + global knowledge<br/>• Triple nodes = relational structure<br/>• Tiêm năng tốt nhất cho hybrid<br/>• Có thể giải thích qua triple importance |
| **Nhược điểm** | • Ít bài báo dùng làm core (CVSE, ERNIE only)<br/>• Node explosion (image count × concept count)<br/>• Khó giải thích từ cơ chế (explanation post-hoc)<br/>• Query-to-concept matching không rõ<br/>• K-reciprocal re-ranking là post-hoc | • Đã được làm rất kỹ, ít novelty<br/>• Không dùng external knowledge<br/>• Mỗi ảnh độc lập (no cross-image KG)<br/>• Cần text parser tốt (bottleneck) | • Chưa được công bố làm approach chính<br/>• Thiết kế chưa rõ (node/edge cấu trúc?)<br/>• ConceptNet + VG mapping phức tạp<br/>• GNN design complexity cao |
| **Công sức Implementation** | High | **Low-Medium** | **High-Very High** |
| **Công sức Evaluation** | Medium | Medium | High |
| **GNN cổ điển (yêu cầu)** | GCN/GAT/SAGE có thể | GCN chủ yếu, có GAT | GCN/GAT cả hai |
| **Ablation qua edges (yêu cầu)** | K-reciprocal suffices (post-hoc) | Có thể: remove edges từ VSG/TSG | Cần: random / remove triple edges |
| **Giải thích từ mechanism (yêu cầu)** | Khó: k-reciprocal không semantically meaningful | **Có**: matched graph/triplets trực tiếp | **Có**: triple importance scores |
| **Fidelity/Sparsity (yêu cầu)** | Limited ground truth | Deletion test on sub-graphs | Deletion test on triples |
| **Phù hợp yêu cầu "at least one classic GNN"** | ✓ GCN in re-ranking | ✓ GCN core | ✓ GCN + GAT |
| **Phù hợp yêu cầu "alpha*CLIP + (1-alpha)*graph"** | ✓ (alpha blending) | ✓ (easy to combine) | ✓ (easy to combine) |
| **Phù hợp yêu cầu "ablation removes/shuffles edges"** | ✗ k-reciprocal not interpretable | ✓ remove VSG/TSG edges | ✓ remove triple edges |
| **Phù hợp yêu cầu "explanations from mechanism"** | ✗ post-hoc | ✓ matched sub-graphs | ✓ triple scores |

**Phân tích:**

- **A (Global)**: Thoải mái về khái niệm (external knowledge), nhưng ít công bố làm điều này cho retrieval, và cơ chế giải thích yếu (k-reciprocal re-ranking không semantically meaningful, chỉ là neighbor co-retrieval).
- **B (Per-image)**: Cứng (proven 8×), nhưng ít novelty. Có thể đạt yêu cầu nếu extend với text parser + GCN matching + sub-graph explanations. Không dùng ConceptNet.
- **C (Hybrid)**: Tiềm năng cao (combines (B) clarity + (A) knowledge), rất phù hợp với "triple nodes" lean, nhưng **chưa được công bố như phương pháp chính** ở text-to-image retrieval. Thiết kế phức tạp nhất.

---

## 5. Khuyến nghị

**Khuyến nghị: Chọn PHƯƠNG ÁN B (Per-image scene graphs) với hybrid architecture hướng tới C**.

### 5.1 Lý do

1. **B là proven**: VSRN, SGM, GSMN, SGRAF, Structure-CLIP, SPAN, SEMScene, CORA tất cả làm việc. Publication track record mạnh. Implementation ít vấn đề.
2. **C thiết kế cụ thể có sẵn**: Team lean to C = per-image scene graphs with triple nodes. Đây vẫn **Option B core + (C) node enrichment**. Có thể làm được bằng cách:
   - Bắt đầu với B (per-image VSG, text TSG parse)
   - Mỗi scene graph được enrich: thay vì chỉ object nodes, add **triple nodes** (subject-predicate-object as explicit nodes)
   - Edges: object→object (relation), object→triple (membership)
   - Optionally add ConceptNet edges (object concept nodes linked)
   → **Hybrid C architecture, xây dựng trên proven B framework**.
3. **Giải thích từ mechanism**: TSG matching với VSG → matched triplets = explanation. Triple importance (via attention hoặc ablation) = explainability. Đáp ứng yêu cầu "explanations from ranking mechanism".
4. **Ablation cạnh**: Remove/shuffle triple edges → measure performance drop → sparsity/fidelity.
5. **GNN cổ điển**: GCN core matching, optional GAT cho triple importance.

### 5.2 Thiết kế cụ thể

```
Per-image Scene Graph (VSG) with Triple Nodes:
- Node types: {Object, Attribute, Triple}
- Object node: object class (bounding box in image)
- Attribute node: visual attribute
- Triple node: (subject_obj, predicate, object_obj)
- Edges: object--relation--object, object--has--attribute, triple--contains--object

Text-to-Query Graph (TSG) with Triple Nodes:
- Parsed từ caption via Stanford/SceneGraphParser
- Same structure: Object, Attribute, Triple nodes
- Triple nodes: (subject_noun, verb/prep, object_noun)

Matching mechanism:
- GCN encoder cho VSG → visual embedding
- GCN encoder cho TSG → text embedding (frozen từ CLIP hoặc retrained)
- Graph matching: triple-to-triple similarity + object-to-object (aggregated từ GCN)
- Score = alpha * CLIP_cosine(image, text) + (1-alpha) * GCN_graph_similarity(VSG, TSG)

Explanation:
- Matched triples (TSG→VSG) = explanation
- Triple importance: attention weights hoặc ablation score (remove triple edges)
- Sparsity: fraction of triples used in explanation
- Fidelity: drop in score when top-K triples removed
```

### 5.3 Ba bài nên đọc kỹ

1. **VSRN (Li et al., ICCV 2019)**: Foundation. GCN cho scene graphs, simple setup. Code clear. Start here.
   - https://github.com/KunpengLi1994/VSRN
   - https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Visual_Semantic_Reasoning_for_Image-Text_Matching_ICCV_2019_paper.pdf

2. **SEMScene (2024)**: Multi-level matching idea (node/triplet/graph). Closest to team's "triple nodes" concept. Most recent.
   - https://dl.acm.org/doi/10.1145/3664816

3. **CORA (CVPR 2024)**: Dual-GAT for relations + attributes. Explicit compositional modeling (team's bag-of-words fix need).
   - https://mlanthology.org/cvpr/2024/pham2024cvpr-composing/

### 5.4 Support từ evidence

**Team's lean towards (C) với triple nodes được support**:
- SEMScene làm "semantic triplet level matching" (2024, SOTA)
- CORA làm "object relations + attributes" với separate GATs (CVPR 2024)
- VQA-GNN demonstrates per-image scene graph + external KG pattern (C concept)
- **Nhưng**: Chưa có published method nào explicitly tên gọi là "triple nodes hybrid". Team sẽ innovate here.

**Chống lại (A) global heterogeneous**:
- Chỉ CVSE, một số re-ranking papers dùng làm core
- K-reciprocal explanation không semantic
- Node explosion problem
- Better để dùng per-image clarity

---

## 6. Những điều chưa kiểm chứng được

1. **SGRAF (AAAI 2021)**: Tìm được title nhưng abstract không chi tiết. Possible sắp xếp giữa B và A. Nên xem full paper.
2. **FACTUAL parser (2023)**: Được mention nhưng không tìm được PDF/code. Chỉ có Stanford, SceneGraphParser, LLM-based.
3. **Exact CLIP baseline details trong team's setup**: 38.5% R@1, 96.7% R@50 chắc đúng, nhưng "bag-of-words behavior" cụ thể cần verify qua ARO benchmark.
4. **ConceptNet edges cho Visual Genome objects**: Mapping chính xác chưa clear từ papers. Team sẽ cần align object classes với ConceptNet concepts.
5. **Hybrid (C) implementation complexity**: Chưa có bài paper làm chính xác như team description. Estimate effort là High.
6. **Fidelity/Sparsity metrics trên text-to-image**: Hầu hết papers dùng cho node/graph classification, chưa standard cho image retrieval explanation evaluation.

---

## 7. Danh sách nguồn

### 7.1 Main Papers

- https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Visual_Semantic_Reasoning_for_Image-Text_Matching_ICCV_2019_paper.pdf (VSRN)
- https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_Cross-modal_Scene_Graph_Matching_for_Relationship-aware_Image-Text_Retrieval_WACV_2020_paper.pdf (SGM)
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Graph_Structured_Network_for_Image-Text_Matching_CVPR_2020_paper.pdf (GSMN)
- https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690018.pdf (CVSE)
- https://arxiv.org/abs/2305.06152 (Structure-CLIP)
- https://arxiv.org/pdf/2210.01936 (NegCLIP/ARO)
- https://arxiv.org/pdf/2304.00590 (SPAN)
- https://dl.acm.org/doi/10.1145/3664816 (SEMScene)
- https://mlanthology.org/cvpr/2024/pham2024cvpr-composing/ (CORA)
- https://www.ijcai.org/proceedings/2019/0720.pdf (Knowledge Aware Semantic Concept Expansion)
- https://arxiv.org/pdf/2012.07620 (GNN Perspective Re-ranking)
- https://arxiv.org/abs/2306.08792 (Graph Convolution Re-ranking 2023)
- https://arxiv.org/pdf/2205.11501 (VQA-GNN)
- https://arxiv.org/pdf/1602.07332 (Visual Genome)

### 7.2 Benchmarks & Tools

- https://www.emergentmind.com/papers/2204.03162 (Winoground)
- https://nlp.stanford.edu/software/scenegraph-parser.shtml (Stanford Scene Graph Parser)
- https://github.com/vacancy/SceneGraphParser (SceneGraphParser Python)

### 7.3 Surveys & Reviews

- https://arxiv.org/pdf/2402.05391 (Knowledge Graphs Meet Multi-Modal Learning Survey)
- https://arxiv.org/pdf/2005.08045 (Visual Relationship Detection using Scene Graphs Survey)
- https://arxiv.org/pdf/2104.01111 (Scene Graphs: Generation and Application Survey)
- https://dl.acm.org/doi/10.1145/3711122 (GNN Explainability Survey)

### 7.4 Recent Work & Applications

- https://arxiv.org/pdf/2401.14111 (Image Synthesis with Graph Conditioning 2024)
- https://arxiv.org/pdf/2406.02343 (Cluster-Aware Similarity Diffusion 2024)
- https://www.sciencedirect.com/science/article/abs/pii/S0957417424007085 (Heterogeneous Graph Fusion Network 2024)
- https://arxiv.org/pdf/2508.05318 (mKG-RAG: Multimodal KG + RAG for VQA)
- https://arxiv.org/pdf/2203.02985 (Dynamic Key-value Memory Enhanced GNN for KB-VQA)

### 7.5 Code Repositories

- https://github.com/KunpengLi1994/VSRN (VSRN PyTorch)
- https://github.com/CrossmodalGroup/GSMN (GSMN PyTorch)
- https://github.com/BruceW91/CVSE (CVSE official)

---

**Kết luận**: **Option B per-image scene graphs**, enriched với **triple node architecture (hướng C)**, là lựa chọn tốt nhất. Xây dựng trên VSRN/GSMN foundation nhưng add triple nodes để capture compositional structure (fix bag-of-words), dùng GCN/GAT classic, enable explanation qua matched triples + ablation sparsity/fidelity. Team's lean towards C là reasonable, nhưng implement từ proven B framework.
