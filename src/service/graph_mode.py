""""CLIP + Đồ thị" ranking mode for the demo: re-rank CLIP's top-50 with the trained graph reranker.

A typed query is parsed with sng_parser into the same query graph as the packaged captions; part texts that the
frozen part table lacks are encoded on the fly with the same frozen CLIP text encoder and the same conventions
(nodes through the concept prompt, relations and triples verbatim). Every explanation row comes from the
ranking pass itself (src/explain/graph_explainer.matched_pairs), never from a search after the fact.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.data.graph_store import GraphIndex, GraphStore
from src.explain.graph_explainer import matched_pairs
from src.graph.parse_query import parse_caption
from src.train_graph_reranker import load_run_model
from src.utils.config import resolve


class GraphRerankMode:
    """Callable reranker (query, hits) -> hits for SearchService.add_mode()."""

    def __init__(self, cfg, split, encoder, run="main_r3_seed0", device=None):
        self.cfg, self.encoder = cfg, encoder
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        features = resolve(cfg, "features")
        meta = json.loads((features / "vg_coco_graph_parts.ids.json").read_text(encoding="utf-8"))
        self.table = np.load(features / "vg_coco_graph_parts.npy", mmap_mode="r")
        self.row = {key: r for r, key in enumerate(meta["ids"])}
        self.extra = {}  # part key -> vector encoded on demand for typed queries
        self.graphs = {}
        with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                if graph["split"] == split:
                    self.graphs[int(graph["image_id"])] = graph
        self.run = run
        self.model = load_run_model(Path(resolve(cfg, "experiments")) / "level1" / "gat" / run, self.device)[0]

    def _vectors(self, keys):
        """A compact table holding exactly these part keys, encoding unseen texts once with frozen CLIP."""
        missing = [k for k in keys if k not in self.row and k not in self.extra]
        if missing:
            prompt = self.cfg["clip"]["concept_prompt"]
            texts = [prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1] for k in missing]
            for key, vector in zip(missing, self.encoder.encode_texts(texts)):
                self.extra[key] = np.asarray(vector, dtype=np.float32)
        local = {key: r for r, key in enumerate(keys)}
        table = np.stack([self.table[self.row[k]] if k in self.row else self.extra[k] for k in keys]).astype(np.float32)
        return local, torch.from_numpy(table).to(self.device)

    @staticmethod
    def _keys(parts):
        return ([f"node:{x}" for x in parts["node_labels"]] + [f"rel:{x}" for x in parts["relation_labels"]]
                + [f"triple:{x}" for x in parts["triple_labels"]])

    @torch.no_grad()
    def __call__(self, query, hits):
        query_graph = parse_caption(query)
        query_parts = GraphStore._graph_parts(query_graph, query=True)
        image_parts = [GraphStore._graph_parts(self.graphs[h.image_id]) for h in hits]
        keys = list(dict.fromkeys(self._keys(query_parts) + [k for p in image_parts for k in self._keys(p)]))
        local, table = self._vectors(keys)
        sentence = torch.from_numpy(np.asarray(self.encoder.encode_texts([query]), dtype=np.float32)).to(self.device)
        batch = {"queries": GraphIndex.build([query_parts], local).to_batch(table, self.device),
                 "images": GraphIndex.build(image_parts, local).to_batch(table, self.device),
                 "candidate_index": torch.arange(len(hits), device=self.device)[None],
                 "clip": torch.tensor([[h.clip_score for h in hits]], dtype=torch.float32, device=self.device),
                 "sentence": sentence}
        fused, detail = self.model(batch, return_encoded=True)
        weights = detail["weights"][0].cpu().tolist()
        names = lambda graph: {o["id"]: o["name"] for o in graph["objects"]}  # noqa: E731
        for row, (hit, parts) in enumerate(zip(hits, image_parts)):
            pairs = matched_pairs(self.model, detail, query_parts, parts, row)
            image_graph, qnames, inames = self.graphs[hit.image_id], names(query_graph), names(self.graphs[hit.image_id])
            for item in pairs["objects"]:
                item["query_part"] = query_graph["objects"][item["query_index"]]["name"]
                item["image_part"] = image_graph["objects"][item["image_index"]]["name"]
            for item in pairs["triples"]:
                q, i = query_graph["relations"][item["query_index"]], image_graph["relations"][item["image_index"]]
                item["query_part"] = (qnames[q["subject"]], q["predicate"].lower(), qnames[q["object"]])
                item["image_part"] = (inames[i["subject"]], i["predicate"], inames[i["object"]])
            hit.score = float(fused[0, row])
            hit.graph_score = float(detail["graph"][0, row])
            hit.explanation = [{"weights_abc": weights, **pairs,
                                "query_graph": {"objects": [o["name"] for o in query_graph["objects"]],
                                                "relations": [(qnames[r["subject"]], r["predicate"].lower(), qnames[r["object"]])
                                                              for r in query_graph["relations"]
                                                              if r["subject"] in qnames and r["object"] in qnames]}}]
        return hits
