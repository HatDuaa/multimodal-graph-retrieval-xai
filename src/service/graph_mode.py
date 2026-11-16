""""CLIP + Đồ thị" ranking mode for the demo: re-rank CLIP's top-50 with a trained graph reranker.

The run decides the data: its config.json names the training config, and DEMO_DATA gives the config of the demo split
and the caption parser that built its packaged caption graphs. Level-1 runs (no config entry) read configs/default.yaml
and parse with sng_parser; countonly_factual_r3 reads count_v1 (merge_v1 image graphs with box geometry, completion
edges and a global node as in its config.json, count buckets) and parses with FACTUAL-T5 large plus the count rules
(src/graph/caption_parsers.FactualQueryParser). Part texts that the frozen part table lacks are encoded on the fly with
the same frozen CLIP text encoder and the same conventions (nodes through the concept prompt, relations and triples
verbatim). Every explanation row comes from the ranking pass itself (src/explain/graph_explainer.matched_pairs),
never from a search after the fact.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.data.graph_store import GraphIndex, GraphStore, attach_count_buckets
from src.explain.graph_explainer import matched_pairs
from src.explain.score_breakdown import top50_deltas
from src.graph.soft_match import PRONOUNS
from src.models.graph_reranker import fill_known_mean
from src.train_graph_reranker import load_run_model, structure_options
from src.utils.config import REPO_ROOT, load_config

# Training config of a run (config.json "config") -> config read for each demo split, and the caption parser.
DEMO_DATA = {
    None: {"val": None, "test": None, "parser": "sng"},
    "configs/count_v1.yaml": {"val": "configs/count_v1.yaml", "test": "configs/count_v1_test.yaml", "parser": "factual-large"},
}
# Folders searched for a run given by name; the default is the improvement phase's main model, seed chosen on val R@1.
RUN_ROOTS = ("experiments/improve/count-only", "experiments/level1/gat")
DEFAULT_RUN = "countonly_factual_r3_seed0"


def find_run(run) -> Path:
    path = Path(run)
    if path.is_dir():
        return path
    for root in RUN_ROOTS:
        if (REPO_ROOT / root / run / "config.json").exists():
            return REPO_ROOT / root / run
    raise FileNotFoundError(f"run {run!r} not found in {RUN_ROOTS}")


def unmatched_parts(query_graph, query_parts, image_parts):
    """Query parts that get no explanation row in each channel, with the reason the ranking skipped them.

    Mirrors GraphStore._graph_parts(query=True) and matched_pairs: pronoun objects are dropped from the query graph
    together with every relation touching them, empty object labels are hidden by part_mask, and a channel whose
    image side is empty yields no rows. Reason codes: pronoun, empty_label, dangling, image_empty.
    """
    names = {o["id"]: o["name"] for o in query_graph["objects"]}
    scorable = {i for i, keep in zip(query_parts["node_indices"], query_parts["part_mask"]) if keep}
    kept_relations = set(query_parts["relation_indices"])
    out = {"objects": [], "triples": []}
    for index, obj in enumerate(query_graph["objects"]):
        if index in scorable:
            reason = None if image_parts["node_indices"] else "image_empty"
        else:
            reason = "pronoun" if obj["name"] in PRONOUNS else "empty_label"
        if reason:
            out["objects"].append({"query_part": obj["name"], "reason": reason})
    for index, rel in enumerate(query_graph["relations"]):
        part = (names.get(rel["subject"], "?"), rel["predicate"].lower(), names.get(rel["object"], "?"))
        if index in kept_relations:
            reason = None if image_parts["relation_indices"] else "image_empty"
        else:
            reason = "pronoun" if part[0] in PRONOUNS or part[2] in PRONOUNS else "dangling"
        if reason:
            out["triples"].append({"query_part": part, "reason": reason})
    return out


class GraphRerankMode:
    """Callable reranker (query, hits) -> hits for SearchService.add_mode()."""

    def __init__(self, cfg, split, encoder, run=DEFAULT_RUN, device=None, parser_device=None):
        self.encoder = encoder
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run_dir = find_run(run)
        self.run = self.run_dir.name
        self.options = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))["args"]
        data = DEMO_DATA[self.options.get("config")]
        self.cfg = cfg if data[split] is None else load_config(REPO_ROOT / data[split])
        self.fusion = self.options.get("fusion") or "zscore"
        self.structure = structure_options(self.options)
        paths = self.cfg["paths"]
        features = REPO_ROOT / paths["features"]
        meta = json.loads((features / "vg_coco_graph_parts.ids.json").read_text(encoding="utf-8"))
        self.table = np.load(features / "vg_coco_graph_parts.npy", mmap_mode="r")
        self.row = {key: r for r, key in enumerate(meta["ids"])}
        self.extra = {}  # part key -> vector: the data version's extra parts, then texts encoded for typed queries
        if paths.get("extra_parts"):
            stem = REPO_ROOT / paths["extra_parts"]
            keys = json.loads(stem.with_suffix(".ids.json").read_text(encoding="utf-8"))["ids"]
            vectors = np.load(stem.with_suffix(".npy"))
            self.extra.update((k, np.asarray(v, dtype=np.float32)) for k, v in zip(keys, vectors) if k not in self.row)
        self.graphs = {}
        with (REPO_ROOT / paths["graphs"] / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                if graph["split"] == split:
                    self.graphs[int(graph["image_id"])] = graph
        if paths.get("count_buckets"):
            counts = json.loads((REPO_ROOT / paths["count_buckets"]).read_text(encoding="utf-8"))
            attach_count_buckets(self.graphs, counts)
        self.image_parts = {}  # image id -> graph parts with the run's structure, built on first use
        if data["parser"] == "sng":
            from src.graph.parse_query import parse_caption
            self.parse = parse_caption
        else:
            from src.graph.caption_parsers import FactualQueryParser
            self.parse = FactualQueryParser(str(parser_device or self.device))
        self.model = load_run_model(self.run_dir, self.device)[0]

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

    def _image(self, image_id):
        if image_id not in self.image_parts:
            self.image_parts[image_id] = GraphStore._graph_parts(self.graphs[image_id], **self.structure)
        return self.image_parts[image_id]

    @staticmethod
    def _deltas(clip_weight, hits, all_pairs):
        """top50_deltas of every row; columns[channel] maps a query part index to its column."""
        out = {"columns": {}}
        terms = {}
        for key in ("objects", "triples"):
            indices = sorted({item["query_index"] for pairs in all_pairs for item in pairs[key]})
            out["columns"][key] = {index: col for col, index in enumerate(indices)}
            terms[key] = np.full((len(hits), len(indices)), np.nan)
            for row, pairs in enumerate(all_pairs):
                for item in pairs[key]:
                    terms[key][row, out["columns"][key][item["query_index"]]] = item["channel_term"]
        out.update(top50_deltas(clip_weight, [h.clip_score for h in hits], terms))
        return out

    @torch.no_grad()
    def __call__(self, query, hits):
        query_graph = self.parse(query)
        query_parts = GraphStore._graph_parts(query_graph, query=True, global_node=self.structure["global_node"])
        image_parts = [self._image(h.image_id) for h in hits]
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
        # fuse_tensor leaves a channel out of the graph score when it is inactive for this query (no scorable parts,
        # or constant over the pool); the demo states this instead of implying the channel weight was used
        active = [bool(detail["object_active"][0]), bool(detail["triple_active"][0])]
        # fuse_raw fills a missing channel score (image without a part to match) with the candidates' known mean
        filled = {"objects": fill_known_mean(detail["objects"])[0], "triples": fill_known_mean(detail["triples"])[0]}
        names = lambda graph: {o["id"]: o["name"] for o in graph["objects"]}  # noqa: E731
        all_pairs = [matched_pairs(self.model, detail, query_parts, parts, row) for row, parts in enumerate(image_parts)]
        deltas = self._deltas(weights[0], hits, all_pairs) if self.fusion == "raw" else None
        mean_fused = float(fused[0].mean())
        for row, (hit, parts, pairs) in enumerate(zip(hits, image_parts, all_pairs)):
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
            info = {"weights_abc": weights, **pairs, "fusion": self.fusion, "channel_active": active,
                    "unmatched": unmatched_parts(query_graph, query_parts, parts),
                    "query_graph": {"objects": [o["name"] for o in query_graph["objects"]],
                                    "relations": [(qnames[r["subject"]], r["predicate"].lower(), qnames[r["object"]])
                                                  for r in query_graph["relations"]
                                                  if r["subject"] in qnames and r["object"] in qnames]}}
            if self.fusion == "raw":
                # fused = a * cosine + b * S_obj + c * S_tri; a channel without rows here took the filled mean
                info["fills"] = {key: (weights[k] * float(filled[key][row])
                                       if not bool(torch.isfinite(detail[key][0, row])) else None)
                                 for k, key in ((1, "objects"), (2, "triples"))}
                # each row against its mean over the 50 candidates; the deltas add up to fused - mean(fused)
                info["clip_delta"] = float(deltas["clip"][row])
                info["delta_total"] = float(fused[0, row]) - mean_fused
                info["fill_deltas"] = {}
                for key in ("objects", "triples"):
                    for item in pairs[key]:
                        item["delta"] = float(deltas[key][row, deltas["columns"][key][item["query_index"]]])
                    info["fill_deltas"][key] = (float(deltas[key][row].sum())
                                                if info["fills"][key] is not None else None)
            else:
                # z-scored CLIP term of fused = a * z_clip + (1 - a) * graph (fuse_tensor)
                info["z_clip"] = float(detail["z_clip"][0, row])
            hit.explanation = [info]
        return hits
