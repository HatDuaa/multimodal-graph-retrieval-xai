"""Connectivity of the graphs the GAT sees, per structure variant (experiment 3). CPU only, no model.

Image graphs on train and val: level-1 graphs (factual_v1), merge_v1 (duplicates merged), merge_v1 with the
5-nearest completion edges and bridges, and that plus the global node; group_v1 (group nodes, experiment 4) alone and
with completion edges and the global node. Sentence graphs (FACTUAL, factual_v1) on
train and val, without and with the global node. Per graph, edge direction ignored, isolated nodes counted as
components: number of components, share of nodes in the largest one, share of isolated nodes; plus labelled,
completion and bridge edges per graph and the share of image graphs that needed a bridge. Edges are built by
GraphStore._graph_parts, the code the training batches use.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.structure import structure_stats  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "full-graph"
IMAGE_VARIANTS = {"level1_factual_v1": ("configs/factual_v1.yaml", 0, False), "merge_v1": ("configs/merge_v1.yaml", 0, False),
                  "merge_v1+knn5": ("configs/merge_v1.yaml", 5, False), "merge_v1+knn5+global": ("configs/merge_v1.yaml", 5, True),
                  "group_v1": ("configs/group_v1.yaml", 0, False), "group_v1+knn5+global": ("configs/group_v1.yaml", 5, True)}


def graph_record(parts):
    n = len(parts["node_labels"])
    edges = [e[:2] for e in parts["edges"]] + [e[:2] for e in parts.get("added_edges", ())]
    if parts.get("global_node"):
        edges += [(n, i) for i in range(n)]
        n += 1
    stats = structure_stats(n, edges)
    degree = np.bincount(np.asarray(edges, dtype=np.int64).reshape(-1), minlength=n) if n else np.zeros(0)
    return {**stats, "isolated": int((degree == 0).sum()), "labelled": len(parts["edges"]),
            "completion": len(parts.get("added_edges", ())) - parts.get("bridge_edges", 0),
            "bridges": parts.get("bridge_edges", 0)}


def aggregate(records):
    graphs = [r for r in records if r["nodes"]]
    nodes = sum(r["nodes"] for r in graphs)
    return {"graphs": len(records), "empty_graphs": len(records) - len(graphs),
            "nodes_per_graph": nodes / max(len(graphs), 1),
            "components_per_graph": float(np.mean([r["components"] for r in graphs])),
            "components_median": float(np.median([r["components"] for r in graphs])),
            "connected_share": float(np.mean([r["components"] == 1 for r in graphs])),
            "largest_component_share": float(np.mean([r["largest_share"] for r in graphs])),
            "isolated_node_share": sum(r["isolated"] for r in graphs) / max(nodes, 1),
            "labelled_edges_per_graph": float(np.mean([r["labelled"] for r in records])),
            "completion_edges_per_graph": float(np.mean([r["completion"] for r in records])),
            "bridge_edges_per_graph": float(np.mean([r["bridges"] for r in records])),
            "graphs_with_bridge": float(np.mean([r["bridges"] > 0 for r in records]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--variants", nargs="+", default=list(IMAGE_VARIANTS), choices=list(IMAGE_VARIANTS))
    parser.add_argument("--no-sentences", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT / "graph_structure.json")
    args = parser.parse_args()
    result = {"images": {}, "sentences": {}}
    for name in args.variants:
        config, knn, global_node = IMAGE_VARIANTS[name]
        cfg = load_config(REPO_ROOT / config)
        graphs = {}
        with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                if graph["split"] in args.splits:
                    graphs.setdefault(graph["split"], []).append(graph)
        for split in args.splits:
            records = [graph_record(GraphStore._graph_parts(g, knn_edges=knn, global_node=global_node)) for g in graphs[split]]
            result["images"].setdefault(name, {})[split] = aggregate(records)
            print("image", name, split, json.dumps(result["images"][name][split]), flush=True)
    cfg = load_config(REPO_ROOT / "configs" / "factual_v1.yaml")
    for split in ([] if args.no_sentences else args.splits):
        queries = json.loads((resolve(cfg, "processed") / f"query_graphs_{split}.json").read_text(encoding="utf-8"))
        for name, global_node in (("factual_v1", False), ("factual_v1+global", True)):
            records = [graph_record(GraphStore._graph_parts(g, query=True, global_node=global_node)) for g in queries.values()]
            result["sentences"].setdefault(name, {})[split] = aggregate(records)
            print("sentence", name, split, json.dumps(result["sentences"][name][split]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
