"""Package data version merge_v1: factual_v1 with duplicate boxes merged in the image graphs (experiment 3.1).

Image side only; the caption graphs, CLIP top-50 candidates and caption vectors stay those of factual_v1.
Per train/val image (src/graph/dedup.py, the definition of the graph-cleaning audit): two nodes are one thing when
they have the same type (synsets equal or one a WordNet hypernym of the other, every person synset counting as
person.n.01; equal names without synsets) and IoU >= 0.7, closed transitively. A merged node takes the label and
synset of its largest box (ties: the lowest node index) and the union box; edges are re-pointed, then duplicate
triples and self-loops dropped. Writes, never overwriting:
- data/graphs/merge_v1/vg_coco_scene_graphs.jsonl: train and val graphs with image width/height, a box per node and
  the source relation index per edge (test is not packaged here: it is built the same way before the one-time
  final evaluation only);
- data/features/merge_v1/vg_coco_graph_parts_extra.*: frozen-CLIP vectors of the part texts that the factual_v1 table
  lacks (triples whose subject or object label changed), same encoder and conventions as package_factual.py;
- data/graphs/merge_v1/MANIFEST.json.
Read it through configs/merge_v1.yaml.
"""
import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.package_factual import fresh, sha256  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.dedup import merge_duplicates, with_boxes  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

VERSION = "merge_v1"
THRESHOLD = .7
SPLITS = ("train", "val")


def merged_graph(args):
    graph, objects = args
    merged, _ = merge_duplicates(with_boxes(graph, objects["boxes"], objects["synsets"]), THRESHOLD)
    return {"image_id": graph["image_id"], "split": graph["split"], "width": objects["width"], "height": objects["height"],
            "objects": [{"id": o["id"], "name": o["name"], "synset": o["synset"], "box": o["box"], "boxes": o["boxes"],
                         "members": o["members"]} for o in merged["objects"]],
            "relations": merged["relations"]}


def part_keys(graph):
    parts = GraphStore._graph_parts(graph)
    return ([f"node:{x}" for x in parts["node_labels"]] + [f"rel:{x}" for x in parts["relation_labels"]]
            + [f"triple:{x}" for x in parts["triple_labels"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    base_cfg, factual_cfg = load_config(), load_config(REPO_ROOT / "configs" / "factual_v1.yaml")
    out_graphs = REPO_ROOT / "data" / "graphs" / VERSION
    out_features = REPO_ROOT / "data" / "features" / VERSION
    out_graphs.mkdir(parents=True, exist_ok=True)
    out_features.mkdir(parents=True, exist_ok=True)
    target = fresh(out_graphs / "vg_coco_scene_graphs.jsonl")
    source = resolve(base_cfg, "graphs") / "vg_coco_scene_graphs.jsonl"
    graphs = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] in SPLITS:
                graphs[graph["image_id"]] = graph
    manifest = {"version": VERSION, "date": time.strftime("%F %T %z"), "base": "factual_v1",
                "rule": {"same_type": "src/graph/dedup.py same_type", "iou": THRESHOLD,
                         "label": "largest box (ties: lowest node index)", "box": "union of the merged boxes",
                         "edges": "re-pointed; duplicate (s, p, o) and self-loops dropped"},
                "source_graphs": str(source.relative_to(REPO_ROOT)), "source_graphs_sha256": sha256(source), "splits": {}}
    merged = []
    for split in SPLITS:
        objects = json.loads((resolve(base_cfg, "processed") / "geometry" / f"objects_{split}.json").read_text(encoding="utf-8"))
        manifest[f"objects_{split}_sha256"] = sha256(resolve(base_cfg, "processed") / "geometry" / f"objects_{split}.json")
        ids = sorted(i for i in graphs if graphs[i]["split"] == split)
        with Pool(args.workers) as pool:
            out = pool.map(merged_graph, [(graphs[i], objects[str(i)]) for i in ids], chunksize=64)
        merged.extend(out)
        manifest["splits"][split] = {
            "images": len(out), "nodes_before": sum(len(graphs[i]["objects"]) for i in ids),
            "nodes_after": sum(len(g["objects"]) for g in out),
            "edges_before": sum(len(graphs[i]["relations"]) for i in ids), "edges_after": sum(len(g["relations"]) for g in out),
            "merged_nodes": sum(o["boxes"] > 1 for g in out for o in g["objects"])}
        print(split, json.dumps(manifest["splits"][split]), f"{time.time() - started:.0f}s", flush=True)
    with target.open("w", encoding="utf-8") as handle:
        for graph in merged:
            handle.write(json.dumps(graph) + "\n")
    from src.features.extract_clip import ClipEncoder, load_features, save_features
    table, keys, encoder_name = load_features(resolve(factual_cfg, "features") / "vg_coco_graph_parts")
    row = {k: i for i, k in enumerate(keys)}
    missing = sorted({k for g in merged for k in part_keys(g)} - set(row))
    encoder = ClipEncoder.from_config(base_cfg)
    assert encoder.name == encoder_name, (encoder.name, encoder_name)
    prompt = base_cfg["clip"]["concept_prompt"]
    text = lambda k: prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1]  # noqa: E731
    sample = random.Random(base_cfg["seed"]).sample(keys, 1000)
    error = float(np.max(np.abs(encoder.encode_texts([text(k) for k in sample]) - table[[row[k] for k in sample]])))
    assert error <= 1e-4, f"encoder differs from the factual_v1 table: max abs {error}"
    extra = encoder.encode_texts([text(k) for k in missing]) if missing else np.zeros((0, table.shape[1]), np.float32)
    stem = out_features / "vg_coco_graph_parts_extra"
    fresh(stem.with_suffix(".ids.json"))
    save_features(fresh(stem.with_suffix(".npy")).with_suffix(""), extra, missing, encoder_name, dtype=table.dtype)
    manifest["extra_parts"] = {"keys": len(missing),
                               "by_kind": {kind: sum(k.startswith(f"{kind}:") for k in missing) for kind in ("node", "rel", "triple")},
                               "encoder": encoder_name, "reencoded_factual_v1_sample": len(sample),
                               "reencoded_max_abs_error": error}
    manifest["seconds"] = time.time() - started
    fresh(out_graphs / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
