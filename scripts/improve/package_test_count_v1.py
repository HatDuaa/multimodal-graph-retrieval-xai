"""Package one split through the data chain of countonly_factual_r3 (count_v1), for the one-time test evaluation.

The same functions that built the train/val data, applied to --split:
- image graphs: the level-1 graphs with VG boxes and synsets (data/processed/geometry/objects_<split>.json, from
  build_boxes.py --synsets), duplicates merged as in merge_v1 (package_merged.merged_graph), per-node counts as in
  count_v1 (package_counts.counts_of);
- caption graphs: FACTUAL-T5 large (factual_v1) with count classes as in group_v1 (package_group.caption_side);
- candidates: symlink to factual_v1's CLIP top-50 (level-1 candidates);
- part vectors that the factual_v1 table lacks, encoded with the same frozen CLIP and conventions.
--split test writes data/{graphs,processed,features}/count_v1_test (read through configs/count_v1_test.yaml), never
overwriting. --split val writes to --out-root (a scratch folder) and checks that the result equals the val data the runs
were trained and selected on (merge_v1 graphs, count_v1 buckets, group_v1 caption graphs). Nothing here scores a model.
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.package_counts import counts_of  # noqa: E402
from scripts.improve.package_factual import fresh, sha256  # noqa: E402
from scripts.improve.package_group import caption_side  # noqa: E402
from scripts.improve.package_merged import merged_graph, part_keys  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

VERSION = "count_v1_test"


def query_keys(graph):
    from src.data.graph_store import GraphStore
    parts = GraphStore._graph_parts(graph, query=True)
    return ([f"node:{x}" for x in parts["node_labels"]] + [f"rel:{x}" for x in parts["relation_labels"]]
            + [f"triple:{x}" for x in parts["triple_labels"]])


def val_check(graphs, buckets, out_processed):
    """The val outputs must equal the data the runs used."""
    merged = {}
    for line in (REPO_ROOT / "data" / "graphs" / "merge_v1" / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8"):
        graph = json.loads(line)
        if graph["split"] == "val":
            merged[graph["image_id"]] = graph
    stored = json.loads((REPO_ROOT / "data" / "graphs" / "count_v1" / "count_buckets.json").read_text(encoding="utf-8"))
    queries = json.loads((REPO_ROOT / "data" / "processed" / "group_v1" / "query_graphs_val.json").read_text(encoding="utf-8"))
    new_queries = json.loads((out_processed / "query_graphs_val.json").read_text(encoding="utf-8"))
    return {"graphs_equal_merge_v1": {g["image_id"]: g for g in graphs} == merged,
            "buckets_equal_count_v1": all(buckets[str(i)] == stored[str(i)] for i in merged) and len(buckets) == len(merged),
            "caption_graphs_equal_group_v1": new_queries == queries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "val"), required=True)
    parser.add_argument("--out-root", type=Path, help="Root for data/{graphs,processed,features}; required for val.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="CLIP text encoder device for the missing part vectors.")
    args = parser.parse_args()
    if args.split == "val" and args.out_root is None:
        raise SystemExit("--split val is a consistency check: give a scratch --out-root")
    started = time.time()
    root = args.out_root or REPO_ROOT
    base_cfg, factual_cfg = load_config(), load_config(REPO_ROOT / "configs" / "factual_v1.yaml")
    out_graphs, out_processed, out_features = (root / "data" / kind / VERSION for kind in ("graphs", "processed", "features"))
    for folder in (out_graphs, out_processed, out_features):
        folder.mkdir(parents=True, exist_ok=True)
    source = resolve(base_cfg, "graphs") / "vg_coco_scene_graphs.jsonl"
    boxes_path = resolve(base_cfg, "processed") / "geometry" / f"objects_{args.split}.json"
    objects = json.loads(boxes_path.read_text(encoding="utf-8"))
    level1 = [json.loads(line) for line in source.open(encoding="utf-8")]
    level1 = sorted((g for g in level1 if g["split"] == args.split), key=lambda g: g["image_id"])
    with Pool(args.workers) as pool:
        graphs = pool.map(merged_graph, [(g, objects[str(g["image_id"])]) for g in level1], chunksize=32)
        rows = pool.map(counts_of, graphs, chunksize=32)
    with fresh(out_graphs / "vg_coco_scene_graphs.jsonl").open("w", encoding="utf-8") as handle:
        for graph in graphs:
            handle.write(json.dumps(graph) + "\n")
    buckets = {str(i): {"counts": c, "buckets": b} for i, _, c, b in rows}
    fresh(out_graphs / "count_buckets.json").write_text(json.dumps(buckets), encoding="utf-8")
    manifest = {"version": VERSION, "split": args.split, "date": time.strftime("%F %T %z"),
                "chain": "level-1 graphs + objects_<split> boxes -> merge_v1 merge -> count_v1 counts; factual_v1 captions -> "
                         "group_v1 count classes; factual_v1 candidates",
                "source_graphs_sha256": sha256(source), "objects_sha256": sha256(boxes_path), "captions": {},
                "images": len(graphs), "nodes_before": sum(len(g["objects"]) for g in level1),
                "nodes_after": sum(len(g["objects"]) for g in graphs)}
    caption_side(factual_cfg, args.split, out_processed, manifest)
    link = fresh(out_processed / f"candidates_{args.split}.npz")
    os.symlink(os.path.relpath(resolve(factual_cfg, "processed") / f"candidates_{args.split}.npz", out_processed), link)
    from src.features.extract_clip import ClipEncoder, load_features, save_features
    table, keys, encoder_name = load_features(resolve(factual_cfg, "features") / "vg_coco_graph_parts")
    row = {k: i for i, k in enumerate(keys)}
    queries = json.loads((out_processed / f"query_graphs_{args.split}.json").read_text(encoding="utf-8"))
    missing = sorted(({k for g in graphs for k in part_keys(g)} | {k for g in queries.values() for k in query_keys(g)}) - set(row))
    encoder = ClipEncoder.from_config(factual_cfg, device=args.device)
    assert encoder.name == encoder_name, (encoder.name, encoder_name)
    prompt = factual_cfg["clip"]["concept_prompt"]
    text = lambda k: prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1]  # noqa: E731
    sample = random.Random(factual_cfg["seed"]).sample(keys, 1000)
    error = float(np.max(np.abs(encoder.encode_texts([text(k) for k in sample]) - table[[row[k] for k in sample]])))
    assert error <= 1e-4, f"encoder differs from the factual_v1 table: max abs {error}"
    extra = encoder.encode_texts([text(k) for k in missing]) if missing else np.zeros((0, table.shape[1]), np.float32)
    stem = out_features / "vg_coco_graph_parts_extra"
    fresh(stem.with_suffix(".ids.json"))
    save_features(fresh(stem.with_suffix(".npy")).with_suffix(""), extra, missing, encoder_name, dtype=table.dtype)
    manifest["extra_parts"] = {"keys": len(missing), "by_kind": {kind: sum(k.startswith(f"{kind}:") for k in missing)
                                                                 for kind in ("node", "rel", "triple")},
                               "encoder": encoder_name, "reencoded_max_abs_error": error}
    if args.split == "val":
        merge_extra = json.loads((REPO_ROOT / "data" / "features" / "merge_v1" / "vg_coco_graph_parts_extra.ids.json")
                                 .read_text(encoding="utf-8"))["ids"]
        manifest["val_check"] = {**val_check(graphs, buckets, out_processed),
                                 "extra_keys_within_merge_v1_extras": set(missing) <= set(merge_extra)}
        print("val_check", json.dumps(manifest["val_check"]), flush=True)
    manifest["seconds"] = time.time() - started
    fresh(out_graphs / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "captions"}), flush=True)


if __name__ == "__main__":
    main()
