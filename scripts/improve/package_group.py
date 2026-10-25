"""Package data version group_v1: merge_v1 plus group nodes and counts on both sides (experiment 4).

Image side, per train/val graph of merge_v1 (duplicates already merged at IoU >= 0.7): same-type nodes that relate to
one anchor node in the same way, (a) the same annotated predicate or (b) the same box-geometry class, are replaced by
one group node (src/graph/dedup.py group_by_role, criterion "ab", as in the graph-cleaning audit). A group node takes
the most frequent member label; a tie goes to the lowest common WordNet hypernym of the tied labels (majority_name);
its box is the union box and its edges the union of the members' edges. Every node then gets a count and a bucket
1 / 2 / 3-4 / >=5 (src/graph/counts.py): members for a group node, same-type instances in the image for the others.
Caption side: the factual_v1 caption graphs with a count class per node (src/graph/counts.py rules, spaCy parse of the
caption text and the raw FACTUAL output). Writes, never overwriting:
- data/graphs/group_v1/vg_coco_scene_graphs.jsonl and MANIFEST.json;
- data/processed/group_v1/query_graphs_{train,val}.json, and candidates_{split}.npz symlinks to factual_v1's;
- data/features/group_v1/vg_coco_graph_parts_extra.*: frozen-CLIP vectors of part texts the factual_v1 table lacks.
Test is not packaged (only before the one-time final evaluation). Read it through configs/group_v1.yaml.
"""
import argparse
import collections
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.package_factual import fresh, sha256  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS, caption_classes, image_bucket, image_counts  # noqa: E402
from src.graph.dedup import group_by_role, majority_name, same_type  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

VERSION = "group_v1"
SPLITS = ("train", "val")


def grouped_graph(graph):
    grouped, _ = group_by_role({"objects": graph["objects"], "relations": graph["relations"]}, "ab", naming=majority_name)
    counts = image_counts(grouped["objects"], same_type)
    objects = []
    for obj, count in zip(grouped["objects"], counts):
        node = {key: obj[key] for key in ("id", "name", "synset", "box", "boxes", "members")}
        if obj.get("group"):
            node.update(group=True, naming=obj["naming"])
        node.update(count=count, count_bucket=image_bucket(count))
        objects.append(node)
    return {"image_id": graph["image_id"], "split": graph["split"], "width": graph["width"], "height": graph["height"],
            "objects": objects, "relations": grouped["relations"]}


def entity_heads(raw_text):
    from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
    parser = object.__new__(SceneGraphParser)
    try:
        return [e["head"] for e in parser.graph_string_to_object(raw_text)["entities"]]
    except Exception:  # noqa: BLE001 - malformed FACTUAL output: fall back to lemma matching
        return None


def part_keys(graph):
    parts = GraphStore._graph_parts(graph)
    return ([f"node:{x}" for x in parts["node_labels"]] + [f"rel:{x}" for x in parts["relation_labels"]]
            + [f"triple:{x}" for x in parts["triple_labels"]])


def caption_side(cfg, split, out_processed, manifest):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    graphs = json.loads((resolve(cfg, "processed") / f"query_graphs_{split}.json").read_text(encoding="utf-8"))
    raw = json.loads((REPO_ROOT / "data" / "processed" / "parsers" / f"raw_{split}_factual-large.json").read_text(encoding="utf-8"))
    texts = {c["caption_id"]: c["text"] for im in json.loads((resolve(cfg, "splits") / f"vg_coco_{split}.json")
                                                             .read_text(encoding="utf-8"))["images"] for c in im["captions"]}
    ids = sorted(graphs)
    rules, classes, captions_known, heads_aligned = collections.Counter(), collections.Counter(), 0, 0
    for cid, doc in zip(ids, nlp.pipe((texts[c] for c in ids), batch_size=512, n_process=8)):
        graph = graphs[cid]
        heads = entity_heads(raw.get(cid, ""))
        if heads is not None and len(heads) != len(graph["objects"]):
            heads = None
        heads_aligned += heads is not None
        known = False
        for obj, (index, rule) in zip(graph["objects"], caption_classes(doc, graph["objects"], heads)):
            obj["count_class"] = index
            obj["count_rule"] = rule
            scorable = obj["name"] and obj["name"] not in PRONOUNS
            if scorable:
                rules[rule] += 1
                classes[CAPTION_CLASSES[index]] += 1
                known |= index > 0
        captions_known += known
    fresh(out_processed / f"query_graphs_{split}.json").write_text(json.dumps(graphs), encoding="utf-8")
    nodes = sum(classes.values())
    manifest["captions"][split] = {"captions": len(ids), "captions_with_known_count": captions_known,
                                   "captions_with_known_count_share": captions_known / len(ids),
                                   "raw_heads_aligned_share": heads_aligned / len(ids), "scorable_nodes": nodes,
                                   "node_classes": {k: classes[k] for k in CAPTION_CLASSES},
                                   "node_class_shares": {k: classes[k] / nodes for k in CAPTION_CLASSES},
                                   "node_rules": dict(rules)}
    print("captions", split, json.dumps(manifest["captions"][split]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    factual_cfg = load_config(REPO_ROOT / "configs" / "factual_v1.yaml")
    merge_cfg = load_config(REPO_ROOT / "configs" / "merge_v1.yaml")
    out_graphs, out_features = REPO_ROOT / "data" / "graphs" / VERSION, REPO_ROOT / "data" / "features" / VERSION
    out_processed = REPO_ROOT / "data" / "processed" / VERSION
    for folder in (out_graphs, out_features, out_processed):
        folder.mkdir(parents=True, exist_ok=True)
    source = resolve(merge_cfg, "graphs") / "vg_coco_scene_graphs.jsonl"
    target = fresh(out_graphs / "vg_coco_scene_graphs.jsonl")
    manifest = {"version": VERSION, "date": time.strftime("%F %T %z"), "base": "merge_v1",
                "source_graphs": str(source.relative_to(REPO_ROOT)), "source_graphs_sha256": sha256(source),
                "rule": {"grouping": "src/graph/dedup.py group_by_role criterion ab", "naming": "majority_name",
                         "counts": "src/graph/counts.py", "image_buckets": IMAGE_BUCKETS, "caption_classes": CAPTION_CLASSES},
                "images": {}, "captions": {}}
    graphs = [json.loads(line) for line in source.open(encoding="utf-8")]
    with Pool(args.workers) as pool:
        grouped = pool.map(grouped_graph, graphs, chunksize=64)
    for split in SPLITS:
        before = [g for g in graphs if g["split"] == split]
        after = [g for g in grouped if g["split"] == split]
        groups = [o for g in after for o in g["objects"] if o.get("group")]
        manifest["images"][split] = {
            "images": len(after), "nodes_before": sum(len(g["objects"]) for g in before),
            "nodes_after": sum(len(g["objects"]) for g in after), "edges_before": sum(len(g["relations"]) for g in before),
            "edges_after": sum(len(g["relations"]) for g in after), "group_nodes": len(groups),
            "group_nodes_per_image": len(groups) / len(after), "mean_group_size": float(np.mean([o["count"] for o in groups])),
            "group_naming": dict(collections.Counter(o["naming"] for o in groups)),
            "node_buckets": {b: sum(o["count_bucket"] == k for g in after for o in g["objects"]) for k, b in enumerate(IMAGE_BUCKETS)}}
        print("images", split, json.dumps(manifest["images"][split]), flush=True)
    with target.open("w", encoding="utf-8") as handle:
        for graph in grouped:
            handle.write(json.dumps(graph) + "\n")
    for split in SPLITS:
        caption_side(factual_cfg, split, out_processed, manifest)
        link = fresh(out_processed / f"candidates_{split}.npz")
        os.symlink(os.path.relpath(resolve(factual_cfg, "processed") / f"candidates_{split}.npz", out_processed), link)
    from src.features.extract_clip import ClipEncoder, load_features, save_features
    table, keys, encoder_name = load_features(resolve(factual_cfg, "features") / "vg_coco_graph_parts")
    row = {k: i for i, k in enumerate(keys)}
    missing = sorted({k for g in grouped for k in part_keys(g)} - set(row))
    encoder = ClipEncoder.from_config(factual_cfg)
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
    manifest["extra_parts"] = {"keys": len(missing),
                               "by_kind": {kind: sum(k.startswith(f"{kind}:") for k in missing) for kind in ("node", "rel", "triple")},
                               "encoder": encoder_name, "reencoded_max_abs_error": error}
    manifest["seconds"] = time.time() - started
    fresh(out_graphs / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["extra_parts"]), f"{manifest['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
