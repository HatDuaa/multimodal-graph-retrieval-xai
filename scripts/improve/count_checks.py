"""Checks before training the count table (experiment 4), val only, no training.

zero: the fullgraph_factual_r3_seed0 checkpoint (GPU)
  1. on merge_v1 with the current code, without the count table: val metrics must equal its metrics.json;
  2. on --config (group_v1 for experiment 4, count_v1 for experiment 5), without and with a zero count table g (same
     weights), twice: all val metrics must be equal (and, for count_v1, whose graphs are merge_v1's, equal to 1).
match: count agreement on the gold image (CPU). For every scorable val caption node with a known count class, the
  gold image's matching nodes (label equal after VG aliases, or frozen-CLIP node cosine >= 0.955, as in the
  graph-cleaning audit): has a match at all, and has a match whose count bucket is compatible with the class
  (src/graph/counts.py compatible). Image counts: level-1 graphs and merge_v1 without grouping (every node counts the
  same-type instances), group_v1 (group nodes count their members), and count_v1 (the stored buckets of experiment 5,
  which must reproduce the merge_v1 row).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.counts import CAPTION_CLASSES, compatible, image_bucket, image_counts  # noqa: E402
from src.graph.dedup import same_type, with_boxes  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.train_graph_reranker import build_model, evaluate_val, structure_options  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "group-count"
RUN = REPO_ROOT / "experiments" / "improve" / "full-graph" / "fullgraph_factual_r3_seed0"
KEYS = ("recall@1", "recall@5", "recall@10", "mrr")
THRESHOLD = .955


def zero_check(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = json.loads((RUN / "config.json").read_text(encoding="utf-8"))["args"]
    state = torch.load(RUN / "checkpoint_best.pt", map_location=device, weights_only=False)
    plain = build_model(options, device)
    plain.load_state_dict(state["model"])
    counted = build_model({**options, "count_table": True}, device)
    missing, unexpected = counted.load_state_dict(state["model"], strict=False)
    assert missing == ["count_table"] and not unexpected, (missing, unexpected)
    out = {"run": str(RUN.relative_to(REPO_ROOT)), "metrics_json": {k: state["metrics"][k] for k in KEYS}}
    for model in (plain, counted):
        model.eval()
    store = GraphStore(load_config(REPO_ROOT / "configs" / "merge_v1.yaml"), splits=("val",), structure=structure_options(options))
    out["merge_v1_without_table"] = {k: evaluate_val(plain, store, store.candidates("val"), device)[k] for k in KEYS}
    del store
    store = GraphStore(load_config(REPO_ROOT / config), splits=("val",), structure=structure_options(options))
    version = Path(config).stem
    out["config"] = str(config)
    for name, model in ((f"{version}_without_table", plain), (f"{version}_zero_table", counted),
                        (f"{version}_zero_table_again", counted)):
        out[name] = {k: evaluate_val(model, store, store.candidates("val"), device)[k] for k in KEYS}
    out["merge_v1_equals_metrics_json"] = out["merge_v1_without_table"] == out["metrics_json"]
    out["zero_table_equals_without"] = (out[f"{version}_zero_table"] == out[f"{version}_without_table"]
                                        == out[f"{version}_zero_table_again"])
    out["zero_table_equals_metrics_json"] = out[f"{version}_zero_table"] == out["metrics_json"]
    return out


def match_check():
    cfg = load_config(REPO_ROOT / "configs" / "group_v1.yaml")
    store = GraphStore(cfg, splits=("val",))
    vocab = Vocab.from_config(cfg)
    arrays = store.candidates("val")
    gold_of = {str(c): int(g) for c, g in zip(arrays["caption_ids"], arrays["gold"])}
    graphs = {}
    objects = json.loads((REPO_ROOT / "data" / "processed" / "geometry" / "objects_val.json").read_text(encoding="utf-8"))
    for line in (REPO_ROOT / "data" / "graphs" / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8"):
        g = json.loads(line)
        if g["split"] == "val":
            graphs.setdefault("level1", {})[g["image_id"]] = with_boxes(g, objects[str(g["image_id"])]["boxes"],
                                                                        objects[str(g["image_id"])]["synsets"])
    for name, config in (("merge_v1", "merge_v1"), ("group_v1", "group_v1"), ("count_v1", "merge_v1")):
        for line in (resolve(load_config(REPO_ROOT / "configs" / f"{config}.yaml"), "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8"):
            g = json.loads(line)
            if g["split"] == "val":
                graphs.setdefault(name, {})[g["image_id"]] = g
    vector = lambda label: store.part_vectors[store.part_row[f"node:{label}"]]  # noqa: E731
    cache = {}

    def match(q, i):
        if (q, i) not in cache:
            same = vocab.canon_object(q) == vocab.canon_object(i)
            known = f"node:{q}" in store.part_row and f"node:{i}" in store.part_row
            cache[(q, i)] = same or (known and float(vector(q) @ vector(i)) >= THRESHOLD)
        return cache[(q, i)]

    stored = json.loads((REPO_ROOT / "data" / "graphs" / "count_v1" / "count_buckets.json").read_text(encoding="utf-8"))
    buckets = {}
    for name, by_image in graphs.items():
        buckets[name] = {}
        for image_id, g in by_image.items():
            if name == "count_v1":
                buckets[name][image_id] = stored[str(image_id)]["buckets"]
                continue
            counts = [o["count"] for o in g["objects"]] if name == "group_v1" else image_counts(g["objects"], same_type)
            buckets[name][image_id] = [image_bucket(c) for c in counts]
    result = {"threshold": THRESHOLD, "variants": list(graphs), "by_class": {}}
    totals = {name: collections.Counter() for name in graphs}
    for cid, graph in store.queries.items():
        image_id = gold_of[cid]
        for obj in graph["objects"]:
            cls = obj.get("count_class", 0)
            if not cls or not obj["name"] or obj["name"] in PRONOUNS:
                continue
            for name in graphs:
                nodes = graphs[name][image_id]["objects"]
                hits = [buckets[name][image_id][j] for j, o in enumerate(nodes) if match(obj["name"], o["name"])]
                for key in ("nodes", f"class:{CAPTION_CLASSES[cls]}"):
                    totals[name][key] += 1
                    totals[name][f"{key}|matched"] += bool(hits)
                    totals[name][f"{key}|compatible"] += any(compatible(cls, b) for b in hits)
    for name, counter in totals.items():
        result[name] = {"caption_nodes": counter["nodes"], "matched": counter["nodes|matched"] / counter["nodes"],
                        "compatible": counter["nodes|compatible"] / counter["nodes"],
                        "compatible_given_matched": counter["nodes|compatible"] / max(counter["nodes|matched"], 1),
                        "by_class": {c: {"nodes": counter[f"class:{c}"],
                                         "compatible": counter[f"class:{c}|compatible"] / max(counter[f"class:{c}"], 1)}
                                     for c in CAPTION_CLASSES[1:]}}
        print(name, json.dumps(result[name]), flush=True)
    known = sum(any(o.get("count_class", 0) and o["name"] and o["name"] not in PRONOUNS for o in g["objects"])
                for g in store.queries.values())
    result["val_captions"] = len(store.queries)
    result["val_captions_with_known_count"] = known
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("zero", "match"))
    parser.add_argument("--config", default="configs/group_v1.yaml", help="zero: data version with count fields.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = zero_check(args.config) if args.check == "zero" else match_check()
    print(json.dumps(out, indent=2), flush=True)
    path = args.out or OUT / f"check_{args.check}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
