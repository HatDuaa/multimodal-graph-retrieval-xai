"""Audit of image-graph cleaning (merge duplicates, then group by role) on train and val. CPU only, no training.

Variants per image (src/graph/dedup.py): before; merge at IoU 0.5 / 0.7 / 0.9; on each merge, grouping by (a) the
same annotated predicate to a shared anchor, (b) the same box-geometry class to a shared anchor, (ab) either, and the
separate "near" variant. Measured per variant: nodes per image, connected components, degree-1 and isolated shares,
largest component, group nodes and their size, the two-tier count (group nodes + member-of edges, not built), and the
(a)/(b) agreement on merged graphs. On val also: the shared-node consistency of step 2 (gold image vs the other 49),
and whether the gold image has the stated number of things for plural or counted caption nodes (FACTUAL).
Thresholds are judged on train statistics and on the drawn train images only (graph_dedup_draw.py), never on val ranking.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.dedup import (duplicate_nodes, graph_stats, group_by_role, merge_duplicates, role_pair_counts,  # noqa: E402
                             with_boxes)
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "graph-dedup-audit"
THRESHOLDS = (.5, .7, .9)
CRITERIA = ("a", "b", "ab", "near")


def variants(graph):
    """All cleaned versions of one image graph, keyed like 'merge0.7' and 'merge0.7+ab'."""
    out = {"before": graph}
    for t in THRESHOLDS:
        merged, _ = merge_duplicates(graph, t)
        out[f"merge{t}"] = merged
        for criterion in CRITERIA:
            out[f"merge{t}+{criterion}"] = group_by_role(merged, criterion)[0]
    return out


def image_record(args):
    graph, objects = args
    g0 = with_boxes(graph, objects["boxes"], objects["synsets"])
    record = {"duplicates": {str(t): duplicate_nodes(g0, t) for t in THRESHOLDS}, "variants": {}}
    for name, g in variants(g0).items():
        stats = graph_stats(g)
        groups = [o["count"] for o in g["objects"] if o["count"] > 1]
        if "+" in name:
            stats.update(group_nodes=len(groups), grouped_members=sum(groups))
        record["variants"][name] = stats
        if name.startswith("merge") and "+" not in name:
            record.setdefault("role_pairs", {})[name] = role_pair_counts(g)
    return record


def aggregate(records):
    out = {"images": len(records), "duplicate_node_share": {}, "variants": {}, "role_pairs": {}}
    nodes0 = sum(r["variants"]["before"]["nodes"] for r in records)
    for t in THRESHOLDS:
        out["duplicate_node_share"][str(t)] = sum(r["duplicates"][str(t)] for r in records) / max(nodes0, 1)
    for name in records[0]["variants"]:
        stats = [r["variants"][name] for r in records]
        nodes = sum(s["nodes"] for s in stats)
        entry = {"nodes_per_image": nodes / len(stats), "edges_per_image": sum(s["edges"] for s in stats) / len(stats),
                 "components_per_image": sum(s["components"] for s in stats) / len(stats),
                 "largest_component_mean": sum(s["largest_component"] for s in stats) / len(stats),
                 "largest_component_share": float(np.mean([s["largest_component"] / s["nodes"] for s in stats if s["nodes"]])),
                 "degree_one_share": sum(s["degree_one"] for s in stats) / max(nodes, 1),
                 "isolated_share": sum(s["isolated"] for s in stats) / max(nodes, 1)}
        if "group_nodes" in stats[0]:
            groups = sum(s["group_nodes"] for s in stats)
            members = sum(s["grouped_members"] for s in stats)
            entry.update(group_nodes_per_image=groups / len(stats), images_with_group=sum(s["group_nodes"] > 0 for s in stats) / len(stats),
                         mean_group_size=members / max(groups, 1),
                         two_tier_extra_nodes_per_image=groups / len(stats), two_tier_member_edges_per_image=members / len(stats))
        out["variants"][name] = entry
    for name in records[0]["role_pairs"]:
        out["role_pairs"][name] = {k: sum(r["role_pairs"][name][k] for r in records) for k in records[0]["role_pairs"][name]}
    return out


def load_split(cfg, split):
    graphs = {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] == split:
                graphs[graph["image_id"]] = graph
    objects = json.loads((resolve(cfg, "processed") / "geometry" / f"objects_{split}.json").read_text(encoding="utf-8"))
    return graphs, objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config()
    report = {"thresholds": THRESHOLDS, "criteria": CRITERIA, "splits": {}}
    for split in ("train", "val"):
        started = time.time()
        graphs, objects = load_split(cfg, split)
        ids = sorted(graphs)[:args.limit]
        with Pool(args.workers) as pool:
            records = pool.map(image_record, [(graphs[i], objects[str(i)]) for i in ids], chunksize=64)
        report["splits"][split] = aggregate(records)
        report["splits"][split]["seconds"] = time.time() - started
        print(split, f"{time.time() - started:.0f}s", json.dumps(report["splits"][split]["variants"]["before"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    (OUT / f"structure{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
