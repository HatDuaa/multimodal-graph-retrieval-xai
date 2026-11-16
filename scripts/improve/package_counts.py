"""Package data version count_v1: merge_v1 image graphs with a count per node, no grouping (experiment 5).

Every node of a merge_v1 graph stays one thing with its own label. Its count is the number of same-type instances in
the image after merging duplicates, itself included ("same type" as in the graph-cleaning audit, src/graph/dedup.py
same_type), bucketed 1 / 2 / 3-4 / >=5 (src/graph/counts.py). To save disk, the graphs are not copied: only
data/graphs/count_v1/count_buckets.json ({image_id: {"counts": [...], "buckets": [...]}}, node order of merge_v1) and
MANIFEST.json are written, never overwriting. configs/count_v1.yaml reads the merge_v1 graphs with these buckets
(GraphStore, paths.count_buckets) and the group_v1 caption graphs, which carry the caption count classes.
"""
import argparse
import collections
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.package_factual import fresh, sha256  # noqa: E402
from src.graph.counts import IMAGE_BUCKETS, image_bucket, image_counts  # noqa: E402
from src.graph.dedup import same_type  # noqa: E402
from src.utils.config import REPO_ROOT  # noqa: E402

VERSION = "count_v1"


def counts_of(graph):
    counts = image_counts(graph["objects"], same_type)
    return graph["image_id"], graph["split"], counts, [image_bucket(c) for c in counts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    source = REPO_ROOT / "data" / "graphs" / "merge_v1" / "vg_coco_scene_graphs.jsonl"
    out = REPO_ROOT / "data" / "graphs" / VERSION
    out.mkdir(parents=True, exist_ok=True)
    target = fresh(out / "count_buckets.json")
    graphs = [json.loads(line) for line in source.open(encoding="utf-8")]
    with Pool(args.workers) as pool:
        rows = pool.map(counts_of, graphs, chunksize=64)
    target.write_text(json.dumps({str(i): {"counts": c, "buckets": b} for i, _, c, b in rows}), encoding="utf-8")
    manifest = {"version": VERSION, "date": time.strftime("%F %T %z"), "base": "merge_v1",
                "source_graphs": str(source.relative_to(REPO_ROOT)), "source_graphs_sha256": sha256(source),
                "rule": "count = same-type instances in the image after merging (src/graph/counts.py image_counts), labels kept",
                "image_buckets": IMAGE_BUCKETS, "splits": {}}
    for split in ("train", "val"):
        buckets = collections.Counter(b for _, s, _, bs in rows if s == split for b in bs)
        nodes = sum(buckets.values())
        manifest["splits"][split] = {"images": sum(s == split for _, s, _, _ in rows), "nodes": nodes,
                                     "node_buckets": {IMAGE_BUCKETS[k]: buckets[k] for k in range(len(IMAGE_BUCKETS))},
                                     "node_bucket_shares": {IMAGE_BUCKETS[k]: buckets[k] / nodes for k in range(len(IMAGE_BUCKETS))}}
        print(split, json.dumps(manifest["splits"][split]), flush=True)
    manifest["seconds"] = time.time() - started
    fresh(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
