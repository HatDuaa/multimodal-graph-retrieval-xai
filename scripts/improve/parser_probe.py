"""Training-free three-channel probe on val with the query graphs of one parser (part 1c).

Same reranker and grid as scripts/checks/three_channel_probe.py at K = 50: CLIP top-50 candidates
(experiments/level1/clip_baseline/top50_val.json), image graphs of the val pool, one CLIP vector per object
instance ("a photo of a {}") and per relation phrase, uniform-weight softmax matching (channel_scores), z-score
fusion (fuse_rows), alpha and beta swept on a 0.1 grid for each temperature in {0.02, 0.05, 0.1}. Only the query
graphs change. Part texts missing from the level-1 val part cache are encoded with the same frozen CLIP on CPU
(the GPU stays free for the parser jobs). Reports the best grid point and the level-1 reference point
(alpha 0.6, beta 0.7, temperature 0.05) so that parsers are also compared at identical settings.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks.three_channel_probe import GRID, fuse_rows, ranked_metrics, sweep  # noqa: E402
from src.features.extract_clip import ClipEncoder, load_features  # noqa: E402
from src.graph.soft_match import channel_scores, instance_parts, query_instance_parts  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "parser-compare"
REFERENCE = {"alpha": 0.6, "beta": 0.7, "temperature": 0.05}
_SHARED = {}


def _channel(args):
    start, stop, temperature = args
    queries, images, ranked = _SHARED["queries"], _SHARED["images"], _SHARED["ranked"]
    return [channel_scores(queries[i], [images[item] for item in ranked[i]], temperature) for i in range(start, stop)]


def channel(queries, images, ranked, temperature, workers):
    _SHARED.update(queries=queries, images=images, ranked=ranked)
    chunks = [(s, min(s + 256, len(ranked)), temperature) for s in range(0, len(ranked), 256)]
    with Pool(workers) as pool:  # fork: the workers read _SHARED without copying it through pickles
        rows = [row for part in pool.map(_channel, chunks) for row in part]
    return np.asarray(rows)


def vectors_for(cfg, node_texts, triple_texts, table=None):
    """Level-1 val part cache (or another table), extended in memory by frozen-CLIP vectors of the texts it lacks."""
    base, keys, encoder_name = load_features(table or resolve(cfg, "features") / "vg_coco_val_graph_parts")
    row = {key: i for i, key in enumerate(keys)}
    missing_nodes = [t for t in node_texts if f"node:{t}" not in row]
    missing_triples = [t for t in triple_texts if f"triple:{t}" not in row]
    if not missing_nodes and not missing_triples:
        return base, row, 0
    encoder = ClipEncoder.from_config(cfg, device="cpu")
    assert encoder.name == encoder_name, (encoder.name, encoder_name)
    extra = encoder.encode_texts([cfg["clip"]["concept_prompt"].format(t) for t in missing_nodes] + missing_triples)
    for i, key in enumerate([f"node:{t}" for t in missing_nodes] + [f"triple:{t}" for t in missing_triples]):
        row[key] = len(base) + i
    return np.concatenate([base, extra]), row, len(extra)


def lookup(vectors, row, kind, parts):
    return vectors[[row[f"{kind}:{p}"] for p in parts]] if parts else np.empty((0, vectors.shape[1]), vectors.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--temperatures", default="0.02,0.05,0.1")
    parser.add_argument("--queries", type=Path, help="Query graphs JSON (default: the parser's val parse).")
    parser.add_argument("--table", type=Path, help="Part vector table stem (default: the level-1 val part cache).")
    parser.add_argument("--out-name", help="Output file name (default probe_<parser>.json).")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    queries_path = args.queries or resolve(cfg, "processed") / "parsers" / f"query_graphs_val_{args.parser}.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    rows = json.loads((REPO_ROOT / "experiments" / "level1" / "clip_baseline" / "top50_val.json").read_text(encoding="utf-8"))
    caption_ids = [r["caption_id"] for r in rows]
    ranked, clip = [r["ranked"] for r in rows], np.asarray([r["scores"] for r in rows])
    images = json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
    gold_of = {c["caption_id"]: im["image_id"] for im in images for c in im["captions"]}
    golds = [gold_of[c] for c in caption_ids]
    graphs = {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] == "val":
                graphs[graph["image_id"]] = graph
    query_parts = [query_instance_parts(queries[c]) for c in caption_ids]
    image_parts = {i: instance_parts(g) for i, g in graphs.items()}
    node_texts = sorted({p for nodes, _ in query_parts for p in nodes} | {p for nodes, _ in image_parts.values() for p in nodes})
    triple_texts = sorted({p for _, ts in query_parts for p in ts} | {p for _, ts in image_parts.values() for p in ts})
    vectors, row, encoded = vectors_for(cfg, node_texts, triple_texts, args.table)
    q_nodes = [lookup(vectors, row, "node", n) for n, _ in query_parts]
    q_triples = [lookup(vectors, row, "triple", t) for _, t in query_parts]
    i_nodes = {i: lookup(vectors, row, "node", p[0]) for i, p in image_parts.items()}
    i_triples = {i: lookup(vectors, row, "triple", p[1]) for i, p in image_parts.items()}
    report = {"parser": args.parser, "split": "val", "queries": len(caption_ids), "newly_encoded_parts": encoded,
              "query_graphs": str(queries_path), "table": str(args.table or "data/features/vg_coco_val_graph_parts"),
              "queries_without_objects": sum(not n for n, _ in query_parts),
              "queries_without_triples": sum(not t for _, t in query_parts),
              "clip_only": ranked_metrics(clip, ranked, golds)[0], "temperatures": {}}
    for temperature in [float(t) for t in args.temperatures.split(",")]:
        objects = channel(q_nodes, i_nodes, ranked, temperature, args.workers)
        triples = channel(q_triples, i_triples, ranked, temperature, args.workers)
        best, grid, _ = sweep(clip, objects, triples, ranked, golds)
        entry = {"best": best, "r1_grid": grid}
        if temperature == REFERENCE["temperature"]:
            fused, _ = fuse_rows(clip, objects, triples, REFERENCE["alpha"], REFERENCE["beta"])
            report["reference_point"] = {**REFERENCE, **ranked_metrics(fused, ranked, golds)[0]}
            for name, beta, obj, tri in (("objects_only", 1.0, objects, np.full_like(triples, np.nan)),
                                         ("triples_only", 0.0, np.full_like(objects, np.nan), triples)):
                report[name] = max(({"alpha": a, **ranked_metrics(fuse_rows(clip, obj, tri, a, beta)[0], ranked, golds)[0]}
                                    for a in GRID), key=lambda m: (m["recall@1"], m["mrr"]))
        report["temperatures"][str(temperature)] = entry
        print(args.parser, temperature, json.dumps({k: best[k] for k in ("alpha", "beta", "recall@1", "mrr")}), flush=True)
    report["best"] = max(({"temperature": float(t), **e["best"]} for t, e in report["temperatures"].items()),
                         key=lambda m: (m["recall@1"], m["mrr"]))
    report["runtime_seconds"] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (args.out_name or f"probe_{args.parser}.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best", "reference_point")}), flush=True)


if __name__ == "__main__":
    main()
