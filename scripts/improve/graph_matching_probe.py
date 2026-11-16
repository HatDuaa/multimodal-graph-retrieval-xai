"""Training-free whole-graph matching channels on val (improvement step 2), CPU only.

Scores every (val caption, top-50 candidate) pair with src/graph/graph_matching.py on data version factual_v1:
fgw (balanced FGW), srfgw (semi-relaxed FGW), consistency (shared-node consistency of best-matching triples), each with
frozen CLIP part vectors ("frozen") and with the encoded parts of main_factual_r3 seed k ("seed<k>", from
scripts/improve/dump_val_scores.py). FGW alpha 0.5 and the hop cap 3 are fixed in advance, not tuned.
Caption graphs are taken as the ranking sees them: scorable nodes (no pronoun, no empty label) and the relations
between them. Scores go to data/processed/graph_matching/scores_<vectors>_<method>.npy (rows = candidates_val order).
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.features.extract_clip import load_features  # noqa: E402
from src.graph.graph_matching import consistency_score, fgw_score, hop_distance, shared_node_pairs  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

METHODS = ("fgw", "srfgw", "consistency")
G = {}


def structures(cfg):
    queries = json.loads((resolve(cfg, "processed") / "query_graphs_val.json").read_text(encoding="utf-8"))
    images = {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] == "val":
                images[graph["image_id"]] = graph
    return queries, images


def query_structure(graph):
    parts = GraphStore._graph_parts(graph, query=True)
    scorable = [i for i, keep in enumerate(parts["part_mask"]) if keep]
    position = {old: new for new, old in enumerate(scorable)}
    kept = [(k, position[s], position[o]) for k, (s, o, _) in enumerate(parts["edges"]) if s in position and o in position]
    return {"labels": [parts["node_labels"][i] for i in scorable], "edges": [(s, o) for _, s, o in kept],
            "edge_rows": [k for k, _, _ in kept], "triples": [parts["triple_labels"][k] for k, _, _ in kept],
            "node_rows": scorable}


def image_structure(graph):
    parts = GraphStore._graph_parts(graph)
    edges = [(s, o) for s, o, _ in parts["edges"]]
    return {"labels": parts["node_labels"], "edges": edges, "triples": parts["triple_labels"],
            "dist": hop_distance(len(parts["node_labels"]), edges)}


def _rows(chunk):
    start, stop = chunk
    out = []
    for qi in range(start, stop):
        q = G["queries"][qi]
        qn, qt = G["qnodes"][qi], G["qtriples"][qi]
        qd = hop_distance(len(q["labels"]), q["edges"]) if G["method"] != "consistency" else None
        pairs = shared_node_pairs(q["edges"]) if G["method"] == "consistency" else None
        row = []
        for image_id in G["candidates"][qi]:
            im = G["images"][int(image_id)]
            if G["method"] == "consistency":
                row.append(consistency_score(pairs, qt, G["itriples"][int(image_id)], im["edges"]))
            else:
                mode = "balanced" if G["method"] == "fgw" else "semirelaxed"
                row.append(fgw_score(qn, qd, G["inodes"][int(image_id)], im["dist"], mode=mode)[0])
        out.append(row)
    return out


def frozen_vectors(cfg, queries, images):
    vectors, keys, _ = load_features(resolve(cfg, "features") / "vg_coco_graph_parts")
    row = {k: i for i, k in enumerate(keys)}

    def get(kind, labels):
        if not labels:
            return np.zeros((0, vectors.shape[1]), np.float32)
        return np.asarray(vectors[[row[f"{kind}:{x}"] for x in labels]], dtype=np.float32)

    return ([get("node", q["labels"]) for q in queries], [get("triple", q["triples"]) for q in queries],
            {i: get("node", im["labels"]) for i, im in images.items()}, {i: get("triple", im["triples"]) for i, im in images.items()})


def trained_vectors(dump, queries, images):
    """Encoded parts of one run: scorable caption nodes in order, caption triples by relation row, image parts."""
    d = dict(np.load(dump))

    def piece(name, k):
        offsets = d[name[:-1] + "_offsets"]  # query_nodes -> query_node_offsets
        return d[name][offsets[k]:offsets[k + 1]]

    order = {int(i): k for k, i in enumerate(d["image_ids"])}
    return ([piece("query_nodes", k) for k in range(len(queries))],
            [piece("query_triples", k)[q["edge_rows"]] for k, q in enumerate(queries)],
            {i: piece("image_nodes", order[i]) for i in images}, {i: piece("image_triples", order[i]) for i in images})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/factual_v1.yaml"))
    parser.add_argument("--vectors", nargs="+", default=["frozen", "seed0", "seed1", "seed2"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, help="Smoke test only: the first N val queries.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = resolve(load_config(), "processed") / "graph_matching"
    dump0 = np.load(out_dir / "main_factual_r3_seed0.npz")
    caption_ids = [str(c) for c in dump0["caption_ids"]][:args.limit]
    raw_queries, raw_images = structures(cfg)
    queries = [query_structure(raw_queries[c]) for c in caption_ids]
    images = {i: image_structure(g) for i, g in raw_images.items()}
    timing = {}
    for vectors in args.vectors:
        if vectors == "frozen":
            qnodes, qtriples, inodes, itriples = frozen_vectors(cfg, queries, images)
        else:
            qnodes, qtriples, inodes, itriples = trained_vectors(out_dir / f"main_factual_r3_{vectors}.npz", queries, images)
        for method in args.methods:
            started = time.time()
            G.update(queries=queries, images=images, qnodes=qnodes, qtriples=qtriples, inodes=inodes, itriples=itriples,
                     candidates=dump0["candidate_ids"][:len(caption_ids)], method=method)
            chunks = [(s, min(s + 64, len(queries))) for s in range(0, len(queries), 64)]
            with Pool(args.workers) as pool:
                scores = np.asarray([r for part in pool.map(_rows, chunks) for r in part], dtype=np.float64)
            suffix = f"_limit{args.limit}" if args.limit else ""
            np.save(out_dir / f"scores_{vectors}_{method}{suffix}.npy", scores)
            timing[f"{vectors}/{method}"] = time.time() - started
            print(vectors, method, f"{timing[f'{vectors}/{method}']:.0f}s", "finite", float(np.isfinite(scores).mean()), flush=True)
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
