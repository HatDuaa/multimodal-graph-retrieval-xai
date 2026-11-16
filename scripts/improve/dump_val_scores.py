"""Val scores and encoded parts of trained runs, for CPU probes that must not hold the GPU (graph matching, step 2).

Per run, writes data/processed/graph_matching/<run>.npz (gitignored; outside the data version folders):
- fused, objects, triples: (queries, 50) final score and raw channel scores, rows in candidates_val order;
- image_nodes / image_node_offsets, image_triples / image_triple_offsets: encoded parts of every val image graph in
  packaged object and relation order (image_ids gives the image order);
- query_nodes / query_node_offsets, query_triples / query_triple_offsets: the same for every val caption graph
  (scorable nodes only, as the ranking sees them; caption order = candidates_val).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def flat(parts, mask):
    counts = mask.sum(-1).cpu().numpy()
    return parts[mask].float().cpu().numpy(), np.concatenate([[0], np.cumsum(counts)])


@torch.no_grad()
def encode_all(model, index, positions, table, device, query, chunk=512):
    nodes, node_counts, triples, triple_counts = [], [], [], []
    for start in range(0, len(positions), chunk):
        batch = index.select(positions[start:start + chunk]).to_batch(table, device)
        n, nm, t, tm = model.encode(batch, query=query)
        nodes.append(n[nm].float().cpu().numpy())
        node_counts.append(nm.sum(-1).cpu().numpy())
        triples.append(t[tm].float().cpu().numpy())
        triple_counts.append(tm.sum(-1).cpu().numpy())
    offsets = lambda counts: np.concatenate([[0], np.cumsum(np.concatenate(counts))])  # noqa: E731
    return np.concatenate(nodes), offsets(node_counts), np.concatenate(triples), offsets(triple_counts)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(cfg, splits=("val",))
    store.precompute()
    arrays = store.candidates("val")
    caption_ids = [str(c) for c in arrays["caption_ids"]]
    image_ids = sorted(store.graphs)
    table = store.part_table(device)
    out_dir = resolve(load_config(), "processed") / "graph_matching"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        model = load_run_model(Path(run), device)[0]
        scores = {"fused": [], "objects": [], "triples": []}
        for start in range(0, len(caption_ids), 256):
            fused, details = model(store.batch(caption_ids[start:start + 256], device))
            scores["fused"].append(fused.cpu().numpy())
            scores["objects"].append(details["objects"].cpu().numpy())
            scores["triples"].append(details["triples"].cpu().numpy())
        image = encode_all(model, store.image_index, [store.image_position[i] for i in image_ids], table, device, False)
        query = encode_all(model, store.query_index, [store.query_position[c] for c in caption_ids], table, device, True)
        np.savez(out_dir / f"{Path(run).name}.npz", caption_ids=np.asarray(caption_ids), image_ids=np.asarray(image_ids),
                 candidate_ids=arrays["candidate_ids"], gold=arrays["gold"],
                 **{k: np.concatenate(v) for k, v in scores.items()},
                 image_nodes=image[0], image_node_offsets=image[1], image_triples=image[2], image_triple_offsets=image[3],
                 query_nodes=query[0], query_node_offsets=query[1], query_triples=query[2], query_triple_offsets=query[3])
        print(run, "saved", flush=True)


if __name__ == "__main__":
    main()
