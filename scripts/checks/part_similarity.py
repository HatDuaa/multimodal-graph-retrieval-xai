"""Spread of encoded part vectors: mean pairwise similarity of object and triple vectors of val image graphs.

Compares the frozen CLIP part vectors (the untrained step-0 model) with trained checkpoints. A mean close to 1
means the vectors of different parts have become nearly identical, so the channel can only rank by tiny
differences that the candidate-wise z-score then magnifies.

Triples are compared with the similarity their channel uses: cosine for phrase vectors, c(s) * c(p) * c(o) for
factorized triples (whose per-slot cosines are reported too). Graph-channel models also report the spread of the
attention-pooled image graph vectors.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.models.graph_reranker import GraphReranker, attention_pool, slot_similarity  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

RUNS = Path("experiments/level1/gat")
LEVEL1_RUNS = [RUNS / f"main_r3_seed{k}" for k in range(3)] + [RUNS / f"no_gat_r3_seed{k}" for k in range(3)]
STEP0 = {"phrase": {}, "factorized": {"triple_match": "factorized"}, "graph": {"graph_channel": True}}


def spread(vectors, pairs=3000, similarity=None):
    generator = torch.Generator().manual_seed(0)
    i = torch.randint(len(vectors), (pairs,), generator=generator)
    j = torch.randint(len(vectors), (pairs,), generator=generator)
    keep = i != j
    a, b = vectors[i[keep]], vectors[j[keep]]
    sims = (a * b).sum(-1) if similarity is None else similarity(a, b)
    return {"mean": float(sims.mean()), "p05": float(sims.quantile(.05)), "p95": float(sims.quantile(.95))}


def factorized_similarity(a, b):
    return slot_similarity((a * b).sum(-1)).prod(-1)


def measure(model, batch):
    nodes, node_mask, triples, triple_mask = model.encode(batch)
    out = {"objects": spread(nodes[node_mask].cpu())}
    if model.options["graph_channel"]:
        graphs, _ = attention_pool(nodes, node_mask, model.u_pool_image)
        out["graph_vectors"] = spread(graphs[node_mask.any(-1)].cpu())
        return out
    triples = triples[triple_mask].cpu()
    if model.options["triple_match"] == "factorized":
        out["triples"] = spread(triples, similarity=factorized_similarity)
        out["triple_slots"] = {slot: spread(triples[:, k]) for k, slot in enumerate(("subject", "predicate", "object"))}
    else:
        out["triples"] = spread(triples)
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", type=Path, default=LEVEL1_RUNS)
    parser.add_argument("--step0", nargs="*", choices=sorted(STEP0), default=["phrase"],
                        help="Untrained reference models (u = 0: frozen CLIP parts).")
    parser.add_argument("--out", type=Path, default=Path("experiments/checks/part_similarity.json"))
    parser.add_argument("--config", type=Path, help="Data configuration (default configs/default.yaml).")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(load_config(args.config), splits=("val",))
    image_ids = sorted(store.graphs)[:300]
    models = {("step0_frozen_clip" if kind == "phrase" else f"step0_{kind}"):
              GraphReranker(adaptive_weights=False, **STEP0[kind]).to(device).eval() for kind in args.step0}
    models.update({run.name: run for run in args.runs})
    result = {"images": len(image_ids), "pairs": 3000}
    for name, model in models.items():
        if isinstance(model, Path):
            model = load_run_model(model, device, store=store)[0]  # also sets the run's graph structure
        else:
            store.set_structure()
        store.precompute()
        batch = store.image_index.select([store.image_position[i] for i in image_ids]).to_batch(store.part_table(device), device)
        result[name] = measure(model, batch)
        print(name, json.dumps(result[name]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
