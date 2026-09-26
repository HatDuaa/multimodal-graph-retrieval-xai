"""Spread of encoded part vectors: mean pairwise cosine of object and triple vectors of val image graphs.

Compares the frozen CLIP part vectors (the untrained step-0 model) with trained checkpoints. A mean close to 1
means the vectors of different parts have become nearly identical, so the channel can only rank by tiny
differences that the candidate-wise z-score then magnifies.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402

RUNS = Path("experiments/level1/gat")


def spread(vectors, pairs=3000):
    generator = torch.Generator().manual_seed(0)
    i = torch.randint(len(vectors), (pairs,), generator=generator)
    j = torch.randint(len(vectors), (pairs,), generator=generator)
    sims = (vectors[i] * vectors[j]).sum(-1)[i != j]
    return {"mean": float(sims.mean()), "p05": float(sims.quantile(.05)), "p95": float(sims.quantile(.95))}


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(splits=("val",))
    store.precompute()
    image_ids = sorted(store.graphs)[:300]
    batch = store.image_index.select([store.image_position[i] for i in image_ids]).to_batch(store.part_table(device), device)
    models = {"step0_frozen_clip": GraphReranker(adaptive_weights=False).to(device).eval()}
    for name in [f"main_r3_seed{k}" for k in range(3)] + [f"no_gat_r3_seed{k}" for k in range(3)]:
        models[name] = load_run_model(RUNS / name, device)[0]
    result = {"images": len(image_ids), "pairs": 3000}
    for name, model in models.items():
        nodes, node_mask, triples, triple_mask = model.encode(batch)
        result[name] = {"objects": spread(nodes[node_mask].cpu()), "triples": spread(triples[triple_mask].cpu())}
        print(name, json.dumps(result[name]), flush=True)
    Path("experiments/checks/part_similarity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
