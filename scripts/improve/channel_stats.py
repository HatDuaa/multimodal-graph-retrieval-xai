"""Per-query spread of each channel's raw scores at step 0, on a seeded sample of train queries (no val, no test).

The untrained model scores with frozen CLIP parts (u = 0), so these are the channel scales the training starts from.
The improvement runs take two constants from here, both fixed before training and never tuned on val:
- --fusion raw: --fusion-scale = median per-query std of (CLIP, objects, triples), for a start equivalent to z-scoring;
- --zscore-floor: the 5th percentile of the per-query std of each channel, i.e. a spread that 95% of train queries
  exceed at step 0; a trained channel squeezed below it is no longer stretched back to unit spread.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import rows_for  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

CHANNELS = ("clip", "objects", "triples")


def row_std(values):
    known = np.isfinite(values)
    out = []
    for row, mask in zip(values, known):
        if mask.sum() >= 2:
            out.append(float(row[mask].std()))
    return np.asarray(out)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(cfg, splits=("train",))
    rows = [cid for _, cid in rows_for(store, store.candidates("train"))]
    rng = np.random.default_rng(cfg["seed"])
    sample = [rows[i] for i in rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False)]
    set_seed(0)
    model = GraphReranker().to(device).eval()
    collected = {c: [] for c in CHANNELS}
    for start in range(0, len(sample), 256):
        batch = store.batch(sample[start:start + 256], device)
        _, details = model(batch)
        for name, values in (("clip", batch["clip"]), ("objects", details["objects"]), ("triples", details["triples"])):
            collected[name].append(row_std(values.float().cpu().numpy()))
    result = {"config": str(args.config or "configs/default.yaml"), "split": "train", "sample": len(sample),
              "seed": cfg["seed"], "model": "GraphReranker() at initialisation (u = 0: frozen CLIP parts, tau 0.05)",
              "per_query_std": {}}
    for name in CHANNELS:
        stds = np.concatenate(collected[name])
        result["per_query_std"][name] = {"queries": int(len(stds)), "mean": float(stds.mean()),
                                         **{f"p{q:02d}": float(np.percentile(stds, q)) for q in (1, 5, 10, 25, 50, 75)}}
    stats = result["per_query_std"]
    result["fusion_scale"] = [stats[c]["p50"] for c in CHANNELS]
    result["zscore_floor"] = [stats[c]["p05"] for c in CHANNELS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
