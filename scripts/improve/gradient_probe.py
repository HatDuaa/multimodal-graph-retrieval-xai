"""Gradient norm per module at a run's best checkpoint, on a fixed set of train batches (experiment 3).

The training log keeps the norms of the last batch of each epoch only, one noisy number. Here every run gets the
same 10 train batches of 256 queries (the first 2,560 of the seed-0 epoch-1 order, micro-batches of 32 as in
training) and the training loss cross_entropy(fused / T); per batch, the norm of each module's gradient
(parameters grouped by their first name part, as in the log), then mean and std over the batches. The model is in
eval mode (no dropout) so the numbers do not depend on dropout masks; every parameter takes gradients (the epoch-1
freeze of the weight map does not apply). Each run is read with its own data configuration and graph structure.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.train_graph_reranker import batch_loss, load_run_model, rows_for, shuffled_rows  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402


def module_norms(model):
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            module = name.split('.')[0]
            norms[module] = norms.get(module, 0.) + float(parameter.grad.detach().norm()) ** 2
    return {name: value ** .5 for name, value in norms.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", nargs="+", action="append", required=True, help="NAME RUN_DIR [RUN_DIR ...]")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stores, result = {}, {"batches": args.batches, "batch_size": args.batch_size, "per_run": {}, "aggregate": {}}
    for name, *runs in args.group:
        for run in runs:
            config = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))["args"].get("config")
            key = str(config or "configs/default.yaml")
            if key not in stores:
                stores.clear()  # one part table in memory at a time
                stores[key] = GraphStore(load_config(REPO_ROOT / key if config else None), splits=("train",))
            store = stores[key]
            model = load_run_model(Path(run), device, store=store)[0]
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            rows = shuffled_rows(list(rows_for(store, store.candidates("train"))), 0, 1)[:args.batches * args.batch_size]
            per_batch = []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                model.zero_grad(set_to_none=True)
                for micro in range(0, len(batch), args.micro_batch_size):
                    part = batch[micro:micro + args.micro_batch_size]
                    loss = batch_loss(model, store, [cid for _, cid in part], device)
                    (loss * (len(part) / len(batch))).backward()
                per_batch.append(module_norms(model))
            modules = sorted(per_batch[0])
            result["per_run"][run] = {m: {"mean": float(np.mean([b[m] for b in per_batch])),
                                          "std": float(np.std([b[m] for b in per_batch], ddof=1))} for m in modules}
            print(name, run, json.dumps({m: round(v["mean"], 4) for m, v in result["per_run"][run].items()}), flush=True)
        modules = sorted(set.intersection(*(set(result["per_run"][r]) for r in runs)))
        result["aggregate"][name] = {m: {"mean": float(np.mean([result["per_run"][r][m]["mean"] for r in runs])),
                                         "std": float(np.std([result["per_run"][r][m]["mean"] for r in runs], ddof=1))
                                         if len(runs) > 1 else 0.} for m in modules}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
