"""Sweep the CLIP weight a on val for a fixed-weights checkpoint (evaluation only).

The fixed_weights model has W = 0, so its channel weights are softmax(bias) for every query. The sweep keeps the
learned graph split beta = b / (b + c) and sets a on a grid, which is the plan's "choose alpha on validation".
The val number at the chosen a is optimistic, because val also chose it; the learned-a number is reported apart.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.train_graph_reranker import evaluate_val, load_run_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--grid", nargs="+", type=float, default=[round(x, 2) for x in np.arange(0.05, 0.96, 0.05)])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/level1/controls"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    store = GraphStore()
    for run in args.runs:
        output = args.output_dir / f"alpha_sweep_{run}.json"
        if output.exists():
            print(f"skip existing {output}", flush=True)
            continue
        model, epoch, _ = load_run_model(Path("experiments/level1/gat") / run, device)
        if model.options["adaptive_weights"] or bool(model.weight_map.detach().abs().sum()):
            raise SystemExit(f"{run} is not a fixed-weights run (W must be zero)")
        learned = torch.softmax(model.weight_bias.detach(), -1).cpu().numpy()
        beta = float(learned[1] / (learned[1] + learned[2]))
        rows = []
        for a in sorted(set(args.grid) | {round(float(learned[0]), 6)}):
            weights = torch.tensor([a, (1 - a) * beta, (1 - a) * (1 - beta)], dtype=torch.float32)
            with torch.no_grad():
                model.weight_bias.copy_(weights.log().to(device))
                metrics = evaluate_val(model, store, store.candidates("val"), device)
            rows.append({"a": a, **{k: metrics[k] for k in ("recall@1", "recall@5", "recall@10", "mrr")}})
            print(run, json.dumps(rows[-1]), flush=True)
        best = max(rows, key=lambda row: (row["recall@1"], row["mrr"]))
        learned_row = min(rows, key=lambda row: abs(row["a"] - float(learned[0])))
        result = {"run": run, "checkpoint_epoch": epoch, "learned_abc": learned.tolist(), "beta": beta,
                  "learned_a_val": learned_row, "best_a_val": best, "grid": rows,
                  "note": "best_a_val is selected on val, so its val numbers are optimistic"}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
