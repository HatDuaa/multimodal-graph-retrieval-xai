"""Evaluation-only graph-corruption controls on val for trained reranker checkpoints.

other_image: every val image is scored with the scene graph of another val image (a fixed derangement per seed).
rewire:      every image graph has its relation targets permuted (degree-preserving, label travels with the edge).
Nothing is trained and the test split is never loaded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.corruption import graph_derangement  # noqa: E402
from src.train_graph_reranker import evaluate_val, load_run_model  # noqa: E402

RUNS = Path("experiments/level1/gat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--control", choices=("other_image", "rewire"), required=True)
    parser.add_argument("--control-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/level1/controls"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    store = GraphStore()
    original = dict(store.graphs)
    val_images = sorted({int(i) for i in store.candidates("val")["candidate_ids"].ravel()})
    for run in args.runs:
        output = args.output_dir / f"{args.control}_{run}.json"
        if output.exists():
            print(f"skip existing {output}", flush=True)
            continue
        model, epoch, saved = load_run_model(RUNS / run, device)
        started = time.time()
        with torch.no_grad():
            clean = evaluate_val(model, store, store.candidates("val"), device, args.eval_batch_size)
        result = {"run": run, "control": args.control, "checkpoint_epoch": epoch, "device": str(device),
                  "saved_best_val": {k: saved[k] for k in ("recall@1", "recall@5", "recall@10", "mrr")},
                  "clean_val": clean, "corrupted_val": {}}
        for seed in args.control_seeds:
            if args.control == "other_image":
                mapping = graph_derangement(val_images, seed)
                store.replace_image_graphs({i: original[mapping.get(i, i)] for i in original})
            else:
                store.use_rewired_graphs(seed)
            with torch.no_grad():
                result["corrupted_val"][str(seed)] = evaluate_val(model, store, store.candidates("val"), device,
                                                                  args.eval_batch_size)
            store.replace_image_graphs(original)
            print(run, args.control, seed, json.dumps({k: result["corrupted_val"][str(seed)][k]
                                                        for k in ("recall@1", "mrr")}), flush=True)
        result["seconds"] = time.time() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
