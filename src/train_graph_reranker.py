"""Train the graph reranker on train-group candidates and select by validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_store import GraphStore
from src.eval.metrics import evaluate
from src.models.graph_reranker import GraphReranker
from src.utils.config import load_config, resolve
from src.utils.seed import set_seed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(store, arrays, limit=None):
    keep = arrays["gold_rank"] >= 0
    indices = np.flatnonzero(keep)
    if limit:
        indices = indices[:limit]
    for row in indices:
        yield int(row), str(arrays["caption_ids"][row])


def batch_loss(model, store, arrays, indices, device):
    queries = [store.query(str(arrays["caption_ids"][row])) for row in indices]
    image_ids = sorted({int(image_id) for row in indices for image_id in arrays["candidate_ids"][row] if int(image_id) >= 0})
    image_row = {image_id: index for index, image_id in enumerate(image_ids)}
    batch = {'queries': store.batch_graphs(queries, device), 'images': store.batch_graphs([store.image(i) for i in image_ids], device),
             'candidate_index': torch.tensor([[image_row[int(i)] for i in arrays["candidate_ids"][row]] for row in indices], dtype=torch.long, device=device),
             'clip': torch.from_numpy(arrays["clip_scores"][indices]).to(device),
             'sentence': torch.stack([store.sentence(str(arrays["caption_ids"][row])) for row in indices]).to(device)}
    scores, details = model(batch)
    targets = torch.tensor([int(arrays["gold_rank"][row] - 1) for row in indices], device=device)
    return F.cross_entropy(scores / model.loss_temperature(), targets)


@torch.no_grad()
def evaluate_val(model, store, arrays, device):
    ranked, golds, weights = [], [], []
    for row in range(len(arrays["caption_ids"])):
        caption_id = str(arrays["caption_ids"][row])
        query = store.query(caption_id)
        ids = [int(image_id) for image_id in arrays["candidate_ids"][row] if int(image_id) >= 0]
        images = [store.image(image_id) for image_id in ids]
        clip = torch.from_numpy(arrays["clip_scores"][row, :len(ids)]).to(device)
        batch = {'queries': store.batch_graphs([query], device), 'images': store.batch_graphs(images, device),
                 'candidate_index': torch.arange(len(ids), device=device, dtype=torch.long)[None],
                 'clip': clip[None], 'sentence': store.sentence(caption_id).to(device)[None]}
        scores, details = model(batch)
        scores = scores.squeeze(0)
        order = torch.argsort(scores.squeeze(0), descending=True, stable=True).cpu().tolist()
        ranked.append([ids[index] for index in order])
        golds.append(int(arrays["gold"][row]))
        weights.append(details["weights"].cpu().numpy().squeeze(0))
    metrics = evaluate(ranked, golds)
    weights = np.asarray(weights)
    metrics["weights_mean"] = weights.mean(axis=0).tolist() if len(weights) else [0, 0, 0]
    metrics["weights_std"] = weights.std(axis=0).tolist() if len(weights) else [0, 0, 0]
    metrics["tau_objects"] = float(model.tau_objects())
    metrics["tau_triples"] = float(model.tau_triples())
    metrics["temperature"] = float(model.loss_temperature())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adaptive-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-triple-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-graph-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-only", type=Path)
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store, model = GraphStore(cfg), GraphReranker(args.adaptive_weights, args.use_gat, args.use_triple_channel,
                                                   args.use_edges, args.query_graph_encoder).to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device)["model"])
    train_arrays, val_arrays = store.candidates("train"), store.candidates("val")
    run_dir = resolve(cfg, "experiments") / "level1" / "gat" / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.eval_only:
        raise FileExistsError(f"run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model.eval(); print(json.dumps(evaluate_val(model, store, val_arrays, device))); return
    if not args.adaptive_weights:
        model.weight_map.requires_grad_(False)
    model.weight_bias.requires_grad_(False)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("weight_map") and not args.adaptive_weights: continue
        (no_decay if name.startswith(("log_tau", "log_temperature", "weight_bias")) else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0}], lr=args.lr)
    metadata = {"args": vars(args), "config": cfg, "device": str(device), "train_candidates_sha256": sha256(
        Path(cfg["paths"]["processed"]) / "candidates_train.npz"), "val_candidates_sha256": sha256(
        Path(cfg["paths"]["processed"]) / "candidates_val.npz")}
    try: metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: metadata["git_commit"] = "unknown"
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log = run_dir / "log.jsonl"
    best, stale = -1.0, 0
    last_metrics = None
    rows = list(rows_for(store, train_arrays, args.limit_train))
    for epoch in range(1, args.epochs + 1):
        started = time.time(); model.train()
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, store, train_arrays, [row for row, _ in batch], device)
            if loss is None: continue
            loss.backward(); optimizer.step()
        if epoch == 1:
            for parameter in ((model.weight_map,) if args.adaptive_weights else ()) + (model.weight_bias,):
                parameter.requires_grad_(True)
        model.eval(); metrics = evaluate_val(model, store, val_arrays, device)
        gradient_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                module = name.split('.')[0]
                gradient_norms[module] = gradient_norms.get(module, 0.) + float(parameter.grad.detach().norm()) ** 2
        gradient_norms = {name: value ** .5 for name, value in gradient_norms.items()}
        metrics.update({"epoch": epoch, "seconds": time.time() - started, "train_queries": len(rows),
                        "gradient_norms": gradient_norms,
                        "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        last_metrics = metrics
        with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        score = metrics["recall@1"]
        if score > best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, run_dir / "checkpoint_best.pt")
        else:
            stale += 1
            if stale >= 2: break
    if last_metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
