"""Train the graph reranker on train-group candidates and select by validation only."""
from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

# Variable-size graph batches fragment the default CUDA cache (reserved ~1.5x allocated without this).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def shuffled_rows(rows, seed, epoch):
    """Deterministic per-epoch order, independent of process hash/random state."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    return [rows[index] for index in rng.permutation(len(rows))]


def lambda_mrr_loss(scores, target):
    """LambdaRank-style pairwise loss for one relevant candidate, weighted by the change in reciprocal rank.

    Each (gold, other) pair costs log(1 + exp(-(s_gold - s_other))), weighted by |1/rank_gold - 1/rank_other|
    under the current ranking (no gradient through the weights); weights are normalised per query.
    """
    ranks = scores.detach().argsort(-1, descending=True).argsort(-1).float() + 1
    gold_rank = ranks.gather(1, target[:, None])
    weights = (1 / gold_rank - 1 / ranks).abs()
    weights = weights.scatter(1, target[:, None], 0.)
    margins = scores.gather(1, target[:, None]) - scores
    pair = F.softplus(-margins) * weights
    return (pair.sum(-1) / weights.sum(-1).clamp_min(1e-12)).mean()


def batch_loss(model, store, caption_ids, device, loss="cross_entropy"):
    batch = store.batch(caption_ids, device)
    scores, _ = model(batch)
    if loss == "lambda_mrr":
        return lambda_mrr_loss(scores / model.loss_temperature(), batch['target'])
    return F.cross_entropy(scores / model.loss_temperature(), batch['target'])


@torch.no_grad()
def evaluate_val(model, store, arrays, device, batch_size=256):
    ranked, golds, weights = [], [], []
    caption_ids = [str(cid) for cid in arrays["caption_ids"]]
    for start in range(0, len(caption_ids), batch_size):
        batch = store.batch(caption_ids[start:start + batch_size], device)
        scores, details = model(batch)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True).cpu().numpy()
        ranked.extend(np.take_along_axis(batch['candidate_ids'], order, axis=1).tolist())
        golds.extend(batch['gold'])
        weights.append(details["weights"].cpu().numpy())
    metrics = evaluate(ranked, golds)
    weights = np.concatenate(weights) if weights else np.zeros((0, 3))
    metrics["weights_mean"] = weights.mean(axis=0).tolist() if len(weights) else [0, 0, 0]
    metrics["weights_std"] = weights.std(axis=0).tolist() if len(weights) else [0, 0, 0]
    metrics["tau_objects"] = float(model.tau_objects())
    metrics["tau_triples"] = float(model.tau_triples())
    metrics["temperature"] = float(model.loss_temperature())
    return metrics


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(payload, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_run_model(run_dir: Path, device, checkpoint="checkpoint_best.pt"):
    """Rebuild a finished run's model from its config.json flags and load one of its checkpoints."""
    options = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["args"]
    model = GraphReranker(options["adaptive_weights"], options["use_gat"], options["use_triple_channel"],
                          options["use_edges"], options["query_graph_encoder"]).to(device)
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval(), state["epoch"], state["metrics"]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=128,
                        help="Per-forward micro batch; gradients accumulate to the effective batch-size.")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adaptive-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-triple-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-graph-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss", choices=("cross_entropy", "lambda_mrr"), default="cross_entropy")
    parser.add_argument("--rewire-seed", type=int,
                        help="Train and validate on image graphs with degree-preserving rewired edges (control).")
    parser.add_argument("--resume", action="store_true",
                        help="Continue this run from its checkpoint_last.pt (start fresh if it has none).")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Testing only: exit after saving this epoch's checkpoint, as if interrupted.")
    parser.add_argument("--eval-only", type=Path)
    return parser


def train(args, store, run_dir: Path, device, cfg=None):
    """One training run; resumable from run_dir/checkpoint_last.pt with identical behaviour."""
    set_seed(args.seed)
    model = GraphReranker(args.adaptive_weights, args.use_gat, args.use_triple_channel,
                          args.use_edges, args.query_graph_encoder).to(device)
    train_arrays, val_arrays = store.candidates("train"), store.candidates("val")
    last_path = run_dir / "checkpoint_last.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("weight_map") and not args.adaptive_weights: continue
        (no_decay if name.startswith(("log_tau", "log_temperature", "weight_bias")) else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0}], lr=args.lr)
    log = run_dir / "log.jsonl"
    best, stale, best_metrics, last_metrics, start_epoch = -1.0, 0, None, None, 1
    if args.resume and last_path.exists():
        # RNG states must stay CPU byte tensors; the model and optimizer copy their tensors to the device.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, stale = state["best"], state["stale"]
        best_metrics, last_metrics = state["best_metrics"], state["last_metrics"]
        start_epoch = state["epoch"] + 1
        set_rng_state(state["rng"])
        if log.exists():
            # Drop log lines of an epoch that finished after the checkpoint was written.
            lines = log.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if json.loads(line)["epoch"] <= state["epoch"]]
            if len(kept) != len(lines):
                log.with_name(f"log.before-resume-{int(time.time())}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
                log.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        with (run_dir / "resume.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.strftime("%F %T %z"), "from_epoch": state["epoch"]}) + "\n")
    else:
        if log.exists():
            log.rename(log.with_name(f"log.before-restart-{int(time.time())}.jsonl"))
        metadata = {"args": vars(args), "config": cfg, "device": str(device),
                    "train_candidates_sha256": sha256(store.base / store.cfg["paths"]["processed"] / "candidates_train.npz"),
                    "val_candidates_sha256": sha256(store.base / store.cfg["paths"]["processed"] / "candidates_val.npz")}
        try: metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception: metadata["git_commit"] = "unknown"
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    rows = list(rows_for(store, train_arrays, args.limit_train))
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= 2:
            break
        started = time.time(); model.train()
        # Warm up graph parameters for epoch 1; gates open only after that epoch.
        model.set_epoch(epoch)
        # Only the fault reproduction uses the legacy order; all reported runs shuffle.
        epoch_rows = rows if os.environ.get("MGRX_DEBUG_LEGACY_ORDER") else shuffled_rows(rows, args.seed, epoch)
        for start in range(0, len(epoch_rows), args.batch_size):
            batch = epoch_rows[start:start + args.batch_size]
            if os.environ.get("MGRX_DEBUG_BATCH_LOG") and start % (args.batch_size * 50) == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                print(json.dumps({"debug": "batch", "epoch": epoch, "start": start,
                                  "rss_bytes": rss,
                                  "gpu_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}),
                      flush=True)
            optimizer.zero_grad(set_to_none=True)
            micro = max(1, min(args.micro_batch_size, len(batch)))
            for micro_start in range(0, len(batch), micro):
                part = batch[micro_start:micro_start + micro]
                loss = batch_loss(model, store, [caption_id for _, caption_id in part], device,
                                  getattr(args, "loss", "cross_entropy"))
                (loss * (len(part) / len(batch))).backward()
                del loss
            optimizer.step()
        model.eval(); metrics = evaluate_val(model, store, val_arrays, device, args.eval_batch_size)
        gradient_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                module = name.split('.')[0]
                gradient_norms[module] = gradient_norms.get(module, 0.) + float(parameter.grad.detach().norm()) ** 2
        gradient_norms = {name: value ** .5 for name, value in gradient_norms.items()}
        metrics.update({"epoch": epoch, "seconds": time.time() - started, "train_queries": len(rows),
                        "gradient_norms": gradient_norms,
                        "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        last_metrics = metrics
        with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        gc.collect()
        score = metrics["recall@1"]
        if score > best:
            best, stale = score, 0
            best_metrics = metrics
            save_atomic({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, run_dir / "checkpoint_best.pt")
        else:
            stale += 1
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best,
                     "stale": stale, "best_metrics": best_metrics, "last_metrics": last_metrics, "rng": rng_state()},
                    last_path)
        if args.stop_after_epoch == epoch:
            return None
    if last_metrics is not None:
        result = {"best": best_metrics, "last": last_metrics}
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return result
    return None


def main() -> None:
    faulthandler.enable()
    args = build_parser().parse_args()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(cfg)
    if args.rewire_seed is not None:
        store.use_rewired_graphs(args.rewire_seed)
    run_dir = resolve(cfg, "experiments") / "level1" / "gat" / args.run_name
    if args.eval_only:
        set_seed(args.seed)
        model = GraphReranker(args.adaptive_weights, args.use_gat, args.use_triple_channel,
                              args.use_edges, args.query_graph_encoder).to(device)
        model.load_state_dict(torch.load(args.eval_only, map_location=device)["model"])
        model.eval(); print(json.dumps(evaluate_val(model, store, store.candidates("val"), device, args.eval_batch_size)))
        return
    train(args, store, run_dir, device, cfg)


if __name__ == "__main__": main()
