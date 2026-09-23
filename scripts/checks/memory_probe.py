"""Peak host RAM and GPU memory of graph reranker training per micro-batch size (train split only).

One micro-batch size per process, so the RSS peak belongs to that size alone. The effective batch stays
256; the probe runs a fixed number of optimizer steps of epoch 1 of the main recipe and appends one record
to the output file, together with what other processes on this shared machine were using at the time.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import batch_loss, rows_for, shuffled_rows  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def shell(command):
    try:
        return subprocess.check_output(command, shell=True, text=True).strip()
    except subprocess.CalledProcessError as error:
        return f"error: {error}"


def machine_snapshot():
    return {"gpu_total_used_mib": shell("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"),
            "gpu_processes": shell("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"),
            "host_free_mib": shell("free -m | awk 'NR==2{print $2\" total, \"$3\" used, \"$7\" available\"}'")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("experiments/checks/memory_probe.jsonl"))
    args = parser.parse_args()
    before = machine_snapshot()
    set_seed(0)
    device = torch.device("cuda")
    store = GraphStore()
    store.precompute()
    model = GraphReranker().to(device).train()
    model.set_epoch(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = shuffled_rows(list(rows_for(store, store.candidates("train"))), 0, 1)
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started, during = time.time(), None
    for step in range(args.steps):
        batch = rows[step * args.batch_size:(step + 1) * args.batch_size]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(batch), args.micro_batch_size):
            part = batch[start:start + args.micro_batch_size]
            loss = batch_loss(model, store, [caption_id for _, caption_id in part], device)
            (loss * (len(part) / len(batch))).backward()
            del loss
        optimizer.step()
        if step == args.steps // 2:
            during = machine_snapshot()
    torch.cuda.synchronize()
    seconds = time.time() - started
    record = {"micro_batch_size": args.micro_batch_size, "batch_size": args.batch_size, "steps": args.steps,
              "seconds_per_step": seconds / args.steps,
              "estimated_train_seconds_per_epoch": seconds / args.steps * len(rows) / args.batch_size,
              "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "rss_after_load_bytes": rss_after_load,
              "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(),
              "machine_before": before, "machine_during": during, "pid": os.getpid(),
              "time": time.strftime("%F %T %z")}
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
