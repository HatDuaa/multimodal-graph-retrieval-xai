"""One-time test evaluation of the main model countonly_factual_r3 and of CLIP only, on the same test captions.

Run --split test only on Lộc's go, once. Each seed's checkpoint_best.pt (selected on val R@1 during training) is scored
with its own flags and graph structure (config.json); results are reported per seed and as mean +- std. Before any model
is scored, the packaged test candidates must reproduce the level-1 CLIP baseline on test (the same check as
scripts/evaluate_test.py). The result file is written once: if it exists, nothing is computed.

--split val is the dry run of the same code on val (configs/count_v1.yaml): every run must reproduce the val metrics of
its metrics.json and CLIP only the level-1 CLIP val baseline. It reads no test data and writes no test result.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import aggregate_seeds, evaluate, write_metrics  # noqa: E402
from src.train_graph_reranker import evaluate_val, load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

RUNS = [REPO_ROOT / "experiments" / "improve" / "count-only" / f"countonly_factual_r3_seed{k}" for k in range(3)]
CLIP_BASELINE = REPO_ROOT / "experiments" / "level1" / "clip_baseline"
OUT = REPO_ROOT / "experiments" / "improve" / "test"
CONFIGS = {"test": "configs/count_v1_test.yaml", "val": "configs/count_v1.yaml"}
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def scalars(metrics):
    a, b, c = metrics["weights_mean"]
    return {"n_queries": metrics["n_queries"], **{k: metrics[k] for k in METRICS}, "a": a, "b": b, "c": c}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "val"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = OUT / "metrics_countonly_factual_r3.json"
    clip_output = OUT / "metrics_clip_only.json"
    if args.split == "test" and (output.exists() or clip_output.exists()):
        raise SystemExit(f"{output} or {clip_output} exists: test results are computed once and never recomputed")
    device = torch.device(args.device)
    cfg = load_config(REPO_ROOT / CONFIGS[args.split])
    store = GraphStore(cfg, splits=(args.split,), allow_test=args.split == "test")
    arrays = store.candidates(args.split)
    candidates = REPO_ROOT / cfg["paths"]["processed"] / f"candidates_{args.split}.npz"
    provenance = {"split": args.split, "config": CONFIGS[args.split], "candidates_sha256": sha256(candidates.resolve()),
                  "time": time.strftime("%F %T %z"),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()}
    clip = evaluate(arrays["candidate_ids"].tolist(), arrays["gold"].tolist())
    reference = json.loads((CLIP_BASELINE / f"metrics_{args.split}.json").read_text(encoding="utf-8"))["per_seed"]["0"]
    for key in METRICS:
        if abs(clip[key] - reference[key]) > 1e-9:
            raise SystemExit(f"{args.split} candidates do not reproduce the CLIP baseline: {key} {clip[key]} vs {reference[key]}")
    print("clip_only", args.split, json.dumps({k: clip[k] for k in METRICS}), flush=True)
    per_seed, runs, mismatches = {}, {}, []
    for run in RUNS:
        model, epoch, val = load_run_model(run, device, store=store)  # also sets the run's graph structure
        metrics = evaluate_val(model, store, arrays, device)
        options = json.loads((run / "config.json").read_text(encoding="utf-8"))["args"]
        per_seed[int(options["seed"])] = scalars(metrics)
        runs[run.name] = {"epoch": epoch, "val_recall@1": val["recall@1"],
                          "checkpoint_sha256": sha256(run / "checkpoint_best.pt")}
        print(run.name, args.split, json.dumps({k: metrics[k] for k in METRICS}), flush=True)
        if args.split == "val":
            # Dry run: the selected checkpoint must reproduce its own val numbers (R@k exactly; MRR up to the
            # run-to-run GPU reduction noise seen in experiments/improve/count-only/logs/check_zero_repeat.log).
            stored = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["best"]
            for key in METRICS:
                tolerance = 1e-6 if key == "mrr" else 0.
                if abs(metrics[key] - stored[key]) > tolerance:
                    mismatches.append((run.name, key, metrics[key], stored[key]))
    aggregate = aggregate_seeds(list(per_seed.values()))
    print("countonly_factual_r3", args.split, json.dumps({k: aggregate[k] for k in METRICS}), flush=True)
    if args.split == "val":
        print("dry run on val:", "every run reproduces its metrics.json" if not mismatches else f"MISMATCH {mismatches}",
              flush=True)
        if mismatches:
            raise SystemExit(1)
        return
    write_metrics(clip_output, method="clip_only", dataset="vg_coco", split="test", per_seed={0: {"n_queries": clip["n_queries"],
                  **{k: clip[k] for k in METRICS}}}, config=provenance)
    write_metrics(output, method="countonly_factual_r3", dataset="vg_coco", split="test", per_seed=per_seed,
                  config={**provenance, "runs": runs})
    print(f"wrote {output} and {clip_output}", flush=True)


if __name__ == "__main__":
    main()
