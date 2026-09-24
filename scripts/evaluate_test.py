"""One-time final evaluation of locked level 1 models on the test split.

Run only after every design choice was made on validation. It packages the CLIP top-50 of the test split
(from the baseline's top50_test.json, the same way val was packaged), checks that the packaged candidates
reproduce the CLIP baseline test numbers, then scores each requested run's best-on-val checkpoint once.
Each result file is written once; an existing file is never recomputed or overwritten.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import evaluate, write_metrics  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import evaluate_val, load_run_model  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

RUNS = Path("experiments/level1/gat")
OUT = Path("experiments/level1/test")
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def scalars(metrics):
    """The ranking metrics plus mean channel weights a, b, c, as flat numbers for aggregate_seeds."""
    a, b, c = metrics["weights_mean"]
    return {"n_queries": metrics["n_queries"], **{k: metrics[k] for k in METRICS}, "a": a, "b": b, "c": c}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def package_test_candidates(cfg) -> Path:
    path = resolve(cfg, "processed") / "candidates_test.npz"
    if path.exists():
        return path
    baseline = resolve(cfg, "experiments") / "level1" / "clip_baseline" / "top50_test.json"
    split = json.loads((resolve(cfg, "splits") / "vg_coco_test.json").read_text(encoding="utf-8"))
    gold_by_caption = {cap["caption_id"]: image["image_id"] for image in split["images"] for cap in image["captions"]}
    rows = [row for row in json.loads(baseline.read_text(encoding="utf-8")) if row["caption_id"] in gold_by_caption]
    if len(rows) != len(gold_by_caption):
        raise AssertionError(f"baseline test rows {len(rows)} != captions {len(gold_by_caption)}")
    candidate_ids = np.asarray([row["ranked"] for row in rows], dtype=np.int64)
    clip_scores = np.asarray([row["scores"] for row in rows], dtype=np.float32)
    golds = np.asarray([gold_by_caption[row["caption_id"]] for row in rows], dtype=np.int64)
    if candidate_ids.shape != (len(rows), 50) or any(int(row["gold"]) != int(g) for row, g in zip(rows, golds)):
        raise AssertionError("baseline test top-50 has unexpected shape or gold ids")
    gold_rank = np.asarray([(list(ids).index(g) + 1) if g in ids else -1 for ids, g in zip(candidate_ids, golds)])
    with path.open("xb") as handle:
        np.savez(handle, caption_ids=np.asarray([row["caption_id"] for row in rows], dtype=str), gold=golds,
                 candidate_ids=candidate_ids, clip_scores=clip_scores, gold_rank=gold_rank.astype(np.int64),
                 group_index=np.full(len(rows), -1, dtype=np.int64), baseline_sha256=np.asarray(sha256(baseline)))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[f"main_r3_seed{k}" for k in range(3)])
    parser.add_argument("--method", default="main_r3", help="name of the result file for these runs")
    parser.add_argument("--training-free", action="store_true", help="also score the untrained step-0 model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    candidates_path = package_test_candidates(cfg)
    store = GraphStore(cfg, splits=("test",), allow_test=True)
    arrays = store.candidates("test")
    provenance = {"candidates_test_sha256": sha256(candidates_path), "time": time.strftime("%F %T %z"),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    # The packaged candidates must reproduce the CLIP baseline on test before anything else is scored.
    clip = evaluate(arrays["candidate_ids"].tolist(), arrays["gold"].tolist())
    reference = json.loads((resolve(cfg, "experiments") / "level1" / "clip_baseline" / "metrics_test.json")
                           .read_text(encoding="utf-8"))["per_seed"]["0"]
    for key in METRICS:
        if abs(clip[key] - reference[key]) > 1e-9:
            raise SystemExit(f"test candidates do not reproduce the CLIP baseline: {key} {clip[key]} vs {reference[key]}")
    print("clip_only_test", json.dumps({k: clip[k] for k in METRICS}), flush=True)
    if args.training_free and not (OUT / "metrics_training_free.json").exists():
        model = GraphReranker(adaptive_weights=False).to(device).eval()
        metrics = evaluate_val(model, store, arrays, device)
        write_metrics(OUT / "metrics_training_free.json", method="three_channel_training_free", dataset="vg_coco",
                      split="test", per_seed={0: scalars(metrics)}, config=provenance)
        print("training_free_test", json.dumps({k: metrics[k] for k in METRICS}), flush=True)
    output = OUT / f"metrics_{args.method}.json"
    if output.exists():
        raise SystemExit(f"{output} exists: test results are computed once and never recomputed")
    per_seed, checkpoints = {}, {}
    original = dict(store.graphs)
    for run in args.runs:
        options = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))["args"]
        model, epoch, val = load_run_model(RUNS / run, device)
        # A model trained on rewired graphs is scored on rewired test graphs (same rewiring seed).
        if options.get("rewire_seed") is not None:
            store.use_rewired_graphs(options["rewire_seed"], "_test")
        metrics = evaluate_val(model, store, arrays, device)
        store.replace_image_graphs(original)
        per_seed[int(options["seed"])] = scalars(metrics)
        checkpoints[run] = {"epoch": epoch, "val_recall@1": val["recall@1"], "rewire_seed": options.get("rewire_seed"),
                            "checkpoint_sha256": sha256(RUNS / run / "checkpoint_best.pt")}
        print(run, "test", json.dumps({k: metrics[k] for k in METRICS}), flush=True)
    write_metrics(output, method=args.method, dataset="vg_coco", split="test", per_seed=per_seed,
                  config={**provenance, "runs": checkpoints})
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
