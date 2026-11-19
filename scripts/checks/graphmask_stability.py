"""Does GraphMask training (scripts/xai_metrics.py, v3) give the same gates when it is run again?

For each model and seed, trains GraphMask on the same 200 val captions as scripts/xai_metrics.py, several times in one
process and under several CPU thread counts, and records the last epoch's L0 and divergence of each try.
Nothing else is measured; the reranker is not trained and the test split is never loaded.
Writes experiments/xai-metrics-v3/graphmask_stability.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xai_metrics as xm  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-metrics-v3" / "graphmask_stability.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["countonly_factual_r3", "main_r3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, torch.get_num_threads()])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device("cpu")
    result = {"rule": __doc__, "torch": torch.__version__, "default_threads": torch.get_num_threads(), "tries": []}
    for name in args.models:
        root, config = xm.MODELS[name]
        store = GraphStore(load_config(REPO_ROOT / config), splits=("val",)) if config else GraphStore(splits=("val",))
        # the same training captions as scripts/xai_metrics.py (defaults --compare 1000, --pg-train 200)
        caption_ids = [str(c) for c in store.candidates("val")["caption_ids"]]
        compare_ids = set(np.random.default_rng(2026).choice(caption_ids, size=1000, replace=False).tolist())
        cases = json.loads(xm.CASES.read_text(encoding="utf-8"))
        case_ids = [c["caption_id"] for c in cases["cases"]] + [e["caption_id"] for e in cases["comparison"]["examples"]]
        pool = sorted(set(caption_ids) - compare_ids - set(case_ids))
        for seed in args.seeds:
            model, _, _ = xm.load_run_model(root / f"{name}_seed{seed}", device, store=store)
            chosen = xm.training_captions(model, store, device, np.random.default_rng(2027).permutation(pool), 200)
            for threads in args.threads:
                torch.set_num_threads(threads)
                for repeat in range(args.repeats):
                    started = time.time()
                    _, info = xm.train_gm(model, store, device, chosen, seed)
                    row = {"model": name, "seed": seed, "threads": threads, "repeat": repeat,
                           "l0": info["l0"], "divergence": info["divergence"], "lagrange": info["lagrange"],
                           "step": info.get("step"),
                           "seconds": time.time() - started}
                    result["tries"].append(row)
                    print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
