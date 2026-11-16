"""How often does the z-score floor bind after training? Per-query std of each raw channel on val vs the floors.

For each run: the share of val queries (channel active: >= 2 finite scores) whose top-50 std of the object and triple
channel scores lies below the floor of train_channel_stats.json (zscore_floor), and the median std.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/factual_v1.yaml"))
    parser.add_argument("--stats", type=Path, default=Path("experiments/improve/fusion/train_channel_stats.json"))
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    floors = json.loads(args.stats.read_text(encoding="utf-8"))["zscore_floor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(load_config(args.config), splits=("val",))
    caption_ids = [str(c) for c in store.candidates("val")["caption_ids"]]
    result = {"floors": dict(zip(("clip", "objects", "triples"), floors)), "runs": {}}
    for run in args.runs:
        model = load_run_model(Path(run), device)[0]
        stds = {"objects": [], "triples": []}
        for start in range(0, len(caption_ids), 256):
            _, details = model(store.batch(caption_ids[start:start + 256], device))
            for name in stds:
                values = details[name].float().cpu().numpy()
                for row in values:
                    known = row[np.isfinite(row)]
                    if len(known) >= 2:
                        stds[name].append(float(known.std()))
        result["runs"][run] = {name: {"queries": len(v), "median_std": float(np.median(v)),
                                      "share_below_floor": float(np.mean(np.asarray(v) < floors[k + 1]))}
                               for k, (name, v) in enumerate(stds.items())}
        print(run, json.dumps(result["runs"][run]), flush=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
