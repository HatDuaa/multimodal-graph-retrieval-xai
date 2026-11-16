"""Untrained reranker (step 0: frozen CLIP parts, a/b/c = 0.60/0.28/0.12, tau 0.05) on val through the training data path.

At step 0 the model is the training-free three-channel probe at alpha 0.6, beta 0.7, temperature 0.05, so this checks
that a data version is read as intended (level 1: R@1 0.43496661337346 on both paths).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import evaluate_val  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(load_config(args.config), splits=("val",))
    torch.manual_seed(0)
    model = GraphReranker(adaptive_weights=False).to(device).eval()
    metrics = evaluate_val(model, store, store.candidates("val"), device)
    metrics["config"] = str(args.config or "configs/default.yaml")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
