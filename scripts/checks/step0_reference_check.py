"""Verify the fresh graph reranker reproduces the frozen three-channel reference on val.

Runs through the same precomputed batch path (``GraphStore.batch``) and batched validation
(``evaluate_val``) that training uses, on the training device unless ``--device`` says otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import evaluate_val  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = torch.device(args.device)
    store = GraphStore()
    model = GraphReranker(adaptive_weights=False).to(device).eval()
    reference = np.load("experiments/checks/three_channel_probe_reference_scores.npz")
    caption_ids = [str(caption) for caption in reference["caption_ids"]]
    with torch.no_grad():
        batch = store.batch(caption_ids, device)
        if not np.array_equal(batch["candidate_ids"], reference["candidate_ids"]):
            raise SystemExit("first-200 candidate ids differ from the reference")
        # The reference was scored with its own CLIP column; use it so only the graph path is compared.
        batch["clip"] = torch.from_numpy(reference["clip"]).to(device)
        fused, details = model(batch)
    checks = {"objects": details["objects"], "triples": details["triples"], "z_objects": details["z_objects"],
              "z_triples": details["z_triples"], "graph": details["graph"], "fused": fused}
    checks = {key: value.cpu().numpy() for key, value in checks.items()}
    max_diff = {}
    for key, actual in checks.items():
        expected = reference[key]
        max_diff[key] = float(np.nanmax(np.abs(actual - expected)))
        for row in range(len(caption_ids)):
            if not np.allclose(actual[row], expected[row], atol=1e-4, equal_nan=True):
                raise SystemExit(f"first-200 mismatch: caption={caption_ids[row]}, row={row}, channel={key}, "
                                 f"max_abs={np.nanmax(np.abs(actual[row] - expected[row]))}")
    for row in range(len(caption_ids)):
        if not np.array_equal(np.argsort(-checks["fused"][row], kind="stable"),
                              np.argsort(-reference["fused"][row], kind="stable")):
            raise SystemExit(f"first-200 order mismatch: caption={caption_ids[row]}, row={row}")
    print("first_200_max_abs_diff", json.dumps(max_diff))
    print("reference_first_200_order", "identical")
    metrics = evaluate_val(model, store, store.candidates("val"), device, args.batch_size)
    print("full_val_metrics", json.dumps(metrics))
    if abs(metrics["recall@1"] - 0.434967) > 1e-6:
        raise SystemExit(f"full-val R@1 mismatch: {metrics['recall@1']}, expected 0.434967 +/- 1e-6")


if __name__ == "__main__":
    main()
