"""How much does the third channel repeat the object channel? Per-query channel correlations on val (no training).

For every val query, the Pearson correlation across its top-50 candidates between the raw scores of two channels
(only candidates where both are finite, only queries where both channels are active, at least 3 candidates).
z-scoring is affine per query, so these are also the correlations of the z-scores that fuse_tensor mixes.
Reported per run (mean over queries) and as mean +- std over the runs of a group.
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

PAIRS = (("objects", "third"), ("clip", "third"), ("clip", "objects"))


def row_correlation(a, b, mask):
    """Pearson correlation per row over masked entries; nan for rows with fewer than 3 entries or no variance."""
    count = mask.sum(-1)
    a, b = torch.where(mask, a, 0.), torch.where(mask, b, 0.)
    mean_a, mean_b = a.sum(-1) / count.clamp_min(1), b.sum(-1) / count.clamp_min(1)
    da, db = torch.where(mask, a - mean_a[:, None], 0.), torch.where(mask, b - mean_b[:, None], 0.)
    cov, var_a, var_b = (da * db).sum(-1), da.square().sum(-1), db.square().sum(-1)
    corr = cov / (var_a * var_b).sqrt()
    return torch.where((count >= 3) & (var_a > 1e-12) & (var_b > 1e-12), corr, torch.nan)


@torch.no_grad()
def run_correlations(model, store, caption_ids, device, batch_size=256):
    values = {pair: [] for pair in PAIRS}
    active = []
    for start in range(0, len(caption_ids), batch_size):
        batch = store.batch(caption_ids[start:start + batch_size], device)
        _, details = model(batch)
        scores = {"clip": batch["clip"], "objects": details["objects"], "third": details["triples"]}
        both = details["object_active"] & details["triple_active"]
        active.append(details["triple_active"].cpu())
        for a, b in PAIRS:
            mask = torch.isfinite(scores[a]) & torch.isfinite(scores[b])
            corr = row_correlation(scores[a].nan_to_num(), scores[b].nan_to_num(), mask)
            if "third" in (a, b):
                corr = torch.where(both, corr, torch.nan)
            values[(a, b)].append(corr.cpu())
    out = {"third_channel_active": float(torch.cat(active).float().mean())}
    for pair, chunks in values.items():
        corr = torch.cat(chunks).numpy()
        corr = corr[np.isfinite(corr)]
        out[f"{pair[0]}~{pair[1]}"] = {"mean": float(corr.mean()), "median": float(np.median(corr)), "queries": int(len(corr))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", nargs="+", action="append", required=True,
                        help="NAME RUN_DIR [RUN_DIR ...]; repeat for each group.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(splits=("val",))
    caption_ids = [str(cid) for cid in store.candidates("val")["caption_ids"]]
    result = {"split": "val", "n_queries": len(caption_ids), "per_run": {}, "aggregate": {}}
    for name, *runs in args.group:
        for run in runs:
            result["per_run"][run] = run_correlations(load_run_model(Path(run), device)[0], store, caption_ids, device)
            print(name, run, json.dumps(result["per_run"][run]), flush=True)
        keys = [k for k in result["per_run"][runs[0]] if "~" in k] + ["third_channel_active"]
        result["aggregate"][name] = {}
        for key in keys:
            values = np.array([result["per_run"][r][key]["mean"] if "~" in key else result["per_run"][r][key] for r in runs])
            result["aggregate"][name][key] = {"mean": float(values.mean()),
                                              "std": float(values.std(ddof=1)) if len(values) > 1 else 0.}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
