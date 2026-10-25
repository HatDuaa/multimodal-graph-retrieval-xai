"""Evaluation-only graph controls on val for countonly_factual_r3, as scripts/checks/graph_controls.py for level 1.

other_image: every val image is scored with the scene graph of another val image (src/graph/corruption.graph_derangement,
             the same fixed derangement per control seed as level 1); the graph keeps its own boxes and count buckets.
rewire:      every val image graph has its relation targets permuted (rewire_graphs: degree-preserving, the predicate
             label travels with the edge, same generator per (seed, image) as level 1). The completion edges ([none]
             kNN + bridges) and the edge geometry are rebuilt from the boxes by the run's structure, as for any graph.
             Triple phrases the part table lacks are encoded with the same frozen CLIP text encoder, in memory only.
Control seeds 0, 1, 2 on each of the three run seeds; nothing is trained and test is never loaded.
Writes experiments/improve/controls/<control>_<run>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.corruption import graph_derangement, rewire_graphs, triple_texts  # noqa: E402
from src.train_graph_reranker import evaluate_val, load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

RUNS = REPO_ROOT / "experiments" / "improve" / "count-only"
OUT = REPO_ROOT / "experiments" / "improve" / "controls"
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def rewired(store, original, seed, encoder):
    """Rewired val graphs and the vectors of their triple phrases that the part table lacks."""
    graphs = rewire_graphs(original, seed)
    missing = sorted(f"triple:{t}" for t in triple_texts(graphs) if f"triple:{t}" not in store.part_row)
    vectors = encoder.encode_texts([k.split(":", 1)[1] for k in missing]) if missing else np.zeros((0, store.part_dim))
    return graphs, np.asarray(vectors, dtype=np.float32), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[f"countonly_factual_r3_seed{k}" for k in range(3)])
    parser.add_argument("--control-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = load_config(REPO_ROOT / "configs" / "count_v1.yaml")
    store = GraphStore(cfg, splits=("val",))
    original = dict(store.graphs)
    base_rows, base_vectors = len(store.part_vectors), store.part_vectors
    val_images = sorted({int(i) for i in store.candidates("val")["candidate_ids"].ravel()})
    from src.features.extract_clip import ClipEncoder, load_features
    encoder = ClipEncoder.from_config(cfg, device=args.device)
    _, _, table_encoder = load_features(REPO_ROOT / cfg["paths"]["features"] / "vg_coco_graph_parts")
    assert encoder.name == table_encoder, (encoder.name, table_encoder)
    rewires = {seed: rewired(store, original, seed, encoder) for seed in args.control_seeds}
    OUT.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        model, epoch, saved = load_run_model(RUNS / run, device, store=store)
        for control in ("other_image", "rewire"):
            output = OUT / f"{control}_{run}.json"
            if output.exists():
                print(f"skip existing {output}", flush=True)
                continue
            started = time.time()
            store.replace_image_graphs(original)
            with torch.no_grad():
                clean = evaluate_val(model, store, store.candidates("val"), device)
            result = {"run": run, "control": control, "checkpoint_epoch": epoch, "device": str(device),
                      "saved_best_val": {k: saved[k] for k in METRICS}, "clean_val": clean, "corrupted_val": {}}
            for seed in args.control_seeds:
                if control == "other_image":
                    mapping = graph_derangement(val_images, seed)
                    store.replace_image_graphs({i: original[mapping.get(i, i)] for i in original})
                else:
                    graphs, vectors, keys = rewires[seed]
                    result.setdefault("encoded_triple_phrases", {})[str(seed)] = len(keys)
                    store.replace_image_graphs(graphs, vectors, keys)
                with torch.no_grad():
                    result["corrupted_val"][str(seed)] = evaluate_val(model, store, store.candidates("val"), device)
                # drop the appended rewire vectors so every control starts from the same table
                store.part_vectors = base_vectors
                store.part_row = {k: r for k, r in store.part_row.items() if r < base_rows}
                store.replace_image_graphs(original)
                print(run, control, seed, json.dumps({k: result["corrupted_val"][str(seed)][k] for k in ("recall@1", "mrr")}),
                      flush=True)
            result["seconds"] = time.time() - started
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
