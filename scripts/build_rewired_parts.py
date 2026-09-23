"""Encode the triple phrases created by degree-preserving edge rewiring (train/val image graphs only).

Writes data/features/vg_coco_graph_parts_rewire_seed<k>.npy/.ids.json holding only phrases that the main part
table lacks, with the same frozen CLIP text encoder and the same raw-text convention as triples in that table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.extract_clip import ClipEncoder, save_features  # noqa: E402
from src.graph.corruption import rewire_graphs, triple_texts  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    cfg = load_config()
    stem = resolve(cfg, "features") / f"vg_coco_graph_parts_rewire_seed{args.seed}"
    if stem.with_suffix(".npy").exists() or stem.with_suffix(".ids.json").exists():
        raise SystemExit(f"refusing to overwrite {stem}")
    graphs = {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] in ("train", "val"):
                graphs[int(graph["image_id"])] = graph
    known = set(json.loads((resolve(cfg, "features") / "vg_coco_graph_parts.ids.json").read_text(encoding="utf-8"))["ids"])
    texts = sorted(t for t in triple_texts(rewire_graphs(graphs, args.seed)) if f"triple:{t}" not in known)
    print(f"images {len(graphs)}, new triple phrases {len(texts)}, "
          f"expected {len(texts) * 512 * 4} bytes", flush=True)
    encoder = ClipEncoder(cfg["clip"]["model"], cfg["clip"]["pretrained"], None, args.batch_size)
    vectors = encoder.encode_texts(texts) if texts else np.zeros((0, 512), dtype=np.float32)
    save_features(stem, vectors.astype(np.float32), [f"triple:{t}" for t in texts], encoder.name)
    print(f"wrote {stem}.npy ({len(texts)} rows)", flush=True)


if __name__ == "__main__":
    main()
