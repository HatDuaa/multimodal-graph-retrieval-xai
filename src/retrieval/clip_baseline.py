"""CLIP-only baseline: caption -> top-k images by cosine, no graph, nothing trained.

For each split the candidate pool is every image of that split and each caption is one
query whose gold answer is its own image. Writes to experiments/level1/clip_baseline/:
  metrics_<split>.json          Recall@1/5/10, MRR (+ Recall@top_k, the ceiling for re-ranking)
  top<k>_<split>.json           per query: caption_id, gold image_id, ranked image_ids, scores

The top-k file is the input of the graph re-ranking step. The baseline is deterministic,
so it is run once (recorded under seed 0) and its std is 0.

Usage:  python -m src.retrieval.clip_baseline [--splits val test]
"""
import argparse
import json

import numpy as np

from src.eval.metrics import DEFAULT_KS, evaluate, format_row, write_metrics
from src.features.extract_clip import load_features
from src.retrieval.faiss_index import CosineIndex
from src.utils.config import REPO_ROOT, load_config, resolve

DATASET = "vg_coco"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    cfg = load_config()
    top_k = cfg["retrieval"]["top_k"]
    feat_dir, split_dir = resolve(cfg, "features"), resolve(cfg, "splits")
    out_dir = REPO_ROOT / cfg["paths"]["experiments"] / "level1" / "clip_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_feats, img_ids, encoder = load_features(feat_dir / f"{DATASET}_image")
    cap_feats, cap_ids, cap_encoder = load_features(feat_dir / f"{DATASET}_caption")
    assert encoder == cap_encoder, "image and caption features come from different encoders"
    img_row = {i: r for r, i in enumerate(img_ids)}
    cap_row = {c: r for r, c in enumerate(cap_ids)}

    for split in args.splits:
        images = json.loads((split_dir / f"{DATASET}_{split}.json").read_text(encoding="utf-8"))["images"]
        pool_ids = [im["image_id"] for im in images]
        queries = [(c["caption_id"], im["image_id"]) for im in images for c in im["captions"]]

        index = CosineIndex(img_feats[[img_row[i] for i in pool_ids]], pool_ids)
        scores, ranked = index.search(cap_feats[[cap_row[c] for c, _ in queries]], top_k)
        golds = [g for _, g in queries]

        ks = tuple(DEFAULT_KS) + ((top_k,) if top_k not in DEFAULT_KS else ())
        metrics = evaluate(ranked, golds, ks=ks)
        payload = write_metrics(out_dir / f"metrics_{split}.json", method="clip", dataset=DATASET, split=split,
                                per_seed={0: metrics},
                                config={"encoder": encoder, "top_k": top_k, "pool_size": len(pool_ids)})
        rows = [{"caption_id": c, "gold": g, "ranked": r, "scores": np.round(s, 5).tolist()}
                for (c, g), r, s in zip(queries, ranked, scores)]
        (out_dir / f"top{top_k}_{split}.json").write_text(json.dumps(rows), encoding="utf-8")

        print(f"[{split}] pool {len(pool_ids)} images, {len(queries)} queries | R@1 | R@5 | R@10 | MRR = "
              f"{format_row(payload['aggregate'])} | R@{top_k} = {100 * metrics[f'recall@{top_k}']:.2f}")


if __name__ == "__main__":
    main()
