"""Extract frozen CLIP features for every image and caption of a dataset's splits.

Output (format: docs/interfaces.md, section 2):
  data/features/<dataset>_image.npy   + .ids.json   (ids = image_id)
  data/features/<dataset>_caption.npy + .ids.json   (ids = caption_id)

All splits go into the same two files; filter by the ids found in the split files.
The script refuses to run if any image of the splits is missing on disk, so that nobody
silently evaluates on a smaller pool.

Usage:  python scripts/extract_features.py [--dataset vg_coco]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.extract_clip import ClipEncoder, save_features  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

CHUNK = 2048  # images encoded per progress-bar step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vg_coco", help="split file prefix, e.g. vg_coco or mkgw")
    dataset = ap.parse_args().dataset

    cfg = load_config()
    raw, split_dir, feat_dir = resolve(cfg, "raw"), resolve(cfg, "splits"), resolve(cfg, "features")

    images = []
    for name in ("train", "val", "test"):
        images += json.loads((split_dir / f"{dataset}_{name}.json").read_text(encoding="utf-8"))["images"]
    paths = [raw / "images" / im["file_name"] for im in images]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing (e.g. {missing[:3]}); run scripts/download_images.py first")

    enc = ClipEncoder.from_config(cfg)
    print("encoder:", enc.name, "| device:", enc.device, "| images:", len(images))

    feats = [enc.encode_images(paths[i:i + CHUNK]) for i in tqdm(range(0, len(paths), CHUNK), desc="images")]
    save_features(feat_dir / f"{dataset}_image", np.concatenate(feats), [im["image_id"] for im in images], enc.name)

    captions = [c for im in images for c in im["captions"]]
    texts = [c["text"] for c in captions]
    feats = [enc.encode_texts(texts[i:i + CHUNK * 8]) for i in tqdm(range(0, len(texts), CHUNK * 8), desc="captions")]
    save_features(feat_dir / f"{dataset}_caption", np.concatenate(feats), [c["caption_id"] for c in captions], enc.name)
    print("saved", len(images), "image vectors and", len(captions), "caption vectors to", feat_dir)


if __name__ == "__main__":
    main()
