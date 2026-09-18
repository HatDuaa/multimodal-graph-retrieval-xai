"""End-to-end check of the phase 0 plumbing on a tiny sample (default 100 images).

download_data.py must have run first. Steps: join Visual Genome with COCO on coco_id ->
take N Karpathy-test images -> download them -> CLIP features -> FAISS -> caption-to-image
search. Prints the size of the intersection and a rough Recall@1 so a broken pipeline is
obvious. This is NOT an experiment: nothing is written to experiments/.

Usage:  python scripts/smoke_test.py [--n 100]
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.extract_clip import ClipEncoder, load_features, save_features  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    raw = resolve(cfg, "raw")

    vg = json.loads((raw / "visual_genome" / "image_data.json").read_text())
    coco = json.loads((raw / "coco" / "dataset_coco.json").read_text())["images"]
    by_coco = {im["cocoid"]: im for im in coco}
    joined = [(v, by_coco[v["coco_id"]]) for v in vg if v.get("coco_id") in by_coco]
    print(f"Visual Genome images: {len(vg)} | with a coco_id found in the Karpathy file: {len(joined)}")
    print("per Karpathy split:", dict(Counter(c["split"] for _, c in joined)))

    rng = np.random.default_rng(cfg["seed"])
    test = [(v, c) for v, c in joined if c["split"] == "test"]
    sample = [test[i] for i in rng.choice(len(test), size=args.n, replace=False)]

    img_dir = raw / "images_smoke"
    img_dir.mkdir(exist_ok=True)
    paths = [img_dir / f"{v['image_id']}.jpg" for v, _ in sample]
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(fetch, [v["url"] for v, _ in sample], paths))
    sample = [s for s, good in zip(sample, ok) if good]
    paths = [p for p, good in zip(paths, ok) if good]
    print(f"images downloaded: {len(paths)}/{args.n}")

    enc = ClipEncoder.from_config(cfg)
    print("encoder:", enc.name, "| device:", enc.device)
    image_ids = [v["image_id"] for v, _ in sample]
    captions = [(f"{c['cocoid']}_{i}", s["raw"], v["image_id"])
                for v, c in sample for i, s in enumerate(c["sentences"])]

    out = resolve(cfg, "features") / "smoke_image"
    save_features(out, enc.encode_images(paths), image_ids, enc.name)
    feats, ids, _ = load_features(out)
    queries = enc.encode_texts([text for _, text, _ in captions])

    _, found = CosineIndex(feats, ids).search(queries, k=10)
    gold = [g for _, _, g in captions]
    r1 = np.mean([f[0] == g for f, g in zip(found, gold)])
    r10 = np.mean([g in f for f, g in zip(found, gold)])
    print(f"queries: {len(captions)} | pool: {len(ids)} images | Recall@1 = {r1:.3f} | Recall@10 = {r10:.3f}")
    print("example:", captions[0][1], "->", found[0][:3], "| gold:", gold[0])
    assert r10 > 0.9, "Recall@10 on a 100-image pool should be near 1; the pipeline is probably broken"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
