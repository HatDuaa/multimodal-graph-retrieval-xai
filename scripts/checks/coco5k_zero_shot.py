"""Pipeline sanity check: reproduce the published zero-shot CLIP retrieval numbers on MSCOCO 5K.

Our own test pool (Visual Genome ∩ COCO, 2 138 images) is a subset of the standard 5 000-image
Karpathy test, so our baseline numbers cannot be compared with papers directly. This script runs
the SAME encoder, index and metric code on the standard pool and compares with the reference
table of the open_clip repository. A close match means the pipeline is correct.

Downloads the 5 000 test images from images.cocodataset.org into data/raw/coco5k_images/
(~800 MB, kept on disk; resumable) and saves features to data/features/coco5k_*.
Writes experiments/checks/coco5k_zero_shot.json.

Usage:  python scripts/checks/coco5k_zero_shot.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder, save_features  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

# https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_retrieval_results.csv
# rows "<model>,<pretrained>", columns "MSCOCO image retr." (text->image) and "MSCOCO text retr." (image->text)
REFERENCE = {
    "ViT-B-32/openai": {"text_to_image": [30.44, 55.94, 66.87], "image_to_text": [50.12, 75.00, 83.52]},
    "ViT-L-14-336/openai": {"text_to_image": [37.09, 61.62, 71.47], "image_to_text": [57.94, 81.20, 87.92]},
}
KS = (1, 5, 10)


def fetch(job: tuple[str, Path]) -> bool:
    url, dest = job
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except requests.RequestException:
            pass
    return False


def main() -> None:
    cfg = load_config()
    raw = resolve(cfg, "raw")
    test = [im for im in json.loads((raw / "coco" / "dataset_coco.json").read_text())["images"] if im["split"] == "test"]
    img_dir = raw / "coco5k_images"
    img_dir.mkdir(exist_ok=True)
    jobs = [(f"http://images.cocodataset.org/{im['filepath']}/{im['filename']}", img_dir / im["filename"]) for im in test]
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(fetch, jobs))
    print(f"Karpathy test images: {len(test)} | on disk: {sum(ok)}", flush=True)
    assert all(ok), "some images failed to download; run again"

    enc = ClipEncoder.from_config(cfg)
    image_ids = [im["cocoid"] for im in test]
    paths = [dest for _, dest in jobs]
    img = np.concatenate([enc.encode_images(paths[i:i + 1000]) for i in range(0, len(paths), 1000)])
    caps = [(f"{im['cocoid']}_{i}", " ".join(s["raw"].split()), im["cocoid"])
            for im in test for i, s in enumerate(im["sentences"])]
    txt = enc.encode_texts([t for _, t, _ in caps])
    feat_dir = resolve(cfg, "features")
    save_features(feat_dir / "coco5k_image", img, image_ids, enc.name)
    save_features(feat_dir / "coco5k_caption", txt, [c for c, _, _ in caps], enc.name)

    _, ranked = CosineIndex(img, image_ids).search(txt, max(KS))
    t2i = evaluate(ranked, [g for _, _, g in caps], ks=KS)
    captions_of: dict[int, set] = {}
    for cap_id, _, gold in caps:
        captions_of.setdefault(gold, set()).add(cap_id)
    _, ranked = CosineIndex(txt, [c for c, _, _ in caps]).search(img, max(KS))
    i2t = evaluate(ranked, [captions_of[i] for i in image_ids], ks=KS)

    ref = REFERENCE.get(enc.name)
    result = {"encoder": enc.name, "pool_images": len(image_ids), "captions": len(caps), "reference_source":
              "open_clip docs/openclip_retrieval_results.csv", "directions": {}}
    print(f"encoder {enc.name} | {len(image_ids)} images | {len(caps)} captions")
    for name, m in (("text_to_image", t2i), ("image_to_text", i2t)):
        ours = [round(100 * m[f"recall@{k}"], 2) for k in KS]
        entry = {"ours": dict(zip((f"recall@{k}" for k in KS), ours))}
        line = f"{name}: ours R@1/5/10 = {ours}"
        if ref:
            entry["published"] = dict(zip((f"recall@{k}" for k in KS), ref[name]))
            entry["difference"] = dict(zip((f"recall@{k}" for k in KS), [round(o - p, 2) for o, p in zip(ours, ref[name])]))
            line += f" | published = {ref[name]} | diff = {list(entry['difference'].values())}"
        result["directions"][name] = entry
        print(line)

    out = REPO_ROOT / cfg["paths"]["experiments"] / "checks" / "coco5k_zero_shot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
