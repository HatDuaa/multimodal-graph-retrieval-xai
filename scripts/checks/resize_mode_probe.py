"""Does it matter how a non-square image is made square before CLIP sees it?

CLIP's default preprocessing resizes the short side to 224 px and takes the CENTER CROP, so the
ends of wide or tall images are never seen. open_clip offers two alternatives: "longest" (resize
the long side, pad to a square) and "squash" (distort to a square). This script compares the three
on the VALIDATION split only, so the choice is made without touching test (brief, section 6).

Caption features are reused; only validation images are re-encoded (seconds on a GPU).
Writes experiments/checks/resize_mode_probe.json.

Usage:  python scripts/checks/resize_mode_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import load_features  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

MODES = {"shortest": "center crop (default, used by the project)", "longest": "pad to square", "squash": "distort to square"}
KS = (1, 5, 10, 50)


def main() -> None:
    cfg = load_config()
    images = json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
    ids = [im["image_id"] for im in images]
    paths = [resolve(cfg, "raw") / "images" / im["file_name"] for im in images]

    ratios = np.array([im["width"] / im["height"] for im in images])
    kept = np.where(ratios >= 1, 1 / ratios, ratios)   # share of the long side that survives the center crop
    shape = {"images": len(ids), "median_aspect_ratio": round(float(np.median(ratios)), 2),
             "landscape_share": round(float(np.mean(ratios > 1.05)), 3),
             "portrait_share": round(float(np.mean(ratios < 0.95)), 3),
             "long_side_kept_mean": round(float(kept.mean()), 3), "long_side_kept_min": round(float(kept.min()), 3)}
    print(shape)

    cap_feats, cap_ids, encoder = load_features(resolve(cfg, "features") / "vg_coco_caption")
    row = {c: r for r, c in enumerate(cap_ids)}
    queries = [(c["caption_id"], im["image_id"]) for im in images for c in im["captions"]]
    q = cap_feats[[row[c] for c, _ in queries]]
    golds = [g for _, g in queries]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for mode, label in MODES.items():
        model, _, pre = open_clip.create_model_and_transforms(cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"],
                                                              image_resize_mode=mode)
        model.eval().to(device)
        feats = []
        with torch.no_grad():
            for i in range(0, len(paths), 256):
                x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 256]]).to(device)
                f = model.encode_image(x).float()
                feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        _, ranked = CosineIndex(np.concatenate(feats).astype(np.float32), ids).search(q, max(KS))
        m = evaluate(ranked, golds, ks=KS)
        results[mode] = {"description": label, **{k: round(100 * v, 2) for k, v in m.items() if k != "n_queries"}}
        print(mode, results[mode])

    out = REPO_ROOT / cfg["paths"]["experiments"] / "checks" / "resize_mode_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"encoder": encoder, "split": "val", "queries": len(queries), "image_shapes": shape,
                               "modes": results}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
