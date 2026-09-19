"""Measure the QuickGELU model choice on validation and the public MSCOCO 5K pool.

The validation run uses only ``vg_coco_val.json``. The COCO 5K pool is expected to be on disk;
download it with ``python scripts/checks/coco5k_zero_shot.py`` when it is absent.
Writes ``experiments/checks/quickgelu_probe.json`` and does not save feature files.
"""
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks.coco5k_zero_shot import REFERENCE, evaluate_coco5k, prepare_coco5k, encode_coco5k  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

VG_KS = (1, 5, 10, 50)
COCO_KS = (1, 5, 10)
MODELS = ("ViT-B-32", "ViT-B-32-quickgelu")


def validation_metrics(enc: ClipEncoder, cfg: dict) -> dict:
    images = json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
    ids = [im["image_id"] for im in images]
    paths = [resolve(cfg, "raw") / "images" / im["file_name"] for im in images]
    captions = [(c["text"], im["image_id"]) for im in images for c in im["captions"]]
    image_features = enc.encode_images(paths)
    text_features = enc.encode_texts([text for text, _ in captions])
    _, ranked = CosineIndex(image_features, ids).search(text_features, max(VG_KS))
    return evaluate(ranked, [gold for _, gold in captions], ks=VG_KS)


def main() -> None:
    cfg = load_config()
    _, coco_paths, coco_ids, coco_caps = prepare_coco5k(cfg, download=False)
    missing = [p for p in coco_paths if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("MSCOCO 5K images are missing; run python scripts/checks/coco5k_zero_shot.py")

    results = {}
    for model_name in MODELS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc = ClipEncoder(model_name, "openai", batch_size=cfg["clip"]["batch_size"])
        warning_flag = any("QuickGELU mismatch" in str(w.message) for w in caught)
        vg = validation_metrics(enc, cfg)
        image_features, text_features = encode_coco5k(enc, coco_paths, coco_caps)
        directions = evaluate_coco5k(image_features, text_features, coco_ids, coco_caps)
        coco_result = {}
        for direction in ("text_to_image", "image_to_text"):
            ours = {f"recall@{k}": round(100 * directions[direction][f"recall@{k}"], 2) for k in COCO_KS}
            reference = REFERENCE.get(f"{model_name}/openai", REFERENCE["ViT-B-32/openai"]).get(direction)
            entry = {"ours": ours, "published": dict(zip(ours, reference)) if reference else None}
            entry["difference"] = (dict(zip(ours, [round(ours[f"recall@{k}"] - p, 2) for k, p in zip(COCO_KS, reference)]))
                                    if reference else None)
            coco_result[direction] = entry
        results[model_name] = {"quickgelu_warning": warning_flag, "vg_coco_val": vg, "coco5k": coco_result}
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {"date": date.today().isoformat(), "open_clip_version": getattr(open_clip, "__version__", "unknown"),
              "torch_version": torch.__version__, "models": results}
    out = REPO_ROOT / cfg["paths"]["experiments"] / "checks" / "quickgelu_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("model | warning | val R@1 | COCO t2i R@1/5/10 | COCO i2t R@1/5/10")
    for name, result in results.items():
        val = result["vg_coco_val"]
        fmt = lambda d: "/".join(f"{d['ours'][f'recall@{k}']:.2f}" for k in COCO_KS)
        print(f"{name} | {result['quickgelu_warning']} | {100 * val['recall@1']:.2f} | "
              f"{fmt(result['coco5k']['text_to_image'])} | {fmt(result['coco5k']['image_to_text'])}")
    print("wrote", out)


if __name__ == "__main__":
    main()
