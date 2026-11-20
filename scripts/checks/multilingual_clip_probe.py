"""Row 9 of the task sheet: try multilingual CLIP encoders on Vietnamese queries and pick one.

Where: MKG-W retrieval VAL split (public data, independent of the Vietnamese MMKG,
whose main result is zero-shot and must not be used to choose anything —
vietnamese-mmkg/DATA_CARD.md section 8). Pool = all 892 val images.
Queries: the val entities whose Wikidata description exists in Vietnamese (84),
name masked exactly like the English queries (src/data/build_mkgw_split.mask_name).
Each such entity is asked twice, in Vietnamese and in English, so the vi/en gap of
one encoder is measured on the same answers and the same pool.

Encoders (all frozen, L2-normalised, cosine ranking):
  openai_b32  ViT-B-32-quickgelu / openai — the project encoder; English-only text tower
  xlmr_b32    xlm-roberta-base-ViT-B-32 / laion5b_s13b_b90k — multilingual text tower
  mclip_b32   sentence-transformers clip-ViT-B-32-multilingual-v1 (text, distilled into the
              OpenAI ViT-B/32 space) + the project's OpenAI ViT-B/32 image tower
  xlmr_h14    xlm-roberta-large-ViT-H-14 / frozen_laion5b_s13b_b90k (--large, GPU advised)

Deterministic (frozen encoders, no sampling): one run.
Output: experiments/checks/multilingual_clip_probe.json

Usage:  python scripts/checks/multilingual_clip_probe.py [--large]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.build_mkgw_split import mask_name  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OPEN_CLIP = {
    "openai_b32": ("ViT-B-32-quickgelu", "openai"),
    "xlmr_b32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    "xlmr_h14": ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"),
}
MCLIP_TEXT = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def bilingual_queries(images: list[dict], cache: dict) -> list[dict]:
    """Val entities with a non-empty masked Vietnamese description, plus their English query."""
    out = []
    for im in images:
        row = cache[im["qid"]]
        vi = mask_name(row.get("desc_vi") or "", [row.get("label_vi")] + list(row.get("aliases_vi") or []))
        if vi:
            out.append({"image_id": im["image_id"], "qid": im["qid"], "vi": vi, "en": im["captions"][0]["text"]})
    return out


def rank(q: np.ndarray, pool: np.ndarray, pool_ids: list) -> list[list]:
    order = np.argsort(-(q @ pool.T), axis=1, kind="stable")
    return [[pool_ids[j] for j in row] for row in order]


def encode_mclip_text(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MCLIP_TEXT)
    return m.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="also run xlm-roberta-large-ViT-H-14")
    args = ap.parse_args()

    cfg = load_config()
    images = json.loads((resolve(cfg, "splits") / "mkgw_val.json").read_text(encoding="utf-8"))["images"]
    raw = REPO_ROOT / "data" / "raw"
    cache = {json.loads(line)["qid"]: json.loads(line)
             for line in (raw / "mkgw" / "wikidata_entities.jsonl").read_text(encoding="utf-8").splitlines()}
    paths = [raw / "images" / im["file_name"] for im in images]
    missing = [p.name for p in paths if not p.exists()]
    assert not missing, f"missing MKG-W images: {missing[:5]}"
    pool_ids = [im["image_id"] for im in images]
    qs = bilingual_queries(images, cache)
    golds = [q["image_id"] for q in qs]

    names = ["openai_b32", "xlmr_b32", "mclip_b32"] + (["xlmr_h14"] if args.large else [])
    results, openai_img = {}, None
    for name in names:
        t0 = time.time()
        res = {}
        if name == "mclip_b32":
            img, encoder = openai_img, f"{MCLIP_TEXT} + ViT-B-32-quickgelu/openai image"
            q_feats = {lang: encode_mclip_text([q[lang] for q in qs]) for lang in ("vi", "en")}
        else:
            enc = ClipEncoder(*OPEN_CLIP[name], batch_size=64)
            img, encoder = enc.encode_images(paths), enc.name
            q_feats = {lang: enc.encode_texts([q[lang] for q in qs]) for lang in ("vi", "en")}
            if name == "openai_b32":
                openai_img = img
        for lang, f in q_feats.items():
            res[lang] = evaluate(rank(f, img, pool_ids), golds, ks=(1, 5, 10))
        results[name] = {"encoder": encoder, **res, "seconds": round(time.time() - t0, 1)}
        print(name, {lang: round(100 * m["recall@1"], 2) for lang, m in res.items()})

    out = REPO_ROOT / "experiments" / "checks" / "multilingual_clip_probe.json"
    out.write_text(json.dumps({"split": "mkgw val", "pool": len(pool_ids), "queries_per_language": len(qs),
                               "examples": qs[:5], "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
