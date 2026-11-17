"""Pull the MKG-W outputs that notebooks/mkgw_molab.py pushed to Hugging Face.

Copies into the repository layout (all of it is git-ignored):
  data/raw/mkgw/*, data/processed/mkgw_triples.jsonl, data/features/mkgw_*   (inputs of evaluation)
  experiments/level2/native-train/*.log, experiments/checks/multilingual_clip_probe.json
  native/checkpoint/*.ckpt -> data/raw/native_checkpoints/                      (input of export)
Images (images/mkgw_images_640.zip, ~0.5 GB) only with --images: evaluation needs features, not images.

Then, per run:  python scripts/export_native_embeddings.py --checkpoint data/raw/native_checkpoints/<run>.ckpt
                    --what ent --suffix _seed0

Usage:  python scripts/pull_mkgw_from_hf.py [--repo duongkhang2100/graphml-mkgw-retrieval] [--images]
The token is read from the HF_TOKEN environment variable or the huggingface-cli login cache.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="duongkhang2100/graphml-mkgw-retrieval")
    ap.add_argument("--images", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    patterns = ["data/*", "data/**", "experiments/**", "native/checkpoint/*"] + (["images/*"] if args.images else [])
    local = Path(snapshot_download(args.repo, repo_type="dataset", allow_patterns=patterns,
                                   local_dir=str(REPO_ROOT / "data" / "raw" / "hf_mkgw")))
    for sub in ("data/raw", "data/processed", "data/features", "experiments"):
        if (local / sub).exists():
            shutil.copytree(local / sub, REPO_ROOT / sub, dirs_exist_ok=True)
    ckpt_dir = REPO_ROOT / "data" / "raw" / "native_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for p in (local / "native" / "checkpoint").glob("*.ckpt"):
        shutil.copy2(p, ckpt_dir / p.name)
        print("checkpoint:", ckpt_dir / p.name)
    if args.images and (local / "images" / "mkgw_images_640.zip").exists():
        zipfile.ZipFile(local / "images" / "mkgw_images_640.zip").extractall(REPO_ROOT / "data" / "raw" / "images")
    print("done")


if __name__ == "__main__":
    main()
