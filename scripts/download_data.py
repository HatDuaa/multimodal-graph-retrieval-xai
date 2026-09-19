"""Download the annotation files for Visual Genome ∩ COCO (no images).

Usage:  python scripts/download_data.py [--force]

Files land in data/raw/. Images are NOT downloaded here: only the images of the fixed
subset are fetched later, once the split has been built. After downloading, the script
prints one MANIFEST line per file (name, size, sha256) to paste into data/MANIFEST.md.
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve  # noqa: E402

VG = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"
VG150 = "https://raw.githubusercontent.com/danfeiX/scene-graph-TF-release/master/data_tools/VG/"
# (url, sub-folder under data/raw, unzip?)
FILES = [
    (VG + "image_data.json.zip", "visual_genome", True),      # has the coco_id field used for the join
    (VG + "objects.json.zip", "visual_genome", True),
    (VG + "relationships.json.zip", "visual_genome", True),
    (VG + "object_alias.txt", "visual_genome", False),
    (VG + "relationship_alias.txt", "visual_genome", False),
    (VG150 + "object_list.txt", "vg150", False),
    (VG150 + "predicate_list.txt", "vg150", False),
    (VG150 + "object_alias.txt", "vg150", False),
    (VG150 + "predicate_alias.txt", "vg150", False),
    # Karpathy split; dataset_coco.json holds split, cocoid and the 5 captions of every image
    ("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip", "coco", True),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    raw = resolve(load_config(), "raw")
    rows = []
    for url, sub, unzip in FILES:
        folder = raw / sub
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / url.rsplit("/", 1)[1]
        if args.force or not dest.exists():
            download(url, dest)
        if unzip:
            with zipfile.ZipFile(dest) as z:
                z.extractall(folder)
        rows.append((dest.relative_to(raw.parent.parent), dest.stat().st_size, sha256(dest)))

    print("\n| File | Bytes | SHA-256 |\n|---|---|---|")
    for name, size, digest in rows:
        print(f"| `{name.as_posix()}` | {size} | `{digest}` |")


if __name__ == "__main__":
    main()
