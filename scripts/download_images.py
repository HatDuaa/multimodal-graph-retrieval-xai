"""Download the images listed in the split files into data/raw/images/<image_id>.jpg.

Resumable: existing, readable files are skipped. Failed ids are written to
data/raw/images_failed.txt so the run can be repeated until the list is empty.

Usage:  python scripts/download_images.py [--splits test val train] [--workers 16] [--dataset vg_coco]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve  # noqa: E402


def is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def fetch(job: tuple[str, Path]) -> bool:
    url, dest = job
    if dest.exists() and is_valid(dest):
        return True
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=30,
                             headers={"User-Agent": "mm-graph-retrieval-course-project/0.1 (academic use)"})
            if r.status_code == 429:      # rate-limited (Wikimedia Commons): wait and retry
                time.sleep(int(r.headers.get("Retry-After", 15 * (attempt + 1))))
                continue
            r.raise_for_status()
            dest.write_bytes(r.content)
            if is_valid(dest):
                return True
        except requests.RequestException:
            time.sleep(2)
    dest.unlink(missing_ok=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dataset", default="vg_coco", help="split file prefix, e.g. vg_coco or mkgw")
    args = ap.parse_args()

    cfg = load_config()
    raw, split_dir = resolve(cfg, "raw"), resolve(cfg, "splits")
    out = raw / "images"
    out.mkdir(exist_ok=True)

    jobs = []
    for name in args.splits:
        images = json.loads((split_dir / f"{args.dataset}_{name}.json").read_text(encoding="utf-8"))["images"]
        jobs += [(im["url"], out / im["file_name"]) for im in images]

    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(tqdm(pool.map(fetch, jobs), total=len(jobs), desc="images"))
    failed = [dest.stem for (_, dest), good in zip(jobs, ok) if not good]
    (raw / "images_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"done: {sum(ok)}/{len(jobs)} ok, {len(failed)} failed (see data/raw/images_failed.txt)")


if __name__ == "__main__":
    main()
