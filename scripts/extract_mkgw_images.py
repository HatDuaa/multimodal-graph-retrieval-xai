"""Extract one canonical image per MKG-W entity from the dataset authors' archive.

The MMRNS release (Google Drive of quqxui/MMRNS — the group that built MKG-W)
ships MKG-W_img.zip: DW_open/<DBpedia name>/<images>. These are the images the
official visual embeddings (NativE's MKG-W-visual.pth) were extracted from, so
they are the coherent image source for the retrieval task — and they avoid
Wikimedia's per-IP rate limits entirely.

For every entity that has at least one readable image, the FIRST file in
alphabetical order is converted to RGB JPEG (longest side <= --max-side) at
data/raw/images/<qid>.jpg, and an index {qid: zip member} is written to
data/raw/mkgw/author_images_index.json. Folder names are mapped to qids through
the authors' ent_links file (DBpedia resource <-> Wikidata entity).

Usage:  python scripts/extract_mkgw_images.py [--max-side 640]
"""
import argparse
import io
import json
import sys
import urllib.parse
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

Image.MAX_IMAGE_PIXELS = None      # some Commons scans are huge; we downscale anyway


def name_to_qid(ent_links: Path) -> dict[str, str]:
    out = {}
    for line in ent_links.read_text(encoding="utf-8").splitlines():
        db, wd = line.split("\t")
        out[urllib.parse.unquote(db.rsplit("/", 1)[-1])] = wd.rsplit("/", 1)[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=640)
    args = ap.parse_args()

    cfg = load_config()
    raw = resolve(cfg, "raw")
    mkgw = raw / "mkgw"
    out_dir = raw / "images"
    out_dir.mkdir(exist_ok=True)

    mapping = name_to_qid(mkgw / "ent_links.tsv")
    zf = zipfile.ZipFile(mkgw / "mkgw_img.zip")
    by_folder: dict[str, list[str]] = {}
    for n in sorted(zf.namelist()):
        parts = n.split("/")
        if len(parts) >= 3 and parts[2]:
            by_folder.setdefault(parts[1], []).append(n)

    index: dict[str, str] = {}
    unmatched_folders, unreadable = 0, 0
    for folder, members in tqdm(sorted(by_folder.items()), desc="entities"):
        qid = mapping.get(folder) or mapping.get(urllib.parse.unquote(folder))
        if qid is None:
            unmatched_folders += 1
            continue
        dest = out_dir / f"{qid}.jpg"
        for member in members:                  # first readable image wins
            if dest.exists():
                index[qid] = index.get(qid, member)
                break
            try:
                with Image.open(io.BytesIO(zf.read(member))) as im:
                    im = im.convert("RGB")
                    im.thumbnail((args.max_side, args.max_side))
                    im.save(dest, "JPEG", quality=90)
                index[qid] = member
                break
            except Exception:
                unreadable += 1

    (mkgw / "author_images_index.json").write_text(json.dumps(index, indent=0), encoding="utf-8")
    print(f"entities with image: {len(index)}, unmatched folders: {unmatched_folders}, "
          f"unreadable files skipped: {unreadable}")


if __name__ == "__main__":
    main()
