"""Step 3 of the Vietnamese MMKG pipeline: image candidates from Wikimedia Commons.

For each of the top `max_landmark_candidates` landmarks: list the files directly in its
Commons category (P373, no subcategories), add its P18 image, read size, MIME type and
license metadata (imageinfo), pick up to `max_image_candidates` files (src/data/vnmmkg_commons.py),
download a Commons thumbnail at a standard width and store it as a 640 px JPEG with its SHA-256.

Usage:  python scripts/fetch_vnmmkg_images.py
Output: data/raw/vnmmkg/commons/<qid>.json, data/raw/vnmmkg/candidates/<sha1-of-title>.jpg
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.vnmmkg_cache import landmark_candidates  # noqa: E402
from src.data.vnmmkg_commons import local_file_name, pick_candidates, to_thumbnail_jpeg  # noqa: E402
from src.data.vnmmkg_wikidata import truthy  # noqa: E402
from src.data.wikimedia_client import COMMONS_API, client_from_config  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402


def category_files(client, category: str) -> list[str]:
    files, cont = [], {}
    while True:
        data = client.get_json(COMMONS_API, {"action": "query", "list": "categorymembers", "cmtype": "file",
                                             "cmtitle": f"Category:{category}", "cmlimit": 500,
                                             "formatversion": 2, "maxlag": 5, **cont})
        files += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return files
        cont = {"cmcontinue": data["continue"]["cmcontinue"]}


def image_infos(client, titles: list[str], request_width: int) -> dict[str, dict]:
    infos = {}
    for start in range(0, len(titles), 50):
        data = client.get_json(COMMONS_API, {"action": "query", "prop": "imageinfo",
                                             "iiprop": "url|size|mime|extmetadata", "iiurlwidth": request_width,
                                             "iiextmetadatafilter": "License|LicenseShortName|LicenseUrl|Artist",
                                             "titles": "|".join(titles[start:start + 50]),
                                             "formatversion": 2, "maxlag": 5})
        normalized = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
        for page in data["query"]["pages"]:
            if page.get("imageinfo"):
                infos[normalized.get(page["title"], page["title"])] = page["imageinfo"][0]
    return infos


def main() -> None:
    cfg = load_config()
    v = cfg["vnmmkg"]
    raw_dir = REPO_ROOT / v["raw_dir"]
    client = client_from_config(cfg)
    ranked, entities = landmark_candidates(raw_dir, v)
    (raw_dir / "commons").mkdir(parents=True, exist_ok=True)
    (raw_dir / "candidates").mkdir(parents=True, exist_ok=True)
    totals = Counter()
    for i, qid in enumerate(ranked, 1):
        out = raw_dir / "commons" / f"{qid}.json"
        if out.exists():
            totals["candidates"] += len(json.loads(out.read_text(encoding="utf-8"))["candidates"])
            continue
        category = truthy(entities[qid], "P373")[0]
        p18 = truthy(entities[qid], "P18")
        p18 = f"File:{p18[0]}" if p18 else None
        files = category_files(client, category)
        titles = sorted(set(files) | ({p18} if p18 else set()))
        picked, drops = pick_candidates(p18, files, image_infos(client, titles, v["thumb_request_width"]), v)
        for cand in picked:
            dest = raw_dir / "candidates" / local_file_name(cand["file_title"])
            if not dest.exists():
                jpeg, _, _ = to_thumbnail_jpeg(client.download(cand["thumb_url"]), v["thumb_width"])
                dest.write_bytes(jpeg)
            with Image.open(dest) as im:
                cand["width"], cand["height"] = im.size
            cand["local_file"] = dest.name
            cand["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        out.write_text(json.dumps({"qid": qid, "category": category, "p18": p18, "n_category_files": len(files),
                                   "drops": drops, "candidates": picked}, ensure_ascii=False, indent=1),
                       encoding="utf-8", newline="\n")
        totals["candidates"] += len(picked)
        totals.update(drops)
        print(f"landmarks {i}/{len(ranked)}, candidates so far {totals['candidates']}", end="\r")
    print(f"\ndone: {dict(totals)}")


if __name__ == "__main__":
    main()
