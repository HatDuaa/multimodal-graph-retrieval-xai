"""Wikimedia Commons helpers for the Vietnamese MMKG images (spec 4.3). Pure functions.

Licenses are read from the machine-readable `License` field of Commons `extmetadata`
(e.g. `cc-by-sa-4.0`); only CC0, public domain, CC BY and CC BY-SA are accepted, so NC/ND
variants and GFDL-only files are rejected. Candidate order: the Wikidata main image (P18)
first, then the category's files by title; the first `max_image_candidates` files that pass
the format, width and license filters are kept for human review.
"""
import hashlib
import html
import io
import re
import unicodedata
from collections import Counter

from PIL import Image, ImageOps


def normalize_license(extmeta: dict) -> str | None:
    code = ((extmeta.get("License") or {}).get("value") or "").strip().lower()
    if code.startswith("cc0"):
        return "CC0"
    if code.startswith("pd"):
        return "Public Domain"
    m = re.fullmatch(r"cc-by(-sa)?-(\d(?:\.\d)?)(?:-[a-z]{2,})?", code)
    if m:
        return f"CC BY{'-SA' if m.group(1) else ''} {m.group(2)}"
    return None


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


def local_file_name(file_title: str) -> str:
    return hashlib.sha1(unicodedata.normalize("NFC", file_title).encode("utf-8")).hexdigest()[:16] + ".jpg"


def pick_candidates(p18: str | None, category_files: list[str], infos: dict[str, dict],
                    cfg_v: dict) -> tuple[list[dict], dict[str, int]]:
    order = ([p18] if p18 else []) + sorted(f for f in set(category_files) if f != p18)
    picked, drops = [], Counter()
    for title in order:
        if len(picked) == cfg_v["max_image_candidates"]:
            break
        info = infos.get(title)
        if info is None:
            drops["no_info"] += 1
            continue
        if info["mime"] not in cfg_v["allowed_mime"]:
            drops["mime"] += 1
            continue
        if info["width"] < cfg_v["min_image_width"]:
            drops["too_small"] += 1
            continue
        meta = info.get("extmetadata", {})
        license_name = normalize_license(meta)
        if license_name is None:
            drops["license"] += 1
            continue
        picked.append({"file_title": title, "source_url": info["descriptionurl"], "thumb_url": info["thumburl"],
                       "orig_width": info["width"], "orig_height": info["height"], "license": license_name,
                       "license_url": (meta.get("LicenseUrl") or {}).get("value") or None,
                       "artist": strip_html((meta.get("Artist") or {}).get("value") or "") or None})
    return picked, dict(sorted(drops.items()))


def to_thumbnail_jpeg(data: bytes, width: int) -> tuple[bytes, int, int]:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), image.width, image.height
