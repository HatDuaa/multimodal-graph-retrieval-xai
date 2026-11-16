"""Pack the Vietnamese MMKG raw cache into a zip for the group Drive and print its checksum.

Only what the build needs: candidates.json, wikidata/, viwiki/, viwiki_links.json, commons/,
candidates/. Entries are sorted and timestamps fixed, so the same cache gives the same zip.
Usage:  python scripts/archive_vnmmkg_raw.py
Output: data/raw/vnmmkg_raw_<fetch date>.zip
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

PARTS = ["candidates.json", "wikidata", "viwiki", "viwiki_links.json", "commons", "candidates"]


def main() -> None:
    v = load_config()["vnmmkg"]
    raw_dir = REPO_ROOT / v["raw_dir"]
    fetched = json.loads((raw_dir / "candidates.json").read_text(encoding="utf-8"))["fetched"]
    out = raw_dir.parent / f"vnmmkg_raw_{fetched}.zip"
    files = sorted(p for part in PARTS for p in ([raw_dir / part] if (raw_dir / part).is_file()
                                                 else (raw_dir / part).rglob("*")) if p.is_file())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(raw_dir.parent).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    print(f"{out}  {out.stat().st_size} bytes  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}  ({len(files)} files)")


if __name__ == "__main__":
    main()
