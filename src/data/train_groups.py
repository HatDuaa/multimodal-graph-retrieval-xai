"""Deterministic train image groups for graph reranker training."""
import hashlib
import json
import random
from pathlib import Path

from src.utils.config import load_config, resolve


def make_train_groups(train_image_ids, seed, dev_size=0, group_size=2140) -> dict:
    """Split sorted train ids into a deterministic optional dev pool and balanced groups."""
    ids = sorted(train_image_ids)
    if dev_size < 0 or group_size <= 0 or dev_size > len(ids):
        raise ValueError("invalid dev_size or group_size")
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    dev = shuffled[:dev_size]
    remaining = shuffled[dev_size:]
    n_groups = (len(remaining) + group_size - 1) // group_size
    if not n_groups:
        return {"seed": seed, "dev": dev, "groups": []}
    base, extra = divmod(len(remaining), n_groups)
    groups = []
    offset = 0
    for index in range(n_groups):
        width = base + (index < extra)
        groups.append(remaining[offset:offset + width])
        offset += width
    return {"seed": seed, "dev": dev, "groups": groups}


def split_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = load_config()
    split_path = resolve(cfg, "splits") / "vg_coco_train.json"
    output = resolve(cfg, "splits") / "vg_coco_train_groups.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    image_ids = [image["image_id"] for image in payload["images"]]
    result = make_train_groups(image_ids, cfg["seed"])
    result["train_split_sha256"] = split_sha256(split_path)
    text = json.dumps(result, indent=2) + "\n"
    print(f"expected {output}: {len(text.encode('utf-8'))} bytes")
    output.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {output} ({output.stat().st_size} bytes), dev={len(result['dev'])}, "
          f"groups={[len(group) for group in result['groups']]}")


if __name__ == "__main__":
    main()
