"""Build the Visual Genome ∩ COCO split files (format: docs/interfaces.md, section 1).

Every Visual Genome image whose coco_id appears in the Karpathy file is kept; nothing is
sampled, so the split depends only on the two source files. Karpathy's "restval" images are
merged into train, as is customary. Output: data/splits/vg_coco_{train,val,test}.json and
a statistics table in experiments/data_stats.md.

Usage:  python -m src.data.build_split
"""
import json
from collections import Counter

from src.utils.config import REPO_ROOT, load_config, resolve

DATASET = "vg_coco"
SPLIT_OF = {"train": "train", "restval": "train", "val": "val", "test": "test"}


def main() -> None:
    cfg = load_config()
    raw = resolve(cfg, "raw")
    vg = json.loads((raw / "visual_genome" / "image_data.json").read_text())
    coco = json.loads((raw / "coco" / "dataset_coco.json").read_text())["images"]
    by_coco = {im["cocoid"]: im for im in coco}
    assert len(by_coco) == len(coco), "duplicate cocoid in the Karpathy file"

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    seen_coco, dropped_dup = set(), 0
    karpathy_counts = Counter()
    for v in sorted(vg, key=lambda x: x["image_id"]):
        c = by_coco.get(v.get("coco_id"))
        if c is None:
            continue
        if c["cocoid"] in seen_coco:  # two Visual Genome ids for one COCO image: keep the first
            dropped_dup += 1
            continue
        seen_coco.add(c["cocoid"])
        karpathy_counts[c["split"]] += 1
        splits[SPLIT_OF[c["split"]]].append({
            "image_id": v["image_id"],
            "coco_id": c["cocoid"],
            "file_name": f"{v['image_id']}.jpg",
            "url": v["url"],
            "width": v["width"],
            "height": v["height"],
            "captions": [{"caption_id": f"{c['cocoid']}_{i}", "text": " ".join(s["raw"].split())}
                         for i, s in enumerate(c["sentences"])],
        })

    ids = [im["image_id"] for part in splits.values() for im in part]
    assert len(ids) == len(set(ids)), "an image appears in two splits"

    out_dir = resolve(cfg, "splits")
    for name, images in splits.items():
        payload = {"dataset": DATASET, "split": name, "images": images}
        (out_dir / f"{DATASET}_{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Thống kê dữ liệu — Visual Genome ∩ COCO",
        "",
        "Sinh bởi `python -m src.data.build_split`. Không sửa tay.",
        "",
        f"- Ảnh trong Visual Genome: {len(vg)}",
        f"- Ảnh có `coco_id` nằm trong file split Karpathy: {sum(karpathy_counts.values()) + dropped_dup}",
        f"- Bỏ vì trùng `coco_id` (hai ảnh Visual Genome cho cùng một ảnh COCO): {dropped_dup}",
        f"- Theo split Karpathy gốc: {dict(sorted(karpathy_counts.items()))} (`restval` gộp vào train)",
        "",
        "| Split | Số ảnh | Số caption (truy vấn) |",
        "|---|---|---|",
    ]
    for name, images in splits.items():
        lines.append(f"| {name} | {len(images)} | {sum(len(im['captions']) for im in images)} |")
    stats = REPO_ROOT / cfg["paths"]["experiments"] / "data_stats.md"
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
