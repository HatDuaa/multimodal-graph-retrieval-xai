"""Visual Genome boxes for the packaged image graphs of one split (part 2a).

The packaged graphs (data/graphs/vg_coco_scene_graphs.jsonl) keep labels only. They are rebuilt here with the same
code (iter_scene_graphs, open vocabulary) plus the source VG object ids; the rebuilt objects and relations must
equal the packaged ones, so object i of a packaged graph gets the box of its source object. Boxes come from
objects.json, or from the relationship endpoint for objects that appear in relationships.json only.
Writes data/processed/geometry/boxes_<split>.json: {image_id: {"width", "height", "boxes": [[x, y, w, h], ...]}};
with --synsets, data/processed/geometry/objects_<split>.json instead, which adds "synsets" (the first VG synset of each
object, or null).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.scene_graph import Vocab, build_scene_graph, load_raw  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--synsets", action="store_true")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    images = json.loads((resolve(cfg, "splits") / f"vg_coco_{args.split}.json").read_text(encoding="utf-8"))["images"]
    size = {im["image_id"]: (im["width"], im["height"]) for im in images}
    packaged = {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] == args.split:
                packaged[graph["image_id"]] = graph
    objects, relationships = load_raw(cfg)
    vocab = Vocab.from_config(cfg)
    out, missing_boxes = {}, 0
    for image_id in sorted(packaged):
        empty = {"image_id": image_id}
        entry_objects, entry_rels = objects.get(image_id, empty), relationships.get(image_id, empty)
        rebuilt = build_scene_graph(entry_objects, entry_rels, vocab, vg150_only=False, include_source_ids=True)
        reference = packaged[image_id]
        assert [(o["id"], o["name"]) for o in rebuilt["objects"]] == [(o["id"], o["name"]) for o in reference["objects"]], image_id
        assert rebuilt["relations"] == reference["relations"], image_id
        box_of, synset_of = {}, {}
        for rel in entry_rels.get("relationships", []):
            for endpoint in (rel.get("subject", {}), rel.get("object", {})):
                if "x" in endpoint:
                    box_of.setdefault(endpoint["object_id"], [endpoint["x"], endpoint["y"], endpoint["w"], endpoint["h"]])
                    synset_of.setdefault(endpoint["object_id"], (endpoint.get("synsets") or [None])[0])
        for obj in entry_objects.get("objects", []):
            box_of[obj["object_id"]] = [obj["x"], obj["y"], obj["w"], obj["h"]]
            synset_of[obj["object_id"]] = (obj.get("synsets") or [None])[0]
        boxes = [box_of.get(o["source_object_id"]) for o in rebuilt["objects"]]
        missing_boxes += sum(b is None for b in boxes)
        out[image_id] = {"width": size[image_id][0], "height": size[image_id][1], "boxes": boxes}
        if args.synsets:
            out[image_id]["synsets"] = [synset_of.get(o["source_object_id"]) for o in rebuilt["objects"]]
    name = f"objects_{args.split}.json" if args.synsets else f"boxes_{args.split}.json"
    path = resolve(cfg, "processed") / "geometry" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({"split": args.split, "images": len(out), "objects": sum(len(v["boxes"]) for v in out.values()),
                      "objects_without_box": missing_boxes, "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
