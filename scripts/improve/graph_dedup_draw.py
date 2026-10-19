"""Draw image graphs before and after cleaning on train images (graph cleaning audit, visual check). CPU only.

- 30 random train images (seed 0): row 1 = before, merged at IoU 0.5 / 0.7 / 0.9 (merged nodes thick, with the number
  of boxes they absorbed); row 2 = grouping on the 0.7 merge by (a), (b), (a or b), near (groups thick, "dog x4").
- Examples of each grouping rule and of their disagreements: images whose (a) groups, (b)-only groups, and anchors
  where (a) and (b) disagree, drawn with the anchor and the pair.
Writes PNG files to experiments/improve/graph-dedup-audit/img/ and an index markdown file.
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.graph_dedup_audit import load_split  # noqa: E402
from src.graph.dedup import (geometry_class, group_by_role, merge_duplicates, pair_relation, roles, same_type,  # noqa: E402
                             with_boxes)
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "graph-dedup-audit"
PANEL = 300
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#469990", "#800000"]


def panel(image, graph, title, highlight):
    """One panel: the image scaled to PANEL px wide, every node's box; highlight(node) -> (thick, label) or None."""
    scale = PANEL / image.width
    canvas = image.resize((PANEL, int(image.height * scale))).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    names = sorted({o["name"] for o in graph["objects"]})
    for o in graph["objects"]:
        x, y, w, h = [v * scale for v in o["box"]]
        mark = highlight(o)
        color = COLORS[names.index(o["name"]) % len(COLORS)] if mark else "#bbbbbb"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3 if mark else 1)
        if mark and mark[1]:
            draw.text((x + 2, y + 1), mark[1], fill=color)
    out = Image.new("RGB", (PANEL, canvas.height + 14), "white")
    out.paste(canvas, (0, 14))
    ImageDraw.Draw(out).text((2, 1), title, fill="black")
    return out


def grid(panels, columns=4):
    rows = [panels[i:i + columns] for i in range(0, len(panels), columns)]
    height = sum(max(p.height for p in row) for row in rows)
    out = Image.new("RGB", (PANEL * columns, height), "white")
    y = 0
    for row in rows:
        for k, p in enumerate(row):
            out.paste(p, (k * PANEL, y))
        y += max(p.height for p in row)
    return out


def save(image, path):
    image.quantize(colors=128).save(path, optimize=True)


def merged_mark(o):
    return (True, f"{o['name']} x{o['boxes']} boxes") if o["boxes"] > 1 else None


def group_mark(o):
    return (True, f"{o['name']} x{o['count']}") if o["count"] > 1 else None


def overview(image_path, g0):
    panels = [panel(Image.open(image_path), g0, "before", lambda o: (True, ""))]
    for t in (.5, .7, .9):
        panels.append(panel(Image.open(image_path), merge_duplicates(g0, t)[0], f"merge IoU>={t}", merged_mark))
    merged = merge_duplicates(g0, .7)[0]
    for criterion, title in (("a", "group (a) predicate"), ("b", "group (b) geometry"), ("ab", "group (a) or (b)"),
                             ("near", "group near")):
        panels.append(panel(Image.open(image_path), group_by_role(merged, criterion)[0], title, group_mark))
    return grid(panels)


def pair_figure(image_path, graph, i, j, anchor, title):
    objects = graph["objects"]
    keep = {i: f"{objects[i]['name']} ({geometry_class(objects[i]['box'], objects[anchor]['box'])})",
            j: f"{objects[j]['name']} ({geometry_class(objects[j]['box'], objects[anchor]['box'])})",
            anchor: f"anchor {objects[anchor]['name']}"}
    return panel(Image.open(image_path), graph, title, lambda o: (True, keep[o["id"]]) if o["id"] in keep else None)


def predicates(graph, node, anchor):
    return sorted({r["predicate"] for r in graph["relations"] if {r["subject"], r["object"]} == {node, anchor}})


def main() -> None:
    cfg = load_config()
    graphs, objects = load_split(cfg, "train")
    images = {im["image_id"]: im["file_name"] for im in json.loads(
        (resolve(cfg, "splits") / "vg_coco_train.json").read_text(encoding="utf-8"))["images"]}
    folder = resolve(cfg, "raw") / "images"
    img_dir = OUT / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    ids = sorted(i for i in graphs if (folder / images[i]).exists())
    sample = rng.sample(ids, 30)
    lines = ["# Kiểm tra bằng mắt: gộp trùng và gom nhóm (30 ảnh train ngẫu nhiên, seed 0)", "",
             "Hàng trên: trước, gộp ở IoU ≥ 0,5 / 0,7 / 0,9 (node được gộp vẽ đậm, ghi số khung đã gộp; khung xám là node "
             "không đổi). Hàng dưới: gom nhóm trên bản gộp 0,7 theo (a) vị từ, (b) hình học, (a) hoặc (b), và gần nhau "
             "(node nhóm vẽ đậm, ghi \"tên x số lượng\").", ""]
    for image_id in sample:
        g0 = with_boxes(graphs[image_id], objects[str(image_id)]["boxes"], objects[str(image_id)]["synsets"])
        name = f"overview_{image_id}.png"
        save(overview(folder / images[image_id], g0), img_dir / name)
        lines += [f"## Ảnh {image_id}", "", f"![{image_id}](img/{name})", ""]
    examples = {"a_group": [], "b_only_group": [], "a_not_b": [], "b_not_a": []}
    for image_id in rng.sample(ids, 3000):
        if all(len(v) >= 3 for v in examples.values()):
            break
        g0 = with_boxes(graphs[image_id], objects[str(image_id)]["boxes"], objects[str(image_id)]["synsets"])
        merged = merge_duplicates(g0, .7)[0]
        role, objs = roles(merged), merged["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if not same_type(objs[i]["synset"], objs[j]["synset"], objs[i]["name"], objs[j]["name"]):
                    continue
                by_a, by_b, a_not_b, b_not_a = pair_relation(merged, role, i, j)
                anchors = ({a for _, _, a in role[i]} & {a for _, _, a in role[j]}) - {i, j}
                for kind, hit in (("a_group", by_a), ("b_only_group", by_b and not by_a), ("a_not_b", a_not_b),
                                  ("b_not_a", b_not_a)):
                    if hit and len(examples[kind]) < 3 and image_id not in [e[0] for e in examples[kind]]:
                        candidates = anchors or ({a for _, _, a in role[i]} | {a for _, _, a in role[j]}) - {i, j}
                        anchor = min(candidates)
                        examples[kind].append((image_id, merged, i, j, anchor))
    titles = {"a_group": "(a) cùng vị từ với cùng node neo", "b_only_group": "(b) chỉ cùng hình học (VG thiếu nhãn quan hệ)",
              "a_not_b": "mâu thuẫn: cùng vị từ, khác hình học", "b_not_a": "mâu thuẫn: cùng hình học, khác vị từ"}
    lines += ["# Ví dụ cho từng tiêu chí gom nhóm (gộp trùng ở IoU 0,7)", ""]
    for kind, items in examples.items():
        lines += [f"## {titles[kind]}", ""]
        for image_id, merged, i, j, anchor in items:
            name = f"{kind}_{image_id}.png"
            ascii_title = {"a_group": "(a) same predicate, same anchor", "b_only_group": "(b) same geometry only",
                           "a_not_b": "conflict: same predicate, other geometry",
                           "b_not_a": "conflict: same geometry, other predicate"}[kind]  # PIL's default font has no diacritics
            save(pair_figure(folder / images[image_id], merged, i, j, anchor, ascii_title), img_dir / name)
            lines += [f"- Ảnh {image_id}: {merged['objects'][i]['name']} {predicates(merged, i, anchor)} / "
                      f"{merged['objects'][j]['name']} {predicates(merged, j, anchor)} — neo "
                      f"{merged['objects'][anchor]['name']}", "", f"  ![{kind} {image_id}](img/{name})", ""]
    (OUT / "visual_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("drawn", len(sample), {k: len(v) for k, v in examples.items()}, flush=True)


if __name__ == "__main__":
    main()
