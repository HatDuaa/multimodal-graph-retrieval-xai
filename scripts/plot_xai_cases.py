"""Figure of the level-3 explanation cases: the rank-1 image of countonly_factual_r3 (seed 0) with the top-3 parts of its
explanation E drawn on it, for a few captions of experiments/xai-metrics-v2/cases.json.

An object node is drawn as a solid box; a labelled triple as the dashed boxes of its subject and object joined by a line.
Each image node is drawn once, whatever the number of parts that use it: its box is solid when a node part uses it, and
its tag lists the position in E (1, 2, 3) of every part that touches it, so a tag is never hidden and the colour is never
the only cue. Parts with the same label cannot be told apart in the graph, so they share one tag ("2, 3"); a label that
matches more nodes (or triples) than parts is drawn on all of them, and the caption under the image says so. Paths come
from the countonly configuration (configs/count_v1.yaml), whose image graphs are the ones the model scores.

Usage:
  python scripts/plot_xai_cases.py [--cases 574350_4 445695_1 422698_1 1153_3] [--out report/latex/figures/xai/xai-cases.pdf]
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

MODEL, CONFIG = "countonly_factual_r3", REPO_ROOT / "configs" / "count_v1.yaml"
DEFAULT_CASES = ("574350_4", "445695_1", "422698_1", "1153_3")
# categorical slots 1-3 of the dataviz reference palette (validated: CVD and normal-vision separation pass)
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")


def load_graphs(path, image_ids):
    graphs = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["image_id"]) in image_ids:
                graphs[str(row["image_id"])] = row
    return graphs


def triple_text(graph, relation):
    names = {o["id"]: o["name"] for o in graph["objects"]}
    return f'{names[relation["subject"]]} {relation["predicate"]} {names[relation["object"]]}'


def locate(graph, kind, label):
    """Object ids of every node (kind "node") or every triple's (subject, object) (kind "triple") with this label."""
    boxed = {o["id"] for o in graph["objects"] if o.get("box")}
    if kind == "node":
        return [(o["id"],) for o in graph["objects"] if o["name"] == label and o["id"] in boxed]
    return [(r["subject"], r["object"]) for r in graph["relations"]
            if triple_text(graph, r) == label and r["subject"] in boxed and r["object"] in boxed]


def centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def draw_box(ax, box, colour, dashed):
    x, y, w, h = box
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=4.0))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colour, linewidth=2.2,
                           linestyle=(0, (4, 2)) if dashed else "solid"))


def tag(ax, x, y, text, colour, placed):
    """A position tag at the box corner, moved down until it overlaps none of the tags already placed."""
    label = ax.text(x, y, text, fontsize=8, fontweight="bold", color="white", ha="left", va="top", zorder=5,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": colour, "edgecolor": "white", "linewidth": 0.8})
    renderer = ax.figure.canvas.get_renderer()
    extent = label.get_window_extent(renderer).padded(4)
    while any(extent.overlaps(other) for other in placed):
        x_px, y_px = ax.transData.transform((x, y))
        y = ax.transData.inverted().transform((x_px, y_px - extent.height - 1))[1]
        label.set_position((x, y))
        extent = label.get_window_extent(renderer).padded(4)
    placed.append(extent)


def rank_text(entry):
    plus, minus = entry["fidelity"]["3"]["plus"], entry["fidelity"]["3"]["minus"]
    return (f'Bỏ top-3: hạng {plus["top"]["rank"]} (ngẫu nhiên {plus["random"]["rank"]:.1f}). '
            f'Giữ top-3: hạng {minus["top"]["rank"]} (ngẫu nhiên {minus["random"]["rank"]:.1f}).')


def vn_number(text):
    """Decimal commas, as in the report."""
    out, previous = [], ""
    for i, ch in enumerate(text):
        nxt = text[i + 1] if i + 1 < len(text) else ""
        out.append("," if ch == "." and previous.isdigit() and nxt.isdigit() else ch)
        previous = ch
    return "".join(out)


def draw_case(ax, cid, caption, entry, graph, image):
    """Draw the top-3 parts of E on the image, each image node once; return the legend lines and the "many" flag."""
    scale = image.width / graph["width"] if graph.get("width") else 1.0
    boxes = {o["id"]: [v * scale for v in o["box"]] for o in graph["objects"] if o.get("box")}
    ax.imshow(image)
    ax.apply_aspect()  # imshow fixes the axes box only at draw time; the tag overlap test needs it now
    groups = {}
    for position, part in enumerate(entry["explanation"][:3], start=1):
        groups.setdefault((part["kind"], part["part"]), []).append(position)
    marks, links, legend, many = {}, [], [], False
    for (kind, label), positions in groups.items():
        found = locate(graph, kind, label)
        if not found:
            raise SystemExit(f"{cid}: part {label!r} ({kind}) of E is not in the image graph")
        many |= len(found) > len(positions)
        for ids in found:
            for oid in ids:
                mark = marks.setdefault(oid, {"positions": set(), "solid": False})
                mark["positions"].update(positions)
                mark["solid"] |= kind == "node"
            if kind == "triple":
                links.append((ids, min(positions)))
        legend.append(f'{", ".join(map(str, positions))}. {label} ({"node" if kind == "node" else "bộ ba"})')
    # dashed (triple ends) first, so a solid node box is never painted over; lines next; tags last, always on top
    for oid, mark in sorted(marks.items(), key=lambda item: item[1]["solid"]):
        draw_box(ax, boxes[oid], COLOURS[min(mark["positions"]) - 1], dashed=not mark["solid"])
    for (subject, target), position in links:
        (x1, y1), (x2, y2) = centre(boxes[subject]), centre(boxes[target])
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=4.0, solid_capstyle="round", zorder=3)
        ax.plot([x1, x2], [y1, y2], color=COLOURS[position - 1], linewidth=2.0, solid_capstyle="round", zorder=4)
    placed = []
    for oid, mark in sorted(marks.items(), key=lambda item: (boxes[item[0]][1], boxes[item[0]][0])):
        positions = sorted(mark["positions"])
        box = boxes[oid]
        tag(ax, box[0] + 2, box[1] + 2, ", ".join(map(str, positions)), COLOURS[positions[0] - 1], placed)

    verdict = "ảnh đúng" if entry["correct"] else "ảnh sai"
    detail = "; ".join(legend) + (" (đồ thị có nhiều phần cùng nhãn hơn; vẽ tất cả)" if many else "")
    text = (textwrap.fill(f"“{caption}”", 58) + "\n" + textwrap.fill(detail, 62) + "\n"
            + textwrap.fill(vn_number(rank_text(entry)), 62))
    ax.set_title(f"{cid} ({verdict})", fontsize=9, fontweight="bold", loc="left")
    ax.text(0, -0.03, text, transform=ax.transAxes, fontsize=7, va="top", ha="left", color="#222222")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--out", default=str(REPO_ROOT / "report" / "latex" / "figures" / "xai" / "xai-cases.pdf"))
    args = parser.parse_args()

    cfg = load_config(CONFIG)
    cases = json.loads((resolve(cfg, "experiments") / "xai-metrics-v2" / "cases.json").read_text(encoding="utf-8"))
    entries = {cid: cases[cid][MODEL] for cid in args.cases}
    graphs = load_graphs(resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl",
                         {str(e["image_id"]) for e in entries.values()})
    images = resolve(cfg, "raw") / "images"

    columns = 2
    rows = (len(args.cases) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(7.0, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    # final layout before drawing, so the tag overlap test sees the pixel sizes that are saved
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02, wspace=0.05, hspace=0.78)
    for ax, cid in zip(axes.flat, args.cases):
        entry = entries[cid]
        image = Image.open(images / f'{entry["image_id"]}.jpg').convert("RGB")
        draw_case(ax, cid, cases[cid]["caption"], entry, graphs[str(entry["image_id"])], image)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(out)


if __name__ == "__main__":
    main()
