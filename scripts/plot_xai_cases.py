"""Figure of the level-3 explanation cases: the rank-1 image of countonly_factual_r3 (seed 0) with the top-3 parts of its
explanation E drawn on it, for a few captions of experiments/xai-metrics-v2/cases.json.

An object node is drawn as its box; a labelled triple as the boxes of its subject and object joined by a line. Each part
carries its position in E (1, 2, 3) as a label, so the colour is never the only cue. Boxes come from the merge_v1 image
graphs, the graphs countonly scores. A part whose label matches several nodes (or triples) of the graph is drawn on all
of them, and the caption under the image says so.

Usage:
  python scripts/plot_xai_cases.py [--cases 574350_4 445695_1 422698_1 1153_3] [--out report/latex/figures/xai/xai-cases.pdf]
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "experiments" / "xai-metrics-v2" / "cases.json"
GRAPHS = REPO / "data" / "graphs" / "merge_v1" / "vg_coco_scene_graphs.jsonl"
IMAGES = REPO / "data" / "raw" / "images"
MODEL = "countonly_factual_r3"
DEFAULT_CASES = ("574350_4", "445695_1", "422698_1", "1153_3")
# categorical slots 1-3 of the dataviz reference palette (validated: CVD and normal-vision separation pass)
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")


def load_graphs(image_ids):
    graphs = {}
    with GRAPHS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["image_id"]) in image_ids:
                graphs[str(row["image_id"])] = row
    return graphs


def triple_text(graph, relation):
    names = {o["id"]: o["name"] for o in graph["objects"]}
    return f'{names[relation["subject"]]} {relation["predicate"]} {names[relation["object"]]}'


def locate(graph, part):
    """Boxes of every node (or every triple's two ends) of the graph that carry this part's label."""
    objects = {o["id"]: o for o in graph["objects"]}
    if part["kind"] == "node":
        return [("node", o["box"]) for o in graph["objects"] if o["name"] == part["part"] and o.get("box")]
    found = []
    for relation in graph["relations"]:
        if triple_text(graph, relation) == part["part"]:
            subject, target = objects[relation["subject"]], objects[relation["object"]]
            if subject.get("box") and target.get("box"):
                found.append(("triple", subject["box"], target["box"]))
    return found


def centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def draw_box(ax, box, colour, dashed=False):
    x, y, w, h = box
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=4.0))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colour, linewidth=2.2,
                           linestyle=(0, (4, 2)) if dashed else "solid"))


def tag(ax, x, y, text, colour):
    ax.text(x, y, text, fontsize=8, fontweight="bold", color="white", ha="left", va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": colour, "edgecolor": "white", "linewidth": 0.8})


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--out", default=str(REPO / "report" / "latex" / "figures" / "xai" / "xai-cases.pdf"))
    args = parser.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    entries = {cid: cases[cid][MODEL] for cid in args.cases}
    graphs = load_graphs({str(e["image_id"]) for e in entries.values()})

    columns = 2
    rows = (len(args.cases) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(7.0, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, cid in zip(axes.flat, args.cases):
        entry = entries[cid]
        graph = graphs[str(entry["image_id"])]
        image = Image.open(IMAGES / f'{entry["image_id"]}.jpg').convert("RGB")
        scale = image.width / graph["width"] if graph.get("width") else 1.0
        ax.imshow(image)
        top = entry["explanation"][:3]
        # parts with the same label cannot be told apart in the graph: draw them once, tagged with all their positions
        groups = {}
        for position, part in enumerate(top, start=1):
            groups.setdefault((part["kind"], part["part"]), []).append(position)
        legend, many = [], False
        for (kind, label), positions in groups.items():
            colour = COLOURS[positions[0] - 1]
            text = ", ".join(str(p) for p in positions)
            found = locate(graph, {"kind": kind, "part": label})
            many |= len(found) > len(positions)
            for item in found:
                if item[0] == "node":
                    box = [v * scale for v in item[1]]
                    draw_box(ax, box, colour)
                    tag(ax, box[0] + 2, box[1] + 2, text, colour)
                else:
                    sbox, obox = ([v * scale for v in b] for b in item[1:])
                    draw_box(ax, sbox, colour, dashed=True)
                    draw_box(ax, obox, colour, dashed=True)
                    (x1, y1), (x2, y2) = centre(sbox), centre(obox)
                    ax.plot([x1, x2], [y1, y2], color="white", linewidth=4.0, solid_capstyle="round")
                    ax.plot([x1, x2], [y1, y2], color=colour, linewidth=2.0, solid_capstyle="round")
                    tag(ax, sbox[0] + 2, sbox[1] + 2, text, colour)
            name = "node" if kind == "node" else "bộ ba"
            legend.append(f'{text}. {label} ({name})')
        verdict = "ảnh đúng" if entry["correct"] else "ảnh sai"
        title = f'{cid} ({verdict})'
        caption = textwrap.fill(f'“{cases[cid]["caption"]}”', 58)
        detail = "; ".join(legend)
        if many:
            detail += " (đồ thị có nhiều phần cùng nhãn hơn; vẽ tất cả)"
        note = textwrap.fill(detail, 62) + "\n" + textwrap.fill(vn_number(rank_text(entry)), 62)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.text(0, -0.03, caption + "\n" + note, transform=ax.transAxes, fontsize=7, va="top", ha="left",
                color="#222222")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02, wspace=0.05, hspace=0.78)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(out)


if __name__ == "__main__":
    main()
