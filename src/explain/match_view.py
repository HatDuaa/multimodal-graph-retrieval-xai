"""Where each explanation row lands in the image: boxes drawn over the image and the caption ↔ image match graph.

Everything drawn here comes from the ranking pass (graph_explainer.matched_pairs): a row's top_matches are the image
parts with the largest a_ij * sim_ij, the terms of its row_score; nothing is searched after the fact. Boxes are the
[x, y, w, h] pixel boxes of the image graph (data version merge_v1 and later, where a merged node keeps the box around
its members); a graph without boxes (level 1) gets no overlay, and the global node and the kNN [none] edges are never
matched, so they are never drawn. The SVG itself is static: focus on hover comes from focus_css(), focus on click from
FOCUS_JS, both keyed by the data-row attribute that the table rows, boxes and graph edges share.
"""
import base64
import html
import math
from pathlib import Path

from PIL import Image

COLORS = {"objects": "#ea580c", "triples": "#9333ea"}
MAX_ROWS = 40  # focus_css covers row numbers 1..MAX_ROWS
DIM = 0.12

# Click a row, a box or a graph node to keep its focus; click it again to release it, or another one to move it.
FOCUS_JS = """
element.addEventListener('click', (event) => {
  const hit = event.target.closest('[data-row]');
  const root = event.target.closest('.mx');
  if (!hit || !root) return;
  const row = hit.getAttribute('data-row');
  const same = root.getAttribute('data-pin') === row;
  root.querySelectorAll('.pin').forEach((el) => el.classList.remove('pin'));
  if (same) { root.removeAttribute('data-pin'); root.classList.remove('pinned'); return; }
  root.setAttribute('data-pin', row);
  root.classList.add('pinned');
  root.querySelectorAll('[data-row="' + row + '"], [data-rows~="' + row + '"]').forEach((el) => el.classList.add('pin'));
});
"""


def esc(value) -> str:
    return html.escape(str(value))


def part_text(value) -> str:
    return f"{value[0]} –{value[1]}→ {value[2]}" if isinstance(value, (list, tuple)) else str(value)


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def valid_box(box):
    """(x, y, w, h) floats, or None for a missing or empty box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return (x, y, w, h) if w > 0 and h > 0 else None


def image_part(graph: dict, channel: str, index: int):
    """Label and pixel boxes of one matched image part, or None when index is not a scorable part of the graph.

    objects: {"part": name, "box", "merged": number of original boxes}; triples: {"part": (subject, predicate, object),
    "subject_box", "object_box"}. A box is None when the graph has none for that node.
    """
    objects = graph.get("objects") or []
    if channel == "objects":
        if not 0 <= index < len(objects):
            return None
        obj = objects[index]
        return {"part": obj.get("name", "?"), "box": valid_box(obj.get("box")), "merged": int(obj.get("boxes") or 1)}
    relations = graph.get("relations") or []
    if not 0 <= index < len(relations) or relations[index].get("predicate") == "[none]":
        return None
    rel = relations[index]
    by_id = {obj.get("id"): obj for obj in objects}
    subject, target = by_id.get(rel.get("subject")) or {}, by_id.get(rel.get("object")) or {}
    return {"part": (subject.get("name", "?"), rel.get("predicate", "?"), target.get("name", "?")),
            "subject_box": valid_box(subject.get("box")), "object_box": valid_box(target.get("box"))}


def ordered_rows(info: dict, raw: bool) -> list[tuple[int, str, int, dict]]:
    """(row number, channel, index in info[channel], row) in table order: objects, then triples, each by decreasing
    part of the final score (channel_term under the raw fusion, contribution under the z-score one)."""
    key = "channel_term" if raw else "contribution"
    out = []
    for channel in ("objects", "triples"):
        rows = sorted(enumerate(info.get(channel) or []), key=lambda pair: -pair[1][key])
        out += [(len(out) + n + 1, channel, index, row) for n, (index, row) in enumerate(rows)]
    return out


def has_boxes(rows) -> bool:
    return any(match.get(k) for _, _, _, row in rows for match in row.get("top_matches", [])
               for k in ("box", "subject_box", "object_box"))


def image_frame(path, graph_size):
    """The image as a data URI, its file size and the scale from the graph's pixel frame; None when the graph has no
    size or the file is missing. Visual Genome sizes match the files; the scale only guards against a resized copy."""
    width, height = (list(graph_size or []) + [None, None])[:2]
    path = Path(path)
    if not width or not height or not path.exists():
        return None
    with Image.open(path) as image:
        size, mime = image.size, Image.MIME.get(image.format, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"href": f"data:{mime};base64,{data}", "size": size, "scale": (size[0] / width, size[1] / height)}


def focus_css(max_rows: int = MAX_ROWS) -> str:
    """Hover a row, a box or a graph node: dim every other mark, show the row's 2nd and 3rd matches and the captions.

    A click keeps that focus (FOCUS_JS adds .pin to the marks of the row and .pinned to the container) until the
    pointer hovers another row."""
    def rules(fmt):
        return ",\n".join(fmt.format(n=n) for n in range(1, max_rows + 1))
    hover = '.mx:has([data-row="{n}"]:hover)'
    return (rules(hover + ' .mk:not([data-row="{n}"]):not([data-rows~="{n}"])') + f" {{ opacity: {DIM}; }}\n"
            + rules(hover + ' .mk.alt[data-row="{n}"]') + " { display: inline; }\n"
            + rules(hover + ' .mk[data-row="{n}"] .cap') + " { display: inline; }\n"
            + rules(hover + ' tr[data-row="{n}"] > td') + " { background: var(--hl); }\n"
            + f".mx.pinned:not(:has([data-row]:hover)) .mk:not(.pin) {{ opacity: {DIM}; }}\n"
            ".mx.pinned:not(:has([data-row]:hover)) .mk.alt.pin { display: inline; }\n"
            ".mx.pinned:not(:has([data-row]:hover)) .mk.pin .cap { display: inline; }\n"
            ".mx.pinned:not(:has([data-row]:hover)) tr.pin > td { background: var(--hl); }\n"
            ".mx { --hl: rgba(250, 204, 21, 0.28); }\n"
            ".mx .mk { transition: opacity 0.12s; }\n"
            ".mx .mk.alt, .mx .mk .cap { display: none; }\n"
            ".mx [data-row] { cursor: pointer; }\n")


def _center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _border_point(box, start):
    """Where the segment from start to the box centre enters the box (the centre itself when start is inside)."""
    cx, cy = _center(box)
    x, y, w, h = box
    dx, dy = start[0] - cx, start[1] - cy
    if x <= start[0] <= x + w and y <= start[1] <= y + h:
        return cx, cy
    t = min(w / 2 / abs(dx) if dx else math.inf, h / 2 / abs(dy) if dy else math.inf)
    return cx + dx * t, cy + dy * t


def _scaled(box, scale):
    if box is None:
        return None
    sx, sy = scale
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def _caption(x, y, text, color, unit, width, anchor="start"):
    """Row caption (part · a), white halo, kept inside the image; hidden until the row has the focus."""
    size = 3.4 * unit
    if anchor == "start":
        x = min(max(x, 0.), max(width - len(text) * 0.58 * size, 0.))
    return (f'<text class="cap" x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" font-weight="600" fill="{color}" '
            f'stroke="#fff" stroke-width="{0.7 * unit:.1f}" paint-order="stroke" text-anchor="{anchor}">{esc(text)}</text>')


def _rect(box, color, rank):
    dash = ' stroke-dasharray="6 4"' if rank else ""
    return (f'<rect x="{box[0]:.1f}" y="{box[1]:.1f}" width="{box[2]:.1f}" height="{box[3]:.1f}" fill="{color}" '
            f'fill-opacity="0.06" stroke="{color}" stroke-width="{1.5 if rank else 2.5}"{dash} '
            'vector-effect="non-scaling-stroke"/>')


def _badge(x, y, number, color, unit):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.3 * unit:.1f}" fill="{color}" stroke="#fff" stroke-width="1.5" '
            f'vector-effect="non-scaling-stroke"/><text x="{x:.1f}" y="{y:.1f}" font-size="{2.6 * unit:.1f}" '
            f'font-weight="700" fill="#fff" text-anchor="middle" dominant-baseline="central">{number}</text>')


def match_title(number, match) -> str:
    share = "" if match.get("share") is None else f" = {match['share']:.0%} Độ khớp của dòng"
    merged = f" · node gộp {match['merged']} hộp" if match.get("merged", 1) > 1 else ""
    return (f"dòng {number} · {part_text(match.get('part', '?'))}: a = {match['a_ij']:.3f}, sim = {match['sim_ij']:.3f}, "
            f"a × sim = {match['a_ij'] * match['sim_ij']:.3f}{share}{merged}")


def _triple_shapes(subject, target, color, rank, unit, uid):
    """Subject and object boxes joined by an arrow; the anchor for the badge and caption is the arrow's midpoint."""
    boxes = [b for b in (subject, target) if b is not None]
    shapes = [_rect(b, color, rank) for b in boxes]
    if subject is None or target is None or _center(subject) == _center(target):
        return shapes, (boxes[0][0], boxes[0][1])
    start = _center(subject)
    end = _border_point(target, start)
    dash = ' stroke-dasharray="6 4"' if rank else ""
    shapes.append(f'<circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="{0.7 * unit:.1f}" fill="{color}"/>'
                  f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="{color}" '
                  f'stroke-width="{1.5 if rank else 2.5}" vector-effect="non-scaling-stroke"{dash} '
                  f'marker-end="url(#mx-arrow-{uid})"/>')
    return shapes, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)


def overlay_svg(rows, frame: dict, uid: str = "0") -> str:
    """The image with the boxes of every row's top matches, numbered like the table rows.

    rows: ordered_rows(info, raw). The best match of a row is drawn solid with the row number; the 2nd and 3rd are
    dashed and only shown while the row has the focus. Object rows box the matched node; triple rows box the subject
    and the object and join them with an arrow. Matches without a box are skipped."""
    width, height = frame["size"]
    unit = max(width, height) / 100
    marks, badges = [], []
    for number, channel, _, row in rows:
        color = COLORS[channel]
        for rank, match in enumerate(row.get("top_matches", [])):
            text = f"{part_text(match.get('part', '?'))} · a {match['a_ij']:.2f}"
            if channel == "objects":
                box = _scaled(match.get("box"), frame["scale"])
                if box is None:
                    continue
                shapes, anchor, area = [_rect(box, color, rank)], (box[0], box[1]), box[2] * box[3]
                # captions of one row step up (objects) or down (triples) by rank, so overlapping boxes stay readable
                caption = _caption(box[0] + 0.4 * unit, box[1] + box[3] - (0.8 + 4 * rank) * unit, text, color, unit, width)
            else:
                subject, target = (_scaled(match.get(k), frame["scale"]) for k in ("subject_box", "object_box"))
                if subject is None and target is None:
                    continue
                shapes, anchor = _triple_shapes(subject, target, color, rank, unit, uid)
                area = sum(b[2] * b[3] for b in (subject, target) if b is not None)
                caption = _caption(anchor[0], anchor[1] + (5.6 + 4 * rank) * unit, text, color, unit, width, "middle")
            title = f"<title>{esc(match_title(number, match))}</title>"
            marks.append((area, f'<g class="mk{" alt" if rank else ""}" data-row="{number}">{title}{"".join(shapes)}'
                                f'{caption}</g>'))
            if rank == 0:
                badges.append(f'<g class="mk" data-row="{number}">{title}{_badge(*anchor, number, color, unit)}</g>')
    marks.sort(key=lambda mark: -mark[0])  # large boxes first, so the small ones stay on top for the pointer
    arrow = (f'<marker id="mx-arrow-{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{2.2 * unit:.1f}" '
             f'markerHeight="{2.2 * unit:.1f}" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{COLORS["triples"]}"/></marker>')
    return (f'<svg class="mx-image" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="Ảnh với hộp của các phần khớp"><defs>{arrow}</defs>'
            f'<image id="mx-img-{uid}" href="{frame["href"]}" x="0" y="0" width="{width}" height="{height}"/>'
            + "".join(svg for _, svg in marks) + "".join(badges) + "</svg>")


def _crop(match, frame, uid, x, y, size=24):
    """A size × size window on the image around the matched part (its box, or both boxes of a triple)."""
    boxes = [_scaled(match.get(k), frame["scale"]) for k in ("box", "subject_box", "object_box")]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return ""
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return (f'<svg x="{x}" y="{y:.1f}" width="{size}" height="{size}" viewBox="{x0:.1f} {y0:.1f} {x1 - x0:.1f} '
            f'{y1 - y0:.1f}" preserveAspectRatio="xMidYMid slice"><use href="#mx-img-{uid}"/></svg>'
            f'<rect x="{x}" y="{y:.1f}" width="{size}" height="{size}" fill="none" stroke="#9ca3af" stroke-width="0.8"/>')


def match_graph_svg(rows, frame: dict | None = None, uid: str = "0") -> str:
    """Caption parts (left, numbered like the table) joined to the image parts in their top matches (right).

    Edge width grows with w_i * a_ij * sim_ij, the part of row i's channel score that image part j carries. An image
    part in the top matches of several rows is one node. With a frame (and the overlay of the same uid on the page),
    each image node shows its crop of the image."""
    left = [(number, channel, row) for number, channel, _, row in rows if row.get("top_matches")]
    if not left:
        return ""
    right, edges = {}, []
    for i, (number, channel, row) in enumerate(left):
        for rank, match in enumerate(row["top_matches"]):
            value = row["w_i"] * match["a_ij"] * match["sim_ij"]
            node = right.setdefault((channel, match["image_index"]),
                                    {"channel": channel, "match": match, "rows": [], "weight": 0., "pull": 0.})
            node["rows"].append(number)
            node["weight"] += max(value, 1e-9)
            node["pull"] += i * max(value, 1e-9)
            edges.append((i, (channel, match["image_index"]), value, number, channel, row, match, rank))
    # barycentre order: each image node sits next to the caption parts that pull on it most, which limits crossings
    order = sorted(right, key=lambda key: (right[key]["pull"] / right[key]["weight"], key[0] != "objects"))
    step, top, x_left, x_right = 30, 24, 236, 372
    n = max(len(left), len(order))
    left_y = [top + (i + (n - len(left)) / 2) * step for i in range(len(left))]
    right_y = {key: top + (k + (n - len(order)) / 2) * step for k, key in enumerate(order)}
    vmax = max((abs(edge[2]) for edge in edges), default=0.) or 1.
    out = []
    for i, key, value, number, channel, row, match, rank in edges:
        y1, y2, middle = left_y[i], right_y[key], (x_left + x_right) / 2
        title = (f"{part_text(row.get('query_part', '?'))} → {part_text(match.get('part', '?'))}: w {row['w_i']:.2f} × "
                 f"a {match['a_ij']:.2f} × sim {match['sim_ij']:.3f} = {value:.3f}")
        out.append(f'<path class="mk" data-row="{number}" d="M{x_left},{y1:.1f} C{middle},{y1:.1f} {middle},{y2:.1f} '
                   f'{x_right},{y2:.1f}" fill="none" stroke="{COLORS[channel]}" stroke-opacity="{0.7 if rank == 0 else 0.45}" '
                   f'stroke-width="{0.8 + 7.2 * max(value, 0.) / vmax:.1f}" stroke-linecap="round">'
                   f'<title>{esc(title)}</title></path>')
    for (number, channel, row), y in zip(left, left_y):
        label = part_text(row.get("query_part", "?"))
        badge = x_left - 10  # the row number sits where the row's edges start
        out.append(f'<g class="mk" data-row="{number}"><title>{esc(label)} · w = {row["w_i"]:.2f}</title>'
                   f'<circle cx="{badge}" cy="{y:.1f}" r="9" fill="{COLORS[channel]}"/><text x="{badge}" y="{y:.1f}" '
                   f'font-size="11" font-weight="700" fill="#fff" text-anchor="middle" dominant-baseline="central">'
                   f'{number}</text><text x="{badge - 14}" y="{y:.1f}" font-size="12.5" text-anchor="end" '
                   f'dominant-baseline="central" fill="currentColor">{esc(shorten(label, 30))}</text></g>')
    for key in order:
        node, y = right[key], right_y[key]
        match = node["match"]
        crop = _crop(match, frame, uid, x_right + 6, y - 12) if frame else ""
        merged = f" (gộp {match['merged']})" if match.get("merged", 1) > 1 else ""
        label = part_text(match.get("part", "?")) + merged
        rows_attr = " ".join(str(r) for r in dict.fromkeys(node["rows"]))
        out.append(f'<g class="mk" data-rows="{rows_attr}"><title>{esc(label)} · trong top-3 của dòng '
                   f'{", ".join(rows_attr.split())}</title><rect x="{x_right - 2}" y="{y - 13:.1f}" width="3" '
                   f'height="26" fill="{COLORS[node["channel"]]}"/>{crop}<text x="{x_right + (36 if crop else 8)}" '
                   f'y="{y:.1f}" font-size="12.5" dominant-baseline="central" fill="currentColor">'
                   f'{esc(shorten(label, 30))}</text></g>')
    height = top + (n - 1) * step + 18
    return (f'<svg class="mx-graph" viewBox="0 0 600 {height:.0f}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="Đồ thị con khớp câu với ảnh"><text x="2" y="8" font-size="11" fill="currentColor" '
            f'opacity="0.65">Phần của câu</text><text x="{x_right}" y="8" font-size="11" fill="currentColor" '
            f'opacity="0.65">Phần của ảnh (top-3 của từng dòng)</text>' + "".join(out) + "</svg>")
