"""Image-graph structure for the full-graph experiment (experiment 3): box geometry on edges and completion edges.

Boxes are (x, y, w, h) in pixels, y pointing down; widths and heights below 1 px count as 1 px.
- edge_geometry(s, o): six numbers of the target box o relative to the source box s of an edge s -> o (the subject
  and object of a labelled triple; the message direction of an added edge): (cx_o - cx_s) / w_s and
  (cy_o - cy_s) / h_s, clipped to [-10, 10]; log(w_o / w_s); log(h_o / h_s); IoU; and the area share
  area_o / (area_s + area_o), the area ratio in a bounded form.
- completion_edges(...): every node receives an edge from each of its k nearest nodes (centre distance divided by
  the image diagonal; ties by node index). If the graph (labelled and k-nearest edges, direction ignored) still has
  several components, the closest pair of nodes in two different components is linked in both directions, again
  and again until one component is left (Kruskal on the component graph). Directed pairs that already carry a
  labelled edge get no added edge.
"""
from __future__ import annotations

import math

import numpy as np

OFFSET_CLIP = 10.
GEOMETRY_FEATURES = ("dx_over_w_s", "dy_over_h_s", "log_w_ratio", "log_h_ratio", "iou", "area_share")


def _sides(box):
    return max(float(box[2]), 1.), max(float(box[3]), 1.)


def box_iou(a, b) -> float:
    aw, ah = _sides(a)
    bw, bh = _sides(b)
    ix = max(0., min(a[0] + aw, b[0] + bw) - max(a[0], b[0]))
    iy = max(0., min(a[1] + ah, b[1] + bh) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def edge_geometry(s, o) -> list[float]:
    sw, sh = _sides(s)
    ow, oh = _sides(o)
    dx = (o[0] + ow / 2 - s[0] - sw / 2) / sw
    dy = (o[1] + oh / 2 - s[1] - sh / 2) / sh
    clip = lambda v: min(max(v, -OFFSET_CLIP), OFFSET_CLIP)  # noqa: E731
    return [clip(dx), clip(dy), math.log(ow / sw), math.log(oh / sh), box_iou(s, o), ow * oh / (sw * sh + ow * oh)]


def centre_distances(boxes, width, height) -> np.ndarray:
    """Pairwise centre distances divided by the image diagonal, shape (n, n)."""
    sides = np.array([_sides(b) for b in boxes], dtype=np.float64).reshape(-1, 2)
    corner = np.array([b[:2] for b in boxes], dtype=np.float64).reshape(-1, 2)
    centres = corner + sides / 2
    diagonal = math.hypot(max(width, 1), max(height, 1))
    return np.linalg.norm(centres[:, None] - centres[None], axis=-1) / diagonal


def components(n, edges) -> np.ndarray:
    """Component label (smallest member) of every node; edge direction ignored."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(x) for x in range(n)], dtype=np.int64)


def completion_edges(boxes, width, height, k, labelled=()):
    """(k-nearest edges, bridge edges) as lists of directed (source, target) node pairs."""
    n = len(boxes)
    if n < 2 or k <= 0:
        return [], []
    distance = centre_distances(boxes, width, height)
    taken = {(int(s), int(t)) for s, t in labelled}
    knn = []
    for target in range(n):
        row = distance[target].copy()
        row[target] = np.inf
        for source in np.argsort(row, kind="stable")[:min(k, n - 1)]:
            pair = (int(source), target)
            if pair not in taken:
                taken.add(pair)
                knn.append(pair)
    label = components(n, list(taken))
    bridges = []
    if len(set(label.tolist())) > 1:
        parent = {int(c): int(c) for c in set(label.tolist())}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        rows, cols = np.triu_indices(n, 1)
        order = np.argsort(distance[rows, cols], kind="stable")  # pairs are in (i, j) order, so ties keep it
        left = len(parent) - 1
        for p in order:
            i, j = int(rows[p]), int(cols[p])
            a, b = find(int(label[i])), find(int(label[j]))
            if a == b:
                continue
            parent[max(a, b)] = min(a, b)
            for pair in ((i, j), (j, i)):
                if pair not in taken:
                    taken.add(pair)
                    bridges.append(pair)
            left -= 1
            if not left:
                break
    return knn, bridges


def structure_stats(n, edges) -> dict:
    """Components (isolated nodes included) and the share of nodes in the largest component of one graph."""
    if not n:
        return {"nodes": 0, "components": 0, "largest_share": None}
    _, sizes = np.unique(components(n, edges), return_counts=True)
    return {"nodes": n, "components": int(len(sizes)), "largest_share": float(sizes.max() / n)}
