"""Clean Visual Genome image graphs: merge duplicate boxes of one thing, then group same-type things by role (audit).

Visual Genome annotators often draw a new box for the same thing in every relation they write, so one horse becomes
three "horse" nodes that each carry one edge, and chains such as man -riding-> horse -on-> beach break apart.
- Same type: both object synsets map to person.n.01 when they are people (VG names one person by role: man, skier,
  player, boy...), otherwise one synset equals or is a WordNet hypernym of the other (animal ~ dog, not dog ~ cat).
  Objects without a synset fall back to equal canonical names.
- Step 1, merge: same type and IoU >= threshold, closed transitively (union-find). Same name with different boxes
  stays separate. Edges are re-pointed to the merged node; duplicates and self-loops dropped.
- Step 2, group (no IoU: different things rarely overlap): two same-type nodes are linked when, for one anchor node
  (the other end of a triple, after merging), they relate to it in the same way:
  (a) the same annotated predicate in the same direction, both annotated; or
  (b) the same box-geometry class relative to the anchor (left, right, above, below, overlap/inside, beside; front and
      behind need depth), where at least one of the two has an annotated edge with the anchor. (b) also groups
      things whose relation VG did not annotate.
  The "near" variant instead links same-type nodes of similar size that are close (area ratio <= 2, box gap <= half
  the mean box side). Linked components of two or more become one group node: count = members, box = union, edges =
  union of the members' edges. Components are closed transitively across anchors.
"""
from __future__ import annotations

from functools import lru_cache

from src.graph.geometry import iou

PERSON = "person.n.01"


@lru_cache(maxsize=None)
def _synset(name):
    from nltk.corpus import wordnet as wn
    try:
        return wn.synset(name)
    except Exception:  # noqa: BLE001 - malformed or retired VG synset names
        return None


@lru_cache(maxsize=None)
def ancestors(name) -> frozenset:
    synset = _synset(name)
    if synset is None:
        return frozenset()
    return frozenset(s.name() for s in synset.closure(lambda s: s.hypernyms()))


@lru_cache(maxsize=None)
def coarse_type(name) -> str:
    return PERSON if name == PERSON or PERSON in ancestors(name) else name


def same_type(a, b, name_a, name_b) -> bool:
    if a is None or b is None:
        return name_a == name_b
    if coarse_type(a) == coarse_type(b):
        return True
    return a in ancestors(b) or b in ancestors(a)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def groups(self):
        out = {}
        for x in range(len(self.parent)):
            out.setdefault(self.find(x), []).append(x)
        return list(out.values())


def union_box(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def collapse(graph, members, same_thing=False):
    """New graph with one node per member list; label of the largest box, edges re-pointed.

    same_thing=True (merging duplicates): the members are boxes of one thing, so count stays that of one member and
    "boxes" adds up the boxes absorbed. Otherwise (grouping): count = number of distinct things in the group.
    """
    nodes, old_to_new = [], {}
    for new, group in enumerate(members):
        largest = max(group, key=lambda i: graph["objects"][i]["box"][2] * graph["objects"][i]["box"][3])
        source = graph["objects"][largest]
        nodes.append({"id": new, "name": source["name"], "synset": source.get("synset"),
                      "box": union_box([graph["objects"][i]["box"] for i in group]),
                      "count": (max(graph["objects"][i].get("count", 1) for i in group) if same_thing
                                else sum(graph["objects"][i].get("count", 1) for i in group)),
                      "boxes": sum(graph["objects"][i].get("boxes", 1) for i in group),
                      "members": sorted(m for i in group for m in graph["objects"][i].get("members", [i]))})
        old_to_new.update({i: new for i in group})
    relations, seen = [], set()
    for k, rel in enumerate(graph["relations"]):
        s, o = old_to_new[rel["subject"]], old_to_new[rel["object"]]
        key = (s, rel["predicate"], o)
        if s != o and key not in seen:
            seen.add(key)
            # source: index of the packaged relation this edge comes from (its phrase vector stays usable)
            relations.append({"subject": s, "predicate": rel["predicate"], "object": o, "source": rel.get("source", k)})
    return {"objects": nodes, "relations": relations}, old_to_new


def with_boxes(graph, boxes, synsets):
    """The packaged graph with each object's VG box and synset attached (count 1, its own member)."""
    return {"objects": [{**o, "box": boxes[i], "synset": synsets[i], "count": 1, "boxes": 1, "members": [i]}
                        for i, o in enumerate(graph["objects"])], "relations": graph["relations"]}


def merge_duplicates(graph, threshold):
    objects = graph["objects"]
    uf = UnionFind(len(objects))
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if same_type(a["synset"], b["synset"], a["name"], b["name"]) and iou(a["box"], b["box"]) >= threshold:
                uf.union(i, j)
    return collapse(graph, uf.groups(), same_thing=True)


def near(a, b) -> bool:
    """Similar size (area ratio <= 2) and a box gap of at most half the mean box side."""
    area_a, area_b = max(a[2] * a[3], 1), max(b[2] * b[3], 1)
    if max(area_a, area_b) / min(area_a, area_b) > 2:
        return False
    gap_x = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    gap_y = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return max(gap_x, gap_y) <= .5 * (area_a ** .5 + area_b ** .5) / 2


def geometry_class(box, anchor) -> str:
    """Where box sits relative to anchor: overlap (intersection >= half the smaller box), beside (vertical overlap,
    horizontal gap <= the wider box's width), else left/right/above/below by the larger centre offset."""
    bx, by, bw, bh = box
    ax, ay, aw, ah = anchor
    ix = max(0., min(bx + bw, ax + aw) - max(bx, ax))
    iy = max(0., min(by + bh, ay + ah) - max(by, ay))
    if ix * iy >= .5 * max(min(bw * bh, aw * ah), 1):
        return "overlap"
    gap_x = max(0., max(bx, ax) - min(bx + bw, ax + aw))
    if iy > 0 and gap_x <= max(bw, aw):
        return "beside"
    dx, dy = (bx + bw / 2) - (ax + aw / 2), (by + bh / 2) - (ay + ah / 2)
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "above" if dy < 0 else "below"


def roles(graph):
    """Per node: annotated (direction, predicate, anchor) roles and the set of anchors it has an edge with."""
    out = [set() for _ in graph["objects"]]
    for rel in graph["relations"]:
        out[rel["subject"]].add(("out", rel["predicate"], rel["object"]))
        out[rel["object"]].add(("in", rel["predicate"], rel["subject"]))
    return out


def pair_relation(graph, role, i, j):
    """(a, b, conflict_a_not_b, conflict_b_not_a) for same-type nodes i, j.

    a: one shared annotated (direction, predicate, anchor). b: one anchor linked to i or j by an annotated edge with the
    same geometry class for both. Conflicts count anchors that both i and j are annotated with, where the two rules
    disagree.
    """
    boxes = [o["box"] for o in graph["objects"]]
    shared = {(d, p, a) for d, p, a in role[i] & role[j] if a not in (i, j)}
    anchors_i, anchors_j = {a for _, _, a in role[i]} - {i, j}, {a for _, _, a in role[j]} - {i, j}
    same_geometry = {a for a in anchors_i | anchors_j if geometry_class(boxes[i], boxes[a]) == geometry_class(boxes[j], boxes[a])}
    both = anchors_i & anchors_j
    by_predicate = {a for _, _, a in shared}
    return (bool(shared), bool(same_geometry), len((by_predicate - same_geometry) & both),
            len((both & same_geometry) - by_predicate))


def group_by_role(graph, criterion="ab"):
    """Group same-type nodes; criterion: a (annotated predicate), b (geometry class), ab (either), near (distance)."""
    objects, role = graph["objects"], roles(graph)
    uf = UnionFind(len(objects))
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if not same_type(a["synset"], b["synset"], a["name"], b["name"]):
                continue
            if criterion == "near":
                linked = near(a["box"], b["box"])
            else:
                by_a, by_b, _, _ = pair_relation(graph, role, i, j)
                linked = {"a": by_a, "b": by_b, "ab": by_a or by_b}[criterion]
            if linked:
                uf.union(i, j)
    return collapse(graph, uf.groups())


def role_pair_counts(graph) -> dict:
    """Same-type node pairs linked by (a), by (b), by both, and the pairs/anchors where (a) and (b) disagree."""
    objects, role = graph["objects"], roles(graph)
    counts = dict(pairs_a=0, pairs_b=0, pairs_both=0, conflict_a_not_b=0, conflict_b_not_a=0)
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if same_type(a["synset"], b["synset"], a["name"], b["name"]):
                by_a, by_b, a_not_b, b_not_a = pair_relation(graph, role, i, j)
                counts["pairs_a"] += by_a
                counts["pairs_b"] += by_b
                counts["pairs_both"] += by_a and by_b
                counts["conflict_a_not_b"] += a_not_b
                counts["conflict_b_not_a"] += b_not_a
    return counts


def graph_stats(graph) -> dict:
    """Nodes, connected components (isolated nodes included), degree-1 share, largest component, isolated share."""
    n = len(graph["objects"])
    uf, degree = UnionFind(n), [0] * n
    for rel in graph["relations"]:
        uf.union(rel["subject"], rel["object"])
        degree[rel["subject"]] += 1
        degree[rel["object"]] += 1
    components = uf.groups() if n else []
    return {"nodes": n, "edges": len(graph["relations"]), "components": len(components),
            "largest_component": max((len(c) for c in components), default=0),
            "degree_one": sum(d == 1 for d in degree), "isolated": sum(d == 0 for d in degree)}


def duplicate_nodes(graph, threshold) -> int:
    """Nodes that have a same-type partner with IoU >= threshold."""
    objects, flagged = graph["objects"], set()
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if same_type(a["synset"], b["synset"], a["name"], b["name"]) and iou(a["box"], b["box"]) >= threshold:
                flagged.update((i, j))
    return len(flagged)
