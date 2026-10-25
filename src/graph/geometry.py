"""Rule-based count and spatial checks between a caption and an image's Visual Genome boxes (no training).

Boxes are (x, y, w, h) in pixels, y pointing down. Every check returns a value in [0, 1], or None when the image
has nothing to check against (no matching instance): a missing label is the object channel's evidence, not this one's.
- counts: explicit numbers in the caption ("two dogs", "3 cats", "a pair of skis") against the number of matching
  instances in the image after merging duplicate boxes of one label (IoU >= 0.5): 1 - min(|n - c| / c, 1).
- spatial: caption triples whose predicate is left of / right of / above / below / under / beneath / underneath /
  on top of / next to / beside, checked on every (subject instance, object instance) pair: the share of pairs that
  satisfy the predicate (centres for left/right/above/below; on top of also needs horizontal overlap; next to/beside
  needs vertical overlap and a horizontal gap of at most the wider box's width). "in front of" and "behind" need
  depth and are not checked.
"""
from __future__ import annotations

import re

NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
SPATIAL = {"left of": "left", "to the left of": "left", "on the left of": "left", "right of": "right",
           "to the right of": "right", "on the right of": "right", "above": "above", "below": "below",
           "under": "below", "underneath": "below", "beneath": "below", "on top of": "on_top",
           "next to": "beside", "beside": "beside"}


def spatial_kind(predicate: str) -> str | None:
    return SPATIAL.get(re.sub(r"\s+", " ", predicate.lower().strip()))


def caption_counts(doc) -> list[tuple[int, str]]:
    """(count, noun lemma) for explicit numbers 2-10 in a spaCy doc: number words, digits, "a pair of"."""
    out = []
    for token in doc:
        value = NUMBER_WORDS.get(token.lower_)
        if value is None and token.text.isdigit() and 2 <= int(token.text) <= 10:
            value = int(token.text)
        if value is not None and token.dep_ == "nummod" and token.head.pos_ in ("NOUN", "PROPN"):
            out.append((value, token.head.lemma_.lower()))
        if token.lower_ == "pair" and token.i + 2 < len(doc) and doc[token.i + 1].lower_ == "of":
            noun = next((t for t in doc[token.i + 2:] if t.pos_ in ("NOUN", "PROPN")), None)
            if noun is not None:
                out.append((2, noun.lemma_.lower()))
    return list(dict.fromkeys(out))


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0., min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0., min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.


def merge_duplicates(boxes, threshold=0.5) -> list:
    """Greedy: keep a box unless it overlaps an already kept box with IoU >= threshold (VG repeats objects)."""
    kept = []
    for box in sorted(boxes, key=lambda b: -b[2] * b[3]):
        if all(iou(box, k) < threshold for k in kept):
            kept.append(box)
    return kept


def count_score(stated: int, boxes) -> float | None:
    if not boxes:
        return None
    found = len(merge_duplicates(boxes))
    return 1. - min(abs(found - stated) / stated, 1.)


def satisfies(kind: str, s, o) -> bool:
    sx, sy, sw, sh = s
    ox, oy, ow, oh = o
    scx, scy, ocx, ocy = sx + sw / 2, sy + sh / 2, ox + ow / 2, oy + oh / 2
    x_overlap = min(sx + sw, ox + ow) - max(sx, ox)
    y_overlap = min(sy + sh, oy + oh) - max(sy, oy)
    if kind == "left":
        return scx < ocx
    if kind == "right":
        return scx > ocx
    if kind == "above":
        return scy < ocy
    if kind == "below":
        return scy > ocy
    if kind == "on_top":
        return scy < ocy and x_overlap > 0
    if kind == "beside":
        return y_overlap > 0 and -x_overlap <= max(sw, ow)
    raise ValueError(kind)


def spatial_score(kind: str, subject_boxes, object_boxes) -> float | None:
    pairs = [(s, o) for s in subject_boxes for o in object_boxes if s != o]
    if not pairs:
        return None
    return sum(satisfies(kind, s, o) for s, o in pairs) / len(pairs)
