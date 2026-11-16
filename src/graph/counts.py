"""Count classes of caption nodes and count buckets of image nodes (experiment 4: group nodes and counts).

Image node (graph after merging duplicates and grouping): a group node counts its members; any other node counts the
instances of its type in the image (itself included; group members count one each), with "type" as in
src/graph/dedup.py same_type. Buckets: 1 / 2 / 3-4 / >=5.

Caption node (FACTUAL graph, matched to its noun in the spaCy parse of the caption), first rule that applies:
1. an explicit number 2-10 on the noun (nummod, words or digits), or "a pair of X" -> that number;
2. "a couple of X" or "both X" -> 2;
3. "lone X" or "single X" -> 1;
4. "a group / herd / crowd / lot of X", "lots of X", "several X", "many X", a FACTUAL attribute holding one of these
   words, or a node that is itself a group / herd / crowd -> >=3;
5. a plural noun (tag NNS or NNPS) that is not a plural-only word (pants, glasses...) -> >=2;
6. anything else, "a X" included -> unknown.
When the noun is not found in the parse, the FACTUAL count (2-10) is used if any, else the attributes (rule 4).
Explicit numbers take the image buckets: 2 -> "2", 3-4 -> "3-4", 5-10 -> ">=5".
"""
from __future__ import annotations

IMAGE_BUCKETS = ("1", "2", "3-4", ">=5")
CAPTION_CLASSES = ("unknown", "1", "2", "3-4", ">=5", ">=2", ">=3")
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
GROUP_WORDS = {"group", "herd", "crowd", "lot"}
MANY_WORDS = {"several", "many"}
PLURAL_ONLY = {"pants", "jeans", "shorts", "trousers", "glasses", "sunglasses", "goggles", "scissors", "tongs", "pliers",
               "clothes", "stairs", "headphones", "binoculars", "overalls", "tights", "leggings", "pajamas", "suspenders"}


def image_bucket(count: int) -> int:
    return 0 if count <= 1 else 1 if count == 2 else 2 if count <= 4 else 3


def number_class(value: int) -> int:
    return CAPTION_CLASSES.index("1" if value == 1 else "2" if value == 2 else "3-4" if value <= 4 else ">=5")


def compatible(caption_class: int, bucket: int) -> bool | None:
    """Does an image bucket satisfy a caption class? None for unknown."""
    name = CAPTION_CLASSES[caption_class]
    if name == "unknown":
        return None
    if name == ">=2":
        return bucket >= 1
    if name == ">=3":
        return bucket >= 2
    return IMAGE_BUCKETS[bucket] == name


def image_counts(objects, same_type) -> list[int]:
    """Count of every node: members for a group node (count > 1 from grouping), else same-type instances."""
    out = []
    for i, obj in enumerate(objects):
        if obj.get("group"):
            out.append(int(obj["count"]))
            continue
        out.append(sum(int(o.get("count", 1)) for o in objects
                       if o is obj or same_type(obj.get("synset"), o.get("synset"), obj["name"], o["name"])))
    return out


def _number(token):
    if token.lower_ in NUMBER_WORDS:
        return NUMBER_WORDS[token.lower_]
    if token.text.isdigit() and 2 <= int(token.text) <= 10:
        return int(token.text)
    return None


def _of_head(token):
    """The noun X in "X of <token>" (a group of men -> group), or None."""
    head = token.head
    if head is not token and head.lower_ == "of" and token.dep_ == "pobj" and head.head is not head:
        return head.head
    return None


def token_class(token, attributes=()) -> tuple[int, str]:
    for child in token.children:
        if child.dep_ == "nummod" and _number(child) is not None:
            return number_class(_number(child)), "number"
    of = _of_head(token)
    if of is not None and of.lemma_.lower() == "pair":
        return CAPTION_CLASSES.index("2"), "pair"
    if (of is not None and of.lemma_.lower() == "couple") or any(c.lower_ == "both" for c in token.children):
        return CAPTION_CLASSES.index("2"), "couple_both"
    if any(c.dep_ == "amod" and c.lower_ in ("lone", "single") for c in token.children):
        return CAPTION_CLASSES.index("1"), "lone_single"
    if (of is not None and of.lemma_.lower() in GROUP_WORDS) or any(c.lower_ in MANY_WORDS for c in token.children) \
            or token.lemma_.lower() in GROUP_WORDS - {"lot"} or _group_attribute(attributes):
        return CAPTION_CLASSES.index(">=3"), "group_many"
    if token.tag_ in ("NNS", "NNPS") and token.lower_ not in PLURAL_ONLY:
        return CAPTION_CLASSES.index(">=2"), "plural"
    return 0, "unknown"


def _group_attribute(attributes) -> bool:
    words = {w for a in attributes for w in str(a).lower().replace(",", " ").split()}
    return bool(words & (GROUP_WORDS | MANY_WORDS | {"lots"}))


def caption_classes(doc, objects, heads=None) -> list[tuple[int, str]]:
    """(class index, rule) per FACTUAL object; heads are the raw FACTUAL entity heads in object order, if known."""
    out = []
    for k, obj in enumerate(objects):
        name = obj["name"].lower().split()[-1] if obj["name"].strip() else ""
        head = heads[k].lower().split()[-1] if heads and k < len(heads) and heads[k].strip() else None
        nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
        token = next((t for t in nouns if head and t.lower_ == head), None) or \
            next((t for t in nouns if name and t.lemma_.lower() == name), None)
        if token is not None:
            out.append(token_class(token, obj.get("attributes", ())))
            continue
        count = obj.get("count")
        if isinstance(count, int) and 2 <= count <= 10:
            out.append((number_class(count), "factual_count"))
        elif _group_attribute(obj.get("attributes", ())) or name in GROUP_WORDS - {"lot"}:
            out.append((CAPTION_CLASSES.index(">=3"), "group_many"))
        else:
            out.append((0, "unknown" if name else "empty"))
    return out
