"""Synthetic scene-graph retrieval with known answers, for the level-3 explanation benchmark (GraphXAI style).

GraphXAI (Agarwal et al., Scientific Data 2023) plants a motif whose presence decides the label, so the motif is the
ground-truth explanation. Here the task is the reranker's: a caption is a small motif (two objects and one relation,
or a chain of three objects and two relations), its gold image is a random scene holding the motif, and every other
candidate is a scene that misses exactly one part of it:
- predicate: one relation of the motif has another predicate;
- object: one object of the motif has another name (its relations keep the new name), and no other object of the
  image has the caption's name;
- relation: one relation of the motif is missing.
Every object and every relation of the motif is broken in at least one negative of each caption, so the model needs
all of them to put the gold image first (a ground truth the model may ignore is the failure GraphXAI and Faber et al.,
KDD 2021, warn about). The gold image's answer is the motif's object nodes and labelled triples; the links from the
motif to the background, which every candidate has, are not part of it. Object order, relation order and boxes are
random, so a part's position says nothing. Scene sizes come from one distribution for every candidate. The CLIP
scores are one constant, so only the graph can rank.

Labels and their frozen CLIP vectors are real (common Visual Genome names and predicates); only the scenes are
synthetic. A model of the main architecture is trained on these captions (scripts/xai_ground_truth.py).
"""
import numpy as np

from src.data.graph_store import GraphStore
from src.graph.counts import image_bucket

OBJECTS = ("man", "woman", "boy", "girl", "dog", "cat", "horse", "bird", "car", "bus", "bike", "boat", "table",
           "chair", "bench", "bed", "couch", "cup", "plate", "bowl", "bottle", "pizza", "cake", "banana", "apple",
           "phone", "laptop", "book", "clock", "lamp", "umbrella", "bag", "hat", "shirt", "kite", "ball", "tree",
           "fence", "sign", "window")
PREDICATES = ("on", "holding", "wearing", "next to", "behind", "under", "riding", "in front of")
NEGATIVE_KINDS = ("predicate", "object", "relation")
WIDTH, HEIGHT = 640, 480


def motif(rng):
    """(names, relations as (subject index, predicate, object index)) of a caption's motif."""
    size = 2 if rng.random() < .5 else 3
    names = [str(x) for x in rng.choice(OBJECTS, size=size, replace=False)]
    relations = [(i, str(rng.choice(PREDICATES)), i + 1) for i in range(size - 1)]
    return names, relations


def caption_text(names, relations):
    s, p, o = relations[0]
    text = f"a {names[s]} {p} a {names[o]}"
    for s, p, o in relations[1:]:
        text += f" and the {names[s]} {p} a {names[o]}"
    return text


def scene(rng, names, relations, exclude=(), background=(5, 11), extra=(3, 8), links=(1, 2)):
    """An image graph: random background objects (none with a motif name or a name in exclude) and relations among
    them, the motif's objects and relations, and a few relations linking the motif to the background; returns the
    graph and the positions of the motif's objects and relations in it."""
    others = [o for o in OBJECTS if o not in names and o not in exclude]
    k = int(rng.integers(background[0], background[1] + 1))
    labels = [str(x) for x in rng.choice(others, size=k)] + list(names)
    order = rng.permutation(len(labels))
    position = {int(old): new for new, old in enumerate(order)}
    objects = []
    for new, old in enumerate(order):
        w, h = int(rng.integers(20, 200)), int(rng.integers(20, 200))
        x, y = int(rng.integers(0, WIDTH - w)), int(rng.integers(0, HEIGHT - h))
        objects.append({"id": new, "name": labels[old], "synset": None, "box": [x, y, w, h]})
    motif_nodes = [position[k + i] for i in range(len(names))]
    rels = [(position[k + s], p, position[k + o], True) for s, p, o in relations]
    for _ in range(int(rng.integers(extra[0], extra[1] + 1))):
        s, o = rng.choice(k, size=2, replace=False)
        rels.append((position[int(s)], str(rng.choice(PREDICATES)), position[int(o)], False))
    for _ in range(int(rng.integers(links[0], links[1] + 1))):
        s, o = int(rng.choice(motif_nodes)), position[int(rng.integers(k))]
        if rng.random() < .5:
            s, o = o, s
        rels.append((s, str(rng.choice(PREDICATES)), o, False))
    unique, seen = [], set()
    for r in (rels[i] for i in rng.permutation(len(rels))):
        if r[:3] not in seen and r[0] != r[2]:
            seen.add(r[:3])
            unique.append(r)
    counts = {}
    for obj in objects:
        counts[obj["name"]] = counts.get(obj["name"], 0) + 1
    for obj in objects:
        obj["count_bucket"] = image_bucket(counts[obj["name"]])
    graph = {"width": WIDTH, "height": HEIGHT, "objects": objects,
             "relations": [{"subject": s, "predicate": p, "object": o} for s, p, o, _ in unique]}
    return graph, motif_nodes, [i for i, r in enumerate(unique) if r[3]]


def broken(rng, names, relations, kind, which):
    """The motif with part which broken as kind (module docstring)."""
    names, relations = list(names), list(relations)
    if kind == "object":
        names[which] = str(rng.choice([o for o in OBJECTS if o not in names]))
    elif kind == "predicate":
        s, p, o = relations[which]
        relations[which] = (s, str(rng.choice([q for q in PREDICATES if q != p])), o)
    else:
        del relations[which]
    return names, relations


def make_dataset(seed, n_train, n_val, candidates=50):
    """Graphs, query graphs, captions, candidate arrays and answers of a synthetic split pair.

    Returns {"graphs": {image_id: graph}, "queries": {cid: graph}, "captions": {cid: text},
    "arrays": {split: candidate arrays as GraphStore keeps them}, "answers": {cid: {"image", "nodes", "triples"}}}.
    The gold image is at a random candidate position.
    """
    rng = np.random.default_rng(seed)
    graphs, queries, captions, answers, arrays = {}, {}, {}, {}, {}
    next_id = 0
    for split, count in (("train", n_train), ("val", n_val)):
        rows = {"caption_ids": [], "gold": [], "candidate_ids": [], "clip_scores": [], "gold_rank": [], "group_index": []}
        for q in range(count):
            cid = f"{split}_{q}"
            names, relations = motif(rng)
            queries[cid] = {"objects": [{"id": i, "name": n, "count_class": 0} for i, n in enumerate(names)],
                            "relations": [{"subject": s, "predicate": p, "object": o} for s, p, o in relations]}
            captions[cid] = caption_text(names, relations)
            gold, nodes, triples = scene(rng, names, relations)
            ids = [next_id]
            graphs[next_id] = gold
            answers[cid] = {"image": next_id, "nodes": nodes, "triples": triples}
            next_id += 1
            # every motif object and relation broken at least once, then kinds at random
            plan = [("object", i) for i in range(len(names))] + [("predicate", i) for i in range(len(relations))] + \
                   [("relation", i) for i in range(len(relations))]
            while len(plan) < candidates - 1:
                kind = str(rng.choice(NEGATIVE_KINDS))
                plan.append((kind, int(rng.integers(len(names) if kind == "object" else len(relations)))))
            for kind, which in plan:
                # the background never holds a caption name, so a renamed object is missing from the image
                graphs[next_id] = scene(rng, *broken(rng, names, relations, kind, which), exclude=names)[0]
                ids.append(next_id)
                next_id += 1
            order = rng.permutation(candidates)
            ids = [ids[i] for i in order]
            rows["caption_ids"].append(cid)
            rows["gold"].append(answers[cid]["image"])
            rows["candidate_ids"].append(ids)
            rows["clip_scores"].append(np.zeros(candidates, dtype=np.float32))
            rows["gold_rank"].append(ids.index(answers[cid]["image"]) + 1)
            rows["group_index"].append(0)
        arrays[split] = {k: np.array(v) for k, v in rows.items()}
        arrays[split]["clip_scores"] = arrays[split]["clip_scores"].astype(np.float32)
    return {"graphs": graphs, "queries": queries, "captions": captions, "arrays": arrays, "answers": answers}


def part_keys(data):
    """Every part key the synthetic graphs and captions use (node, relation and triple texts)."""
    keys = set()
    for graph in list(data["graphs"].values()) + list(data["queries"].values()):
        names = {o["id"]: o["name"] for o in graph["objects"]}
        keys.update(f"node:{n}" for n in names.values())
        for r in graph["relations"]:
            keys.add(f"rel:{r['predicate']}")
            keys.add(f"triple:{names[r['subject']]} {r['predicate']} {names[r['object']]}")
    return sorted(keys)


def synthetic_store(data, part_vectors, part_row, caption_vectors, knn_edges=5, global_node=True):
    """A GraphStore over the synthetic data, built in memory (no files)."""
    store = GraphStore.__new__(GraphStore)
    store.part_vectors, store.part_row = np.ascontiguousarray(part_vectors, dtype=np.float32), dict(part_row)
    store.part_dim = store.part_vectors.shape[1]
    store.graphs, store.queries, store.arrays = dict(data["graphs"]), dict(data["queries"]), data["arrays"]
    ids = sorted(data["captions"])
    store.caption_row = {cid: i for i, cid in enumerate(ids)}
    store.caption_vectors = np.ascontiguousarray(caption_vectors, dtype=np.float32)
    store.caption_location = {str(cid): (split, row) for split, arr in data["arrays"].items()
                              for row, cid in enumerate(arr["caption_ids"])}
    store.concept_vocab, store.concept_folder, store.scene_terms = None, None, False
    store.set_structure(knn_edges=knn_edges, global_node=global_node)
    return store
