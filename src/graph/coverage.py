"""How much of a parsed caption can be found in an image's scene graph. Pure functions on plain dicts.

Three levels, from loose to strict:
  entity  the caption's object label is one of the image's object labels
  pair    subject and object of a caption relation are connected in the image graph (same or either direction)
  triple  (subject, predicate, object) is a relation of the image graph
"""
from collections import Counter
from statistics import mean, median

from src.graph.scene_graph import Vocab, label_pairs, label_triples


def image_index(graph: dict) -> dict:
    """Label-level view of one image graph."""
    pairs = label_pairs(graph)
    return {"labels": {o["name"] for o in graph["objects"]}, "pairs": pairs,
            "either_pairs": pairs | {(b, a) for a, b in pairs}, "triples": label_triples(graph)}


def canon_query(query: dict, vocab: Vocab) -> dict:
    """Canonical labels of a parsed caption. A relation keeps both predicate candidates: raw form, then lemma."""
    names = [vocab.canon_object(o["name"]) for o in query["objects"]]
    relations = [(names[r["subject"]], names[r["object"]],
                  [vocab.canon_predicate(r["predicate"]), vocab.canon_predicate(r.get("predicate_lemma", r["predicate"]))])
                 for r in query["relations"]]
    return {"entities": names, "relations": relations}


def coverage(cq: dict, index: dict) -> dict:
    """Per-entity and per-relation hit flags of a canonical query against one image index."""
    return {"entity": [e in index["labels"] for e in cq["entities"]],
            "pair": [(s, o) in index["pairs"] for s, o, _ in cq["relations"]],
            "either_pair": [(s, o) in index["either_pairs"] for s, o, _ in cq["relations"]],
            "triple": [any((s, p, o) in index["triples"] for p in preds) for s, o, preds in cq["relations"]]}


def _share(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def summarize(results: list[dict]) -> dict:
    """Aggregate coverage() outputs over captions: item-level shares and caption-level 'at least one hit' shares."""
    out = {"captions": len(results)}
    for key in ("entity", "pair", "either_pair", "triple"):
        flat = [h for r in results for h in r[key]]
        out[f"{key}_items"] = len(flat)
        out[f"{key}_item_share"] = _share(sum(flat), len(flat))
        out[f"captions_with_{key}_hit"] = _share(sum(any(r[key]) for r in results), len(results))
    out["captions_with_all_entities_hit"] = _share(sum(bool(r["entity"]) and all(r["entity"]) for r in results), len(results))
    return out


def graph_stats(graphs: list[dict]) -> dict:
    n_obj = [len(g["objects"]) for g in graphs]
    n_rel = [len(g["relations"]) for g in graphs]
    n_tri = [len(label_triples(g)) for g in graphs]
    stats = {"images": len(graphs)}
    for name, values in (("objects", n_obj), ("relations", n_rel), ("unique_label_triples", n_tri)):
        stats[f"{name}_mean"], stats[f"{name}_median"] = round(mean(values), 2), median(values)
    stats["share_images_without_objects"] = _share(sum(v == 0 for v in n_obj), len(graphs))
    stats["share_images_without_relations"] = _share(sum(v == 0 for v in n_rel), len(graphs))
    return stats


def parser_stats(queries: list[dict]) -> dict:
    n_ent = [len(q["objects"]) for q in queries]
    n_rel = [len(q["relations"]) for q in queries]
    return {"captions": len(queries), "share_with_entity": _share(sum(v > 0 for v in n_ent), len(queries)),
            "share_with_relation": _share(sum(v > 0 for v in n_rel), len(queries)),
            "entities_mean": round(mean(n_ent), 2), "relations_mean": round(mean(n_rel), 2)}


def pool_selectivity(canon_queries: list[dict], golds: list[int], indexes: dict[int, dict]) -> dict:
    """For captions whose gold image is hit, the share of the whole pool that is hit as well (lower = more selective).

    A signal that fires on the gold image but also on half of the pool cannot re-rank anything.
    """
    by_label, by_pair, by_triple = {}, {}, {}
    for image_id, index in indexes.items():
        for label in index["labels"]:
            by_label.setdefault(label, set()).add(image_id)
        for pair in index["either_pairs"]:
            by_pair.setdefault(pair, set()).add(image_id)
        for triple in index["triples"]:
            by_triple.setdefault(triple, set()).add(image_id)

    shares = {"any_entity": [], "all_entities": [], "any_pair": [], "any_triple": []}
    for cq, gold in zip(canon_queries, golds):
        label_sets = [by_label.get(e, set()) for e in set(cq["entities"])]
        matched = {"any_entity": set().union(*label_sets),
                   "all_entities": set.intersection(*label_sets) if label_sets else set(),
                   "any_pair": set().union(*(by_pair.get((s, o), set()) for s, o, _ in cq["relations"])),
                   "any_triple": set().union(*(by_triple.get((s, p, o), set())
                                               for s, o, preds in cq["relations"] for p in preds))}
        for key, images in matched.items():
            if gold in images:
                shares[key].append(len(images) / len(indexes))
    return {key: {"captions_with_gold_hit": len(v), "pool_share_mean": round(mean(v), 4) if v else None,
                  "pool_share_median": round(median(v), 4) if v else None} for key, v in shares.items()}


def top_misses(queries: list[dict], canon_queries: list[dict], results: list[dict], n: int = 30) -> dict:
    entities, predicates = Counter(), Counter()
    for q, cq, r in zip(queries, canon_queries, results):
        entities.update(e for e, hit in zip(cq["entities"], r["entity"]) if not hit)
        predicates.update(rel["predicate"].lower() for rel, hit in zip(q["relations"], r["triple"]) if not hit)
    return {"entities_not_in_gold_image": entities.most_common(n), "predicates_without_exact_match": predicates.most_common(n)}


def frequent_entities(canon_queries: list[dict], results: list[dict], n: int = 30) -> list[dict]:
    """The n most frequent caption entity labels with the share of their mentions found in the gold image."""
    total, hits = Counter(), Counter()
    for cq, r in zip(canon_queries, results):
        total.update(cq["entities"])
        hits.update(e for e, hit in zip(cq["entities"], r["entity"]) if hit)
    return [{"label": label, "mentions": count, "hit_share": _share(hits[label], count)} for label, count in total.most_common(n)]
