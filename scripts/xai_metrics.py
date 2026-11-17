"""Level-3 explanation metrics on val, version 2: fidelity, validity, sparsity, subgraphs and other explainers.

Version 2 of scripts/xai_metrics_v1.py (PR #36). Same models, captions, edits and random draws; it fixes what an
audit of version 1 found and adds a comparison with other explainers. Results go to experiments/xai-metrics-v2/;
version 1 and its results stay as they were.

Models: countonly_factual_r3 (count_v1) and main_r3 (level-1 data), three seeds each, checkpoint_best. For every val
caption (or a fixed sample, rng 2026, with --sample) the image the model ranks first is explained.

Explanation E: the image's parts (object nodes and labelled triples) ranked by their exact term (matched_pairs
image_terms) times QueryContext.fusion_scales, which puts both channels on the final score's scale. Under the raw
fusion (countonly) the factor is 1 and E is the v1 ranking. Under the z-score fusion (main_r3) v1 compared raw terms
of the two channels, which the fusion weighs by channel weight / spread over the 50 candidates; v2 uses that slope.

Changes from version 1:
- Base: deltas and ranks are taken against the unedited image rescored the same way (noop), not the forward pass;
  its rank must be 1 (captions where it is not are counted and left out). Captions whose image has no part (E
  empty) are reported apart and left out of the fidelity means, where v1 counted them as "no change" for both E and
  random.
- Sparsity is measured on E itself, in image parts: the fewest top parts of E whose scaled terms reach 80% / 90% of
  the positive term mass, and the smallest k in 1..K whose keep-only edit keeps >= 80% / 90% of the graph effect
  (base - empty). v1 counted caption rows of the demo table and divided them by image parts, two different units;
  that measure is kept as "caption_rows", without the share of image parts.
- Semantic validity, strict: the image node's synset equals the caption word's first noun sense or is a hyponym of
  it; people match only for a generic caption word (person, people, ...). v1's matcher (dedup.same_type: any two
  people, hypernym in either direction, so "woman" ~ "man" and "object" ~ "table") is kept as "lenient", with the
  alias and CLIP-cosine criteria. Triple rows of E are now checked too: subject and object (strict) and predicate
  (VG alias).
- Count compatibility only for models with a count table. v1 also scored main_r3 on undeduplicated level-1 graphs,
  whose duplicate boxes inflate counts, so the two models were not compared on the same counts.
- Subgraph: dropping the top-alpha incoming edges now has two forms. "messages" (main): the edges stop carrying GAT
  messages but keep their triples (message_drop). "edges": v1's form, which also removes labelled edges' triples.
  The random draw takes, for each node, as many of its own incoming edges as were chosen for it (v1 drew from the
  pooled edges of the three nodes).
- Counterfactual: the smallest k whose removal costs the image rank 1 (k_flip, from the deletion curve), and, for a
  wrong rank-1 image, the gold image's rank after the top-k of E are removed.

Comparison of explainers (fixed sample of --compare captions, rng 2026, inside the val captions; only images with a
positive graph effect, base - empty > 1e-4, since the curves are shares of it): every explainer
orders the same image parts and is measured the same way, by deletion and insertion curves for k = 0..K (K = 10):
- model: E above;
- random: a uniform permutation (rng per caption);
- lowest: E reversed;
- clip: post hoc, frozen CLIP cosine of the caption vector with each part's vector (no model);
- gradient: gradient x input of the final score with respect to each part's input vectors;
- gnnexplainer (on the first --gnn captions of the sample): a soft mask over the parts' input vectors, learned to keep
  the final score with a size and an entropy penalty (Ying et al., NeurIPS 2019, on features), parts ordered by mask.
Per explainer: Fid+@k / Fid-@k, rank, lost rank 1, k_flip, and the normalized areas under the curves
(deletion: mean over k = 1..K of the share of the graph effect still there, lower is better; insertion: mean share
kept when only the first k parts stay, higher is better), with a paired Wilcoxon test against "model".

Writes experiments/xai-metrics-v2/{metrics.json, cases.json}.
"""
import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.graph_store import EDGE_COMPLETION, EDGE_GLOBAL, EDGE_LABELLED, GraphStore  # noqa: E402
from src.explain.score_breakdown import top50_deltas  # noqa: E402
from src.explain.xai_metrics import (QueryContext, edit, image_index, keep_only, labelled_count,  # noqa: E402
                                     message_drop, rank_of, real_nodes)
from src.graph.counts import CAPTION_CLASSES, compatible  # noqa: E402
from src.graph.dedup import PERSON, ancestors, coarse_type, same_type  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-metrics-v2"
MODELS = {"countonly_factual_r3": (REPO_ROOT / "experiments" / "improve" / "count-only", "configs/count_v1.yaml"),
          "main_r3": (REPO_ROOT / "experiments" / "level1" / "gat", None)}
KS, RANDOM, THRESHOLDS, SUB_NODES, SUB_EDGES, COSINE, CURVE = (1, 3, 5), 10, (0.8, 0.9), 3, 3, .955, 10
EXPLAINERS = ("model", "random", "lowest", "clip", "gradient", "gnnexplainer")
GENERIC_PEOPLE = {"person", "people", "persons", "someone", "somebody", "human", "humans", "individual", "crowd"}
KIND_NAMES = {EDGE_LABELLED: "labelled", EDGE_COMPLETION: "completion", EDGE_GLOBAL: "global"}
CASES = REPO_ROOT / "experiments" / "improve" / "explanations" / "cases.json"
MIN_FREE_GB = 10


def first_sense(name):
    from nltk.corpus import wordnet as wn
    for word in (name.replace(" ", "_"), name.split()[-1] if name.split() else name):
        senses = wn.synsets(word, pos="n")
        if senses:
            return senses[0].name()
    return None


def strict_wordnet(name, sense, synset):
    """Image synset equals the caption sense or is a hyponym of it; people only for a generic caption word."""
    if not sense or not synset:
        return False
    if synset == sense or sense in ancestors(synset):
        return True
    return name in GENERIC_PEOPLE and coarse_type(synset) == PERSON


class LabelMatcher:
    """Does an image node carry a caption node's label? Strict (alias or WordNet hyponym) and v1's lenient test."""

    def __init__(self, vectors, rows, vocab):
        self.vectors, self.rows, self.vocab, self.cache = vectors, rows, vocab, {}

    def __call__(self, query, image, synset):
        key = (query, image, synset)
        if key not in self.cache:
            alias = self.vocab.canon_object(query) == self.vocab.canon_object(image)
            sense = first_sense(query)
            strict = strict_wordnet(query, sense, synset)
            lenient = bool(sense and synset) and same_type(sense, synset, query, image)
            known = f"node:{query}" in self.rows and f"node:{image}" in self.rows
            cosine = known and float(self.vectors[self.rows[f"node:{query}"]] @ self.vectors[self.rows[f"node:{image}"]]) >= COSINE
            head = self.vocab.canon_object(query.split()[-1] if query.split() else query) == \
                self.vocab.canon_object(image.split()[-1] if image.split() else image)
            self.cache[key] = {"alias": alias, "wordnet_strict": strict, "strict": alias or strict,
                               "wordnet_lenient": lenient, "cosine": cosine, "head": head,
                               "lenient": alias or lenient or cosine}
        return self.cache[key]


def ranked_parts(pairs, index, scales):
    """E: [(kind, position, scaled term)] by decreasing term; parts without a term get 0."""
    nodes, triples = np.zeros(real_nodes(index)), np.zeros(labelled_count(index))
    for j, t in enumerate(pairs["image_terms"]["objects"]):
        nodes[j] = t["term"] * scales["objects"]
    for j, t in enumerate(pairs["image_terms"]["triples"]):
        triples[j] = t["term"] * scales["triples"]
    parts = [("node", j, float(t)) for j, t in enumerate(nodes)] + [("triple", j, float(t)) for j, t in enumerate(triples)]
    return sorted(parts, key=lambda p: -p[2]), nodes, triples


def split(parts):
    return [p[1] for p in parts if p[0] == "node"], [p[1] for p in parts if p[0] == "triple"]


def random_parts(rng, index, n_nodes, n_triples):
    nodes = rng.choice(real_nodes(index), size=n_nodes, replace=False).tolist() if n_nodes else []
    triples = rng.choice(labelled_count(index), size=n_triples, replace=False).tolist() if n_triples else []
    return nodes, triples


def caption_rows(deltas, column, with_clip):
    """v1's sparsity: rows of the demo's "So với TB top-50" column (one per caption part) for 80% / 90% of the margin."""
    values = [("graph", float(v)) for key in ("objects", "triples") for v in deltas[key][column]]
    if with_clip:
        values.append(("clip", float(deltas["clip"][column])))
    margin = sum(v for _, v in values)
    if margin <= 0:
        return None
    ranked = sorted(values, key=lambda kv: -kv[1])
    cumulative = np.cumsum([v for _, v in ranked])
    out = {"margin": margin, "rows": len(values)}
    for threshold in THRESHOLDS:
        n = int(np.argmax(cumulative >= threshold * margin - 1e-12)) + 1
        out[f"n{int(100 * threshold)}"] = n
        out[f"share_of_rows{int(100 * threshold)}"] = n / len(values)
    return out


def term_sparsity(ranked):
    """Fewest top parts of E whose positive scaled terms reach 80% / 90% of all positive term mass."""
    positive = np.array([max(t, 0.) for _, _, t in ranked])
    total = positive.sum()
    if total <= 1e-12:
        return None
    cumulative = np.cumsum(positive)
    out = {"parts": len(ranked)}
    for threshold in THRESHOLDS:
        n = int(np.argmax(cumulative >= threshold * total - 1e-12)) + 1
        out[f"n{int(100 * threshold)}"] = n
        out[f"share{int(100 * threshold)}"] = n / len(ranked)
    return out


def part_name(graph, parts, kind, j):
    if kind == "node":
        return graph["objects"][parts["node_indices"][j]]["name"]
    names = {o["id"]: o["name"] for o in graph["objects"]}
    rel = graph["relations"][parts["relation_indices"][j]]
    return f"{names[rel['subject']]} {rel['predicate']} {names[rel['object']]}"


def curves(ctx, column, order, index, base, empty):
    """Deletion and insertion of the first k parts of an order, k = 0..K: score, rank, share of the graph effect."""
    sizes = list(range(min(CURVE, len(order)) + 1))
    edits = [edit(index, drop_nodes=split(order[:n])[0], drop_edges=split(order[:n])[1]) for n in sizes]
    edits += [keep_only(index, *split(order[:n])) for n in sizes]
    scores, ranks = ctx.rescore(column, edits)
    effect = base - empty
    count = len(sizes)
    out = {"sizes": sizes, "deletion": scores[:count].tolist(), "deletion_rank": ranks[:count].tolist(),
           "insertion": scores[count:].tolist(), "insertion_rank": ranks[count:].tolist()}
    flips = [n for n, r in zip(sizes, ranks[:count]) if r > 1]
    out["k_flip"] = flips[0] if flips else None
    if effect > 1e-4 and count > 1:
        out["deletion_auc"] = float(np.mean([(s - empty) / effect for s in scores[1:count]]))
        out["insertion_auc"] = float(np.mean([(s - empty) / effect for s in scores[count + 1:]]))
    return out


def gnnexplainer_order(ctx, column, index, base, empty, steps=100, size=2., entropy=0.1, lr=0.1):
    """Parts ordered by a soft mask over their input vectors, learned to keep the target's final score.

    A part with mask m takes m * its vector + (1 - m) * the mean vector of its kind in this image (a neutral part):
    the model normalizes node vectors, so scaling alone would barely change a score. The loss is
    -(final - empty) / (base - empty) + size * mean(mask) + entropy * mean binary entropy(mask), as GNNExplainer's
    feature mask, on the scale of the image's graph effect so that one penalty fits both fusions. Called only when
    base - empty > 1e-4 (measure), so that scale is positive. Gradients are taken for the mask alone.
    """
    images = index.to_batch(ctx.store.part_table(ctx.device), ctx.device)
    n, m = real_nodes(index), labelled_count(index)
    logits = torch.full((n + m,), 3., device=ctx.device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    x0, t0, e0 = images.x.detach(), images.triple_f.detach(), images.edge_attr.detach()
    x_mean = x0[:n].mean(0, keepdim=True) if n else x0[:0]
    t_mean, e_mean = (t0[:m].mean(0, keepdim=True), e0[:m].mean(0, keepdim=True)) if m else (t0, e0)
    effect = base - empty
    assert effect > 1e-4, effect
    for _ in range(steps):
        mask = torch.sigmoid(logits)
        node_mask = torch.cat((mask[:n], torch.ones(len(x0) - n, device=ctx.device)))[:, None]
        neutral = torch.cat((x_mean.expand(n, -1), x0[n:]))
        images.x = node_mask * x0 + (1 - node_mask) * neutral
        if m:
            images.triple_f = mask[n:, None] * t0[:m] + (1 - mask[n:, None]) * t_mean
            images.edge_attr = mask[n:, None] * e0[:m] + (1 - mask[n:, None]) * e_mean
        with torch.enable_grad():
            fused, _ = ctx.score_images(column, images, 1)
            bits = mask.clamp(1e-6, 1 - 1e-6)
            loss = (-(fused[0, column] - empty) / effect + size * mask.mean()
                    - entropy * (bits * bits.log() + (1 - bits) * (1 - bits).log()).mean())
            logits.grad, = torch.autograd.grad(loss, [logits])
        optimizer.step()
    weights = torch.sigmoid(logits).detach().cpu().numpy()
    parts = [("node", j, float(weights[j])) for j in range(n)] + [("triple", j, float(weights[n + j])) for j in range(m)]
    return sorted(parts, key=lambda p: -p[2])


def explainer_orders(ctx, column, index, ranked, rng, gnn, base, empty):
    """Every explainer's order of the same image parts."""
    store = ctx.store
    every = [(kind, j) for kind, j, _ in ranked]
    orders = {"model": [p[:2] for p in ranked], "lowest": [p[:2] for p in ranked[::-1]]}
    orders["random"] = [every[i] for i in rng.permutation(len(every))]
    caption = store.caption_vectors[store.caption_row[ctx.caption_id]]
    nodes = store.part_vectors[index.fields["node_rows"][:real_nodes(index)]] @ caption
    labelled = index.fields["edge_kind"] == EDGE_LABELLED
    triples = store.part_vectors[index.fields["triple_rows"][labelled]] @ caption
    clip = [("node", j, float(v)) for j, v in enumerate(nodes)] + [("triple", j, float(v)) for j, v in enumerate(triples)]
    orders["clip"] = [p[:2] for p in sorted(clip, key=lambda p: -p[2])]
    node_grad, triple_grad = ctx.gradient_terms(column, index)
    grad = [("node", j, float(v)) for j, v in enumerate(node_grad)] + [("triple", j, float(v)) for j, v in enumerate(triple_grad)]
    orders["gradient"] = [p[:2] for p in sorted(grad, key=lambda p: -p[2])]
    if gnn:
        orders["gnnexplainer"] = [p[:2] for p in gnnexplainer_order(ctx, column, index, base, empty)]
    return orders


def measure(ctx, rng, info, detail=False, compare=False, gnn=False):
    """Every metric for the rank-1 image of one caption; info gives synsets, buckets and count classes."""
    store, column = ctx.store, ctx.top
    image_id = ctx.ids[column]
    graph = store.graphs[image_id]
    parts_info = store.graph_parts(graph)
    index = image_index(store, graph)
    pairs = ctx.explanation(column)
    ranked, node_terms, triple_terms = ranked_parts(pairs, index, ctx.fusion_scales(column))
    n_parts = len(ranked)
    composition = {k: sum(kind == "triple" for kind, _, _ in ranked[:k]) for k in KS}
    concentration = {}
    for channel, terms in (("objects", node_terms), ("triples", triple_terms)):
        positive = np.clip(terms, 0, None)
        concentration[channel] = (float(np.sort(positive)[::-1][:3].sum() / positive.sum())
                                  if len(terms) > 3 and positive.sum() > 1e-12 else None)
    edits, labels = [edit(index), edit(index, drop_nodes=range(real_nodes(index)))], ["noop", "empty"]
    gold_column = ctx.ids.index(ctx.gold) if ctx.gold in ctx.ids else None
    for k in KS:
        nodes, triples = split(ranked[:k])
        edits += [edit(index, drop_nodes=nodes, drop_edges=triples), keep_only(index, nodes, triples)]
        labels += [f"plus_top_{k}", f"minus_top_{k}"]
        for _ in range(RANDOM):
            rn, rt = random_parts(rng, index, len(nodes), len(triples))
            edits += [edit(index, drop_nodes=rn, drop_edges=rt), keep_only(index, rn, rt)]
            labels += [f"plus_rand_{k}", f"minus_rand_{k}"]
    # explanation subgraph: the top-alpha incoming edges of E's top nodes, per node; random draws per node too
    top_nodes = [j for kind, j, _ in ranked if kind == "node"][:SUB_NODES]
    alpha = ctx.attention(index)
    dst, src, kinds = index.fields["edge_dst"], index.fields["edge_src"], index.fields["edge_kind"]
    chosen, per_node = [], []
    for v in top_nodes:
        incoming = [int(e) for e in np.flatnonzero(dst == v) if src[e] != v and np.isfinite(alpha[e])]
        picked = sorted(incoming, key=lambda e: -alpha[e])[:SUB_EDGES]
        chosen += picked
        per_node.append((incoming, len(picked)))
    draws = [[int(e) for incoming, n in per_node if n for e in rng.choice(incoming, size=n, replace=False)]
             for _ in range(RANDOM)]
    masks = [None] * len(edits)
    for form in ("messages", "edges"):
        for name, dropped in [("top", chosen)] + [("random", d) for d in draws]:
            if form == "messages":
                edits.append(edit(index))
                masks.append(message_drop(index, dropped))
            else:
                edits.append(edit(index, drop_edges=dropped))
                masks.append(None)
            labels.append(f"{form}_{name}")
    scores, ranks, full = ctx.rescore(column, edits, message_masks=masks, full=True)
    by = collections.defaultdict(list)
    for label, s, r in zip(labels, scores, ranks):
        by[label].append((float(s), int(r)))
    base, noop_rank = by["noop"][0]
    empty = by["empty"][0][0]
    effect = base - empty
    ordered = torch.sort(ctx.fused, descending=True).values
    out = {"image_id": image_id, "correct": image_id == ctx.gold, "parts": n_parts, "nodes": real_nodes(index),
           "triples": labelled_count(index), "empty_explanation": n_parts == 0, "noop_rank": noop_rank,
           "noop_error": abs(base - float(ctx.fused[column])), "top1_kind": ranked[0][0] if ranked else None,
           "gap": float(ordered[0] - ordered[1]), "base": base, "empty": empty, "effect": effect,
           "composition": composition, "concentration": concentration, "fidelity": {}}
    for k in KS:
        row = {}
        for sign in ("plus", "minus"):
            top_score, top_rank = by[f"{sign}_top_{k}"][0]
            rand = by[f"{sign}_rand_{k}"]
            rand_score = float(np.mean([s for s, _ in rand]))
            share = ((lambda s, sign=sign: ((base - s) if sign == "plus" else (s - empty)) / effect) if effect > 1e-4
                     else (lambda s: None))
            row[sign] = {"top": {"delta": top_score - base, "rank": top_rank, "lost_rank1": top_rank > 1, "share": share(top_score)},
                         "random": {"delta": rand_score - base, "rank": float(np.mean([r for _, r in rand])),
                                    "lost_rank1": float(np.mean([r > 1 for _, r in rand])), "share": share(rand_score)}}
        out["fidelity"][k] = row
    out["subgraph"] = {"n_edges": len(chosen), "pool": sum(len(i) for i, _ in per_node),
                       "with_choice": any(len(i) > n for i, n in per_node if n),
                       "kinds": dict(collections.Counter(KIND_NAMES[int(kinds[e])] for e in chosen))}
    for form in ("messages", "edges"):
        top_score, top_rank = by[f"{form}_top"][0]
        rand = by[f"{form}_random"]
        out["subgraph"][form] = {"top": {"delta": top_score - base, "rank": top_rank, "lost_rank1": top_rank > 1},
                                 "random": {"delta": float(np.mean([s for s, _ in rand])) - base,
                                            "rank": float(np.mean([r for _, r in rand])),
                                            "lost_rank1": float(np.mean([r > 1 for _, r in rand]))}}
    # sparsity: on E itself (terms, and the insertion curve below), and v1's caption rows
    out["sparsity"] = {"terms": term_sparsity(ranked)}
    rows = ctx.row_terms()
    deltas = top50_deltas(ctx.detail["weights"][0].tolist()[0], ctx.batch["clip"][0].cpu().numpy(), rows)
    raw = ctx.model.options["fusion"] == "raw"
    out["sparsity"]["caption_rows"] = caption_rows(deltas, column, with_clip=False)
    out["sparsity"]["caption_rows_with_clip"] = caption_rows(deltas, column, with_clip=True) if raw else None
    # counterfactual: the smallest k that costs rank 1, and the gold image after the top-k of a wrong image go
    model_curve = curves(ctx, column, [p[:2] for p in ranked], index, base, empty) if n_parts else None
    out["k_flip"] = model_curve["k_flip"] if model_curve else None
    if model_curve and effect > 1e-4:
        kept = [(s - empty) / effect for s in model_curve["insertion"]]
        out["sparsity"]["fidelity"] = {f"k{int(100 * t)}": next((n for n, v in zip(model_curve["sizes"], kept) if v >= t), None)
                                       for t in THRESHOLDS}
    if not out["correct"] and gold_column is not None and n_parts:
        # before and after from the same rescoring (the noop and plus_top_k rows), with rank_of's tie rule
        row = {label: i for i, label in reversed(list(enumerate(labels)))}
        gold = lambda i: int(rank_of(torch.from_numpy(full[i:i + 1]), gold_column)[0])  # noqa: E731
        out["gold_rank_after"] = {k: gold(row[f"plus_top_{k}"]) for k in KS}
        out["gold_rank_before"] = gold(row["noop"])
    # semantic validity of the object rows and of the triple rows
    query = store.queries[ctx.caption_id]
    validity = []
    for row in pairs["objects"]:
        q_obj, i_obj = query["objects"][row["query_index"]], graph["objects"][row["image_index"]]
        match = info["matcher"](q_obj["name"], i_obj["name"], info["synset"](image_id, row["image_index"], i_obj))
        cls = info["query_class"](ctx.caption_id, row["query_index"], q_obj)
        bucket = info["bucket"](image_id, row["image_index"], i_obj)
        validity.append({**match, "count_class": CAPTION_CLASSES[cls] if cls is not None else None,
                         "count_compatible": compatible(cls, bucket) if cls and bucket is not None else None})
    out["validity"] = validity
    q_names = {o["id"]: o["name"] for o in query["objects"]}
    i_objects = {o["id"]: (i, o) for i, o in enumerate(graph["objects"])}
    triple_validity = []
    for row in pairs["triples"]:
        q_rel, i_rel = query["relations"][row["query_index"]], graph["relations"][row["image_index"]]
        ends = {}
        for end in ("subject", "object"):
            i_index, i_obj = i_objects[i_rel[end]]
            ends[end] = info["matcher"](q_names[q_rel[end]], i_obj["name"], info["synset"](image_id, i_index, i_obj))["strict"]
        vocab = info["matcher"].vocab
        predicate = vocab.canon_predicate(q_rel["predicate"]) == vocab.canon_predicate(i_rel["predicate"])
        triple_validity.append({"subject": ends["subject"], "object": ends["object"], "predicate": predicate,
                                "all": ends["subject"] and ends["object"] and predicate})
    out["triple_validity"] = triple_validity
    if compare and n_parts and effect > 1e-4:
        orders = explainer_orders(ctx, column, index, ranked, rng, gnn, base, empty)
        out["explainers"] = {name: (model_curve if name == "model" else curves(ctx, column, order, index, base, empty))
                             for name, order in orders.items()}
    if detail:
        out["explanation"] = [{"kind": kind, "part": part_name(graph, parts_info, kind, j), "term": t}
                              for kind, j, t in ranked[:5]]
        names = [o["name"] for o in graph["objects"]]
        node_name = lambda v: names[parts_info["node_indices"][v]] if v < real_nodes(index) else "[global]"  # noqa: E731
        out["subgraph"]["chosen"] = [{"source": node_name(int(src[e])), "target": node_name(int(dst[e])),
                                      "kind": KIND_NAMES[int(kinds[e])], "alpha": float(alpha[e])} for e in chosen]
        out["object_rows"] = [{"query": query["objects"][r["query_index"]]["name"],
                               "image": graph["objects"][r["image_index"]]["name"], "a": r["a_ij"], **v}
                              for r, v in zip(pairs["objects"], validity)]
        out["curve"] = model_curve
    return out


def level1_info(matcher):
    """Synsets of level-1 image nodes (data/processed/geometry/objects_val.json); no counts (no count table)."""
    objects = json.loads((resolve(load_config(), "processed") / "geometry" / "objects_val.json").read_text(encoding="utf-8"))
    return {"matcher": matcher, "synset": lambda image_id, index, obj: objects[str(image_id)]["synsets"][index],
            "bucket": lambda image_id, index, obj: None, "query_class": lambda cid, index, obj: None}


def count_v1_info(matcher):
    return {"matcher": matcher, "synset": lambda image_id, index, obj: obj.get("synset"),
            "bucket": lambda image_id, index, obj: int(obj.get("count_bucket", 0)),
            "query_class": lambda cid, index, obj: int(obj.get("count_class", 0))}


def mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def wilcoxon(a, b):
    from scipy import stats
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 10 or all(x == y for x, y in pairs):
        return None
    x, y = np.array(pairs).T
    return {"n": len(pairs), "mean_difference": float(np.mean(x - y)), "p": float(stats.wilcoxon(x, y).pvalue)}


def aggregate(records):
    if not records:
        return {"captions": 0}
    # the explained image must be rank 1 when rescored unedited, or every edit would read as a lost rank 1
    usable = [r for r in records if not r["empty_explanation"] and r["noop_rank"] == 1]
    out = {"captions": len(records), "empty_explanations": len(records) - len(usable),
           "noop_not_rank1": sum(r["noop_rank"] != 1 for r in records),
           "noop_error_max": max(r["noop_error"] for r in records), "parts_mean": mean([r["parts"] for r in usable]),
           "gap_mean": mean([r["gap"] for r in records]), "fidelity": {}}
    for k in KS:
        out["fidelity"][k] = {sign: {who: {m: mean([r["fidelity"][k][sign][who][m] for r in usable])
                                         for m in ("delta", "rank", "lost_rank1", "share")}
                                     for who in ("top", "random")}
                              for sign in ("plus", "minus")}
        defined = [r for r in usable if r["effect"] > 1e-4]
        total = sum(r["effect"] for r in defined)
        out["fidelity"][k]["share_defined"] = len(defined)
        for sign in ("plus", "minus"):
            for who in ("top", "random"):
                kept = [(-f if sign == "plus" else f + r["effect"]) for r in defined
                        for f in [r["fidelity"][k][sign][who]["delta"]]]
                out["fidelity"][k][sign][who]["share_pooled"] = sum(kept) / total if total else None
                out["fidelity"][k][sign][who]["share_median"] = (
                    float(np.median([r["fidelity"][k][sign][who]["share"] for r in defined])) if defined else None)
    out["by_top1_kind"] = {}
    for kind in ("node", "triple"):
        chosen = [r for r in usable if r["top1_kind"] == kind]
        out["by_top1_kind"][kind] = {"captions": len(chosen), **{
            f"{sign}_{who}_{m}": mean([r["fidelity"][KS[0]][sign][who][m] for r in chosen])
            for sign in ("plus", "minus") for who in ("top", "random") for m in ("delta", "rank", "lost_rank1")}}
    out["subgraph"] = {}
    for scope, keep in (("all", lambda s: True), ("with_choice", lambda s: s["with_choice"])):
        sub = [r["subgraph"] for r in usable if r["subgraph"]["n_edges"] and keep(r["subgraph"])]
        kinds = collections.Counter()
        for s in sub:
            kinds.update(s["kinds"])
        out["subgraph"][scope] = {"captions": len(sub), "edges_mean": mean([s["n_edges"] for s in sub]),
                                  "pool_mean": mean([s["pool"] for s in sub]),
                                  "kinds": {k: v / max(sum(kinds.values()), 1) for k, v in kinds.items()},
                                  **{f"{form}_{who}_{m}": mean([s[form][who][m] for s in sub])
                                     for form in ("messages", "edges") for who in ("top", "random")
                                     for m in ("delta", "rank", "lost_rank1")}}
    out["composition"] = {k: mean([r["composition"][k] / min(k, r["parts"]) for r in usable]) for k in KS}
    out["concentration"] = {c: {"mean": mean([r["concentration"][c] for r in usable]),
                                "defined": sum(r["concentration"][c] is not None for r in usable)}
                            for c in ("objects", "triples")}
    out["parts"] = {"nodes": mean([r["nodes"] for r in usable]), "triples": mean([r["triples"] for r in usable])}
    terms = [r["sparsity"]["terms"] for r in usable if r["sparsity"]["terms"]]
    fid = [r["sparsity"].get("fidelity") for r in usable if r["sparsity"].get("fidelity")]
    out["sparsity"] = {"terms": {"defined": len(terms), **{f"{m}{int(100 * t)}": mean([x[f"{m}{int(100 * t)}"] for x in terms])
                                                           for m in ("n", "share") for t in THRESHOLDS}},
                       "fidelity": {"defined": len(fid), **{f"k{int(100 * t)}": mean([x[f"k{int(100 * t)}"] for x in fid])
                                                            for t in THRESHOLDS},
                                    **{f"reached{int(100 * t)}": mean([x[f"k{int(100 * t)}"] is not None for x in fid])
                                       for t in THRESHOLDS}}}
    for variant in ("caption_rows", "caption_rows_with_clip"):
        rows = [r["sparsity"][variant] for r in records if r["sparsity"].get(variant)]
        out["sparsity"][variant] = {"defined": len(rows), "rows_mean": mean([x["rows"] for x in rows]),
                                    **{f"{m}{int(100 * t)}": mean([x[f"{m}{int(100 * t)}"] for x in rows])
                                       for m in ("n", "share_of_rows") for t in THRESHOLDS}} if rows else None
    flips = [r["k_flip"] for r in usable]
    out["counterfactual"] = {"k_flip_reached": mean([f is not None for f in flips]),
                             "k_flip_mean": mean(flips), "k_flip_median": (float(np.median([f for f in flips if f is not None]))
                                                                        if any(f is not None for f in flips) else None)}
    wrong = [r for r in usable if "gold_rank_after" in r]
    if wrong:
        out["counterfactual"]["gold"] = {"captions": len(wrong), "rank_before": mean([r["gold_rank_before"] for r in wrong]),
                                         **{f"rank_after_{k}": mean([r["gold_rank_after"][k] for r in wrong]) for k in KS},
                                         **{f"rose_{k}": mean([r["gold_rank_after"][k] < r["gold_rank_before"] for r in wrong])
                                            for k in KS},
                                         **{f"first_{k}": mean([r["gold_rank_after"][k] == 1 for r in wrong]) for k in KS}}
    rows = [v for r in records for v in r["validity"]]
    counted = [v["count_compatible"] for v in rows if v["count_compatible"] is not None]
    triples = [v for r in records for v in r["triple_validity"]]
    out["validity"] = {"object_rows": len(rows),
                       **{k: mean([v[k] for v in rows]) for k in ("strict", "alias", "wordnet_strict", "lenient",
                                                                  "wordnet_lenient", "cosine", "head")},
                       "count_rows": len(counted), "count_compatible": mean(counted),
                       "triple_rows": len(triples),
                       **{f"triple_{k}": mean([v[k] for v in triples]) for k in ("subject", "object", "predicate", "all")}}
    compared = [r for r in usable if "explainers" in r]
    if compared:
        out["explainers"] = {}
        names = [n for n in EXPLAINERS if any(n in r["explainers"] for r in compared)]
        for name in names:
            rows = [r["explainers"][name] for r in compared if name in r["explainers"]]
            entry = {"captions": len(rows), "deletion_auc": mean([c.get("deletion_auc") for c in rows]),
                     "insertion_auc": mean([c.get("insertion_auc") for c in rows]),
                     "k_flip_reached": mean([c["k_flip"] is not None for c in rows]),
                     "k_flip_mean": mean([c["k_flip"] for c in rows])}
            for k in KS:
                entry[f"plus_{k}_lost_rank1"] = mean([c["deletion_rank"][k] > 1 for c in rows if len(c["sizes"]) > k])
                entry[f"minus_{k}_rank"] = mean([c["insertion_rank"][k] for c in rows if len(c["sizes"]) > k])
            if name != "model":
                both = [r for r in compared if name in r["explainers"]]
                entry["vs_model"] = {m: wilcoxon([r["explainers"][name].get(m) for r in both],
                                                 [r["explainers"]["model"].get(m) for r in both])
                                     for m in ("deletion_auc", "insertion_auc")}
            out["explainers"][name] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sample", type=int, help="fixed random sample of val captions (rng 2026); default all")
    parser.add_argument("--compare", type=int, default=1000, help="captions of the explainer comparison (rng 2026)")
    parser.add_argument("--gnn", type=int, default=200, help="first captions of the comparison that also run GNNExplainer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--merge", action="store_true", help="add these runs to an existing metrics.json / cases.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(2026)
    stores = {}

    def store_of(name):
        if name not in stores:
            config = MODELS[name][1]
            stores[name] = (GraphStore(load_config(REPO_ROOT / config), splits=("val",)) if config
                            else GraphStore(splits=("val",)))
        return stores[name]
    # The CLIP-cosine criterion reads node vectors from the count_v1 table, as v1 did, when that data is here.
    count_ready = (REPO_ROOT / "data" / "features" / "factual_v1" / "vg_coco_graph_parts.npy").exists()
    table = store_of("countonly_factual_r3") if count_ready else store_of(args.models[0])
    matcher = LabelMatcher(table.part_vectors, table.part_row, Vocab.from_config(load_config()))
    caption_ids = [str(c) for c in table.candidates("val")["caption_ids"]]
    if args.sample:
        caption_ids = sorted(np.random.default_rng(2026).choice(caption_ids, size=args.sample, replace=False).tolist())
    compare_ids = set(np.random.default_rng(2026).choice(caption_ids, size=min(args.compare, len(caption_ids)),
                                                         replace=False).tolist())
    gnn_ids = set(sorted(compare_ids)[:args.gnn])
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    case_ids = [c["caption_id"] for c in cases["cases"]] + [e["caption_id"] for e in cases["comparison"]["examples"]]
    case_text = {c["caption_id"]: c["caption"] for c in cases["cases"] + cases["comparison"]["examples"]}
    result = {"rule": __doc__, "captions": len(caption_ids), "sample": args.sample, "compare": len(compare_ids),
              "gnn": len(gnn_ids), "device": str(device), "torch": torch.__version__,
              "matcher_table": "count_v1" if count_ready else args.models[0], "runs": {}}
    detailed = {}
    if args.merge and (args.out / "metrics.json").exists():
        old = json.loads((args.out / "metrics.json").read_text(encoding="utf-8"))
        assert (old["captions"], old["sample"], old["compare"]) == (result["captions"], result["sample"], result["compare"])
        result["runs"] = old["runs"]
        detailed = {cid: {k: v for k, v in case.items() if k != "caption"}
                    for cid, case in json.loads((args.out / "cases.json").read_text(encoding="utf-8")).items()}

    def save():
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.out / "cases.json").write_text(json.dumps({cid: {"caption": case_text[cid], **detailed[cid]}
                                                         for cid in case_ids if cid in detailed},
                                                        indent=2, ensure_ascii=False), encoding="utf-8")
    for name in args.models:
        root, config = MODELS[name]
        store = store_of(name)
        info = count_v1_info(matcher) if config else level1_info(matcher)
        for seed in args.seeds:
            run = f"{name}_seed{seed}"
            free = shutil.disk_usage(REPO_ROOT).free / 1e9
            if free < MIN_FREE_GB:
                sys.exit(f"stop before {run}: {free:.1f} GB free < {MIN_FREE_GB} GB")
            model, _, saved = load_run_model(root / run, device, store=store)
            started = time.time()
            records, hits = [], 0
            for n, cid in enumerate(caption_ids):
                ctx = QueryContext(model, store, cid, device)
                rng = np.random.default_rng([2026, seed, int(cid.split("_")[0]), int(cid.split("_")[1])])
                keep = seed == 0 and cid in case_ids
                record = measure(ctx, rng, info, detail=keep, compare=cid in compare_ids, gnn=cid in gnn_ids)
                hits += record["correct"]
                if keep:
                    detailed.setdefault(cid, {})[name] = record
                records.append(record)
                if (n + 1) % 500 == 0:
                    print(run, n + 1, f"{time.time() - started:.0f}s", flush=True)
            if not args.sample:
                assert abs(hits / len(caption_ids) - saved["recall@1"]) < 1e-9, (run, hits / len(caption_ids), saved["recall@1"])
            if seed == 0:
                for cid in case_ids:
                    if cid not in detailed or name not in detailed[cid]:
                        ctx = QueryContext(model, store, cid, device)
                        rng = np.random.default_rng([2026, seed, int(cid.split("_")[0]), int(cid.split("_")[1])])
                        detailed.setdefault(cid, {})[name] = measure(ctx, rng, info, detail=True)
            result["runs"][run] = {"model": name, "seed": seed, "seconds": time.time() - started,
                                   "recall@1": hits / len(caption_ids),
                                   "groups": {group: aggregate([r for r in records if keep(r)])
                                              for group, keep in (("all", lambda r: True), ("correct", lambda r: r["correct"]),
                                                                  ("wrong", lambda r: not r["correct"]))}}
            print(run, json.dumps({k: v for k, v in result["runs"][run].items() if k != "groups"}), flush=True)
            save()


if __name__ == "__main__":
    main()
