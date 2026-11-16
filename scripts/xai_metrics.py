"""Level-3 explanation metrics on val: fidelity, validity, sparsity and explanation subgraphs (no training, no test).

Models: countonly_factual_r3 (count_v1) and main_r3 (level-1 data), three seeds each, checkpoint_best. For every val
caption (or a fixed sample, rng 2026, with --sample) the image the model ranks first is explained. Its explanation E
ranks the image's parts (object nodes and labelled triples) by their exact term (src/explain/xai_metrics.py, which
also defines the edits). All scores are recomputed in memory; nothing is cached on disk.

- Fidelity, k = 1, 3, 5. Fid+: drop the top-k parts of E. Fid-: keep only the top-k parts (and the global node).
  Each against the mean of 10 random draws with the same number of nodes and of triples (uniform over all the
  image's nodes / triples). Reported: change of the final score, rank, and the share of the image's graph effect
  base - empty that is removed (Fid+) or kept (Fid-), where empty is the image with every part dropped (both
  channels then take fuse_raw's filled mean); shares only when base - empty > 1e-4.
- Validity. Counterfactual: share of captions whose image loses rank 1 when the top-k parts are dropped, against
  random. Semantic: for every object row of E, does the image node with the highest attention carry the caption
  node's label (VG alias, WordNet same type as src/graph/dedup.same_type with the caption word's first noun sense, or
  frozen-CLIP node cosine >= 0.955, the criterion of scripts/improve/count_match_changes.py), with the same head noun
  after VG aliases reported apart ("dining table" ~ "table": WordNet puts dining_table.n.01 under table.n.03, VG
  tables are table.n.02); and, for caption nodes
  with a stated count class, is the node's count bucket compatible (src/graph/counts.compatible). Level-1 images get
  synsets from data/processed/geometry/objects_val.json and buckets from the same count rule; level-1 caption nodes
  get count classes from the same spaCy rules.
- Sparsity. Rows of the demo's "So với TB top-50" column (each row minus its mean over the 50 candidates,
  src/explain/score_breakdown.top50_deltas): the fewest rows, by decreasing delta, whose deltas reach 80% / 90% of the
  image's margin over the mean. "with CLIP": CLIP row + graph rows, margin fused - mean(fused) (raw fusion only);
  "graph": graph rows only, margin of the graph part (raw channel terms; for main_r3 before its z-scores). Also as a
  share of the image's parts (nodes + labelled triples). Only when the margin is positive.
- Explanation subgraph. For the top-3 nodes of E, the 3 incoming edges with the highest GAT alpha (both layers, all
  heads); dropping these edges against dropping as many random incoming edges of the same nodes (10 draws).
Groups: all captions, captions whose rank-1 image is the gold image, and those whose rank-1 image is wrong.
Writes experiments/xai-metrics/{metrics.json, cases.json}.
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
                                     real_nodes)
from src.graph.counts import CAPTION_CLASSES, caption_classes, compatible, image_bucket, image_counts  # noqa: E402
from src.graph.dedup import same_type  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-metrics"
MODELS = {"countonly_factual_r3": (REPO_ROOT / "experiments" / "improve" / "count-only", "configs/count_v1.yaml"),
          "main_r3": (REPO_ROOT / "experiments" / "level1" / "gat", None)}
KS, RANDOM, THRESHOLDS, SUB_NODES, SUB_EDGES, COSINE = (1, 3, 5), 10, (0.8, 0.9), 3, 3, .955
KIND_NAMES = {EDGE_LABELLED: "labelled", EDGE_COMPLETION: "completion", EDGE_GLOBAL: "global"}
CASES = REPO_ROOT / "experiments" / "improve" / "explanations" / "cases.json"
MIN_FREE_GB = 10


class LabelMatcher:
    """Does an image node carry a caption node's label? VG alias, WordNet same type, or frozen-CLIP cosine."""

    def __init__(self, vectors, rows, vocab):
        self.vectors, self.rows, self.vocab, self.cache = vectors, rows, vocab, {}

    @staticmethod
    def first_sense(name):
        from nltk.corpus import wordnet as wn
        for word in (name.replace(" ", "_"), name.split()[-1] if name.split() else name):
            senses = wn.synsets(word, pos="n")
            if senses:
                return senses[0].name()
        return None

    def __call__(self, query, image, synset):
        key = (query, image, synset)
        if key not in self.cache:
            alias = self.vocab.canon_object(query) == self.vocab.canon_object(image)
            sense = self.first_sense(query)
            wordnet = bool(sense and synset) and same_type(sense, synset, query, image)
            known = f"node:{query}" in self.rows and f"node:{image}" in self.rows
            cosine = known and float(self.vectors[self.rows[f"node:{query}"]] @ self.vectors[self.rows[f"node:{image}"]]) >= COSINE
            # extra, outside "any": the same head noun after VG aliases ("dining table" ~ "table")
            head = self.vocab.canon_object(query.split()[-1] if query.split() else query) == \
                self.vocab.canon_object(image.split()[-1] if image.split() else image)
            any_ = alias or wordnet or cosine
            self.cache[key] = {"alias": alias, "wordnet": wordnet, "cosine": cosine, "any": any_, "head": head,
                               "any_or_head": any_ or head}
        return self.cache[key]


def explanation_parts(pairs, index):
    """E: [(kind, position, term)] by decreasing term; nodes and labelled triples without a term get 0."""
    nodes, triples = np.zeros(real_nodes(index)), np.zeros(labelled_count(index))
    for j, t in enumerate(pairs["image_terms"]["objects"]):
        nodes[j] = t["term"]
    for j, t in enumerate(pairs["image_terms"]["triples"]):
        triples[j] = t["term"]
    parts = [("node", j, float(t)) for j, t in enumerate(nodes)] + [("triple", j, float(t)) for j, t in enumerate(triples)]
    return sorted(parts, key=lambda p: -p[2]), nodes, triples


def split(parts):
    return [j for kind, j, _ in parts if kind == "node"], [j for kind, j, _ in parts if kind == "triple"]


def random_parts(rng, index, n_nodes, n_triples):
    nodes = rng.choice(real_nodes(index), size=n_nodes, replace=False).tolist() if n_nodes else []
    triples = rng.choice(labelled_count(index), size=n_triples, replace=False).tolist() if n_triples else []
    return nodes, triples


def sparsity(deltas, column, parts_total, with_clip):
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
        graph_rows = sum(kind == "graph" for kind, _ in ranked[:n])
        out[f"n{int(100 * threshold)}"] = n
        out[f"graph_rows{int(100 * threshold)}"] = graph_rows
        out[f"share_of_parts{int(100 * threshold)}"] = graph_rows / parts_total if parts_total else None
        out[f"share_of_rows{int(100 * threshold)}"] = n / len(values)
    return out


def part_name(store, graph, parts, kind, j):
    if kind == "node":
        return graph["objects"][parts["node_indices"][j]]["name"]
    names = {o["id"]: o["name"] for o in graph["objects"]}
    rel = graph["relations"][parts["relation_indices"][j]]
    return f"{names[rel['subject']]} {rel['predicate']} {names[rel['object']]}"


def measure(ctx, rng, info, detail=False):
    """Every metric for the rank-1 image of one caption; info gives labels, synsets, buckets, classes."""
    store, column = ctx.store, ctx.top
    image_id = ctx.ids[column]
    graph = store.graphs[image_id]
    parts_info = store.graph_parts(graph)
    index = image_index(store, graph)
    pairs = ctx.explanation(column)
    ranked, node_terms, triple_terms = explanation_parts(pairs, index)
    n_parts = len(ranked)
    # how E is made: triples among its top-k, and per channel the share of the positive term mass in its top 3 parts
    composition = {k: sum(kind == "triple" for kind, _, _ in ranked[:k]) for k in KS}
    concentration = {}
    for channel, terms in (("objects", node_terms), ("triples", triple_terms)):
        positive = np.clip(terms, 0, None)
        concentration[channel] = (float(np.sort(positive)[::-1][:3].sum() / positive.sum())
                                  if len(terms) > 3 and positive.sum() > 1e-12 else None)
    edits, labels = [edit(index), edit(index, drop_nodes=range(real_nodes(index)))], ["noop", "empty"]
    for k in KS:
        nodes, triples = split(ranked[:k])
        edits += [edit(index, drop_nodes=nodes, drop_edges=triples), keep_only(index, nodes, triples)]
        labels += [f"plus_top_{k}", f"minus_top_{k}"]
        for r in range(RANDOM):
            rn, rt = random_parts(rng, index, len(nodes), len(triples))
            edits += [edit(index, drop_nodes=rn, drop_edges=rt), keep_only(index, rn, rt)]
            labels += [f"plus_rand_{k}", f"minus_rand_{k}"]
    # explanation subgraph: the top incoming GAT edges of E's top nodes
    top_nodes = [j for kind, j, _ in ranked if kind == "node"][:SUB_NODES]
    alpha = ctx.attention(index)
    dst, src, kinds = index.fields["edge_dst"], index.fields["edge_src"], index.fields["edge_kind"]
    chosen, pool = [], []
    for v in top_nodes:
        incoming = [e for e in np.flatnonzero(dst == v) if src[e] != v]
        pool += incoming
        chosen += sorted(incoming, key=lambda e: -alpha[e])[:SUB_EDGES]
    edits.append(edit(index, drop_edges=chosen))
    labels.append("edge_top")
    for r in range(RANDOM):
        edits.append(edit(index, drop_edges=rng.choice(pool, size=len(chosen), replace=False).tolist() if chosen else []))
        labels.append("edge_rand")
    scores, ranks = ctx.rescore(column, edits)
    by = collections.defaultdict(list)
    for label, s, r in zip(labels, scores, ranks):
        by[label].append((float(s), int(r)))
    base = float(ctx.fused[column])
    empty = by["empty"][0][0]
    effect = base - empty
    ordered = torch.sort(ctx.fused, descending=True).values
    out = {"image_id": image_id, "correct": image_id == ctx.gold, "parts": n_parts, "nodes": real_nodes(index),
           "top1_kind": ranked[0][0] if ranked else None, "gap": float(ordered[0] - ordered[1]), "effect": effect,
           "triples": labelled_count(index), "noop_error": abs(by["noop"][0][0] - base), "base": base, "empty": empty,
           "composition": composition, "concentration": concentration, "fidelity": {}}
    for k in KS:
        row = {}
        for sign in ("plus", "minus"):
            top_score, top_rank = by[f"{sign}_top_{k}"][0]
            rand = by[f"{sign}_rand_{k}"]
            rand_score = float(np.mean([s for s, _ in rand]))
            share = (lambda s: ((base - s) if sign == "plus" else (s - empty)) / effect) if effect > 1e-4 else (lambda s: None)
            row[sign] = {"top": {"delta": top_score - base, "rank": top_rank, "lost_rank1": top_rank > 1, "share": share(top_score)},
                         "random": {"delta": rand_score - base, "rank": float(np.mean([r for _, r in rand])),
                                    "lost_rank1": float(np.mean([r > 1 for _, r in rand])),
                                    "share": share(rand_score)}}
        out["fidelity"][k] = row
    edge_score, edge_rank = by["edge_top"][0]
    rand = by["edge_rand"]
    out["subgraph"] = {"edges": len(chosen), "pool": len(pool),
                       "kinds": dict(collections.Counter(KIND_NAMES[int(kinds[e])] for e in chosen)),
                       "top": {"delta": edge_score - base, "rank": edge_rank},
                       "random": {"delta": float(np.mean([s for s, _ in rand])) - base,
                                  "rank": float(np.mean([r for _, r in rand])),
                                  "lost_rank1": float(np.mean([r > 1 for _, r in rand]))}}
    # sparsity on the demo's delta rows
    rows = ctx.row_terms()
    weights = ctx.detail["weights"][0].tolist()
    deltas = top50_deltas(weights[0], ctx.batch["clip"][0].cpu().numpy(), rows)
    raw = ctx.model.options["fusion"] == "raw"
    out["sparsity"] = {"graph": sparsity(deltas, column, n_parts, with_clip=False),
                       "with_clip": sparsity(deltas, column, n_parts, with_clip=True) if raw else None}
    # semantic validity of the object rows
    query = store.queries[ctx.caption_id]
    validity = []
    for row in pairs["objects"]:
        q_obj, i_obj = query["objects"][row["query_index"]], graph["objects"][row["image_index"]]
        match = info["matcher"](q_obj["name"], i_obj["name"], info["synset"](image_id, row["image_index"], i_obj))
        cls = info["query_class"](ctx.caption_id, row["query_index"], q_obj)
        bucket = info["bucket"](image_id, row["image_index"], i_obj)
        validity.append({**match, "count_class": CAPTION_CLASSES[cls],
                         "count_compatible": compatible(cls, bucket) if cls else None})
    out["validity"] = validity
    if detail:
        out["explanation"] = [{"kind": kind, "part": part_name(store, graph, parts_info, kind, j), "term": t}
                              for kind, j, t in ranked[:5]]
        names = [o["name"] for o in graph["objects"]]
        node_name = lambda v: names[parts_info["node_indices"][v]] if v < real_nodes(index) else "[global]"  # noqa: E731
        out["subgraph"]["chosen"] = [{"source": node_name(int(src[e])), "target": node_name(int(dst[e])),
                                      "kind": KIND_NAMES[int(kinds[e])], "alpha": float(alpha[e])} for e in chosen]
        out["object_rows"] = [{"query": query["objects"][r["query_index"]]["name"],
                               "image": graph["objects"][r["image_index"]]["name"], "a": r["a_ij"], **v}
                              for r, v in zip(pairs["objects"], validity)]
    return out


def level1_info(store, matcher, captions):
    """Synsets (objects_<split>.json), count buckets (same rule as count_v1) and spaCy count classes for level 1."""
    import spacy
    objects = json.loads((resolve(load_config(), "processed") / "geometry" / "objects_val.json").read_text(encoding="utf-8"))
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    texts = {cid: text for cid, text in captions.items() if cid in store.queries}
    classes = {}
    for cid, doc in zip(texts, nlp.pipe(texts.values(), batch_size=512)):
        classes[cid] = [c for c, _ in caption_classes(doc, store.queries[cid]["objects"])]
    buckets = {}

    def bucket(image_id, index, obj):
        if image_id not in buckets:
            synsets = objects[str(image_id)]["synsets"]
            plain = [{"name": o["name"], "synset": synsets[i]} for i, o in enumerate(store.graphs[image_id]["objects"])]
            buckets[image_id] = [image_bucket(c) for c in image_counts(plain, same_type)]
        return buckets[image_id][index]
    return {"matcher": matcher, "synset": lambda image_id, index, obj: objects[str(image_id)]["synsets"][index],
            "bucket": bucket, "query_class": lambda cid, index, obj: classes[cid][index]}


def count_v1_info(matcher):
    return {"matcher": matcher, "synset": lambda image_id, index, obj: obj.get("synset"),
            "bucket": lambda image_id, index, obj: int(obj.get("count_bucket", 0)),
            "query_class": lambda cid, index, obj: int(obj.get("count_class", 0))}


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def aggregate(records):
    out = {"captions": len(records), "parts_mean": mean([r["parts"] for r in records]),
           "noop_error_max": max(r["noop_error"] for r in records), "fidelity": {}}
    for k in KS:
        out["fidelity"][k] = {sign: {who: {m: mean([float(r["fidelity"][k][sign][who][m]) if r["fidelity"][k][sign][who][m] is not None
                                                   else None for r in records])
                                         for m in ("delta", "rank", "lost_rank1", "share")}
                                     for who in ("top", "random")}
                              for sign in ("plus", "minus")}
        out["fidelity"][k]["share_defined"] = sum(r["fidelity"][k]["plus"]["top"]["share"] is not None for r in records)
        # ratio of sums and median: the mean of per-caption shares is driven by captions with a tiny base - empty
        defined = [r for r in records if r["effect"] > 1e-4]
        total = sum(r["effect"] for r in defined)
        for sign in ("plus", "minus"):
            for who in ("top", "random"):
                kept = [(-f if sign == "plus" else f + r["effect"])
                        for r in defined for f in [r["fidelity"][k][sign][who]["delta"]]]
                out["fidelity"][k][sign][who]["share_pooled"] = sum(kept) / total if total else None
                out["fidelity"][k][sign][who]["share_median"] = (
                    float(np.median([r["fidelity"][k][sign][who]["share"] for r in defined])) if defined else None)
    out["gap_mean"] = mean([r["gap"] for r in records])
    out["by_top1_kind"] = {}
    for kind in ("node", "triple"):
        chosen = [r for r in records if r["top1_kind"] == kind]
        out["by_top1_kind"][kind] = {"captions": len(chosen), **{
            f"{sign}_{who}_{m}": mean([float(r["fidelity"][KS[0]][sign][who][m]) for r in chosen])
            for sign in ("plus", "minus") for who in ("top", "random") for m in ("delta", "rank", "lost_rank1")}}
    sub = [r["subgraph"] for r in records if r["subgraph"]["edges"]]
    # a random draw can differ from the top edges only when the nodes have more incoming edges than were dropped
    choice = [s for s in sub if s["pool"] > s["edges"]]
    kinds = collections.Counter()
    for s in sub:
        kinds.update(s["kinds"])
    out["subgraph"] = {"captions": len(sub), "edges_mean": mean([s["edges"] for s in sub]),
                       "kinds": {k: v / max(sum(kinds.values()), 1) for k, v in kinds.items()},
                       "top_delta": mean([s["top"]["delta"] for s in sub]), "random_delta": mean([s["random"]["delta"] for s in sub]),
                       "top_rank": mean([s["top"]["rank"] for s in sub]), "random_rank": mean([s["random"]["rank"] for s in sub]),
                       "top_lost_rank1": mean([s["top"]["rank"] > 1 for s in sub]),
                       "random_lost_rank1": mean([s["random"]["lost_rank1"] for s in sub]),
                       "with_choice": {"captions": len(choice), "edges_mean": mean([s["edges"] for s in choice]),
                                       "pool_mean": mean([s["pool"] for s in choice]),
                                       "top_delta": mean([s["top"]["delta"] for s in choice]),
                                       "random_delta": mean([s["random"]["delta"] for s in choice]),
                                       "top_rank": mean([s["top"]["rank"] for s in choice]),
                                       "random_rank": mean([s["random"]["rank"] for s in choice]),
                                       "top_lost_rank1": mean([s["top"]["rank"] > 1 for s in choice]),
                                       "random_lost_rank1": mean([s["random"]["lost_rank1"] for s in choice])}}
    out["composition"] = {k: mean([r["composition"][k] / min(k, r["parts"]) for r in records if r["parts"]]) for k in KS}
    out["concentration"] = {c: {"mean": mean([r["concentration"][c] for r in records]),
                                "defined": sum(r["concentration"][c] is not None for r in records)}
                            for c in ("objects", "triples")}
    out["parts"] = {"nodes": mean([r["nodes"] for r in records]), "triples": mean([r["triples"] for r in records])}
    out["sparsity"] = {}
    for variant in ("graph", "with_clip"):
        rows = [r["sparsity"][variant] for r in records if r["sparsity"][variant]]
        out["sparsity"][variant] = {"defined": len(rows), **{f"{m}{int(100 * t)}": mean([x[f"{m}{int(100 * t)}"] for x in rows])
                                                             for m in ("n", "graph_rows", "share_of_parts", "share_of_rows")
                                                             for t in THRESHOLDS},
                                    "rows_mean": mean([x["rows"] for x in rows])} if rows else None
    rows = [v for r in records for v in r["validity"]]
    counted = [v["count_compatible"] for v in rows if v["count_compatible"] is not None]
    out["validity"] = {"object_rows": len(rows), **{k: mean([v[k] for v in rows]) for k in ("any", "alias", "wordnet", "cosine", "head", "any_or_head")},
                       "count_rows": len(counted), "count_compatible": mean(counted)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sample", type=int, help="fixed random sample of val captions (rng 2026); default all")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--merge", action="store_true", help="add these runs to an existing metrics.json / cases.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count_store = GraphStore(load_config(REPO_ROOT / "configs" / "count_v1.yaml"), splits=("val",))
    stores = {"countonly_factual_r3": count_store, "main_r3": GraphStore(splits=("val",))}
    matcher = LabelMatcher(count_store.part_vectors, count_store.part_row, Vocab.from_config(load_config()))
    captions = {c["caption_id"]: c["text"] for im in json.loads((resolve(load_config(), "splits") / "vg_coco_val.json")
                                                                .read_text(encoding="utf-8"))["images"] for c in im["captions"]}
    caption_ids = [str(c) for c in count_store.candidates("val")["caption_ids"]]
    if args.sample:
        caption_ids = sorted(np.random.default_rng(2026).choice(caption_ids, size=args.sample, replace=False).tolist())
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    case_ids = [c["caption_id"] for c in cases["cases"]] + [e["caption_id"] for e in cases["comparison"]["examples"]]
    result = {"rule": __doc__, "captions": len(caption_ids), "sample": args.sample, "device": str(device),
              "torch": torch.__version__, "runs": {}}
    detailed = {}
    if args.merge and (args.out / "metrics.json").exists():
        old = json.loads((args.out / "metrics.json").read_text(encoding="utf-8"))
        assert (old["captions"], old["sample"]) == (result["captions"], result["sample"]), "merge needs the same captions"
        result["runs"] = old["runs"]
        detailed = {cid: {k: v for k, v in case.items() if k != "caption"}
                    for cid, case in json.loads((args.out / "cases.json").read_text(encoding="utf-8")).items()}
    case_text = {c["caption_id"]: c["caption"] for c in cases["cases"] + cases["comparison"]["examples"]}

    def save():
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.out / "cases.json").write_text(json.dumps({cid: {"caption": case_text[cid], **detailed[cid]}
                                                         for cid in case_ids if cid in detailed},
                                                        indent=2, ensure_ascii=False), encoding="utf-8")
    for name in args.models:
        root, config = MODELS[name]
        store = stores[name]
        info = count_v1_info(matcher) if config else level1_info(store, matcher, captions)
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
                record = measure(ctx, rng, info, detail=seed == 0 and cid in case_ids)
                hits += record["correct"]
                if seed == 0 and cid in case_ids:
                    detailed.setdefault(cid, {})[name] = record
                records.append(record)
                if (n + 1) % 1000 == 0:
                    print(run, n + 1, f"{time.time() - started:.0f}s", flush=True)
            if not args.sample:
                assert abs(hits / len(caption_ids) - saved["recall@1"]) < 1e-9, (run, hits / len(caption_ids), saved["recall@1"])
            # the case captions that a sample left out are measured on their own
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
