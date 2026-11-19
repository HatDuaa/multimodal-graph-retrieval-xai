"""Level-3 explanations against known answers: three benchmarks where the parts that should explain the rank-1 image
are known before any explainer runs, scored for all 16 explainers of scripts/xai_metrics.py (version 3).

The explained image is always the rank-1 candidate, and only captions whose rank-1 image is the one with the answer
are kept. Each explainer orders the image's parts (object nodes and labelled triples); with g the size of the answer:
- GEA, graph explanation accuracy (Agarwal et al., GraphXAI, Scientific Data 2023): Jaccard of the first g parts of
  the order with the answer;
- precision@g: share of the answer among the first g parts;
- AUROC: probability that an answer part comes before a part outside the answer (from the order's positions);
- first: 1-based position of the first answer part.

Benchmarks (results in experiments/xai-ground-truth/<benchmark>/):
- planted (C), real models: a val caption's relation (subject, predicate, object) is planted into one of its
  candidates ranked 2 to 5 (two new object nodes with the caption's names, the caption's predicate between them),
  with a decoy of the same shape whose names and predicate are not in the caption. The case is kept only when the
  planted image becomes rank 1, so the model needs the planted parts (removing them gives back the old, lower rank).
  Answer: the three planted parts; the decoy's three parts are scored apart (their mean position percentile). New
  nodes take boxes of random nodes of the image and a count bucket from their name's count. Part texts that the
  tables lack are encoded with the frozen CLIP of the configuration, as the demo does for typed queries.
- vg-regions (A), real models: Visual Genome region descriptions of val images, with their region graphs
  (data/raw/visual_genome/region_graphs.json, scripts/download_data.py --regions) as the known answer: the image's
  nodes that are the region's annotated objects, and its triples between them with the region's predicates. The
  query graph is the region graph itself (canonical names and predicates; no parser, so a wrong explanation cannot
  come from parsing), the sentence vector is CLIP of the phrase, and the candidates are the CLIP top-50 of the val
  images. Two answers: instance (exactly the annotated nodes) and label (every node with the name of an annotated one,
  since the model matches labels and cannot tell two instances of one name apart). This is plausibility (agreement with
  people), not faithfulness. `prepare` writes the region queries of val once (data/processed/xai_regions/).
- synthetic (B), a model of the main architecture trained here: src/explain/synthetic_graphs.py, where every part of
  the answer is needed to beat some negative. `train` trains three seeds; the explained captions are val captions
  0..1999, PGExplainer and GraphMask train on val captions 2000..2199.

Usage:
  python scripts/xai_ground_truth.py planted [--models ...] [--seeds 0 1 2] [--cases 100]
  python scripts/xai_ground_truth.py vg-regions prepare
  python scripts/xai_ground_truth.py vg-regions [--models ...]
  python scripts/xai_ground_truth.py synthetic train
  python scripts/xai_ground_truth.py synthetic
Writes <benchmark>/metrics.json (per run: per explainer means over cases) and cases.json (per case and explainer).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xai_metrics as xm  # noqa: E402
from src.data.graph_store import GraphIndex  # noqa: E402
from src.explain import synthetic_graphs as synthetic  # noqa: E402
from src.explain.xai_metrics import QueryContext  # noqa: E402
from src.graph.counts import IMAGE_BUCKETS  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.train_graph_reranker import batch_loss, build_model, load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "xai-ground-truth"
REGIONS = REPO_ROOT / "data" / "processed" / "xai_regions" / "regions_val.json"  # 200 MB, not committed
DECOY_NAMES = ("lamp", "fence", "cloud", "bench", "sign", "pole", "window", "plant", "curtain", "rock")
DECOY_PREDICATES = ("near", "behind", "next to", "above", "under")
SYNTHETIC = {"seed": 2031, "train": 6000, "val": 2400, "candidates": 20, "explain": 2000, "epochs": 8, "lr": 1e-3,
             "batch": 32}


# scoring ------------------------------------------------------------------------------------------------------------

def gt_scores(order, answer, decoys=()):
    """GEA, precision@g, AUROC and first position of one explainer's order against an answer (module docstring)."""
    answer = {tuple(p) for p in answer}
    order = [tuple(p) for p in order]
    position = {p: i for i, p in enumerate(order)}
    g = len(answer)
    hit = len(set(order[:g]) & answer)
    inside = [position[p] for p in answer if p in position]
    outside = [position[p] for p in order if p not in answer]
    auroc = (float(np.mean([a < b for a in inside for b in outside])) if inside and outside else None)
    out = {"gea": hit / (2 * g - hit), "precision": hit / g, "auroc": auroc,
           "first": 1 + min(inside) if inside else None}
    if decoys:
        out["decoy_percentile"] = float(np.mean([position[tuple(d)] / max(len(order) - 1, 1) for d in decoys]))
    return out


def summarize(cases, keys):
    """Per explainer, the mean of every score over the cases (None scores left out)."""
    out = {"cases": len(cases)}
    names = [n for n in xm.EXPLAINERS if any(n in c["scores"] for c in cases)]
    for name in names:
        rows = [c["scores"][name] for c in cases if name in c["scores"]]
        out[name] = {"cases": len(rows)}
        for answer in keys:
            for metric in ("gea", "precision", "auroc", "first", "decoy_percentile"):
                values = [r[answer].get(metric) for r in rows if r.get(answer)]
                if any(v is not None for v in values):
                    out[name][f"{answer}_{metric}"] = xm.mean(values)
    out["answer_size"] = xm.mean([len(c["answer"]) for c in cases])
    out["parts"] = xm.mean([c["parts"] for c in cases])
    out["k_flip_reached"] = xm.mean([c["k_flip"] is not None for c in cases])
    return out


def explain(ctx, column, info, pg, gm, seed, answers, decoys=()):
    """measure() of xai_metrics with every explainer on the rank-1 image, and the scores of each explainer's order
    against each answer in answers ({name: parts}); None when the image has no positive graph effect."""
    assert ctx.top == column
    rng = np.random.default_rng([2026, seed, *xm.caption_numbers(ctx.caption_id)])
    record = xm.measure(ctx, rng, info, compare=True, gnn=True, pg=pg, seed=seed, gm=gm, keep_orders=True)
    if "orders" not in record:
        return None
    scores = {name: {key: gt_scores(order, parts, decoys if key == "main" else ()) for key, parts in answers.items()}
              for name, order in record["orders"].items()}
    return {"caption_id": ctx.caption_id, "image_id": int(ctx.ids[column]), "parts": record["parts"],
            "answer": [list(p) for p in answers["main"]], "decoys": [list(p) for p in decoys],
            "k_flip": record["k_flip"], "cf_exact": record.get("cf_exact"), "effect": record["effect"],
            "scores": scores}


def info_for(name, matcher, config):
    """xai_metrics' synset / bucket / count lookups, safe for image nodes added after packaging (planted)."""
    if config:
        return xm.count_v1_info(matcher)
    base = xm.level1_info(matcher)
    synset = base["synset"]

    def safe(image_id, index, obj):
        try:
            return synset(image_id, index, obj)
        except (IndexError, KeyError):
            return obj.get("synset")
    return {**base, "synset": safe}


def add_parts(store, encoder, keys, prompt):
    """Append CLIP vectors of part keys the table lacks (node texts with the concept prompt, as the packaging)."""
    missing = sorted({k for k in keys if k not in store.part_row})
    if not missing:
        return 0
    texts = [prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1] for k in missing]
    vectors = encoder.encode_texts(texts)
    offset = len(store.part_vectors)
    store.part_vectors = np.concatenate([store.part_vectors, vectors.astype(np.float32)])
    store.part_row.update({k: offset + i for i, k in enumerate(missing)})
    store._device_tables = {}
    return len(missing)


def graph_keys(graph):
    names = {o["id"]: o["name"] for o in graph["objects"]}
    return ([f"node:{n}" for n in names.values()] + [f"rel:{r['predicate']}" for r in graph["relations"]]
            + [f"triple:{names[r['subject']]} {r['predicate']} {names[r['object']]}" for r in graph["relations"]])


def positions(store, graph, objects=(), relations=()):
    """Part positions (("node", j) / ("triple", j)) of graph objects and relations, by their list indices."""
    parts = store.graph_parts(graph)
    return ([("node", parts["node_indices"].index(i)) for i in objects]
            + [("triple", parts["relation_indices"].index(i)) for i in relations])


def trained_explainers(model, store, device, pool, count, seed):
    chosen = xm.training_captions(model, store, device, pool, count)
    pg, pg_info = xm.train_pg(model, store, device, chosen, seed)
    gm, gm_info = xm.train_gm(model, store, device, chosen, seed)
    return pg, gm, {"pgexplainer": pg_info, "graphmask": gm_info}


def save(folder, metrics, cases):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (folder / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")


def load_previous(folder, merge):
    if merge and (folder / "metrics.json").exists():
        return (json.loads((folder / "metrics.json").read_text(encoding="utf-8")),
                json.loads((folder / "cases.json").read_text(encoding="utf-8")))
    return {"rule": __doc__, "runs": {}}, {}


def store_of(config):
    """The val GraphStore of a level-3 model's data configuration (None: level 1)."""
    from src.data.graph_store import GraphStore
    return GraphStore(load_config(REPO_ROOT / config), splits=("val",)) if config else GraphStore(splits=("val",))


def matcher_of(store):
    return xm.LabelMatcher(store.part_vectors, store.part_row, Vocab.from_config(load_config()))


def explainer_pool(store, exclude):
    ids = [str(c) for c in store.candidates("val")["caption_ids"]]
    compare = set(np.random.default_rng(2026).choice(ids, size=1000, replace=False).tolist())
    cases = json.loads(xm.CASES.read_text(encoding="utf-8"))
    case_ids = {c["caption_id"] for c in cases["cases"]} | {e["caption_id"] for e in cases["comparison"]["examples"]}
    pool = sorted(set(ids) - compare - case_ids - set(exclude))
    return np.random.default_rng(2027).permutation(pool)


# C. planted ---------------------------------------------------------------------------------------------------------

def plant(graph, motifs, rng):
    """A copy of graph with each motif (subject name, predicate, object name) added as two new nodes and one relation,
    in random order; returns the copy and, per motif, (object indices, relation index)."""
    graph = copy.deepcopy(graph)
    objects, relations = graph["objects"], graph["relations"]
    boxes = [o.get("box") for o in objects]
    placed = []
    for i in rng.permutation(len(motifs)):
        s, p, o = motifs[i]
        ids = []
        for name in (s, o):
            new = {"id": max([x["id"] for x in objects], default=-1) + 1, "name": name, "synset": None}
            if boxes and all(b is not None for b in boxes):
                new["box"] = list(boxes[int(rng.integers(len(boxes)))])
            objects.append(new)
            ids.append(new["id"])
        relations.append({"subject": ids[0], "predicate": p, "object": ids[1]})
        placed.append((int(i), [len(objects) - 2, len(objects) - 1], len(relations) - 1))
    counts = {}
    for obj in objects:
        counts[obj["name"]] = counts.get(obj["name"], 0) + 1
    for obj in objects[-2 * len(motifs):]:
        if "count_bucket" in objects[0] or any("count_bucket" in x for x in objects[:-2 * len(motifs)]):
            c = counts[obj["name"]]
            obj["count_bucket"] = IMAGE_BUCKETS.index("1" if c == 1 else "2" if c == 2 else "3-4" if c <= 4 else ">=5")
    return graph, {i: (objs, rel) for i, objs, rel in sorted(placed)}


def caption_motif(store, cid, rng):
    """(caption relation, decoy) for planting, or None when the caption has no relation with two named ends."""
    query = store.queries[cid]
    names = {o["id"]: o["name"] for o in query["objects"]}
    rels = [r for r in query["relations"] if names.get(r["subject"]) and names.get(r["object"])
            and r["subject"] != r["object"]]
    if not rels:
        return None
    r = rels[0]
    words = set(" ".join(names.values()).split())
    decoy_names = [n for n in DECOY_NAMES if n not in words]
    decoy_preds = [p for p in DECOY_PREDICATES if p != r["predicate"]]
    a, b = rng.choice(decoy_names, size=2, replace=False)
    return ((names[r["subject"]], r["predicate"], names[r["object"]]),
            (str(a), str(rng.choice(decoy_preds)), str(b)))


def run_planted(args):
    from src.features.extract_clip import ClipEncoder
    folder = args.out / "planted"
    metrics, cases_out = load_previous(folder, args.merge)
    metrics.update({"benchmark": "planted", "cases_per_run": args.cases})
    cfg = load_config()
    encoder = ClipEncoder.from_config(cfg, device="cpu")
    for name in args.models:
        root, config = xm.MODELS[name]
        store = store_of(config)
        info = info_for(name, matcher_of(store), config)
        ids = [str(c) for c in store.candidates("val")["caption_ids"]]
        for seed in args.seeds:
            run = f"{name}_seed{seed}"
            model, _, _ = load_run_model(root / run, args.device, store=store)
            started = time.time()
            pool = explainer_pool(store, [])
            pg, gm, trained = trained_explainers(model, store, args.device, pool, args.explainer_train, seed)
            training = set(pool[:400].tolist())
            order = [c for c in np.random.default_rng(2029).permutation(ids) if c not in training]
            cases, tried = [], 0
            for cid in order:
                if len(cases) == args.cases:
                    break
                rng = np.random.default_rng([2029, seed, *[int(x) for x in cid.split("_")]])
                motifs = caption_motif(store, cid, rng)
                if motifs is None:
                    continue
                tried += 1
                base_ctx = QueryContext(model, store, cid, args.device)
                ranked = torch.argsort(base_ctx.fused, descending=True, stable=True).tolist()
                for column in ranked[1:5]:
                    image_id = base_ctx.ids[column]
                    planted, placed = plant(store.graphs[image_id], motifs, rng)
                    add_parts(store, encoder, graph_keys(planted), cfg["clip"]["concept_prompt"])
                    index = GraphIndex.build([store.graph_parts(planted)], store.part_row)
                    _, rank = base_ctx.rescore(column, [index])
                    if int(rank[0]) != 1:
                        continue
                    original = store.graphs[image_id]
                    store.graphs[image_id] = planted
                    try:
                        ctx = QueryContext(model, store, cid, args.device, overrides={image_id: planted})
                        if ctx.top != column:
                            continue
                        answer = positions(store, planted, placed[0][0], [placed[0][1]])
                        decoy = positions(store, planted, placed[1][0], [placed[1][1]])
                        case = explain(ctx, column, info, pg, gm, seed, {"main": answer}, decoy)
                    finally:
                        store.graphs[image_id] = original
                    if case is not None:
                        case.update({"motif": motifs[0], "decoy_motif": motifs[1], "rank_before": ranked.index(column) + 1})
                        cases.append(case)
                        if len(cases) % 10 == 0:
                            print(run, len(cases), f"{time.time() - started:.0f}s", flush=True)
                    break
            metrics["runs"][run] = {"model": name, "seed": seed, "captions_tried": tried, **trained,
                                    "seconds": time.time() - started, "summary": summarize(cases, ["main"])}
            cases_out[run] = cases
            save(folder, metrics, cases_out)
            print(run, json.dumps({k: v for k, v in metrics["runs"][run].items() if k != "summary"}), flush=True)


# A. Visual Genome region graphs -------------------------------------------------------------------------------------

def json_items(path, chunk=1 << 24):
    """The items of a top-level JSON list, decoded one at a time (region_graphs.json is 2.8 GB)."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, started = "", False
        while True:
            data = handle.read(chunk)
            buffer += data
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    assert buffer[0] == "[", "expected a JSON list"
                    buffer, started = buffer[1:], True
                    continue
                buffer = buffer.lstrip(",").lstrip()
                if not buffer or buffer[0] == "]":
                    break
                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not data:
                        raise
                    break
                yield item
                buffer = buffer[end:]
            if not data:
                return


def prepare_regions(args):
    """Region queries of val images with their known answers, for both models' graphs; writes REGIONS and the
    counts of kept and dropped regions (prepare.json)."""
    from src.features.extract_clip import ClipEncoder
    from src.graph.scene_graph import build_scene_graph, load_raw
    cfg = load_config()
    vocab = Vocab.from_config(cfg)
    folder = args.out / "vg-regions"
    level1 = {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            g = json.loads(line)
            level1[int(g["image_id"])] = g
    merged = {}
    count_cfg = load_config(REPO_ROOT / xm.MODELS["countonly_factual_r3"][1])
    with (REPO_ROOT / count_cfg["paths"]["graphs"] / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            g = json.loads(line)
            if g["split"] == "val":
                merged[int(g["image_id"])] = g
    val = sorted(merged)
    raw = load_raw(cfg)
    rng = np.random.default_rng(2030)
    by_image = {int(r["image_id"]): r["regions"]
                for r in json_items(resolve(cfg, "raw") / "visual_genome" / "region_graphs.json")
                if r.get("image_id") is not None and int(r["image_id"]) in merged}
    queries, stats = [], {"images": len(val), "regions": 0, "unmapped_object": 0, "no_triple": 0, "duplicate": 0,
                          "level1_name_mismatch": 0}
    for image_id in val:
        rebuilt = build_scene_graph(raw[0].get(image_id, {"image_id": image_id}),
                                    raw[1].get(image_id, {"image_id": image_id}), vocab, vg150_only=False,
                                    include_source_ids=True)
        stored = level1[image_id]
        if [o["name"] for o in rebuilt["objects"]] != [o["name"] for o in stored["objects"]]:
            stats["level1_name_mismatch"] += 1
            continue
        level1_of = {o["source_object_id"]: o["id"] for o in rebuilt["objects"]}
        merge_of = {m: o["id"] for o in merged[image_id]["objects"] for m in o["members"]}
        seen = set()
        for region in by_image.get(image_id, []):
            stats["regions"] += 1
            phrase = " ".join(str(region.get("phrase", "")).split()).strip().rstrip(".").lower()
            objs = {int(o["object_id"]): vocab.canon_object(o.get("name") or (o.get("names") or [""])[0])
                    for o in region.get("objects", [])}
            rels = [(int(r["subject_id"]), vocab.canon_predicate(r.get("predicate", "")), int(r["object_id"]))
                    for r in region.get("relationships", [])]
            rels = [r for r in rels if r[0] in objs and r[2] in objs and r[0] != r[2] and r[1]]
            if not phrase or not rels or not all(objs.values()):
                stats["no_triple"] += 1
                continue
            if any(o not in level1_of for o in objs):
                stats["unmapped_object"] += 1
                continue
            if phrase in seen:
                stats["duplicate"] += 1
                continue
            seen.add(phrase)
            answer = {}
            for graph_name, graph, node_of in (("level1", stored, lambda o: level1_of[o]),
                                               ("merge_v1", merged[image_id], lambda o: merge_of.get(level1_of[o]))):
                nodes = {o: node_of(o) for o in objs}
                if None in nodes.values():
                    answer = None
                    break
                triples = sorted({i for i, r in enumerate(graph["relations"]) for s, p, t in rels
                                  if r["subject"] == nodes[s] and r["object"] == nodes[t]
                                  and vocab.canon_predicate(r["predicate"]) == p})
                names = {o["id"]: o["name"] for o in graph["objects"]}
                index_of = {o["id"]: i for i, o in enumerate(graph["objects"])}
                instance = sorted({index_of[n] for n in nodes.values()})
                label = sorted({i for i, o in enumerate(graph["objects"]) if o["name"] in {names[n] for n in nodes.values()}})
                if not triples:
                    answer = None
                    break
                answer[graph_name] = {"objects": instance, "label_objects": label, "relations": triples}
            if answer is None:
                stats["no_triple"] += 1
                continue
            ids = {o: i for i, o in enumerate(sorted(objs))}
            query = {"objects": [{"id": ids[o], "name": objs[o], "count_class": 0} for o in sorted(objs)],
                     "relations": [{"subject": ids[s], "predicate": p, "object": ids[t]} for s, p, t in rels]}
            queries.append({"caption_id": f"region_{region['region_id']}", "image_id": image_id, "phrase": phrase,
                            "query": query, "answer": answer})
    order = rng.permutation(len(queries))
    queries = [queries[i] for i in order]
    encoder = ClipEncoder.from_config(cfg, device="cpu")
    vectors = encoder.encode_texts([q["phrase"] for q in queries])
    image_vectors = np.load(resolve(cfg, "features") / "vg_coco_image.npy")
    image_ids = json.loads((resolve(cfg, "features") / "vg_coco_image.ids.json").read_text(encoding="utf-8"))["ids"]
    rows = {int(i): r for r, i in enumerate(image_ids)}
    val_rows = np.array([rows[i] for i in val])
    sims = vectors @ image_vectors[val_rows].T
    kept = []
    for q, v, s in zip(queries, vectors, sims):
        top = np.argsort(-s, kind="stable")[:50]
        candidates = [val[i] for i in top]
        if q["image_id"] not in candidates:
            continue
        q.update({"candidate_ids": candidates, "clip_scores": s[top].astype(float).tolist(),
                  "gold_rank": candidates.index(q["image_id"]) + 1, "sentence": v.astype(float).tolist()})
        kept.append(q)
    stats.update({"queries": len(queries), "gold_in_top50": len(kept)})
    REGIONS.parent.mkdir(parents=True, exist_ok=True)
    REGIONS.write_text(json.dumps({"rule": prepare_regions.__doc__, "stats": stats, "queries": kept}), encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prepare.json").write_text(json.dumps({"regions": str(REGIONS.relative_to(REPO_ROOT)), "stats": stats},
                                                    indent=2), encoding="utf-8")
    print(json.dumps(stats), flush=True)


def add_queries(store, queries):
    """Put region queries into a loaded store as a split "regions" (candidates, CLIP scores, sentence vectors) and
    rebuild its query index."""
    store.precompute()
    offset = len(store.caption_vectors)
    store.caption_vectors = np.concatenate([store.caption_vectors, np.array([q["sentence"] for q in queries],
                                                                            dtype=np.float32)])
    arrays = {"caption_ids": [], "gold": [], "candidate_ids": [], "clip_scores": [], "gold_rank": [], "group_index": []}
    for i, q in enumerate(queries):
        cid = q["caption_id"]
        store.queries[cid] = q["query"]
        store.caption_row[cid] = offset + i
        store.caption_location[cid] = ("regions", i)
        arrays["caption_ids"].append(cid)
        arrays["gold"].append(q["image_id"])
        arrays["candidate_ids"].append(q["candidate_ids"])
        arrays["clip_scores"].append(q["clip_scores"])
        arrays["gold_rank"].append(q["gold_rank"])
        arrays["group_index"].append(0)
    store.arrays["regions"] = {k: np.array(v) for k, v in arrays.items()}
    store.arrays["regions"]["clip_scores"] = store.arrays["regions"]["clip_scores"].astype(np.float32)
    caption_ids = list(store.queries)
    store.query_position = {cid: i for i, cid in enumerate(caption_ids)}
    store.query_index = GraphIndex.build((store.graph_parts(store.queries[c], query=True) for c in caption_ids),
                                         store.part_row)


def run_regions(args):
    from src.features.extract_clip import ClipEncoder
    folder = args.out / "vg-regions"
    data = json.loads(REGIONS.read_text(encoding="utf-8"))
    queries = data["queries"]
    metrics, cases_out = load_previous(folder, args.merge)
    metrics.update({"benchmark": "vg-regions", "prepared": data["stats"], "cases_per_run": args.cases})
    cfg = load_config()
    encoder = ClipEncoder.from_config(cfg, device="cpu")
    for name in args.models:
        root, config = xm.MODELS[name]
        store = store_of(config)
        add_parts(store, encoder, [k for q in queries for k in graph_keys(q["query"])], cfg["clip"]["concept_prompt"])
        add_queries(store, queries)
        info = info_for(name, matcher_of(store), config)
        graph_name = "merge_v1" if config else "level1"
        for seed in args.seeds:
            run = f"{name}_seed{seed}"
            model, _, _ = load_run_model(root / run, args.device, store=store)
            started = time.time()
            pg, gm, trained = trained_explainers(model, store, args.device, explainer_pool(store, []),
                                                 args.explainer_train, seed)
            cases, tried, rank1 = [], 0, 0
            for q in queries:
                if len(cases) == args.cases:
                    break
                tried += 1
                ctx = QueryContext(model, store, q["caption_id"], args.device)
                if ctx.ids[ctx.top] != q["image_id"]:
                    continue
                rank1 += 1
                graph = store.graphs[q["image_id"]]
                a = q["answer"][graph_name]
                answers = {"main": positions(store, graph, a["objects"], a["relations"]),
                           "label": positions(store, graph, a["label_objects"], a["relations"])}
                case = explain(ctx, ctx.top, info, pg, gm, seed, answers)
                if case is not None:
                    case["phrase"] = q["phrase"]
                    cases.append(case)
                    if len(cases) % 10 == 0:
                        print(run, len(cases), f"{time.time() - started:.0f}s", flush=True)
            metrics["runs"][run] = {"model": name, "seed": seed, "queries_tried": tried, "gold_rank1": rank1, **trained,
                                    "seconds": time.time() - started, "summary": summarize(cases, ["main", "label"])}
            cases_out[run] = cases
            save(folder, metrics, cases_out)
            print(run, json.dumps({k: v for k, v in metrics["runs"][run].items() if k != "summary"}), flush=True)


# B. synthetic -------------------------------------------------------------------------------------------------------

def synthetic_data(args, encoder=None):
    """The synthetic dataset and its store; CLIP vectors cached in data/features/synthetic_xai/ (not committed)."""
    cfg = load_config()
    data = synthetic.make_dataset(SYNTHETIC["seed"], SYNTHETIC["train"], SYNTHETIC["val"], SYNTHETIC["candidates"])
    cache = REPO_ROOT / "data" / "features" / "synthetic_xai" / f"vectors_{SYNTHETIC['seed']}.npz"
    keys = synthetic.part_keys(data)
    captions = sorted(data["captions"])
    if not cache.exists():
        from src.features.extract_clip import ClipEncoder
        encoder = encoder or ClipEncoder.from_config(cfg, device="cpu")
        prompt = cfg["clip"]["concept_prompt"]
        parts = encoder.encode_texts([prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1]
                                      for k in keys])
        sentences = encoder.encode_texts([data["captions"][c] for c in captions])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, parts=parts, sentences=sentences, keys=np.array(keys), captions=np.array(captions))
    with np.load(cache) as saved:
        assert list(saved["keys"]) == keys and list(saved["captions"]) == captions
        parts, sentences = saved["parts"], saved["sentences"]
    store = synthetic.synthetic_store(data, parts, {k: i for i, k in enumerate(keys)}, sentences)
    return data, store


def synthetic_options():
    """The countonly run's flags (experiments/improve/count-only), without its fusion scale (fitted on COCO)."""
    root, _ = xm.MODELS["countonly_factual_r3"]
    options = json.loads((root / "countonly_factual_r3_seed0" / "config.json").read_text(encoding="utf-8"))["args"]
    return {**options, "fusion_scale": None}


@torch.no_grad()
def recall_at_1(model, store, split, device, limit=None):
    ids = [str(c) for c in store.arrays[split]["caption_ids"]][:limit]
    hits = 0
    for start in range(0, len(ids), 64):
        batch = store.batch(ids[start:start + 64], device)
        scores, _ = model(batch)
        top = scores.argmax(-1).cpu().numpy()
        hits += sum(int(batch["candidate_ids"][i][t]) == batch["gold"][i] for i, t in enumerate(top))
    return hits / len(ids)


def train_synthetic(args):
    folder = args.out / "synthetic"
    data, store = synthetic_data(args)
    log = {"data": SYNTHETIC, "options": synthetic_options(), "runs": {}}
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(synthetic_options(), args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=SYNTHETIC["lr"])
        train_ids = [str(c) for c in store.arrays["train"]["caption_ids"]]
        best, history = None, []
        for epoch in range(SYNTHETIC["epochs"]):
            model.train()
            order = np.random.default_rng([seed, epoch]).permutation(len(train_ids))
            losses = []
            for start in range(0, len(order), SYNTHETIC["batch"]):
                ids = [train_ids[i] for i in order[start:start + SYNTHETIC["batch"]]]
                loss = batch_loss(model, store, ids, args.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
            model.eval()
            val = recall_at_1(model, store, "val", args.device, limit=SYNTHETIC["explain"])
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_recall@1": val})
            print(f"synthetic seed {seed} epoch {epoch}", json.dumps(history[-1]), flush=True)
            if best is None or val > best[0]:
                best = (val, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        (folder / "models").mkdir(parents=True, exist_ok=True)
        torch.save({"model": best[2], "epoch": best[1], "metrics": {"recall@1": best[0]}},
                   folder / "models" / f"seed{seed}.pt")
        log["runs"][f"seed{seed}"] = {"best_epoch": best[1], "val_recall@1": best[0], "history": history}
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "training.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def run_synthetic(args):
    folder = args.out / "synthetic"
    data, store = synthetic_data(args)
    metrics, cases_out = load_previous(folder, args.merge)
    metrics.update({"benchmark": "synthetic", "cases_per_run": args.cases, "data": SYNTHETIC})
    info = xm.count_v1_info(matcher_of(store))
    val_ids = [str(c) for c in store.arrays["val"]["caption_ids"]]
    explain_ids, pool = val_ids[:SYNTHETIC["explain"]], val_ids[SYNTHETIC["explain"]:]
    for seed in args.seeds:
        run = f"synthetic_seed{seed}"
        model = build_model(synthetic_options(), args.device)
        state = torch.load(folder / "models" / f"seed{seed}.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        started = time.time()
        pg, gm, trained = trained_explainers(model, store, args.device, pool, args.explainer_train, seed)
        cases, tried = [], 0
        for cid in explain_ids:
            if len(cases) == args.cases:
                break
            tried += 1
            ctx = QueryContext(model, store, cid, args.device)
            answer = data["answers"][cid]
            if ctx.ids[ctx.top] != answer["image"]:
                continue
            graph = store.graphs[answer["image"]]
            case = explain(ctx, ctx.top, info, pg, gm, seed,
                           {"main": positions(store, graph, answer["nodes"], answer["triples"])})
            if case is not None:
                case["caption"] = data["captions"][cid]
                cases.append(case)
                if len(cases) % 10 == 0:
                    print(run, len(cases), f"{time.time() - started:.0f}s", flush=True)
        metrics["runs"][run] = {"model": "synthetic", "seed": seed, "captions_tried": tried,
                                "val_recall@1": state["metrics"]["recall@1"], **trained,
                                "seconds": time.time() - started, "summary": summarize(cases, ["main"])}
        cases_out[run] = cases
        save(folder, metrics, cases_out)
        print(run, json.dumps({k: v for k, v in metrics["runs"][run].items() if k != "summary"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("planted", "vg-regions", "synthetic"))
    parser.add_argument("step", nargs="?", choices=("prepare", "train"))
    parser.add_argument("--models", nargs="+", default=list(xm.MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cases", type=int, default=100, help="explained cases per run")
    parser.add_argument("--explainer-train", type=int, default=200, help="captions that train PGExplainer and GraphMask")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--merge", action="store_true", help="add these runs to existing results")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    torch.manual_seed(2026)
    if args.benchmark == "planted":
        run_planted(args)
    elif args.benchmark == "vg-regions":
        prepare_regions(args) if args.step == "prepare" else run_regions(args)
    else:
        train_synthetic(args) if args.step == "train" else run_synthetic(args)


if __name__ == "__main__":
    main()
