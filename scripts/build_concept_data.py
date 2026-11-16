"""Build data version concept_v1: the concept graph, image node attributes and fixed scene terms (scene-ranker options).

Reads configs/concept_v1.yaml (the count_v1 data plus paths.concepts and a `concepts` settings block) and writes, under
paths.concepts, never overwriting without --force:
- concepts.json: vocabulary (concept names, WordNet synsets, train image counts) and relation names;
- concept_graph.npz: CLIP text vector of every concept name (the part table's node vectors) and the typed, weighted
  edges between concepts (src/graph/concepts.py states the rules);
- image_attributes.json: {image_id: one list of Visual Genome attributes per node, in the packaged graph's node order}
  for train and val; a merged node (merge_v1) takes the union of its members' attributes;
- attribute_texts.npy / .ids.json: CLIP text vectors of every "attribute label" text, same encoder and prompt as the
  node vectors;
- terms_train.npz / terms_val.npz: the four fixed scene terms (src/graph/scene_terms.py) of every (caption, candidate)
  pair, row-aligned with candidates_<split>.npz and divided by their train scale;
- MANIFEST.json: settings, counts, term scales, input checksums and run time.

Everything is fitted on train: the vocabulary, the co-occurrence prior, the IDF of soft_objects and the term scales.
Val only gets its terms computed. Test is never read.

Usage: python scripts/build_concept_data.py [--config configs/concept_v1.yaml] [--force] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.extract_clip import ClipEncoder, load_features, save_features  # noqa: E402
from src.graph.concepts import RELATION_NAMES, SCENE_TERMS, ConceptVocab  # noqa: E402
from src.graph.scene_graph import Vocab, build_scene_graph, normalize_label  # noqa: E402
from src.graph.scene_terms import attribute_text, caption_side, caption_terms, image_side  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

SPLITS = ("train", "val")
CONCEPTNET_RELATIONS = {"/r/RelatedTo": "cn_related", "/r/Synonym": "cn_related", "/r/UsedFor": "cn_function",
                        "/r/CapableOf": "cn_function", "/r/AtLocation": "cn_function", "/r/PartOf": "cn_function"}
DIRECTED = {"vg_prior": True, "hypernym": True, "cn_related": False, "cn_function": True}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str) -> None:
    print(f"{time.strftime('%T')} {message}", flush=True)


def load_graphs(cfg) -> dict[int, dict]:
    graphs = {}
    with (REPO_ROOT / cfg["paths"]["graphs"] / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] in SPLITS:
                graphs[int(graph["image_id"])] = graph
    return graphs


def part_table(cfg) -> tuple[np.ndarray, dict[str, int]]:
    """The frozen CLIP part table (features + extra_parts), as GraphStore reads it."""
    vectors, keys, _ = load_features(REPO_ROOT / cfg["paths"]["features"] / "vg_coco_graph_parts")
    rows = {key: row for row, key in enumerate(keys)}
    if cfg["paths"].get("extra_parts"):
        extra, extra_keys, _ = load_features(REPO_ROOT / cfg["paths"]["extra_parts"])
        new = [i for i, key in enumerate(extra_keys) if key not in rows]
        rows.update({extra_keys[i]: len(vectors) + n for n, i in enumerate(new)})
        vectors = np.concatenate([vectors, extra[new]])
    return np.ascontiguousarray(vectors, dtype=np.float32), rows


# --- vocabulary and concept graph ---------------------------------------------------------------------------------

def vocabulary(graphs, settings):
    images, synsets = Counter(), defaultdict(Counter)
    for graph in graphs.values():
        if graph["split"] != "train":
            continue
        images.update({obj["name"] for obj in graph["objects"] if obj["name"]})
        for obj in graph["objects"]:
            if obj.get("synset"):
                synsets[obj["name"]][obj["synset"]] += 1
    names = sorted(name for name, count in images.items() if count >= settings["min_images"])
    return names, [synsets[name].most_common(1)[0][0] if synsets[name] else None for name in names], \
        [images[name] for name in names]


def prior_edges(graphs, vocab, settings):
    pairs, train = Counter(), 0
    for graph in graphs.values():
        if graph["split"] != "train":
            continue
        train += 1
        concept = [vocab.concept_of(obj["name"]) for obj in graph["objects"]]
        pairs.update({(concept[r["subject"]], concept[r["object"]]) for r in graph["relations"]
                      if concept[r["subject"]] >= 0 and concept[r["object"]] >= 0
                      and concept[r["subject"]] != concept[r["object"]]})
    return [(s, o, "vg_prior", math.log1p(n) / math.log1p(train)) for (s, o), n in pairs.items()
            if n >= settings["prior_min_images"]]


def hypernym_edges(synsets):
    """concept -> its nearest WordNet hypernyms (fewest steps up) that are vocabulary concepts."""
    from nltk.corpus import wordnet as wn
    owner = defaultdict(list)
    for concept, synset in enumerate(synsets):
        if synset:
            owner[synset].append(concept)
    edges = []
    for concept, name in enumerate(synsets):
        try:
            level = wn.synset(name).hypernyms() if name else []
        except Exception:  # noqa: BLE001 - malformed or retired VG synset names
            level = []
        seen = set()
        while level:
            found = {c for s in level for c in owner.get(s.name(), ()) if c != concept}
            if found:
                edges.extend((concept, parent, "hypernym", 1.) for parent in sorted(found))
                break
            seen.update(s.name() for s in level)
            level = [h for s in level for h in s.hypernyms() if h.name() not in seen]
    return edges


def conceptnet_path(cfg, settings) -> Path:
    path = resolve(cfg, "raw") / "conceptnet" / settings["conceptnet_url"].rsplit("/", 1)[1]
    if not path.exists():
        from scripts.download_data import download
        log(f"downloading ConceptNet to {path}")
        download(settings["conceptnet_url"], path)
    return path


def conceptnet_edges(path, vocab, settings):
    best = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            family = CONCEPTNET_RELATIONS.get(row[1])
            if family is None or not row[2].startswith("/c/en/") or not row[3].startswith("/c/en/"):
                continue
            ends = [vocab.index.get(normalize_label(uri.split("/")[3].replace("_", " "))) for uri in row[2:4]]
            if None in ends or ends[0] == ends[1]:
                continue
            weight = float(json.loads(row[4]).get("weight", 1.))
            if weight < settings["conceptnet_min_weight"]:
                continue
            s, o = ends if DIRECTED[family] else sorted(ends)
            best[(s, o, family)] = max(best.get((s, o, family), 0.), weight)
    top = Counter()
    for (_, _, family), weight in best.items():
        top[family] = max(top[family], weight)
    return [(s, o, family, weight / top[family]) for (s, o, family), weight in best.items()]


def prune(edges, names, counts, settings):
    """Keep, per concept, `quota_per_family` best neighbours of every relation type, then the best others up to
    `neighbors`; an edge survives when either endpoint keeps it."""
    incident = [defaultdict(dict) for _ in names]
    for s, o, family, weight in edges:
        for a, b in ((s, o), (o, s)):
            incident[a][family][b] = max(incident[a][family].get(b, 0.), weight)
    # Better neighbour first: higher weight, then more train images, then the name.
    rank = lambda item: (-item[1], -counts[item[0]], names[item[0]])  # noqa: E731
    limit, choices = settings["neighbors"], []
    for families in incident:
        chosen, best = set(), {}
        for family in sorted(families):
            chosen.update(n for n, _ in sorted(families[family].items(), key=rank)[:settings["quota_per_family"]])
            for n, w in families[family].items():
                best[n] = max(best.get(n, 0.), w)
        ranked = [n for n, _ in sorted(best.items(), key=rank)]
        chosen = [n for n in ranked if n in chosen][:limit]
        chosen += [n for n in ranked if n not in chosen][:limit - len(chosen)]
        choices.append(set(chosen))
    return [e for e in edges if e[1] in choices[e[0]] or e[0] in choices[e[1]]]


# --- image attributes --------------------------------------------------------------------------------------------

def image_attributes(cfg, graphs):
    """Visual Genome attributes of every packaged node: level-1 graphs are rebuilt with their source object ids
    (as in scripts/improve/build_boxes.py), and a merged node takes the union over its members."""
    folder = resolve(cfg, "raw") / "visual_genome"
    wanted = set(graphs)
    loaded = {}
    for name in ("objects.json", "relationships.json", "attributes.json"):
        path = folder / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run: python scripts/download_data.py")
        log(f"reading {path.name}")
        entries = json.loads(path.read_text(encoding="utf-8"))
        loaded[name] = {x["image_id"]: x for x in entries if x["image_id"] in wanted}
        del entries
    vocab = Vocab.from_config(cfg)
    out, attributed = {}, 0
    for image_id, graph in graphs.items():
        empty = {"image_id": image_id}
        level1 = build_scene_graph(loaded["objects.json"].get(image_id, empty),
                                   loaded["relationships.json"].get(image_id, empty), vocab, vg150_only=False,
                                   include_source_ids=True)
        by_object = {}
        for item in loaded["attributes.json"].get(image_id, empty).get("attributes", []):
            values = {normalize_label(a) for a in item.get("attributes") or () if normalize_label(a)}
            by_object.setdefault(item["object_id"], set()).update(values)
        sources = [obj["source_object_id"] for obj in level1["objects"]]
        nodes = []
        for obj in graph["objects"]:
            members = obj.get("members", [obj["id"]])
            # A merged node is named after one of its members, so its label must be among theirs.
            if max(members) >= len(sources) or obj["name"] not in {level1["objects"][m]["name"] for m in members}:
                raise ValueError(f"image {image_id}: node members do not match the rebuilt level-1 graph")
            nodes.append(sorted(set().union(*(by_object.get(sources[m], set()) for m in members))))
        attributed += sum(bool(values) for values in nodes)
        out[str(image_id)] = nodes
    return out, attributed


def encode_attribute_texts(cfg, texts, stem: Path) -> tuple[np.ndarray, dict[str, int]]:
    encoder = ClipEncoder.from_config(cfg)
    prompt = cfg["clip"]["concept_prompt"]
    log(f"encoding {len(texts)} attribute texts with {encoder.name}")
    vectors = encoder.encode_texts([prompt.format(text) for text in texts]).astype(np.float32)
    save_features(stem, vectors, texts, encoder.name)
    return vectors, {text: row for row, text in enumerate(texts)}


# --- scene terms -------------------------------------------------------------------------------------------------

def query_objects(graph):
    return [obj for obj in graph["objects"] if obj["name"] not in PRONOUNS]


def label_idf(queries_train):
    documents = Counter(name for graph in queries_train.values() for name in {o["name"] for o in query_objects(graph)})
    total = len(queries_train)
    idf = {name: math.log((total + 1) / (n + 1)) + 1 for name, n in documents.items()}
    return idf, math.log(total + 1) + 1


def split_terms(arrays, queries, sides, vocab, rows, tables, idf, limit=None):
    label_row, attribute_row = rows
    label_table, attribute_table = tables
    count = len(arrays["caption_ids"]) if limit is None else min(limit, len(arrays["caption_ids"]))
    terms = np.zeros((count, arrays["candidate_ids"].shape[1], len(SCENE_TERMS)), dtype=np.float32)
    for i in range(count):
        caption = caption_side(query_objects(queries[str(arrays["caption_ids"][i])]), vocab, label_row,
                               attribute_row, *idf)
        images = [sides[int(image_id)] for image_id in arrays["candidate_ids"][i]]
        terms[i] = caption_terms(caption, images, label_table, attribute_table)
        if (i + 1) % 20000 == 0:
            log(f"terms {i + 1}/{count}")
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/concept_v1.yaml"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing concept_v1 folder's files")
    parser.add_argument("--limit", type=int, help="testing only: terms for the first N captions of each split")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config(REPO_ROOT / args.config)
    settings = cfg["concepts"]
    out = REPO_ROOT / cfg["paths"]["concepts"]
    out.mkdir(parents=True, exist_ok=True)
    targets = ["concepts.json", "concept_graph.npz", "image_attributes.json", "attribute_texts.npy",
               "attribute_texts.ids.json", "terms_train.npz", "terms_val.npz", "MANIFEST.json"]
    existing = [name for name in targets if (out / name).exists()]
    if existing and not args.force:
        raise FileExistsError(f"{out} already holds {existing}; use --force to rebuild")

    graphs = load_graphs(cfg)
    log(f"graphs: {sum(g['split'] == 'train' for g in graphs.values())} train, "
        f"{sum(g['split'] == 'val' for g in graphs.values())} val")
    names, synsets, image_counts = vocabulary(graphs, settings)
    vocab = ConceptVocab(names)
    vectors, part_row = part_table(cfg)
    missing = [name for name in names if f"node:{name}" not in part_row]
    if missing:
        raise ValueError(f"{len(missing)} concept names have no node vector, e.g. {missing[:5]}")
    features = vectors[[part_row[f"node:{name}"] for name in names]]

    cn_path = conceptnet_path(cfg, settings)
    raw_edges = (prior_edges(graphs, vocab, settings) + hypernym_edges(synsets)
                 + conceptnet_edges(cn_path, vocab, settings))
    edges = prune(raw_edges, names, image_counts, settings)
    log(f"concepts: {len(names)}, edges: {len(raw_edges)} before pruning, {len(edges)} kept")
    relation = {name: i for i, name in enumerate(RELATION_NAMES)}
    np.savez(out / "concept_graph.npz", features=features,
             source=np.array([e[0] for e in edges], dtype=np.int64),
             target=np.array([e[1] for e in edges], dtype=np.int64),
             relation=np.array([relation[e[2]] for e in edges], dtype=np.int64),
             weight=np.array([e[3] for e in edges], dtype=np.float64),
             directed=np.array([DIRECTED[e[2]] for e in edges], dtype=bool))
    (out / "concepts.json").write_text(json.dumps({"names": names, "synsets": synsets, "train_images": image_counts,
                                                   "relation_names": list(RELATION_NAMES)}), encoding="utf-8")

    attributes, attributed = image_attributes(cfg, graphs)
    (out / "image_attributes.json").write_text(json.dumps(attributes), encoding="utf-8")
    queries = {split: json.loads((resolve(cfg, "processed") / f"query_graphs_{split}.json").read_text(encoding="utf-8"))
               for split in SPLITS}
    texts = {attribute_text(value, obj["name"]) for image_id, graph in graphs.items()
             for obj, values in zip(graph["objects"], attributes[str(image_id)], strict=True) for value in values}
    texts |= {attribute_text(value, obj["name"]) for split in SPLITS for graph in queries[split].values()
              for obj in query_objects(graph) for value in obj.get("attributes") or ()}
    attribute_vectors, attribute_row = encode_attribute_texts(cfg, sorted(texts), out / "attribute_texts")

    label_row = {key.split(":", 1)[1]: row for key, row in part_row.items() if key.startswith("node:")}
    sides = {image_id: image_side(graph, attributes[str(image_id)], vocab, label_row, attribute_row)
             for image_id, graph in graphs.items()}
    idf = label_idf(queries["train"])
    manifest_terms = {}
    scales = None
    for split in SPLITS:
        with np.load(resolve(cfg, "processed") / f"candidates_{split}.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in ("caption_ids", "candidate_ids")}
        log(f"terms for {split}: {len(arrays['caption_ids'])} captions")
        terms = split_terms(arrays, queries[split], sides, vocab, (label_row, attribute_row),
                            (vectors, attribute_vectors), idf, args.limit)
        if split == "train":
            # Fixed scale of every term: its std over a caption's candidates, averaged over train captions.
            scales = terms.std(axis=1).mean(axis=0)
            scales = np.where(scales > 0, scales, 1.)
        count = len(terms)
        np.savez(out / f"terms_{split}.npz", names=np.array(SCENE_TERMS), caption_ids=arrays["caption_ids"][:count],
                 candidate_ids=arrays["candidate_ids"][:count], terms=(terms / scales).astype(np.float32))
        manifest_terms[split] = {
            "captions": count,
            "nonzero_share": dict(zip(SCENE_TERMS, (terms != 0).mean((0, 1)).tolist(), strict=True)),
            "mean": dict(zip(SCENE_TERMS, terms.mean((0, 1)).tolist(), strict=True))}

    by_relation = Counter(e[2] for e in edges)
    manifest = {"version": Path(cfg["paths"]["concepts"]).name, "date": time.strftime("%F %T %z"),
                "config": str(args.config), "settings": settings, "concepts": len(names),
                "edges": {name: by_relation[name] for name in RELATION_NAMES}, "edges_before_pruning": len(raw_edges),
                "image_nodes_with_attributes": attributed, "attribute_texts": len(texts),
                "term_scales": dict(zip(SCENE_TERMS, scales.tolist(), strict=True)), "terms": manifest_terms,
                "limit": args.limit, "seconds": round(time.time() - started, 1),
                "inputs_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in (
                    REPO_ROOT / cfg["paths"]["graphs"] / "vg_coco_scene_graphs.jsonl", cn_path,
                    resolve(cfg, "raw") / "visual_genome" / "attributes.json")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(json.dumps({k: manifest[k] for k in ("concepts", "edges", "term_scales", "seconds")}))


if __name__ == "__main__":
    main()
