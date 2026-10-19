"""Val checks of image-graph cleaning (audit, no training, no ranking): consistency and plural/count coverage.

For every cleaned variant of graph_dedup_audit.py:
- shared-node consistency of step 2 on the val captions (FACTUAL, data version factual_v1) with >= 1 pair of triples
  sharing a node: gold image vs the other 49 candidates; frozen CLIP phrase vectors, each cleaned edge keeping the
  vector of the packaged relation it comes from;
- caption nodes with a stated count (>= 2) or a plural head in the FACTUAL output: does the gold image have that many
  matching things (label equal after VG aliases, or CLIP cosine >= 0.955 as in the geometry channel), and, after
  grouping, a matching group node of that size.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.graph_dedup_audit import load_split, variants  # noqa: E402
from scripts.improve.graph_matching_probe import frozen_vectors, image_structure, query_structure, structures  # noqa: E402
from src.graph.dedup import with_boxes  # noqa: E402
from src.graph.graph_matching import consistency_score, shared_node_pairs  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "graph-dedup-audit"
THRESHOLD = .955


def plural_heads(raw_text):
    """FACTUAL entity heads in output order (the order of the converted caption graph objects)."""
    from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
    parser = object.__new__(SceneGraphParser)
    return [e["head"] for e in parser.graph_string_to_object(raw_text)["entities"]]


def main() -> None:
    cfg = load_config(Path("configs/factual_v1.yaml"))
    base_cfg = load_config()
    graphs, objects = load_split(base_cfg, "val")
    cleaned = {i: variants(with_boxes(g, objects[str(i)]["boxes"], objects[str(i)]["synsets"])) for i, g in graphs.items()}
    names = list(next(iter(cleaned.values())))
    raw_queries, raw_images = structures(cfg)
    candidates = np.load(resolve(cfg, "processed") / "candidates_val.npz")
    caption_ids = [str(c) for c in candidates["caption_ids"]]
    golds = [int(g) for g in candidates["gold"]]
    queries = [query_structure(raw_queries[c]) for c in caption_ids]
    images = {i: image_structure(g) for i, g in raw_images.items()}
    _, q_triples, _, i_triples = frozen_vectors(cfg, queries, images)
    result = {"split": "val", "variants": names, "consistency": {}, "counts": {}}
    subset = [k for k, q in enumerate(queries) if shared_node_pairs(q["edges"])]
    for name in names:
        gold_scores, other_scores, gold_full, other_full = [], [], [], []
        for k in subset:
            pairs = shared_node_pairs(queries[k]["edges"])
            row = []
            for image_id in candidates["candidate_ids"][k]:
                g = cleaned[int(image_id)][name]
                edges = [(r["subject"], r["object"]) for r in g["relations"]]
                vectors = i_triples[int(image_id)][[r.get("source", n) for n, r in enumerate(g["relations"])]] \
                    if g["relations"] else i_triples[int(image_id)][:0]
                row.append(consistency_score(pairs, q_triples[k], vectors, edges))
            row = np.asarray(row)
            gold = candidates["candidate_ids"][k] == golds[k]
            if gold.any() and np.isfinite(row[gold]).all() and np.isfinite(row[~gold]).any():
                gold_scores.append(float(row[gold][0]))
                other_scores.append(float(np.nanmean(row[~gold])))
                gold_full.append(float(row[gold][0] == 1.))
                other_full.append(float(np.nanmean(row[~gold] == 1.)))
        result["consistency"][name] = {"captions": len(gold_scores), "gold_mean": float(np.mean(gold_scores)),
                                       "others_mean": float(np.mean(other_scores)),
                                       "gold_minus_others": float(np.mean(gold_scores) - np.mean(other_scores)),
                                       "gold_above_others_mean": float(np.mean(np.asarray(gold_scores) > np.asarray(other_scores))),
                                       "gold_fully_consistent": float(np.mean(gold_full)),
                                       "others_fully_consistent": float(np.mean(other_full))}
        print(name, json.dumps({k: round(v, 4) for k, v in result["consistency"][name].items()}), flush=True)
    vocab = Vocab.from_config(base_cfg)
    from src.features.extract_clip import load_features
    table, keys, _ = load_features(resolve(cfg, "features") / "vg_coco_graph_parts")
    row_of = {k: i for i, k in enumerate(keys)}
    raw_text = json.loads((resolve(base_cfg, "processed") / "parsers" / "raw_val_factual-large.json").read_text(encoding="utf-8"))
    cache = {}

    def match(q, i):
        if (q, i) not in cache:
            same = vocab.canon_object(q) == vocab.canon_object(i)
            cache[(q, i)] = same or float(table[row_of[f"node:{q}"]] @ table[row_of[f"node:{i}"]]) >= THRESHOLD
        return cache[(q, i)]

    items = []
    for k, cid in enumerate(caption_ids):
        graph = raw_queries[cid]
        heads = plural_heads(raw_text[cid])
        for obj, head in zip(graph["objects"], heads):
            name = obj["name"]
            if not name or name in PRONOUNS or f"node:{name}" not in row_of:
                continue
            plural = head.lower().split()[-1] != name.split()[-1] if head.strip() else False
            count = obj.get("count")
            if (count and count >= 2) or plural:
                items.append((k, name, count if count and count >= 2 else None))
    for name in names:
        stated, stated_hit_instances, stated_hit_group, plural_n, plural_enough, plural_group = 0, 0, 0, 0, 0, 0
        for k, label, count in items:
            g = cleaned[golds[k]][name]
            matched = [o for o in g["objects"] if match(label, o["name"])]
            instances = sum(o["count"] for o in matched) if "+" in name else len(matched)
            group = max([o["count"] for o in matched], default=0)
            if count:
                stated += 1
                stated_hit_instances += instances == count
                stated_hit_group += group == count
            else:
                plural_n += 1
                plural_enough += instances >= 2
                plural_group += group >= 2
        grouped = "+" in name  # group-node columns only mean something once nodes are grouped
        result["counts"][name] = {"stated_count_nodes": stated, "stated_count_match_instances": stated_hit_instances / max(stated, 1),
                                  "stated_count_match_largest_node": stated_hit_group / max(stated, 1) if grouped else None,
                                  "plural_nodes_without_count": plural_n, "plural_with_two_or_more": plural_enough / max(plural_n, 1),
                                  "plural_with_group_node": plural_group / max(plural_n, 1) if grouped else None}
        print(name, json.dumps(result["counts"][name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
