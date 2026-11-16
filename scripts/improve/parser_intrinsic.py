"""Intrinsic comparison of caption parsers on val (part 1b): no ranking, no training.

For each parser's val query graphs (data/processed/parsers/query_graphs_val_<parser>.json):
- the query-side graph the ranking sees (GraphStore._graph_parts(query=True): pronoun objects and every relation
  touching them dropped, empty labels hidden): share of captions with >= 1 triple, mean objects and triples;
- raw parser nodes that are pronouns, empty, or not a Visual Genome object label of any train image graph (OOV);
- coverage of the query-side graph by the gold image's scene graph, exact labels after VG alias normalisation
  (src/graph/coverage.py, open vocabulary, as scripts/checks/scene_graph_coverage.py);
- parse time per 1,000 captions (timing_val_<parser>.json);
- 20 random val captions (seed 0) side by side, written to side_by_side.md.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.coverage import canon_query, coverage, image_index, summarize  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "parser-compare"


def query_side(graph):
    """The query graph exactly as the ranking sees it, in the plain schema used by coverage()."""
    parts = GraphStore._graph_parts(graph, query=True)
    kept = [i for i, keep in zip(parts["node_indices"], parts["part_mask"]) if keep]
    remap = {old: new for new, old in enumerate(parts["node_indices"])}
    objects = [{"id": remap[i], "name": graph["objects"][i]["name"]} for i in parts["node_indices"]]
    relations = [{"subject": e[0], "predicate": e[2], "object": e[1]} for e in parts["edges"]]
    return {"objects": objects, "relations": relations}, len(kept), len(relations)


def describe(graph):
    names = [o["name"] + (f" ×{o['count']}" if o.get("count") else "") for o in graph["objects"]]
    rels = [f"{names[r['subject']]} —{r['predicate']}→ {names[r['object']]}" for r in graph["relations"]]
    return "; ".join(dict.fromkeys(names)) or "∅", "; ".join(rels) or "∅"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsers", nargs="+", default=["sng", "factual-base", "factual-large", "qwen2.5-3b"])
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config()
    vocab = Vocab.from_config(cfg)
    images = json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
    gold = {c["caption_id"]: im["image_id"] for im in images for c in im["captions"]}
    text = {c["caption_id"]: c["text"] for im in images for c in im["captions"]}
    train_labels, val_graphs = set(), {}
    with (resolve(cfg, "graphs") / "vg_coco_scene_graphs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            graph = json.loads(line)
            if graph["split"] == "train":
                train_labels.update(vocab.canon_object(o["name"]) for o in graph["objects"])
            elif graph["split"] == "val":
                val_graphs[graph["image_id"]] = graph
    indexes = {image_id: image_index(g) for image_id, g in val_graphs.items()}
    parsed = {p: json.loads((resolve(cfg, "processed") / "parsers" / f"query_graphs_val_{p}.json").read_text(encoding="utf-8"))
              for p in args.parsers}
    caption_ids = list(parsed[args.parsers[0]])
    report = {"split": "val", "captions": len(caption_ids), "train_image_labels": len(train_labels), "parsers": {}}
    for name, graphs in parsed.items():
        assert set(graphs) == set(caption_ids), name
        sides = [query_side(graphs[c]) for c in caption_ids]
        raw_nodes = [o["name"] for c in caption_ids for o in graphs[c]["objects"]]
        pronoun = sum(n in PRONOUNS for n in raw_nodes)
        empty = sum(not n.strip() for n in raw_nodes)
        oov = sum(bool(n.strip()) and n not in PRONOUNS and vocab.canon_object(n) not in train_labels for n in raw_nodes)
        raw_relations = [len(graphs[c]["relations"]) for c in caption_ids]
        results = [coverage(canon_query(side, vocab), indexes[gold[c]]) for c, (side, _, _) in zip(caption_ids, sides)]
        timing_path = OUT / f"timing_val_{name}.json"
        timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
        report["parsers"][name] = {
            "share_with_triple": mean(t > 0 for _, _, t in sides),
            "objects_mean": mean(n for _, n, _ in sides), "triples_mean": mean(t for _, _, t in sides),
            "share_raw_relations_but_no_triple": mean(r > 0 and t == 0 for r, (_, _, t) in zip(raw_relations, sides)),
            "relation_drop_share": 1 - sum(t for _, _, t in sides) / max(1, sum(raw_relations)),
            "raw_nodes": len(raw_nodes), "pronoun_node_share": pronoun / len(raw_nodes),
            "empty_node_share": empty / len(raw_nodes), "oov_node_share": oov / len(raw_nodes),
            "junk_node_share": (pronoun + empty + oov) / len(raw_nodes),
            "coverage": summarize(results),
            "seconds_per_1000": timing.get("seconds_per_1000"), "device": timing.get("device"),
            "failed_outputs": timing.get("failed_outputs", 0)}
        print(name, json.dumps({k: v for k, v in report["parsers"][name].items() if k != "coverage"}), flush=True)
    (OUT / "intrinsic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    sample = random.Random(0).sample(caption_ids, args.examples)
    lines = ["# 20 câu val ngẫu nhiên qua các parser (seed 0)", "",
             "Đồ thị phía câu đúng như parser trả ra (trước khi lọc đại từ). Dạng: vật thể; quan hệ chủ ngữ —vị từ→ tân ngữ.", ""]
    for k, cid in enumerate(sample, 1):
        lines += [f"## {k}. `{cid}` — {text[cid]}", "", "| Parser | Vật thể | Quan hệ |", "|---|---|---|"]
        for name, graphs in parsed.items():
            nodes, rels = describe(graphs[cid])
            lines.append(f"| {name} | {nodes} | {rels} |".replace("\n", " "))
        lines.append("")
    (OUT / "side_by_side.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
