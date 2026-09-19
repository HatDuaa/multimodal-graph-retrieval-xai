"""Can a caption be matched against the scene graph of its own image? Measured before any GNN is written.

The level 1 design parses the query into a small graph and matches it, triple by triple, against the
Visual Genome scene graph of every candidate image. That only works if (1) the parser gets triples out
of captions, (2) those triples can be found in the scene graph of the caption's own image, and (3) they
are not found in most other images too. This script measures the three, with exact label matching only
(no embeddings, no training), so the numbers are a LOWER bound on what soft matching can reach.

Two label settings: "vg150" keeps only the 150 object classes and 50 predicates of the usual scene graph
benchmark; "open_vocab" keeps every label after alias normalisation.

Uses all validation captions and a seeded random sample of train captions. The test split is never read:
this check drives design decisions, and those must not look at test (brief, section 6).

Writes experiments/checks/scene_graph_coverage.json and keeps the parsed captions in
data/processed/query_graphs_check_{train,val}.json.

Usage:  python scripts/checks/scene_graph_coverage.py [--train-sample 20000] [--workers 16]
"""
import argparse
import json
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.coverage import (canon_query, coverage, graph_stats, image_index, parser_stats,  # noqa: E402
                                pool_selectivity, summarize, top_misses)
from src.graph.parse_query import parse_many  # noqa: E402
from src.graph.scene_graph import Vocab, iter_scene_graphs, load_raw  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

SETTINGS = {"vg150": True, "open_vocab": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sample", type=int, default=20000, help="number of train captions to parse")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    started = time.time()
    cfg = load_config()
    vocab = Vocab.from_config(cfg)
    raw = load_raw(cfg)

    report = {"date": time.strftime("%Y-%m-%d"),
              "config": {"seed": cfg["seed"], "train_sample": args.train_sample, "matching": "exact label match",
                         "parser": f"SceneGraphParser {version('SceneGraphParser')}",
                         "spacy": f"{version('spacy')} / en_core_web_sm {version('en_core_web_sm')}"},
              "splits": {}}

    for split in ("train", "val"):
        path = resolve(cfg, "splits") / f"vg_coco_{split}.json"
        if not path.exists():
            sys.exit(f"missing {path}; run: python -m src.data.build_split")
        images = json.loads(path.read_text(encoding="utf-8"))["images"]
        captions = [(c["caption_id"], c["text"], im["image_id"]) for im in images for c in im["captions"]]
        if split == "train":
            captions = random.Random(cfg["seed"]).sample(captions, min(args.train_sample, len(captions)))

        queries = parse_many([text for _, text, _ in captions], args.workers)
        golds = [image_id for _, _, image_id in captions]
        kept = resolve(cfg, "processed") / f"query_graphs_check_{split}.json"
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps({cid: q for (cid, _, _), q in zip(captions, queries)}), encoding="utf-8")

        entry = {"parser": parser_stats(queries), "settings": {}}
        for setting, vg150_only in SETTINGS.items():
            graphs = list(iter_scene_graphs(raw, vocab, [im["image_id"] for im in images], vg150_only))
            indexes = {g["image_id"]: image_index(g) for g in graphs}
            canon = [canon_query(q, vocab) for q in queries]
            results = [coverage(cq, indexes[gold]) for cq, gold in zip(canon, golds)]
            entry["settings"][setting] = {"scene_graphs": graph_stats(graphs), "gold_image_coverage": summarize(results)}
            if split == "val":   # the val images are exactly the retrieval pool of the val queries
                entry["settings"][setting]["pool_selectivity"] = pool_selectivity(canon, golds, indexes)
                entry["settings"][setting]["top_misses"] = top_misses(queries, canon, results)
        report["splits"][split] = entry
        print(split, json.dumps({k: v["gold_image_coverage"] for k, v in entry["settings"].items()}, indent=1))

    report["runtime_seconds"] = round(time.time() - started, 1)
    out = REPO_ROOT / "experiments" / "checks" / "scene_graph_coverage.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
