"""Does a graph score help CLIP at all, before anything is trained?

scene_graph_coverage.py showed that exact label matching of caption triples covers too few queries. This
probe replaces exact matching by cosine between frozen CLIP TEXT vectors of the parts (object labels, and
triples written as phrases such as "boy holding ball"), then re-ranks the CLIP top-k of every VALIDATION
query with  alpha * z(CLIP) + (1 - alpha) * z(graph)  and sweeps alpha. Nothing is trained, no GNN.

Variants of the graph score:
  node             object labels only (no relations): what a bag of objects can do
  triple_max       each query triple takes its best matching image triple
  triple_softmax_T image triples weighted by softmax(similarity / T)
  node+triple_max  mean of z(node) and z(triple_max)
  *_shuffled       control: every image gets the scene graph of another image; must not beat CLIP

Validation only (brief, section 6). Scene graphs use open-vocabulary labels (see experiments/checks/README.md,
section 3). Writes experiments/checks/soft_graph_rerank_probe.json and keeps the text vectors of all parts in
data/features/vg_coco_val_graph_parts.npy.

Usage:  python scripts/checks/soft_graph_rerank_probe.py [--workers 16]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder, save_features  # noqa: E402
from src.graph.parse_query import parse_many  # noqa: E402
from src.graph.scene_graph import Vocab, iter_scene_graphs, load_raw  # noqa: E402
from src.graph.soft_match import image_parts, match, query_parts, rerank, zscore_rows  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

ALPHAS = [round(0.1 * i, 1) for i in range(11)]
TEMPERATURES = (0.02, 0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    started = time.time()
    cfg = load_config()
    top_k = cfg["retrieval"]["top_k"]

    top_path = REPO_ROOT / cfg["paths"]["experiments"] / "level1" / "clip_baseline" / f"top{top_k}_val.json"
    if not top_path.exists():
        sys.exit(f"missing {top_path}; run: python -m src.retrieval.clip_baseline --splits val")
    rows = json.loads(top_path.read_text(encoding="utf-8"))
    images = json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
    text_of = {c["caption_id"]: c["text"] for im in images for c in im["captions"]}
    pool = [im["image_id"] for im in images]

    queries = [query_parts(q) for q in parse_many([text_of[r["caption_id"]] for r in rows], args.workers)]
    graphs = {g["image_id"]: image_parts(g) for g in iter_scene_graphs(load_raw(cfg), Vocab.from_config(cfg), pool, False)}
    shuffled_ids = pool[:]
    random.Random(cfg["seed"]).shuffle(shuffled_ids)

    # one text vector per distinct part; object labels use the concept prompt, triples are plain phrases
    prompt = cfg["clip"]["concept_prompt"]
    node_texts = sorted({n for nodes, _ in queries for n in nodes} | {n for nodes, _ in graphs.values() for n in nodes})
    triple_texts = sorted({t for _, ts in queries for t in ts} | {t for _, ts in graphs.values() for t in ts})
    encoder = ClipEncoder.from_config(cfg)
    vecs = encoder.encode_texts([prompt.format(n) for n in node_texts] + triple_texts)
    keys = [f"node:{n}" for n in node_texts] + [f"triple:{t}" for t in triple_texts]
    save_features(resolve(cfg, "features") / "vg_coco_val_graph_parts", vecs, keys, f"{cfg['clip']['model']}/{cfg['clip']['pretrained']}")
    row_of = {k: i for i, k in enumerate(keys)}

    def lookup(kind: str, parts: list[str]) -> np.ndarray:
        return vecs[[row_of[f"{kind}:{p}"] for p in parts]]

    q_vecs = [(lookup("node", n), lookup("triple", t)) for n, t in queries]
    g_vecs = {i: (lookup("node", n), lookup("triple", t)) for i, (n, t) in graphs.items()}
    s_vecs = {i: g_vecs[j] for i, j in zip(pool, shuffled_ids)}

    variants = {"node": (0, "max", None), "triple_max": (1, "max", None)}
    variants.update({f"triple_softmax_{t}": (1, "softmax", t) for t in TEMPERATURES})
    shape = (len(rows), top_k)
    scores = {name: np.full(shape, np.nan) for name in variants}
    scores.update({f"{name}_shuffled": np.full(shape, np.nan) for name in ("node", "triple_max")})
    for qi, (row, qv) in enumerate(zip(rows, q_vecs)):
        for ci, image_id in enumerate(row["ranked"]):
            for name, (level, mode, temp) in variants.items():
                scores[name][qi, ci] = match(qv[level], g_vecs[image_id][level], mode, temp or 1.0)[0]
            for name in ("node", "triple_max"):
                level = variants[name][0]
                scores[f"{name}_shuffled"][qi, ci] = match(qv[level], s_vecs[image_id][level], "max")[0]
    scores["node+triple_max"] = (zscore_rows(scores["node"]) + zscore_rows(scores["triple_max"])) / 2
    scores["node+triple_max_shuffled"] = (zscore_rows(scores["node_shuffled"]) + zscore_rows(scores["triple_max_shuffled"])) / 2

    clip = np.array([r["scores"] for r in rows])
    ranked = [r["ranked"] for r in rows]
    golds = [r["gold"] for r in rows]
    report = {"date": time.strftime("%Y-%m-%d"), "split": "val", "queries": len(rows), "pool": len(pool), "top_k": top_k,
              "config": {"encoder": f"{cfg['clip']['model']}/{cfg['clip']['pretrained']}", "labels": "open_vocab",
                         "node_prompt": prompt, "normalisation": "z-score inside each query's top-k, missing graph score = 0",
                         "seed_for_shuffle": cfg["seed"]},
              "parts": {"node_texts": len(node_texts), "triple_texts": len(triple_texts),
                        "queries_without_triple": sum(len(t) == 0 for _, t in queries),
                        "images_without_triple": sum(len(t) == 0 for _, t in graphs.values())},
              "clip_only": evaluate(ranked, golds), "variants": {}}
    for name, graph_scores in scores.items():
        sweep = {str(a): evaluate(rerank(clip, graph_scores, ranked, a), golds) for a in ALPHAS}
        best = max(sweep, key=lambda a: (sweep[a]["recall@1"], sweep[a]["mrr"]))
        report["variants"][name] = {"best_alpha_by_recall@1": float(best), "best": sweep[best], "sweep": sweep}
        print(f"{name:28s} best alpha {best}: R@1 {100 * sweep[best]['recall@1']:.2f}  R@5 {100 * sweep[best]['recall@5']:.2f}  "
              f"R@10 {100 * sweep[best]['recall@10']:.2f}  MRR {100 * sweep[best]['mrr']:.2f}")
    print(f"{'clip_only':28s} R@1 {100 * report['clip_only']['recall@1']:.2f}")

    report["runtime_seconds"] = round(time.time() - started, 1)
    out = REPO_ROOT / "experiments" / "checks" / "soft_graph_rerank_probe.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
