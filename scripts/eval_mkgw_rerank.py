"""CLIP baseline and graph re-ranking on the MKG-W retrieval splits (level 2).

Two evaluations on one split (default: val — test is reserved for the one-shot
final run after the method is locked):

  clip      frozen CLIP, cosine ranking over the split's own image pool
  rerank    top-K candidates re-scored with
            alpha * z(CLIP) + (1-alpha) * z(beta * z(emb) + (1-beta) * z(triple))
            emb    = best NativE-embedding cosine between candidate and entities
                     linked in the query (EntityLinker)
            triple = number of direct KGC-train edges candidate <-> linked entities
            alpha, beta are swept and reported per value; choosing them is only
            allowed on val (assignment section 6).

Deterministic (frozen encoders, no sampling): a single run, std = 0.
Results: experiments/level2/<name>/metrics_<split>.json via write_metrics().

Usage:  python scripts/eval_mkgw_rerank.py [--split val] [--native-feats mkgw_native_ent]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import evaluate, write_metrics  # noqa: E402
from src.features.extract_clip import load_features  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.retrieval.mkgw_graph_score import (  # noqa: E402
    EntityLinker, MkgwGraphScorer, load_train_triples, zscore)
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

ALPHAS = [round(a * 0.1, 1) for a in range(11)]
BETAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--native-feats", default="mkgw_native_ent")
    ap.add_argument("--top-k", type=int, default=None, help="default: retrieval.top_k from config")
    args = ap.parse_args()
    if args.split == "test":
        print("WARNING: test is the one-shot split; run this only for the locked final method.")

    cfg = load_config()
    k = args.top_k or cfg["retrieval"]["top_k"]
    feat_dir, split_dir = resolve(cfg, "features"), resolve(cfg, "splits")
    images = json.loads((split_dir / f"mkgw_{args.split}.json").read_text(encoding="utf-8"))["images"]
    qid_of_image = {im["image_id"]: im["qid"] for im in images}

    img_feats, img_ids, encoder = load_features(feat_dir / "mkgw_image")
    cap_feats, cap_ids, _ = load_features(feat_dir / "mkgw_caption")
    pool_ids = [im["image_id"] for im in images]
    pool_rows = {i: r for r, i in enumerate(img_ids)}
    pool = np.stack([img_feats[pool_rows[i]] for i in pool_ids])
    cap_rows = {c: r for r, c in enumerate(cap_ids)}
    queries = [im["captions"][0] for im in images]
    q_feats = np.stack([cap_feats[cap_rows[q["caption_id"]]] for q in queries])
    golds = [im["image_id"] for im in images]

    index = CosineIndex(pool, pool_ids)
    scores, ranked = index.search(q_feats, k)
    clip_metrics = evaluate([list(r) for r in ranked], golds, ks=(1, 5, 10))
    out_dir = resolve(cfg, "experiments") / "level2"
    write_metrics(out_dir / "clip_baseline" / f"metrics_{args.split}.json",
                  method="clip", dataset="mkgw", split=args.split,
                  per_seed={0: clip_metrics},
                  config={"encoder": encoder, "pool": len(pool_ids), "top_k": k})
    print("CLIP:", {m: round(v, 4) if isinstance(v, float) else v for m, v in clip_metrics.items()})

    # graph channels
    raw = REPO_ROOT / "data" / "raw" / "mkgw"
    cache_rows = [json.loads(l) for l in (raw / "wikidata_entities.jsonl").read_text(encoding="utf-8").splitlines()]
    id2qid = {r["id"]: r["qid"] for r in cache_rows}
    rels = json.loads((raw / "wikidata_relations.json").read_text(encoding="utf-8"))
    linker = EntityLinker.from_cache(raw / "wikidata_entities.jsonl", lang=cfg["mkgw"]["query_lang"])
    nat_feats, nat_qids, nat_encoder = load_features(feat_dir / args.native_feats)
    scorer = MkgwGraphScorer(nat_feats, {q: i for i, q in enumerate(nat_qids)},
                             load_train_triples(REPO_ROOT / "data" / "processed" / "mkgw_triples.jsonl",
                                                id2qid, {v["id"]: pid for pid, v in rels.items()}),
                             {pid: v["label"] or pid for pid, v in rels.items()})

    emb_ch = np.zeros((len(queries), k), dtype=np.float32)
    tri_ch = np.zeros((len(queries), k), dtype=np.float32)
    for qi, (im, query) in enumerate(zip(images, queries)):
        linked = linker.link(query["text"])          # the answer is unknown at inference: no exclusions
        for ci, cand_image_id in enumerate(ranked[qi]):
            s = scorer.score(linked, qid_of_image[cand_image_id])
            emb_ch[qi, ci] = 0.0 if s["emb"] is None else s["emb"]
            tri_ch[qi, ci] = s["triple"]

    sweep = {}
    for beta in BETAS:
        for alpha in ALPHAS:
            fused_ranked = []
            for qi in range(len(queries)):
                graph = beta * zscore(emb_ch[qi]) + (1.0 - beta) * zscore(tri_ch[qi])
                fused = alpha * zscore(scores[qi]) + (1.0 - alpha) * zscore(graph)
                order = np.argsort(-fused, kind="stable")
                fused_ranked.append([ranked[qi][j] for j in order])
            m = evaluate(fused_ranked, golds, ks=(1, 5, 10))
            sweep[f"alpha={alpha},beta={beta}"] = m
    best_key = max(sweep, key=lambda kk: (sweep[kk]["recall@1"], sweep[kk]["mrr"]))
    write_metrics(out_dir / "rerank_sweep" / f"metrics_{args.split}.json",
                  method=f"clip+graph[{args.native_feats}]", dataset="mkgw", split=args.split,
                  per_seed={0: sweep[best_key]},
                  config={"encoder": encoder, "native": nat_encoder, "top_k": k,
                          "best": best_key, "sweep": sweep})
    print("best on", args.split, "->", best_key, {m: round(v, 4) if isinstance(v, float) else v
                                                  for m, v in sweep[best_key].items()})
    for probe in ("alpha=1.0,beta=0.0", "alpha=0.6,beta=0.0", "alpha=0.6,beta=1.0", "alpha=0.5,beta=0.5"):
        print(f"  {probe}:", round(sweep[probe]["recall@1"], 4))


if __name__ == "__main__":
    main()
