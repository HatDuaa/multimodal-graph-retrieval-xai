"""Row 32 of the task sheet: run the level-2 method on the Vietnamese MMKG and analyse transfer.

Method transferred from MKG-W (scripts/eval_mkgw_level2.py, row "kg_triple"):
CLIP top-K -> entity linking in the (Vietnamese) query -> number of direct KG edges
between the candidate image's landmark and the linked entities -> fused score
alpha * z(CLIP) + (1 - alpha) * z(triple). The explanation of a candidate is the list of
those edges (MkgwGraphScorer.score()["matches"]), produced by the ranking itself.
NativE's embedding channel is not transferred: it needs a NativE model trained on this
150-entity graph, which is out of scope; the structural channel is what transfers.

Protocol (vietnamese-mmkg/DATA_CARD.md, section 8):
  A0 zero-shot (main): pool = every reviewed image (val + test), queries = all automatic
     queries + the locked hand-written set. Nothing is chosen here: the encoder comes
     from row 9 (MKG-W val, experiments/checks/multilingual_clip_probe.json) and alpha
     from MKG-W val (experiments/level2/val/choice.json, row kg_triple).
  A1 alpha adaptation (secondary): alpha chosen on vnmmkg val (pool = val), test run once.
The triples are curated knowledge (Wikidata, infobox, two fixed rules), not
co-occurrence statistics, so the graph uses all of them (data card section 8).

Deterministic (frozen encoder, no training): one run; std = 0 is reported as such.
Output: experiments/level3/vnmmkg/<encoder>/{a0,a1}_*.json

Usage:  python scripts/eval_vnmmkg_transfer.py --encoder mclip_b32 [--alpha-a0 0.6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.checks.multilingual_clip_probe import MCLIP_TEXT, OPEN_CLIP  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder  # noqa: E402
from src.retrieval.mkgw_graph_score import EntityLinker, MkgwGraphScorer, zscore  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

ALPHAS = [round(a * 0.05, 2) for a in range(21)]
KS = (1, 5, 10)
VN = REPO_ROOT / "vietnamese-mmkg"


def encode(encoder: str, image_paths: list[Path], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if encoder == "mclip_b32":
        from sentence_transformers import SentenceTransformer
        img = ClipEncoder(*OPEN_CLIP["openai_b32"], batch_size=32).encode_images(image_paths)
        txt = SentenceTransformer(MCLIP_TEXT).encode(texts, batch_size=64, convert_to_numpy=True,
                                                      normalize_embeddings=True).astype(np.float32)
        return img, txt
    enc = ClipEncoder(*OPEN_CLIP[encoder], batch_size=32)
    return enc.encode_images(image_paths), enc.encode_texts(texts)


def load_kg():
    entities = [json.loads(l) for l in (VN / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    triples = [json.loads(l) for l in (VN / "triples.jsonl").read_text(encoding="utf-8").splitlines()]
    names = {e["qid"]: [e.get("label_vi")] + list(e.get("aliases_vi") or []) for e in entities}
    linker = EntityLinker(names, max_ngram=12)
    adj = [{"h_qid": t["h"], "t_qid": t["t"], "r_pid": t["r"], "kgc_split": "train"} for t in triples]
    scorer = MkgwGraphScorer(np.zeros((0, 1), np.float32), {}, adj)
    return entities, linker, scorer


def run(pool: list[dict], queries: list[dict], img_feats: dict, q_feats: np.ndarray, linker, scorer, k: int):
    """CLIP top-k over `pool` and the triple channel for each query; returns what fusion needs."""
    pool_ids = [im["image_id"] for im in pool]
    qid_of = {im["image_id"]: im["qid"] for im in pool}
    P = np.stack([img_feats[i] for i in pool_ids])
    sims = q_feats @ P.T
    k = min(k, len(pool_ids))
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    clip = np.take_along_axis(sims, order, axis=1)
    ranked = [[pool_ids[j] for j in row] for row in order]
    images_of = {}
    for im in pool:
        images_of.setdefault(im["qid"], []).append(im["image_id"])
    golds = [[i for g in q["gold_qids"] for i in images_of.get(g, [])] for q in queries]
    tri = np.zeros_like(clip)
    explanations = []
    for qi, q in enumerate(queries):
        linked = linker.link(q["text"])
        row = []
        for ci, img in enumerate(ranked[qi]):
            s = scorer.score(linked, qid_of[img])
            tri[qi, ci] = s["triple"]
            row.append(s["matches"])
        explanations.append({"linked": linked, "matches": dict(zip(ranked[qi], row))})
    return clip, tri, ranked, golds, explanations


def fused(clip, tri, ranked, alpha):
    out = []
    for i in range(len(ranked)):
        f = alpha * zscore(clip[i]) + (1 - alpha) * zscore(tri[i])
        out.append([ranked[i][j] for j in np.argsort(-f, kind="stable")])
    return out


def by_group(rank_lists, golds, queries):
    res = {"all": evaluate(rank_lists, golds, ks=KS)}
    for grp in sorted({q["group"] for q in queries}):
        idx = [i for i, q in enumerate(queries) if q["group"] == grp]
        res[grp] = evaluate([rank_lists[i] for i in idx], [golds[i] for i in idx], ks=KS)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["openai_b32", "xlmr_b32", "mclip_b32", "xlmr_h14"])
    ap.add_argument("--alpha-a0", type=float, default=None,
                    help="default: kg_triple alpha of experiments/level2/val/choice.json (seed 0)")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config()
    if args.alpha_a0 is None:
        choice = json.loads((resolve(cfg, "experiments") / "level2" / "val" / "choice.json").read_text())
        args.alpha_a0 = choice["0"]["kg_triple"][0]
    splits = {s: json.loads((resolve(cfg, "splits") / f"vnmmkg_{s}.json").read_text(encoding="utf-8"))
              for s in ("val", "test")}
    hand = json.loads((VN / "queries_hand.json").read_text(encoding="utf-8"))
    for q in hand:
        q.setdefault("group", "hand")
    auto = {s: [{"query_id": q["query_id"], "text": q["text"], "gold_qids": [q["gold_qid"]], "group": "auto"}
                for q in d["queries"]] for s, d in splits.items()}

    pool_all = splits["val"]["images"] + splits["test"]["images"]
    img_dir = REPO_ROOT / "data" / "raw" / "vnmmkg" / "images"
    queries_all = auto["val"] + auto["test"] + [{**q, "group": f"hand_{q['group']}"} for q in hand]
    img, txt = encode(args.encoder, [img_dir / im["file_name"] for im in pool_all],
                      [q["text"] for q in queries_all])
    img_feats = {im["image_id"]: img[i] for i, im in enumerate(pool_all)}
    q_index = {q["query_id"]: i for i, q in enumerate(queries_all)}
    _, linker, scorer = load_kg()
    out_dir = resolve(cfg, "experiments") / "level3" / "vnmmkg" / args.encoder

    # A0: zero-shot on the whole set
    clip, tri, ranked, golds, expl = run(pool_all, queries_all, img_feats, txt, linker, scorer, args.top_k)
    a0 = {"encoder": args.encoder, "alpha": args.alpha_a0, "pool": len(pool_all), "top_k": args.top_k,
          "linked_pct": round(100 * np.mean([bool(e["linked"]) for e in expl]), 2),
          "clip": by_group(ranked, golds, queries_all),
          "clip+kg_triple": by_group(fused(clip, tri, ranked, args.alpha_a0), golds, queries_all)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a0_metrics.json").write_text(json.dumps(a0, indent=1), encoding="utf-8")
    # explanation of the image the fused ranking returns first: the KG edges that were counted for it
    a0_ranked = fused(clip, tri, ranked, args.alpha_a0)
    (out_dir / "a0_explanations.json").write_text(json.dumps(
        [{"query_id": q["query_id"], "group": q["group"], "text": q["text"], "linked": e["linked"],
          "top1_image": a0_ranked[i][0], "top1_is_gold": a0_ranked[i][0] in golds[i],
          "top1_matches": e["matches"][a0_ranked[i][0]]}
         for i, (q, e) in enumerate(zip(queries_all, expl))], ensure_ascii=False, indent=1), encoding="utf-8")
    print("A0", {m: {g: round(100 * r["recall@1"], 2) for g, r in v.items()}
                 for m, v in a0.items() if isinstance(v, dict)})

    # A1: alpha chosen on val (pool = val), test once (pool = test). Hand queries stay out.
    res = {}
    for s in ("val", "test"):
        qs = auto[s]
        res[s] = run(splits[s]["images"], qs, img_feats, txt[[q_index[q["query_id"]] for q in qs]],
                     linker, scorer, args.top_k)
    c, t, r, g, _ = res["val"]
    sweep = {a: evaluate(fused(c, t, r, a), g, ks=KS) for a in ALPHAS}
    a_best = max(sweep, key=lambda a: (sweep[a]["recall@1"], sweep[a]["mrr"]))
    c, t, r, g, _ = res["test"]
    a1 = {"encoder": args.encoder, "alpha_val": a_best, "val_sweep_r1": {a: m["recall@1"] for a, m in sweep.items()},
          "test": {"clip": evaluate(r, g, ks=KS), "clip+kg_triple": evaluate(fused(c, t, r, a_best), g, ks=KS),
                   "clip+kg_triple@alpha_a0": evaluate(fused(c, t, r, args.alpha_a0), g, ks=KS)}}
    (out_dir / "a1_metrics.json").write_text(json.dumps(a1, indent=1), encoding="utf-8")
    print("A1 alpha", a_best, {m: round(100 * v["recall@1"], 2) for m, v in a1["test"].items()})
    print("->", out_dir)


if __name__ == "__main__":
    main()
