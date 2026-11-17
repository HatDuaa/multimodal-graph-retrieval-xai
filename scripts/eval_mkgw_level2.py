"""Level 2 evaluation on MKG-W: CLIP vs CLIP + graph, 3 seeds, ablations, query groups.

Extends Đạt's scripts/eval_mkgw_rerank.py (single checkpoint, val sweep) to the full
level-2 protocol. Graph channels are Đạt's MkgwGraphScorer (src/retrieval/mkgw_graph_score.py):

  triple  number of direct KGC-train edges between the candidate and the entities
          linked in the query (training-free, deterministic)
  emb     best cosine between the candidate's NativE structural embedding and a
          linked entity; NativE trained on the leakage-safe MKG-W-rmask copy,
          one embedding file per seed (mkgw_native_ent_seed{0,1,2})

  score = alpha * z(CLIP) + (1 - alpha) * z(beta * z(emb) + (1 - beta) * z(triple))
  over the CLIP top-K of each query (z-score within the query's list).

Protocol (assignment section 6):
  * alpha, beta are chosen on VAL, separately for every seed; test reuses them.
  * test is read only with --split test, once, after the method is locked.
  * the hand-written query set (data/queries/mkgw_queries_hand.json, pool = test
    images) is evaluation-only: it reuses the val choice and never tunes anything.

Rows written to experiments/level2/<split>/metrics_<row>.json (mean ± std over seeds):
  clip            frozen CLIP only
  kg_triple       CLIP + triple channel (beta = 0); alpha from val
  native          CLIP + emb + triple, alpha/beta from val per seed       (main level-2 row)
  native_emb      CLIP + emb only (beta = 1): the "remove edges" ablation of the triple channel
  shuffled_edges  native, but the triple channel counts edges of a KGC-train graph whose tails
                  are permuted within each relation (degree of heads kept), seed = run seed
  random_emb      native, but NativE vectors replaced by Gaussian noise, seed = run seed
New rows (17/11, after val chose beta = 0 for every seed, i.e. no weight for the cosine channel), with the channel

  kgc     NativE's own triple score (AdvRelRotatE.forward of the rmask checkpoint of the seed), max over
          linked entities e != candidate, the 169 relations and both directions (src/retrieval/native_kgc.py)

  native_kgc      CLIP + kgc + triple, alpha/beta from val per seed                (NEW main level-2 row)
  kgc_only        CLIP + kgc only (beta = 1): the "remove edges" ablation of native_kgc
  random_kgc      native_kgc's alpha/beta, but NativE's structural entity vectors replaced by Gaussian noise
                  (random_emb's rng recipe, rows rescaled to the trained norms), seed = run seed
The six rows above and their choice.json keys are computed exactly as before; the new keys sit next to them.
Also: per-group metrics (query has a linked entity or not), the alpha sweep of the
main row on val (sensitivity), and, with --hand, direct vs indirect hand queries.

Usage:  python scripts/eval_mkgw_level2.py --split val
        python scripts/eval_mkgw_level2.py --split test --hand      # once, at the end
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import aggregate_seeds, evaluate, write_metrics  # noqa: E402
from src.features.extract_clip import load_features  # noqa: E402
from src.retrieval.mkgw_graph_score import EntityLinker, MkgwGraphScorer, load_train_triples, zscore  # noqa: E402
from src.retrieval.native_kgc import NativeKgcScorer, default_device, random_entities  # noqa: E402
from src.retrieval.native_kgc import channel as kgc_channel  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

ALPHAS = [round(a * 0.05, 2) for a in range(21)]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = (0, 1, 2)
KS = (1, 5, 10)
OLD_CHOICE_KEYS = ("native", "kg_triple", "native_emb")   # choice.json keys of the six original rows


def shuffle_tails(triples: list[dict], seed: int) -> list[dict]:
    """Permute tails within each relation: every head keeps its out-degree per relation,
    every tail its in-degree, but which entity is linked to which is random."""
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for t in triples:
        by_rel[t["r_pid"]].append(t)
    out = []
    for pid in sorted(by_rel):
        ts = by_rel[pid]
        tails = [t["t_qid"] for t in ts]
        rng.shuffle(tails)
        out += [{**t, "t_qid": tail} for t, tail in zip(ts, tails)]
    return out


def fuse_rank(clip: np.ndarray, emb: np.ndarray, tri: np.ndarray, ranked: list, alpha: float, beta: float) -> list:
    graph = beta * zscore(emb) + (1.0 - beta) * zscore(tri)
    fused = alpha * zscore(clip) + (1.0 - alpha) * zscore(graph)
    order = np.argsort(-fused, kind="stable")
    return [ranked[j] for j in order]


def rerank_all(ch: dict, alpha: float, beta: float, emb_key: str, tri_key: str) -> list[list]:
    return [fuse_rank(ch["clip"][i], ch[emb_key][i], ch[tri_key][i], ch["ranked"][i], alpha, beta)
            for i in range(len(ch["ranked"]))]


def channels(scorer: MkgwGraphScorer, linked: list[dict], ranked: list[list], qid_of: dict, k: int):
    emb = np.zeros((len(ranked), k), dtype=np.float32)
    tri = np.zeros((len(ranked), k), dtype=np.float32)
    for qi, cands in enumerate(ranked):
        for ci, img in enumerate(cands):
            s = scorer.score(linked[qi], qid_of[img])
            emb[qi, ci] = 0.0 if s["emb"] is None else s["emb"]
            tri[qi, ci] = s["triple"]
    return emb, tri


def best_on_val(sweep: dict) -> str:
    return max(sweep, key=lambda key: (sweep[key]["recall@1"], sweep[key]["mrr"]))


def choose_on_val(ev, ch: dict) -> tuple[dict, dict, dict]:
    """alpha/beta of one seed, chosen on val by R@1 then MRR. The original keys come first and depend only on the
    emb and triple channels; with a kgc channel, native_kgc and kgc_only are chosen the same way on its own sweep."""
    sweep = {(a, b): ev(rerank_all(ch, a, b, "emb", "tri")) for b in BETAS for a in ALPHAS}
    a_n, b_n = best_on_val({f"{a},{b}": v for (a, b), v in sweep.items()}).split(",")
    tri_only = {a: sweep[(a, 0.0)] for a in ALPHAS}
    emb_only = {a: sweep[(a, 1.0)] for a in ALPHAS}
    a_t = max(tri_only, key=lambda a: (tri_only[a]["recall@1"], tri_only[a]["mrr"]))
    a_e = max(emb_only, key=lambda a: (emb_only[a]["recall@1"], emb_only[a]["mrr"]))
    choice = {"native": [float(a_n), float(b_n)], "kg_triple": [a_t, 0.0], "native_emb": [a_e, 1.0]}
    kgc_sweep = {}
    if "kgc" in ch:
        kgc_sweep = {(a, b): ev(rerank_all(ch, a, b, "kgc", "tri")) for b in BETAS for a in ALPHAS}
        a_k, b_k = best_on_val({f"{a},{b}": v for (a, b), v in kgc_sweep.items()}).split(",")
        kgc_only = {a: kgc_sweep[(a, 1.0)] for a in ALPHAS}
        a_o = max(kgc_only, key=lambda a: (kgc_only[a]["recall@1"], kgc_only[a]["mrr"]))
        choice.update({"native_kgc": [float(a_k), float(b_k)], "kgc_only": [a_o, 1.0]})
    return choice, sweep, kgc_sweep


def kgc_scorers(path: Path, seed: int, device: str) -> tuple[NativeKgcScorer, NativeKgcScorer]:
    """NativE's scorer for one seed's rmask checkpoint, and the same model with random structural entity vectors."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    rnd = random_entities(state["ent_embeddings.weight"].numpy(), seed)
    return NativeKgcScorer(state, device), NativeKgcScorer(state, device, ent_embeddings=rnd)


class Data:
    """Everything that does not depend on the seed: pools, CLIP top-K, links, KG."""

    def __init__(self, cfg: dict, split: str, k: int):
        feat_dir, split_dir = resolve(cfg, "features"), resolve(cfg, "splits")
        self.k = k
        self.images = json.loads((split_dir / f"mkgw_{split}.json").read_text(encoding="utf-8"))["images"]
        self.qid_of = {im["image_id"]: im["qid"] for im in self.images}
        self.image_of_qid = {im["qid"]: im["image_id"] for im in self.images}
        img_feats, img_ids, self.encoder = load_features(feat_dir / "mkgw_image")
        rows = {i: r for r, i in enumerate(img_ids)}
        self.pool_ids = [im["image_id"] for im in self.images]
        self.pool = np.stack([img_feats[rows[i]] for i in self.pool_ids])

        raw = REPO_ROOT / "data" / "raw" / "mkgw"
        cache = [json.loads(l) for l in (raw / "wikidata_entities.jsonl").read_text(encoding="utf-8").splitlines()]
        self.id2qid = {r["id"]: r["qid"] for r in cache}
        self.nid_of = {q: i for i, q in self.id2qid.items()}
        self.label = {r["qid"]: r.get("label_en") for r in cache}
        rels = json.loads((raw / "wikidata_relations.json").read_text(encoding="utf-8"))
        self.rel_label = {pid: v["label"] or pid for pid, v in rels.items()}
        self.train_triples = load_train_triples(REPO_ROOT / "data" / "processed" / "mkgw_triples.jsonl",
                                                self.id2qid, {v["id"]: pid for pid, v in rels.items()})
        self.linker = EntityLinker.from_cache(raw / "wikidata_entities.jsonl", lang=cfg["mkgw"]["query_lang"])

    def topk(self, q_feats: np.ndarray):
        sims = q_feats @ self.pool.T
        order = np.argsort(-sims, axis=1, kind="stable")[:, : self.k]
        clip = np.take_along_axis(sims, order, axis=1).astype(np.float32)
        ranked = [[self.pool_ids[j] for j in row] for row in order]
        return clip, ranked

    def scorer(self, ent_feats: np.ndarray, ent_qids: list, triples: list[dict]) -> MkgwGraphScorer:
        return MkgwGraphScorer(ent_feats, {q: i for i, q in enumerate(ent_qids)}, triples, self.rel_label)


def split_queries(data: Data, cfg: dict):
    feat_dir = resolve(cfg, "features")
    cap_feats, cap_ids, _ = load_features(feat_dir / "mkgw_caption")
    rows = {c: r for r, c in enumerate(cap_ids)}
    qs = [im["captions"][0] for im in data.images]
    feats = np.stack([cap_feats[rows[q["caption_id"]]] for q in qs])
    golds = [im["image_id"] for im in data.images]
    return [q["text"] for q in qs], feats, golds


def hand_queries(data: Data, cfg: dict):
    from src.features.extract_clip import ClipEncoder
    hand = json.loads((REPO_ROOT / "data" / "queries" / "mkgw_queries_hand.json").read_text(encoding="utf-8"))
    enc = ClipEncoder.from_config(cfg, device="cpu")
    feats = enc.encode_texts([q["text"] for q in hand])
    golds = [[data.image_of_qid[g] for g in q["gold_qids"]] for q in hand]
    return hand, feats, golds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--hand", action="store_true", help="also evaluate the hand-written queries (test pool)")
    ap.add_argument("--native-prefix", default="mkgw_native_ent_seed")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--kgc-checkpoint", default="checkpoint/rmask_seed{seed}_e1000.ckpt",
                    help="NativE checkpoint of each seed for the kgc channel, relative to mkgw.native_repo")
    ap.add_argument("--device", default=None, help="kgc channel device (default: cuda if available, else cpu)")
    args = ap.parse_args()
    if args.split == "test":
        print("NOTE: test split — alpha/beta come from experiments/level2/val/choice.json, nothing is tuned here.")

    cfg = load_config()
    k = args.top_k or cfg["retrieval"]["top_k"]
    out_root = resolve(cfg, "experiments") / "level2"
    data = Data(cfg, args.split, k)
    texts, q_feats, golds = split_queries(data, cfg)
    clip, ranked = data.topk(q_feats)
    linked = [data.linker.link(t) for t in texts]
    has_link = np.array([len(l) > 0 for l in linked])
    groups = {"linked": has_link, "no_link": ~has_link}

    def ev(rank_lists, mask=None):
        if mask is None:
            return evaluate(rank_lists, golds, ks=KS)
        idx = np.flatnonzero(mask)
        return evaluate([rank_lists[i] for i in idx], [golds[i] for i in idx], ks=KS)

    feat_dir = resolve(cfg, "features")
    seed_feats = {s: load_features(feat_dir / f"{args.native_prefix}{s}") for s in SEEDS}
    choice_path = out_root / "val" / "choice.json"
    choice = json.loads(choice_path.read_text()) if args.split == "test" else {}
    native_repo = Path(cfg["mkgw"]["native_repo"]).expanduser()
    device = args.device or default_device()
    kgc_paths = {s: native_repo / args.kgc_checkpoint.format(seed=s) for s in SEEDS}

    per_row: dict[str, dict[int, dict]] = defaultdict(dict)
    per_group: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    sweeps, kgc_sweeps = {}, {}
    for s in SEEDS:
        feats, qids, native_name = seed_feats[s]
        scorer = data.scorer(feats, qids, data.train_triples)
        emb, tri = channels(scorer, linked, ranked, data.qid_of, k)
        ch = {"clip": clip, "ranked": ranked, "emb": emb, "tri": tri}

        # shuffled KG edges (triple channel only; emb stays from the real model)
        sh_scorer = data.scorer(feats, qids, shuffle_tails(data.train_triples, s))
        _, ch["tri_shuffled"] = channels(sh_scorer, linked, ranked, data.qid_of, k)
        # random entity vectors (emb channel only)
        rnd = np.random.default_rng(s).standard_normal(feats.shape).astype(np.float32)
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        ch["emb_random"], _ = channels(data.scorer(rnd, qids, data.train_triples), linked, ranked, data.qid_of, k)
        # NativE's own triple score (kgc), and the same with random structural entity vectors
        kgc, kgc_rnd = kgc_scorers(kgc_paths[s], s, device)
        ch["kgc"] = kgc_channel(kgc, linked, ranked, data.qid_of, data.nid_of, k)
        ch["kgc_random"] = kgc_channel(kgc_rnd, linked, ranked, data.qid_of, data.nid_of, k)

        if args.split == "val":
            choice[str(s)], sweep, kgc_sweep = choose_on_val(ev, ch)
            sweeps[s] = {f"alpha={a},beta={b}": v["recall@1"] for (a, b), v in sweep.items()}
            kgc_sweeps[s] = {f"alpha={a},beta={b}": v["recall@1"] for (a, b), v in kgc_sweep.items()}
        c = choice[str(s)]
        rows = {
            "kg_triple": rerank_all(ch, *c["kg_triple"], "emb", "tri"),
            "native": rerank_all(ch, *c["native"], "emb", "tri"),
            "native_emb": rerank_all(ch, *c["native_emb"], "emb", "tri"),
            "shuffled_edges": rerank_all(ch, *c["native"], "emb", "tri_shuffled"),
            "random_emb": rerank_all(ch, *c["native"], "emb_random", "tri"),
            "native_kgc": rerank_all(ch, *c["native_kgc"], "kgc", "tri"),
            "kgc_only": rerank_all(ch, *c["kgc_only"], "kgc", "tri"),
            "random_kgc": rerank_all(ch, *c["native_kgc"], "kgc_random", "tri"),
        }
        per_row["clip"][s] = ev(ranked)
        for g, mask in groups.items():
            per_group["clip"][g][s] = ev(ranked, mask)
        for name, lists in rows.items():
            per_row[name][s] = ev(lists)
            for g, mask in groups.items():
                per_group[name][g][s] = ev(lists, mask)
        print(f"seed {s} ({native_name}):", {n: round(100 * m[s]["recall@1"], 2) for n, m in per_row.items()},
              "choice", c)

        if args.hand and args.split == "test":
            hand, h_feats, h_golds = hand_queries(data, cfg)
            h_clip, h_ranked = data.topk(h_feats)
            h_linked = [data.linker.link(q["text"]) for q in hand]
            h_emb, h_tri = channels(scorer, h_linked, h_ranked, data.qid_of, k)
            h_kgc = kgc_channel(kgc, h_linked, h_ranked, data.qid_of, data.nid_of, k)
            hch = {"clip": h_clip, "ranked": h_ranked, "emb": h_emb, "tri": h_tri, "kgc": h_kgc}
            h_rows = {"clip": h_ranked, "kg_triple": rerank_all(hch, *c["kg_triple"], "emb", "tri"),
                      "native": rerank_all(hch, *c["native"], "emb", "tri"),
                      "native_kgc": rerank_all(hch, *c["native_kgc"], "kgc", "tri")}
            for grp in ("direct", "indirect"):
                idx = [i for i, q in enumerate(hand) if q["group"] == grp]
                for name, lists in h_rows.items():
                    per_group[f"hand_{name}"][grp][s] = evaluate([lists[i] for i in idx],
                                                                 [h_golds[i] for i in idx], ks=KS)

    split_dir = out_root / args.split
    # the original rows keep their config (choice limited to the original keys), so their files do not change
    old_choice = {s: {key: v for key, v in c.items() if key in OLD_CHOICE_KEYS} for s, c in choice.items()}
    for name, runs in per_row.items():
        config = {"top_k": k, "clip": data.encoder, "choice": old_choice, "native_prefix": args.native_prefix}
        if name in ("native_kgc", "kgc_only", "random_kgc"):
            config = {**config, "choice": choice, "kgc_checkpoints": {str(s): str(p) for s, p in kgc_paths.items()},
                      "kgc": "NativE AdvRelRotatE.forward, max over linked e != candidate, 169 relations, both "
                             "directions; no link -> 0; only self-linked -> query mean"}
        write_metrics(split_dir / f"metrics_{name}.json", method=name, dataset="mkgw", split=args.split,
                      per_seed=runs, config=config)
    summary = {
        "split": args.split, "n_queries": len(texts), "linked_pct": round(100 * has_link.mean(), 2),
        "rows": {n: aggregate_seeds(list(r.values())) for n, r in per_row.items()},
        "groups": {n: {g: aggregate_seeds(list(r.values())) for g, r in gs.items()} for n, gs in per_group.items()},
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if args.split == "val":
        (split_dir / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
        (split_dir / "alpha_beta_sweep.json").write_text(json.dumps(sweeps, indent=1), encoding="utf-8")
        (split_dir / "alpha_beta_sweep_kgc.json").write_text(json.dumps(kgc_sweeps, indent=1), encoding="utf-8")
    print("->", split_dir)


if __name__ == "__main__":
    main()
