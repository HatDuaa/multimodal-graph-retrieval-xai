"""Does the trained triple channel read the predicate and the subject/object order? Validation only, no training.

Three conditions on the sentence-side (query) graphs of val, image graphs untouched:
- baseline: unchanged.
- verb_swap: every query relation gets a random predicate from the train query-graph predicate vocabulary
  (sampled by train frequency, never the original predicate; fixed seed). Both the edge feature (rel:<p>) and
  the triple phrase (triple:<s> <p> <o>) change.
- subject_object_swap: subject and object of every query relation are exchanged, so the edge is reversed and the
  phrase becomes "<o> <p> <s>".
Part texts missing from the frozen part table are encoded with the same frozen CLIP and the same conventions as
src/service/graph_mode.py (relations and triples verbatim). Metrics are reported on all val queries and on the
queries whose graph keeps at least one triple; "triple_only" ranks the top-50 by the triple channel score alone
(the third channel: graph-level cosine for --graph-channel models), "third_only_all" does so on all queries.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import aggregate_seeds, evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

CONDITIONS = ("baseline", "verb_swap", "subject_object_swap")


def train_predicate_vocabulary(cfg):
    path = resolve(cfg, "processed") / "query_graphs_train.json"
    graphs = json.loads(path.read_text(encoding="utf-8"))
    counts = collections.Counter(rel["predicate"].lower() for g in graphs.values() for rel in g["relations"])
    words = sorted(counts)
    freq = np.array([counts[w] for w in words], dtype=np.float64)
    return words, freq / freq.sum()


def verb_swap(queries, words, probs, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for cid in sorted(queries):
        graph = copy.deepcopy(queries[cid])
        for rel in graph["relations"]:
            original = rel["predicate"].lower()
            new = original
            while new == original:
                new = words[rng.choice(len(words), p=probs)]
            rel["predicate"] = rel["predicate_lemma"] = new
        out[cid] = graph
    return out


def subject_object_swap(queries):
    out = {}
    for cid, graph in queries.items():
        graph = copy.deepcopy(graph)
        for rel in graph["relations"]:
            rel["subject"], rel["object"] = rel["object"], rel["subject"]
        out[cid] = graph
    return out


def part_keys(queries):
    keys = set()
    for graph in queries.values():
        parts = GraphStore._graph_parts(graph, query=True)
        keys.update(f"rel:{x}" for x in parts["relation_labels"])
        keys.update(f"triple:{x}" for x in parts["triple_labels"])
    return keys


def add_part_vectors(store, cfg, keys, device):
    """Append frozen-CLIP vectors for part keys the table lacks (relations and triples verbatim)."""
    missing = sorted(k for k in keys if k not in store.part_row)
    if not missing:
        return 0
    encoder = ClipEncoder.from_config(cfg, device=str(device))
    vectors = encoder.encode_texts([k.split(":", 1)[1] for k in missing])
    del encoder
    torch.cuda.empty_cache()
    offset = len(store.part_vectors)
    store.part_vectors = np.concatenate([store.part_vectors, vectors.astype(np.float32)])
    store.part_row.update({k: offset + i for i, k in enumerate(missing)})
    store._device_tables = {}
    return len(missing)


def use_queries(store, queries):
    store.queries = queries
    store.image_index = None
    store.precompute()


@torch.no_grad()
def rank_val(model, store, caption_ids, device, batch_size=256):
    fused_lists, triple_lists, golds = [], [], []
    for start in range(0, len(caption_ids), batch_size):
        batch = store.batch(caption_ids[start:start + batch_size], device)
        scores, details = model(batch)
        for name, values, sink in (("fused", scores, fused_lists), ("triple", details["triples"], triple_lists)):
            values = torch.nan_to_num(values, nan=-torch.inf)
            order = torch.argsort(values, dim=-1, descending=True, stable=True).cpu().numpy()
            sink.extend(np.take_along_axis(batch["candidate_ids"], order, axis=1).tolist())
        golds.extend(batch["gold"])
    return fused_lists, triple_lists, golds


def subset_metrics(ranked, golds, mask):
    return evaluate([r for r, m in zip(ranked, mask) if m], [g for g, m in zip(golds, mask) if m])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="Run directories, relative to the repository root.")
    parser.add_argument("--step0", action="store_true",
                        help="Score with the untrained model instead (frozen CLIP parts, fixed a/b/c, tau 0.05).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verb-seed", type=int, default=0)
    parser.add_argument("--config", type=Path, help="Data configuration (default configs/default.yaml).")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(cfg, splits=("val",))
    original = store.queries
    words, probs = train_predicate_vocabulary(cfg)
    variants = {"baseline": original, "verb_swap": verb_swap(original, words, probs, args.verb_seed),
                "subject_object_swap": subject_object_swap(original)}
    added = add_part_vectors(store, cfg, set().union(*(part_keys(v) for v in variants.values())), device)
    use_queries(store, original)
    caption_ids = [str(cid) for cid in store.candidates("val")["caption_ids"]]
    has_triple = [store.query_index.edge_counts[store.query_position[cid]] > 0 for cid in caption_ids]
    result = {"split": "val", "verb_seed": args.verb_seed, "config": str(args.config or "configs/default.yaml"), "train_predicate_vocabulary": len(words),
              "new_part_vectors": added, "n_queries": len(caption_ids), "n_with_triple": int(sum(has_triple)),
              "runs": [str(r) for r in args.runs], "per_run": {}, "aggregate": {}}
    models = {str(run): load_run_model(Path(run), device)[0] for run in args.runs}
    if args.step0:
        models["step0_frozen_clip"] = GraphReranker(adaptive_weights=False).to(device).eval()
        result["runs"].append("step0_frozen_clip")
    for condition in CONDITIONS:
        use_queries(store, variants[condition])
        for run, model in models.items():
            fused, triple, golds = rank_val(model, store, caption_ids, device)
            row = {"all": evaluate(fused, golds), "with_triple": subset_metrics(fused, golds, has_triple),
                   "triple_only_with_triple": subset_metrics(triple, golds, has_triple),
                   "third_only_all": evaluate(triple, golds)}
            result["per_run"].setdefault(run, {})[condition] = row
            print(condition, run, json.dumps({k: round(v["recall@1"], 4) for k, v in row.items()}), flush=True)
        result["aggregate"][condition] = {
            view: aggregate_seeds([result["per_run"][run][condition][view] for run in models])
            for view in ("all", "with_triple", "triple_only_with_triple", "third_only_all")}
    result["seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out} in {result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
