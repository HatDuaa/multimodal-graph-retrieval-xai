"""Re-score val with the trained main_r3 models (3 seeds) and the query graphs of other parsers (part 1c).

No training: only the sentence-side graphs change, image graphs stay. Part texts that the frozen part table lacks
are encoded with the same frozen CLIP and conventions as src/service/graph_mode.py (nodes through the concept prompt,
relations and triples verbatim). The models were trained on sng_parser graphs, so other parsers are scored under a
distribution shift. Reports all val queries, the queries with >= 1 triple under that parser, and the queries with
>= 1 triple under sng (a fixed subset shared by every parser).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.triple_perturbation import rank_val, subset_metrics, use_queries  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import aggregate_seeds, evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "parser-compare"
VIEWS = ("all", "with_triple", "with_sng_triple")


def part_keys(queries):
    keys = set()
    for graph in queries.values():
        parts = GraphStore._graph_parts(graph, query=True)
        keys.update(f"node:{x}" for x in parts["node_labels"])
        keys.update(f"rel:{x}" for x in parts["relation_labels"])
        keys.update(f"triple:{x}" for x in parts["triple_labels"])
    return keys


def add_part_vectors(store, cfg, keys, device):
    missing = sorted(k for k in keys if k not in store.part_row)
    if not missing:
        return 0
    prompt = cfg["clip"]["concept_prompt"]
    encoder = ClipEncoder.from_config(cfg, device=str(device))
    vectors = encoder.encode_texts([prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1]
                                    for k in missing])
    del encoder
    torch.cuda.empty_cache()
    offset = len(store.part_vectors)
    store.part_vectors = np.concatenate([store.part_vectors, vectors.astype(np.float32)])
    store.part_row.update({k: offset + i for i, k in enumerate(missing)})
    store._device_tables = {}
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsers", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", default=[f"experiments/level1/gat/main_r3_seed{k}" for k in range(3)])
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(cfg, splits=("val",))
    caption_ids = [str(c) for c in store.candidates("val")["caption_ids"]]
    store.precompute()
    with_sng_triple = [store.query_index.edge_counts[store.query_position[c]] > 0 for c in caption_ids]
    models = {run: load_run_model(Path(run), device)[0] for run in args.runs}
    for name in args.parsers:
        queries = json.loads((resolve(cfg, "processed") / "parsers" / f"query_graphs_val_{name}.json").read_text(encoding="utf-8"))
        assert set(queries) == set(caption_ids), name
        added = add_part_vectors(store, cfg, part_keys(queries), device)
        use_queries(store, queries)
        with_triple = [store.query_index.edge_counts[store.query_position[c]] > 0 for c in caption_ids]
        result = {"parser": name, "split": "val", "runs": args.runs, "newly_encoded_parts": added,
                  "n_with_triple": int(sum(with_triple)), "n_with_sng_triple": int(sum(with_sng_triple)),
                  "per_run": {}, "aggregate": {}}
        for run, model in models.items():
            fused, _, golds = rank_val(model, store, caption_ids, device)
            result["per_run"][run] = {"all": evaluate(fused, golds), "with_triple": subset_metrics(fused, golds, with_triple),
                                      "with_sng_triple": subset_metrics(fused, golds, with_sng_triple)}
        result["aggregate"] = {v: aggregate_seeds([result["per_run"][r][v] for r in models]) for v in VIEWS}
        result["seconds"] = time.time() - started
        (OUT / f"rerank_main_r3_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, json.dumps({v: round(result["aggregate"][v]["recall@1"]["mean"], 4) for v in VIEWS}), flush=True)


if __name__ == "__main__":
    main()
