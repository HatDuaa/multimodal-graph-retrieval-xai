"""Val ranking of runs on the captions whose graph has a known count, and the learned count tables (experiment 4).

Subsets are fixed by the group_v1 caption graphs (src/graph/counts.py classes of scorable nodes), the same for every
run: "known_count" (>= 1 node with a known class), "explicit" (>= 1 node with an explicit 1 / 2 / 3-4 / >=5 class),
"plural_or_group_only" (known, none explicit), and "unknown_only". Each run is ranked on all val captions with its own
data configuration and graph structure (config.json); metrics per subset, then mean +- std over the runs of a group.
Count-table runs also report their table g (best checkpoint), per seed and mean +- std.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import aggregate_seeds, evaluate  # noqa: E402
from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

EXPLICIT = {CAPTION_CLASSES.index(c) for c in ("1", "2", "3-4", ">=5")}


def subsets():
    graphs = json.loads((REPO_ROOT / "data" / "processed" / "group_v1" / "query_graphs_val.json").read_text(encoding="utf-8"))
    out = {}
    for cid, graph in graphs.items():
        known = {o.get("count_class", 0) for o in graph["objects"] if o["name"] and o["name"] not in PRONOUNS} - {0}
        out[cid] = {"all": True, "known_count": bool(known), "explicit": bool(known & EXPLICIT),
                    "plural_or_group_only": bool(known) and not known & EXPLICIT, "unknown_only": not known}
    return out


@torch.no_grad()
def ranked_val(model, store, device):
    caption_ids = [str(c) for c in store.candidates("val")["caption_ids"]]
    ranked, golds = [], []
    for start in range(0, len(caption_ids), 256):
        batch = store.batch(caption_ids[start:start + 256], device)
        scores, _ = model(batch)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True).cpu().numpy()
        ranked.extend(np.take_along_axis(batch["candidate_ids"], order, axis=1).tolist())
        golds.extend(batch["gold"])
    return caption_ids, ranked, golds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", nargs="+", action="append", required=True, help="NAME RUN_DIR [RUN_DIR ...]")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    membership = subsets()
    names = list(next(iter(membership.values())))
    result = {"split": "val", "subset_sizes": {n: int(sum(m[n] for m in membership.values())) for n in names},
              "per_run": {}, "aggregate": {}, "count_tables": {}}
    stores = {}
    for group, *runs in args.group:
        tables = []
        for run in runs:
            config = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))["args"].get("config")
            if config not in stores:
                stores.clear()
                stores[config] = GraphStore(load_config(REPO_ROOT / config), splits=("val",))
            store = stores[config]
            model = load_run_model(Path(run), device, store=store)[0]
            caption_ids, ranked, golds = ranked_val(model, store, device)
            result["per_run"][run] = {n: evaluate([r for c, r in zip(caption_ids, ranked) if membership[c][n]],
                                                  [g for c, g in zip(caption_ids, golds) if membership[c][n]]) for n in names}
            print(group, run, json.dumps({n: round(v["recall@1"], 4) for n, v in result["per_run"][run].items()}), flush=True)
            if model.options.get("count_table"):
                tables.append(model.counts_table().detach().cpu().numpy())
        result["aggregate"][group] = {n: aggregate_seeds([result["per_run"][r][n] for r in runs]) for n in names}
        if tables:
            stack = np.stack(tables)
            result["count_tables"][group] = {"rows": CAPTION_CLASSES, "columns": IMAGE_BUCKETS,
                                             "per_run": {r: t.tolist() for r, t in zip(runs, tables)},
                                             "mean": stack.mean(0).tolist(),
                                             "std": stack.std(0, ddof=1).tolist() if len(tables) > 1 else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
