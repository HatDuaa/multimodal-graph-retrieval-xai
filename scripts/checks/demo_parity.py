"""Score parity of the demo path with the packaged evaluation path, on val captions (no test data is read).

For --n val captions sampled with --seed, the caption text goes through the demo exactly as typed (SearchService:
CLIP top-50 of the text; GraphRerankMode: caption graph parsed on the fly, image graph parts built in the service)
and through the packaged path (GraphStore of the run's config, store.batch, the same checkpoint). Per caption:
- the 50 candidates are the packaged ones;
- the parsed caption graph equals the packaged one (count classes and rules included), except for the captions whose
  packaged FACTUAL parse lost its last token to the batch truncation (scripts/checks/factual_truncation.py), which
  the demo does not reproduce; those are flagged;
- max |fused demo - fused packaged| over the 50 candidates;
- the parts shown by the demo add up to the final score: exactly under the raw fusion (CLIP + every row + filled
  channels), CLIP part + graph part under the z-score fusion (its rows are a proportional share).
Writes experiments/improve/demo/parity_<run>_val.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks.factual_truncation import batch_truncated  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.explain.score_breakdown import split_final_score, split_raw_score  # noqa: E402
from src.service.graph_mode import DEMO_DATA, GraphRerankMode, find_run  # noqa: E402
from src.service.search_service import SearchService  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "demo"


def parts_total(info, hit):
    """Sum of the parts the demo page shows for one hit."""
    a, b, c = info["weights_abc"]
    terms = {key: [item["channel_term"] for item in info[key]] for key in ("objects", "triples")}
    if info["fusion"] == "raw":
        split = split_raw_score(hit.score, a, hit.clip_score, terms, info["fills"])
        rows = sum(sum(v) for v in split["row_parts"].values()) + sum(v for v in split["fill_parts"].values() if v is not None)
        return split["clip_part"] + rows, sum(s for s in split["row_shares"]["objects"] + split["row_shares"]["triples"]
                                              + [split["clip_share"]] + list(split["fill_shares"].values()) if s is not None)
    active = dict(zip(("objects", "triples"), info["channel_active"]))
    contributions = {key: [item["contribution"] for item in info[key]] for key in ("objects", "triples")}
    split = split_final_score(hit.score, a, info["z_clip"], hit.graph_score, contributions, active)
    return split["clip_part"] + split["graph_part"], None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="countonly_factual_r3_seed0")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_dir = find_run(args.run)
    options = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["args"]
    config = DEMO_DATA[options.get("config")]["val"]
    cfg_level1 = load_config()
    store = GraphStore(load_config(REPO_ROOT / config) if config else cfg_level1, splits=("val",))
    device = torch.device(args.device)
    model = load_run_model(run_dir, device, store=store)[0]
    service = SearchService.from_config(split="val", cfg=cfg_level1)
    mode = GraphRerankMode(cfg_level1, "val", service.encoder, str(run_dir), args.device)
    service.add_mode("clip_graph", mode)
    texts = {c["caption_id"]: c["text"] for im in json.loads((REPO_ROOT / "data" / "splits" / "vg_coco_val.json")
                                                             .read_text(encoding="utf-8"))["images"] for c in im["captions"]}
    caption_ids = random.Random(args.seed).sample(sorted(store.queries), args.n)
    factual = DEMO_DATA[options.get("config")]["parser"] != "sng"
    truncated = set(batch_truncated(cfg_level1, "val")[0]) if factual else set()
    rows, seconds = [], []
    for cid in caption_ids:
        batch = store.batch([cid], device)
        with torch.no_grad():
            packaged = model(batch)[0][0].cpu().tolist()
        packaged = dict(zip(batch["candidate_ids"][0].tolist(), packaged))
        started = time.time()
        hits = service.search(texts[cid], "clip_graph", k=50)
        seconds.append(time.time() - started)
        graph = mode.parse(" ".join(texts[cid].split()))
        totals = [parts_total(h.explanation[0], h) for h in hits]
        rows.append({"caption_id": cid, "text": texts[cid],
                     "text_changed_by_whitespace": " ".join(texts[cid].split()) != texts[cid],
                     "same_candidates": {h.image_id for h in hits} == set(packaged),
                     "packaged_parse_truncated": cid in truncated,
                     "same_caption_graph": graph == store.queries[cid],
                     "max_abs_fused_diff": max(abs(h.score - packaged[h.image_id]) for h in hits),
                     "max_abs_parts_minus_fused": max(abs(total - h.score) for (total, _), h in zip(totals, hits)),
                     "max_abs_shares_minus_1": (max(abs(share - 1) for _, share in totals)
                                                if totals[0][1] is not None else None),
                     "top1_demo": hits[0].image_id, "top1_packaged": max(packaged, key=packaged.get)})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"run": run_dir.name, "split": "val", "n": args.n, "seed": args.seed, "device": args.device,
               "fusion": options.get("fusion") or "zscore", "config": config or "configs/default.yaml",
               "all_same_candidates": all(r["same_candidates"] for r in rows),
               "packaged_parse_truncated": sum(r["packaged_parse_truncated"] for r in rows),
               "same_caption_graph_untruncated": sum(r["same_caption_graph"] for r in rows if not r["packaged_parse_truncated"]),
               "untruncated": sum(not r["packaged_parse_truncated"] for r in rows),
               "max_abs_fused_diff_same_graph": max((r["max_abs_fused_diff"] for r in rows if r["same_caption_graph"]),
                                                    default=None),
               "max_abs_fused_diff": max(r["max_abs_fused_diff"] for r in rows),
               "max_abs_parts_minus_fused": max(r["max_abs_parts_minus_fused"] for r in rows),
               "max_abs_shares_minus_1": max((r["max_abs_shares_minus_1"] for r in rows
                                              if r["max_abs_shares_minus_1"] is not None), default=None),
               "same_top1": sum(r["top1_demo"] == r["top1_packaged"] for r in rows),
               "seconds_per_query_median": sorted(seconds)[len(seconds) // 2], "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"parity_{run_dir.name}_val.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
