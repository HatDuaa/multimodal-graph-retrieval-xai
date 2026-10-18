"""Parse the captions of one split with one parser into the query graph schema (improvement phase, part 1a).

Writes data/processed/parsers/query_graphs_<split>_<parser>.json ({caption_id: graph}, gitignored like every
processed file), the parser's raw output next to it, and experiments/improve/parser-compare/timing_<split>_<parser>.json.
The sng run is also checked against the packaged level-1 query graphs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.caption_parsers import FactualParser, LLMParser, convert  # noqa: E402
from src.graph.parse_query import parse_many  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

PARSERS = {"sng": None, "factual-base": "lizhuang144/flan-t5-base-VG-factual-sg",
           "factual-large": "lizhuang144/flan-t5-large-VG-factual-sg", "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct"}


def captions_of(cfg, split):
    images = json.loads((resolve(cfg, "splits") / f"vg_coco_{split}.json").read_text(encoding="utf-8"))["images"]
    return [(c["caption_id"], c["text"]) for im in images for c in im["captions"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", choices=sorted(PARSERS), required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16, help="sng only: worker processes")
    args = parser.parse_args()
    cfg = load_config()
    captions = captions_of(cfg, args.split)[:args.limit]
    texts = [text for _, text in captions]
    started = time.time()
    if args.parser == "sng":
        graphs, raw_text = parse_many(texts, args.workers), None
        device = f"cpu, {args.workers} processes"
    else:
        model = (FactualParser(PARSERS[args.parser], batch_size=args.batch_size) if args.parser.startswith("factual")
                 else LLMParser(PARSERS[args.parser], batch_size=args.batch_size))
        loaded = time.time()
        raw, raw_text = model.parse(texts)
        parsed = time.time()
        graphs = convert(raw)
        failed = sum(bool(r.get("failed")) for r in raw)
        device = "cuda (RTX 4090)"
    seconds = time.time() - started
    out_dir = resolve(cfg, "processed") / "parsers"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{args.parser}" + (f"_limit{args.limit}" if args.limit else "")
    (out_dir / f"query_graphs_{suffix}.json").write_text(
        json.dumps({cid: g for (cid, _), g in zip(captions, graphs)}), encoding="utf-8")
    timing = {"parser": args.parser, "checkpoint": PARSERS[args.parser], "split": args.split, "captions": len(captions),
              "device": device, "batch_size": None if args.parser == "sng" else args.batch_size,
              "seconds_total": seconds, "seconds_per_1000": 1000 * seconds / len(captions)}
    if raw_text is not None:
        (out_dir / f"raw_{suffix}.json").write_text(
            json.dumps({cid: t for (cid, _), t in zip(captions, raw_text)}), encoding="utf-8")
        timing.update(seconds_model_load=loaded - started, seconds_generate=parsed - loaded,
                      seconds_per_1000_generate=1000 * (parsed - loaded) / len(captions), failed_outputs=failed)
    if args.parser == "sng" and not args.limit:
        packaged = json.loads((resolve(cfg, "processed") / f"query_graphs_{args.split}.json").read_text(encoding="utf-8"))
        timing["matches_level1_packaged"] = all(packaged[cid] == g for (cid, _), g in zip(captions, graphs))
    report = REPO_ROOT / "experiments" / "improve" / "parser-compare"
    report.mkdir(parents=True, exist_ok=True)
    (report / f"timing_{suffix}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
