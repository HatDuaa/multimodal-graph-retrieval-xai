"""Parse the captions of one split with one parser into the query graph schema (improvement phase, part 1a).

Writes data/processed/parsers/query_graphs_<split>_<parser>.json ({caption_id: graph}, gitignored like every
processed file), the parser's raw output next to it, and experiments/improve/parser-compare/timing_<split>_<parser>.json.
The sng run is also checked against the packaged level-1 query graphs. With --chunk-size, a long split is parsed in
chunks saved under data/processed/parsers/chunks_<split>_<parser>/; a rerun skips finished chunks (resume). Names are
normalised name by name, so per-chunk conversion gives the same graphs as one pass.
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
    parser.add_argument("--split", choices=("train", "val", "test"), default="val",
                        help="test is parsed only to have the data ready; nothing here evaluates it")
    parser.add_argument("--chunk-size", type=int, help="Parse in resumable chunks of this many captions.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16, help="sng only: worker processes")
    args = parser.parse_args()
    cfg = load_config()
    captions = captions_of(cfg, args.split)[:args.limit]
    texts = [text for _, text in captions]
    started = time.time()
    out_dir = resolve(cfg, "processed") / "parsers"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{args.parser}" + (f"_limit{args.limit}" if args.limit else "")
    build = lambda: (FactualParser(PARSERS[args.parser], batch_size=args.batch_size)  # noqa: E731
                     if args.parser.startswith("factual") else LLMParser(PARSERS[args.parser], batch_size=args.batch_size))
    if args.chunk_size and args.parser != "sng":
        chunk_dir = out_dir / f"chunks_{suffix}"
        chunk_dir.mkdir(exist_ok=True)
        model, graphs, raw_text, generate, failed, loaded = None, [], [], 0., 0, started
        for k, start in enumerate(range(0, len(captions), args.chunk_size)):
            path = chunk_dir / f"chunk_{k:04d}.json"
            if not path.exists():
                if model is None:
                    model = build()
                    loaded = time.time()
                began = time.time()
                raw, text = model.parse(texts[start:start + args.chunk_size])
                payload = {"caption_ids": [c for c, _ in captions[start:start + args.chunk_size]], "graphs": convert(raw),
                           "raw": text, "failed": sum(bool(r.get("failed")) for r in raw),
                           "seconds_generate": time.time() - began}
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(path)
                print(json.dumps({"chunk": k, "captions": len(payload["graphs"]),
                                  "seconds": payload["seconds_generate"]}), flush=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["caption_ids"] == [c for c, _ in captions[start:start + args.chunk_size]], path
            graphs += payload["graphs"]
            raw_text += payload["raw"]
            generate += payload["seconds_generate"]
            failed += payload["failed"]
        device, parsed = "cuda (RTX 4090)", loaded + generate
    elif args.parser == "sng":
        graphs, raw_text = parse_many(texts, args.workers), None
        device = f"cpu, {args.workers} processes"
    else:
        model = build()
        loaded = time.time()
        raw, raw_text = model.parse(texts)
        parsed = time.time()
        graphs = convert(raw)
        failed = sum(bool(r.get("failed")) for r in raw)
        device = "cuda (RTX 4090)"
    seconds = time.time() - started if not args.chunk_size else parsed - started
    (out_dir / f"query_graphs_{suffix}.json").write_text(
        json.dumps({cid: g for (cid, _), g in zip(captions, graphs)}), encoding="utf-8")
    timing = {"parser": args.parser, "checkpoint": PARSERS[args.parser], "split": args.split, "captions": len(captions),
              "device": device, "batch_size": None if args.parser == "sng" else args.batch_size,
              "chunk_size": args.chunk_size,
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
