"""Packaged FACTUAL caption parses that lost tokens to the batch truncation of FactualSceneGraph 0.7.3.

SceneGraphParser.parse sets a batch's max length to its longest prompt counted without the end-of-sequence token, then
tokenizes with that token and truncation, so the longest prompt(s) of each batch lose their last token; prompts over 63
tokens lose more. scripts/improve/parse_captions.py parsed every split in its caption order in batches of 64 (train
and test in chunks of 8,192, a multiple of 64), so the cut captions follow from the tokenizer alone. Reads only caption
texts; writes experiments/improve/demo/factual_truncation.json. --verify-val (GPU) parses the val captions that lost a
word piece again: alone, a caption is its batch's longest, so parse() must give the packaged raw output, and
FactualParser.generate_one the untruncated one.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

BATCH, MAX_INPUT = 64, 64
OUT = REPO_ROOT / "experiments" / "improve" / "demo" / "factual_truncation.json"


def batch_truncated(cfg, split, tokenizer=None):
    """{caption id: dropped text} of the captions whose packaged FACTUAL prompt was cut."""
    from transformers import AutoTokenizer

    from scripts.improve.parse_captions import captions_of
    from src.graph.caption_parsers import FactualQueryParser
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(FactualQueryParser.CHECKPOINT)
    captions = captions_of(cfg, split)
    ids = [tokenizer.encode("Generate Scene Graph: " + text.strip(), add_special_tokens=False) for _, text in captions]
    cut = {}
    for start in range(0, len(captions), BATCH):
        limit = min(MAX_INPUT, max(len(x) for x in ids[start:start + BATCH]))
        for k in range(start, min(start + BATCH, len(captions))):
            if len(ids[k]) + 1 > limit:  # the end token takes the place of the last kept token
                cut[captions[k][0]] = tokenizer.decode(ids[k][limit - 1:]).strip()
    return cut, len(captions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-val", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from src.graph.caption_parsers import FactualParser, FactualQueryParser
    tokenizer = AutoTokenizer.from_pretrained(FactualQueryParser.CHECKPOINT)
    cfg, out = load_config(), {}
    for split in ("train", "val", "test"):
        cut, total = batch_truncated(cfg, split, tokenizer)
        words = {cid: text for cid, text in cut.items() if any(ch.isalnum() for ch in text)}
        out[split] = {"captions": total, "truncated": len(cut), "truncated_share": len(cut) / total,
                      "dropped_word_piece": len(words), "dropped_word_piece_share": len(words) / total,
                      "dropped_only_punctuation": len(cut) - len(words), "examples": dict(sorted(words.items())[:10])}
        print(split, json.dumps({k: v for k, v in out[split].items() if k != "examples"}), flush=True)
        if split == "val" and args.verify_val:
            from scripts.improve.parse_captions import captions_of
            texts = dict(captions_of(cfg, "val"))
            raw = json.loads((REPO_ROOT / "data" / "processed" / "parsers" / "raw_val_factual-large.json").read_text(encoding="utf-8"))
            model = FactualParser(FactualQueryParser.CHECKPOINT, batch_size=1)
            out["val_verify"] = {"captions": len(words),
                                 "alone_equals_packaged": sum(model.parse([texts[c]])[1][0] == raw[c] for c in words),
                                 "untruncated_equals_packaged": sum(model.generate_one(texts[c]) == raw[c] for c in words)}
            print("val_verify", json.dumps(out["val_verify"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
