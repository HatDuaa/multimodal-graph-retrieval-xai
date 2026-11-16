"""Package data version factual_v1: level-1 data with the caption graphs of FACTUAL-T5 large (improvement, step 1d).

Only the sentence side changes. Writes, never overwriting an existing file:
- data/processed/factual_v1/query_graphs_{train,val,test}.json from data/processed/parsers/query_graphs_<split>_factual-large.json;
- data/processed/factual_v1/candidates_{split}.npz: symlinks to the level-1 CLIP top-50 candidates (unchanged);
- data/features/factual_v1/vg_coco_caption.*: symlinks to the level-1 caption vectors (unchanged);
- data/features/factual_v1/vg_coco_graph_parts.*: the level-1 part table plus the part texts of the new caption graphs
  that it lacks, encoded with the same frozen CLIP and conventions (nodes through the concept prompt, relations and
  triples verbatim). A seeded sample of 1,000 level-1 keys is re-encoded to confirm the encoder matches (max abs <= 1e-4);
- data/processed/factual_v1/MANIFEST.json.
Read it through configs/factual_v1.yaml.
"""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.features.extract_clip import ClipEncoder, load_features, save_features  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

SPLITS = ("train", "val", "test")
VERSION = "factual_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    return path


def query_keys(graph):
    parts = GraphStore._graph_parts(graph, query=True)
    return ([f"node:{x}" for x in parts["node_labels"]] + [f"rel:{x}" for x in parts["relation_labels"]]
            + [f"triple:{x}" for x in parts["triple_labels"]])


def main() -> None:
    started = time.time()
    cfg = load_config()
    processed, features = resolve(cfg, "processed"), resolve(cfg, "features")
    out_processed, out_features = processed / VERSION, features / VERSION
    out_processed.mkdir(exist_ok=True)
    out_features.mkdir(exist_ok=True)
    manifest = {"version": VERSION, "date": time.strftime("%F %T %z"), "parser": "lizhuang144/flan-t5-large-VG-factual-sg",
                "parse_script": "scripts/improve/parse_captions.py --parser factual-large", "splits": {}}
    new_keys = set()
    for split in SPLITS:
        source = processed / "parsers" / f"query_graphs_{split}_factual-large.json"
        graphs = json.loads(source.read_text(encoding="utf-8"))
        images = json.loads((resolve(cfg, "splits") / f"vg_coco_{split}.json").read_text(encoding="utf-8"))["images"]
        expected = {c["caption_id"] for im in images for c in im["captions"]}
        assert set(graphs) == expected, f"{split}: caption ids differ from the split"
        target = fresh(out_processed / f"query_graphs_{split}.json")
        target.write_text(json.dumps(graphs), encoding="utf-8")
        link = fresh(out_processed / f"candidates_{split}.npz")
        os.symlink(os.path.relpath(processed / f"candidates_{split}.npz", out_processed), link)
        for graph in graphs.values():
            new_keys.update(query_keys(graph))
        manifest["splits"][split] = {"captions": len(graphs), "source": str(source.relative_to(REPO_ROOT)),
                                     "source_sha256": sha256(source), "candidates": f"symlink to level-1 candidates_{split}.npz",
                                     "captions_with_triple": sum(bool(GraphStore._graph_parts(g, query=True)["edges"])
                                                                 for g in graphs.values())}
    for suffix in (".npy", ".ids.json"):
        link = fresh(out_features / f"vg_coco_caption{suffix}")
        os.symlink(os.path.relpath(features / f"vg_coco_caption{suffix}", out_features), link)
    base, base_keys, encoder_name = load_features(features / "vg_coco_graph_parts")
    base_row = {k: i for i, k in enumerate(base_keys)}
    missing = sorted(new_keys - set(base_row))
    encoder = ClipEncoder.from_config(cfg)
    assert encoder.name == encoder_name, (encoder.name, encoder_name)
    prompt = cfg["clip"]["concept_prompt"]
    text = lambda k: prompt.format(k.split(":", 1)[1]) if k.startswith("node:") else k.split(":", 1)[1]  # noqa: E731
    sample = random.Random(cfg["seed"]).sample(base_keys, 1000)
    check = encoder.encode_texts([text(k) for k in sample])
    error = float(np.max(np.abs(check - base[[base_row[k] for k in sample]])))
    assert error <= 1e-4, f"encoder differs from the level-1 table: max abs {error}"
    extra = encoder.encode_texts([text(k) for k in missing]) if missing else np.zeros((0, base.shape[1]), np.float32)
    keys = list(base_keys) + missing
    fresh(out_features / "vg_coco_graph_parts.ids.json")
    save_features(fresh(out_features / "vg_coco_graph_parts.npy").with_suffix(""), np.concatenate([base, extra]), keys,
                  encoder_name, dtype=base.dtype)
    manifest["part_table"] = {"level1_keys": len(base_keys), "new_keys": len(missing),
                              "new_by_kind": {kind: sum(k.startswith(f"{kind}:") for k in missing) for kind in ("node", "rel", "triple")},
                              "caption_part_keys": len(new_keys), "encoder": encoder_name,
                              "reencoded_level1_sample": len(sample), "reencoded_max_abs_error": error,
                              "level1_table_sha256": sha256(features / "vg_coco_graph_parts.npy")}
    manifest["seconds"] = time.time() - started
    fresh(out_processed / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
