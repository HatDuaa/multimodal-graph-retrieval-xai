"""Build a leakage-safe copy of MKG-W for NativE training (level 2, retrieval use).

Why: NativE's frozen modality tables are extracted from exactly what the retrieval
task evaluates. MKG-W-textual.pth encodes each entity's description (our query
text is that description with the name masked) and MKG-W-visual.pth encodes the
entity's images (our candidates). Training NativE on all 15 000 rows would let the
projection layers and the structural embeddings see val/test queries and
candidates, which assignment section 6 forbids.

What: copy benchmarks/MKG-W to benchmarks/<name> and write
embeddings/<name>-{visual,textual}.pth where every row of an entity in the
retrieval val or test split is set to zero (the "missing modality" case that
NativE is built for). KGC triples are left untouched: they hold no query text
and no image, and only the KGC-train part is used for training anyway.

The original MKG-W stays as it is for step a (reproducing the paper's link
prediction numbers).

Usage:  python scripts/mask_native_modalities.py [--name MKG-W-rmask]
Run from the repository root; NativE location comes from configs/default.yaml.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_mkgw_wikidata import read_entity2id  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def held_out_rows(split_dir: Path, id_of_qid: dict[str, int], splits=("val", "test")) -> list[int]:
    """NativE entity ids of every entity that appears in the retrieval val/test splits."""
    rows = set()
    for split in splits:
        images = json.loads((split_dir / f"mkgw_{split}.json").read_text(encoding="utf-8"))["images"]
        for im in images:
            rows.add(id_of_qid[im["qid"]])
    return sorted(rows)


def mask_rows(table: torch.Tensor, rows: list[int]) -> torch.Tensor:
    out = table.clone()
    out[rows] = 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="MKG-W-rmask")
    args = ap.parse_args()

    cfg = load_config()
    native = Path(cfg["mkgw"]["native_repo"]).expanduser()
    id_of_qid = dict(read_entity2id(native))
    rows = held_out_rows(resolve(cfg, "splits"), id_of_qid)

    src, dst = native / "benchmarks" / "MKG-W", native / "benchmarks" / args.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    report = {"name": args.name, "masked_entities": len(rows), "n_entities": len(id_of_qid)}
    for modality in ("visual", "textual"):
        table = torch.load(native / "embeddings" / f"MKG-W-{modality}.pth", map_location="cpu")
        masked = mask_rows(table, rows)
        torch.save(masked, native / "embeddings" / f"{args.name}-{modality}.pth")
        report[f"{modality}_shape"] = list(table.shape)
        report[f"{modality}_zero_rows"] = int((masked.abs().sum(1) == 0).sum())
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
