"""Export entity embeddings from a trained NativE checkpoint to the shared format.

Reads the AdvRelRotatE state dict saved by NativE's trainer and writes
data/features/mkgw_native_<what>.npy + .ids.json (docs/interfaces.md, section 2),
with one row per MKG-W entity, ids = Wikidata qids in entity2id order. These
vectors replace the GNN graph score in the re-ranking step (task sheet, level 2
row "buoc b"). Three views can be exported:

  ent        structural embedding                    (15000 x 500)
  img_proj   image feature pushed through img_proj   (15000 x 500)
  text_proj  text feature pushed through text_proj   (15000 x 500)

Rows are L2-normalised (interface convention: dot product = cosine). Note that
NativE's own scoring uses unnormalised, relation-gated fusion; the normalised
export is a retrieval-side representation, not the KGC score.

Usage:  python scripts/export_native_embeddings.py --checkpoint <path> [--what ent img_proj text_proj]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_mkgw_wikidata import read_entity2id  # noqa: E402
from src.features.extract_clip import save_features  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def project(x: torch.Tensor, sd: dict, prefix: str) -> torch.Tensor:
    """Apply NativE's Sequential(Linear, ReLU, Linear) projection from raw state-dict keys."""
    h = torch.relu(x @ sd[f"{prefix}.0.weight"].T + sd[f"{prefix}.0.bias"])
    return h @ sd[f"{prefix}.2.weight"].T + sd[f"{prefix}.2.bias"]


def export(sd: dict, what: str) -> torch.Tensor:
    if what == "ent":
        return sd["ent_embeddings.weight"]
    if what == "img_proj":
        return project(sd["img_embeddings.weight"], sd, "img_proj")
    if what == "text_proj":
        return project(sd["text_embeddings.weight"], sd, "text_proj")
    raise ValueError(what)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--what", nargs="+", default=["ent"], choices=["ent", "img_proj", "text_proj"])
    args = ap.parse_args()

    cfg = load_config()
    native_repo = Path(cfg["mkgw"]["native_repo"]).expanduser()
    qids = [qid for qid, _ in sorted(read_entity2id(native_repo), key=lambda p: p[1])]

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert sd["ent_embeddings.weight"].shape[0] == len(qids), "checkpoint does not match entity2id"
    ckpt_name = Path(args.checkpoint).name
    for what in args.what:
        with torch.no_grad():
            feats = export(sd, what)
            feats = torch.nn.functional.normalize(feats, dim=-1).numpy().astype(np.float32)
        out = resolve(cfg, "features") / f"mkgw_native_{what}"
        save_features(out, feats, qids, encoder_name=f"NativE-AdvRelRotatE[{ckpt_name}]/{what}")
        print(f"{what}: {feats.shape} -> {out}.npy")


if __name__ == "__main__":
    main()
