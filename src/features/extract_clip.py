"""Frozen CLIP encoder: images and texts -> L2-normalised float32 vectors.

The encoder is never trained. Features are extracted once and saved with save_features(),
so every experiment of every member reads exactly the same vectors.
"""
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


class ClipEncoder:
    def __init__(self, model: str = "ViT-B-32", pretrained: str = "openai",
                 device: str | None = None, batch_size: int = 256):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.name = f"{model}/{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model)
        self.model.eval().to(self.device)

    @classmethod
    def from_config(cls, cfg: dict, device: str | None = None) -> "ClipEncoder":
        c = cfg["clip"]
        return cls(c["model"], c["pretrained"], device, c["batch_size"])

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(list(texts[i:i + self.batch_size])).to(self.device)
            out.append(self._normalise(self.model.encode_text(tokens)))
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)

    @torch.no_grad()
    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        out = []
        for i in range(0, len(paths), self.batch_size):
            batch = [self.preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + self.batch_size]]
            out.append(self._normalise(self.model.encode_image(torch.stack(batch).to(self.device))))
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _normalise(x: torch.Tensor) -> np.ndarray:
        x = x.float()
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


def save_features(path: str | Path, feats: np.ndarray, ids: Sequence, encoder_name: str) -> None:
    """Write <path>.npy plus <path>.ids.json; row i of the matrix belongs to ids[i]."""
    path = Path(path)
    assert len(ids) == len(feats), (len(ids), len(feats))
    assert len(set(ids)) == len(ids), "ids must be unique"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), feats.astype(np.float32))
    meta = {"encoder": encoder_name, "dim": int(feats.shape[1]), "ids": list(ids)}
    path.with_suffix(".ids.json").write_text(json.dumps(meta), encoding="utf-8")


def load_features(path: str | Path) -> tuple[np.ndarray, list, str]:
    path = Path(path)
    feats = np.load(path.with_suffix(".npy"))
    meta = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))
    assert len(meta["ids"]) == len(feats)
    return feats, meta["ids"], meta["encoder"]
