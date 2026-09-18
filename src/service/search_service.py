"""Search service behind the demo: text query -> ranked images (+ explanation).

The UI only talks to SearchService.search(), so new ranking modes and new datasets plug in
here without touching the interface. Mode "clip" is the frozen-CLIP baseline. A graph
re-ranker registers itself with add_mode(); its scorer follows docs/interfaces.md section 4
and returns, per candidate, a graph score and the paths that produced it.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from src.features.extract_clip import load_features
from src.retrieval.faiss_index import CosineIndex
from src.utils.config import load_config, resolve


class TextEncoder(Protocol):
    name: str

    def encode_texts(self, texts): ...


@dataclass
class Hit:
    rank: int
    image_id: int
    score: float
    clip_score: float
    image_path: Path
    captions: list[str]
    graph_score: float | None = None
    explanation: list = field(default_factory=list)  # paths that contributed to graph_score


# (query text, candidate hits from CLIP) -> the same hits, re-scored and carrying explanations
Reranker = Callable[[str, list[Hit]], list[Hit]]


class SearchService:
    def __init__(self, image_feats: np.ndarray, image_ids: list, meta: dict, encoder: TextEncoder, pool_k: int = 50):
        """meta maps image_id -> {"path": Path, "captions": [str]}; pool_k = candidates handed to a re-ranker."""
        self.index = CosineIndex(image_feats, image_ids)
        self.meta = meta
        self.encoder = encoder
        self.pool_k = pool_k
        self.modes: dict[str, Reranker | None] = {"clip": None}

    @classmethod
    def from_config(cls, split: str = "test", dataset: str = "vg_coco", cfg: dict | None = None,
                    encoder: TextEncoder | None = None) -> "SearchService":
        cfg = cfg or load_config()
        images = json.loads((resolve(cfg, "splits") / f"{dataset}_{split}.json").read_text(encoding="utf-8"))["images"]
        feats, ids, feat_encoder = load_features(resolve(cfg, "features") / f"{dataset}_image")
        row = {i: r for r, i in enumerate(ids)}
        pool = [im["image_id"] for im in images]
        img_dir = resolve(cfg, "raw") / "images"
        meta = {im["image_id"]: {"path": img_dir / im["file_name"], "captions": [c["text"] for c in im["captions"]]}
                for im in images}
        if encoder is None:
            from src.features.extract_clip import ClipEncoder
            encoder = ClipEncoder.from_config(cfg)
        assert encoder.name == feat_encoder, f"query encoder {encoder.name} != feature encoder {feat_encoder}"
        return cls(feats[[row[i] for i in pool]], pool, meta, encoder, cfg["retrieval"]["top_k"])

    def add_mode(self, name: str, reranker: Reranker) -> None:
        self.modes[name] = reranker

    def search(self, query: str, mode: str = "clip", k: int = 10) -> list[Hit]:
        if mode not in self.modes:
            raise ValueError(f"unknown mode {mode!r}; available: {sorted(self.modes)}")
        query = " ".join(query.split())
        if not query:
            return []
        reranker = self.modes[mode]
        n = k if reranker is None else max(k, self.pool_k)
        scores, ranked = self.index.search(self.encoder.encode_texts([query]), n)
        hits = [Hit(rank=r, image_id=i, score=float(s), clip_score=float(s),
                    image_path=self.meta[i]["path"], captions=self.meta[i]["captions"])
                for r, (i, s) in enumerate(zip(ranked[0], scores[0]), start=1)]
        if reranker is not None:
            hits = sorted(reranker(query, hits), key=lambda h: -h.score)
            for r, h in enumerate(hits, start=1):
                h.rank = r
        return hits[:k]

    def rank_of(self, query: str, image_id: int, mode: str = "clip") -> int | None:
        """Rank of a known image for this query within the candidate pool (None if outside it)."""
        for h in self.search(query, mode, k=self.pool_k):
            if h.image_id == image_id:
                return h.rank
        return None
