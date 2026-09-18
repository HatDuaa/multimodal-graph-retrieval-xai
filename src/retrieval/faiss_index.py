"""Exact cosine top-k search over L2-normalised vectors (inner product == cosine)."""
from typing import Sequence

import faiss
import numpy as np


class CosineIndex:
    def __init__(self, feats: np.ndarray, ids: Sequence):
        assert len(feats) == len(ids)
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "vectors must be L2-normalised"
        self.ids = list(ids)
        self.index = faiss.IndexFlatIP(feats.shape[1])
        self.index.add(feats)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, list[list]]:
        """Return (scores [n, k], ids [n][k]), best first."""
        k = min(k, len(self.ids))
        scores, rows = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return scores, [[self.ids[j] for j in r] for r in rows]
