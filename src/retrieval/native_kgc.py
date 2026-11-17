"""NativE's own triple score as a retrieval channel (level 2, `kgc`).

The cosine channel `emb` compares a candidate's structural embedding with a linked entity's and ignores relations.
NativE is a KGC model: its score says how plausible a triple (h, r, t) is, also for links that are missing from the
train graph. This module recomputes exactly what AdvRelRotatE.forward (zjukg/NativE c72291c,
mmkgc/module/model/AdvRelRotatE.py) returns, from a trained checkpoint's state dict, without importing NativE:

  joint_r(x) = sum_m softmax_m(ent_attn(tanh(e_m(x))) / sigmoid(rel_gate[r])) * e_m(x)
               e_s = ent_embeddings[x], e_v = img_proj(img_embeddings[x]), e_t = text_proj(text_embeddings[x])
  score(h, r, t) = margin - sum_d |joint_r(h)_d * rot_r,d - joint_r(t)_d|        (complex, 250 dims)
               rot_r = exp(i * rel_embeddings[r] * pi / rel_embedding_range)

Higher means more plausible: NativE's tester ranks by predict() = -forward(). Head- and tail-batch modes give the
same value because |rot| = 1. The modality tables are the ones stored in the checkpoint, i.e. the MKG-W-rmask tables
the model was trained with (zero rows for retrieval val/test entities, as intended).

  kgc(c, L) = max over e in L, e != c, over the 169 relations and both directions of score(c, r, e), score(e, r, c)

It is undefined when L holds no entity other than c; the caller fills it (see channel()).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

DIRECTIONS = ("candidate->linked", "linked->candidate")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class NativeKgcScorer:
    """AdvRelRotatE triple scores from a state dict (CPU or CUDA)."""

    def __init__(self, state_dict: dict, device: str | None = None, ent_embeddings: np.ndarray | None = None):
        self.device = device or default_device()
        sd = {k: v.detach().to(self.device, torch.float32) for k, v in state_dict.items()}
        es = sd["ent_embeddings.weight"] if ent_embeddings is None else \
            torch.as_tensor(ent_embeddings, dtype=torch.float32, device=self.device)
        assert es.shape == sd["ent_embeddings.weight"].shape, "replacement entity vectors must keep the shape"
        with torch.no_grad():
            ev = self._mlp(sd["img_embeddings.weight"], sd, "img_proj")
            et = self._mlp(sd["text_embeddings.weight"], sd, "text_proj")
            self.parts = torch.stack((es, ev, et), dim=1)                                   # (N, 3, 2d)
            self.attn_logits = torch.tanh(self.parts) @ sd["ent_attn.weight"][0]             # (N, 3)
            self.temperature = torch.sigmoid(sd["rel_gate.weight"][:, 0])                    # (R,)
            phase = sd["rel_embeddings.weight"] / (sd["rel_embedding_range"].item() / math.pi)
            self.cos, self.sin = torch.cos(phase), torch.sin(phase)                          # (R, d)
        self.margin = sd["margin"].item()
        self.n_entities, self.n_relations = es.shape[0], phase.shape[0]

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None,
                        ent_embeddings: np.ndarray | None = None) -> "NativeKgcScorer":
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state_dict, device, ent_embeddings)

    @staticmethod
    def _mlp(x: torch.Tensor, sd: dict, prefix: str) -> torch.Tensor:
        h = torch.relu(x @ sd[f"{prefix}.0.weight"].T + sd[f"{prefix}.0.bias"])
        return h @ sd[f"{prefix}.2.weight"].T + sd[f"{prefix}.2.bias"]

    def joint(self, entities: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """(len(relations), len(entities), 2d) joint embeddings, one per relation (the gate depends on it)."""
        logits = self.attn_logits[entities][None] / self.temperature[relations][:, None, None]
        weights = torch.softmax(logits, dim=-1)
        return (weights[..., None] * self.parts[entities][None]).sum(2)

    @torch.no_grad()
    def score(self, heads, relations, tails) -> np.ndarray:
        """NativE's forward() for a batch of triples, elementwise."""
        h, r, t = (torch.as_tensor(np.asarray(x), dtype=torch.long, device=self.device)
                   for x in (heads, relations, tails))
        out = []
        for i in range(len(h)):
            jh = self.joint(h[i:i + 1], r[i:i + 1])[0, 0]
            jt = self.joint(t[i:i + 1], r[i:i + 1])[0, 0]
            out.append(self._distance(jh[None, None], jt[None, None], r[i:i + 1])[0, 0, 0])
        return (self.margin - torch.stack(out)).cpu().numpy()

    def _distance(self, heads: torch.Tensor, tails: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """sum_d |h * rot_r - t| for heads (R, H, 2d) and tails (R, T, 2d): (R, H, T)."""
        re_h, im_h = heads.chunk(2, dim=-1)
        re_t, im_t = tails.chunk(2, dim=-1)
        cos, sin = self.cos[relations][:, None], self.sin[relations][:, None]
        re_rot, im_rot = re_h * cos - im_h * sin, re_h * sin + im_h * cos                  # (R, H, d)
        re = re_rot[:, :, None] - re_t[:, None]
        im = im_rot[:, :, None] - im_t[:, None]
        return torch.sqrt(re * re + im * im).sum(-1)

    @torch.no_grad()
    def best(self, candidates, linked, chunk: int = 8) -> tuple[np.ndarray, list]:
        """kgc(c, L) for every candidate (NaN where L has no entity other than c) and its argmax
        (linked entity id, relation id, direction) or None."""
        cands = torch.as_tensor(np.asarray(candidates), dtype=torch.long, device=self.device)
        links = torch.as_tensor(np.asarray(linked), dtype=torch.long, device=self.device)
        n_c, n_l = len(cands), len(links)
        values = np.full(n_c, np.nan)
        argmax: list = [None] * n_c
        if n_c == 0 or n_l == 0:
            return values, argmax
        best = torch.full((n_c,), -math.inf, device=self.device)
        best_index = torch.zeros((n_c, 3), dtype=torch.long, device=self.device)
        same = cands[:, None] == links[None]                                                 # (C, L)
        rels = torch.arange(self.n_relations, device=self.device)
        jc = self.joint(cands, rels)                                                         # (R, C, 2d)
        for start in range(0, n_l, chunk):
            part = links[start:start + chunk]
            jl = self.joint(part, rels)                                                      # (R, l, 2d)
            for direction in range(2):
                dist = self._distance(jc, jl, rels) if direction == 0 else self._distance(jl, jc, rels).transpose(1, 2)
                s = self.margin - dist                                                       # (R, C, l)
                s = s.masked_fill(same[None, :, start:start + chunk], -math.inf)
                flat = s.permute(1, 0, 2).reshape(n_c, -1)                                   # (C, R * l)
                value, index = flat.max(1)
                better = value > best
                best = torch.where(better, value, best)
                r, e = index // len(part), index % len(part) + start
                new = torch.stack((e, r, torch.full_like(r, direction)), dim=1)
                best_index = torch.where(better[:, None], new, best_index)
        found = torch.isfinite(best)
        values[found.cpu().numpy()] = best[found].double().cpu().numpy()
        for i in torch.nonzero(found).flatten().tolist():
            e, r, direction = best_index[i].tolist()
            argmax[i] = (int(linked[e]), r, DIRECTIONS[direction])
        return values, argmax


def channel(scorer: NativeKgcScorer, linked: list[dict], ranked: list[list], qid_of: dict, nid_of: dict,
            k: int) -> np.ndarray:
    """(queries, k) kgc values. A query with no linked entity gets 0 everywhere (neutral, as the other channels).
    A candidate whose only linked entity is itself gets the mean of the query's defined values, which is neutral
    after the z-score (a raw 0 would rank it far above every real score, margin - distance being far below 0)."""
    out = np.zeros((len(ranked), k), dtype=np.float32)
    for qi, cands in enumerate(ranked):
        links = sorted({nid_of[q] for q in linked[qi] if q in nid_of})
        if not links:
            continue
        values, _ = scorer.best([nid_of[qid_of[img]] for img in cands], links)
        defined = ~np.isnan(values)
        if defined.any():
            values[~defined] = values[defined].mean()
            out[qi, : len(cands)] = values
    return out


def random_entities(ent: np.ndarray, seed: int) -> np.ndarray:
    """Gaussian entity vectors with the recipe of random_emb (default_rng(seed), unit rows), rescaled to each row's
    trained norm so that the structural part keeps its weight in the joint embedding."""
    rnd = np.random.default_rng(seed).standard_normal(ent.shape).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    return rnd * np.linalg.norm(ent, axis=1, keepdims=True).astype(np.float32)
