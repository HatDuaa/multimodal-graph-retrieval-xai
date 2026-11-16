"""Level 2 helpers: leakage mask for NativE, edge shuffling and score fusion (Khang)."""
from collections import Counter

import numpy as np
import torch

from scripts.eval_mkgw_level2 import fuse_rank, shuffle_tails
from scripts.mask_native_modalities import mask_rows


def test_mask_rows_zeroes_only_held_out_entities():
    table = torch.ones(5, 3)
    out = mask_rows(table, [1, 3])
    assert out[[1, 3]].abs().sum() == 0
    assert torch.equal(out[[0, 2, 4]], torch.ones(3, 3))
    assert torch.equal(table, torch.ones(5, 3)), "input must not be modified"


def test_shuffle_tails_keeps_degrees_per_relation():
    triples = [{"h_qid": f"H{i}", "t_qid": f"T{i % 3}", "r_pid": "P1" if i < 6 else "P2", "kgc_split": "train"}
               for i in range(10)]
    out = shuffle_tails(triples, seed=0)
    assert len(out) == len(triples)
    for pid in ("P1", "P2"):
        before = [t for t in triples if t["r_pid"] == pid]
        after = [t for t in out if t["r_pid"] == pid]
        assert Counter(t["h_qid"] for t in before) == Counter(t["h_qid"] for t in after)
        assert Counter(t["t_qid"] for t in before) == Counter(t["t_qid"] for t in after)
    assert shuffle_tails(triples, seed=0) == out, "same seed, same graph"


def test_fuse_rank_alpha_one_is_clip_order():
    clip = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tri = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    ids = ["a", "b", "c"]
    assert fuse_rank(clip, emb, tri, ids, alpha=1.0, beta=0.5) == ["b", "c", "a"]
    assert fuse_rank(clip, emb, tri, ids, alpha=0.0, beta=0.0)[0] == "a"
