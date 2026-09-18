import numpy as np
import pytest

from src.features.extract_clip import load_features, save_features
from src.retrieval.faiss_index import CosineIndex
from src.utils.config import load_config
from src.utils.seed import set_seed


def unit_vectors(n, d, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_config_has_required_sections():
    cfg = load_config()
    for key in ("seed", "seeds", "paths", "clip", "retrieval"):
        assert key in cfg
    assert len(cfg["seeds"]) >= 3  # the brief requires at least 3 seeds


def test_set_seed_is_reproducible():
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    assert np.allclose(a, np.random.rand(3))


def test_feature_roundtrip(tmp_path):
    feats, ids = unit_vectors(5, 8), [10, 11, 12, 13, 14]
    save_features(tmp_path / "img", feats, ids, "test/enc")
    got, got_ids, enc = load_features(tmp_path / "img")
    assert np.array_equal(got, feats) and got_ids == ids and enc == "test/enc"


def test_save_rejects_duplicate_ids(tmp_path):
    with pytest.raises(AssertionError):
        save_features(tmp_path / "x", unit_vectors(2, 4), [1, 1], "e")


def test_index_returns_self_first():
    feats = unit_vectors(50, 16)
    ids = [f"img{i}" for i in range(50)]
    scores, found = CosineIndex(feats, ids).search(feats, k=5)
    assert [f[0] for f in found] == ids
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)
    assert (np.diff(scores, axis=1) <= 1e-6).all()  # best first


def test_index_rejects_unnormalised_vectors():
    with pytest.raises(AssertionError):
        CosineIndex(unit_vectors(3, 4) * 2.0, [0, 1, 2])
