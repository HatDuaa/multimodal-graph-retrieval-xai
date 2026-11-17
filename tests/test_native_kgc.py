"""The kgc channel recomputes NativE's own triple score (AdvRelRotatE.forward) from a state dict."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.retrieval.native_kgc import DIRECTIONS, NativeKgcScorer, channel, random_entities
from src.utils.config import load_config

N_ENT, N_REL, DIM = 7, 4, 6


def native_repo() -> Path:
    return Path(load_config()["mkgw"]["native_repo"]).expanduser()


def native_class():
    """NativE's own AdvRelRotatE (mkgw.native_repo in configs/default.yaml); the test is skipped without it."""
    repo = native_repo()
    if not (repo / "mmkgc" / "module" / "model" / "AdvRelRotatE.py").exists():
        pytest.skip(f"NativE code not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from mmkgc.module.model.AdvRelRotatE import AdvRelRotatE
    return AdvRelRotatE


def toy_state_dict(seed: int = 0) -> dict:
    """A random AdvRelRotatE state dict, with zero modality rows for entities 5 and 6 (as MKG-W-rmask)."""
    g = torch.Generator().manual_seed(seed)
    img, text = torch.randn(N_ENT, 5, generator=g), torch.randn(N_ENT, 3, generator=g)
    img[5:], text[5:] = 0., 0.
    d2 = 2 * DIM
    rnd = lambda *shape: 0.5 * torch.randn(*shape, generator=g)  # noqa: E731
    return {"zero_const": torch.zeros(1), "pi_const": torch.tensor([np.pi]), "ent_embedding_range": torch.tensor([.5]),
            "rel_embedding_range": torch.tensor([1.]), "margin": torch.tensor([4.]),
            "ent_embeddings.weight": rnd(N_ENT, d2), "rel_embeddings.weight": rnd(N_REL, DIM),
            "img_embeddings.weight": img, "text_embeddings.weight": text,
            "img_proj.0.weight": rnd(d2, 5), "img_proj.0.bias": rnd(d2), "img_proj.2.weight": rnd(d2, d2),
            "img_proj.2.bias": rnd(d2), "text_proj.0.weight": rnd(d2, 3), "text_proj.0.bias": rnd(d2),
            "text_proj.2.weight": rnd(d2, d2), "text_proj.2.bias": rnd(d2), "ent_attn.weight": rnd(1, d2),
            "rel_gate.weight": rnd(N_REL, 1), "adv_scores.0.weight": rnd(d2, d2), "adv_scores.0.bias": rnd(d2),
            "adv_scores.2.weight": rnd(1, d2), "adv_scores.2.bias": rnd(1)}


def native_scores(model, h, r, t, mode):
    data = {"batch_h": torch.tensor(h), "batch_t": torch.tensor(t), "batch_r": torch.tensor(r), "mode": mode}
    with torch.no_grad():
        return model(data).numpy()


def test_score_is_natives_own_forward_on_a_toy_model():
    AdvRelRotatE = native_class()
    sd = toy_state_dict()
    model = AdvRelRotatE(ent_tot=N_ENT, rel_tot=N_REL, dim=DIM, margin=4.0, epsilon=2.0,
                         img_emb=sd["img_embeddings.weight"], text_emb=sd["text_embeddings.weight"])
    model.load_state_dict(sd)
    rng = np.random.default_rng(0)
    h, r, t = (rng.integers(0, n, 20).tolist() for n in (N_ENT, N_REL, N_ENT))
    ours = NativeKgcScorer(sd, device="cpu").score(h, r, t)
    for mode in ("normal", "head_batch"):
        np.testing.assert_allclose(ours, native_scores(model, h, r, t, mode), rtol=1e-5, atol=1e-5)


def test_score_is_natives_own_forward_on_the_trained_rmask_checkpoint():
    AdvRelRotatE = native_class()
    path = native_repo() / "checkpoint" / "rmask_seed0_e1000.ckpt"
    if not path.exists():
        pytest.skip(f"no checkpoint at {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    n_ent, n_rel = sd["ent_embeddings.weight"].shape[0], sd["rel_embeddings.weight"].shape[0]
    model = AdvRelRotatE(ent_tot=n_ent, rel_tot=n_rel, dim=sd["rel_embeddings.weight"].shape[1], margin=4.0,
                         epsilon=2.0, img_emb=sd["img_embeddings.weight"], text_emb=sd["text_embeddings.weight"])
    model.load_state_dict(sd)
    rng = np.random.default_rng(1)
    h, r, t = (rng.integers(0, n, 16).tolist() for n in (n_ent, n_rel, n_ent))
    np.testing.assert_allclose(NativeKgcScorer(sd, device="cpu").score(h, r, t),
                               native_scores(model, h, r, t, "normal"), rtol=1e-4, atol=1e-3)


def test_best_is_the_max_over_linked_entities_relations_and_directions():
    scorer = NativeKgcScorer(toy_state_dict(1), device="cpu")
    cands, links = [0, 2, 5, 6], [2, 3, 6]
    values, argmax = scorer.best(cands, links, chunk=2)
    for i, c in enumerate(cands):
        brute = {}
        for e in links:
            if e == c:
                continue
            for r in range(N_REL):
                brute[(e, r, DIRECTIONS[0])] = scorer.score([c], [r], [e])[0]
                brute[(e, r, DIRECTIONS[1])] = scorer.score([e], [r], [c])[0]
        key = max(brute, key=brute.get)
        assert values[i] == pytest.approx(brute[key], abs=1e-5)
        assert argmax[i] == key


def test_a_candidate_linked_only_to_itself_is_undefined():
    scorer = NativeKgcScorer(toy_state_dict(), device="cpu")
    values, argmax = scorer.best([3, 4], [3])
    assert np.isnan(values[0]) and argmax[0] is None
    assert np.isfinite(values[1]) and argmax[1][0] == 3


def test_channel_is_zero_without_links_and_fills_self_links_with_the_query_mean():
    scorer = NativeKgcScorer(toy_state_dict(), device="cpu")
    qid_of = {f"img{i}": f"Q{i}" for i in range(N_ENT)}
    nid_of = {f"Q{i}": i for i in range(N_ENT)}
    ranked = [["img0", "img1", "img2"], ["img3", "img4", "img1"]]
    linked = [{}, {"Q3": "three"}]
    out = channel(scorer, linked, ranked, qid_of, nid_of, k=3)
    assert out.shape == (2, 3) and not out[0].any()
    expected, _ = scorer.best([3, 4, 1], [3])
    assert out[1, 1:] == pytest.approx(expected[1:].astype(np.float32))
    assert out[1, 0] == pytest.approx(np.mean(out[1, 1:]))


def test_random_entities_keep_row_norms_and_the_random_emb_directions():
    ent = np.random.default_rng(5).standard_normal((10, 8)).astype(np.float32) * 3
    rnd = random_entities(ent, seed=2)
    np.testing.assert_allclose(np.linalg.norm(rnd, axis=1), np.linalg.norm(ent, axis=1), rtol=1e-5)
    unit = np.random.default_rng(2).standard_normal(ent.shape).astype(np.float32)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    np.testing.assert_allclose(rnd / np.linalg.norm(rnd, axis=1, keepdims=True), unit, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(random_entities(ent, seed=2), rnd)


def test_replacing_entity_vectors_changes_only_the_structural_part():
    sd = toy_state_dict()
    base = NativeKgcScorer(sd, device="cpu")
    same = NativeKgcScorer(sd, device="cpu", ent_embeddings=sd["ent_embeddings.weight"].numpy())
    assert torch.equal(base.parts, same.parts)
    other = NativeKgcScorer(sd, device="cpu", ent_embeddings=np.zeros((N_ENT, 2 * DIM), np.float32))
    assert torch.equal(base.parts[:, 1:], other.parts[:, 1:]) and not other.parts[:, 0].any()


def toy_channels(seed: int, n_queries: int = 12, k: int = 6) -> tuple[dict, list]:
    rng = np.random.default_rng(seed)
    ranked = [[f"q{q}c{c}" for c in range(k)] for q in range(n_queries)]
    golds = [row[int(rng.integers(0, k))] for row in ranked]
    ch = {"ranked": ranked, "clip": np.sort(rng.random((n_queries, k)).astype(np.float32))[:, ::-1].copy(),
          "emb": rng.random((n_queries, k)).astype(np.float32),
          "tri": rng.integers(0, 3, (n_queries, k)).astype(np.float32)}
    return ch, golds


def test_the_original_choice_keys_do_not_depend_on_the_kgc_channel():
    from scripts.eval_mkgw_level2 import KS, OLD_CHOICE_KEYS, choose_on_val
    from src.eval.metrics import evaluate
    ch, golds = toy_channels(0)
    ev = lambda lists: evaluate(lists, golds, ks=KS)  # noqa: E731
    before, sweep_before, kgc_sweep = choose_on_val(ev, ch)
    assert list(before) == list(OLD_CHOICE_KEYS) and kgc_sweep == {}
    for seed in (1, 2):
        with_kgc = {**ch, "kgc": np.random.default_rng(seed).normal(size=ch["emb"].shape).astype(np.float32)}
        after, sweep_after, kgc_sweep = choose_on_val(ev, with_kgc)
        assert {key: after[key] for key in OLD_CHOICE_KEYS} == before and sweep_after == sweep_before
        assert set(after) == set(OLD_CHOICE_KEYS) | {"native_kgc", "kgc_only"} and after["kgc_only"][1] == 1.0
        best = max(kgc_sweep.values(), key=lambda v: (v["recall@1"], v["mrr"]))
        assert kgc_sweep[tuple(after["native_kgc"])] == best
