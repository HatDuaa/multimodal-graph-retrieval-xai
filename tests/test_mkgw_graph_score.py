import numpy as np
import pytest

from src.retrieval.mkgw_graph_score import EntityLinker, MkgwGraphScorer, fuse, zscore

NAMES = {
    "Q1": ["John Ford"],
    "Q2": ["Lolita"],
    "Q3": ["New York City"],
    "Q4": ["York"],
    "Q5": ["film"],   # generic word that IS an entity name
}


def make_linker():
    return EntityLinker(NAMES)


def test_linker_matches_word_bounded_and_case_insensitive():
    linked = make_linker().link("1962 film by JOHN FORD")
    assert linked["Q1"] == "john ford"
    assert "Q5" in linked          # "film" matched too
    # "afford" must not match "Ford"-like fragments
    assert make_linker().link("they can afford it") == {}


def test_linker_longest_match_wins():
    linked = make_linker().link("born in New York City")
    assert "Q3" in linked
    assert "Q4" not in linked      # "York" consumed by the longer match


def test_linker_exclude_answer_entity():
    linked = make_linker().link("Lolita is a film", exclude={"Q2"})
    assert "Q2" not in linked and "Q5" in linked


def make_scorer():
    feats = np.eye(4, dtype=np.float32)     # Q1..Q4 orthogonal
    feats[3] = feats[0]                     # Q4 identical to Q1
    index = {f"Q{i+1}": i for i in range(4)}
    triples = [{"h_qid": "Q2", "t_qid": "Q1", "r_pid": "P57", "kgc_split": "train"}]
    return MkgwGraphScorer(feats, index, triples, {"P57": "director"})


def test_embedding_channel_takes_best_match():
    s = make_scorer().score({"Q1": "john ford", "Q3": "new york city"}, "Q4")
    assert s["emb"] == pytest.approx(1.0)   # Q4 == Q1
    best = [m for m in s["matches"] if m["via"] == "embedding"]
    assert len(best) == 1 and best[0]["qid"] == "Q1"


def test_triple_channel_and_explanation():
    s = make_scorer().score({"Q1": "john ford"}, "Q2")
    assert s["triple"] == 1.0
    t = [m for m in s["matches"] if m["via"] == "triple"][0]
    assert (t["pid"], t["label"], t["direction"]) == ("P57", "director", "->")


def test_score_without_removes_the_used_triple():
    scorer = make_scorer()
    linked = {"Q1": "john ford"}
    s = scorer.score(linked, "Q2")
    s2 = scorer.score_without(linked, "Q2", removed=s["matches"])
    assert s2["triple"] == 0.0 and s2["emb"] is None


def test_no_link_is_neutral():
    s = make_scorer().score({}, "Q2")
    assert s["emb"] is None and s["triple"] == 0.0 and s["matches"] == []


def test_only_train_triples_accepted():
    with pytest.raises(AssertionError):
        MkgwGraphScorer(np.eye(2, dtype=np.float32), {"Q1": 0, "Q2": 1},
                        [{"h_qid": "Q1", "t_qid": "Q2", "r_pid": "P1", "kgc_split": "test"}])


def test_fuse_alpha_one_is_pure_clip_order():
    clip = np.array([3.0, 2.0, 1.0])
    emb = np.array([1.0, 2.0, 3.0])
    tri = np.zeros(3)
    fused = fuse(clip, emb, tri, alpha=1.0, beta=0.5)
    assert list(np.argsort(-fused)) == [0, 1, 2]
    fused0 = fuse(clip, emb, tri, alpha=0.0, beta=1.0)
    assert list(np.argsort(-fused0)) == [2, 1, 0]


def test_zscore_constant_input_is_finite():
    assert np.isfinite(zscore(np.ones(5))).all()
