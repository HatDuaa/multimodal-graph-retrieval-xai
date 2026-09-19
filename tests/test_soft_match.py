"""Soft matching and re-ranking on hand-made unit vectors."""
import numpy as np

from src.graph.soft_match import image_parts, match, query_parts, rerank, zscore_rows

E = np.eye(3, dtype=np.float32)


def test_parts_drop_pronouns_and_duplicates():
    query = {"objects": [{"id": 0, "name": "child"}, {"id": 1, "name": "ball"}, {"id": 2, "name": "it"}],
             "relations": [{"subject": 0, "predicate": "Playing With", "object": 1}, {"subject": 2, "predicate": "on", "object": 1}]}
    assert query_parts(query) == (["child", "ball"], ["child playing with ball"])
    graph = {"objects": [{"id": 0, "name": "boy"}, {"id": 1, "name": "ball"}, {"id": 2, "name": "ball"}],
             "relations": [{"subject": 0, "predicate": "holding", "object": 1}, {"subject": 0, "predicate": "holding", "object": 2}]}
    assert image_parts(graph) == (["boy", "ball"], ["boy holding ball"])


def test_match_max_ignores_unrelated_image_parts():
    score, details = match(E[:1], E, "max")
    assert score == 1.0 and details == [(0, 1.0, 1.0)]
    score, _ = match(E[:2], E[:1], "max")           # second query part has no counterpart
    assert score == 0.5


def test_softmax_tends_to_max_and_is_diluted_when_flat():
    sharp, _ = match(E[:1], E, "softmax", temperature=0.01)
    flat, _ = match(E[:1], E, "softmax", temperature=100.0)
    assert abs(sharp - 1.0) < 1e-6 and abs(flat - 1 / 3) < 1e-2


def test_no_parts_means_no_evidence():
    score, details = match(np.zeros((0, 3), dtype=np.float32), E, "max")
    assert np.isnan(score) and details == []


def test_zscore_and_rerank():
    z = zscore_rows(np.array([[1.0, 3.0, np.nan], [2.0, 2.0, 2.0]]))
    assert z.tolist() == [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    clip = np.array([[0.9, 0.8, 0.7]])
    graph = np.array([[0.1, 0.2, 0.9]])
    ids = [["a", "b", "c"]]
    assert rerank(clip, graph, ids, 1.0) == [["a", "b", "c"]]
    assert rerank(clip, graph, ids, 0.0) == [["c", "b", "a"]]
    assert rerank(clip, np.full((1, 3), np.nan), ids, 0.5) == [["a", "b", "c"]]   # no graph evidence keeps the CLIP order
