from pathlib import Path

import numpy as np
import pytest

from src.service.search_service import SearchService

VOCAB = ["dog", "cat", "car", "tree"]


class FakeEncoder:
    """One axis per word, so a query is closest to the image whose vector shares its words."""
    name = "fake"

    def encode_texts(self, texts):
        out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                if w in VOCAB:
                    out[r, VOCAB.index(w)] = 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


@pytest.fixture
def service():
    feats = np.eye(len(VOCAB), dtype=np.float32)          # image i shows exactly VOCAB[i]
    ids = [100, 101, 102, 103]
    meta = {i: {"path": Path(f"{i}.jpg"), "captions": [f"a {w}"]} for i, w in zip(ids, VOCAB)}
    return SearchService(feats, ids, meta, FakeEncoder(), pool_k=4)


def test_clip_mode_ranks_matching_image_first(service):
    hits = service.search("a cat", k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].image_id == 101 and hits[0].score == pytest.approx(1.0)
    assert hits[0].captions == ["a cat"] and hits[0].explanation == [] and hits[0].graph_score is None


def test_blank_query_and_unknown_mode(service):
    assert service.search("   ") == []
    with pytest.raises(ValueError):
        service.search("dog", mode="nope")


def test_reranker_can_reorder_and_explain(service):
    def prefer_tree(query, hits):
        for h in hits:
            h.graph_score = 1.0 if h.image_id == 103 else 0.0
            h.score = 0.5 * h.clip_score + 0.5 * h.graph_score
            if h.graph_score:
                h.explanation = [[("c:dog", "near", "c:tree")]]
        return hits

    service.add_mode("clip_graph", prefer_tree)
    # CLIP alone puts the dog image first; the re-ranker sees the whole pool and lifts image 103
    assert service.search("dog", mode="clip", k=1)[0].image_id == 100
    hits = service.search("dog", mode="clip_graph", k=2)
    assert {hits[0].image_id, hits[1].image_id} == {100, 103}
    assert [h.rank for h in hits] == [1, 2]
    top_tree = next(h for h in hits if h.image_id == 103)
    assert top_tree.clip_score == pytest.approx(0.0) and top_tree.explanation


def test_rank_of(service):
    assert service.rank_of("car", 102) == 1
    assert service.rank_of("car", 999) is None
