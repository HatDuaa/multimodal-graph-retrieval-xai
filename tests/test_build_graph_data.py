import numpy as np

from scripts.build_graph_data import build_candidates


def test_build_candidates_stays_inside_pool_and_records_gold_rank():
    pool = [10, 11, 12]
    pool_features = np.eye(3, dtype=np.float32)
    queries = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
    result = build_candidates(queries, ["a", "b"], [11, 99], pool_features, pool, 4, top_k=2)
    assert result["candidate_ids"].shape == (2, 2)
    assert set(result["candidate_ids"].ravel()).issubset(pool)
    assert result["gold_rank"].tolist() == [1, -1]
    assert result["group_index"].tolist() == [4, 4]
