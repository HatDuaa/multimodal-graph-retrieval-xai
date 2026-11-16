"""Whole-graph matching probes: hop distances, FGW and the shared-node consistency score."""
import numpy as np
import pytest

from src.graph.graph_matching import consistency_score, fgw_score, hop_distance, shared_node_pairs

pytest.importorskip("ot")


def test_hop_distance_is_capped_and_scaled():
    dist = hop_distance(5, [(0, 1), (1, 2), (2, 3)])
    assert dist[0, 1] == pytest.approx(1 / 3) and dist[0, 2] == pytest.approx(2 / 3)
    assert dist[0, 3] == 1. and dist[0, 4] == 1. and dist[4, 4] == 0.
    np.testing.assert_allclose(dist, dist.T)


def two_horses():
    """Caption: man riding horse, horse on beach. Image A uses one horse; image B uses two different horses."""
    man, horse, beach = np.eye(3)
    query = np.stack([man, horse, beach]), [(0, 1), (1, 2)]
    image_a = np.stack([man, horse, beach]), [(0, 1), (1, 2)]
    image_b = np.stack([man, horse, horse, beach]), [(0, 1), (2, 3)]
    return query, image_a, image_b


@pytest.mark.parametrize("mode", ["balanced", "semirelaxed"])
def test_fgw_prefers_the_image_where_the_triples_share_the_horse(mode):
    (qv, qe), (av, ae), (bv, be) = two_horses()
    qd = hop_distance(len(qv), qe)
    score_a, plan = fgw_score(qv, qd, av, hop_distance(len(av), ae), mode=mode)
    score_b, _ = fgw_score(qv, qd, bv, hop_distance(len(bv), be), mode=mode)
    assert score_a > score_b
    assert list(plan.argmax(-1)) == [0, 1, 2]
    assert np.isnan(fgw_score(qv[:0], qd[:0, :0], av, hop_distance(3, ae))[0])


def test_consistency_checks_the_shared_node_of_the_best_matching_triples():
    (_, qe), (_, ae), (_, be) = two_horses()
    pairs = shared_node_pairs(qe)
    assert pairs == [(0, 1, 1, 0)]
    riding, on = np.eye(2)
    queries = np.stack([riding, on])
    assert consistency_score(pairs, queries, np.stack([riding, on]), ae) == 1.
    assert consistency_score(pairs, queries, np.stack([riding, on]), be) == 0.
    assert np.isnan(consistency_score([], queries, np.stack([riding, on]), ae))
