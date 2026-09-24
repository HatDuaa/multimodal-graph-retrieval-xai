from collections import Counter

import numpy as np

from src.graph.corruption import graph_derangement, rewire_graph, rewire_graphs, triple_texts
from tests.test_graph_reranker import fake_store


def star_graph():
    objects = [{'id': k, 'name': f'o{k}'} for k in range(6)]
    relations = [{'subject': s, 'predicate': p, 'object': o}
                 for s, p, o in [(0, 'on', 1), (0, 'near', 2), (3, 'has', 4), (5, 'on', 0), (2, 'by', 3)]]
    return {'objects': objects, 'relations': relations}


def test_derangement_is_a_bijection_without_fixed_points():
    ids = list(range(10, 60))
    mapping = graph_derangement(ids, 3)
    assert sorted(mapping) == ids and sorted(mapping.values()) == ids
    assert all(k != v for k, v in mapping.items())
    assert mapping == graph_derangement(reversed(ids), 3)
    assert mapping != graph_derangement(ids, 4)


def test_rewire_keeps_degrees_labels_and_edge_count():
    graph = star_graph()
    changed = False
    for seed in range(20):
        out = rewire_graph(graph, np.random.default_rng(seed))
        assert out['objects'] == graph['objects']
        assert len(out['relations']) == len(graph['relations'])
        before, after = graph['relations'], out['relations']
        assert [(r['subject'], r['predicate']) for r in before] == [(r['subject'], r['predicate']) for r in after]
        assert Counter(r['object'] for r in before) == Counter(r['object'] for r in after)
        changed |= [r['object'] for r in before] != [r['object'] for r in after]
    assert changed
    assert graph['relations'][0]['object'] == 1  # the input is not modified


def test_rewire_graphs_is_seeded_per_image():
    graphs = {1: star_graph(), 2: star_graph()}
    assert rewire_graphs(graphs, 0) == rewire_graphs(dict(reversed(list(graphs.items()))), 0)
    assert triple_texts({1: star_graph()}) == {'o0 on o1', 'o0 near o2', 'o3 has o4', 'o5 on o0', 'o2 by o3'}


def test_replacing_image_graphs_changes_the_batch_and_extends_the_table():
    store = fake_store()
    clean = store.batch(['q'])
    rows_before = len(store.part_vectors)
    swapped = {1: store.graphs[2], 2: store.graphs[1]}
    store.replace_image_graphs(swapped, np.ones((2, 512), dtype='float32'), ['triple:new phrase', 'node:person'])
    assert len(store.part_vectors) == rows_before + 1  # an existing key is not duplicated
    assert store.part_row['triple:new phrase'] == rows_before
    corrupted = store.batch(['q'])
    assert clean['images'].x.shape != corrupted['images'].x.shape or not np.allclose(clean['images'].x, corrupted['images'].x)


def test_graph_store_opens_test_only_when_explicitly_allowed():
    import pytest
    from src.data.graph_store import GraphStore
    for kwargs in ({'splits': ('test',)}, {'splits': ('train', 'test')}, {'splits': ('train', 'test'), 'allow_test': True},
                   {'splits': ('val',), 'allow_test': True}, {'splits': ()}):
        with pytest.raises(ValueError):
            GraphStore(cfg={'paths': {}}, **kwargs)
