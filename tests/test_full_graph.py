"""Full image graphs (experiment 3): edge geometry, completion edges, global node, and their default-off behaviour."""
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.data.graph_store import EDGE_COMPLETION, EDGE_GLOBAL, EDGE_LABELLED, GraphIndex, GraphStore
from src.graph.structure import completion_edges, components, edge_geometry, structure_stats
from src.models.graph_reranker import GraphReranker
from src.train_graph_reranker import build_model, structure_options


def test_edge_geometry_of_a_box_to_the_right():
    s, o = [0, 0, 10, 10], [10, 0, 20, 10]
    dx, dy, lw, lh, iou, share = edge_geometry(s, o)
    assert dx == pytest.approx(1.5) and dy == 0
    assert lw == pytest.approx(math.log(2)) and lh == 0
    assert iou == 0 and share == pytest.approx(200 / 300)
    assert edge_geometry(s, s)[4] == pytest.approx(1.)


def test_edge_geometry_clips_offsets_of_tiny_boxes():
    dx, dy, *_ = edge_geometry([0, 0, 0, 0], [500, -500, 10, 10])
    assert dx == 10 and dy == -10  # a 0-px box counts as 1 px, then the offset is clipped


def boxes_in_two_clusters():
    left = [[x * 5, 0, 4, 4] for x in range(6)]
    right = [[1000 + x * 5, 0, 4, 4] for x in range(6)]
    return left + right


def test_every_node_receives_k_nearest_edges_and_bridges_join_components():
    boxes = boxes_in_two_clusters()
    knn, bridges = completion_edges(boxes, 1100, 100, 5)
    incoming = np.bincount([t for _, t in knn], minlength=12)
    assert incoming.tolist() == [5] * 12
    assert all(s != t for s, t in knn)
    assert all((s < 6) == (t < 6) for s, t in knn)  # k = 5 stays inside each cluster of 6
    assert sorted(bridges) == [(5, 6), (6, 5)]  # the closest pair across the two clusters, both directions
    assert structure_stats(12, knn + bridges) == {"nodes": 12, "components": 1, "largest_share": 1.}


def test_completion_skips_labelled_pairs_and_breaks_ties_by_index():
    boxes = [[0, 0, 2, 2], [10, 0, 2, 2], [-10, 0, 2, 2]]  # nodes 1 and 2 are equally far from node 0
    knn, bridges = completion_edges(boxes, 100, 100, 1, labelled=[(1, 0)])
    assert (1, 0) not in knn and (0, 1) in knn and (0, 2) in knn
    assert bridges == [] and len(set(components(3, knn + [(1, 0)]).tolist())) == 1
    knn, _ = completion_edges(boxes, 100, 100, 1)
    assert (1, 0) in knn and (2, 0) not in knn


def test_small_graphs_connect_to_every_other_node():
    knn, bridges = completion_edges([[0, 0, 1, 1], [5, 5, 1, 1]], 10, 10, 5)
    assert sorted(knn) == [(0, 1), (1, 0)] and bridges == []
    assert completion_edges([[0, 0, 1, 1]], 10, 10, 5) == ([], [])


def image_graph():
    return {'image_id': 1, 'width': 100, 'height': 100,
            'objects': [{'id': 0, 'name': 'person', 'box': [0, 0, 10, 10]}, {'id': 1, 'name': 'ball', 'box': [12, 0, 4, 4]},
                        {'id': 2, 'name': 'ball', 'box': [90, 90, 5, 5]}],
            'relations': [{'subject': 0, 'predicate': 'holding', 'object': 1}]}


def test_graph_parts_list_added_edges_after_labelled_ones_with_geometry():
    graph = image_graph()
    plain = GraphStore._graph_parts(graph)
    assert 'added_edges' not in plain and len(plain['edge_geometry']) == 1
    parts = GraphStore._graph_parts(graph, knn_edges=1)
    assert parts['edges'] == plain['edges']
    assert all(kind == EDGE_COMPLETION for _, _, kind in parts['added_edges'])
    assert len(parts['edge_geometry']) == 1 + len(parts['added_edges'])
    pairs = [e[:2] for e in parts['edges']] + [e[:2] for e in parts['added_edges']]
    assert structure_stats(3, pairs)['components'] == 1
    query = GraphStore._graph_parts(graph, query=True, knn_edges=1)
    assert 'added_edges' not in query and 'edge_geometry' not in query
    with pytest.raises(ValueError):
        GraphStore._graph_parts({**graph, 'objects': [{'id': 0, 'name': 'a'}, {'id': 1, 'name': 'b'}]}, knn_edges=1)


def fake_store(structure=None):
    torch.manual_seed(2)
    store = GraphStore.__new__(GraphStore)
    store.part_dim = 512
    keys = ['node:person', 'node:ball', 'rel:holding', 'triple:person holding ball']
    store.part_row = {key: i for i, key in enumerate(keys)}
    store.part_vectors = F.normalize(torch.randn(4, 512), dim=-1).numpy()
    store.graphs = {1: image_graph(), 2: {'width': 50, 'height': 50, 'objects': [{'id': 0, 'name': 'ball', 'box': [1, 1, 3, 3]}],
                                          'relations': []}}
    store.queries = {'q': {'objects': [{'id': 3, 'name': 'person'}, {'id': 8, 'name': 'ball'}],
                           'relations': [{'subject': 3, 'predicate': 'holding', 'object': 8}]}}
    store.caption_vectors = F.normalize(torch.randn(1, 512), dim=-1).numpy()
    store.caption_row = {'q': 0}
    store.arrays = {'val': {'candidate_ids': np.array([[1, 2]]), 'clip_scores': np.array([[.5, .2]], dtype='float32'),
                            'gold_rank': np.array([1]), 'gold': np.array([1])}}
    store.caption_location = {'q': ('val', 0)}
    if structure:
        store.set_structure(**structure)
    return store


def test_default_store_batches_keep_the_old_fields():
    packed = fake_store().batch(['q'])
    for side in ('queries', 'images'):
        assert packed[side].edge_kind is None and packed[side].labelled_index is None and packed[side].global_mask is None
    assert packed['queries'].edge_geometry is None
    assert packed['images'].edge_geometry.shape == (1, 6)  # image 1's labelled edge has boxes


def test_global_node_and_completion_edges_in_the_index():
    store = fake_store({'knn_edges': 1, 'global_node': True})
    store.precompute()
    images, queries = store.image_index, store.query_index
    assert images.node_counts.tolist() == [4, 2]  # 3 + global, 1 + global
    assert images.labelled_counts().tolist() == [1, 0] and queries.labelled_counts().tolist() == [1]
    kinds = images.fields['edge_kind'][:images.edge_counts[0]]
    assert kinds[0] == EDGE_LABELLED and (kinds[1:] != EDGE_LABELLED).all()
    assert (kinds == EDGE_GLOBAL).sum() == 6
    assert images.fields['global_mask'].tolist() == [False, False, False, True, False, True]
    assert not images.fields['part_mask'][images.fields['global_mask']].any()
    packed = store.batch(['q'])
    batch = packed['images']
    assert batch.edge_attr.shape == (1, 512) and batch.triple_f.shape == (1, 512)
    assert batch.labelled_index.tolist() == [0]
    torch.testing.assert_close(batch.edge_attr[0], torch.from_numpy(store.part_vectors[2]))
    geometry = batch.edge_geometry
    assert torch.isfinite(geometry[batch.edge_kind != EDGE_GLOBAL]).all()
    assert torch.isnan(geometry[batch.edge_kind == EDGE_GLOBAL]).all()
    assert packed['queries'].edge_geometry is None and int((packed['queries'].edge_kind == EDGE_GLOBAL).sum()) == 4


def test_set_structure_rebuilds_the_index():
    store = fake_store()
    store.precompute()
    assert store.image_index.edge_counts.tolist() == [1, 0]
    store.set_structure(knn_edges=2)
    store.precompute()
    assert store.image_index.edge_counts[0] > 1
    assert structure_options({'knn_edges': 5}) == {'knn_edges': 5, 'global_node': False}
    assert structure_options({}) == {'knn_edges': 0, 'global_node': False}


def scores(model, store):
    torch.manual_seed(0)
    return model(store.batch(['q']))[0]


def test_zero_initialized_geometry_starts_like_the_model_without_it():
    store = fake_store()
    torch.manual_seed(5)
    plain = GraphReranker(dropout=0.).eval()
    with_geometry = GraphReranker(dropout=0., edge_geometry=True).eval()
    with_geometry.load_state_dict(plain.state_dict(), strict=False)
    torch.nn.init.normal_(with_geometry.u.weight)  # non-zero u, so the GAT output reaches the scores
    plain.u.weight.data.copy_(with_geometry.u.weight.data)
    torch.testing.assert_close(scores(plain, store), scores(with_geometry, store))
    raw = GraphReranker(dropout=0., edge_geometry=True, fusion='raw')
    torch.nn.init.normal_(raw.u.weight)
    raw(store.batch(['q']))[0][0, 0].backward()  # one raw score: a z-score of two candidates is always +-1
    assert raw.geometry_mlp[2].weight.grad.abs().sum() > 0


def test_completion_edges_change_gat_messages_only():
    store, full = fake_store(), fake_store({'knn_edges': 2})
    torch.manual_seed(5)
    model = GraphReranker(dropout=0., use_gat=False, completion_edges=True).eval()
    torch.nn.init.normal_(model.u.weight)
    _, plain_details = model(store.batch(['q']))
    _, full_details = model(full.batch(['q']))
    torch.testing.assert_close(plain_details['triples'], full_details['triples'], equal_nan=True)
    torch.testing.assert_close(plain_details['objects'], full_details['objects'], equal_nan=True)
    gat = GraphReranker(dropout=0., completion_edges=True).eval()
    torch.nn.init.normal_(gat.u.weight)
    encoded = gat.encode(full.batch(['q'])['images'])
    assert int(encoded[3].sum()) == 1  # still one triple: the labelled edge
    assert not torch.allclose(gat.encode(store.batch(['q'])['images'])[0], encoded[0])
    raw = GraphReranker(dropout=0., completion_edges=True, fusion='raw')
    torch.nn.init.normal_(raw.u.weight)
    raw(full.batch(['q']))[0][0, 0].backward()
    assert raw.none_predicate.grad.abs().sum() > 0


def test_global_node_is_mean_initialized_and_not_an_object():
    store = fake_store({'global_node': True})
    model = GraphReranker(dropout=0., global_node=True).eval()
    batch = store.batch(['q'])['images']
    nodes, node_mask, _, _ = model.encode(batch)
    assert node_mask.sum(-1).tolist() == [3, 1]  # global nodes excluded from the object channel
    with pytest.raises(ValueError):
        GraphReranker(dropout=0.).eval().encode(batch)
    model.record_attention = True
    model.encode(batch)
    (index, alpha), _ = model.attention
    assert alpha.shape == (index.shape[1], 4)
    expected = F.normalize(torch.from_numpy(store.part_vectors[[0, 1, 1]]).sum(0), dim=0)
    seen = {}

    def hook(_, args):
        seen['x'] = args[0]

    handle = model.p_node.register_forward_pre_hook(hook)
    model.encode(batch)
    handle.remove()
    torch.testing.assert_close(seen['x'][3], expected)


def test_build_model_reads_the_new_flags_with_old_defaults():
    old = {'adaptive_weights': True, 'use_gat': True, 'use_triple_channel': True, 'use_edges': True,
           'query_graph_encoder': True}
    model = build_model(old, 'cpu')
    assert not hasattr(model, 'geometry_mlp') and not hasattr(model, 'none_predicate')
    model = build_model({**old, 'edge_geometry': True, 'knn_edges': 5, 'global_node': True}, 'cpu')
    assert hasattr(model, 'geometry_mlp') and hasattr(model, 'none_predicate') and hasattr(model, 'global_predicate')
    assert GraphIndex.NODE_FIELDS[-1] == 'global_mask'
