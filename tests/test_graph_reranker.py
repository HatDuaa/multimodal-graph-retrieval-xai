import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from src.data.graph_store import GraphStore
from src.explain.graph_explainer import remove_parts, score, score_without
from src.graph.soft_match import fuse_three_channels
from src.models.graph_reranker import GraphReranker, fuse_tensor, zscore_tensor
from src.train_graph_reranker import shuffled_rows


def test_training_order_is_deterministic_but_changes_each_epoch():
    rows = [(i, str(i)) for i in range(100)]
    first = shuffled_rows(rows, 7, 1)
    assert first == shuffled_rows(rows, 7, 1)
    assert first != shuffled_rows(rows, 7, 2)


def test_microbatch_mean_loss_has_full_batch_weighting():
    torch.manual_seed(3)
    logits = torch.randn(256, 50, requires_grad=True)
    targets = torch.arange(256) % 50
    full = F.cross_entropy(logits, targets)
    full.backward()
    full_grad = logits.grad.detach().clone()
    logits.grad.zero_()
    for start in range(0, 256, 32):
        part = F.cross_entropy(logits[start:start + 32], targets[start:start + 32])
        (part * (32 / 256)).backward()
    torch.testing.assert_close(logits.grad, full_grad, atol=1e-6, rtol=1e-6)


def graph(n=4, edges=True):
    x = F.normalize(torch.randn(n, 512), dim=-1)
    pairs = torch.tensor([[0, 1, 2], [1, 2, 3]]) if edges and n >= 4 else torch.empty((2, 0), dtype=torch.long)
    e = pairs.shape[1]
    return Data(x=x, edge_index=pairs, edge_attr=F.normalize(torch.randn(e, 512), dim=-1),
                triple_f=F.normalize(torch.randn(e, 512), dim=-1), part_mask=torch.ones(n, dtype=torch.bool))


def batch():
    torch.manual_seed(14)
    return {'queries': Batch.from_data_list([graph(), graph()]),
            'images': Batch.from_data_list([graph() for _ in range(5)]),
            'candidate_index': torch.tensor([[0, 1, 2, 3], [4, 2, 1, 0]]),
            'clip': torch.randn(2, 4), 'sentence': F.normalize(torch.randn(2, 512), dim=-1)}


def loss(model, data):
    return F.cross_entropy(model(data)[0] / model.loss_temperature(), torch.tensor([1, 2]))


def test_zero_init_and_missing_gradients_are_finite():
    model = GraphReranker(dropout=0.).eval()
    data = batch()
    encoded = model.encode(data['queries'], query=True)
    torch.testing.assert_close(encoded[0][encoded[1]], data['queries'].x, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(encoded[2][encoded[3]], data['queries'].triple_f, atol=1e-7, rtol=1e-6)
    model.train()
    loss(model, data).backward()
    assert model.u.weight.grad.abs().sum() > 0
    assert model.gat1.lin_l.weight.grad.abs().sum() == 0  # U=0 blocks upstream gradients initially.
    inputs = torch.tensor([[1., 1., float('nan')], [1., float('nan'), float('nan')],
                           [float('nan')] * 3], requires_grad=True)
    z, active = zscore_tensor(inputs)
    assert not active.any()
    z.sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_fusion_matches_numpy_with_missing_channels():
    rng = np.random.default_rng(4)
    clip, obj, tri = rng.normal(size=(3, 6, 8)).astype('float32')
    obj[0] = np.nan
    tri[1] = np.nan
    obj[2] = tri[2] = np.nan
    obj[3, :7] = np.nan
    tri[4] = 1.
    obj[5, 2:5] = np.nan
    weights = torch.tensor([[.6, .28, .12]]).expand(6, -1)
    fused, _ = fuse_tensor(torch.tensor(clip), torch.tensor(obj), torch.tensor(tri), weights)
    expected = np.array([fuse_three_channels(c, o, t, .6, .7) for c, o, t in zip(clip, obj, tri)])
    np.testing.assert_allclose(fused.numpy(), expected, atol=1e-6)


def test_gradients_after_residual_update_and_gate_warmup():
    model, data = GraphReranker(dropout=0.), batch()
    model.set_epoch(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss(model, data).backward()
    assert model.weight_map.grad is None and model.weight_bias.grad is None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.set_epoch(2)
    loss(model, data).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in ('gat1.lin_l.weight', 'gat2.lin_l.weight', 'triple_mlp.0.weight', 'p_node.weight',
                 'p_edge.weight', 'u.weight', 'u_objects', 'u_triples', 'log_tau_objects', 'log_tau_triples',
                 'log_temperature', 'weight_map', 'weight_bias'):
        assert dict(model.named_parameters())[name].grad.norm() > 0, name


@pytest.mark.parametrize('option', ['use_gat', 'use_edges', 'use_triple_channel', 'query_graph_encoder', 'adaptive_weights'])
def test_ablation_changes_computation(option):
    model, data = GraphReranker(**{option: False}, dropout=0.), batch()
    with torch.no_grad():
        model.u.weight.normal_(0, .01)
    model.set_epoch(2)
    scores, details = model(data)
    loss(model, data).backward()
    if option == 'use_gat':
        assert all(p.grad is None for p in model.gat1.parameters())
        assert model.p_node.weight.grad.norm() > 0
    if option in ('use_edges', 'use_triple_channel'):
        assert details['triples'].isnan().all()
        assert all(p.grad is None for p in model.triple_mlp.parameters())
    if option == 'use_edges':
        assert model.gat1.lin_l.weight.grad.norm() > 0  # self-loop GAT remains active
    if option == 'query_graph_encoder':
        encoded = model.encode(data['queries'], query=True)
        torch.testing.assert_close(encoded[0][encoded[1]], data['queries'].x, rtol=0, atol=0)
    if option == 'adaptive_weights':
        assert model.weight_map.grad is None
        assert model.weight_bias.grad.norm() > 0


def fake_store():
    torch.manual_seed(2)
    store = GraphStore.__new__(GraphStore)
    store.part_dim = 512
    keys = ['node:person', 'node:ball', 'rel:holding', 'triple:person holding ball']
    store.part_row = {key: i for i, key in enumerate(keys)}
    store.part_vectors = F.normalize(torch.randn(4, 512), dim=-1).numpy()
    raw = {'objects': [{'id': 3, 'name': 'person'}, {'id': 8, 'name': 'ball'}],
           'relations': [{'subject': 3, 'predicate': 'holding', 'object': 8}]}
    store.graphs = {1: raw, 2: {'objects': [{'id': 0, 'name': 'ball'}], 'relations': []}}
    store.queries = {'q': raw}
    store.caption_vectors = F.normalize(torch.randn(1, 512), dim=-1).numpy()
    store.caption_row = {'q': 0}
    store.arrays = {'val': {'candidate_ids': np.array([[1, 2]]), 'clip_scores': np.array([[.5, .2]], dtype='float32'),
                             'gold_rank': np.array([1]), 'gold': np.array([1])}}
    store.caption_location = {'q': ('val', 0)}
    return store


def test_explanations_use_candidate_context_and_remove_indexed_parts():
    store, model = fake_store(), GraphReranker(dropout=0.).eval()
    expected = score(model, store, 'q', 1)
    assert expected == score_without(model, store, 'q', 1)
    assert expected['score'] != 0  # A single-candidate z-score would wrongly produce zero.
    removed = score_without(model, store, 'q', 1, [0, 1], [])
    assert removed['graph'] == 0
    assert removed['objects'] == [] and removed['triples'] == []
    assert len(store.graphs[1]['objects']) == 2
    remaining = remove_parts(store.graphs[1], [0], [])
    assert remaining['objects'][0]['id'] == 8 and remaining['relations'] == []


def test_packed_batch_equals_pyg_batch_of_per_graph_data():
    store = fake_store()
    store.graphs[3] = {'objects': [], 'relations': []}
    store.arrays['val']['candidate_ids'] = np.array([[2, 3, 1]])
    store.arrays['val']['clip_scores'] = np.array([[.5, .2, .1]], dtype='float32')
    packed = store.batch(['q'])
    table = torch.from_numpy(store.part_vectors)
    expected = []
    for image_id in packed['image_ids']:
        parts = store._graph_parts(store.graphs[int(image_id)])
        rows = lambda prefix, labels: table[[store.part_row[f'{prefix}:{x}'] for x in labels]].reshape(-1, 512)  # noqa: E731
        expected.append(Data(x=rows('node', parts['node_labels']),
                             edge_index=torch.tensor([[e[0] for e in parts['edges']], [e[1] for e in parts['edges']]],
                                                     dtype=torch.long).reshape(2, -1),
                             edge_attr=rows('rel', parts['relation_labels']),
                             triple_f=rows('triple', parts['triple_labels']),
                             part_mask=torch.tensor(parts['part_mask'], dtype=torch.bool)))
    expected = Batch.from_data_list(expected)
    for key in ('x', 'edge_index', 'edge_attr', 'triple_f', 'part_mask', 'batch'):
        torch.testing.assert_close(getattr(packed['images'], key), getattr(expected, key), rtol=0, atol=0)
    assert packed['images'].num_graphs == expected.num_graphs == 3
    assert packed['image_ids'][packed['candidate_index'][0].numpy()].tolist() == [2, 3, 1]


def test_override_graph_keeps_candidate_mapping():
    store = fake_store()
    modified = remove_parts(store.graphs[1], [1], [])
    packed = store.batch(['q'], overrides={1: modified})
    assert packed['image_ids'][packed['candidate_index'][0].numpy()].tolist() == [1, 2]
    row = int(packed['candidate_index'][0, 0])
    assert int((packed['images'].batch == row).sum()) == 1
