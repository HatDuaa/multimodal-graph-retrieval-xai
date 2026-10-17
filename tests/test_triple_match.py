"""Factorized triple matching and the graph-level channel (improvement phase, steps 2 and 3)."""
import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from src.explain.graph_explainer import score
from src.models.graph_reranker import GraphReranker, attention_pool, part_similarity, slot_similarity
from src.train_graph_reranker import build_model
from tests.test_graph_reranker import batch, fake_store, loss


def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def two_node_graph(nodes, relation, phrase, reverse=False):
    edges = torch.tensor([[1], [0]]) if reverse else torch.tensor([[0], [1]])
    return Data(x=nodes, edge_index=edges, edge_attr=relation[None], triple_f=phrase[None],
                part_mask=torch.ones(2, dtype=torch.bool))


def test_default_model_keeps_level1_parameters_and_options():
    model = GraphReranker()
    assert model.options['triple_match'] == 'phrase' and not model.options['graph_channel']
    assert not any(name.startswith('u_pool') for name in model.state_dict())
    old_run_args = dict(adaptive_weights=True, use_gat=True, use_triple_channel=True, use_edges=True,
                        query_graph_encoder=True)
    assert build_model(old_run_args, 'cpu').options['triple_match'] == 'phrase'
    with pytest.raises(ValueError):
        GraphReranker(triple_match='factorized', graph_channel=True)


def test_factorized_similarity_is_the_product_of_slot_similarities():
    torch.manual_seed(0)
    query, images = unit(2, 3, 3, 512), unit(6, 5, 3, 512)
    candidate_index = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 0]])
    sims = part_similarity(query, images, candidate_index)
    assert sims.shape == (2, 4, 3, 5)
    b, k, i, j = 1, 2, 0, 3
    expected = 1.
    for slot in range(3):
        expected *= (1 + query[b, i, slot] @ images[candidate_index[b, k], j, slot]) / 2
    torch.testing.assert_close(sims[b, k, i, j], expected)
    assert ((sims >= 0) & (sims <= 1)).all()
    phrase = part_similarity(query[..., 0, :], images[..., 0, :], candidate_index)
    candidates = images[candidate_index][..., 0, :]
    torch.testing.assert_close(phrase, torch.matmul(query[:, None, :, 0], candidates.transpose(-1, -2)))
    # The per-slot checkpoint gives the same gradients as the plain product.
    query.requires_grad_(True)
    part_similarity(query, images, candidate_index).sum().backward()
    grad = query.grad.clone()
    query.grad = None
    plain = 1.
    for slot in range(3):
        plain = plain * (1 + torch.matmul(query[:, None, :, slot], images[candidate_index][..., slot, :].transpose(-1, -2))) / 2
    plain.sum().backward()
    torch.testing.assert_close(query.grad, grad)


def test_factorized_encoding_at_init_uses_node_and_relation_vectors():
    torch.manual_seed(1)
    model = GraphReranker(triple_match='factorized', dropout=0.).eval()
    data = batch()
    triples, mask = model.encode(data['queries'], query=True)[2:]
    graphs = data['queries']
    src, dst = graphs.edge_index
    expected = torch.stack((graphs.x[src], graphs.edge_attr, graphs.x[dst]), -2)
    torch.testing.assert_close(triples[mask], expected, atol=1e-6, rtol=1e-6)


def test_factorized_match_sees_subject_object_order():
    torch.manual_seed(2)
    nodes, relation, phrase = unit(2, 512), unit(512), unit(512)
    model = GraphReranker(triple_match='factorized', dropout=0.).eval()
    # Same phrase vector for both directions: the bag-of-words limit of a CLIP phrase.
    data = {'queries': Batch.from_data_list([two_node_graph(nodes, relation, phrase)]),
            'images': Batch.from_data_list([two_node_graph(nodes, relation, phrase),
                                            two_node_graph(nodes, relation, phrase, True)]),
            'candidate_index': torch.tensor([[0, 1]]), 'clip': torch.zeros(1, 2), 'sentence': unit(1, 512)}
    triples = model(data)[1]['triples'][0]
    torch.testing.assert_close(triples[0], torch.tensor(1.))
    assert triples[1] < triples[0] - .1
    phrase = GraphReranker(dropout=0.).eval()(data)[1]['triples'][0]
    assert phrase[0] == phrase[1]  # the phrase match at init cannot tell the reversed edge apart


@pytest.mark.parametrize('kwargs', [dict(triple_match='factorized'), dict(graph_channel=True)])
def test_new_modes_train_with_finite_gradients(kwargs):
    model, data = GraphReranker(dropout=0., **kwargs), batch()
    model.set_epoch(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss(model, data).backward()
    optimizer.step()  # moves U away from zero so the GAT receives gradients
    optimizer.zero_grad(set_to_none=True)
    loss(model, data).backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
    for name in ('gat1.lin_l.weight', 'p_node.weight', 'u.weight', 'u_objects', 'weight_bias'):
        assert grads[name].norm() > 0, name
    assert all(grads[name] is None for name in grads if name.startswith('triple_mlp'))
    if kwargs.get('triple_match') == 'factorized':
        assert grads['u_triples'].norm() > 0 and grads['log_tau_triples'].norm() > 0
    else:
        assert grads['u_pool_query'].norm() > 0 and grads['u_pool_image'].norm() > 0
        assert grads['u_triples'] is None and grads['log_tau_triples'] is None


def test_graph_channel_is_cosine_of_mean_pooled_nodes_at_init():
    torch.manual_seed(3)
    model, data = GraphReranker(graph_channel=True, dropout=0.).eval(), batch()
    _, details = model(data)
    queries, images = data['queries'], data['images']
    mean = lambda graphs, g: F.normalize(graphs.x[graphs.batch == g].mean(0), dim=-1)  # noqa: E731
    for b in range(2):
        for k in range(4):
            image = int(data['candidate_index'][b, k])
            expected = mean(queries, b) @ mean(images, image)
            torch.testing.assert_close(details['triples'][b, k], expected, atol=1e-6, rtol=1e-6)
    assert details['pool_query'].shape[0] == 2 and details['pool_image'].shape[0] == 5
    torch.testing.assert_close(details['pool_query'].sum(-1), torch.ones(2))


def test_attention_pool_ignores_masked_nodes():
    nodes = unit(1, 3, 512)
    pooled, alpha = attention_pool(nodes, torch.tensor([[True, True, False]]), torch.zeros(512))
    torch.testing.assert_close(alpha, torch.tensor([[.5, .5, 0.]]))
    torch.testing.assert_close(pooled[0], F.normalize(nodes[0, :2].mean(0), dim=-1))


def test_explanations_follow_the_new_scoring():
    store = fake_store()
    factorized = GraphReranker(triple_match='factorized', dropout=0.).eval()
    result = score(factorized, store, 'q', 1)
    row = result['triples'][0]
    slots = row['slot_similarity']
    assert set(slots) == {'subject', 'predicate', 'object'}
    assert abs(slots['subject'] * slots['predicate'] * slots['object'] - row['sim_ij']) < 1e-5
    assert abs(row['sim_ij'] - 1) < 1e-5  # the query graph is the image graph
    graph = GraphReranker(graph_channel=True, dropout=0.).eval()
    result = score(graph, store, 'q', 1)
    assert result['triples'] == []
    level = result['graph_level']
    assert abs(level['cosine'] - 1) < 1e-5
    assert [n['alpha'] for n in level['query_nodes']] == pytest.approx([.5, .5])
    assert abs(slot_similarity(torch.tensor(1.)) - 1) < 1e-7
