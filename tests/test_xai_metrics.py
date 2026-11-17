"""Level-3 explanation metrics: part-level graph edits and rescoring among the candidates."""
import numpy as np
import pytest
import torch

from src.data.graph_store import EDGE_LABELLED, GraphIndex
from src.explain.xai_metrics import QueryContext, edit, image_index, keep_only, labelled_count, real_nodes
from tests.test_demo_countonly import count_model_and_batch


def context():
    store, model, _ = count_model_and_batch()
    return store, model, QueryContext(model, store, "q", "cpu")


def test_empty_edit_and_keeping_every_part_leave_the_score_unchanged():
    store, _, ctx = context()
    for column, image_id in enumerate(ctx.ids):
        index = image_index(store, store.graphs[image_id])
        scores, ranks = ctx.rescore(column, [edit(index), keep_only(index, range(real_nodes(index)),
                                                                     range(labelled_count(index)))])
        assert scores == pytest.approx([float(ctx.fused[column])] * 2, abs=1e-6)
        baseline = 1 + int((ctx.fused > ctx.fused[column]).sum())
        assert list(ranks) == [baseline] * 2


def test_dropping_every_part_of_a_channel_gives_the_filled_value():
    store, model, ctx = context()
    column = ctx.ids.index(1)                              # image 1: person, ball, ball and one labelled triple
    other = 1 - column
    index = image_index(store, store.graphs[1])
    a, b, c = ctx.detail["weights"][0].tolist()
    clip = ctx.batch["clip"][0]
    # every triple dropped: the triple channel takes the known mean of the others (image 2 has none: 0)
    score, _, objects, triples = ctx.rescore(column, [edit(index, drop_edges=range(labelled_count(index)))], channels=True)
    assert np.isnan(triples[0]) and np.isnan(float(ctx.detail["triples"][0, other]))
    assert score[0] == pytest.approx(float(a * clip[column] + b * objects[0]), abs=1e-6)
    # every node dropped: the object channel takes image 2's object score, the only known one
    score, _, objects, _ = ctx.rescore(column, [edit(index, drop_nodes=range(real_nodes(index)))], channels=True)
    assert np.isnan(objects[0])
    assert score[0] == pytest.approx(float(a * clip[column] + b * ctx.detail["objects"][0, other]), abs=1e-6)


def test_dropping_a_node_drops_its_edges_without_rebuilding_completion_edges():
    store, _, _ = context()
    index = image_index(store, store.graphs[1])
    src, dst = index.fields["edge_src"], index.fields["edge_dst"]
    touching = int(((src == 0) | (dst == 0)).sum())
    edited = edit(index, drop_nodes=[0])
    assert int(edited.node_counts[0]) == int(index.node_counts[0]) - 1
    assert int(edited.edge_counts[0]) == int(index.edge_counts[0]) - touching
    assert labelled_count(edited) == 0                     # the only triple started at node 0
    assert edited.fields["edge_src"].max() < edited.node_counts[0]


def test_a_kept_triple_keeps_its_end_nodes_as_unscored_carriers():
    store, _, _ = context()
    index = image_index(store, store.graphs[1])
    kept = keep_only(index, nodes=[], triples=[0])
    real = real_nodes(kept)
    assert real == 2 and not kept.fields["part_mask"][:real].any()   # person and its ball, neither scored
    assert labelled_count(kept) == 1 and (kept.fields["edge_kind"] == EDGE_LABELLED).sum() == 1


def test_attention_has_one_alpha_per_edge():
    store, _, ctx = context()
    index = image_index(store, store.graphs[1])
    alpha = ctx.attention(index)
    assert alpha.shape == (int(index.edge_counts[0]),) and np.all(alpha >= 0) and np.all(alpha <= 1)


def test_attention_skips_a_relation_from_a_node_to_itself():
    store, _, ctx = context()
    index = image_index(store, store.graphs[1])
    fields = dict(index.fields)
    fields["edge_dst"] = fields["edge_dst"].copy()
    fields["edge_dst"][0] = fields["edge_src"][0]          # the labelled triple now loops on its subject
    looped = GraphIndex(index.node_counts, index.edge_counts, has_concepts=index.has_concepts, **fields)
    alpha = ctx.attention(looped)
    assert np.isnan(alpha[0]) and np.all((alpha[1:] >= 0) & (alpha[1:] <= 1))


def test_row_terms_add_up_to_the_channel_parts():
    _, _, ctx = context()
    rows = ctx.row_terms()
    weights = ctx.detail["weights"][0]
    for channel, w in (("objects", weights[1]), ("triples", weights[2])):
        expected = (w * ctx.detail[channel][0]).double().numpy()
        total = np.where(np.isnan(rows[channel]).all(1), np.nan, np.nansum(rows[channel], 1))
        np.testing.assert_allclose(total, expected, atol=1e-6)
