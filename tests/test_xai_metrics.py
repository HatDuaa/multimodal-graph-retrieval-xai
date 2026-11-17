"""Level-3 explanation metrics: part-level graph edits and rescoring among the candidates."""
import numpy as np
import pytest

from src.data.graph_store import EDGE_LABELLED, GraphIndex
from src.explain.xai_metrics import (QueryContext, edit, image_index, keep_only, labelled_count, message_drop,
                                     real_nodes)
from tests.test_demo_countonly import count_model_and_batch


def wordnet_ready():
    try:
        from nltk.corpus import wordnet as wn
        wn.synset("dog.n.01")
        return True
    except (ImportError, LookupError):
        return False


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


def test_dropping_messages_keeps_the_triple_while_dropping_the_edge_removes_it():
    store, _, ctx = context()
    column = ctx.ids.index(1)                              # image 1 has one labelled triple, edge 0
    index = image_index(store, store.graphs[1])
    every = np.ones(int(index.edge_counts[0]), dtype=bool)
    scores, _ = ctx.rescore(column, [edit(index), edit(index)], message_masks=[None, every])
    assert scores[0] == pytest.approx(scores[1], abs=1e-6)  # an all-True mask changes nothing
    _, _, _, kept = ctx.rescore(column, [edit(index)], channels=True, message_masks=[message_drop(index, [0])])
    _, _, _, dropped = ctx.rescore(column, [edit(index, drop_edges=[0])], channels=True)
    assert np.isfinite(kept[0]) and np.isnan(dropped[0])
    silent = message_drop(index, range(int(index.edge_counts[0])))
    _, _, _, triples = ctx.rescore(column, [edit(index)], channels=True, message_masks=[silent])
    assert np.isfinite(triples[0])                         # no message at all, the triple still scores


def test_fusion_scales_are_one_under_the_raw_fusion():
    _, _, ctx = context()
    assert ctx.fusion_scales(0) == {"objects": 1., "triples": 1.}


def test_fusion_scales_match_a_finite_difference_under_the_z_score_fusion():
    from src.explain.xai_metrics import fuse
    store, model, _ = count_model_and_batch()
    model.options["fusion"] = "zscore"                     # same weights, level-1 fusion
    ctx = QueryContext(model, store, "q", "cpu")
    column = ctx.ids.index(1)                              # the only candidate with both channels
    scales = ctx.fusion_scales(column)
    weights = ctx.detail["weights"]
    for key, channel in (("objects", 1), ("triples", 2)):
        values = {k: ctx.detail[k].clone().double() for k in ("objects", "triples")}
        step = 1e-6
        base = fuse(model, ctx.batch["clip"].double(), values["objects"], values["triples"], weights.double())[0, column]
        values[key][0, column] += step
        moved = fuse(model, ctx.batch["clip"].double(), values["objects"], values["triples"], weights.double())[0, column]
        slope = float((moved - base) / step) / float(weights[0, channel])
        assert scales[key] == pytest.approx(slope, rel=1e-3, abs=1e-6)


def test_gradient_terms_give_one_value_per_object_node_and_labelled_triple():
    store, _, ctx = context()
    column = ctx.ids.index(1)
    index = image_index(store, store.graphs[1])
    nodes, triples = ctx.gradient_terms(column, index)
    assert nodes.shape == (real_nodes(index),) and triples.shape == (labelled_count(index),)
    assert np.all(np.isfinite(nodes)) and np.all(np.isfinite(triples))


def test_gradient_terms_on_an_image_without_triples():
    store, _, ctx = context()
    column = ctx.ids.index(2)                              # image 2 has object nodes and no labelled triple
    index = image_index(store, store.graphs[2])
    assert labelled_count(index) == 0 and real_nodes(index) > 0
    nodes, triples = ctx.gradient_terms(column, index)
    assert nodes.shape == (real_nodes(index),) and triples.shape == (0,) and np.all(np.isfinite(nodes))


def test_full_rescore_matches_the_forward_pass_for_an_unedited_graph():
    store, _, ctx = context()
    for column, image_id in enumerate(ctx.ids):
        *_, fused = ctx.rescore(column, [edit(image_index(store, store.graphs[image_id]))], full=True)
        assert fused[0] == pytest.approx(ctx.fused.cpu().numpy(), abs=1e-6)


@pytest.mark.skipif(not wordnet_ready(), reason="WordNet data not installed")
def test_strict_wordnet_accepts_hyponyms_only_and_people_for_generic_words():
    from scripts.xai_metrics import strict_wordnet
    assert strict_wordnet("animal", "animal.n.01", "dog.n.01")
    assert not strict_wordnet("dog", "dog.n.01", "animal.n.01")
    assert not strict_wordnet("woman", "woman.n.01", "man.n.01")
    assert strict_wordnet("person", "person.n.01", "man.n.01")
