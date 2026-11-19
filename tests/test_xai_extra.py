"""Counterfactuals, GraphMask and GStarX of the level-3 explainer comparison."""
import math
from itertools import combinations, permutations

import numpy as np
import pytest
import torch

from src.explain import xai_baselines as baselines
from src.explain.xai_counterfactual import cf_gnnexplainer, exact_counterfactual, exact_counterfactuals, flipped, removal
from src.explain.xai_metrics import QueryContext, edit, image_index, labelled_count, real_nodes
from tests.test_demo_countonly import count_model_and_batch


def context():
    """The toy count model with a non-zero u (so GAT messages reach the score) and CLIP scores close enough that
    removing node 0 of the rank-1 image costs it rank 1."""
    store, model, _ = count_model_and_batch()
    torch.manual_seed(0)
    with torch.no_grad():
        model.u.weight.normal_(0, .05)
    store.arrays["val"]["clip_scores"] = np.array([[.38, .4]], dtype="float32")
    ctx = QueryContext(model, store, "q", "cpu")
    index = image_index(store, store.graphs[ctx.ids[ctx.top]])
    every = [("node", j) for j in range(real_nodes(index))] + [("triple", j) for j in range(labelled_count(index))]
    return ctx, ctx.top, index, every


def exact_shapley(players, v):
    n = len(players)
    phi = {p: 0. for p in players}
    for order in permutations(players):
        before = frozenset()
        for p in order:
            phi[p] += v(before | {p}) - v(before)
            before = before | {p}
    return {p: x / math.factorial(n) for p, x in phi.items()}


def test_hamiache_navarro_is_the_shapley_value_on_a_complete_graph():
    rng = np.random.default_rng(0)
    players = [0, 1, 2, 3]
    table = {frozenset(c): float(rng.normal()) for r in range(1, 5) for c in combinations(players, r)}
    v = lambda c: table.get(frozenset(c), 0.)  # noqa: E731
    adjacency = [set(players) - {p} for p in players]
    hn = baselines.hamiache_navarro(players, adjacency, table)
    shapley = exact_shapley(players, v)
    for p in players:
        assert hn[p] == pytest.approx(shapley[p], abs=1e-8)


def test_hamiache_navarro_is_efficient_on_a_path():
    rng = np.random.default_rng(1)
    players = [0, 1, 2, 3]
    adjacency = [{1}, {0, 2}, {1, 3}, {2}]
    table = {frozenset(c): float(rng.normal()) for r in range(1, 5) for c in combinations(players, r)
             if baselines._connected(c, adjacency)}
    hn = baselines.hamiache_navarro(players, adjacency, table)
    assert sum(hn.values()) == pytest.approx(table[frozenset(players)], abs=1e-8)
    # a path is not complete, so the value differs from the Shapley value of the Myerson graph game
    assert not all(np.isclose(hn[p], exact_shapley(players, lambda c: table.get(frozenset(c), 0.))[p]) for p in players)


def test_part_graph_links_triples_to_their_end_nodes():
    _, _, index, every = context()
    adjacency = baselines.part_graph(index, every)
    for i, neighbours in enumerate(adjacency):
        assert i not in neighbours and all(i in adjacency[j] for j in neighbours)
    t = every.index(("triple", 0))
    src, dst = index.fields["edge_src"][0], index.fields["edge_dst"][0]
    assert adjacency[t] == {every.index(("node", int(src))), every.index(("node", int(dst)))}


def test_gstarx_scores_every_part():
    ctx, column, index, every = context()
    values, evaluations = baselines.gstarx(ctx, column, index, every, np.random.default_rng(0), size=3)
    assert [p[:2] for p in values] == every and all(np.isfinite(v) for *_, v in values)
    assert evaluations > len(every)


def open_or_closed(bias):
    explainer = baselines.GraphMask()
    with torch.no_grad():
        for gate in explainer.gates:
            gate[-1].weight.zero_()
            gate[-1].bias.fill_(bias)
    return explainer


def test_open_graphmask_gates_leave_the_score_unchanged():
    ctx, column, index, _ = context()
    images = index.to_batch(ctx.store.part_table("cpu"), "cpu")
    plain, _ = ctx.score_images(column, images, 1)
    with baselines.gated_messages(ctx.model, open_or_closed(20.)):
        gated, _ = ctx.score_images(column, index.to_batch(ctx.store.part_table("cpu"), "cpu"), 1)
    assert torch.equal(gated, plain)


def test_closed_graphmask_gates_send_the_baseline_and_learn_it():
    ctx, column, index, _ = context()
    plain, _ = ctx.score_images(column, index.to_batch(ctx.store.part_table("cpu"), "cpu"), 1)
    explainer = open_or_closed(-20.)
    with torch.no_grad():
        explainer.baselines[0].normal_()
    record = []
    with torch.enable_grad(), baselines.gated_messages(ctx.model, explainer, record=record):
        gated, _ = ctx.score_images(column, index.to_batch(ctx.store.part_table("cpu"), "cpu"), 1)
        grad, = torch.autograd.grad(gated[0, column], [explainer.baselines[0]])
    assert float(gated[0, column].detach()) != pytest.approx(float(plain[0, column]), abs=1e-6)
    assert grad.abs().sum() > 0
    # one record per layer, over the input edges that are not self-loops
    src, dst = index.fields["edge_src"], index.fields["edge_dst"]
    assert [layer for layer, _, _ in record] == [0, 1]
    assert record[0][1].tolist() == np.flatnonzero(src != dst).tolist()
    parts, edges = baselines.graphmask(ctx, column, index, explainer)
    assert len(edges) == len(src) and np.all(np.isnan(edges) == (src == dst))
    assert all(0 <= v <= 1 for *_, v in parts)


def test_exact_counterfactual_is_the_smallest_removal_that_costs_rank_one():
    ctx, column, index, every = context()
    out = exact_counterfactual(ctx, column, index, every, max_size=len(every))
    assert out["size"] is not None
    *_, full = ctx.rescore(column, [removal(index, out["parts"])], full=True)
    assert flipped(full, column)[0]
    smaller = [list(c) for r in range(1, out["size"]) for c in combinations(every, r)]
    if smaller:
        *_, full = ctx.rescore(column, [removal(index, c) for c in smaller], full=True)
        assert not flipped(full, column).any()


def test_cf_gnnexplainer_returns_a_valid_counterfactual():
    ctx, column, index, every = context()
    base = float(ctx.fused[column])
    empty = float(ctx.rescore(column, [edit(index, drop_nodes=range(real_nodes(index)))])[0][0])
    parts, out = cf_gnnexplainer(ctx, column, index, base, empty, steps=100)
    assert [p[:2] for p in parts] == every
    if out["parts"] is not None:
        *_, full = ctx.rescore(column, [removal(index, out["parts"])], full=True)
        assert flipped(full, column)[0] and out["size"] == len(out["parts"])


def smallest_by_brute_force(ctx, column, index, every, max_size, target):
    for size in range(1, max_size + 1):
        sets = [list(c) for c in combinations(every, size)]
        *_, full = ctx.rescore(column, [removal(index, c) for c in sets], full=True)
        if flipped(full, column, target).any():
            return size, int(flipped(full, column, target).sum())
    return None, 0


def test_one_rescoring_serves_the_rank_one_and_the_target_counterfactuals():
    ctx, column, index, every = context()
    other = 1 - column
    both = exact_counterfactuals(ctx, column, index, every, max_size=3, targets=(None, other))
    for out, target in zip(both, (None, other)):
        size, sets = smallest_by_brute_force(ctx, column, index, every, 3, target)
        assert (out["size"], out["sets"]) == (size, sets)
        assert out == exact_counterfactual(ctx, column, index, every, max_size=3, target=target)
    # with two candidates, the other one is rank 1 exactly when the explained image loses it
    assert both[0]["size"] == both[1]["size"]


@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(), reason="no accelerator")
def test_trained_explainers_follow_the_model_device():
    device = "cuda" if torch.cuda.is_available() else "mps"
    store, model, _ = count_model_and_batch()
    model = model.to(device)
    ctx = QueryContext(model, store, "q", device)
    index = image_index(store, store.graphs[ctx.ids[ctx.top]])
    (base, empty), _ = ctx.rescore(ctx.top, [edit(index), edit(index, drop_nodes=range(real_nodes(index)))])
    items = [lambda: (ctx, ctx.top, index, float(base), float(empty))]
    gm, _ = baselines.train_graphmask(items, seed=0, epochs=1, device=device)
    pg, _ = baselines.train_pgexplainer(items, seed=0, epochs=1, device=device)
    assert all(p.device.type == device for p in [*gm.parameters(), *pg.parameters()])
