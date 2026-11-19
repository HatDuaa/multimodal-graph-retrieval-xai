"""Four more explainers for the level-3 comparison, adapted to the reranker's image parts (object nodes and labelled
triples of the rank-1 image): PGExplainer, GNN-LRP, GraphSVX and SubgraphX.

Each returns the image's parts with an importance, so scripts/xai_metrics.py measures them like the other explainers
(deletion / insertion curves, Fid_alpha, agreement with Shapley). The model is a reranker, not a node or graph
classifier: the explained output is the rank-1 image's final score (or its rank-1 margin, for the Shapley-type
methods), with the caption and the other 49 candidates fixed (QueryContext.score_images / rescore).

- PGExplainer (Luo et al., NeurIPS 2020): one MLP for every caption maps each part's encoded vector, and its
  product with the caption's weighted part summary, to a mask logit; it is trained once per checkpoint on val captions
  outside the comparison sample, with binary concrete masks and the loss of the GNNExplainer adaptation (keep the
  graph effect, size and entropy penalties, masked parts move to the image's neutral vector). At explanation time one
  forward pass gives the mask.
- GNN-LRP (Schnake et al., TPAMI 2022): relevant walks through the two GATv2 layers, by LRP-0. The rest of the
  model is made linear the usual LRP way (Ali et al., ICML 2022): the vector norms of F.normalize, the softmax
  attention of the part matching and the z-score mean and spread over the 50 candidates are held constant
  (lrp_mode), so gradient x input there is LRP-0 and conserves relevance up to biases. The relevance at the second GAT
  layer's output is split exactly over walks (source input node J -> layer-1 node K -> layer-2 node L) with the GAT
  attention coefficients held constant and each ELU passing relevance in proportion. A part's importance is the
  relevance of every walk it lies on (a node: as J, K or L; a triple: its labelled edge carrying the layer-1 or
  layer-2 message), plus the relevance of its own inputs that skip the GAT (a node's residual vector; a triple's
  phrase vector and predicate vector into the triple MLP). Biases get no relevance (LRP-0).
- GraphSVX (Duval and Malliaros, ECML-PKDD 2021): a weighted linear surrogate fitted on sampled coalitions of parts,
  with Kernel SHAP weights and the efficiency constraint, so its coefficients estimate Shapley values. Value
  function and evaluation budget are those of the permutation Shapley reference: the rank-1 margin when only the
  coalition is kept, 16 x (parts + 1) evaluations. Coalition sizes are enumerated from both ends while the budget
  allows, then sampled with complements.
- SubgraphX (Yuan et al., ICML 2021): Monte Carlo tree search over connected node subgraphs, each child pruning one
  node (the highest-degree ones first) and keeping the largest connected component; a subgraph's reward is its
  Shapley value against coalitions of its 2-hop neighbours (the model has 2 GAT layers), by Monte Carlo sampling;
  connectivity uses the labelled and completion edges, not the global node. The best subgraph within the size limit
  is then pruned greedily to one node. Parts are ordered by that pruning path, last node standing first, and a
  triple comes right after the later of its two end nodes.
"""
import contextlib
from itertools import combinations
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.graph_store import EDGE_GLOBAL, EDGE_LABELLED
from src.explain.xai_metrics import keep_only, labelled_count, margin_values, real_nodes, split
import src.models.graph_reranker as reranker
from src.models.graph_reranker import masked_softmax, part_summary


def neutral_parts(images, n, m):
    """The image's input vectors and their neutral values: the image's mean node vector for its object nodes, the
    mean phrase and predicate vectors for its labelled triples (other nodes, e.g. the global node, keep theirs)."""
    x0, t0, e0 = images.x.detach(), images.triple_f.detach(), images.edge_attr.detach()
    x_mean = x0[:n].mean(0, keepdim=True) if n else x0[:0]
    t_mean, e_mean = (t0[:m].mean(0, keepdim=True), e0[:m].mean(0, keepdim=True)) if m else (t0, e0)
    return x0, t0, e0, x_mean, t_mean, e_mean


def blend(images, mask, n, m, vectors):
    """Set the image's inputs to mask * input + (1 - mask) * neutral, per part (mask: n node values, m triple values)."""
    x0, t0, e0, x_mean, t_mean, e_mean = vectors
    device = x0.device
    node_mask = torch.cat((mask[:n], torch.ones(len(x0) - n, device=device)))[:, None]
    neutral = torch.cat((x_mean.expand(n, -1), x0[n:]))
    images.x = node_mask * x0 + (1 - node_mask) * neutral
    if m:
        images.triple_f = mask[n:, None] * t0[:m] + (1 - mask[n:, None]) * t_mean
        images.edge_attr = mask[n:, None] * e0[:m] + (1 - mask[n:, None]) * e_mean
    return images


def mask_loss(ctx, column, images, mask, empty, effect, size, entropy):
    """GNNExplainer adaptation's loss: -(final - empty) / effect + size * mean(mask) + entropy * mean entropy(mask)."""
    fused, _ = ctx.score_images(column, images, 1)
    bits = mask.clamp(1e-6, 1 - 1e-6)
    return (-(fused[0, column] - empty) / effect + size * mask.mean()
            - entropy * (bits * bits.log() + (1 - bits) * (1 - bits).log()).mean())


class PGExplainer(nn.Module):
    """Amortized part mask: an MLP over each part's encoded vector and its product with the caption summary."""

    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * 512 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    @torch.no_grad()
    def features(ctx, index):
        """(n + m, 1026): [z, z * caption summary, kind one-hot] per node, then per labelled triple, where z is the
        part's vector after the model's encoder and the summary is the caption's parts weighted by w of that channel."""
        model = ctx.model
        images = index.to_batch(ctx.store.part_table(ctx.device), ctx.device)
        concepts = model.concept_vectors() if model.options['concept_gat'] else None
        nodes, _, triples, _ = model.encode(images, concepts=concepts)
        q, qm, qt, qtm = ctx.detail['encoded_query']
        n, m = real_nodes(index), labelled_count(index)
        rows = []
        for vectors, count, query, query_mask, u, kind in ((nodes, n, q, qm, model.u_objects, 0),
                                                           (triples, m, qt, qtm, model.u_triples, 1)):
            if not count:
                continue
            z = vectors[0, :count]
            z = z.mean(-2) if z.dim() == 3 else z
            summary = part_summary(query)
            w = masked_softmax(summary @ u, query_mask, -1)[0]
            weighted = torch.nan_to_num((w[:, None] * summary[0]).sum(0)) if query_mask.any() else torch.zeros_like(z[0])
            onehot = F.one_hot(torch.full((count,), kind, device=z.device), 2).to(z.dtype)
            rows.append(torch.cat((z, z * weighted, onehot), -1))
        return torch.cat(rows)

    def logits(self, features):
        return self.mlp(features).squeeze(-1)

    @torch.no_grad()
    def explain(self, ctx, column, index):
        mask = torch.sigmoid(self.logits(self.features(ctx, index))).cpu().numpy()
        n = real_nodes(index)
        return [("node", j, float(mask[j])) for j in range(n)] + \
               [("triple", j, float(mask[n + j])) for j in range(labelled_count(index))]


def train_pgexplainer(contexts, seed, epochs=10, lr=3e-3, size=2., entropy=0.1, t0=5., t1=1.):
    """Train PGExplainer on (QueryContext factory, ...) items: contexts is a list of callables returning
    (ctx, column, index, base, empty) for one training caption, effect = base - empty > 1e-4.

    Binary concrete masks with temperature t0 -> t1 over the epochs (Luo et al.); the loss is mask_loss on each
    caption in turn. Returns the explainer and the mean loss of the last epoch.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        explainer = PGExplainer()
        optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(seed)
        order_rng = np.random.default_rng(seed)
        last = []
        for epoch in range(epochs):
            temperature = t0 * (t1 / t0) ** (epoch / max(epochs - 1, 1))
            last = []
            for i in order_rng.permutation(len(contexts)):
                ctx, column, index, base, empty = contexts[i]()
                features = explainer.features(ctx, index)
                n, m = real_nodes(index), labelled_count(index)
                images = index.to_batch(ctx.store.part_table(ctx.device), ctx.device)
                vectors = neutral_parts(images, n, m)
                with torch.enable_grad():
                    logits = explainer.logits(features)
                    u = torch.rand(len(logits), generator=generator).clamp(1e-6, 1 - 1e-6).to(logits.device)
                    mask = torch.sigmoid((logits + u.log() - (1 - u).log()) / temperature)
                    loss = mask_loss(ctx, column, blend(images, mask, n, m, vectors), mask, empty, base - empty,
                                     size, entropy)
                    optimizer.zero_grad()
                    grads = torch.autograd.grad(loss, list(explainer.parameters()))
                for p, g in zip(explainer.parameters(), grads):
                    p.grad = g
                optimizer.step()
                last.append(float(loss.detach()))
    explainer.eval()
    return explainer, float(np.mean(last)) if last else None


def _zscore_constant(values, eps=1e-6, floor=0.):
    """zscore_tensor with its mean and spread held constant (no gradient through them)."""
    known = torch.isfinite(values)
    count = known.sum(-1, keepdim=True)
    clean = torch.where(known, values, 0.)
    mean = clean.sum(-1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(known, clean - mean, 0.)
    var = centered.square().sum(-1, keepdim=True) / count.clamp_min(1)
    active = (count >= 2) & (var >= eps ** 2)
    std = var.clamp_min(eps ** 2).sqrt()
    if floor:
        std = std.clamp_min(floor)
    z = torch.where(active, (clean - mean.detach()) / (std + eps).detach(), 0.)
    return z, active.squeeze(-1)


@contextlib.contextmanager
def lrp_mode():
    """Hold F.normalize's norms, masked_softmax's weights and the z-score statistics constant (LRP-0 linearization)."""
    saved = F.normalize, reranker.masked_softmax, reranker.zscore_tensor

    def normalize(v, p=2.0, dim=1, eps=1e-12, out=None):
        return v / v.norm(p, dim, keepdim=True).clamp_min(eps).detach()

    def softmax_constant(*args, **kwargs):
        return saved[1](*args, **kwargs).detach()
    F.normalize, reranker.masked_softmax, reranker.zscore_tensor = normalize, softmax_constant, _zscore_constant
    try:
        yield
    finally:
        F.normalize, reranker.masked_softmax, reranker.zscore_tensor = saved


EPSILON = 1e-6


def gnn_lrp(ctx, column, index):
    """Part relevances from GNN-LRP's relevant walks (module docstring); also returns a check dict.

    Inside the GAT, relevance goes back layer by layer with the epsilon rule on the input contributions only (biases
    take none), so the walks' relevances add up to the layer-2 output's relevance. check: the largest gap between
    each GAT layer's recorded pre-activation and its rebuild from the contributions and biases (float error when the
    split is exact), the output relevance and the walks' total (equal up to epsilon), and the inputs' own relevance.
    """
    model, device = ctx.model, ctx.device
    if not (model.options['use_gat'] and model.options['use_edges']) or model.options['concept_gat']:
        raise ValueError('GNN-LRP here needs the GAT over edges and no concept input')
    table = ctx.store.part_table(device)
    images = index.to_batch(table, device)
    n, m = real_nodes(index), labelled_count(index)
    captured = {}

    def keep(name):
        def hook(module, args, output):
            captured[name] = output[0] if isinstance(output, tuple) else output
        return hook
    handles = [model.gat1.register_forward_hook(keep('z1')), model.gat2.register_forward_hook(keep('z2'))]
    model.attention, model.record_attention = [], True
    try:
        with torch.enable_grad(), lrp_mode():
            fused, _ = ctx.score_images(column, images, 1)
            gz2, = torch.autograd.grad(fused[0, column], [captured['z2']])
    finally:
        model.record_attention = False
        for h in handles:
            h.remove()
    (idx1, a1), (idx2, a2) = model.attention
    model.attention = []
    assert torch.equal(idx1, idx2)
    z1, z2 = captured['z1'].detach(), captured['z2'].detach()
    total, heads = len(images.x), a1.shape[1]
    width = z1.shape[1] // heads
    # Relevance of the layer-2 output h2 = ELU(z2): (dF/dh2) . h2, with dF/dh2 = dF/dz2 / ELU'(z2); ELU passes it to
    # z2 unchanged per unit, and likewise at layer 1.
    slope2 = torch.where(z2 > 0, torch.ones_like(z2), torch.exp(z2))
    relevance2 = gz2.detach() / slope2 * F.elu(z2)

    def stable(z):
        return z + EPSILON * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

    # c0[i, j]: the part of p_node's output at node i that comes from object node j's input vector (no bias).
    x = images.x.detach()
    wx = x[:n] @ model.p_node.weight.T
    c0 = torch.zeros(total, n, wx.shape[1], device=device)
    c0[torch.arange(n), torch.arange(n)] = wx
    global_mask = getattr(images, 'global_mask', None)
    if global_mask is not None and bool(global_mask.any()):
        scorable = images.part_mask[:n].to(x.dtype)
        norm = (x[:n] * scorable[:, None]).sum(0).norm().clamp_min(1e-12)
        for g in torch.nonzero(global_mask).flatten().tolist():
            c0[g] = wx * scorable[:, None] / norm

    def layer(conv, c, edges, alpha):
        """Per-source contributions to a GATv2 layer's output (before its activation) and the bias part."""
        wc = c @ conv.lin_l.weight.T
        src, dst = edges
        per_edge = (alpha.repeat_interleave(width, 1)[:, None, :] * wc[src])
        out = torch.zeros(total, c.shape[1], wc.shape[2], device=device).index_add(0, dst, per_edge)
        b = conv.lin_l.bias if conv.lin_l.bias is not None else torch.zeros(wc.shape[2], device=device)
        bias = torch.zeros(total, wc.shape[2], device=device).index_add(
            0, dst, alpha.repeat_interleave(width, 1) * b) + (conv.bias if conv.bias is not None else 0.)
        return out, bias, wc

    with torch.no_grad():
        contrib1, bias1, wc0 = layer(model.gat1, c0, idx1, a1)
        # check: layer 1 rebuilt from the sources, the bias of p_node flowing like one more source, and the biases
        bias_path1, _, _ = layer(model.gat1, torch.zeros(total, 1, wx.shape[1], device=device) + model.p_node.bias,
                                 idx1, a1)
        gap1 = float((contrib1.sum(1) + bias_path1[:, 0] + bias1 - z1).abs().max()) if n else 0.
        h1 = F.elu(z1)

        # Layer 2, per edge e (K -> L): its input contribution z_e = alpha_e * W h1[K] (no bias); each output unit's
        # relevance is shared among the incoming contributions in proportion (epsilon rule, biases get none).
        src2, dst2 = idx2
        w2 = model.gat2.lin_l.weight
        alpha2 = a2.repeat_interleave(width, 1)
        z_edges = alpha2 * (h1[src2] @ w2.T)
        z_sum = torch.zeros_like(z2).index_add(0, dst2, z_edges)
        gap2 = float((z_sum + layer(model.gat2, h1[:, None, :] * 0, idx2, a2)[1] - z2).abs().max()) if n else 0.
        coefficient = relevance2[dst2] / stable(z_sum[dst2]) * alpha2
        message = h1[src2] * (coefficient @ w2)  # (edges, 256): relevance reaching h1[K] through edge e
        relevance1 = torch.zeros_like(z1).index_add(0, src2, message)

        # Layer 1 (with p_node): h1[K]'s relevance is shared among source nodes J by their contributions (no bias).
        z1_sources = contrib1.sum(1)
        share = contrib1 / stable(z1_sources)[:, None, :]
        walks = torch.einsum('ed,ejd->ej', message, share[src2])  # (edges2, n): walk J -> K -> L through e

        by_source = walks.sum(0)
        node_score = torch.zeros(n, device=device)
        sources = torch.arange(n, device=device)
        for e in range(walks.shape[0]):
            k, target = int(src2[e]), int(dst2[e])
            r = walks[e]
            node_score += r  # the source J of every walk
            if k < n:
                node_score[k] += (r * (sources != k)).sum()
            if target < n and target != k:
                node_score[target] += (r * (sources != target)).sum()

        triple_walks = torch.zeros(m, device=device)
        if m:
            kind = index.fields['edge_kind']
            esrc, edst = index.fields['edge_src'], index.fields['edge_dst']
            proper = esrc != edst
            position = np.cumsum(proper) - 1
            labelled = np.flatnonzero(kind == EDGE_LABELLED)
            src1, dst1 = idx1
            for t, e in enumerate(labelled[:m]):
                if not proper[e]:
                    continue
                p = int(position[e])
                assert int(src1[p]) == int(esrc[e]) and int(dst1[p]) == int(edst[e])
                s, d = int(esrc[e]), int(edst[e])
                r1 = 0.
                if s < n:
                    via = a1[p].repeat_interleave(width) * wc0[s, s]
                    r1 = float((relevance1[d] * via / stable(z1_sources[d])).sum())
                triple_walks[t] = r1 + float(walks[p].sum())

    # inputs that skip the GAT: residual node vectors, triple phrase vectors, predicate vectors into the triple MLP
    own = {}

    def cut_input(module, args):
        return (args[0].detach(),) + tuple(args[1:])

    def cut_edges(module, args):
        return (args[0], args[1], args[2].detach()) + tuple(args[3:])
    handles = [model.p_node.register_forward_pre_hook(cut_input), model.gat1.register_forward_pre_hook(cut_edges),
               model.gat2.register_forward_pre_hook(cut_edges)]
    try:
        again = index.to_batch(table, device)
        inputs = [again.x.detach().clone().requires_grad_(True), again.triple_f.detach().clone().requires_grad_(True),
                  again.edge_attr.detach().clone().requires_grad_(True)]
        again.x, again.triple_f, again.edge_attr = inputs
        with torch.enable_grad(), lrp_mode():
            fused, _ = ctx.score_images(column, again, 1)
            grads = torch.autograd.grad(fused[0, column], inputs, allow_unused=True)
        x_r, t_r, e_r = [((g if g is not None else torch.zeros_like(v)) * v).sum(-1).detach()
                         for g, v in zip(grads, inputs)]
        own['nodes'] = x_r[:n]
        own['triples'] = (t_r + e_r)[:m]
    finally:
        for h in handles:
            h.remove()
    nodes = (node_score + own['nodes']).cpu().numpy()
    triples = (triple_walks + own['triples']).cpu().numpy()
    parts = [("node", j, float(v)) for j, v in enumerate(nodes)] + [("triple", j, float(v)) for j, v in enumerate(triples)]
    check = {"layer1_gap": gap1, "layer2_gap": gap2, "scale": float(z2.abs().max()),
             "output_relevance": float(relevance2.sum()), "walks": float(by_source.sum()),
             "own_nodes": float(own['nodes'].sum()), "own_triples": float(own['triples'].sum())}
    return parts, check


def kernel_weight(size, total):
    return (total - 1) / (math.comb(total, size) * size * (total - size))


def graphsvx(ctx, column, index, every, rng, budget):
    """Shapley estimates of the parts from a Kernel SHAP weighted linear surrogate (GraphSVX), within budget
    evaluations of the rank-1 margin; returns the parts with their coefficient and an efficiency check."""
    total = len(every)
    full, none = margin_values(ctx, column, [keep_only(index, *split(every)), keep_only(index)])
    delta = full - none
    if total == 1:
        return [(every[0][0], every[0][1], float(delta))], {"sum": float(delta), "difference": float(delta)}
    coalitions, weights = [], []
    left = max(budget - 2, 2)
    sizes = []
    for s in range(1, total // 2 + 1):
        sizes += [s] if s == total - s else [s, total - s]
    exhausted = 0
    for i in range(0, len(sizes), 2):
        pair = sizes[i:i + 2]
        count = sum(math.comb(total, s) for s in pair)
        if count > left:
            break
        for s in pair:
            for subset in _combinations(total, s):
                mask = np.zeros(total, dtype=bool)
                mask[list(subset)] = True
                coalitions.append(mask)
                weights.append(kernel_weight(s, total))
        left -= count
        exhausted = i + len(pair)
    rest = sizes[exhausted:]
    if rest and left >= 2:
        mass = np.array([kernel_weight(s, total) * math.comb(total, s) for s in rest])
        probability = mass / mass.sum()
        per = mass.sum() / (left // 2 * 2)
        for _ in range(left // 2):
            s = int(rng.choice(rest, p=probability))
            mask = np.zeros(total, dtype=bool)
            mask[rng.choice(total, size=s, replace=False)] = True
            for c in (mask, ~mask):
                coalitions.append(c)
                weights.append(per)
    X = np.array(coalitions, dtype=float)
    edits = [keep_only(index, *split([every[t] for t in np.flatnonzero(c)])) for c in coalitions]
    y = margin_values(ctx, column, edits) - none
    w = np.array(weights)
    # efficiency: phi_last = delta - sum(phi_others), substituted into the weighted least squares
    A = X[:, :-1] - X[:, -1:]
    b = y - X[:, -1] * delta
    lhs = A.T @ (A * w[:, None])
    rhs = A.T @ (b * w)
    phi = np.linalg.lstsq(lhs + 1e-10 * np.eye(len(lhs)), rhs, rcond=None)[0]
    phi = np.append(phi, delta - phi.sum())
    return ([(kind, j, float(v)) for (kind, j), v in zip(every, phi)],
            {"sum": float(phi.sum()), "difference": float(delta), "evaluations": len(coalitions) + 2})


def _combinations(total, size):
    from itertools import combinations
    return combinations(range(total), size)


def subgraphx(ctx, column, index, every, rng, rollouts=10, expand=4, samples=10, min_nodes=5, c_puct=10., hops=2):
    """SubgraphX's search (module docstring); returns the parts with an importance from the pruning order, and the
    number of margin evaluations."""
    n = real_nodes(index)
    src, dst, kinds = index.fields['edge_src'], index.fields['edge_dst'], index.fields['edge_kind']
    neighbours = [set() for _ in range(n)]
    for s, d, k in zip(src, dst, kinds):
        if k != EDGE_GLOBAL and s < n and d < n and s != d:
            neighbours[int(s)].add(int(d))
            neighbours[int(d)].add(int(s))
    triples = [(t, int(src[e]), int(dst[e])) for t, e in enumerate(np.flatnonzero(kinds == EDGE_LABELLED)[:labelled_count(index)])]
    cache, evaluations = {}, [0]

    def subgraph_edit(nodes):
        nodes = set(nodes)
        return keep_only(index, sorted(nodes), [t for t, s, d in triples if s in nodes and d in nodes])

    def components(nodes):
        left, out = set(nodes), []
        while left:
            stack, comp = [min(left)], set()
            while stack:
                v = stack.pop()
                if v in comp:
                    continue
                comp.add(v)
                stack.extend(neighbours[v] & left - comp)
            left -= comp
            out.append(comp)
        return sorted(out, key=lambda c: (-len(c), min(c)))

    def local_players(state):
        frontier, seen = set(state), set(state)
        for _ in range(hops):
            frontier = {u for v in frontier for u in neighbours[v]} - seen
            seen |= frontier
        return sorted(seen - set(state))

    def rewards(states):
        todo = [s for s in states if s not in cache]
        edits, spans = [], []
        for state in todo:
            players = local_players(state)
            start = len(edits)
            for _ in range(samples):
                order = rng.permutation(len(players) + 1)
                position = int(np.flatnonzero(order == len(players))[0])
                coalition = [players[i] for i in order[:position]]
                edits += [subgraph_edit(coalition + list(state)), subgraph_edit(coalition)]
            spans.append((start, len(edits)))
        if edits:
            values = margin_values(ctx, column, edits)
            evaluations[0] += len(edits)
            for state, (a, b) in zip(todo, spans):
                v = values[a:b]
                cache[state] = float(np.mean(v[0::2] - v[1::2]))
        return [cache[s] for s in states]

    def children(state):
        degree = {v: len(neighbours[v] & state) for v in state}
        candidates = sorted(state, key=lambda v: (-degree[v], v))
        out = []
        for v in candidates:
            comps = components(state - {v})
            child = frozenset(comps[0]) if comps else frozenset()
            removed = [v] + sorted(state - {v} - child)
            out.append((child, removed))
        return out

    if n == 0:
        order = [("triple", t) for t in range(labelled_count(index))]
        return [(k, j, float(len(order) - i)) for i, (k, j) in enumerate(order)], 0
    comps = components(set(range(n)))
    root = frozenset(comps[0])
    removal = [sorted(set(range(n)) - root)]
    tree = {root: {"N": 0, "W": 0., "children": None, "parent": None, "removed": None}}
    for _ in range(rollouts):
        state, path = root, [root]
        while len(state) > min_nodes:
            node = tree[state]
            if node["children"] is None:
                kids = [(c, r) for c, r in children(state)[:expand] if c]
                rewards([c for c, _ in kids])
                node["children"] = [c for c, _ in kids]
                for c, r in kids:
                    tree.setdefault(c, {"N": 0, "W": 0., "children": None, "parent": state, "removed": r})
            if not node["children"]:
                break
            visits = sum(tree[c]["N"] for c in node["children"])
            state = max(node["children"], key=lambda c: (tree[c]["W"] / tree[c]["N"] if tree[c]["N"] else 0.)
                        + c_puct * cache[c] * math.sqrt(visits) / (1 + tree[c]["N"]))
            path.append(state)
        leaf = rewards([state])[0]
        for s in path:
            tree[s]["N"] += 1
            tree[s]["W"] += leaf
    small = [s for s in tree if len(s) <= min_nodes and s in cache]
    best = max(small, key=lambda s: (cache[s], -len(s))) if small else root
    chain, s = [], best
    while tree[s]["parent"] is not None:
        chain.append(tree[s]["removed"])
        s = tree[s]["parent"]
    removal += chain[::-1]
    state = best
    while len(state) > 1:
        kids = [(c, r) for c, r in children(state) if c]
        if not kids:
            break
        values = rewards([c for c, _ in kids])
        child, removed = kids[int(np.argmax(values))]
        removal.append(removed)
        state = child
    removal.append(sorted(state))
    node_order = [v for group in removal[::-1] for v in group]
    seen_nodes = set(node_order)
    node_order += [v for v in range(n) if v not in seen_nodes]
    placed, order = set(), []
    pending = sorted(triples)
    for v in node_order:
        order.append(("node", v))
        placed.add(v)
        ready = [t for t, s, d in pending if s in placed and d in placed]
        order += [("triple", t) for t in ready]
        pending = [(t, s, d) for t, s, d in pending if t not in set(ready)]
    order += [("triple", t) for t, _, _ in pending]
    known = set(every)
    order = [p for p in order if p in known]
    return [(k, j, float(len(order) - i)) for i, (k, j) in enumerate(order)], evaluations[0]


# GraphMask (Schlichtkrull et al., ICLR 2021) ------------------------------------------------------------------------

HC_BETA, HC_GAMMA, HC_ZETA = 2 / 3, -0.1, 1.1          # hard concrete (Louizos et al., ICLR 2018), as GraphMask


class GraphMask(nn.Module):
    """One gate per GAT message and layer, predicted by a small MLP per layer from the message's source and target
    vectors at that layer, its edge vector and the message itself; a closed message is replaced by a learned
    baseline vector of that layer (the messages' softmax weights stay as they are).

    The gates start open (output bias open_bias): from a random start the hard concrete gates of a checkpoint could all
    close at once, where the clamp leaves no gradient to reopen them however large the Lagrange multiplier grows
    (countonly seed 0: score change 0.61 of the graph effect against a tolerance of 0.05)."""

    def __init__(self, layers=2, dim=256, heads=4, channels=64, hidden=64, open_bias=3.):
        super().__init__()
        self.gates = nn.ModuleList(nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                   for _ in range(layers))
        for gate in self.gates:
            nn.init.constant_(gate[-1].bias, open_bias)
        self.baselines = nn.ParameterList(nn.Parameter(torch.zeros(heads, channels)) for _ in range(layers))

    def log_alpha(self, layer, x, edges, attrs, messages):
        src, dst = edges
        return self.gates[layer](torch.cat((x[src], x[dst], attrs, messages.flatten(1)), -1)).squeeze(-1)

    @staticmethod
    def open_probability(log_alpha):
        """P(gate != 0) of the hard concrete gate: GraphMask's measure of how much a message is needed."""
        return torch.sigmoid(log_alpha - HC_BETA * math.log(-HC_GAMMA / HC_ZETA))

    @staticmethod
    def gate(log_alpha, generator=None):
        """A sampled hard concrete gate when a generator is given (training), else the deterministic one."""
        if generator is None:
            s = torch.sigmoid(log_alpha)
        else:
            u = torch.rand(log_alpha.shape, generator=generator).clamp(1e-6, 1 - 1e-6).to(log_alpha.device)
            s = torch.sigmoid((u.log() - (1 - u).log() + log_alpha) / HC_BETA)
        return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0, 1)


@contextlib.contextmanager
def gated_messages(model, explainer, generator=None, record=None):
    """Run the model's two GAT layers with GraphMask's gates on their messages (PyG message hooks; the model's code is
    not changed). GATv2Conv drops the graph's self-loop edges and appends one self-loop per node after the other
    edges; the appended self-loops are not gated. record, a list, receives (layer, proper edge positions of the input
    edges, log alpha) per layer call."""
    handles, inputs = [], {}
    for layer, conv in enumerate((model.gat1, model.gat2)):
        def pre(module, args, layer=layer):
            x, edges, attrs = args[0], args[1], args[2]
            proper = edges[0] != edges[1]
            inputs[layer] = (x, edges[:, proper], attrs[proper], torch.nonzero(proper).squeeze(-1))

        def message(module, msg_inputs, out, layer=layer):
            x, edges, attrs, positions = inputs[layer]
            count = edges.shape[1]
            log_alpha = explainer.log_alpha(layer, x, edges, attrs, out[:count])
            if record is not None:
                record.append((layer, positions, log_alpha))
            g = explainer.gate(log_alpha, generator)[:, None, None]
            gated = g * out[:count] + (1 - g) * explainer.baselines[layer]
            return torch.cat((gated, out[count:]))
        handles += [conv.register_forward_pre_hook(pre), conv.register_message_forward_hook(message)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def graphmask_loss(ctx, column, explainer, index, base, effect, lagrange, generator, tolerance):
    """L0 penalty (mean P(gate != 0)) + lagrange * (the score change as a share of the graph effect - tolerance)."""
    images = index.to_batch(ctx.store.part_table(ctx.device), ctx.device)
    record = []
    with gated_messages(ctx.model, explainer, generator, record):
        fused, _ = ctx.score_images(column, images, 1)
    gates = torch.cat([explainer.open_probability(a) for _, _, a in record]) if record else fused.new_zeros(0)
    # an image without proper edges has no message to gate (the mean of nothing would be NaN)
    l0 = gates.mean() if gates.numel() else fused.new_zeros(())
    divergence = (fused[0, column] - base).abs() / effect
    return l0, divergence


def train_graphmask(contexts, seed, epochs=10, lr=3e-3, lagrange_lr=1e-2, tolerance=0.05):
    """Train GraphMask on the same training captions as PGExplainer (contexts: callables returning
    (ctx, column, index, base, empty)). GraphMask's constrained objective: minimize the L0 penalty subject to a
    score change of at most tolerance x the graph effect, by gradient descent on the gates and ascent on the Lagrange
    multiplier. Returns the explainer and the last epoch's mean L0 penalty and mean divergence."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        explainer = GraphMask()
        optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(seed)
        order_rng = np.random.default_rng(seed)
        lagrange, last = 1., []
        for _ in range(epochs):
            last = []
            for i in order_rng.permutation(len(contexts)):
                ctx, column, index, base, empty = contexts[i]()
                with torch.enable_grad():
                    l0, divergence = graphmask_loss(ctx, column, explainer, index, base, base - empty, lagrange,
                                                    generator, tolerance)
                    loss = l0 + lagrange * (divergence - tolerance)
                    optimizer.zero_grad()
                    grads = torch.autograd.grad(loss, list(explainer.parameters()), allow_unused=True)
                for p, g in zip(explainer.parameters(), grads):
                    p.grad = g
                optimizer.step()
                lagrange = max(0., lagrange + lagrange_lr * float(divergence.detach() - tolerance))
                last.append((float(l0.detach()), float(divergence.detach())))
    explainer.eval()
    return explainer, {"l0": float(np.mean([a for a, _ in last])) if last else None,
                       "divergence": float(np.mean([b for _, b in last])) if last else None, "lagrange": lagrange}


@torch.no_grad()
def graphmask(ctx, column, index, explainer):
    """GraphMask's open probabilities of the image's messages: per input edge, the larger of its two layers (NaN for
    an edge that carries no message, a self-loop relation), and per part, for the shared comparison: a triple takes
    its labelled edge's value, an object node the largest value among the messages it sends."""
    images = index.to_batch(ctx.store.part_table(ctx.device), ctx.device)
    record = []
    with gated_messages(ctx.model, explainer, None, record):
        ctx.score_images(column, images, 1)
    edges = np.full(int(index.edge_counts[0]), np.nan)
    for _, positions, log_alpha in record:
        p = explainer.open_probability(log_alpha).cpu().numpy()
        pos = positions.cpu().numpy()
        edges[pos] = np.fmax(edges[pos], p)
    n, m = real_nodes(index), labelled_count(index)
    src = index.fields['edge_src']
    nodes = np.zeros(n)
    for e, v in enumerate(edges):
        if np.isfinite(v) and src[e] < n:
            nodes[src[e]] = max(nodes[src[e]], v)
    parts = [("node", j, float(nodes[j])) for j in range(n)] + \
            [("triple", j, float(np.nan_to_num(edges[j]))) for j in range(m)]
    return parts, edges


# GStarX (Zhang et al., NeurIPS 2022) --------------------------------------------------------------------------------

def part_graph(index, every):
    """Adjacency (list of sets, positions in every) of the parts: two object nodes are adjacent when a labelled or
    completion edge joins them (the global node is not a part), and a triple is adjacent to its two end nodes."""
    n = real_nodes(index)
    src, dst, kinds = index.fields['edge_src'], index.fields['edge_dst'], index.fields['edge_kind']
    position = {part: i for i, part in enumerate(every)}
    adjacency = [set() for _ in every]
    for e, (s, d, k) in enumerate(zip(src, dst, kinds)):
        if k == EDGE_GLOBAL or s >= n or d >= n:
            continue
        ends = [position.get(("node", int(s))), position.get(("node", int(d)))]
        if s != d and None not in ends:
            adjacency[ends[0]].add(ends[1])
            adjacency[ends[1]].add(ends[0])
        if k == EDGE_LABELLED and ("triple", e) in position:
            t = position[("triple", e)]
            for v in ends:
                if v is not None:
                    adjacency[t].add(v)
                    adjacency[v].add(t)
    return adjacency


def _connected(members, adjacency):
    members = set(members)
    if not members:
        return False
    stack, seen = [min(members)], set()
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(adjacency[v] & members - seen)
    return seen == members


def hamiache_navarro(players, adjacency, value, tau=None, squarings=30, tol=1e-12):
    """Exact Hamiache-Navarro value of a graph game on a few players.

    value: {frozenset (connected coalition): v(S)} with v(empty) = 0. The associated game
    v*(S) = v(S) + tau * sum_{j in N(S)} [v(S + j) - v(S) - v(j)] (Hamiache 2001; Navarro 2020 for graph games) only
    involves connected coalitions, so it is a linear map on them; its iterates converge, for 0 < tau < 2 / n, to an
    inessential game whose value on {i} is i's HN value (on a complete graph, the Shapley value). The limit is taken
    by squaring the map until it stops changing (squaring past that point only adds rounding error).
    """
    coalitions = [c for c in value if c]
    where = {c: i for i, c in enumerate(coalitions)}
    n = len(players)
    tau = tau if tau is not None else 1. / n
    M = np.eye(len(coalitions))
    for c, i in where.items():
        neighbours = set().union(*(adjacency[p] for p in c)) & set(players) - c
        for j in neighbours:
            M[i, where[c | {j}]] += tau
            M[i, i] -= tau
            M[i, where[frozenset([j])]] -= tau
    for _ in range(squarings):
        squared = M @ M
        done = np.abs(squared - M).max() < tol
        M = squared
        if done:
            break
    v = np.array([value[c] for c in coalitions])
    limit = M @ v
    return {p: float(limit[where[frozenset([p])]]) for p in players}


def _grow(seed, adjacency, size, rng):
    """A random connected set of at most size parts containing seed, grown one random neighbour at a time."""
    members = {seed}
    while len(members) < size:
        frontier = sorted(set().union(*(adjacency[p] for p in members)) - members)
        if not frontier:
            break
        members.add(int(rng.choice(frontier)))
    return frozenset(members)


def gstarx(ctx, column, index, every, rng, size=5):
    """GStarX's structure-aware values of the parts.

    The value function is the rank-1 margin when only a coalition of parts is kept, minus its value with no part (the
    one of the Shapley reference). A graph of 44 parts is too big for the exact HN value, so, like GStarX's sampled
    version, one connected set of at most size parts is grown from every part, the exact HN value is taken inside
    each set (the parts outside it removed), and a part's value is its mean over the sets that contain it. Margins
    are cached across sets. Returns the parts with their value and the number of margin evaluations.
    """
    adjacency = part_graph(index, every)
    samples = [_grow(i, adjacency, size, rng) for i in range(len(every))]
    needed = {frozenset()}
    for sample in samples:
        members = sorted(sample)
        for r in range(1, len(members) + 1):
            needed.update(frozenset(c) for c in combinations(members, r) if _connected(c, adjacency))
    needed = sorted(needed, key=lambda c: (len(c), sorted(c)))
    margins = margin_values(ctx, column, [keep_only(index, *split([every[i] for i in sorted(c)])) for c in needed])
    value = {c: float(v - margins[0]) for c, v in zip(needed, margins)}
    totals, counts = np.zeros(len(every)), np.zeros(len(every))
    for sample in samples:
        local = {c: v for c, v in value.items() if c and c <= sample}
        for p, phi in hamiache_navarro(sorted(sample), adjacency, local).items():
            totals[p] += phi
            counts[p] += 1
    scores = totals / np.maximum(counts, 1)
    return [(kind, j, float(s)) for (kind, j), s in zip(every, scores)], len(needed)
