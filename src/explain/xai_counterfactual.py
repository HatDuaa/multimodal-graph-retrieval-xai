"""Counterfactual explanations of the reranker: the fewest image parts whose removal changes the top of the ranking.

The explained image is the rank-1 candidate of a caption. A counterfactual is a set of its parts (object nodes and
labelled triples) such that removing them (xai_metrics.edit, the same edit as the deletion curves) makes the image
lose rank 1, or, with a target column (the gold image of a caption whose rank-1 image is wrong), makes the target
rank 1. Only the explained image is edited; the caption and the other 49 candidates stay as they are.

- exact_counterfactual: every set of at most max_size parts, smallest first; the reference of the smallest
  counterfactual within that size.
- cf_gnnexplainer: CF-GNNExplainer (Lucic et al., AISTATS 2022) on part masks. The original learns a sparse
  perturbation of the adjacency matrix with the loss L_pred + beta * L_dist, where L_pred is active only while the
  prediction has not changed. Here a part with mask m takes m * its vector + (1 - m) * its kind's neutral vector
  (xai_baselines.blend, as the GNNExplainer adaptation), L_pred is a hinge on the rank-1 margin (or the target's
  margin) on the scale of the graph effect, and L_dist is the share of parts removed. After every step the parts with
  m < 0.5 are removed for real and the edit is rescored; the smallest valid removal seen is the counterfactual.
- greedy counterfactuals of any explainer's order are the k_flip of its deletion curve (scripts/xai_metrics.curves).
"""
from itertools import combinations

import numpy as np
import torch

from src.explain.xai_baselines import blend, neutral_parts
from src.explain.xai_metrics import edit, labelled_count, rank_of, real_nodes, split


def removal(index, parts):
    nodes, triples = split(parts)
    return edit(index, drop_nodes=nodes, drop_edges=triples)


def flipped(full, column, target=None):
    """Per row of (n, 50) final scores: True when the edit changed the top as asked (column lost rank 1, or target
    is rank 1)."""
    scores = torch.from_numpy(np.asarray(full))
    if target is None:
        return (rank_of(scores, column) > 1).numpy()
    return (rank_of(scores, target) == 1).numpy()


def exact_counterfactual(ctx, column, index, every, max_size=2, target=None, chunk=256):
    """The smallest counterfactuals among all sets of at most max_size of the parts every.

    Returns {"size": smallest size with a counterfactual (None if none up to max_size), "sets": how many sets of
    that size are counterfactuals, "parts": the one among them with the lowest score of the explained image (target
    None) or the highest target margin, "evaluations": edits rescored}.
    """
    evaluations = 0
    for size in range(1, max_size + 1):
        sets = list(combinations(range(len(every)), size))
        if not sets:
            break
        hits, best, best_value = 0, None, None
        for start in range(0, len(sets), chunk):
            batch = sets[start:start + chunk]
            *_, full = ctx.rescore(column, [removal(index, [every[i] for i in s]) for s in batch], full=True)
            evaluations += len(batch)
            ok = flipped(full, column, target)
            for s, row, good in zip(batch, full, ok):
                if not good:
                    continue
                hits += 1
                value = (-row[column] if target is None
                         else row[target] - np.delete(row, target).max())
                if best_value is None or value > best_value:
                    best, best_value = s, value
        if hits:
            return {"size": size, "sets": hits, "parts": [every[i] for i in best], "evaluations": evaluations}
    return {"size": None, "sets": 0, "parts": None, "evaluations": evaluations}


def cf_gnnexplainer(ctx, column, index, base, empty, target=None, steps=200, beta=0.5, lr=0.1, kappa=0.01):
    """CF-GNNExplainer's search on part masks (module docstring).

    Returns (parts with importance 1 - final mask, for the deletion and insertion curves like the other explainers;
    {"parts": the smallest valid removal seen or None, "size", "step": the step it was first found at,
    "evaluations": edits rescored}). Called only when base - empty > 1e-4.
    """
    effect = base - empty
    assert effect > 1e-4, effect
    images = index.to_batch(ctx.store.part_table(ctx.device), ctx.device)
    n, m = real_nodes(index), labelled_count(index)
    every = [("node", j) for j in range(n)] + [("triple", j) for j in range(m)]
    vectors = neutral_parts(images, n, m)
    logits = torch.full((n + m,), 3., device=ctx.device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    best, best_step, seen, evaluations = None, None, set(), 0
    for step in range(steps):
        mask = torch.sigmoid(logits)
        with torch.enable_grad():
            fused, _ = ctx.score_images(column, blend(images, mask, n, m, vectors), 1)
            row = fused[0]
            if target is None:
                others = torch.cat((row[:column], row[column + 1:]))
                margin = row[column] - others.max()
            else:
                others = torch.cat((row[:target], row[target + 1:]))
                margin = others.max() - row[target]
            loss = torch.relu(margin / effect + kappa) + beta * (1 - mask).mean()
            logits.grad, = torch.autograd.grad(loss, [logits])
        optimizer.step()
        removed = tuple(np.flatnonzero(torch.sigmoid(logits).detach().cpu().numpy() < .5).tolist())
        if not removed or removed in seen or (best is not None and len(removed) >= len(best)):
            continue
        seen.add(removed)
        *_, full = ctx.rescore(column, [removal(index, [every[i] for i in removed])], full=True)
        evaluations += 1
        if flipped(full, column, target)[0]:
            best, best_step = removed, step
    weights = 1 - torch.sigmoid(logits).detach().cpu().numpy()
    parts = [(kind, j, float(w)) for (kind, j), w in zip(every, weights)]
    found = [every[i] for i in best] if best is not None else None
    return parts, {"parts": found, "size": len(best) if best is not None else None, "step": best_step,
                   "evaluations": evaluations}
