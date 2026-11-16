"""Whole-graph matching between a caption graph and an image graph (training-free probes, improvement step 2).

The level-1 channels match every caption part on its own, so "man riding horse" and "horse on beach" may use two
different horses. Two scores that look at the joint assignment instead:
- fgw_score: Fused Gromov-Wasserstein (Vayer et al., 2019) between the graphs. Node cost 1 - cosine of node vectors,
  structure = hop distance on the undirected relation graph, capped and scaled to [0, 1]. "balanced" is the plain FGW
  (both marginals uniform); "semirelaxed" frees the image marginal (Vincent-Cuaz et al., 2022), so a small caption
  graph can map into part of a large image graph. Score = -distance; the transport plan says which caption node went
  to which image node.
- consistency_score: for each pair of caption triples that share a node, take the best-matching image triple of each
  and check that the shared caption node lands on the same image node in both. Score = share of consistent pairs.
"""
from __future__ import annotations

import numpy as np

HOP_CAP = 3


def hop_distance(n: int, edges, cap: int = HOP_CAP) -> np.ndarray:
    """Shortest-path hop counts on the undirected graph, capped at cap (also for disconnected pairs), scaled by cap."""
    dist = np.full((n, n), cap, dtype=np.float64)
    np.fill_diagonal(dist, 0)
    adjacency = [set() for _ in range(n)]
    for s, o in edges:
        if s != o:
            adjacency[s].add(o)
            adjacency[o].add(s)
    for start in range(n):
        frontier, seen, depth = [start], {start}, 0
        while frontier and depth < cap - 1:
            depth += 1
            nxt = []
            for node in frontier:
                for other in adjacency[node] - seen:
                    seen.add(other)
                    dist[start, other] = depth
                    nxt.append(other)
            frontier = nxt
    return dist / cap


def fgw_score(query_vectors, query_dist, image_vectors, image_dist, alpha=.5, mode="balanced"):
    """-FGW distance and the transport plan (query nodes x image nodes); (nan, None) if a side has no node."""
    import ot
    if len(query_vectors) == 0 or len(image_vectors) == 0:
        return float("nan"), None
    cost = 1. - query_vectors @ image_vectors.T
    p = np.full(len(query_vectors), 1. / len(query_vectors))
    if mode == "balanced":
        q = np.full(len(image_vectors), 1. / len(image_vectors))
        plan, log = ot.gromov.fused_gromov_wasserstein(cost, query_dist, image_dist, p, q, loss_fun="square_loss",
                                                       alpha=alpha, log=True)
        return -float(log["fgw_dist"]), plan
    if mode == "semirelaxed":
        plan, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(cost, query_dist, image_dist, p,
                                                                   loss_fun="square_loss", alpha=alpha, log=True)
        return -float(log["srfgw_dist"]), plan
    raise ValueError(mode)


def shared_node_pairs(edges) -> list[tuple[int, int, int, int]]:
    """(triple i, role in i, triple j, role in j) for every pair of triples sharing a node; role 0 subject, 1 object."""
    pairs = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            for ri in (0, 1):
                for rj in (0, 1):
                    if edges[i][ri] == edges[j][rj]:
                        pairs.append((i, ri, j, rj))
    return pairs


def consistency_score(pairs, query_triple_vectors, image_triple_vectors, image_edges) -> float:
    """Share of shared-node caption triple pairs whose best image triples share the corresponding image node."""
    if not pairs or len(image_triple_vectors) == 0:
        return float("nan")
    best = (query_triple_vectors @ image_triple_vectors.T).argmax(-1)
    return float(np.mean([image_edges[best[i]][ri] == image_edges[best[j]][rj] for i, ri, j, rj in pairs]))
