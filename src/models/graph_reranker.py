"""Shared PyG encoder and differentiable three-channel graph reranking."""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

from src.models.concept_gat import ConceptGAT

# Log-domain stand-in for -inf in the Sinkhorn plan: finite, so masked rows and columns never produce nan.
NEG = -1e4


def zscore_tensor(values, eps=1e-6, floor=0.):
    """Population z-score with masked missing values and finite inactive gradients.

    floor > 0 divides by max(std, floor): a query whose candidates barely differ in this channel is no longer
    stretched to unit spread.
    """
    known = torch.isfinite(values)
    count = known.sum(-1, keepdim=True)
    clean = torch.where(known, values, 0.)
    mean = clean.sum(-1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(known, clean - mean, 0.)
    var = centered.square().sum(-1, keepdim=True) / count.clamp_min(1)
    active = (count >= 2) & (var >= eps ** 2)
    # sqrt(0) has an infinite derivative even behind a torch.where mask.
    std = var.clamp_min(eps ** 2).sqrt()
    if floor:
        std = std.clamp_min(floor)
    z = torch.where(active, centered / (std + eps), 0.)
    return z, active.squeeze(-1)


def fill_known_mean(values):
    """Missing scores (nan) take the mean of the query's known scores; rows without any become 0."""
    known = torch.isfinite(values)
    clean = torch.where(known, values, 0.)
    mean = clean.sum(-1, keepdim=True) / known.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(known, clean, mean)


def fuse_raw(clip, objects, triples, weights):
    """fused = a * CLIP + b * S_obj + c * S_tri on raw scores (no z-score).

    Cross-entropy over a query's candidates ignores a constant added to all of them, so neither the missing-score
    fill (the known mean, which z-scoring maps to 0 as well) nor an inactive channel moves the ranking.
    """
    _, obj_active = zscore_tensor(objects)
    _, tri_active = zscore_tensor(triples)
    obj, tri = fill_known_mean(objects), fill_known_mean(triples)
    graph = weights[:, 1:2] * obj + weights[:, 2:3] * tri
    fused = weights[:, :1] * clip + graph
    center = lambda x: x - x.mean(-1, keepdim=True)  # noqa: E731
    return fused, {"z_clip": center(clip), "z_objects": center(obj), "z_triples": center(tri), "graph": graph,
                   "object_active": obj_active, "triple_active": tri_active, "weights": weights}


def fuse_tensor(clip, objects, triples, weights, floors=(0., 0., 0.)):
    z_clip, _ = zscore_tensor(clip, floor=floors[0])
    z_obj, obj_active = zscore_tensor(objects, floor=floors[1])
    z_tri, tri_active = zscore_tensor(triples, floor=floors[2])
    mix = (weights[:, 1:2] * z_obj + weights[:, 2:3] * z_tri) / weights[:, 1:].sum(-1, keepdim=True)
    graph_both, _ = zscore_tensor(mix)
    graph = torch.where((obj_active & tri_active)[:, None], graph_both,
                        torch.where(obj_active[:, None], z_obj, torch.where(tri_active[:, None], z_tri, 0.)))
    fused = weights[:, :1] * z_clip + (1 - weights[:, :1]) * graph
    return fused, {"z_clip": z_clip, "z_objects": z_obj, "z_triples": z_tri, "graph": graph,
                   "object_active": obj_active, "triple_active": tri_active, "weights": weights}


def masked_softmax(logits, mask, dim):
    masked = logits.masked_fill(~mask, -torch.inf)
    masked = torch.where(mask.any(dim, keepdim=True), masked, torch.zeros_like(masked))
    return torch.softmax(masked, dim) * mask


def sinkhorn_plan(sims, row_mask, column_mask, tau, dustbin, iterations):
    """Transport plan (..., R, C) of R caption parts onto C image parts, with one dustbin row and one dustbin column.

    Every valid row (caption part) sends one unit of mass and every valid column (image part) receives at most one,
    so several caption parts cannot all lean on the same image part. Mass a row cannot place goes to the dustbin,
    whose score is ``dustbin`` in similarity units; invalid rows and columns get no mass. Log-domain iterations as in
    SuperGlue (Sarlin et al., 2020). sims is (..., R, C), row_mask (..., R) and column_mask (..., C).
    """
    rows = row_mask.sum(-1).to(sims.dtype)
    columns = column_mask.sum(-1).to(sims.dtype)
    log_total = (rows + columns).clamp_min(1).log()
    pair = row_mask[..., :, None] & column_mask[..., None, :]
    logits = (sims / tau).masked_fill(~pair, NEG)
    bin_logit = (dustbin / tau).expand(sims.shape[:-2])[..., None, None]
    right = bin_logit.expand(*sims.shape[:-1], 1).masked_fill(~row_mask[..., :, None], NEG)
    bottom = bin_logit.expand(*sims.shape[:-2], 1, sims.shape[-1]).masked_fill(~column_mask[..., None, :], NEG)
    full = torch.cat((torch.cat((logits, right), -1), torch.cat((bottom, bin_logit), -1)), -2)
    # Row marginals: 1 per valid row and the columns' total for the dustbin row; columns symmetric; all divided by
    # rows + columns.
    log_mu = torch.cat((torch.where(row_mask, 0., NEG) - log_total[..., None],
                        (columns.clamp_min(1e-9).log() - log_total)[..., None]), -1)
    log_nu = torch.cat((torch.where(column_mask, 0., NEG) - log_total[..., None],
                        (rows.clamp_min(1e-9).log() - log_total)[..., None]), -1)
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(full + v[..., None, :], -1)
        v = log_nu - torch.logsumexp(full + u[..., :, None], -2)
    plan = (full + u[..., :, None] + v[..., None, :] + log_total[..., None, None]).exp()
    return plan[..., :-1, :-1] * pair


def slot_similarity(cos):
    """Map a per-slot cosine into [0, 1]: smooth everywhere, so no slot loses its gradient (unlike relu)."""
    return (1 + cos) / 2


def gathered_cosine(query, images, candidate_index):
    """query (B, Tq, D) . images[candidate_index] (B, K, Ti, D) -> (B, K, Tq, Ti)."""
    return torch.matmul(query[:, None], images[candidate_index].transpose(-1, -2))


def part_similarity(query, images, candidate_index):
    """Similarity of every query part to every part of each candidate image, shape (B, K, Tq, Ti).

    query is (B, Tq, D) and images (N, Ti, D), or factorized triples (B, Tq, 3, D) and (N, Ti, 3, D) holding
    subject, predicate and object vectors; a factorized pair scores c(s) * c(p) * c(o), so one mismatched slot
    pulls the whole pair down. Factorized slots are gathered one at a time, each under its own checkpoint, so the
    (B, K, Ti, D) candidate copy exists for one slot at a time, as in the phrase match.
    """
    if query.dim() == 3:
        return gathered_cosine(query, images, candidate_index)
    sims = 1.
    for slot in range(query.shape[-2]):
        args = (query[..., slot, :], images[..., slot, :], candidate_index)
        cos = checkpoint(gathered_cosine, *args, use_reentrant=False) if torch.is_grad_enabled() else gathered_cosine(*args)
        sims = sims * slot_similarity(cos)
    return sims


def part_summary(query):
    """One vector per query part (B, Tq, D) for the weights w: factorized triples use the mean of their slots."""
    return query if query.dim() == 3 else query.mean(-2)


def count_bias(table, query_class, image_class, candidate_index):
    """table[query count class, image count bucket] for every query node / candidate node pair, (B, K, Tq, Ti)."""
    return table[query_class[:, None, :, None], image_class[candidate_index][:, :, None, :]]


def channel_score(query, query_mask, images, image_mask, candidate_index, tau, u, query_class=None, image_class=None,
                  table=None, match="softmax", dustbin=None, iterations=20):
    """Match all B x K pairs together; no loop or materialized feature cache per candidate.

    With a count table, sim_ij + table[class_i, bucket_j] replaces sim_ij in both the attention and the score.
    match="softmax" lets every caption part attend over all parts of the candidate (several may share one image
    part); match="sinkhorn" replaces the attention by the transport plan of sinkhorn_plan, with a dustbin score.
    """
    masks = image_mask[candidate_index]
    sims = part_similarity(query, images, candidate_index)
    if table is not None:
        sims = sims + count_bias(table, query_class, image_class, candidate_index)
    if match == "sinkhorn":
        rows = query_mask[:, None, :].expand(-1, masks.shape[1], -1)
        attention = sinkhorn_plan(sims, rows, masks, tau, dustbin, iterations)
    else:
        attention = masked_softmax(sims / tau, masks[:, :, None, :], -1)
    per_part = (attention * sims).sum(-1)
    weights = masked_softmax(part_summary(query) @ u, query_mask, -1)
    scores = (weights[:, None, :] * per_part).sum(-1)
    valid = query_mask.any(-1)[:, None] & masks.any(-1)
    return scores.masked_fill(~valid, torch.nan)


def attention_pool(nodes, mask, u):
    """Graph vector = normalize(sum_n alpha_n * node_n), alpha = softmax over the graph's nodes of node_n . u."""
    alpha = masked_softmax(nodes @ u, mask, -1)
    return F.normalize((alpha[..., None] * nodes).sum(-2), dim=-1), alpha


def graph_level_score(query, query_mask, images, image_mask, candidate_index):
    """Cosine of pooled query and candidate graph vectors; nan where either graph has no scorable node."""
    scores = (query[:, None] * images[candidate_index]).sum(-1)
    valid = query_mask.any(-1)[:, None] & image_mask.any(-1)[candidate_index]
    return scores.masked_fill(~valid, torch.nan)


def dense_counts(graphs, width):
    """Count index of every scorable node in the dense (graph, node) layout of encode, padded to width with 0."""
    mask = graphs.part_mask
    values = getattr(graphs, 'count_index', None)
    if values is None:
        values = torch.zeros(len(mask), dtype=torch.long, device=mask.device)
    dense, _ = to_dense_batch(values[mask], graphs.batch[mask], batch_size=graphs.num_graphs, fill_value=0)
    return F.pad(dense, (0, width - dense.shape[1])) if dense.shape[1] < width else dense


TRIPLE_MATCHES = ("phrase", "factorized")
FUSIONS = ("zscore", "raw")
PART_MATCHES = ("softmax", "sinkhorn")


class GraphReranker(nn.Module):
    """Three channels: CLIP, objects, and a third graph channel.

    The third channel is the triple channel by default. triple_match="phrase" matches one vector per triple,
    normalize(CLIP phrase + U triple_mlp([h_s; e; h_o])); "factorized" matches (subject node, CLIP relation,
    object node) slot by slot. graph_channel=True replaces the triple channel by the cosine of attention-pooled
    query and image graph vectors.

    Full image graphs (experiment 3; batches from GraphStore with structure options):
    - edge_geometry: the six box-geometry numbers of an edge (src/graph/structure.py) go through a small MLP and
      are added to its edge vector e (GAT and triple_mlp); edges without boxes (sentence side, global links) add 0.
      The MLP's last layer starts at zero, so step 0 equals the model without geometry.
    - completion_edges: added edges (k-nearest, bridges) take a learned "[none]" predicate vector through p_edge
      instead of a CLIP relation vector; they carry GAT messages only and are never triples.
    - global_node: one node per graph, its input vector the normalized mean of the graph's scorable node vectors
      (so it already summarizes the graph at the first GAT layer), linked both ways to every node by edges with a
      learned "[global]" predicate vector; it is not an object for the object channel.
    - count_table: a learned table g[caption count class, image count bucket] (src/graph/counts.py), zero at start,
      added to every node pair similarity of the object channel; the row of unknown caption counts stays 0.
    Scene-ranker options (docs/scene-ranker.html); all off by default, and each starts at step 0 from the model
    without it except part_match:
    - part_match="sinkhorn": both channels match parts with sinkhorn_plan (one unit per image part, a learned dustbin
      score per channel) instead of a softmax attention; the channel temperatures start at 0.02.
    - graph_loss: a learned temperature for the graph-only loss, CE(graph / T_graph); the training script weights it.
    - concept_graph: a commonsense concept graph (src/graph/concepts.py). A relational GAT encodes its concepts, and
      each node whose label links to a concept adds a projection of that concept vector, zero at start, to its input.
    - scene_terms: number of fixed scene terms per (caption, candidate) in batch['terms'] (scaled offline), added to
      the graph score with learned weights d, zero at start.
    record_attention = True keeps the GAT attention of the last encode calls in self.attention.
    """

    def __init__(self, adaptive_weights=True, use_gat=True, use_triple_channel=True,
                 use_edges=True, query_graph_encoder=True, dropout=0.1, triple_match="phrase", graph_channel=False,
                 fusion="zscore", fusion_scale=None, zscore_floor=None, edge_geometry=False, completion_edges=False,
                 global_node=False, count_table=False, part_match="softmax", sinkhorn_iterations=20, graph_loss=False,
                 concept_graph=None, scene_terms=0):
        super().__init__()
        if part_match not in PART_MATCHES:
            raise ValueError(f"part_match must be one of {PART_MATCHES}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if zscore_floor is not None and fusion != "zscore":
            raise ValueError("zscore_floor applies to the z-score fusion only")
        if triple_match not in TRIPLE_MATCHES:
            raise ValueError(f"triple_match must be one of {TRIPLE_MATCHES}")
        if graph_channel and triple_match != "phrase":
            raise ValueError("graph_channel replaces the triple channel; it takes no triple_match")
        self.options = dict(adaptive_weights=adaptive_weights, use_gat=use_gat,
                            use_triple_channel=use_triple_channel, use_edges=use_edges,
                            query_graph_encoder=query_graph_encoder, dropout=dropout,
                            triple_match=triple_match, graph_channel=graph_channel, fusion=fusion,
                            fusion_scale=fusion_scale, zscore_floor=zscore_floor, edge_geometry=edge_geometry,
                            completion_edges=completion_edges, global_node=global_node, count_table=count_table,
                            part_match=part_match, sinkhorn_iterations=sinkhorn_iterations, graph_loss=graph_loss,
                            concept_gat=concept_graph is not None, scene_terms=scene_terms)
        self.p_node = nn.Linear(512, 256)
        self.p_edge = nn.Linear(512, 256)
        self.gat1 = GATv2Conv(256, 64, heads=4, edge_dim=256, fill_value=0., dropout=dropout)
        self.gat2 = GATv2Conv(256, 64, heads=4, edge_dim=256, fill_value=0., dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.triple_mlp = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 256))
        self.u = nn.Linear(256, 512, bias=False)
        nn.init.zeros_(self.u.weight)
        self.u_objects = nn.Parameter(torch.zeros(512))
        self.u_triples = nn.Parameter(torch.zeros(512))
        self.log_tau_objects = nn.Parameter(torch.tensor(math.log(.05)))
        self.log_tau_triples = nn.Parameter(torch.tensor(math.log(.05)))
        self.log_temperature = nn.Parameter(torch.tensor(0.))
        self.weight_map = nn.Parameter(torch.zeros(3, 512), requires_grad=adaptive_weights)
        self.weight_bias = nn.Parameter(torch.log(torch.tensor([.60, .28, .12])))
        if fusion == "raw" and fusion_scale is not None:
            # Step-0 equivalence with the z-score fusion: z_k = s_k / std_k, so raw weights proportional to
            # base_k / std_k (std measured on train), and the overall scale 1 / sum(base_k / std_k) moves into T.
            raw = torch.tensor([.60, .28, .12]) / torch.tensor(fusion_scale, dtype=torch.float32)
            with torch.no_grad():
                self.weight_bias.copy_(torch.log(raw / raw.sum()))
                self.log_temperature.fill_(-math.log(float(raw.sum())))
        if graph_channel:
            # Zero init: both sides start from mean pooling of their node vectors.
            self.u_pool_query = nn.Parameter(torch.zeros(512))
            self.u_pool_image = nn.Parameter(torch.zeros(512))
        if edge_geometry:
            self.geometry_mlp = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 256))
            nn.init.zeros_(self.geometry_mlp[2].weight)
            nn.init.zeros_(self.geometry_mlp[2].bias)
        # Learned predicate vectors of added edges, unit norm at start like the CLIP relation vectors they replace.
        if completion_edges:
            self.none_predicate = nn.Parameter(F.normalize(torch.randn(512), dim=0))
        if global_node:
            self.global_predicate = nn.Parameter(F.normalize(torch.randn(512), dim=0))
        if count_table:
            from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS
            # Known caption classes only; the unknown row is prepended as a constant 0 in counts_table().
            self.count_table = nn.Parameter(torch.zeros(len(CAPTION_CLASSES) - 1, len(IMAGE_BUCKETS)))
        if part_match == "sinkhorn":
            # Values of the B/32 scene ranker (docs/scene-ranker.html): a sharp plan and a dustbin that a part takes
            # unless some image part is closer than cosine 0.85.
            with torch.no_grad():
                self.log_tau_objects.fill_(math.log(.02))
                self.log_tau_triples.fill_(math.log(.02))
            self.dustbin_objects = nn.Parameter(torch.tensor(.85))
            self.dustbin_triples = nn.Parameter(torch.tensor(.85))
        if graph_loss:
            # Same start as the fused loss's temperature, so both losses begin on one scale.
            self.log_graph_temperature = nn.Parameter(self.log_temperature.detach().clone())
        if concept_graph is not None:
            self.concept_gat = ConceptGAT(concept_graph, hidden=256)
            # Zero init: step 0 equals the model without concepts.
            self.concept_project = nn.Linear(256, 256)
            nn.init.zeros_(self.concept_project.weight)
            nn.init.zeros_(self.concept_project.bias)
        if scene_terms:
            self.term_weights = nn.Parameter(torch.zeros(scene_terms))
        self.record_attention, self.attention = False, []

    def counts_table(self):
        """g with its fixed zero row for unknown caption counts, (caption classes, image buckets)."""
        return torch.cat((self.count_table.new_zeros((1, self.count_table.shape[1])), self.count_table))

    def tau_floor(self):
        # The Sinkhorn plan starts at 0.02, so it needs a lower floor than the softmax attention.
        return .005 if self.options['part_match'] == 'sinkhorn' else .01

    def tau_objects(self):
        return self.log_tau_objects.exp().clamp(self.tau_floor(), 1.)

    def tau_triples(self):
        return self.log_tau_triples.exp().clamp(self.tau_floor(), 1.)

    def loss_floor(self):
        # Raw CLIP cosines differ by about 0.02 inside a top-50, so the raw fusion needs a much lower floor.
        return .005 if self.options['fusion'] == 'raw' else .05

    def loss_temperature(self):
        return self.log_temperature.exp().clamp(self.loss_floor(), 5.)

    def graph_temperature(self):
        """Temperature of the graph-only loss (graph_loss=True)."""
        return self.log_graph_temperature.exp().clamp(self.loss_floor(), 5.)

    def concept_vectors(self):
        """One input vector per concept: the concept GAT output through the zero-initialized projection."""
        return self.concept_project(self.concept_gat())

    def concept_input(self, graphs, concepts):
        """Concept vector of every node (zero for nodes without a linked concept), added to p_node."""
        index = getattr(graphs, "node_concept", None)
        if index is None:
            raise ValueError('concept_gat needs batches with node_concept: use a data version with concepts '
                             '(configs/concept_v1.yaml)')
        return torch.where((index >= 0)[:, None], concepts[index.clamp_min(0)], 0.)

    def match_options(self, channel):
        """channel_score keyword arguments of the part matching for 'objects' or 'triples'."""
        if self.options['part_match'] != 'sinkhorn':
            return {}
        return {'match': 'sinkhorn', 'dustbin': getattr(self, f'dustbin_{channel}'),
                'iterations': self.options['sinkhorn_iterations']}

    def set_epoch(self, epoch):
        self.weight_map.requires_grad_(epoch > 1 and self.options['adaptive_weights'])
        self.weight_bias.requires_grad_(epoch > 1)

    def uses_third_channel(self):
        return self.options['use_triple_channel'] and (self.options['use_edges'] or self.options['graph_channel'])

    def edge_vectors(self, graphs):
        """p_edge of every edge's predicate vector (+ geometry), and the labelled edges' positions (None: all)."""
        kind, labelled = getattr(graphs, 'edge_kind', None), getattr(graphs, 'labelled_index', None)
        edge_attr = self.p_edge(graphs.edge_attr)
        if kind is not None:
            # torch.where broadcasts the two learned vectors: their gradient is a plain sum over edges, unlike a
            # gather of a 3-row table by edge kind, whose backward sorts ~300k duplicate indices per micro-batch.
            added = edge_attr.new_zeros(256)
            if self.options['completion_edges']:
                added = torch.where((kind == 1)[:, None], self.p_edge(self.none_predicate), added)
            if self.options['global_node']:
                added = torch.where((kind == 2)[:, None], self.p_edge(self.global_predicate), added)
            edge_attr = added.expand(len(kind), 256).index_put((labelled,), edge_attr)
        geometry = getattr(graphs, 'edge_geometry', None)
        if self.options['edge_geometry'] and geometry is not None:
            known = torch.isfinite(geometry[:, :1])
            edge_attr = edge_attr + torch.where(known, self.geometry_mlp(torch.nan_to_num(geometry)), 0.)
        return edge_attr, labelled

    def gat(self, layer, h, edges, attrs):
        if not self.record_attention:
            return layer(h, edges, attrs)
        out, (index, alpha) = layer(h, edges, attrs, return_attention_weights=True)
        self.attention.append((index.detach(), alpha.detach()))
        return out

    def encode(self, graphs: Batch, query=False, concepts=None):
        """Encoded nodes and triples of a graph batch; concepts is concept_vectors(), computed here when None."""
        nodes, triples = graphs.x, graphs.triple_f
        global_mask = getattr(graphs, 'global_mask', None)
        kind = getattr(graphs, 'edge_kind', None)
        if (kind is not None and not (self.options['completion_edges'] or self.options['global_node'])) or \
                (global_mask is not None and not self.options['global_node']):
            raise ValueError('batch has added edges or global nodes that this model was not built for')
        if global_mask is not None:
            scorable = graphs.part_mask.to(nodes.dtype)[:, None]
            sums = nodes.new_zeros((graphs.num_graphs, nodes.shape[1])).index_add(0, graphs.batch, nodes * scorable)
            nodes = torch.where(global_mask[:, None], F.normalize(sums, dim=-1).index_select(0, graphs.batch), nodes)
        phrase = self.options['triple_match'] == 'phrase' and not self.options['graph_channel']
        src, dst = graphs.edge_index
        if kind is not None:
            src, dst = src[graphs.labelled_index], dst[graphs.labelled_index]
        if not query or self.options['query_graph_encoder']:
            h = self.p_node(nodes)
            if self.options['concept_gat']:
                h = h + self.concept_input(graphs, self.concept_vectors() if concepts is None else concepts)
            edge_attr, labelled = self.edge_vectors(graphs)
            if self.options['use_gat']:
                edges = graphs.edge_index if self.options['use_edges'] else graphs.edge_index[:, :0]
                attrs = edge_attr if self.options['use_edges'] else edge_attr[:0]
                h = self.dropout(F.elu(self.gat(self.gat1, h, edges, attrs)))
                h = self.dropout(F.elu(self.gat(self.gat2, h, edges, attrs)))
            nodes = F.normalize(nodes + self.u(h), dim=-1)
            if phrase and self.options['use_triple_channel'] and self.options['use_edges']:
                e = edge_attr if labelled is None else edge_attr[labelled]
                g = self.triple_mlp(torch.cat((h[src], e, h[dst]), -1))
                triples = F.normalize(triples + self.u(g), dim=-1)
        if self.options['triple_match'] == 'factorized':
            # Subject and object take the same encoded node vectors as the object channel; the predicate keeps its
            # frozen CLIP relation vector (no learned projection, so it cannot collapse).
            triples = torch.stack((nodes[src], graphs.edge_attr, nodes[dst]), -2)
        node_batch = graphs.batch[graphs.part_mask]
        nodes, node_mask = to_dense_batch(nodes[graphs.part_mask], node_batch, batch_size=graphs.num_graphs)
        if len(triples):
            edge_batch = graphs.batch[src]
            triples, triple_mask = to_dense_batch(triples, edge_batch, batch_size=graphs.num_graphs)
        else:
            triples = nodes.new_zeros((graphs.num_graphs, 1, *triples.shape[1:]))
            triple_mask = node_mask.new_zeros((graphs.num_graphs, 1))
        if self.options['graph_channel'] or not self.options['use_triple_channel'] or not self.options['use_edges']:
            triple_mask = torch.zeros_like(triple_mask)
        # At least one padding position avoids empty softmax reductions.
        if nodes.shape[1] == 0:
            nodes, node_mask = nodes.new_zeros((graphs.num_graphs, 1, 512)), node_mask.new_zeros((graphs.num_graphs, 1))
        if triples.shape[1] == 0:
            triples = triples.new_zeros((graphs.num_graphs, 1, *triples.shape[2:]))
            triple_mask = triple_mask.new_zeros((graphs.num_graphs, 1))
        return nodes, node_mask, triples, triple_mask

    def forward(self, batch, return_encoded=False):
        # The concept GAT runs once per forward and serves both sides.
        concepts = self.concept_vectors() if self.options['concept_gat'] else None
        queries = self.encode(batch['queries'], query=True, concepts=concepts)
        images = self.encode(batch['images'], concepts=concepts)
        return self.match_encoded(batch, queries, images, return_encoded)

    def match_encoded(self, batch, queries, images, return_encoded=False):
        args_obj = (queries[0], queries[1], images[0], images[1], batch['candidate_index'], self.tau_objects(), self.u_objects)
        counts = {}
        if self.options['count_table']:
            counts = {'count_query': dense_counts(batch['queries'], queries[0].shape[1]),
                      'count_images': dense_counts(batch['images'], images[0].shape[1]), 'count_table': self.counts_table()}
            args_obj = args_obj + (counts['count_query'], counts['count_images'], counts['count_table'])
        args_tri = (queries[2], queries[3], images[2], images[3], batch['candidate_index'], self.tau_triples(), self.u_triples)
        score_objects = partial(channel_score, **self.match_options('objects'))
        score_triples = partial(channel_score, **self.match_options('triples'))
        pooled = {}
        if self.options['graph_channel']:
            # The third channel ("triples" in fuse_tensor and details) is the graph-level score here.
            query_graph, pooled['pool_query'] = attention_pool(queries[0], queries[1], self.u_pool_query)
            image_graph, pooled['pool_image'] = attention_pool(images[0], images[1], self.u_pool_image)
            objects = (checkpoint(score_objects, *args_obj, use_reentrant=False) if self.training
                       else score_objects(*args_obj))
            triples = graph_level_score(query_graph, queries[1], image_graph, images[1], batch['candidate_index'])
            pooled.update(graph_query=query_graph, graph_images=image_graph)
        elif self.training:
            objects = checkpoint(score_objects, *args_obj, use_reentrant=False)
            # Factorized slots are already checkpointed one by one inside part_similarity.
            triples = (score_triples(*args_tri) if self.options['triple_match'] == 'factorized'
                       else checkpoint(score_triples, *args_tri, use_reentrant=False))
        else:
            objects, triples = score_objects(*args_obj), score_triples(*args_tri)
        if not self.uses_third_channel():
            triples = torch.full_like(objects, torch.nan)
        weights = F.softmax(F.linear(batch['sentence'], self.weight_map, self.weight_bias), -1)
        if not self.uses_third_channel():
            weights = weights.clone()
            weights[:, 2] = 0.
            weights = weights / weights.sum(-1, keepdim=True)
        if self.options['fusion'] == 'raw':
            fused, details = fuse_raw(batch['clip'], objects, triples, weights)
        else:
            fused, details = fuse_tensor(batch['clip'], objects, triples, weights, self.options['zscore_floor'] or (0., 0., 0.))
        if self.options['scene_terms']:
            if 'terms' not in batch:
                raise ValueError("scene_terms needs batches with 'terms': call GraphStore.use_scene_terms()")
            # sum_t d_t T_t over the fixed terms, already divided by their train scale; part of the graph score, so
            # it enters fused with the graph's weight: 1 in the raw fusion, 1 - a in the z-score fusion.
            terms = (batch['terms'] * self.term_weights).sum(-1)
            graph_weight = 1. if self.options['fusion'] == 'raw' else 1 - weights[:, :1]
            fused = fused + graph_weight * terms
            details.update(graph=details['graph'] + terms, terms=terms)
        details.update(objects=objects, triples=triples, fused=fused, **pooled, **counts)
        if return_encoded:
            details.update(encoded_query=queries, encoded_images=images)
        return fused, details
