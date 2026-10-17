"""Shared PyG encoder and differentiable three-channel graph reranking."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


def zscore_tensor(values, eps=1e-6):
    """Population z-score with masked missing values and finite inactive gradients."""
    known = torch.isfinite(values)
    count = known.sum(-1, keepdim=True)
    clean = torch.where(known, values, 0.)
    mean = clean.sum(-1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(known, clean - mean, 0.)
    var = centered.square().sum(-1, keepdim=True) / count.clamp_min(1)
    active = (count >= 2) & (var >= eps ** 2)
    # sqrt(0) has an infinite derivative even behind a torch.where mask.
    std = var.clamp_min(eps ** 2).sqrt()
    z = torch.where(active, centered / (std + eps), 0.)
    return z, active.squeeze(-1)


def fuse_tensor(clip, objects, triples, weights):
    z_clip, _ = zscore_tensor(clip)
    z_obj, obj_active = zscore_tensor(objects)
    z_tri, tri_active = zscore_tensor(triples)
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


def slot_similarity(cos):
    """Map a per-slot cosine into [0, 1]: smooth everywhere, so no slot loses its gradient (unlike relu)."""
    return (1 + cos) / 2


def part_similarity(query, candidates):
    """Similarity of every query part to every candidate part, shape (..., Tq, Ti).

    query is (B, Tq, D) and candidates (B, K, Ti, D), or factorized triples (B, Tq, 3, D) and (B, K, Ti, 3, D)
    holding subject, predicate and object vectors; a factorized pair scores c(s) * c(p) * c(o), so one
    mismatched slot pulls the whole pair down.
    """
    if query.dim() == 3:
        return torch.matmul(query[:, None], candidates.transpose(-1, -2))
    sims = 1.
    for slot in range(query.shape[-2]):
        sims = sims * slot_similarity(torch.matmul(query[:, None, :, slot], candidates[..., slot, :].transpose(-1, -2)))
    return sims


def part_summary(query):
    """One vector per query part (B, Tq, D) for the weights w: factorized triples use the mean of their slots."""
    return query if query.dim() == 3 else query.mean(-2)


def channel_score(query, query_mask, images, image_mask, candidate_index, tau, u):
    """Match all B x K pairs together; no loop or materialized feature cache per candidate."""
    candidates = images[candidate_index]
    masks = image_mask[candidate_index]
    sims = part_similarity(query, candidates)
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


TRIPLE_MATCHES = ("phrase", "factorized")


class GraphReranker(nn.Module):
    """Three channels: CLIP, objects, and a third graph channel.

    The third channel is the triple channel by default. triple_match="phrase" matches one vector per triple,
    normalize(CLIP phrase + U triple_mlp([h_s; e; h_o])); "factorized" matches (subject node, CLIP relation,
    object node) slot by slot. graph_channel=True replaces the triple channel by the cosine of attention-pooled
    query and image graph vectors.
    """

    def __init__(self, adaptive_weights=True, use_gat=True, use_triple_channel=True,
                 use_edges=True, query_graph_encoder=True, dropout=0.1, triple_match="phrase", graph_channel=False):
        super().__init__()
        if triple_match not in TRIPLE_MATCHES:
            raise ValueError(f"triple_match must be one of {TRIPLE_MATCHES}")
        if graph_channel and triple_match != "phrase":
            raise ValueError("graph_channel replaces the triple channel; it takes no triple_match")
        self.options = dict(adaptive_weights=adaptive_weights, use_gat=use_gat,
                            use_triple_channel=use_triple_channel, use_edges=use_edges,
                            query_graph_encoder=query_graph_encoder, dropout=dropout,
                            triple_match=triple_match, graph_channel=graph_channel)
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
        if graph_channel:
            # Zero init: both sides start from mean pooling of their node vectors.
            self.u_pool_query = nn.Parameter(torch.zeros(512))
            self.u_pool_image = nn.Parameter(torch.zeros(512))

    def tau_objects(self):
        return self.log_tau_objects.exp().clamp(.01, 1.)

    def tau_triples(self):
        return self.log_tau_triples.exp().clamp(.01, 1.)

    def loss_temperature(self):
        return self.log_temperature.exp().clamp(.05, 5.)

    def set_epoch(self, epoch):
        self.weight_map.requires_grad_(epoch > 1 and self.options['adaptive_weights'])
        self.weight_bias.requires_grad_(epoch > 1)

    def uses_third_channel(self):
        return self.options['use_triple_channel'] and (self.options['use_edges'] or self.options['graph_channel'])

    def encode(self, graphs: Batch, query=False):
        nodes, triples = graphs.x, graphs.triple_f
        phrase = self.options['triple_match'] == 'phrase' and not self.options['graph_channel']
        if not query or self.options['query_graph_encoder']:
            h = self.p_node(nodes)
            edge_attr = self.p_edge(graphs.edge_attr)
            if self.options['use_gat']:
                edges = graphs.edge_index if self.options['use_edges'] else graphs.edge_index[:, :0]
                attrs = edge_attr if self.options['use_edges'] else edge_attr[:0]
                h = self.dropout(F.elu(self.gat1(h, edges, attrs)))
                h = self.dropout(F.elu(self.gat2(h, edges, attrs)))
            nodes = F.normalize(nodes + self.u(h), dim=-1)
            if phrase and self.options['use_triple_channel'] and self.options['use_edges']:
                src, dst = graphs.edge_index
                g = self.triple_mlp(torch.cat((h[src], edge_attr, h[dst]), -1))
                triples = F.normalize(triples + self.u(g), dim=-1)
        if self.options['triple_match'] == 'factorized':
            # Subject and object take the same encoded node vectors as the object channel; the predicate keeps its
            # frozen CLIP relation vector (no learned projection, so it cannot collapse).
            src, dst = graphs.edge_index
            triples = torch.stack((nodes[src], graphs.edge_attr, nodes[dst]), -2)
        node_batch = graphs.batch[graphs.part_mask]
        nodes, node_mask = to_dense_batch(nodes[graphs.part_mask], node_batch, batch_size=graphs.num_graphs)
        if len(triples):
            edge_batch = graphs.batch[graphs.edge_index[0]]
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
        queries = self.encode(batch['queries'], query=True)
        images = self.encode(batch['images'])
        return self.match_encoded(batch, queries, images, return_encoded)

    def match_encoded(self, batch, queries, images, return_encoded=False):
        args_obj = (queries[0], queries[1], images[0], images[1], batch['candidate_index'], self.tau_objects(), self.u_objects)
        args_tri = (queries[2], queries[3], images[2], images[3], batch['candidate_index'], self.tau_triples(), self.u_triples)
        pooled = {}
        if self.options['graph_channel']:
            # The third channel ("triples" in fuse_tensor and details) is the graph-level score here.
            query_graph, pooled['pool_query'] = attention_pool(queries[0], queries[1], self.u_pool_query)
            image_graph, pooled['pool_image'] = attention_pool(images[0], images[1], self.u_pool_image)
            objects = checkpoint(channel_score, *args_obj, use_reentrant=False) if self.training else channel_score(*args_obj)
            triples = graph_level_score(query_graph, queries[1], image_graph, images[1], batch['candidate_index'])
            pooled.update(graph_query=query_graph, graph_images=image_graph)
        elif self.training:
            objects = checkpoint(channel_score, *args_obj, use_reentrant=False)
            triples = checkpoint(channel_score, *args_tri, use_reentrant=False)
        else:
            objects, triples = channel_score(*args_obj), channel_score(*args_tri)
        if not self.uses_third_channel():
            triples = torch.full_like(objects, torch.nan)
        weights = F.softmax(F.linear(batch['sentence'], self.weight_map, self.weight_bias), -1)
        if not self.uses_third_channel():
            weights = weights.clone()
            weights[:, 2] = 0.
            weights = weights / weights.sum(-1, keepdim=True)
        fused, details = fuse_tensor(batch['clip'], objects, triples, weights)
        details.update(objects=objects, triples=triples, fused=fused, **pooled)
        if return_encoded:
            details.update(encoded_query=queries, encoded_images=images)
        return fused, details
