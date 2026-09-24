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


def channel_score(query, query_mask, images, image_mask, candidate_index, tau, u):
    """Match all B x K pairs together; no loop or materialized feature cache per candidate."""
    candidates = images[candidate_index]
    masks = image_mask[candidate_index]
    sims = torch.matmul(query[:, None], candidates.transpose(-1, -2))
    attention = masked_softmax(sims / tau, masks[:, :, None, :], -1)
    per_part = (attention * sims).sum(-1)
    weights = masked_softmax(query @ u, query_mask, -1)
    scores = (weights[:, None, :] * per_part).sum(-1)
    valid = query_mask.any(-1)[:, None] & masks.any(-1)
    return scores.masked_fill(~valid, torch.nan)


class GraphReranker(nn.Module):
    def __init__(self, adaptive_weights=True, use_gat=True, use_triple_channel=True,
                 use_edges=True, query_graph_encoder=True, dropout=0.1):
        super().__init__()
        self.options = dict(adaptive_weights=adaptive_weights, use_gat=use_gat,
                            use_triple_channel=use_triple_channel, use_edges=use_edges,
                            query_graph_encoder=query_graph_encoder, dropout=dropout)
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

    def tau_objects(self):
        return self.log_tau_objects.exp().clamp(.01, 1.)

    def tau_triples(self):
        return self.log_tau_triples.exp().clamp(.01, 1.)

    def loss_temperature(self):
        return self.log_temperature.exp().clamp(.05, 5.)

    def set_epoch(self, epoch):
        self.weight_map.requires_grad_(epoch > 1 and self.options['adaptive_weights'])
        self.weight_bias.requires_grad_(epoch > 1)

    def encode(self, graphs: Batch, query=False):
        nodes, triples = graphs.x, graphs.triple_f
        if not query or self.options['query_graph_encoder']:
            h = self.p_node(nodes)
            edge_attr = self.p_edge(graphs.edge_attr)
            if self.options['use_gat']:
                edges = graphs.edge_index if self.options['use_edges'] else graphs.edge_index[:, :0]
                attrs = edge_attr if self.options['use_edges'] else edge_attr[:0]
                h = self.dropout(F.elu(self.gat1(h, edges, attrs)))
                h = self.dropout(F.elu(self.gat2(h, edges, attrs)))
            nodes = F.normalize(nodes + self.u(h), dim=-1)
            if self.options['use_triple_channel'] and self.options['use_edges']:
                src, dst = graphs.edge_index
                g = self.triple_mlp(torch.cat((h[src], edge_attr, h[dst]), -1))
                triples = F.normalize(triples + self.u(g), dim=-1)
        node_batch = graphs.batch[graphs.part_mask]
        nodes, node_mask = to_dense_batch(nodes[graphs.part_mask], node_batch, batch_size=graphs.num_graphs)
        if len(triples):
            edge_batch = graphs.batch[graphs.edge_index[0]]
            triples, triple_mask = to_dense_batch(triples, edge_batch, batch_size=graphs.num_graphs)
        else:
            triples = nodes.new_zeros((graphs.num_graphs, 1, 512))
            triple_mask = node_mask.new_zeros((graphs.num_graphs, 1))
        if not self.options['use_triple_channel'] or not self.options['use_edges']:
            triple_mask = torch.zeros_like(triple_mask)
        # At least one padding position avoids empty softmax reductions.
        if nodes.shape[1] == 0:
            nodes, node_mask = nodes.new_zeros((graphs.num_graphs, 1, 512)), node_mask.new_zeros((graphs.num_graphs, 1))
        if triples.shape[1] == 0:
            triples, triple_mask = triples.new_zeros((graphs.num_graphs, 1, 512)), triple_mask.new_zeros((graphs.num_graphs, 1))
        return nodes, node_mask, triples, triple_mask

    def forward(self, batch, return_encoded=False):
        queries = self.encode(batch['queries'], query=True)
        images = self.encode(batch['images'])
        return self.match_encoded(batch, queries, images, return_encoded)

    def match_encoded(self, batch, queries, images, return_encoded=False):
        args_obj = (queries[0], queries[1], images[0], images[1], batch['candidate_index'], self.tau_objects(), self.u_objects)
        args_tri = (queries[2], queries[3], images[2], images[3], batch['candidate_index'], self.tau_triples(), self.u_triples)
        if self.training:
            objects = checkpoint(channel_score, *args_obj, use_reentrant=False)
            triples = checkpoint(channel_score, *args_tri, use_reentrant=False)
        else:
            objects, triples = channel_score(*args_obj), channel_score(*args_tri)
        weights = F.softmax(F.linear(batch['sentence'], self.weight_map, self.weight_bias), -1)
        if not self.options['use_triple_channel'] or not self.options['use_edges']:
            weights = weights.clone()
            weights[:, 2] = 0.
            weights = weights / weights.sum(-1, keepdim=True)
        fused, details = fuse_tensor(batch['clip'], objects, triples, weights)
        details.update(objects=objects, triples=triples, fused=fused)
        if return_encoded:
            details.update(encoded_query=queries, encoded_images=images)
        return fused, details
