"""Relational GAT over the commonsense concept graph (src/graph/concepts.py), for GraphReranker(concept_graph=...).

Every concept starts from the frozen CLIP text vector of its name. Two attention layers pass messages along typed arcs
(WordNet hypernym, ConceptNet, Visual Genome co-occurrence on train), each arc carrying a fixed weight from the
graph builder. The log of that weight is added to the attention logit after the LeakyReLU, so when the learned logits
are equal the attention reproduces the normalized graph weights; the relation type shifts the logit, not the message.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def arcs(graph):
    """(source, target, relation index, weight) of every traversal direction, plus one self-loop per concept.

    A symmetric relation keeps its type both ways; a directed one gets a separate "<relation>_reverse" type
    backwards. Returns the arrays and the arc relation names.
    """
    names = list(graph["relation_names"])
    arc_names = names + [f"{name}_reverse" for name in names] + ["self_loop"]
    source, target, relation = graph["source"], graph["target"], graph["relation"]
    reverse = np.where(graph["directed"], relation + len(names), relation)
    count = len(graph["features"])
    loops = np.arange(count)
    return (np.concatenate((source, target, loops)), np.concatenate((target, source, loops)),
            np.concatenate((relation, reverse, np.full(count, len(arc_names) - 1))),
            np.concatenate((graph["weight"], graph["weight"], np.ones(count))), arc_names)


class RelationalAttention(nn.Module):
    """One multi-head additive attention layer with relation embeddings and a fixed log-weight prior."""

    def __init__(self, dimension, channels, heads, relations, relation_dimension=16):
        super().__init__()
        self.heads, self.channels = heads, channels
        self.project = nn.Linear(dimension, heads * channels, bias=False)
        self.relation = nn.Embedding(relations, relation_dimension)
        self.relation_logit = nn.Linear(relation_dimension, heads, bias=False)
        self.attend_source = nn.Parameter(torch.empty(heads, channels))
        self.attend_target = nn.Parameter(torch.empty(heads, channels))
        self.bias = nn.Parameter(torch.zeros(heads * channels))
        nn.init.xavier_uniform_(self.attend_source)
        nn.init.xavier_uniform_(self.attend_target)

    def forward(self, x, source, target, relation, log_prior):
        count = len(x)
        z = self.project(x).view(count, self.heads, self.channels)
        logits = F.leaky_relu((z * self.attend_source).sum(-1)[source] + (z * self.attend_target).sum(-1)[target]
                              + self.relation_logit(self.relation(relation)), .2) + log_prior[:, None]
        # Softmax over the incoming arcs of every target, per head.
        index = target[:, None].expand_as(logits)
        maxima = logits.new_full((count, self.heads), -math.inf).scatter_reduce(0, index, logits.detach(), 'amax')
        weights = (logits - maxima[target]).exp()
        attention = weights / weights.new_zeros((count, self.heads)).index_add(0, target, weights)[target]
        out = z.new_zeros((count, self.heads, self.channels)).index_add(0, target, z[source] * attention[..., None])
        return out.reshape(count, -1) + self.bias


class ConceptGAT(nn.Module):
    """Two relational attention layers (4 heads, then 1) over the whole concept graph; one vector per concept."""

    def __init__(self, graph, hidden=256, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        source, target, relation, weight, self.arc_names = arcs(graph)
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
            raise ValueError("concept graph weights must be finite and strictly positive")
        as_long = lambda values: torch.as_tensor(np.asarray(values, dtype=np.int64))  # noqa: E731
        # Fixed data, rebuilt from the data version at load time: not part of the checkpoint.
        self.register_buffer("features", torch.as_tensor(np.asarray(graph["features"], dtype=np.float32)), False)
        self.register_buffer("source", as_long(source), False)
        self.register_buffer("target", as_long(target), False)
        self.register_buffer("relation", as_long(relation), False)
        self.register_buffer("log_prior", torch.as_tensor(np.log(weight), dtype=torch.float32), False)
        self.first = RelationalAttention(self.features.shape[1], hidden // heads, heads, len(self.arc_names))
        self.second = RelationalAttention(hidden, hidden, 1, len(self.arc_names))

    def forward(self):
        arcs_ = (self.source, self.target, self.relation, self.log_prior)
        return self.second(F.gelu(self.first(self.features, *arcs_)), *arcs_)
