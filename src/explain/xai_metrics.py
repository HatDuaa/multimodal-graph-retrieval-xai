"""Level-3 explanation metrics: edit one image graph part by part and rescore it among its query's candidates.

An explanation E(q, image) ranks the image's parts, its object nodes and its labelled triples, by their exact term
term_j = channel weight * sum_i w_i a_ij sim_ij (matched_pairs image_terms). Under the raw fusion these terms add up
to the image's graph part; for the z-score fusion of level 1 they are the same raw terms, before the candidate-wise
z-score.

Edits act on the image graph as the ranking packs it (a one-graph GraphIndex):
- dropping a node drops every edge touching it (labelled, completion [none], global); completion edges are not
  rebuilt for the remaining nodes;
- dropping a triple drops exactly its labelled edge;
- an unscored node stays in the graph as a carrier (its edges and GAT messages remain) but leaves the object channel
  (part_mask False): the end nodes of a kept triple when only some parts are kept.
- a message edit (message_drop) keeps every node and triple but stops some edges from carrying GAT messages, so an
  edge's role in message passing is measured apart from its triple.
The edited image is encoded again by the whole model, GAT included. The other 49 candidates keep their channel
scores and the fusion is redone over the 50, so for them only fuse_raw's filled mean (or the z-scores of level 1) can
move.

Terms are put on the final score's scale by fusion_scales before parts of the two channels are compared: 1 under the
raw fusion (exact), and the local slope of the z-score fusion otherwise (level 1), where a channel's raw term counts
in proportion to channel weight / its standard deviation over the 50 candidates.
"""
import numpy as np
import torch

from src.data.graph_store import EDGE_LABELLED, GraphIndex
from src.explain.graph_explainer import matched_pairs
from src.models.graph_reranker import (count_bias, fuse_raw, fuse_tensor, masked_softmax, part_similarity,
                                       part_summary)


def image_index(store, graph):
    """The one-graph GraphIndex of an image graph with the store's structure (completion edges, global node)."""
    return GraphIndex.build([store.graph_parts(graph)], store.part_row)


def labelled_count(index):
    return int((index.fields['edge_kind'] == EDGE_LABELLED).sum())


def real_nodes(index):
    """Node rows of the image's objects (the global node, if any, is last and excluded)."""
    return int((~index.fields['global_mask']).sum())


def edit(index, drop_nodes=(), drop_edges=(), unscored=()):
    """A copy of a one-graph index without some nodes (and every edge touching them) and edges.

    Node and edge positions are those of the unedited index (labelled edges come first, so triple j is edge j).
    unscored nodes stay as carriers but leave the object channel.
    """
    n, m = int(index.node_counts[0]), int(index.edge_counts[0])
    keep_nodes = np.ones(n, dtype=bool)
    keep_nodes[list(drop_nodes)] = False
    src, dst = index.fields['edge_src'], index.fields['edge_dst']
    keep_edges = keep_nodes[src] & keep_nodes[dst]
    keep_edges[list(drop_edges)] = False
    new_id = np.cumsum(keep_nodes) - 1
    fields = {k: index.fields[k][keep_nodes] for k in GraphIndex.NODE_FIELDS}
    if len(unscored):
        mask = index.fields['part_mask'].copy()
        mask[list(unscored)] = False
        fields['part_mask'] = mask[keep_nodes]
    fields.update({k: index.fields[k][keep_edges] for k in GraphIndex.EDGE_FIELDS})
    fields['edge_src'], fields['edge_dst'] = new_id[src[keep_edges]], new_id[dst[keep_edges]]
    assert m == len(src)
    return GraphIndex([int(keep_nodes.sum())], [int(keep_edges.sum())], has_concepts=index.has_concepts, **fields)


def keep_only(index, nodes=(), triples=()):
    """Edit keeping only these object nodes and labelled triples (and the global node); a kept triple keeps its end
    nodes as unscored carriers."""
    src, dst = index.fields['edge_src'], index.fields['edge_dst']
    carriers = {int(v) for t in triples for v in (src[t], dst[t])} - set(nodes)
    real = real_nodes(index)
    drop = [v for v in range(real) if v not in set(nodes) | carriers]
    labelled = labelled_count(index)
    return edit(index, drop_nodes=drop, drop_edges=[e for e in range(labelled) if e not in set(triples)],
                unscored=sorted(carriers))


def message_drop(index, edges):
    """Message mask of a one-graph index: False on the given edge positions, which then carry no GAT message."""
    mask = np.ones(int(index.edge_counts[0]), dtype=bool)
    mask[list(edges)] = False
    return mask


def _spread(values, floor=0., eps=1e-6):
    """std + eps of the known values as zscore_tensor divides by it, and whether zscore_tensor treats them as active."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    var = float(np.mean((values - values.mean()) ** 2))
    if var < eps ** 2:
        return None
    return max(np.sqrt(max(var, eps ** 2)), floor) + eps


def fuse(model, clip, objects, triples, weights):
    if model.options['fusion'] == 'raw':
        return fuse_raw(clip, objects, triples, weights)[0]
    return fuse_tensor(clip, objects, triples, weights, model.options['zscore_floor'] or (0., 0., 0.))[0]


def rank_of(fused, column):
    """1-based rank of column per row, as a stable descending sort (ties go to the lower candidate index)."""
    target = fused[:, column:column + 1]
    index = torch.arange(fused.shape[1], device=fused.device)
    ahead = (fused > target) | ((fused == target) & (index < column))
    return 1 + ahead.sum(1)


class QueryContext:
    """One val caption scored over its 50 candidates, ready to rescore edits of one candidate's graph."""

    @torch.no_grad()
    def __init__(self, model, store, caption_id, device):
        self.model, self.store, self.caption_id, self.device = model, store, str(caption_id), device
        self.batch = store.batch([self.caption_id], device)
        self.fused, self.detail = model(self.batch, return_encoded=True)
        self.fused = self.fused[0]
        self.ids = [int(i) for i in self.batch['candidate_ids'][0]]
        self.top = int(torch.argsort(self.fused, descending=True, stable=True)[0])
        self.gold = int(self.batch['gold'][0])

    def column_of(self, image_id):
        return self.ids.index(int(image_id))

    def explanation(self, column):
        """matched_pairs of the candidate at column: rows per query part and the exact term of every image part."""
        query_parts = self.store.graph_parts(self.store.queries[self.caption_id], query=True)
        image_parts = self.store.graph_parts(self.store.graphs[self.ids[column]])
        row = int(self.batch['candidate_index'][0, column])
        with torch.no_grad():
            return matched_pairs(self.model, self.detail, query_parts, image_parts, row)

    def fusion_scales(self):
        """{channel: factor} that puts an image part's raw term on the scale of the final score.

        Raw fusion: 1, the terms already add up to the final score. z-score fusion: the local slope of fuse_tensor,
        d final / d term = (1 - a) / ((b + c) * s_channel * s_mix) when both channels are active, (1 - a) /
        (channel weight * s_channel) when only this one is, 0 when it is inactive; s is the spread zscore_tensor divides
        by over the 50 candidates (a candidate's own effect on the mean is ignored). The common factor does not change
        any order inside one caption; the channel ratio does.
        """
        if self.model.options['fusion'] == 'raw':
            return {'objects': 1., 'triples': 1.}
        detail = self.detail
        a, b, c = detail['weights'][0].tolist()
        floors = self.model.options['zscore_floor'] or (0., 0., 0.)
        spread = {'objects': _spread(detail['objects'][0].cpu().numpy(), floors[1]),
                  'triples': _spread(detail['triples'][0].cpu().numpy(), floors[2])}
        weight = {'objects': b, 'triples': c}
        if spread['objects'] and spread['triples']:
            mix = (b * detail['z_objects'][0] + c * detail['z_triples'][0]) / (b + c)
            s_mix = _spread(mix.cpu().numpy())
            return {k: (1 - a) / ((b + c) * spread[k] * s_mix) if s_mix else 0. for k in spread}
        return {k: (1 - a) / (weight[k] * spread[k]) if spread[k] and weight[k] else 0. for k in spread}

    def gradient_terms(self, column, index):
        """Gradient x input of the final score of the candidate at column, per object node and labelled triple.

        The inputs are the frozen CLIP vectors of the one-graph index: a node's label vector, and a triple's phrase and
        predicate vectors. The other 49 candidates are fixed as in rescore.
        """
        model, table = self.model, self.store.part_table(self.device)
        images = index.to_batch(table, self.device)
        images.x = images.x.detach().clone().requires_grad_(True)
        images.triple_f = images.triple_f.detach().clone().requires_grad_(True)
        images.edge_attr = images.edge_attr.detach().clone().requires_grad_(True)
        batch = {'queries': self.batch['queries'], 'images': images,
                 'candidate_index': torch.zeros((1, 1), dtype=torch.long, device=self.device),
                 'clip': torch.zeros((1, 1), device=self.device), 'sentence': self.batch['sentence']}
        with torch.enable_grad():
            _, detail = model(batch)
            objects, triples = self.detail['objects'].clone(), self.detail['triples'].clone()
            objects[:, column], triples[:, column] = detail['objects'][0], detail['triples'][0]
            fused = fuse(model, self.batch['clip'], objects, triples, self.detail['weights'])
            fused[0, column].backward()
        node = (images.x.grad * images.x).sum(-1).detach().cpu().numpy()[:real_nodes(index)]
        triple = ((images.triple_f.grad * images.triple_f).sum(-1)
                  + (images.edge_attr.grad * images.edge_attr).sum(-1)).detach().cpu().numpy()[:labelled_count(index)]
        return node, triple

    @torch.no_grad()
    def rescore(self, column, indexes, channels=False, message_masks=None):
        """Final score and rank of the candidate at column for each edited graph in indexes (with channels, also its
        raw object and triple channel scores, NaN for an emptied channel). message_masks: one message_drop mask (or
        None) per index."""
        model, table = self.model, self.store.part_table(self.device)
        images = GraphIndex.concat(indexes).to_batch(table, self.device)
        if message_masks is not None and any(m is not None for m in message_masks):
            masks = [np.ones(int(i.edge_counts[0]), dtype=bool) if m is None else m for i, m in zip(indexes, message_masks)]
            images.message_mask = torch.from_numpy(np.concatenate(masks)).to(self.device)
        count = len(indexes)
        batch = {'queries': self.batch['queries'], 'images': images,
                 'candidate_index': torch.arange(count, device=self.device)[None],
                 'clip': torch.zeros((1, count), device=self.device), 'sentence': self.batch['sentence']}
        _, detail = model(batch)
        objects = self.detail['objects'].repeat(count, 1)
        triples = self.detail['triples'].repeat(count, 1)
        objects[:, column], triples[:, column] = detail['objects'][0], detail['triples'][0]
        fused = fuse(model, self.batch['clip'].repeat(count, 1), objects, triples, self.detail['weights'].repeat(count, 1))
        out = fused[:, column].cpu().numpy(), rank_of(fused, column).cpu().numpy()
        if channels:
            out += (detail['objects'][0].cpu().numpy(), detail['triples'][0].cpu().numpy())
        return out

    @torch.no_grad()
    def rescore_full(self, column, index):
        """Final scores of all 50 candidates when the candidate at column takes one edited graph."""
        model, table = self.model, self.store.part_table(self.device)
        images = index.to_batch(table, self.device)
        batch = {'queries': self.batch['queries'], 'images': images,
                 'candidate_index': torch.zeros((1, 1), dtype=torch.long, device=self.device),
                 'clip': torch.zeros((1, 1), device=self.device), 'sentence': self.batch['sentence']}
        _, detail = model(batch)
        objects, triples = self.detail['objects'].clone(), self.detail['triples'].clone()
        objects[:, column], triples[:, column] = detail['objects'][0], detail['triples'][0]
        return fuse(model, self.batch['clip'], objects, triples, self.detail['weights'])[0]

    @torch.no_grad()
    def row_terms(self):
        """Per channel, (50, n) array of channel weight * w_i * sum_j a_ij sim_ij for every candidate and scorable
        query part, NaN for a candidate without parts in that channel (channel_score, softmax matching)."""
        model, detail, cand = self.model, self.detail, self.batch['candidate_index']
        weights = detail['weights'][0]
        q, qm, qt, qtm = detail['encoded_query']
        p, pm, t, tm = detail['encoded_images']
        out = {}
        for channel, query, query_mask, images, image_mask, tau, u, weight in (
                ('objects', q, qm, p, pm, model.tau_objects(), model.u_objects, weights[1]),
                ('triples', qt, qtm, t, tm, model.tau_triples(), model.u_triples, weights[2])):
            masks = image_mask[cand]
            sims = part_similarity(query, images, cand)
            if channel == 'objects' and 'count_table' in detail:
                sims = sims + count_bias(detail['count_table'], detail['count_query'], detail['count_images'], cand)
            attention = masked_softmax(sims / tau, masks[:, :, None, :], -1)
            per_part = (attention * sims).sum(-1)[0]
            w = masked_softmax(part_summary(query) @ u, query_mask, -1)[0]
            rows = (weight * w[None, :] * per_part)[:, query_mask[0]]
            if not model.uses_third_channel() and channel == 'triples':
                rows = rows[:, :0]
            rows = torch.where(masks[0].any(-1)[:, None], rows, torch.nan)
            out[channel] = rows.double().cpu().numpy()
        return out

    @torch.no_grad()
    def attention(self, index):
        """Per edge of a one-graph index: GAT alpha averaged over the two layers and the heads.

        GATv2Conv drops the graph's own self-loops (a relation whose subject and object are one node) and appends one
        self-loop per node after the other edges, which keep their order; an edge that is itself a self-loop gets NaN.
        """
        model = self.model
        batch = index.to_batch(self.store.part_table(self.device), self.device)
        model.attention, model.record_attention = [], True
        try:
            model.encode(batch)
        finally:
            model.record_attention = False
        edges = batch.edge_index
        proper = edges[0] != edges[1]
        n_proper = int(proper.sum())
        alphas = []
        for pairs, alpha in model.attention:
            assert torch.equal(pairs[:, :n_proper], edges[:, proper])
            alphas.append(alpha[:n_proper].mean(-1))
        model.attention = []
        out = np.full(edges.shape[1], np.nan)
        out[proper.cpu().numpy()] = torch.stack(alphas).mean(0).cpu().numpy() if alphas else 0.
        return out
