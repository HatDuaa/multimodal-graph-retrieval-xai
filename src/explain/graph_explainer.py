"""Explain and ablate an image inside the same query's full candidate context."""
import copy

import torch

from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS
from src.models.graph_reranker import masked_softmax, part_similarity, part_summary, slot_similarity


def remove_parts(graph, object_indices=(), relation_indices=()):
    """Remove indexed instances and incident relations; preserve original object IDs."""
    out = copy.deepcopy(graph)
    object_indices, relation_indices = set(object_indices), set(relation_indices)
    if not object_indices <= set(range(len(graph['objects']))) or not relation_indices <= set(range(len(graph['relations']))):
        raise IndexError('Removed part index is outside the original graph')
    removed_ids = {obj['id'] for i, obj in enumerate(graph['objects']) if i in object_indices}
    out['objects'] = [obj for i, obj in enumerate(graph['objects']) if i not in object_indices]
    out['relations'] = [rel for i, rel in enumerate(graph['relations']) if i not in relation_indices
                        and rel['subject'] not in removed_ids and rel['object'] not in removed_ids]
    return out


def matched_pairs(model, detail, query_parts, image_parts, image_row, query_row=0):
    """Per channel, each query part's most similar image part with the quantities the ranking used.

    detail must come from model(batch, return_encoded=True); image_row indexes the encoded images of that batch.
    contribution = channel weight (b or c) * w_i * a_ij * sim_ij, before the candidate-wise z-score. For count-table
    models, object rows also carry query_count, image_count and count_bias = g[query_count, image_count], which sim_ij
    includes. row_score = sum_j a_ij * sim_ij over every image part, and channel_term = channel weight * w_i *
    row_score: the channel score is sum_i w_i * row_score_i, so under the raw fusion the channel terms of one image
    add up exactly to its part of the final score.
    """
    weights = detail['weights'][query_row]
    result = {'objects': [], 'triples': []}
    for channel, offset, tau, u, weight in (('objects', 0, model.tau_objects(), model.u_objects, weights[1]),
                                           ('triples', 2, model.tau_triples(), model.u_triples, weights[2])):
        qs, qm = detail['encoded_query'][offset:offset + 2]
        ps, pm = detail['encoded_images'][offset:offset + 2]
        q, p = qs[query_row, qm[query_row]], ps[image_row, pm[image_row]]
        if len(q) == 0 or len(p) == 0:
            continue
        # Same similarity and weights as channel_score; factorized triples carry (subject, predicate, object) slots.
        sims = part_similarity(q[None], p[None], torch.zeros((1, 1), dtype=torch.long, device=p.device))[0, 0]
        counts = None
        if channel == 'objects' and 'count_table' in detail:
            # Count-table models: sim_ij + g[count class of query node i, count bucket of image node j].
            q_class = detail['count_query'][query_row, qm[query_row]]
            p_bucket = detail['count_images'][image_row, pm[image_row]]
            bias = detail['count_table'][q_class[:, None], p_bucket[None, :]]
            counts = (q_class, p_bucket, bias)
            sims = sims + bias
        attention = torch.softmax(sims / tau, -1)
        w = torch.softmax(part_summary(q[None])[0] @ u, -1)
        best = sims.argmax(-1)
        row_scores = (attention * sims).sum(-1)
        slots = slot_similarity(torch.einsum('isd,jsd->ijs', q, p)) if q.dim() == 3 else None
        # Encoded query objects keep only scorable parts (part_mask), so the index list must be masked the same way.
        q_indices = ([i for i, keep in zip(query_parts['node_indices'], query_parts['part_mask']) if keep]
                     if channel == 'objects' else query_parts['relation_indices'])
        p_indices = image_parts['node_indices'] if channel == 'objects' else image_parts['relation_indices']
        for i, j in enumerate(best.tolist()):
            row = {'query_index': q_indices[i], 'image_index': p_indices[j],
                   'w_i': float(w[i]), 'a_ij': float(attention[i, j]),
                   'sim_ij': float(sims[i, j]),
                   'contribution': float(weight * w[i] * attention[i, j] * sims[i, j]),
                   'row_score': float(row_scores[i]), 'channel_term': float(weight * w[i] * row_scores[i])}
            if slots is not None:
                row['slot_similarity'] = dict(zip(('subject', 'predicate', 'object'), slots[i, j].tolist()))
            if counts is not None:
                # e.g. caption "2 dog", image "2 dog": count_bias = g["2", "2"]; sim_ij already includes it.
                row.update(query_count=CAPTION_CLASSES[int(counts[0][i])], image_count=IMAGE_BUCKETS[int(counts[1][j])],
                           count_bias=float(counts[2][i, j]))
            result[channel].append(row)
    if 'pool_query' in detail:
        # Graph-level channel: pooled-vector cosine, with the pooling weight of every scorable node on both sides.
        qm, pm = detail['encoded_query'][1][query_row], detail['encoded_images'][1][image_row]
        q_indices = [i for i, keep in zip(query_parts['node_indices'], query_parts['part_mask']) if keep]
        cosine = float(detail['graph_query'][query_row] @ detail['graph_images'][image_row])
        result['graph_level'] = {'cosine': cosine, 'contribution': float(weights[2] * cosine),
                           'query_nodes': [{'query_index': i, 'alpha': float(a)}
                                           for i, a in zip(q_indices, detail['pool_query'][query_row][qm])],
                           'image_nodes': [{'image_index': i, 'alpha': float(a)}
                                           for i, a in zip(image_parts['node_indices'], detail['pool_image'][image_row][pm])]}
    return result


@torch.no_grad()
def score(model, store, query, image_id, overrides=None):
    """Query is a packaged caption ID; scores retain the complete top-50 z-score context."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        batch = store.batch([query], device, overrides)
        ids = batch['candidate_ids'][0].tolist()
        if image_id not in ids:
            raise ValueError('Image must belong to the query candidate pool')
        rank = ids.index(image_id)
        image_row = int(batch['candidate_index'][0, rank])
        fused, detail = model(batch, return_encoded=True)
        raw_image = (overrides or {}).get(image_id, store.graphs[image_id])
        image_parts = store._graph_parts(raw_image)
        query_parts = store._graph_parts(store.queries[query], query=True)
        weights = detail['weights'][0]
        result = {'score': float(fused[0, rank]), 'graph': float(detail['graph'][0, rank]),
                  'weights': weights.cpu().tolist(), 'objects': [], 'triples': [],
                  'contribution_definition': 'channel_weight * w_i * a_ij * sim_ij before candidate-wise z-score; '
                                             'these terms do not sum to the final normalized score.'}
        result.update(matched_pairs(model, detail, query_parts, image_parts, image_row))
        return result
    finally:
        model.train(was_training)


def score_without(model, store, query, image_id, removed_object_indices=(), removed_relation_indices=()):
    modified = remove_parts(store.graphs[image_id], removed_object_indices, removed_relation_indices)
    return score(model, store, query, image_id, {image_id: modified})
