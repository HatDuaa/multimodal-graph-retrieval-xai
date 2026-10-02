"""Explain and ablate an image inside the same query's full candidate context."""
import copy

import torch

from src.models.graph_reranker import masked_softmax


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
    contribution = channel weight (b or c) * w_i * a_ij * sim_ij, before the candidate-wise z-score.
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
        sims = q @ p.T
        attention = torch.softmax(sims / tau, -1)
        w = torch.softmax(q @ u, -1)
        best = sims.argmax(-1)
        # Encoded query objects keep only scorable parts (part_mask), so the index list must be masked the same way.
        q_indices = ([i for i, keep in zip(query_parts['node_indices'], query_parts['part_mask']) if keep]
                     if channel == 'objects' else query_parts['relation_indices'])
        p_indices = image_parts['node_indices'] if channel == 'objects' else image_parts['relation_indices']
        for i, j in enumerate(best.tolist()):
            result[channel].append({'query_index': q_indices[i], 'image_index': p_indices[j],
                                    'w_i': float(w[i]), 'a_ij': float(attention[i, j]),
                                    'sim_ij': float(sims[i, j]),
                                    'contribution': float(weight * w[i] * attention[i, j] * sims[i, j])})
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
