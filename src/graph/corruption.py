"""Graph-corruption controls for the reranker: another image's graph, and degree-preserving edge rewiring."""
import copy

import numpy as np


def graph_derangement(image_ids, seed):
    """Map every image to a different image of the same pool: shuffle, then shift by one (no fixed points)."""
    ids = sorted(int(i) for i in image_ids)
    if len(ids) < 2:
        raise ValueError("a derangement needs at least two images")
    order = np.random.default_rng(np.random.SeedSequence([seed, 7919])).permutation(len(ids))
    shuffled = [ids[k] for k in order]
    return {shuffled[k]: shuffled[(k + 1) % len(shuffled)] for k in range(len(shuffled))}


def rewire_graph(graph, rng):
    """Permute relation targets within one graph.

    Every relation keeps its subject and its predicate label and receives the object of another relation of the
    same graph, so each node keeps both its out-degree and its in-degree and the number of edges is unchanged.
    """
    out = copy.deepcopy(graph)
    relations = out['relations']
    if len(relations) > 1:
        targets = [rel['object'] for rel in relations]
        for rel, k in zip(relations, rng.permutation(len(targets))):
            rel['object'] = targets[k]
    return out


def rewire_graphs(graphs, seed):
    """Rewire every graph with its own generator, fixed by (seed, image_id), so the result is order-independent."""
    return {image_id: rewire_graph(graph, np.random.default_rng(np.random.SeedSequence([seed, int(image_id)])))
            for image_id, graph in graphs.items()}


def triple_texts(graphs):
    """The triple phrases of image graphs, exactly as ``GraphStore._graph_parts`` builds them."""
    texts = set()
    for graph in graphs.values():
        names = {obj['id']: obj['name'] for obj in graph['objects']}
        for rel in graph['relations']:
            if rel['subject'] in names and rel['object'] in names:
                texts.add(f"{names[rel['subject']]} {rel['predicate']} {names[rel['object']]}")
    return texts
