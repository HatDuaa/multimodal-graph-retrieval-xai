"""Commonsense concept vocabulary and graph of the scene-ranker options (data version concept_v1).

Built once by scripts/build_concept_data.py, from train only:
- vocabulary: canonical image node labels that occur in at least `min_images` train images;
- edges between vocabulary concepts, of four relation types:
  - vg_prior: Visual Genome relation subject -> object, counted once per train image, kept from `prior_min_images`
    images; weight log(1 + count) / log(1 + train images);
  - hypernym: concept -> its nearest WordNet hypernym that is also a vocabulary concept; weight 1;
  - cn_related / cn_function: ConceptNet 5.7 English edges (RelatedTo, Synonym / UsedFor, CapableOf, AtLocation,
    PartOf) of weight >= `conceptnet_min_weight`; weight = ConceptNet weight / the relation type's largest weight;
- per concept, at most `neighbors` neighbours kept (each relation type first gets `quota_per_family` of them, by
  weight); an edge survives when either endpoint keeps it.

A node label links to the concept of the same name, else to the concept named by its last word ("tennis racket" ->
"racket"), else to none.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RELATION_NAMES = ("vg_prior", "hypernym", "cn_related", "cn_function")
SCENE_TERMS = ("attributes", "attributes_soft", "salience", "soft_objects")
NO_CONCEPT = -1


class ConceptVocab:
    def __init__(self, names):
        self.names = list(names)
        self.index = {name: i for i, name in enumerate(self.names)}

    def __len__(self):
        return len(self.names)

    def concept_of(self, label: str) -> int:
        """Concept index of a node label: the whole label, else its last word, else NO_CONCEPT."""
        label = str(label).lower().strip()
        if label in self.index:
            return self.index[label]
        words = label.split()
        return self.index.get(words[-1], NO_CONCEPT) if words else NO_CONCEPT

    @classmethod
    def load(cls, folder: str | Path) -> "ConceptVocab":
        return cls(json.loads((Path(folder) / "concepts.json").read_text(encoding="utf-8"))["names"])


def load_concept_graph(folder: str | Path) -> dict:
    """The concept graph as arrays: features (C, 512) CLIP text vectors of the concept names, and per edge source,
    target, relation (index into relation_names), weight and directed."""
    folder = Path(folder)
    meta = json.loads((folder / "concepts.json").read_text(encoding="utf-8"))
    with np.load(folder / "concept_graph.npz", allow_pickle=False) as data:
        graph = {key: data[key] for key in ("features", "source", "target", "relation", "weight", "directed")}
    graph["relation_names"] = meta["relation_names"]
    if len(graph["features"]) != len(meta["names"]):
        raise ValueError(f"concept graph and vocabulary sizes differ in {folder}")
    return graph
