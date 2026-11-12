"""Fixed scene terms of one caption against its candidate images (src/graph/concepts.py SCENE_TERMS).

Nothing here is learned; scripts/build_concept_data.py computes the terms once per (caption, candidate) and divides
each by its train scale, and GraphReranker(scene_terms=4) adds them to the graph score with learned weights.

- attributes: share of the caption's attribute facts ("red bus": entity bus, attribute red), among facts whose entity
  links to a concept, for which some image node of that concept carries an attribute sharing a content word;
- attributes_soft: mean over attribute facts of the best CLIP-text cosine between "red bus" and any "attribute label"
  text of the image ("red double decker"); 0 when the image has no attributes;
- salience: mean over the caption's concepts of the largest box share (box area / image area) of an image node of
  that concept; 0 for a concept absent from the image;
- soft_objects: IDF-weighted mean over the caption's entity labels of the best CLIP-text cosine to any image node
  label, IDF from train captions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.graph.concepts import NO_CONCEPT, SCENE_TERMS

STOP_WORDS = frozenset("a an the of and or in on with to at by for very is are some".split())


def content_words(text: str) -> frozenset[str]:
    """Lower-case alphabetic words of an attribute, without function words ("light blue" -> {light, blue})."""
    return frozenset(w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in STOP_WORDS)


def attribute_text(attribute: str, label: str) -> str:
    return f"{attribute} {label}".strip()


@dataclass
class ImageSide:
    """What the terms need of one image graph."""
    label_rows: np.ndarray                      # part-table rows of the node labels' CLIP text vectors
    attribute_rows: np.ndarray                  # rows of its "attribute label" texts in the attribute table
    words: dict[int, frozenset] = field(default_factory=dict)   # concept -> attribute content words of its nodes
    salience: dict[int, float] = field(default_factory=dict)    # concept -> largest box share of its nodes


@dataclass
class CaptionSide:
    """What the terms need of one caption graph."""
    facts: list[tuple[int, frozenset]]          # (entity concept, attribute content words), linked entities only
    fact_rows: np.ndarray                       # attribute-table rows of every fact's "attribute entity" text
    concepts: list[int]                         # distinct linked concepts of its entities
    label_rows: np.ndarray                      # part-table rows of its distinct entity labels
    label_weights: np.ndarray                   # their train IDF


def image_side(graph, attributes, vocab, label_row, attribute_row) -> ImageSide:
    """graph: a packaged image graph (merge_v1: box per node, width, height); attributes: one list per node."""
    words, salience, texts = {}, {}, []
    area = float(graph["width"]) * float(graph["height"]) or 1.
    for obj, values in zip(graph["objects"], attributes, strict=True):
        concept = vocab.concept_of(obj["name"])
        texts.extend(attribute_text(value, obj["name"]) for value in values)
        if concept == NO_CONCEPT:
            continue
        words[concept] = words.get(concept, frozenset()).union(*(content_words(v) for v in values))
        box = obj.get("box")
        if box is not None:
            salience[concept] = max(salience.get(concept, 0.), min(1., box[2] * box[3] / area))
    return ImageSide(label_rows=np.array(sorted({label_row[obj["name"]] for obj in graph["objects"]}), dtype=np.int64),
                     attribute_rows=np.array(sorted({attribute_row[t] for t in texts}), dtype=np.int64),
                     words=words, salience=salience)


def caption_side(objects, vocab, label_row, attribute_row, idf, unseen_idf) -> CaptionSide:
    """objects: the caption graph's nodes (name, attributes), pronouns already removed."""
    facts, fact_texts, concepts, labels = [], [], set(), set()
    for obj in objects:
        name = obj["name"]
        if not name:
            continue
        labels.add(name)
        concept = vocab.concept_of(name)
        if concept != NO_CONCEPT:
            concepts.add(concept)
        for value in obj.get("attributes") or ():
            fact_texts.append(attribute_text(value, name))
            if concept != NO_CONCEPT and content_words(value):
                facts.append((concept, content_words(value)))
    labels = sorted(labels)
    return CaptionSide(facts=facts, fact_rows=np.array(sorted({attribute_row[t] for t in fact_texts}), dtype=np.int64),
                       concepts=sorted(concepts), label_rows=np.array([label_row[x] for x in labels], dtype=np.int64),
                       label_weights=np.array([idf.get(x, unseen_idf) for x in labels], dtype=np.float64))


def _best_cosine(queries, images, table):
    """(len(queries), len(images)): best cosine of each query row to the rows of each image; 0 for an empty image."""
    out = np.zeros((len(queries), len(images)), dtype=np.float32)
    if not len(queries):
        return out
    rows = np.unique(np.concatenate([image for image in images if len(image)] or [np.zeros(0, dtype=np.int64)]))
    if not len(rows):
        return out
    products = table[queries] @ table[rows].T
    position = {row: i for i, row in enumerate(rows.tolist())}
    for j, image in enumerate(images):
        if len(image):
            out[:, j] = products[:, [position[row] for row in image.tolist()]].max(1)
    return out


def caption_terms(caption: CaptionSide, images: list[ImageSide], label_table, attribute_table) -> np.ndarray:
    """Terms (len(images), len(SCENE_TERMS)) of one caption, in SCENE_TERMS order."""
    terms = np.zeros((len(images), len(SCENE_TERMS)), dtype=np.float32)
    for k, image in enumerate(images):
        if caption.facts:
            terms[k, 0] = np.mean([bool(words & image.words.get(concept, frozenset()))
                                   for concept, words in caption.facts])
        if caption.concepts:
            terms[k, 2] = np.mean([image.salience.get(concept, 0.) for concept in caption.concepts])
    if len(caption.fact_rows):
        terms[:, 1] = _best_cosine(caption.fact_rows, [i.attribute_rows for i in images], attribute_table).mean(0)
    if len(caption.label_rows):
        best = _best_cosine(caption.label_rows, [i.label_rows for i in images], label_table)
        terms[:, 3] = caption.label_weights @ best / caption.label_weights.sum()
    return terms
