"""Training-free soft matching between a parsed query and an image scene graph. Pure numpy.

Every part (an object label, or a whole triple written as a phrase) is a unit text vector, so a dot product
is a cosine. Each part of the query looks for its counterpart among the parts of the image:
  max      the best matching image part counts, the rest of the image is ignored
  softmax  image parts are weighted by softmax(similarity / temperature); tends to max as temperature -> 0
The query-side weights are uniform here; learning them needs training and is not part of this module.
"""
import numpy as np

PRONOUNS = {"that", "it", "they", "them", "he", "she", "this", "these", "those", "there", "who", "which", "what",
            "one", "other", "each", "some", "something", "someone"}


def triple_phrase(subject: str, predicate: str, obj: str) -> str:
    return f"{subject} {predicate} {obj}"


def query_parts(query: dict) -> tuple[list[str], list[str]]:
    """Object labels and triple phrases of a parsed caption; pronoun 'objects' of the parser are dropped."""
    names = [o["name"] for o in query["objects"]]
    nodes = [n for n in dict.fromkeys(names) if n and n not in PRONOUNS]
    triples = [triple_phrase(names[r["subject"]], r["predicate"].lower(), names[r["object"]]) for r in query["relations"]
               if names[r["subject"]] not in PRONOUNS and names[r["object"]] not in PRONOUNS]
    return nodes, list(dict.fromkeys(triples))


def image_parts(graph: dict) -> tuple[list[str], list[str]]:
    names = {o["id"]: o["name"] for o in graph["objects"]}
    triples = [triple_phrase(names[r["subject"]], r["predicate"], names[r["object"]]) for r in graph["relations"]]
    return list(dict.fromkeys(names.values())), list(dict.fromkeys(triples))


def match(query_vecs: np.ndarray, image_vecs: np.ndarray, mode: str = "max", temperature: float = 0.05) -> tuple[float, list]:
    """Score of one query against one image, plus per query part (best image part index, similarity, contribution).

    Returns (nan, []) when either side has no parts: there is no graph evidence, which is not the same as a low score.
    """
    if len(query_vecs) == 0 or len(image_vecs) == 0:
        return float("nan"), []
    sims = query_vecs @ image_vecs.T
    if mode == "max":
        per_part = sims.max(axis=1)
    elif mode == "softmax":
        weights = np.exp((sims - sims.max(axis=1, keepdims=True)) / temperature)
        per_part = (weights / weights.sum(axis=1, keepdims=True) * sims).sum(axis=1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    best = sims.argmax(axis=1)
    details = [(int(j), float(sims[i, j]), float(per_part[i] / len(per_part))) for i, j in enumerate(best)]
    return float(per_part.mean()), details


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Standardise each query's candidate scores; missing scores (nan) become 0, i.e. neutral."""
    mean = np.nanmean(scores, axis=1, keepdims=True)
    std = np.nanstd(scores, axis=1, keepdims=True)
    z = (scores - mean) / np.where(std > 0, std, 1.0)
    return np.nan_to_num(z, nan=0.0)


def rerank(clip_scores: np.ndarray, graph_scores: np.ndarray, ranked_ids: list[list], alpha: float) -> list[list]:
    """alpha * z(CLIP) + (1 - alpha) * z(graph) inside each query's candidate list; ties keep the CLIP order."""
    final = alpha * zscore_rows(clip_scores) + (1 - alpha) * zscore_rows(graph_scores)
    order = np.argsort(-final, axis=1, kind="stable")
    return [[ids[j] for j in row] for ids, row in zip(ranked_ids, order)]
