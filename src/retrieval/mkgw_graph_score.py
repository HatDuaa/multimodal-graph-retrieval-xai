"""Graph score for MKG-W re-ranking (level 2, step b): NativE embeddings replace the GNN.

The query is a masked entity description. It usually still names OTHER entities
("film by John Ford"), so the graph channel works in three steps:

  1. EntityLinker finds every KG entity whose label/alias appears in the query.
  2. For a candidate image (= entity), the embedding channel scores how close the
     candidate is to the linked entities in NativE's entity-embedding space, and
     the triple channel counts direct KGC-train edges between them.
  3. The matches used for the score are returned with it; they ARE the
     explanation (assignment rule: explanations must come from the ranking
     mechanism itself, never found post-hoc).

Final fusion (evaluation script): alpha * z(CLIP) + (1 - alpha) * z(graph), with
graph = beta * z(emb) + (1 - beta) * z(triple), following the level-1 convention
(experiments/checks/README.md, section 6). alpha and beta are chosen on val only.
Only triples with kgc_split == "train" may enter the adjacency (anti-leakage).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_NGRAM = 5      # longest entity name, in tokens, that the linker will match
MIN_CHARS = 3      # names shorter than this are noise ("US" would still match via aliases)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


class EntityLinker:
    """Match KG entity labels/aliases inside a query string (longest match wins)."""

    def __init__(self, names_of: dict[str, list[str]]):
        # names_of: qid -> every surface form (label + aliases)
        self.by_ngram: dict[tuple, set[str]] = defaultdict(set)
        for qid, names in names_of.items():
            for name in names:
                if not name or len(name) < MIN_CHARS:
                    continue
                toks = tuple(_tokenize(name))
                if 0 < len(toks) <= MAX_NGRAM:
                    self.by_ngram[toks].add(qid)

    @classmethod
    def from_cache(cls, cache_path: str | Path, lang: str = "en") -> "EntityLinker":
        names_of = {}
        for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            names_of[row["qid"]] = [row.get(f"label_{lang}")] + list(row.get(f"aliases_{lang}", []))
        return cls(names_of)

    def link(self, text: str, exclude: set[str] = frozenset()) -> dict[str, str]:
        """qid -> matched surface form. Longer n-grams are matched first; a token
        consumed by a match cannot start a shorter one (so "New York City" does
        not also link "York")."""
        toks = _tokenize(text)
        linked: dict[str, str] = {}
        used = [False] * len(toks)
        for n in range(min(MAX_NGRAM, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(used[i:i + n]):
                    continue
                for qid in self.by_ngram.get(tuple(toks[i:i + n]), ()):
                    if qid not in exclude:
                        linked.setdefault(qid, " ".join(toks[i:i + n]))
                        used[i:i + n] = [True] * n
        return linked


class MkgwGraphScorer:
    """Embedding + triple channels for one (query, candidate) pair."""

    def __init__(self, ent_feats: np.ndarray, qid_index: dict[str, int],
                 triples: list[dict], relation_labels: dict[str, str] | None = None):
        # ent_feats rows are L2-normalised (scripts/export_native_embeddings.py)
        self.feats = ent_feats
        self.index = qid_index
        self.rel_label = relation_labels or {}
        self.adj: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for t in triples:
            assert t.get("kgc_split") == "train", "only KGC-train triples may build the graph"
            self.adj[(t["h_qid"], t["t_qid"])].append((t["r_pid"], "->"))
            self.adj[(t["t_qid"], t["h_qid"])].append((t["r_pid"], "<-"))

    def score(self, linked: dict[str, str], candidate_qid: str,
              exclude_triples: set[tuple] = frozenset()) -> dict:
        """Returns {"emb": float|None, "triple": float, "matches": [...]}.

        emb    = best cosine between the candidate and a linked entity (None when
                 nothing is linked; the caller treats None as neutral).
        triple = number of direct KGC-train edges between the candidate and any
                 linked entity, minus the excluded ones (for fidelity checks).
        """
        matches = []
        emb = None
        cand_row = self.index.get(candidate_qid)
        for qid, surface in linked.items():
            if qid == candidate_qid:
                continue
            row = self.index.get(qid)
            if cand_row is not None and row is not None:
                sim = float(self.feats[cand_row] @ self.feats[row])
                if emb is None or sim > emb:
                    emb = sim
                    matches = [m for m in matches if m["via"] != "embedding"]
                    matches.append({"qid": qid, "surface": surface, "via": "embedding", "score": sim})
            for pid, direction in self.adj.get((candidate_qid, qid), ()):
                if (candidate_qid, pid, qid, direction) in exclude_triples:
                    continue
                matches.append({"qid": qid, "surface": surface, "via": "triple",
                                "pid": pid, "label": self.rel_label.get(pid, pid),
                                "direction": direction})
        n_triples = sum(1 for m in matches if m["via"] == "triple")
        return {"emb": emb, "triple": float(n_triples), "matches": matches}

    def score_without(self, linked: dict[str, str], candidate_qid: str,
                      removed: list[dict]) -> dict:
        """Recompute after removing explanation items (assignment: fidelity)."""
        removed_qids = {m["qid"] for m in removed if m["via"] == "embedding"}
        removed_triples = {(candidate_qid, m["pid"], m["qid"], m["direction"])
                           for m in removed if m["via"] == "triple"}
        kept = {q: s for q, s in linked.items() if q not in removed_qids}
        return self.score(kept, candidate_qid, exclude_triples=removed_triples)


def load_train_triples(path: str | Path, id2qid: dict[int, str],
                       id2pid: dict[int, str]) -> list[dict]:
    """Read data/processed/mkgw_triples.jsonl and keep ONLY the KGC-train part,
    converting NativE integer ids to Wikidata qids/pids."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t["kgc_split"] == "train":
            out.append({"h_qid": id2qid[t["h"]], "t_qid": id2qid[t["t"]],
                        "r_pid": id2pid[t["r"]], "kgc_split": "train"})
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / (std + 1e-6)


def fuse(clip: np.ndarray, emb: np.ndarray, triple: np.ndarray,
         alpha: float, beta: float) -> np.ndarray:
    """Level-1 convention: alpha * z(CLIP) + (1-alpha) * z(beta*z(emb) + (1-beta)*z(triple)).
    All arrays are scores of one query's candidate list."""
    graph = beta * zscore(emb) + (1.0 - beta) * zscore(triple)
    return alpha * zscore(clip) + (1.0 - alpha) * zscore(graph)
