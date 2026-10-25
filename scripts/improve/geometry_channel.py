"""Rule-based count + spatial channel added to the trained main_r3 score on val (part 2c). No training.

final = fused_main_r3 + w * z(geometry), z-scored over each query's top-50 like every other channel; geometry is the
mean of the query's checkable items (src/graph/geometry.py): explicit counts parsed from the caption text with spaCy,
and spatial triples of the level-1 (sng) query graph. A caption label matches an image instance when their Visual
Genome-canonical labels are equal or the frozen CLIP vectors of "a photo of a {}" have cosine >= 0.955 (fixed before
looking at val: synonyms and plurals such as man/men, dog/puppy score >= 0.958, distinct classes such as dog/cat,
man/woman <= 0.945). The weight w is swept on val; a 2-fold split of val by gold image also picks w on one half and
scores the other, to show how much the sweep itself flatters the result.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks.three_channel_probe import zscore_rows  # noqa: E402
from scripts.improve.parser_rerank_eval import add_part_vectors  # noqa: E402
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import aggregate_seeds, evaluate  # noqa: E402
from src.graph.geometry import caption_counts, count_score, merge_duplicates, spatial_kind, spatial_score  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.graph.soft_match import PRONOUNS  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "geometry-channel"
THRESHOLD = 0.955
WEIGHTS = [0., .05, .1, .15, .2, .3, .4, .5, .75, 1., 1.5, 2.]
FEATURES = ("geometry", "count", "spatial")
VIEWS = ("all", "with_count", "with_spatial")


@torch.no_grad()
def fused_scores(model, store, caption_ids, device, batch_size=256):
    scores = []
    for start in range(0, len(caption_ids), batch_size):
        fused, _ = model(store.batch(caption_ids[start:start + batch_size], device))
        scores.append(fused.cpu().numpy())
    return np.concatenate(scores)


class LabelMatcher:
    def __init__(self, store, vocab):
        self.store, self.vocab, self.cache = store, vocab, {}

    def vector(self, label):
        return self.store.part_vectors[self.store.part_row[f"node:{label}"]]

    def __call__(self, query_label, image_label):
        key = (query_label, image_label)
        if key not in self.cache:
            same = self.vocab.canon_object(query_label) == self.vocab.canon_object(image_label)
            self.cache[key] = same or float(self.vector(query_label) @ self.vector(image_label)) >= THRESHOLD
        return self.cache[key]


def item_lists(store, caption_ids, texts):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    counts = [caption_counts(doc) for doc in nlp.pipe([texts[c] for c in caption_ids], batch_size=256)]
    spatial = []
    for cid in caption_ids:
        graph = store.queries[cid]
        names = [o["name"] for o in graph["objects"]]
        items = []
        for rel in graph["relations"]:
            kind = spatial_kind(rel["predicate"])
            s, o = names[rel["subject"]], names[rel["object"]]
            if kind and s and o and s not in PRONOUNS and o not in PRONOUNS:
                items.append((kind, s, o, rel["predicate"].lower()))
        spatial.append(list(dict.fromkeys(items)))
    return counts, spatial


def features(counts, spatial, candidates, store, boxes, match):
    n, k = candidates.shape
    out = {f: np.full((n, k), np.nan) for f in FEATURES}
    for i in range(n):
        if not counts[i] and not spatial[i]:
            continue
        for j, image_id in enumerate(candidates[i]):
            graph, image_boxes = store.graphs[int(image_id)], boxes[str(int(image_id))]["boxes"]
            labels = [o["name"] for o in graph["objects"]]
            found = lambda label: [image_boxes[t] for t, name in enumerate(labels) if match(label, name)]  # noqa: E731
            c = [count_score(stated, found(noun)) for stated, noun in counts[i]]
            s = [spatial_score(kind, found(subj), found(obj)) for kind, subj, obj, _ in spatial[i]]
            c, s = [x for x in c if x is not None], [x for x in s if x is not None]
            if c:
                out["count"][i, j] = np.mean(c)
            if s:
                out["spatial"][i, j] = np.mean(s)
            if c or s:
                out["geometry"][i, j] = np.mean(c + s)
    return out


def ranked(scores, candidates):
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1).tolist()


def metrics(scores, candidates, golds, masks):
    lists = ranked(scores, candidates)
    return {v: evaluate([r for r, m in zip(lists, mask) if m], [g for g, m in zip(golds, mask) if m])
            for v, mask in masks.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[f"experiments/level1/gat/main_r3_seed{k}" for k in range(3)])
    parser.add_argument("--limit", type=int, help="Smoke test only: the first N val queries.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GraphStore(cfg, splits=("val",))
    arrays = store.candidates("val")
    caption_ids = [str(c) for c in arrays["caption_ids"]][:args.limit]
    candidates, golds = arrays["candidate_ids"][:args.limit], [int(g) for g in arrays["gold"]][:args.limit]
    images = json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
    texts = {c["caption_id"]: c["text"] for im in images for c in im["captions"]}
    boxes = json.loads((resolve(cfg, "processed") / "geometry" / "boxes_val.json").read_text(encoding="utf-8"))
    counts, spatial = item_lists(store, caption_ids, texts)
    labels = {n for _, n in sum(counts, [])} | {x for items in spatial for _, s, o, _ in items for x in (s, o)}
    added = add_part_vectors(store, cfg, {f"node:{x}" for x in labels}, device)
    match = LabelMatcher(store, Vocab.from_config(cfg))
    feats = features(counts, spatial, candidates, store, boxes, match)
    masks = {"all": [True] * len(caption_ids), "with_count": [bool(c) for c in counts],
             "with_spatial": [bool(s) for s in spatial]}
    fold = np.array([g % 2 for g in golds])
    base = {run: fused_scores(load_run_model(Path(run), device)[0], store, caption_ids, device) for run in args.runs}
    result = {"split": "val", "runs": args.runs, "threshold": THRESHOLD, "weights": WEIGHTS, "newly_encoded_labels": added,
              "n": {v: int(sum(m)) for v, m in masks.items()},
              "count_items": sum(len(c) for c in counts), "spatial_items": sum(len(s) for s in spatial),
              "spatial_kinds": {k: sum(item[0] == k for s in spatial for item in s) for k in
                                ("left", "right", "above", "below", "on_top", "beside")},
              "candidates_checkable": {f: float(np.isfinite(feats[f]).mean()) for f in FEATURES}, "features": {}}
    for name in FEATURES:
        z, _ = zscore_rows(feats[name])
        sweep = {}
        for w in WEIGHTS:
            per_run = {run: metrics(score + w * z, candidates, golds, masks) for run, score in base.items()}
            sweep[str(w)] = {v: aggregate_seeds([per_run[r][v] for r in base]) for v in VIEWS}
        best = max(WEIGHTS, key=lambda w: sweep[str(w)]["all"]["recall@1"]["mean"])
        cross = {}
        for run, score in base.items():
            final = score.copy()
            for f in (0, 1):
                train = fold != f
                w = max(WEIGHTS, key=lambda w: evaluate(ranked((score + w * z)[train], candidates[train]),
                                                           [g for g, t in zip(golds, train) if t])["recall@1"])
                final[fold == f] = (score + w * z)[fold == f]
                cross.setdefault("chosen_weights", {})[f"{run}:fold{f}"] = w
            cross[run] = metrics(final, candidates, golds, masks)
        result["features"][name] = {"sweep": sweep, "best_weight_on_val": best,
                                    "two_fold": {v: aggregate_seeds([cross[r][v] for r in base]) for v in VIEWS},
                                    "two_fold_weights": cross["chosen_weights"]}
        print(name, "best w", best, json.dumps({v: round(sweep[str(best)][v]["recall@1"]["mean"], 4) for v in VIEWS}),
              "w=0", json.dumps({v: round(sweep["0.0"][v]["recall@1"]["mean"], 4) for v in VIEWS}), flush=True)
    result["seconds"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "geometry_channel.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_examples(args.out, caption_ids, texts, counts, spatial, golds, store, boxes, match)


def write_examples(out, caption_ids, texts, counts, spatial, golds, store, boxes, match, n=12):
    """Readable explanation lines for the gold image of the first queries with checkable items."""
    lines = ["# Ví dụ giải thích kênh số lượng + vị trí (ảnh đúng của câu)", ""]
    shown = 0
    for i, cid in enumerate(caption_ids):
        if shown >= n or not (counts[i] or spatial[i]):
            continue
        graph, image_boxes = store.graphs[golds[i]], boxes[str(golds[i])]["boxes"]
        labels = [o["name"] for o in graph["objects"]]
        rows = []
        for stated, noun in counts[i]:
            found = [image_boxes[t] for t, name in enumerate(labels) if match(noun, name)]
            score = count_score(stated, found)
            if score is not None:
                rows.append(f"câu: {stated} {noun} — ảnh: {len(merge_duplicates(found))} {noun} "
                            f"{'✓' if score == 1 else f'✗ (điểm {score:.2f})'}")
        for kind, subj, obj, predicate in spatial[i]:
            s = [image_boxes[t] for t, name in enumerate(labels) if match(subj, name)]
            o = [image_boxes[t] for t, name in enumerate(labels) if match(obj, name)]
            score = spatial_score(kind, s, o)
            if score is not None:
                rows.append(f"câu: {subj} {predicate} {obj} — ảnh: {round(score * len(s) * len(o))}/{len(s) * len(o)} "
                            f"cặp khung thoả '{kind}' {'✓' if score >= .5 else '✗'}")
        if rows:
            lines += [f"- `{cid}` {texts[cid]}"] + [f"  - {r}" for r in rows]
            shown += 1
    (out / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
