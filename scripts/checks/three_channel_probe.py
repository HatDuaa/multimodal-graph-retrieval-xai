"""Measure the fixed three-channel training-free reranker reference on validation only.

The optional ``sbert`` encoder uses ``sentence-transformers/all-mpnet-base-v2`` and is not a project
dependency. Test is never read: the saved reference is the score the trained model must reproduce at step 0.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import evaluate  # noqa: E402
from src.features.extract_clip import ClipEncoder, load_features, save_features  # noqa: E402
from src.graph.parse_query import parse_many  # noqa: E402
from src.graph.scene_graph import Vocab, iter_scene_graphs, load_raw  # noqa: E402
from src.graph.soft_match import channel_scores, fuse_three_channels, instance_parts, query_instance_parts  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

GRID = [round(i / 10, 1) for i in range(11)]


def parse_top_k(values: list[str] | None) -> list[int]:
    """Parse repeatable and comma-separated candidate-pool sizes."""
    values = values or ["50"]
    result = sorted({int(item) for value in values for item in value.split(",")})
    if any(value < 0 for value in result):
        raise ValueError("--top-k values must be non-negative")
    return result


def candidate_pools(caption_features: np.ndarray, image_features: np.ndarray, image_ids: list[int],
                    ks: list[int]) -> dict:
    """Build candidate pools through the same FAISS path as the CLIP baseline."""
    index = CosineIndex(image_features, image_ids)
    result = {}
    for k in ks:
        width = len(image_ids) if k == 0 else min(k, len(image_ids))
        scores, ranked = index.search(caption_features, width)
        result[k] = (scores, ranked)
    return result


def zscore_rows(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply zscore_channel row-wise and return z-scores plus active-row flags."""
    scores = np.asarray(scores, dtype=float)
    known = ~np.isnan(scores)
    count = known.sum(axis=1)
    safe_count = np.maximum(count, 1)
    means = np.where(known, scores, 0.0).sum(axis=1) / safe_count
    centered = np.where(known, scores - means[:, None], 0.0)
    stds = np.sqrt((centered * centered).sum(axis=1) / safe_count)
    active = (count >= 2) & (stds >= 1e-6)
    filled = np.where(known, scores, means[:, None])
    z = np.zeros_like(scores, dtype=float)
    z[active] = (filled[active] - means[active, None]) / (stds[active, None] + 1e-6)
    return z, active


def fuse_rows(clip: np.ndarray, objects: np.ndarray, triples: np.ndarray, alpha: float,
              beta: float) -> tuple[np.ndarray, dict]:
    """Fuse all query rows with the same inactive-channel rules as fuse_three_channels."""
    z_clip, _ = zscore_rows(clip)
    z_objects, object_active = zscore_rows(objects)
    z_triples, triple_active = zscore_rows(triples)
    both = object_active & triple_active
    object_only = object_active & ~triple_active
    triple_only = triple_active & ~object_active
    graph = np.zeros_like(z_clip)
    graph[object_only] = z_objects[object_only]
    graph[triple_only] = z_triples[triple_only]
    if np.any(both):
        graph[both], _ = zscore_rows(beta * z_objects[both] + (1 - beta) * z_triples[both])
    return alpha * z_clip + (1 - alpha) * graph, {
        "z_clip": z_clip,
        "z_objects": z_objects,
        "z_triples": z_triples,
        "object_active": object_active,
        "triple_active": triple_active,
    }


def ranked_metrics(scores: np.ndarray, ranked: list[list[int]], golds: list[int]) -> tuple[dict, list[list[int]]]:
    """Stable-sort candidate IDs by scores and evaluate them."""
    orders = [list(np.asarray(ids)[np.argsort(-row, kind="stable")]) for row, ids in zip(scores, ranked)]
    return evaluate(orders, golds), orders


def sweep(clip: np.ndarray, objects: np.ndarray, triples: np.ndarray, ranked: list[list[int]], golds: list[int],
          alphas: list[float] = GRID, betas: list[float] = GRID) -> tuple[dict, dict, dict]:
    """Sweep alpha and beta after computing each channel z-score once."""
    z_clip, _ = zscore_rows(clip)
    z_objects, object_active = zscore_rows(objects)
    z_triples, triple_active = zscore_rows(triples)
    both = object_active & triple_active
    object_only = object_active & ~triple_active
    triple_only = triple_active & ~object_active
    grid = {}
    best = None
    for alpha in alphas:
        for beta in betas:
            graph = np.zeros_like(z_clip)
            graph[object_only] = z_objects[object_only]
            graph[triple_only] = z_triples[triple_only]
            if np.any(both):
                graph[both], _ = zscore_rows(beta * z_objects[both] + (1 - beta) * z_triples[both])
            fused = alpha * z_clip + (1 - alpha) * graph
            metrics, _ = ranked_metrics(fused, ranked, golds)
            key = f"{alpha:.1f},{beta:.1f}"
            grid[key] = metrics["recall@1"]
            if best is None or (metrics["recall@1"], metrics["mrr"]) > (best["recall@1"], best["mrr"]):
                best = {"alpha": alpha, "beta": beta, **metrics}
    return best, grid, {"z_clip": z_clip, "z_objects": z_objects, "z_triples": z_triples}


def part_vectors(cfg: dict, encoder_name: str, node_texts: list[str], triple_texts: list[str]
                 ) -> tuple[np.ndarray, dict, str]:
    """Load a complete part cache or encode and save a new cache without overwriting any file."""
    features = resolve(cfg, "features")
    stem = features / ("vg_coco_val_graph_parts" if encoder_name == "clip" else "vg_coco_val_graph_parts_sbert")
    required = [f"node:{text}" for text in node_texts] + [f"triple:{text}" for text in triple_texts]
    candidates = [stem] + [features / f"{stem.name}_v{i}" for i in range(2, 100)]
    for path in candidates:
        if not path.with_suffix(".npy").exists() or not path.with_suffix(".ids.json").exists():
            continue
        vectors, keys, saved_encoder = load_features(path)
        row = {key: index for index, key in enumerate(keys)}
        if all(key in row for key in required):
            return vectors, row, saved_encoder
    if encoder_name == "clip":
        encoder = ClipEncoder.from_config(cfg)
        texts = [cfg["clip"]["concept_prompt"].format(text) for text in node_texts] + triple_texts
        vectors = encoder.encode_texts(texts)
        saved_encoder = f"{cfg['clip']['model']}/{cfg['clip']['pretrained']}"
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        vectors = encoder.encode(node_texts + triple_texts, normalize_embeddings=True)
        saved_encoder = "sentence-transformers/all-mpnet-base-v2"
    target = next(path for path in candidates
                  if not path.with_suffix(".npy").exists() and not path.with_suffix(".ids.json").exists())
    save_features(target, vectors, required, saved_encoder)
    return vectors, {key: index for index, key in enumerate(required)}, saved_encoder


def lookup(vectors: np.ndarray, row: dict, kind: str, parts: list[str]) -> np.ndarray:
    """Look up vectors for an ordered list of instances."""
    if not parts:
        return np.empty((0, vectors.shape[1]), dtype=vectors.dtype)
    return vectors[[row[f"{kind}:{part}"] for part in parts]]


def channel_matrix(query_vectors: list[np.ndarray], image_vectors: dict[int, np.ndarray], ranked: list[list[int]],
                   temperature: float) -> np.ndarray:
    """Compute one channel for every query and candidate exactly once."""
    return np.asarray([channel_scores(query, [image_vectors[item] for item in ids], temperature)
                       for query, ids in zip(query_vectors, ranked)])


def save_reference(path: Path, caption_ids: list[str], candidate_ids: list[list[int]], clip: np.ndarray,
                   objects: np.ndarray, triples: np.ndarray, alpha: float, beta: float, temperature: float) -> None:
    """Save raw channels and the selected reference ranking for the first 200 rows."""
    limit = min(200, len(caption_ids))
    fused, parts = fuse_rows(clip[:limit], objects[:limit], triples[:limit], alpha, beta)
    _, object_active = zscore_rows(objects[:limit])
    _, triple_active = zscore_rows(triples[:limit])
    graph = np.zeros_like(fused)
    graph[object_active & ~triple_active] = parts["z_objects"][object_active & ~triple_active]
    graph[triple_active & ~object_active] = parts["z_triples"][triple_active & ~object_active]
    both = object_active & triple_active
    if np.any(both):
        graph[both], _ = zscore_rows(beta * parts["z_objects"][both] + (1 - beta) * parts["z_triples"][both])
    candidate_array = np.asarray(candidate_ids[:limit])
    order = np.argsort(-fused, axis=1, kind="stable")
    final_ids = np.take_along_axis(candidate_array, order, axis=1)
    np.savez(path, caption_ids=np.asarray(caption_ids[:limit]), candidate_ids=np.asarray(candidate_ids[:limit]),
             clip=clip[:limit], objects=objects[:limit], triples=triples[:limit], z_clip=parts["z_clip"],
             z_objects=parts["z_objects"], z_triples=parts["z_triples"], graph=graph, fused=fused,
             order=final_ids, alpha=alpha, beta=beta, temperature=temperature)


def main() -> None:
    """Run the validation-only reference probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--top-k", action="append", default=None)
    parser.add_argument("--encoder", choices=("clip", "sbert"), default="clip")
    parser.add_argument("--temperatures", default="0.02,0.05,0.1")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    ks = parse_top_k(args.top_k)
    temperatures = [float(value) for value in args.temperatures.split(",")]
    split_path = resolve(cfg, "splits") / "vg_coco_val.json"
    if not split_path.exists():
        raise SystemExit(f"missing {split_path}; run the split builder")
    images = json.loads(split_path.read_text(encoding="utf-8"))["images"]
    pool = [image["image_id"] for image in images]
    text_of = {caption["caption_id"]: caption["text"] for image in images for caption in image["captions"]}
    gold_of = {caption["caption_id"]: image["image_id"] for image in images for caption in image["captions"]}
    feature_dir = resolve(cfg, "features")
    image_features, image_feature_ids, image_encoder = load_features(feature_dir / "vg_coco_image")
    caption_features, caption_feature_ids, caption_encoder = load_features(feature_dir / "vg_coco_caption")
    if image_encoder != caption_encoder:
        raise SystemExit("image and caption features use different encoders")
    image_row = {image_id: index for index, image_id in enumerate(image_feature_ids)}
    caption_row = {caption_id: index for index, caption_id in enumerate(caption_feature_ids)}
    rows_path = REPO_ROOT / cfg["paths"]["experiments"] / "level1" / "clip_baseline" / "top50_val.json"
    if not rows_path.exists():
        raise SystemExit(f"missing {rows_path}; run: python -m src.retrieval.clip_baseline --splits val")
    baseline_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    caption_ids = [row["caption_id"] for row in baseline_rows]
    query_features = caption_features[[caption_row[caption_id] for caption_id in caption_ids]]
    pool_features = image_features[[image_row[image_id] for image_id in pool]]
    pools = candidate_pools(query_features, pool_features, pool, sorted(set(ks) | {50}))
    baseline_ranked = [row["ranked"] for row in baseline_rows]
    baseline_scores = np.asarray([row["scores"] for row in baseline_rows])
    if any(expected != actual for expected, actual in zip(baseline_ranked, pools[50][1])) or not np.allclose(
            pools[50][0], baseline_scores, atol=1e-5):
        raise AssertionError("recomputed K=50 candidates do not match top50_val.json")
    parsed = parse_many([text_of[caption_id] for caption_id in caption_ids], args.workers)
    query_parts = [query_instance_parts(query) for query in parsed]
    raw = load_raw(cfg)
    vocab = Vocab.from_config(cfg)
    graphs = {graph["image_id"]: graph for graph in iter_scene_graphs(raw, vocab, pool, False)}
    image_parts = {image_id: instance_parts(graph) for image_id, graph in graphs.items()}
    node_texts = sorted({part for nodes, _ in query_parts for part in nodes} |
                        {part for nodes, _ in image_parts.values() for part in nodes})
    triple_texts = sorted({part for _, triples in query_parts for part in triples} |
                          {part for _, triples in image_parts.values() for part in triples})
    vectors, vector_row, part_encoder = part_vectors(cfg, args.encoder, node_texts, triple_texts)
    query_nodes = [lookup(vectors, vector_row, "node", nodes) for nodes, _ in query_parts]
    query_triples = [lookup(vectors, vector_row, "triple", triples) for _, triples in query_parts]
    image_nodes_vec = {image_id: lookup(vectors, vector_row, "node", parts[0])
                       for image_id, parts in image_parts.items()}
    image_triples_vec = {image_id: lookup(vectors, vector_row, "triple", parts[1])
                         for image_id, parts in image_parts.items()}
    golds = [gold_of[caption_id] for caption_id in caption_ids]
    report = {
        "date": time.strftime("%Y-%m-%d"), "split": "val", "queries": len(caption_ids), "pool": len(pool),
        "config": {"encoder": part_encoder,
                    "node_prompt": cfg["clip"]["concept_prompt"] if args.encoder == "clip" else None,
                    "temperatures": temperatures, "grid_step": 0.1,
                    "zscore_rule": "nan gets known mean; fewer than 2 known or std < 1e-6 gives all zero; "
                                    "population std",
                    "shuffle_seed": cfg["seed"]},
        "parts": {"distinct_node_texts": len(node_texts), "distinct_triple_texts": len(triple_texts),
                  "mean_object_instances_per_image": float(np.mean([len(parts[0]) for parts in image_parts.values()])),
                  "mean_relation_instances_per_image": float(
                      np.mean([len(parts[1]) for parts in image_parts.values()])),
                  "mean_object_instances_per_query": float(np.mean([len(parts[0]) for parts in query_parts])),
                  "mean_relation_instances_per_query": float(np.mean([len(parts[1]) for parts in query_parts])),
                  "queries_without_triples": sum(not triples for _, triples in query_parts),
                  "images_without_triples": sum(not parts[1] for parts in image_parts.values())},
        "clip_only": evaluate(pools[50][1], golds), "results": {}, "controls": {},
    }
    order_ks = [50] + [k for k in ks if k != 50]
    best_temperature = None
    for k in order_ks:
        ranked = pools[k][1]
        clip = pools[k][0]
        report["results"][str(k)] = {}
        temperatures_for_k = temperatures if k == 50 else [best_temperature[0]]
        for temperature in temperatures_for_k:
            objects = np.asarray([channel_scores(query, [image_nodes_vec[item] for item in ids], temperature)
                                  for query, ids in zip(query_nodes, ranked)])
            triples = np.asarray([channel_scores(query, [image_triples_vec[item] for item in ids], temperature)
                                  for query, ids in zip(query_triples, ranked)])
            check_scores, _ = fuse_rows(clip[:50], objects[:50], triples[:50], 0.3, 0.7)
            for index in range(min(50, len(clip))):
                expected = fuse_three_channels(clip[index], objects[index], triples[index], 0.3, 0.7)
                assert np.allclose(check_scores[index], expected, atol=1e-9), "vectorized fusion mismatch"
            best, grid, _ = sweep(clip, objects, triples, ranked, golds)
            report["results"][str(k)][str(temperature)] = {"best": best, "r1_grid": grid}
            if k == 50 and (best_temperature is None or
                             (best["recall@1"], best["mrr"]) >
                             (best_temperature[1]["recall@1"], best_temperature[1]["mrr"])):
                best_temperature = (temperature, best, objects, triples)
        if k == 50:
            temperature, best, objects, triples = best_temperature
            reference = {"alpha": best["alpha"], "beta": best["beta"], "temperature": temperature,
                         "metrics": {key: best[key] for key in
                                     ("n_queries", "recall@1", "recall@5", "recall@10", "mrr")}}
            shuffled = pool[:]
            random.Random(cfg["seed"]).shuffle(shuffled)
            shuffled_nodes = {image_id: image_nodes_vec[source] for image_id, source in zip(pool, shuffled)}
            shuffled_triples = {image_id: image_triples_vec[source] for image_id, source in zip(pool, shuffled)}
            shuffled_objects = np.asarray([channel_scores(query, [shuffled_nodes[item] for item in ids], temperature)
                                           for query, ids in zip(query_nodes, ranked)])
            shuffled_relation = np.asarray([channel_scores(query, [shuffled_triples[item] for item in ids], temperature)
                                            for query, ids in zip(query_triples, ranked)])
            controls = {"clip_only": evaluate(ranked, golds)}
            controls_to_run = (("objects_only", 1.0, objects, np.full_like(triples, np.nan)),
                               ("triples_only", 0.0, np.full_like(objects, np.nan), triples),
                               ("shuffled", None, shuffled_objects, shuffled_relation))
            for name, beta, object_scores, triple_scores in controls_to_run:
                alpha_best = None
                for alpha in GRID:
                    beta_values = GRID if beta is None else [beta]
                    for beta_value in beta_values:
                        fused, _ = fuse_rows(clip, object_scores, triple_scores, alpha, beta_value)
                        metrics, _ = ranked_metrics(fused, ranked, golds)
                        if (alpha_best is None or
                                (metrics["recall@1"], metrics["mrr"]) >
                                (alpha_best["recall@1"], alpha_best["mrr"])):
                            alpha_best = {"alpha": alpha, "beta": beta_value, **metrics}
                controls[name] = {"temperature": temperature, **alpha_best} if name == "shuffled" else alpha_best
            report["controls"] = controls
            report["reference"] = reference
            suffix = "_sbert" if args.encoder == "sbert" else ""
            save_reference(REPO_ROOT / "experiments" / "checks" / f"three_channel_probe_reference_scores{suffix}.npz",
                           caption_ids, ranked, clip, objects, triples, best["alpha"], best["beta"], temperature)
    report["runtime_seconds"] = round(time.time() - started, 1)
    output_name = "three_channel_probe_sbert.json" if args.encoder == "sbert" else "three_channel_probe.json"
    output = REPO_ROOT / "experiments" / "checks" / output_name
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", output)


if __name__ == "__main__":
    main()
