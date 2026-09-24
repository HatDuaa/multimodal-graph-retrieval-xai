"""Package graph inputs without fitting or evaluating anything on validation/test.

Existing outputs are reused; --force writes a new version alongside them. Count can run first
and builds missing graph/query caches. Run --stage all --limit 200 for a bounded smoke run.
CLIP weights must already be cached: vector encoding forces offline mode.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.train_groups import make_train_groups  # noqa: E402
from src.eval.metrics import evaluate, gold_rank  # noqa: E402
from src.features.extract_clip import ClipEncoder, load_features, save_features  # noqa: E402
from src.graph.parse_query import parse_many  # noqa: E402
from src.graph.scene_graph import Vocab, iter_scene_graphs, load_raw  # noqa: E402
from src.graph.soft_match import PRONOUNS, instance_parts, query_instance_parts  # noqa: E402
from src.retrieval.faiss_index import CosineIndex  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

SPLITS = ("train", "val", "test")
KINDS = ("node", "rel", "triple")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versioned(path: Path) -> Path:
    """Choose an unused sibling without modifying the original file."""
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def write_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    print(f"expected {path}: {len(text.encode('utf-8'))} bytes", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {path}: {path.stat().st_size} bytes", flush=True)


def graph_parts(graph: dict, query: bool = False) -> dict[str, set[str]]:
    """Use precisely the probe's matching parts; retain only usable query relations."""
    nodes, triples = query_instance_parts(graph) if query else instance_parts(graph)
    relations = graph["relations"]
    if query:
        names = [obj["name"] for obj in graph["objects"]]
        relations = [rel for rel in relations
                     if names[rel["subject"]] not in PRONOUNS and names[rel["object"]] not in PRONOUNS]
    return {"node": set(nodes), "rel": {rel["predicate"].lower() if query else rel["predicate"]
                                      for rel in relations}, "triple": set(triples)}


def build_candidates(query_features: np.ndarray, caption_ids: list[str], golds: list[int],
                     pool_features: np.ndarray, pool_ids: list[int], group_index: int,
                     top_k: int = 50) -> dict[str, np.ndarray]:
    """Search only the supplied pool; gold ranks are 1-based (-1 when absent).

    Tiny smoke pools have trailing candidate -1 / score -inf padding to keep a fixed width.
    Padding is never passed to the metrics module.
    """
    if len(caption_ids) != len(golds) or len(caption_ids) != len(query_features):
        raise ValueError("caption, gold and query lengths differ")
    if not pool_ids or top_k <= 0:
        raise ValueError("pool and top_k must be nonempty/positive")
    scores, ranked = CosineIndex(pool_features, pool_ids).search(query_features, top_k)
    candidates = np.full((len(golds), top_k), -1, dtype=np.int64)
    clip_scores = np.full((len(golds), top_k), -np.inf, dtype=np.float32)
    width = min(top_k, len(pool_ids))
    candidates[:, :width] = np.asarray(ranked, dtype=np.int64).reshape(len(golds), width)
    clip_scores[:, :width] = scores
    ranks = [gold_rank(row, gold) for row, gold in zip(ranked, golds)]
    return {"caption_ids": np.asarray(caption_ids, dtype=str), "gold": np.asarray(golds, dtype=np.int64),
            "candidate_ids": candidates, "clip_scores": clip_scores,
            "gold_rank": np.asarray([rank if rank is not None else -1 for rank in ranks], dtype=np.int64),
            "group_index": np.full(len(golds), group_index, dtype=np.int64)}


def candidate_metrics(arrays: dict) -> dict:
    if not len(arrays["gold"]):
        return {"n_queries": 0, "recall@50": None, "mrr": None}
    ranked = [[int(item) for item in row if item != -1] for row in arrays["candidate_ids"]]
    return evaluate(ranked, arrays["gold"].tolist(), ks=(50,))


class Builder:
    def __init__(self, cfg: dict, workers: int = 16, limit: int | None = None,
                 dtype: str = "float32", force: bool = False, progress_batches: int = 20):
        self.cfg, self.workers, self.limit = cfg, workers, limit
        self.dtype, self.force, self.progress_batches = dtype, force, progress_batches
        self.suffix = f"_limit{limit}" if limit else ""
        self.outputs, self.counts, self.runtimes = {}, {}, {}
        self.images, self.paths, self.parts = {}, {}, None
        self.split_hashes = {}

    def path(self, folder: str, stem: str, extension: str) -> Path:
        directory = resolve(self.cfg, folder)
        if folder == "experiments":
            directory = directory / "level1" / "data_packaging"
        return directory / f"{stem}{self.suffix}{extension}"

    def target(self, path: Path, force: bool = False) -> tuple[Path, bool]:
        if path.exists():
            if force:
                other = versioned(path)
                print(f"preserve {path}; --force writes {other}", flush=True)
                return other, True
            print(f"skip existing {path}: {path.stat().st_size} bytes", flush=True)
            return path, False
        return path, True

    def record(self, path: Path, count=None) -> None:
        self.outputs[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if count is not None:
            self.counts[path.name] = count

    def split(self, split: str) -> list[dict]:
        if split not in self.images:
            path = resolve(self.cfg, "splits") / f"vg_coco_{split}.json"
            self.split_hashes[split] = sha256(path)
            images = json.loads(path.read_text(encoding="utf-8"))["images"]
            self.images[split] = images[:self.limit] if self.limit else images
        return self.images[split]

    def graphs(self, force: bool = False) -> Path:
        if "graphs" in self.paths:
            return self.paths["graphs"]
        path, create = self.target(self.path("graphs", "vg_coco_scene_graphs", ".jsonl"), force)
        self.paths["graphs"] = path
        if create:
            raw, vocab = load_raw(self.cfg), Vocab.from_config(self.cfg)
            rows = []
            for split in SPLITS:
                ids = [im["image_id"] for im in self.split(split)]
                for graph in iter_scene_graphs(raw, vocab, ids, vg150_only=False):
                    rows.append(json.dumps({**graph, "split": split}, ensure_ascii=False) + "\n")
            expected = sum(len(row.encode("utf-8")) for row in rows)
            print(f"expected {path}: {len(rows)} graphs, {expected} bytes", flush=True)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.writelines(rows)
        with path.open(encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        self.record(path, {"graphs": count})
        print(f"graphs: {count}, {path.stat().st_size} bytes", flush=True)
        return path

    def queries(self, force: bool = False) -> dict[str, Path]:
        if "queries" in self.paths:
            return self.paths["queries"]
        outputs = {}
        for split in SPLITS:
            path, create = self.target(self.path("processed", f"query_graphs_{split}", ".json"), force)
            captions = [cap for im in self.split(split) for cap in im["captions"]]
            if create:
                print(f"parse {split}: {len(captions)} captions", flush=True)
                parsed = parse_many([cap["text"] for cap in captions], self.workers)
                payload = {cap["caption_id"]: graph for cap, graph in zip(captions, parsed)}
                write_json(path, payload)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if set(payload) != {cap["caption_id"] for cap in captions}:
                    raise ValueError(f"query cache has different caption IDs: {path}")
            self.record(path, {"queries": len(payload)})
            print(f"{split}: {len(payload)} queries, {path.stat().st_size} bytes", flush=True)
            outputs[split] = path
        self.paths["queries"] = outputs
        return outputs

    def collect(self) -> dict:
        if self.parts is not None:
            return self.parts
        sets = {split: {source: {kind: set() for kind in KINDS} for source in ("images", "queries")}
                for split in SPLITS}
        with self.graphs().open(encoding="utf-8") as handle:
            for line in handle:
                graph = json.loads(line)
                for kind, values in graph_parts(graph).items():
                    sets[graph["split"]]["images"][kind].update(values)
        for split, path in self.queries().items():
            queries = json.loads(path.read_text(encoding="utf-8"))
            for graph in queries.values():
                for kind, values in graph_parts(graph, query=True).items():
                    sets[split]["queries"][kind].update(values)
        for split in SPLITS:
            sets[split]["all"] = {kind: sets[split]["images"][kind] | sets[split]["queries"][kind] for kind in KINDS}
        sets["total"] = {kind: set().union(*(sets[split]["all"][kind] for split in SPLITS)) for kind in KINDS}
        self.parts = sets
        return sets

    def count(self) -> Path:
        path, create = self.target(self.path("experiments", "part_counts", ".json"), self.force)
        if create:
            sets = self.collect()
            counts = {split: {source: {kind: len(values) for kind, values in parts.items()}
                              for source, parts in sets[split].items()} for split in SPLITS}
            total = {kind: len(values) for kind, values in sets["total"].items()}
            total_keys = sum(total.values())
            result = {"seed": self.cfg["seed"], "limit": self.limit, "splits": counts, "total": total,
                      "images": {split: len(self.split(split)) for split in SPLITS},
                      "queries": {split: sum(len(im["captions"]) for im in self.split(split)) for split in SPLITS},
                      "unseen_in_train": {split: {kind: len(sets[split]["all"][kind] - sets["train"]["all"][kind])
                                                  for kind in KINDS} for split in ("val", "test")},
                      "total_keys": total_keys,
                      "projected_vector_bytes": {"float32": total_keys * 512 * 4, "float16": total_keys * 512 * 2},
                      "size_note": "Matrix payload only; NPY header and IDs JSON are additional."}
            write_json(path, result)
        else:
            result = json.loads(path.read_text(encoding="utf-8"))
        self.record(path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return path

    def vectors(self) -> Path:
        stem = self.path("features", "vg_coco_graph_parts", "")
        npy, meta = stem.with_suffix(".npy"), stem.with_suffix(".ids.json")
        if npy.exists() or meta.exists():
            if not self.force:
                if not (npy.exists() and meta.exists()):
                    raise FileExistsError(f"incomplete feature pair: {stem}; use --force to write beside it")
                vectors, keys, encoder = load_features(stem)
                if vectors.dtype != np.dtype(self.dtype):
                    raise ValueError(f"existing dtype={vectors.dtype}; requested {self.dtype}; use --force")
                expected = f"{self.cfg['clip']['model']}/{self.cfg['clip']['pretrained']}"
                if encoder != expected:
                    raise ValueError("existing vector encoder differs from configuration")
                self.record(npy, {"vectors": len(keys), "dtype": str(vectors.dtype)})
                self.record(meta)
                print(f"skip existing vectors: {len(keys)}, {npy.stat().st_size} bytes", flush=True)
                return npy
            index = 2
            while True:
                candidate = stem.with_name(f"{stem.name}_v{index}")
                if not candidate.with_suffix(".npy").exists() and not candidate.with_suffix(".ids.json").exists():
                    stem = candidate
                    break
                index += 1
            npy, meta = stem.with_suffix(".npy"), stem.with_suffix(".ids.json")
        sets = self.collect()["total"]
        keys = [f"{kind}:{text}" for kind in KINDS for text in sorted(sets[kind])]
        expected = len(keys) * 512 * np.dtype(self.dtype).itemsize
        print(f"expected vectors: {len(keys)} x 512 {self.dtype}, {expected} bytes plus NPY header; "
              f"IDs JSON about {len(json.dumps(keys).encode('utf-8'))} bytes", flush=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        encoder = ClipEncoder.from_config(self.cfg)
        vectors = np.empty((len(keys), 512), dtype=self.dtype)
        old_stem = resolve(self.cfg, "features") / "vg_coco_val_graph_parts"
        sampled, old_vectors, old_rows = set(), None, {}
        if old_stem.with_suffix(".npy").exists():
            old_vectors, old_keys, old_encoder = load_features(old_stem)
            if old_encoder != encoder.name:
                raise ValueError("reference vector encoder differs from current encoder")
            old_rows = {key: row for row, key in enumerate(old_keys)}
            common = sorted(set(keys) & set(old_rows))
            sampled = set(random.Random(self.cfg["seed"]).sample(common, min(1000, len(common))))
        max_error, compared = 0.0, 0
        batches = math.ceil(len(keys) / encoder.batch_size)
        for start in range(0, len(keys), encoder.batch_size):
            batch_keys = keys[start:start + encoder.batch_size]
            texts = [self.cfg["clip"]["concept_prompt"].format(key.split(":", 1)[1])
                     if key.startswith("node:") else key.split(":", 1)[1] for key in batch_keys]
            encoded = encoder.encode_texts(texts)
            for row, key in enumerate(batch_keys):
                if key in sampled:
                    error = float(np.max(np.abs(encoded[row] - old_vectors[old_rows[key]])))
                    max_error = max(max_error, error)
                    compared += 1
                    if error > 1e-4:
                        raise AssertionError(f"reference vector mismatch: {key}, max abs={error}")
            vectors[start:start + len(batch_keys)] = encoded
            batch = start // encoder.batch_size + 1
            if batch % self.progress_batches == 0 or batch == batches:
                print(f"encoded batch {batch}/{batches}", flush=True)
        save_features(stem, vectors, keys, encoder.name, dtype=np.dtype(self.dtype))
        self.record(npy, {"vectors": len(keys), "dtype": self.dtype, "reference_sample": compared,
                          "reference_max_abs_error_float32": max_error})
        self.record(meta)
        print(f"vectors: {len(keys)}, {npy.stat().st_size} bytes; reference sample={compared}, error={max_error}", flush=True)
        return npy

    def groups(self) -> dict:
        ids = [im["image_id"] for im in self.split("train")]
        dev_size = 0
        expected = make_train_groups(ids, self.cfg["seed"], dev_size=dev_size)
        expected.update({"train_split_sha256": self.split_hashes["train"], "limit": self.limit,
                         "dev_size": dev_size, "group_size": 2140})
        path, create = self.target(self.path("splits", "vg_coco_train_groups", ".json"), self.force)
        if not create:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if any(saved.get(key) != expected[key] for key in ("seed", "dev", "groups", "train_split_sha256")):
                path = versioned(path)
                create = True
                print(f"existing groups use an older split policy; writing {path}", flush=True)
        if create:
            write_json(path, expected)
        else:
            for key in ("seed", "dev", "groups", "train_split_sha256"):
                if saved[key] != expected[key]:
                    raise ValueError(f"group provenance mismatch in {path}: {key}")
        self.record(path, {"dev": len(expected["dev"]), "groups": list(map(len, expected["groups"]))})
        return expected

    def candidates(self) -> None:
        targets = {name: self.target(self.path("processed", f"candidates_{name}", ".npz"), self.force)
                   for name in ("train", "val")}
        groups = self.groups()
        images = {im["image_id"]: im for im in self.split("train")}
        if any(create for _, create in targets.values()):
            feats = resolve(self.cfg, "features")
            image_features, image_ids, image_encoder = load_features(feats / "vg_coco_image")
            caption_features, caption_ids, caption_encoder = load_features(feats / "vg_coco_caption")
            if image_encoder != caption_encoder:
                raise ValueError("image/caption encoders differ")
            image_row = {item: row for row, item in enumerate(image_ids)}
            caption_row = {item: row for row, item in enumerate(caption_ids)}
        for name, (path, create) in targets.items():
            if create:
                if name == "val":
                    self._candidates_val(path, create)
                    continue
                pools = list(enumerate(groups["groups"]))
                chunks = []
                for index, pool in pools:
                    if not pool:
                        continue
                    queries = [(cap["caption_id"], image_id) for image_id in pool for cap in images[image_id]["captions"]]
                    ids, golds = zip(*queries)
                    chunk = build_candidates(caption_features[[caption_row[item] for item in ids]], list(ids), list(golds),
                                             image_features[[image_row[item] for item in pool]], pool, index)
                    chunks.append(chunk)
                if not chunks:
                    raise ValueError("no candidate queries; use --limit >= 2")
                arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
                arrays["group"] = arrays["group_index"].copy()
                # Store the metrics inside the NPZ as JSON so a skipped stage retains its coverage record.
                per_group = {str(index): candidate_metrics({key: values[arrays["group_index"] == index]
                                                            for key, values in arrays.items()})
                             for index in np.unique(arrays["group_index"])}
                coverage = {"overall": candidate_metrics(arrays), "per_group": per_group}
                arrays["coverage_json"] = np.asarray(json.dumps(coverage))
                arrays["train_split_sha256"] = np.asarray(self.split_hashes["train"])
                arrays["seed"] = np.asarray(self.cfg["seed"])
                arrays["limit"] = np.asarray(self.limit or 0)
                expected_bytes = sum(values.nbytes for values in arrays.values()) + 4096
                print(f"expected {path}: about {expected_bytes} bytes", flush=True)
                with path.open("xb") as handle:
                    np.savez(handle, **arrays)
            else:
                if name == "val":
                    self._candidates_val(path, create)
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["train_split_sha256"]) != self.split_hashes["train"] or int(saved["seed"]) != self.cfg["seed"]:
                        raise ValueError(f"candidate provenance mismatch: {path}")
                    coverage = json.loads(str(saved["coverage_json"]))
            self.record(path, coverage)
            print(f"{name}: {json.dumps(coverage)}, {path.stat().st_size} bytes", flush=True)

    def _candidates_val(self, path: Path, create: bool) -> None:
        """Package the already-computed baseline val top-50 without recomputing it."""
        baseline = resolve(self.cfg, "experiments") / "level1" / "clip_baseline" / "top50_val.json"
        if not baseline.exists():
            raise FileNotFoundError(f"missing baseline candidates: {baseline}")
        rows = json.loads(baseline.read_text(encoding="utf-8"))
        selected = {cap["caption_id"] for image in self.split("val") for cap in image["captions"]}
        rows = [row for row in rows if row["caption_id"] in selected]
        expected_n = sum(len(image["captions"]) for image in self.split("val"))
        if len(rows) != expected_n:
            raise AssertionError(f"baseline val rows {len(rows)} != expected {expected_n}")
        candidate_ids = np.asarray([row["ranked"] for row in rows], dtype=np.int64)
        clip_scores = np.asarray([row["scores"] for row in rows], dtype=np.float32)
        if candidate_ids.shape != (expected_n, 50) or clip_scores.shape != (expected_n, 50):
            raise AssertionError("baseline val top-50 has unexpected shapes")
        gold_by_caption = {cap["caption_id"]: image["image_id"]
                           for image in self.split("val") for cap in image["captions"]}
        golds = np.asarray([gold_by_caption[row["caption_id"]] for row in rows], dtype=np.int64)
        ranks = [gold_rank(ids.tolist(), gold) for ids, gold in zip(candidate_ids, golds)]
        arrays = {"caption_ids": np.asarray([row["caption_id"] for row in rows], dtype=str),
                  "gold": golds, "candidate_ids": candidate_ids, "clip_scores": clip_scores,
                  "gold_rank": np.asarray([rank if rank is not None else -1 for rank in ranks], dtype=np.int64),
                  "group_index": np.full(expected_n, -1, dtype=np.int64),
                  "group": np.full(expected_n, -1, dtype=np.int64),
                  "coverage_json": np.asarray(json.dumps({"overall": candidate_metrics({
                      key: value for key, value in {"gold": golds, "candidate_ids": candidate_ids}.items()})})),
                  "baseline_sha256": np.asarray(sha256(baseline)),
                  "train_split_sha256": np.asarray(self.split_hashes["train"]),
                  "seed": np.asarray(self.cfg["seed"]),
                  "limit": np.asarray(self.limit or 0)}
        coverage = json.loads(str(arrays["coverage_json"]))
        if create:
            expected_bytes = sum(value.nbytes for value in arrays.values()) + 4096
            print(f"expected {path}: about {expected_bytes} bytes", flush=True)
            with path.open("xb") as handle:
                np.savez(handle, **arrays)
        else:
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["baseline_sha256"]) != sha256(baseline):
                    raise ValueError(f"baseline provenance mismatch: {path}")
        self.record(path, coverage)
        print(f"val: {json.dumps(coverage)}, {path.stat().st_size} bytes", flush=True)

    def summary(self, stages: list[str], runtime: float) -> None:
        for split in SPLITS:
            self.split(split)
        path = self.path("experiments", "summary", ".json")
        if path.exists():
            path = versioned(path)
        payload = {"date": time.strftime("%Y-%m-%d"), "config": self.cfg, "seed": self.cfg["seed"],
                   "seeds": self.cfg["seeds"], "limit": self.limit, "dtype": self.dtype, "workers": self.workers,
                   "stages": stages, "split_sha256": self.split_hashes, "outputs": self.outputs,
                   "counts": self.counts, "runtimes_seconds": self.runtimes, "runtime_seconds": runtime,
                   "leakage": {"groups_from_train_only": True, "has_dev_split": False,
                               "val_used_for_early_stopping_and_model_selection": True,
                               "test_untouched_by_this_task": True,
                               "no_cross_image_graph_statistics": True,
                               "vocabulary_union_is_only_a_frozen_text_lookup": True,
                               "test_candidates_computed": False, "test_metrics_computed": False,
                               "note": "Per-image annotations only; no edges, fitted statistics or filters cross images. "
                                       "Counts/union across splits are storage bookkeeping only."},
                   "smoke_policy": "First N images per split; no dev split; "
                                   "small pools pad candidate=-1, clip_score=-inf; gold ranks are 1-based."}
        write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("graphs", "queries", "count", "vectors", "candidates", "all"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--progress-batches", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="write versioned siblings; never replace existing files")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0 or args.progress_batches <= 0:
        parser.error("workers and progress-batches must be positive")
    builder = Builder(load_config(), args.workers, args.limit, args.dtype, args.force, args.progress_batches)
    stages = ["graphs", "queries", "count", "vectors", "candidates"] if args.stage == "all" else [args.stage]
    started = time.monotonic()
    for stage in stages:
        begin = time.monotonic()
        if stage in ("graphs", "queries"):
            getattr(builder, stage)(force=args.force)
        else:
            getattr(builder, stage)()
        builder.runtimes[stage] = time.monotonic() - begin
    builder.summary(stages, time.monotonic() - started)


if __name__ == "__main__":
    main()
