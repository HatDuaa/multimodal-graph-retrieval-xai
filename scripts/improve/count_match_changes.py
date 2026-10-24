"""How often does the count table g change which image node a caption node is matched to? (experiment 5, demo)

For every val caption whose gold image is in its top-50, and every scorable caption node with a known count class,
the object channel's best match in the gold image is taken twice with the same trained run: by sim_ij (g left out)
and by sim_ij + g[class_i, bucket_j] (what the model ranks with). A change is "good" when only the new match carries
the caption node's label (label equal after VG aliases, or frozen-CLIP node cosine >= 0.955, as in the graph-cleaning
audit), "bad" when only the old one does, else "same label status". Counts per run, mean +- std over runs, and drawn
examples (first run, random with seed 0) with both matches, their count buckets and similarities.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS, compatible  # noqa: E402
from src.graph.scene_graph import Vocab  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

THRESHOLD = .955


@torch.no_grad()
def changes(model, store, vocab):
    vector = lambda label: store.part_vectors[store.part_row[f"node:{label}"]]  # noqa: E731
    same = {}

    def label_match(q, i):
        if (q, i) not in same:
            known = f"node:{q}" in store.part_row and f"node:{i}" in store.part_row
            same[(q, i)] = vocab.canon_object(q) == vocab.canon_object(i) or (known and float(vector(q) @ vector(i)) >= THRESHOLD)
        return same[(q, i)]

    device = next(model.parameters()).device
    caption_ids = [str(c) for c in store.candidates("val")["caption_ids"]]
    rows = []
    for start in range(0, len(caption_ids), 256):
        ids = caption_ids[start:start + 256]
        batch = store.batch(ids, device)
        _, detail = model(batch, return_encoded=True)
        q, qm = detail["encoded_query"][:2]
        p, pm = detail["encoded_images"][:2]
        table = detail["count_table"]
        for b, cid in enumerate(ids):
            target = int(batch["target"][b])
            if target < 0:
                continue
            row = int(batch["candidate_index"][b, target])
            classes = detail["count_query"][b][qm[b]]
            if not (classes > 0).any() or not pm[row].any():  # no known count, or a gold image without nodes
                continue
            buckets = detail["count_images"][row][pm[row]]
            sims = q[b][qm[b]] @ p[row][pm[row]].T
            biased = sims + table[classes[:, None], buckets[None, :]]
            before, after = sims.argmax(-1).tolist(), biased.argmax(-1).tolist()
            parts = GraphStore._graph_parts(store.queries[cid], query=True)
            q_names = [n for n, keep in zip(parts["node_labels"], parts["part_mask"]) if keep]
            image_id = int(batch["image_ids"][row])
            i_names = GraphStore._graph_parts(store.graphs[image_id])["node_labels"]
            for k, cls in enumerate(classes.tolist()):
                if not cls:
                    continue
                old, new = before[k], after[k]
                old_ok, new_ok = label_match(q_names[k], i_names[old]), label_match(q_names[k], i_names[new])
                kind = ("unchanged" if old == new else "good" if new_ok and not old_ok else "bad" if old_ok and not new_ok
                        else "same_label_status")
                rows.append({"caption_id": cid, "image_id": image_id, "query_node": q_names[k], "query_count": CAPTION_CLASSES[cls],
                             "kind": kind, "without_g": {"node": i_names[old], "count": IMAGE_BUCKETS[int(buckets[old])],
                                                          "sim": float(sims[k, old]), "label_match": old_ok,
                                                          "count_compatible": compatible(cls, int(buckets[old]))},
                             "with_g": {"node": i_names[new], "count": IMAGE_BUCKETS[int(buckets[new])],
                                        "sim_plus_g": float(biased[k, new]), "g": float(biased[k, new] - sims[k, new]),
                                        "label_match": new_ok, "count_compatible": compatible(cls, int(buckets[new]))}})
    return rows


def summary(rows):
    kinds = collections.Counter(r["kind"] for r in rows)
    n = len(rows)
    changed = n - kinds["unchanged"]
    return {"caption_nodes": n, "changed": changed, "changed_share": changed / max(n, 1),
            "good": kinds["good"], "bad": kinds["bad"], "same_label_status": kinds["same_label_status"],
            "label_match_without_g": sum(r["without_g"]["label_match"] for r in rows) / max(n, 1),
            "label_match_with_g": sum(r["with_g"]["label_match"] for r in rows) / max(n, 1),
            "count_compatible_without_g": sum(bool(r["without_g"]["count_compatible"]) for r in rows) / max(n, 1),
            "count_compatible_with_g": sum(bool(r["with_g"]["count_compatible"]) for r in rows) / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads((Path(args.runs[0]) / "config.json").read_text(encoding="utf-8"))["args"]["config"]
    cfg = load_config(REPO_ROOT / config)
    store = GraphStore(cfg, splits=("val",))
    vocab = Vocab.from_config(cfg)
    result, first = {"split": "val", "runs": args.runs, "per_run": {}}, None
    for run in args.runs:
        rows = changes(load_run_model(Path(run), device, store=store)[0], store, vocab)
        result["per_run"][run] = summary(rows)
        print(run, json.dumps(result["per_run"][run]), flush=True)
        first = first if first is not None else rows
    keys = [k for k, v in result["per_run"][args.runs[0]].items() if isinstance(v, (int, float))]
    result["aggregate"] = {k: {"mean": float(np.mean([result["per_run"][r][k] for r in args.runs])),
                               "std": float(np.std([result["per_run"][r][k] for r in args.runs], ddof=1)) if len(args.runs) > 1 else 0.}
                           for k in keys}
    texts = {c["caption_id"]: c["text"] for im in json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
             for c in im["captions"]}
    rng = random.Random(0)
    examples = {kind: rng.sample([r for r in first if r["kind"] == kind], min(args.examples // 2, sum(r["kind"] == kind for r in first)))
                for kind in ("bad", "good")}
    result["examples"] = examples
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "count_match_changes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    titles = {"bad": "g kéo sang node sai nhãn (cũ đúng nhãn, mới sai nhãn)", "good": "g kéo sang node đúng nhãn (cũ sai nhãn, mới đúng nhãn)"}
    lines = [f"# Ca g làm đổi node được ghép ({Path(args.runs[0]).name}, ảnh đúng của câu val)", "",
             "Mỗi dòng: node câu (lớp số lượng) và node ảnh được ghép khi bỏ g (theo sim_ij) và khi có g (theo sim_ij + g), "
             "kèm nhóm số lượng của node ảnh. Chọn ngẫu nhiên (seed 0).", ""]
    for kind, items in examples.items():
        lines += [f"## {titles[kind]}", ""]
        for r in items:
            a, b = r["without_g"], r["with_g"]
            lines.append(f"- {r['caption_id']} \"{texts.get(r['caption_id'], '')}\": câu {r['query_count']} {r['query_node']} — "
                         f"không g: {a['count']} {a['node']} (sim {a['sim']:.3f}) → có g: {b['count']} {b['node']} "
                         f"(sim + g {b['sim_plus_g']:.3f}, g {b['g']:+.3f})")
        lines.append("")
    (args.out_dir / "count_match_changes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
