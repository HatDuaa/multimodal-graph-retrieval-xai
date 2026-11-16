"""Gold-vs-other statistics and readable node-matching explanations for the graph matching probes (step 2)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.improve.graph_matching_probe import frozen_vectors, image_structure, query_structure, structures  # noqa: E402
from src.graph.graph_matching import fgw_score, hop_distance, shared_node_pairs  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "graph-matching"


def main() -> None:
    cfg = load_config(Path("configs/factual_v1.yaml"))
    dump_dir = resolve(load_config(), "processed") / "graph_matching"
    dump = np.load(dump_dir / "main_factual_r3_seed0.npz")
    caption_ids = [str(c) for c in dump["caption_ids"]]
    candidates, golds = dump["candidate_ids"], dump["gold"]
    raw_queries, raw_images = structures(cfg)
    queries = [query_structure(raw_queries[c]) for c in caption_ids]
    subset = np.array([len(shared_node_pairs(q["edges"])) > 0 for q in queries])
    stats = {}
    for method in ("srfgw", "consistency"):
        scores = np.load(dump_dir / f"scores_frozen_{method}.npy")
        gold_col = (candidates == golds[:, None]).argmax(1)
        gold_scores = scores[np.arange(len(scores)), gold_col]
        rows = subset & np.isfinite(gold_scores)
        others = np.where(candidates == golds[:, None], np.nan, scores)
        stats[method] = {"captions": int(rows.sum()), "gold_mean": float(np.nanmean(gold_scores[rows])),
                         "others_mean": float(np.nanmean(others[rows])),
                         "gold_above_others_mean": float(np.mean(gold_scores[rows] > np.nanmean(others[rows], 1)))}
        if method == "consistency":
            stats[method]["gold_fully_consistent"] = float(np.mean(gold_scores[rows] == 1.))
            stats[method]["others_fully_consistent"] = float(np.nanmean(others[rows] == 1.))
    images = {i: image_structure(g) for i, g in raw_images.items()}
    qn, _, inodes, _ = frozen_vectors(cfg, queries, images)
    texts = {c["caption_id"]: c["text"] for im in json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
             for c in im["captions"]}
    lines = ["# Ví dụ ghép node bằng FGW nửa nới lỏng (vector CLIP đóng băng, ảnh đúng)", "",
             "Mỗi node của câu được nối với node ảnh nhận nhiều khối lượng vận chuyển nhất (phần trăm khối lượng của node câu đó).", ""]
    shown = 0
    for k in np.flatnonzero(subset):
        if shown >= 8:
            break
        q, image = queries[k], images[int(golds[k])]
        if len(q["labels"]) < 3:
            continue
        score, plan = fgw_score(qn[k], hop_distance(len(q["labels"]), q["edges"]), inodes[int(golds[k])], image["dist"],
                                mode="semirelaxed")
        rows = []
        for i, label in enumerate(q["labels"]):
            j = int(plan[i].argmax())
            rows.append(f"{label} → {image['labels'][j]} ({100 * plan[i, j] / plan[i].sum():.0f}%)")
        lines.append(f"- `{caption_ids[k]}` {texts[caption_ids[k]]} — điểm −FGW {score:.3f}")
        lines.append("  - bộ ba câu: " + "; ".join(f"{q['labels'][s]} → {q['labels'][o]}" for s, o in q["edges"]))
        lines.append("  - ghép: " + "; ".join(rows))
        shown += 1
    (OUT / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "gold_vs_others.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
