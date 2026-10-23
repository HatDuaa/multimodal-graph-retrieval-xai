"""Explanation lines with the count term g for a few val captions with an explicit count (experiment 4, demo).

Random val captions (seed 0) whose graph has a node with an explicit count class, gold image in the top-50; the
explanation of the gold image by a count-table run (src/explain/graph_explainer.py score, object channel) is printed as
"caption: <class> <label> — image: <bucket> <label> -> g = +x.xx", with sim_ij and the contribution.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.explain.graph_explainer import score  # noqa: E402
from src.graph.counts import CAPTION_CLASSES  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

EXPLICIT = {CAPTION_CLASSES.index(c) for c in ("1", "2", "3-4", ">=5")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(REPO_ROOT / "configs" / "group_v1.yaml")
    store = GraphStore(cfg, splits=("val",))
    model = load_run_model(args.run, device, store=store)[0]
    arrays = store.candidates("val")
    gold = {str(c): (int(g), int(r)) for c, g, r in zip(arrays["caption_ids"], arrays["gold"], arrays["gold_rank"])}
    texts = {c["caption_id"]: c["text"] for im in json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
             for c in im["captions"]}
    pool = sorted(c for c, g in store.queries.items() if gold[c][1] > 0
                  and any(o.get("count_class", 0) in EXPLICIT for o in g["objects"]))
    lines = [f"# Ví dụ giải thích có số lượng ({args.run.name}, val)", "",
             "Mỗi câu val chọn ngẫu nhiên (seed 0) trong các câu có số nêu rõ. Dòng là node vật thể của câu và node ảnh "
             "giống nó nhất trong ảnh đúng: số lượng hai bên, g cộng vào độ giống, sim_ij (đã gồm g) và đóng góp.", ""]
    for cid in random.Random(0).sample(pool, args.n):
        image_id = gold[cid][0]
        result = score(model, store, cid, image_id)
        query, image = store.queries[cid], store.graphs[image_id]
        lines += [f"## {cid}: {texts[cid]}", "", f"Ảnh đúng {image_id}, điểm cuối {result['score']:.4f}.", ""]
        for row in result["objects"]:
            q = query["objects"][row["query_index"]]["name"]
            i = image["objects"][row["image_index"]]["name"]
            lines.append(f"- câu: {row['query_count']} {q} — ảnh: {row['image_count']} {i} → g = {row['count_bias']:+.3f}; "
                         f"sim_ij {row['sim_ij']:.3f}, đóng góp {row['contribution']:+.4f}")
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
