"""Where the GAT of a full-graph run looks: attention by edge kind on val image graphs, and drawn examples.

For each run (val only): every val image graph is encoded with the run's structure, and both GATv2 layers return
their attention alpha (softmax over each target node's incoming edges, self-loop included; mean over the 4 heads).
- Per layer, over the real (non-global) target nodes: the share of attention mass on labelled edges, completion
  edges (k-nearest and bridges, predicate [none]), global-node edges and the self-loop; and the "lift" of each
  kind, alpha x in-degree (1 = a uniform share).
- Completion edges by (source label, target label), pairs seen >= 30 times: mean lift, highest and lowest.
- Examples (first run only): the gold image of 8 val captions whose gold the run ranks first and 4 whose gold it
  ranks 2-10, drawn at random with seed 0 among those with completion edges; the 6 completion edges with the
  highest alpha in layer 1 are drawn as arrows source -> target with alpha, other boxes grey.
Writes attention.json, attention_examples.md and img/attention_<caption>.png under experiments/improve/full-graph/.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import EDGE_COMPLETION, EDGE_GLOBAL, EDGE_LABELLED, GraphStore  # noqa: E402
from src.train_graph_reranker import load_run_model  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

OUT = REPO_ROOT / "experiments" / "improve" / "full-graph"
SELF = 3
KINDS = {EDGE_LABELLED: "labelled", EDGE_COMPLETION: "completion", EDGE_GLOBAL: "global", SELF: "self"}
PANEL = 520


@torch.no_grad()
def image_attention(model, store, image_ids, device, chunk=256):
    """Per image: local (source, target) pairs, kinds and per-layer mean-head alpha of every GAT edge."""
    table = store.part_table(device)
    out = {}
    for start in range(0, len(image_ids), chunk):
        ids = image_ids[start:start + chunk]
        index = store.image_index.select([store.image_position[i] for i in ids])
        batch = index.to_batch(table, device)
        model.attention = []
        model.record_attention = True
        model.encode(batch)
        model.record_attention = False
        n_edges = batch.edge_index.shape[1]
        kind = batch.edge_kind if batch.edge_kind is not None else torch.zeros(n_edges, dtype=torch.long, device=device)
        layers = []
        for pairs, alpha in model.attention:
            assert torch.equal(pairs[:, :n_edges], batch.edge_index)
            assert torch.equal(pairs[0, n_edges:], pairs[1, n_edges:])  # GATv2 appends one self-loop per node
            layers.append(alpha.mean(-1))
        pairs = model.attention[0][0]
        kinds = torch.cat((kind, torch.full((pairs.shape[1] - n_edges,), SELF, device=device)))
        node_start = np.cumsum(index.node_counts) - index.node_counts
        graph_of_node = np.repeat(np.arange(len(ids)), index.node_counts)
        pairs, kinds = pairs.cpu().numpy(), kinds.cpu().numpy()
        layers = [a.cpu().numpy() for a in layers]
        global_mask = batch.global_mask.cpu().numpy() if batch.global_mask is not None else np.zeros(len(graph_of_node), bool)
        owner = graph_of_node[pairs[1]]
        for k, image_id in enumerate(ids):
            keep = owner == k
            out[image_id] = {"pairs": pairs[:, keep] - node_start[k], "kinds": kinds[keep],
                             "alpha": [a[keep] for a in layers],
                             "global": global_mask[node_start[k]:node_start[k] + index.node_counts[k]]}
    return out


def kind_statistics(attention, store):
    layers = len(next(iter(attention.values()))["alpha"])
    share = [collections.defaultdict(list) for _ in range(layers)]
    lift = [collections.defaultdict(list) for _ in range(layers)]
    pair_lift = [collections.defaultdict(list) for _ in range(layers)]
    for image_id, item in attention.items():
        names = [o["name"] for o in store.graphs[image_id]["objects"]]
        src, dst = item["pairs"]
        real = ~item["global"][dst]
        degree = np.bincount(dst, minlength=len(item["global"]))
        for layer, alpha in enumerate(item["alpha"]):
            for kind, name in KINDS.items():
                mass = np.bincount(dst, weights=alpha * (item["kinds"] == kind), minlength=len(item["global"]))
                targets = np.flatnonzero(~item["global"] & (degree > 0))
                share[layer][name].extend(mass[targets].tolist())
                chosen = (item["kinds"] == kind) & real
                lift[layer][name].extend((alpha[chosen] * degree[dst[chosen]]).tolist())
            chosen = np.flatnonzero((item["kinds"] == EDGE_COMPLETION) & real)
            for e in chosen:
                pair_lift[layer][(names[src[e]], names[dst[e]])].append(float(alpha[e] * degree[dst[e]]))
    out = []
    for layer in range(layers):
        pairs = {f"{a} -> {b}": float(np.mean(v)) for (a, b), v in pair_lift[layer].items() if len(v) >= 30}
        ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
        out.append({"mass_share": {k: float(np.mean(v)) for k, v in share[layer].items()},
                    "lift": {k: float(np.mean(v)) if v else None for k, v in lift[layer].items()},
                    "completion_pairs_seen_30": len(pairs), "completion_pairs_highest_lift": ordered[:15],
                    "completion_pairs_lowest_lift": ordered[-15:][::-1]})
    return out


@torch.no_grad()
def gold_ranks(model, store, device):
    arrays = store.candidates("val")
    caption_ids = [str(c) for c in arrays["caption_ids"]]
    ranks = {}
    for start in range(0, len(caption_ids), 256):
        batch = store.batch(caption_ids[start:start + 256], device)
        scores, _ = model(batch)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True).cpu().numpy()
        ranked = np.take_along_axis(batch["candidate_ids"], order, axis=1)
        for cid, row, gold in zip(caption_ids[start:start + 256], ranked, batch["gold"]):
            hit = np.flatnonzero(row == gold)
            ranks[cid] = int(hit[0]) + 1 if len(hit) else None
    return ranks


def draw(path, graph, item, top, title):
    image = Image.open(path).convert("RGB")
    scale = PANEL / image.width
    canvas = image.resize((PANEL, int(image.height * scale)))
    draw = ImageDraw.Draw(canvas)
    objects = graph["objects"]
    ends = {int(item["pairs"][0][e]) for e in top} | {int(item["pairs"][1][e]) for e in top}
    centre = lambda b: ((b[0] + b[2] / 2) * scale, (b[1] + b[3] / 2) * scale)  # noqa: E731
    for i, o in enumerate(objects):
        x, y, w, h = [v * scale for v in o["box"]]
        draw.rectangle([x, y, x + w, y + h], outline="#e6194b" if i in ends else "#bbbbbb", width=2 if i in ends else 1)
        if i in ends:
            draw.text((x + 2, y + 1), o["name"], fill="#e6194b")
    for rank, e in enumerate(top):
        s, t = int(item["pairs"][0][e]), int(item["pairs"][1][e])
        (x0, y0), (x1, y1) = centre(objects[s]["box"]), centre(objects[t]["box"])
        draw.line([x0, y0, x1, y1], fill="#4363d8", width=3)
        draw.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill="#4363d8")  # the dot marks the target (receiving) node
        draw.text(((x0 + x1) / 2 + 3, (y0 + y1) / 2), f"{rank + 1}: {item['alpha'][0][e]:.2f}", fill="#000000")
    out = Image.new("RGB", (PANEL, canvas.height + 16), "white")
    out.paste(canvas, (0, 16))
    ImageDraw.Draw(out).text((2, 2), title, fill="black")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out_dir = args.out
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(REPO_ROOT / "configs" / "merge_v1.yaml")
    store = GraphStore(cfg, splits=("val",))
    image_ids = sorted(store.graphs)
    result = {"split": "val", "images": len(image_ids), "runs": args.runs, "per_run": {}}
    examples = None
    for k, run in enumerate(args.runs):
        model = load_run_model(Path(run), device, store=store)[0]
        store.precompute()
        attention = image_attention(model, store, image_ids, device)
        result["per_run"][run] = kind_statistics(attention, store)
        print(run, json.dumps([layer["mass_share"] for layer in result["per_run"][run]]), flush=True)
        if k == 0:
            examples = (model, attention)
    layers = len(result["per_run"][args.runs[0]])
    result["aggregate"] = [{field: {kind: {"mean": float(np.mean([result["per_run"][r][layer][field][kind] for r in args.runs])),
                                           "std": float(np.std([result["per_run"][r][layer][field][kind] for r in args.runs], ddof=1))
                                           if len(args.runs) > 1 else 0.}
                                    for kind in KINDS.values() if result["per_run"][args.runs[0]][layer][field][kind] is not None}
                            for field in ("mass_share", "lift")} for layer in range(layers)]
    model, attention = examples
    ranks = gold_ranks(model, store, device)
    arrays = store.candidates("val")
    gold_of = {str(c): int(g) for c, g in zip(arrays["caption_ids"], arrays["gold"])}
    with_completion = [c for c in sorted(ranks) if (attention[gold_of[c]]["kinds"] == EDGE_COMPLETION).any()]
    rng = random.Random(0)
    chosen = rng.sample([c for c in with_completion if ranks[c] == 1], 8) + \
        rng.sample([c for c in with_completion if ranks[c] and 2 <= ranks[c] <= 10], 4)
    texts = {c["caption_id"]: c["text"] for im in json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]
             for c in im["captions"]}
    files = {im["image_id"]: im["file_name"] for im in json.loads((resolve(cfg, "splits") / "vg_coco_val.json").read_text(encoding="utf-8"))["images"]}
    folder = resolve(cfg, "raw") / "images"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    lines = [f"# Attention của GAT trên cạnh tự thêm — ví dụ val ({Path(args.runs[0]).name})", "",
             "Mỗi ảnh là ảnh đúng của một câu val. Mũi tên xanh là 6 cạnh tự thêm (vị từ `[none]`) có attention lớp 1 "
             "cao nhất, đi từ node nguồn tới node nhận (chấm tròn); số là thứ hạng và alpha (trung bình 4 head; alpha "
             "là softmax trên các cạnh đi vào node nhận, kể cả self-loop). Khung đỏ là hai đầu của các cạnh đó, khung xám "
             "là node khác. Chọn ngẫu nhiên (seed 0): 8 câu mô hình xếp ảnh đúng hạng 1, 4 câu xếp hạng 2–10.", ""]
    result["examples"] = []
    for cid in chosen:
        image_id = gold_of[cid]
        item, graph = attention[image_id], store.graphs[image_id]
        completion = np.flatnonzero(item["kinds"] == EDGE_COMPLETION)
        top = completion[np.argsort(-item["alpha"][0][completion], kind="stable")[:6]].tolist()
        name = f"attention_{cid}.png"
        draw(folder / files[image_id], graph, item, top, f"caption {cid}, gold rank {ranks[cid]}").save(out_dir / "img" / name)
        rows = []
        for rank, e in enumerate(top):
            s, t = int(item["pairs"][0][e]), int(item["pairs"][1][e])
            degree = int((item["pairs"][1] == t).sum())
            rows.append({"rank": rank + 1, "source": f"{graph['objects'][s]['name']}#{s}",
                         "target": f"{graph['objects'][t]['name']}#{t}",
                         "alpha_layer1": float(item["alpha"][0][e]), "alpha_layer2": float(item["alpha"][1][e]),
                         "target_in_degree": degree})
        result["examples"].append({"caption_id": cid, "image_id": image_id, "gold_rank": ranks[cid], "top_completion_edges": rows})
        lines += [f"## {cid} — hạng ảnh đúng {ranks[cid]}", "", f"> {texts.get(cid, '')}", "", f"![{cid}](img/{name})", "",
                  "| # | nguồn → nhận | alpha lớp 1 | alpha lớp 2 | bậc vào của node nhận |", "|---|---|---|---|---|"]
        lines += [f"| {r['rank']} | {r['source']} → {r['target']} | {r['alpha_layer1']:.3f} | {r['alpha_layer2']:.3f} | "
                  f"{r['target_in_degree']} |" for r in rows] + [""]
    (out_dir / "attention.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "attention_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
