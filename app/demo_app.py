"""Demo: type a query, see the top-k images and why each one was returned.

Runs fully offline once features and the test/val images are on disk:
    python app/demo_app.py [--split test] [--port 7860]
then open http://localhost:7860

The page only calls SearchService.search(); ranking modes added to the service (for example
CLIP + graph) appear in the "Chế độ xếp hạng" selector automatically.
"""
import argparse
import random
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.service.search_service import Hit, SearchService  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

MODE_LABELS = {"clip": "CLIP thuần", "clip_graph": "CLIP + Đồ thị"}

# The page fills the browser window exactly: only the result grid and the detail panel scroll.
CSS = """
html, body { height: 100%; margin: 0; overflow: hidden; }
.gradio-container { height: 100vh !important; max-width: 100% !important; padding: 10px 16px !important; overflow: hidden; }
footer { display: none !important; }
#title h1 { font-size: 1.25rem; margin: 0; }
#title p { margin: 0; }
#note p { margin: 0; }
#results { flex: 1 1 0 !important; min-height: 0 !important; flex-wrap: nowrap !important; }
#results > * { height: 100% !important; min-height: 0 !important; }
#gallery { height: 100% !important; }
/* the result grid keeps its normal square cells and scrolls inside its box; images are shown whole */
#gallery .gallery-container { height: 100% !important; display: flex; flex-direction: column; }
#gallery .grid-wrap { height: 100% !important; max-height: none !important; flex: 1 1 0; min-height: 0; overflow-y: auto !important; }
#gallery .thumbnail-item { background: var(--neutral-100, #f3f4f6); }
#detail { height: 100% !important; overflow-y: auto; padding-right: 6px; }
"""


def graph_explanation(info: dict) -> list[str]:
    """Matched part pairs exactly as the reranker scored them (src/explain/graph_explainer.matched_pairs)."""
    a, b, c = info["weights_abc"]
    lines = [f"- Trọng số của câu này: CLIP **{a:.0%}**, vật thể **{b:.0%}**, bộ ba **{c:.0%}**",
             "- Đồ thị câu: " + ", ".join(f"`{o}`" for o in info["query_graph"]["objects"])
             + ("; " + ", ".join(f"`{h} –{r}→ {t}`" for h, r, t in info["query_graph"]["relations"])
                if info["query_graph"]["relations"] else ""),
             "", "| Phần của câu | Phần khớp trong ảnh | w | a | sim | đóng góp |", "|---|---|---|---|---|---|"]

    def part(value):
        return f"{value[0]} –{value[1]}→ {value[2]}" if isinstance(value, (list, tuple)) else value

    for item in sorted(info["objects"] + info["triples"], key=lambda item: -item["contribution"]):
        lines.append(f"| `{part(item['query_part'])}` | `{part(item['image_part'])}` | {item['w_i']:.2f} | "
                     f"{item['a_ij']:.2f} | {item['sim_ij']:.2f} | {item['contribution']:.4f} |")
    lines.append("\n_đóng góp = trọng số kênh × w × a × sim, trước z-score trong top-50; sắp theo đóng góp._")
    return lines


def describe(hit: Hit, mode: str) -> str:
    lines = [f"### Hạng {hit.rank} · ảnh `{hit.image_id}`",
             f"- Điểm cuối: **{hit.score:.4f}**",
             f"- Điểm CLIP (cosine): {hit.clip_score:.4f}"]
    if hit.graph_score is not None:
        lines.append(f"- Điểm đồ thị (z-score trong top-50): {hit.graph_score:.4f}")
    lines.append("\n**Giải thích**")
    if hit.explanation and isinstance(hit.explanation[0], dict):
        lines += graph_explanation(hit.explanation[0])
    elif hit.explanation:
        for path in hit.explanation:
            lines.append("- " + " ; ".join(f"{h} –{r}→ {t}" for h, r, t in path))
    elif mode == "clip":
        lines.append("- Chế độ CLIP thuần chỉ so khớp vector, không dùng đồ thị nên không có đường đi giải thích.")
    else:
        lines.append("- Không có đường đi nào nối truy vấn với ảnh này; chỉ điểm CLIP đóng góp.")
    lines.append("\n**Caption gốc của ảnh (COCO)**")
    lines += [f"- {c}" for c in hit.captions]
    return "\n".join(lines)


def build(service: SearchService, split: str) -> gr.Blocks:
    gold_pairs = [(c, i) for i, m in service.meta.items() for c in m["captions"]]

    def run(query: str, mode: str, k: int, gold_id):
        hits = service.search(query, mode=mode, k=int(k))
        gallery = [(str(h.image_path), f"#{h.rank} · {h.score:.3f}") for h in hits]
        note = f"{len(hits)} kết quả · pool {len(service.meta)} ảnh ({split}) · chế độ {MODE_LABELS.get(mode, mode)}"
        if gold_id is not None and hits:
            rank = service.rank_of(query, gold_id, mode)
            where = f"hạng **{rank}**" if rank else f"ngoài top-{service.pool_k}"
            note += f"\n\nẢnh đúng của caption này là `{gold_id}`: {where}."
        detail = describe(hits[0], mode) if hits else "Nhập truy vấn để bắt đầu."
        return gallery, note, detail, hits

    def pick(hits, mode, evt: gr.SelectData):
        return describe(hits[evt.index], mode)

    def random_caption():
        text, gold = random.choice(gold_pairs)
        return text, gold

    with gr.Blocks(title="Truy vấn ảnh có giải thích bằng đồ thị", fill_height=True, fill_width=True) as demo:
        gr.Markdown("# Truy vấn văn bản → ảnh, có giải thích bằng đồ thị\n"
                    f"Nhóm 7 · Visual Genome ∩ COCO · pool: {len(service.meta)} ảnh của split `{split}`", elem_id="title")
        hits_state, gold_state = gr.State([]), gr.State(None)
        with gr.Row():
            query = gr.Textbox(label="Truy vấn (tiếng Anh)", placeholder="a child playing football", scale=6)
            mode = gr.Radio([(MODE_LABELS.get(m, m), m) for m in service.modes], value="clip",
                            label="Chế độ xếp hạng", scale=3)
            k = gr.Slider(1, 50, value=12, step=1, label="Số kết quả (k)", scale=3)
            with gr.Column(scale=2, min_width=180):
                go = gr.Button("Tìm", variant="primary")
                lucky = gr.Button("Caption ngẫu nhiên của pool")
        note = gr.Markdown(elem_id="note")
        with gr.Row(elem_id="results"):
            gallery = gr.Gallery(label="Kết quả", columns=6, object_fit="contain", scale=3, elem_id="gallery")
            detail = gr.Markdown("Nhập truy vấn để bắt đầu.", elem_id="detail")

        outputs = [gallery, note, detail, hits_state]
        go.click(run, [query, mode, k, gold_state], outputs)
        query.submit(run, [query, mode, k, gold_state], outputs)
        mode.change(run, [query, mode, k, gold_state], outputs)
        query.input(lambda: None, None, gold_state)          # a hand-typed query has no known gold image
        lucky.click(random_caption, None, [query, gold_state]).then(run, [query, mode, k, gold_state], outputs)
        gallery.select(pick, [hits_state, mode], detail)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--run", default="main_r3_seed0", help="trained graph reranker run for the CLIP + Đồ thị mode")
    ap.add_argument("--no-graph", action="store_true", help="CLIP mode only (no model checkpoint needed)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()

    cfg = load_config()
    service = SearchService.from_config(split=args.split, cfg=cfg)
    if not args.no_graph:
        from src.service.graph_mode import GraphRerankMode
        service.add_mode("clip_graph", GraphRerankMode(cfg, args.split, service.encoder, args.run, args.device))
    missing = [m["path"].name for m in service.meta.values() if not m["path"].exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing; run: python scripts/download_images.py --splits {args.split}")
    build(service, args.split).launch(server_name=args.host, server_port=args.port,
                                      allowed_paths=[str(resolve(cfg, "raw") / "images")], css=CSS)


if __name__ == "__main__":
    main()
