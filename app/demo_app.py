"""Demo: type a query, see the top-k images and why each one was returned.

Runs fully offline once features and the test/val images are on disk:
    python app/demo_app.py [--split test] [--port 7860]
then open http://localhost:7860

The page only calls SearchService.search(); ranking modes added to the service (for example
CLIP + graph) appear in the "Chế độ xếp hạng" selector automatically.
"""
import argparse
import html
import random
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain.score_breakdown import split_final_score  # noqa: E402
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
#detail-col { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; overflow: hidden; }
#detail { flex: 1 1 0 !important; height: auto !important; min-height: 0 !important; overflow-y: auto !important; padding-right: 6px; }
#detail h3 { margin: 0 0 6px; font-size: 1.1rem; }
#detail h4 { margin: 14px 0 6px; }
#detail ul { margin: 0 0 4px; padding-left: 20px; }
#detail code { font-size: 0.85em; padding: 1px 4px; border-radius: 4px; background: var(--neutral-100, #f3f4f6); }
/* explanation tables: text cells wrap only between words, numeric cells (w, a, sim, đóng góp) never wrap */
#detail table { display: table; width: 100%; table-layout: auto; border-collapse: collapse; margin: 4px 0;
                word-break: normal !important; }
#detail th, #detail td { word-break: normal !important; overflow-wrap: normal !important; hyphens: none;
                         padding: 4px 6px; border-bottom: 1px solid var(--neutral-200, #e5e7eb); text-align: left; }
#detail th { white-space: nowrap; font-weight: 600; }
#detail th.text { white-space: normal; }
#detail .num { white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
/* one table, one row group per channel; a light tint of the channel colour tells the groups apart */
#detail .ch-clip { --ch: #2563eb; --tint: rgba(37, 99, 235, 0.06); }
#detail .ch-obj { --ch: #ea580c; --tint: rgba(234, 88, 12, 0.06); }
#detail .ch-tri { --ch: #9333ea; --tint: rgba(147, 51, 234, 0.06); }
#detail tbody tr { background: var(--tint); }
#detail tbody tr.group-start > td { border-top: 2px solid var(--neutral-300, #d1d5db); }
#detail td.channel { border-left: 3px solid var(--ch); vertical-align: top; white-space: nowrap; }
#detail td.channel b { color: var(--ch); }
#detail tr.total > td { font-weight: 600; border-top: 2px solid var(--neutral-400, #9ca3af); background: none; }
#detail .muted { color: var(--body-text-color-subdued, #6b7280); font-size: 0.85rem; margin: 2px 0; }
#detail .summary { margin: 0 0 6px; font-variant-numeric: tabular-nums; }
#detail .summary .clip { color: #2563eb; font-weight: 600; }
#detail .summary .graph { color: #374151; font-weight: 600; }
#detail .share { display: inline-flex; align-items: center; justify-content: flex-end; gap: 6px; }
#detail .sbar { display: inline-block; width: 40px; height: 6px; border-radius: 3px; overflow: hidden;
                background: var(--neutral-200, #e5e7eb); }
#detail .sbar > span { display: block; height: 100%; background: var(--ch); opacity: 0.75; }
#detail .sbar.neg > span { background: var(--neutral-400, #9ca3af); }
"""

# Short reason shown in place of the matched image part; codes come from src.service.graph_mode.unmatched_parts.
UNMATCHED_REASONS = {
    "pronoun": {"objects": "đại từ, bị loại", "triples": "chứa đại từ, bị loại"},
    "empty_label": "nhãn rỗng, bị che",
    "dangling": "thiếu một đầu quan hệ",
    "image_empty": {"objects": "ảnh không có vật thể", "triples": "ảnh không có bộ ba"},
}
NO_QUERY_PARTS = {"objects": "câu không có vật thể", "triples": "câu không có quan hệ"}
CHANNELS = (("objects", "Vật thể", "ch-obj"), ("triples", "Bộ ba", "ch-tri"))
EMPTY = '<td class="num"></td>'
DASH = '<td class="num">—</td>'


def esc(value) -> str:
    return html.escape(str(value))


def part_text(value) -> str:
    return f"{value[0]} –{value[1]}→ {value[2]}" if isinstance(value, (list, tuple)) else str(value)


def signed(value: float, fmt: str) -> str:
    """Number with a typographic minus, so negative parts stand out."""
    return format(value, fmt).replace("-", "−")


def num(value: float | None, digits: int) -> str:
    return EMPTY if value is None else f'<td class="num">{signed(value, f".{digits}f")}</td>'


def share_cell(share: float | None, part: float | None, raw: float | None = None) -> str:
    """% of the final score with a small bar; the absolute part when the final score is not positive."""
    if part is None:
        return DASH
    title = f' title="đóng góp thô = {raw:.4f}"' if raw is not None else ""
    if share is None:
        return f'<td class="num"{title}>{signed(part, ".3f")}</td>'
    bar = (f'<span class="sbar{" neg" if share < 0 else ""}">'
           f'<span style="width:{min(abs(share), 1) * 100:.1f}%"></span></span>')
    return f'<td class="num"{title}><span class="share">{bar}{signed(share, ".0%")}</span></td>'


def channel_rows(key: str, title: str, cls: str, weight: float, info: dict, split: dict) -> list[str]:
    """Row group of one channel: matched parts by contribution, then query parts without a match and why."""
    active = info.get("channel_active", [True, True])[0 if key == "objects" else 1]
    cells = []
    rows = zip(info[key], split["row_parts"][key], split["row_shares"][key])
    for item, part, share in sorted(rows, key=lambda row: -row[0]["contribution"]):
        cells.append(f'<td><code>{esc(part_text(item["query_part"]))}</code></td>' + num(item["w_i"], 2)
                     + f'<td><code>{esc(part_text(item["image_part"]))}</code></td>'
                     + num(item["a_ij"] * item["sim_ij"], 2) + share_cell(share, part, item["contribution"]))
    for item in info.get("unmatched", {}).get(key, []):
        reason = UNMATCHED_REASONS.get(item["reason"], item["reason"])
        reason = reason[key] if isinstance(reason, dict) else reason
        cells.append(f'<td><code>{esc(part_text(item["query_part"]))}</code></td>' + DASH
                     + f'<td>— <span class="muted">{esc(reason)}</span></td>' + DASH + DASH)
    if not cells:
        cells.append("<td>—</td>" + DASH + f'<td>— <span class="muted">{NO_QUERY_PARTS[key]}</span></td>'
                     + DASH + DASH)
    note = '<span class="muted">Σw = 1</span>' if active else '<span class="muted">(không dùng)</span>'
    head = f'<td class="channel" rowspan="{len(cells)}"><b>{title}</b> {weight:.0%}<br>{note}</td>'
    return [f'<tr class="{cls}{" group-start" if i == 0 else ""}">{head if i == 0 else ""}{row}</tr>'
            for i, row in enumerate(cells)]


def score_summary(fused: float, split: dict) -> str:
    def term(name, cls, part, share):
        return (f'<span class="{cls}">{name} {signed(part, ".3f")}</span>'
                + ("" if share is None else f" ({signed(share, '.0%')})"))
    return (f'<p class="summary">Điểm cuối <b>{signed(fused, ".3f")}</b> = '
            + term("CLIP", "clip", split["clip_part"], split["clip_share"]) + " + "
            + term("Đồ thị", "graph", split["graph_part"], split["graph_share"]) + "</p>")


def graph_explanation(info: dict, hit: Hit) -> str:
    """Matched part pairs exactly as the reranker scored them, with each row's share of the final score."""
    a, b, c = info["weights_abc"]
    active = dict(zip(("objects", "triples"), info.get("channel_active", [True, True])))
    split = split_final_score(hit.score, a, info["z_clip"], hit.graph_score,
                              {key: [item["contribution"] for item in info[key]] for key, _, _ in CHANNELS}, active)
    rows = ['<tr class="ch-clip group-start"><td class="channel"><b>CLIP</b> ' f"{a:.0%}</td>"
            "<td>cả câu</td>" + DASH + "<td>cả ảnh</td>"
            f'<td class="num">cosine {hit.clip_score:.4f}</td>'
            + share_cell(split["clip_share"], split["clip_part"]) + "</tr>"]
    for (key, title, cls), weight in zip(CHANNELS, (b, c)):
        rows += channel_rows(key, title, cls, weight, info, split)
    unused = not all(active.values())
    total_share = "100%" if hit.score > 0 else signed(hit.score, ".3f")
    rows.append(f'<tr class="total"><td>Tổng</td><td></td>{EMPTY}<td></td>{EMPTY}'
                f'<td class="num">{total_share}</td></tr>')
    notes = ["Tầng CLIP / đồ thị là tách đúng theo công thức điểm cuối "
             "(điểm cuối = a × z(CLIP) + (1 − a) × điểm đồ thị).",
             "Chia phần đồ thị cho từng dòng là xấp xỉ theo tỉ lệ đóng góp thô (trọng số kênh × w × a × sim, "
             "xem khi rê chuột lên ô %), vì z-score trừ trung bình của 50 ứng viên."]
    if unused:
        notes.append("(không dùng): kênh không có phần nào so khớp được hoặc điểm kênh như nhau trên cả top-50, "
                     "nên bị bỏ khỏi điểm đồ thị; điểm đồ thị chỉ lấy từ kênh còn lại.")
    if hit.score <= 0:
        notes.append("Điểm cuối không dương nên cột % điểm cuối ghi giá trị tuyệt đối của từng phần.")
    return (score_summary(hit.score, split)
            + '<table><thead><tr><th>Kênh</th><th class="text">Phần của câu</th><th class="num">w</th>'
            '<th class="text">Phần khớp trong ảnh</th>'
            '<th class="text">Độ khớp (a × sim)</th><th class="text">% điểm cuối</th></tr>'
            "</thead><tbody>" + "".join(rows) + "</tbody></table>"
            + "".join(f'<p class="muted">{note}</p>' for note in notes))


def describe(hit: Hit, mode: str) -> str:
    out = [f"<h3>Hạng {hit.rank} · ảnh <code>{hit.image_id}</code></h3><ul>",
           f"<li>Điểm cuối: <b>{hit.score:.4f}</b></li>",
           f"<li>Điểm CLIP (cosine): {hit.clip_score:.4f}</li>"]
    if hit.graph_score is not None:
        out.append(f"<li>Điểm đồ thị (z-score trong top-50): {hit.graph_score:.4f}</li>")
    out.append("</ul><h4>Giải thích</h4>")
    if hit.explanation and isinstance(hit.explanation[0], dict):
        out.append(graph_explanation(hit.explanation[0], hit))
    elif hit.explanation:
        out.append("<ul>" + "".join("<li>" + esc(" ; ".join(f"{h} –{r}→ {t}" for h, r, t in path)) + "</li>"
                                    for path in hit.explanation) + "</ul>")
    elif mode == "clip":
        out.append("<ul><li>Chế độ CLIP thuần chỉ so khớp vector, không dùng đồ thị nên không có đường đi giải thích."
                   "</li></ul>")
    else:
        out.append("<ul><li>Không có đường đi nào nối truy vấn với ảnh này; chỉ điểm CLIP đóng góp.</li></ul>")
    out.append("<h4>Caption gốc của ảnh (COCO)</h4><ul>" + "".join(f"<li>{esc(c)}</li>" for c in hit.captions) + "</ul>")
    return "".join(out)


START_TEXT = "<p>Nhập truy vấn để bắt đầu.</p>"


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
        detail = describe(hits[0], mode) if hits else START_TEXT
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
            # 11:9 split gives the detail panel ~45% of the width so the explanation table fits unwrapped;
            # the detail component ignores scale/min_width, so it sits in a Column that carries them
            gallery = gr.Gallery(label="Kết quả", columns=5, object_fit="contain", scale=11, elem_id="gallery")
            with gr.Column(scale=9, min_width=480, elem_id="detail-col"):
                detail = gr.HTML(START_TEXT, elem_id="detail")

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
