"""Step 4 of the Vietnamese MMKG pipeline: a local HTML page to review images and triples.

The page lists every image candidate of the top landmark candidates (keep / drop with a
reason, spec 4.3 criteria) and every proposed triple (infobox and derived ones must be
decided; Wikidata ones are kept unless dropped). Decisions autosave in the browser and are
exported as vietnamese-mmkg/review_images.json and review_triples.json, which the build
(src/data/build_vnmmkg.py) reads. Open the page from data/raw/vnmmkg/ so that the relative
image paths work.

Usage:  python scripts/build_vnmmkg_review.py
Output: data/raw/vnmmkg/review.html
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.vnmmkg_cache import (infobox_rows, landmark_candidates, load_commons, load_link_qids,  # noqa: E402
                                   load_viwiki)
from src.data.vnmmkg_triples import propose_triples, triple_key  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402

TEMPLATE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>vnmmkg review</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
header{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ccc;z-index:2}
.lm{background:#fff;margin:12px 0;padding:8px;border:1px solid #ddd}
.grid{display:flex;flex-wrap:wrap;gap:8px}
.card{width:220px;border:3px solid #bbb;padding:4px;font-size:12px;word-break:break-all}
.card img{width:100%;height:150px;object-fit:cover;cursor:zoom-in}
.card.keep{border-color:#2a2}.card.drop{border-color:#c22;opacity:.6}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:3px}
tr.drop{background:#fdd}tr.keep{background:#dfd}
</style></head><body>
<header>
 Người duyệt: <input id="reviewer" size="20"> <span id="progress"></span>
 <button onclick="exportJson('images')">Xuất review_images.json</button>
 <button onclick="exportJson('triples')">Xuất review_triples.json</button>
 Nhập lại: <input type="file" id="importer" accept=".json" multiple>
 <label><input type="checkbox" id="showWd"> hiện cả triple Wikidata</label>
</header>
<h2>1. Ảnh</h2>
<p>Giữ: ảnh cho thấy rõ chính di tích (mặt ngoài, hoặc nội thất / chi tiết đặc trưng). Bỏ: bản đồ, sơ đồ, tài liệu,
ảnh gần trùng, ảnh có người nhận diện được là chủ thể chính, ảnh không nhận ra là di tích đó.</p>
<div id="images"></div>
<h2>2. Triple</h2>
<p>Triple <b>infobox</b> và <b>derived</b> phải được duyệt; triple Wikidata mặc định giữ, chỉ bỏ nếu sai rõ ràng.</p>
<table id="triples"><thead><tr><th>Đầu</th><th>Quan hệ</th><th>Đuôi</th><th>Nguồn</th><th>Bằng chứng</th>
<th>Quyết định</th></tr></thead><tbody></tbody></table>
<script>
const DATA = __DATA__;
const KEY = "vnmmkg-review-v1";
const REASONS = ["", "bản đồ / sơ đồ", "tài liệu / chữ", "gần trùng ảnh khác", "người là chủ thể chính",
                 "không nhận ra di tích", "triple sai", "khác"];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
let state = {reviewer: "", images: {}, triples: {}};
try { state = Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function controls(kind, key) {
  const span = document.createElement("span");
  for (const [label, keep] of [["Giữ", true], ["Bỏ", false]]) {
    const b = document.createElement("button"); b.textContent = label;
    b.onclick = () => { state[kind][key] = {keep, reason: keep ? "" : ((state[kind][key] || {}).reason || "")}; save(); render(); };
    span.appendChild(b);
  }
  const s = document.createElement("select");
  REASONS.forEach(r => { const o = document.createElement("option"); o.value = r; o.textContent = r || "(lý do bỏ)"; s.appendChild(o); });
  s.value = (state[kind][key] || {}).reason || "";
  s.onchange = () => { state[kind][key] = {keep: false, reason: s.value}; save(); render(); };
  span.appendChild(s);
  return span;
}
function progress() {
  const imgs = DATA.landmarks.flatMap(l => l.images);
  const need = DATA.triples.filter(t => t.source !== "wikidata");
  document.getElementById("progress").textContent =
    `Ảnh đã duyệt ${imgs.filter(i => state.images[i.file_title]).length}/${imgs.length} · ` +
    `triple cần duyệt ${need.filter(t => state.triples[t.key]).length}/${need.length}`;
}
function render() {
  const box = document.getElementById("images"); box.innerHTML = "";
  for (const lm of DATA.landmarks) {
    const div = document.createElement("div"); div.className = "lm";
    const kept = lm.images.filter(im => (state.images[im.file_title] || {}).keep === true).length;
    div.innerHTML = `<h3>${esc(lm.label)} <small>${esc(lm.qid)} · giữ ${kept}/${lm.images.length}</small></h3>`;
    const allKeep = document.createElement("button"); allKeep.textContent = "Giữ mọi ảnh chưa duyệt";
    allKeep.onclick = () => { lm.images.forEach(im => { if (!state.images[im.file_title]) state.images[im.file_title] = {keep: true, reason: ""}; }); save(); render(); };
    div.appendChild(allKeep);
    const grid = document.createElement("div"); grid.className = "grid";
    for (const im of lm.images) {
      const d = state.images[im.file_title];
      const card = document.createElement("div"); card.className = "card" + (d ? (d.keep ? " keep" : " drop") : "");
      const img = document.createElement("img"); img.src = "candidates/" + im.local; img.loading = "lazy";
      img.onclick = () => window.open(img.src, "_blank");
      card.appendChild(img);
      const meta = document.createElement("div");
      meta.innerHTML = `<a href="${esc(im.source_url)}" target="_blank">${esc(im.file_title)}</a><br>${esc(im.license)} · ${esc(im.artist || "?")}<br>`;
      card.appendChild(meta); card.appendChild(controls("images", im.file_title));
      grid.appendChild(card);
    }
    div.appendChild(grid); box.appendChild(div);
  }
  const tbody = document.querySelector("#triples tbody"); tbody.innerHTML = "";
  const showWd = document.getElementById("showWd").checked;
  for (const t of DATA.triples) {
    if (t.source === "wikidata" && !showWd && !state.triples[t.key]) continue;
    const d = state.triples[t.key];
    const tr = document.createElement("tr"); tr.className = d ? (d.keep ? "keep" : "drop") : "";
    tr.innerHTML = `<td>${esc(t.h_label)} <small>${esc(t.h)}</small></td><td>${esc(t.r_label)} <small>${esc(t.r)}</small></td>` +
                   `<td>${esc(t.t_label)} <small>${esc(t.t)}</small></td><td>${esc(t.source)}</td><td>${esc(t.evidence)}</td>`;
    const td = document.createElement("td"); td.appendChild(controls("triples", t.key));
    tr.appendChild(td); tbody.appendChild(tr);
  }
  progress();
}
function exportJson(kind) {
  state.reviewer = document.getElementById("reviewer").value; save();
  const decisions = {};
  Object.keys(state[kind]).sort().forEach(k => decisions[k] = state[kind][k]);
  const payload = {reviewer: state.reviewer, exported: new Date().toISOString().slice(0, 10), decisions};
  const blob = new Blob([JSON.stringify(payload, null, 1) + "\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `review_${kind}.json`; a.click();
}
document.getElementById("importer").onchange = async (ev) => {
  for (const f of ev.target.files) {
    const obj = JSON.parse(await f.text());
    Object.assign(state[f.name.includes("triples") ? "triples" : "images"], obj.decisions || {});
    if (obj.reviewer) state.reviewer = obj.reviewer;
  }
  document.getElementById("reviewer").value = state.reviewer || "";
  save(); render();
};
document.getElementById("reviewer").value = state.reviewer || "";
document.getElementById("showWd").onchange = render;
render();
</script></body></html>
"""


def review_payload(ranked: list[str], entities: dict, commons: dict, triples: list[dict], pid_labels: dict) -> dict:
    def label(qid):
        e = entities.get(qid) or {}
        return e.get("label_vi") or e.get("label_en") or qid

    landmarks = [{"qid": q, "label": label(q),
                  "images": [{"file_title": c["file_title"], "local": c["local_file"], "license": c["license"],
                              "artist": c["artist"], "source_url": c["source_url"]}
                             for c in commons[q]["candidates"]]}
                 for q in ranked if q in commons]
    rows = [{"key": triple_key(t), "h": t["h"], "h_label": label(t["h"]), "r": t["r"],
             "r_label": pid_labels.get(t["r"], t["r"]), "t": t["t"], "t_label": label(t["t"]),
             "source": t["source"], "evidence": t["evidence"]} for t in triples]
    return {"landmarks": landmarks, "triples": rows}


def render_html(payload: dict) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))


def main() -> None:
    cfg = load_config()
    v = cfg["vnmmkg"]
    raw_dir = REPO_ROOT / v["raw_dir"]
    ranked, entities = landmark_candidates(raw_dir, v)
    commons = load_commons(raw_dir)
    triples, counts = propose_triples(ranked, entities, infobox_rows(load_viwiki(raw_dir), v),
                                      load_link_qids(raw_dir), v)
    payload = review_payload(ranked, entities, commons, triples, v["pid_labels_vi"])
    out = raw_dir / "review.html"
    out.write_text(render_html(payload), encoding="utf-8", newline="\n")
    n_images = sum(len(lm["images"]) for lm in payload["landmarks"])
    n_review = sum(t["source"] != "wikidata" for t in triples)
    print(f"{len(payload['landmarks'])} landmarks, {n_images} images, {len(triples)} triples "
          f"({n_review} need review); counters {counts}\n-> {out}")


if __name__ == "__main__":
    main()
