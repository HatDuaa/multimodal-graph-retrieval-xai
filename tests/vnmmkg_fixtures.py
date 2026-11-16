"""A tiny fake raw cache for the Vietnamese MMKG build tests (no network).

Four landmark candidates; Q4 has only 2 images so it is dropped (min_images = 3). One image
of Q1 is rejected in review, one of Q3 is left undecided, Q3 has 6 candidates (max 5).
Triple review approves Q1's derived province and infobox founder, leaves Q1's derived
period undecided.
"""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

LANDMARKS = {  # qid: (label, sitelinks, n_images, extra claims)
    "Q1": ("Chùa Một Cột", 40, 4, {"P131": ["Q10"]}),
    "Q2": ("Chùa Trấn Quốc", 30, 3, {"P131": ["Q12"]}),
    "Q3": ("Chùa Hương", 20, 6, {}),
    "Q4": ("Chùa Keo", 10, 2, {}),
}


def cfg_v() -> dict:
    return {"country": "Q881", "landmark_classes": ["Q50"], "exclude_if_has": ["P195", "P189"],
            "max_landmark_candidates": 10, "min_images": 3, "max_images": 5, "max_entities": 150,
            "landmark_pids": ["P31", "P112"], "context_extra_pids": ["P131"], "province_classes": ["Q2824648"],
            "human_class": "Q5", "max_p131_depth": 4, "infobox_templates": ["Thông tin chùa"],
            "infobox_fields": {"founder": "P112"}, "founded_fields": ["founded"],
            "periods": [{"label_vi": "Nhà Lý", "start": 1009, "end": 1225, "qid": "Q200"}],
            "type_prefixes": ["Chùa"], "max_query_sentences": 3, "min_query_words": 6,
            "non_visual_pids": ["P131", "P112", "P2348"], "max_pattern_golds": 5, "val_fraction": 0.5}


def _claims(**pids) -> dict:
    return {p: [{"value": v, "rank": "normal"} for v in values] for p, values in pids.items()}


def _entity(qid, label, claims, viwiki=None, sitelinks=1, aliases=()):
    return {"qid": qid, "lastrevid": 1, "label_vi": label, "label_en": None, "aliases_vi": list(aliases),
            "desc_vi": None, "viwiki_title": viwiki, "sitelinks": sitelinks, "claims": claims}


def _jpeg(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (seed * 37 % 256, 80, 120)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def make_raw_cache(root: Path) -> tuple[dict, Path, Path]:
    raw, out = root / "raw", root / "out"
    for d in (raw / "wikidata", raw / "viwiki", raw / "commons", raw / "candidates", out):
        d.mkdir(parents=True)
    _dump(raw / "candidates.json", {"fetched": "2026-10-05", "sparql_qids": list(LANDMARKS)})
    entities = [_entity("Q10", "Phường X", _claims(P131=["Q12"])),
                _entity("Q12", "Hà Nội", _claims(P31=["Q2824648"])),
                _entity("Q50", "chùa Phật giáo", {}),
                _entity("Q60", "Lý Thái Tông", _claims(P31=["Q5"])),
                _entity("Q200", "Nhà Lý", {})]
    for qid, (label, sitelinks, _, extra) in LANDMARKS.items():
        claims = _claims(P17=["Q881"], P31=["Q50"], P373=[label], **extra)
        if qid == "Q1":
            claims["P571"] = [{"value": "+1049-00-00T00:00:00Z", "rank": "normal", "precision": 9}]
        entities.append(_entity(qid, label, claims, viwiki=label, sitelinks=sitelinks,
                                aliases=["Diên Hựu tự"] if qid == "Q1" else []))
    for e in entities:
        _dump(raw / "wikidata" / f"{e['qid']}.json", e)
    for qid, (label, *_rest) in LANDMARKS.items():
        _dump(raw / "viwiki" / f"{qid}.json", {
            "qid": qid, "title": label, "page_title": label, "pageid": 1, "revid": 100 + int(qid[1:]),
            "extract": f"{label} (chữ Hán: 一柱寺) là một ngôi chùa cổ nằm ở trung tâm thành phố. "
                       "Ngôi chùa có kiến trúc gỗ rất độc đáo và lâu đời. Ngắn.",
            "wikitext": "{{Thông tin chùa\n| founder = [[Lý Thái Tông]]\n}}" if qid == "Q1" else ""})
    _dump(raw / "viwiki_links.json", {"Lý Thái Tông": "Q60"})
    decisions, seed = {}, 0
    for qid, (label, _, n_images, _) in LANDMARKS.items():
        candidates = []
        for k in range(n_images):
            seed += 1
            data, local, title = _jpeg(seed), f"{qid}_{k}.jpg", f"File:{label} {k}.jpg"
            (raw / "candidates" / local).write_bytes(data)
            candidates.append({"file_title": title, "source_url": f"https://commons.wikimedia.org/wiki/{title}",
                               "thumb_url": f"https://upload.wikimedia.org/{local}", "orig_width": 800,
                               "orig_height": 600, "license": "CC BY-SA 4.0",
                               "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
                               "artist": "Tác giả", "local_file": local,
                               "sha256": hashlib.sha256(data).hexdigest(), "width": 64, "height": 48})
            decisions[title] = {"keep": True, "reason": ""}
        _dump(raw / "commons" / f"{qid}.json", {"qid": qid, "category": label, "p18": None,
                                                "n_category_files": n_images, "drops": {}, "candidates": candidates})
    decisions["File:Chùa Một Cột 0.jpg"] = {"keep": False, "reason": "bản đồ / sơ đồ"}
    del decisions["File:Chùa Hương 5.jpg"]
    _dump(out / "review_images.json", {"reviewer": "test", "exported": "2026-10-05", "decisions": decisions})
    _dump(out / "review_triples.json", {"reviewer": "test", "exported": "2026-10-05", "decisions": {
        "Q1|P131|Q12|derived": {"keep": True, "reason": ""},
        "Q1|P112|Q60|infobox": {"keep": True, "reason": ""}}})
    return cfg_v(), raw, out
