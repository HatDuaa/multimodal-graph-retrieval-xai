import hashlib

import pytest

from src.data.build_vnmmkg import (apply_triple_review, build_dataset, replace_stats_block, run_build,
                                   scale_problems, select_images, split_landmarks)
from src.data.vnmmkg_text import contains_name, name_variants
from tests.vnmmkg_fixtures import make_raw_cache


def cand(title):
    return {"file_title": title}


def test_select_images_rules():
    commons = {"Q1": {"candidates": [cand("a"), cand("b"), cand("c"), cand("d")]},
               "Q2": {"candidates": [cand("a"), cand("e"), cand("f"), cand("g")]},
               "Q3": {"candidates": [cand("h")]}}
    decisions = {t: {"keep": True, "reason": ""} for t in "abcdefgh"}
    decisions["d"] = {"keep": False, "reason": "x"}
    del decisions["g"]
    selected, stats = select_images(["Q1", "Q2", "Q3"], commons, decisions, min_images=3, max_images=2)
    assert [c["file_title"] for c in selected["Q1"]] == ["a", "b"]           # max 2, candidate order kept
    assert set(selected) == {"Q1"}           # Q2: a is taken by Q1, g undecided -> 2 kept < 3; Q3: 1 image
    assert stats == {"images_over_max": 1, "images_rejected": 1, "images_shared_with_earlier_landmark": 1,
                     "images_undecided": 1, "landmarks_too_few_images": 2}


def test_shared_image_counts_once():
    commons = {"Q1": {"candidates": [cand("a"), cand("b"), cand("c")]},
               "Q2": {"candidates": [cand("a"), cand("e"), cand("f")]}}
    decisions = {t: {"keep": True, "reason": ""} for t in "abcef"}
    selected, stats = select_images(["Q1", "Q2"], commons, decisions, min_images=3, max_images=5)
    assert "Q2" not in selected and stats["images_shared_with_earlier_landmark"] == 1


def test_apply_triple_review_rules():
    triples = [{"h": "Q1", "r": "P31", "t": "Q5", "source": "wikidata", "evidence": ""},
               {"h": "Q1", "r": "P140", "t": "Q6", "source": "wikidata", "evidence": ""},
               {"h": "Q1", "r": "P112", "t": "Q7", "source": "infobox", "evidence": ""},
               {"h": "Q1", "r": "P131", "t": "Q8", "source": "derived", "evidence": ""}]
    decisions = {"Q1|P140|Q6|wikidata": {"keep": False, "reason": "triple sai"},
                 "Q1|P112|Q7|infobox": {"keep": True, "reason": ""}}
    kept, stats = apply_triple_review(triples, decisions)
    assert [(t["r"], t["reviewed"]) for t in kept] == [("P31", False), ("P112", True)]
    assert stats == {"derived_undecided": 1, "wikidata_rejected": 1}


def test_split_landmarks_deterministic_disjoint():
    qids = [f"Q{i}" for i in range(1, 92)]
    a = split_landmarks(qids, seed=0, val_fraction=0.5)
    assert a == split_landmarks(list(reversed(qids)), seed=0, val_fraction=0.5)
    assert not set(a["val"]) & set(a["test"]) and len(a["val"]) + len(a["test"]) == 91
    assert abs(len(a["val"]) - len(a["test"])) <= 1
    assert split_landmarks(qids, seed=1, val_fraction=0.5) != a


def test_build_end_to_end(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    r = build_dataset(cfg_v, raw, out, seed=0)
    assert r["landmarks"] == ["Q1", "Q2", "Q3"]                              # Q4: only 2 images
    assert [im["image_id"] for im in r["images"]] == list(range(11))         # 3 + 3 + 5
    assert {e["qid"]: e["category"] for e in r["entities"]} == {
        "Q1": "landmark", "Q2": "landmark", "Q3": "landmark", "Q12": "province", "Q50": "class", "Q60": "person"}
    keys = {(t["h"], t["r"], t["t"], t["source"], t["reviewed"]) for t in r["triples"]}
    assert ("Q1", "P131", "Q12", "derived", True) in keys
    assert ("Q2", "P131", "Q12", "wikidata", False) in keys
    assert not any(t["r"] == "P2348" for t in r["triples"])                  # period left undecided
    assert r["stats"]["triple_review"] == {"derived_undecided": 1}
    assert r["stats"]["image_selection"]["images_undecided"] == 1
    assert len(r["queries_auto"]) == 6
    for q in r["queries_auto"]:
        e = next(x for x in r["entities"] if x["qid"] == q["qid"])
        assert not contains_name(q["text"], name_variants(e["label_vi"], e["aliases_vi"], e["viwiki_title"], ["Chùa"]))
    val, test = r["splits"]["val"], r["splits"]["test"]
    val_q = {im["qid"] for im in val["images"]}
    assert not val_q & {im["qid"] for im in test["images"]}
    assert {q["gold_qid"] for q in val["queries"]} <= val_q
    assert val["images"][0]["captions"] == [] and val["images"][0]["file_name"].endswith(".jpg")


def _digest(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def test_build_is_byte_identical_on_rerun(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    splits, stats_md = tmp_path / "splits", tmp_path / "data_stats.md"
    stats_md.write_text("# Thống kê\n\nPhần khác.\n", encoding="utf-8")
    run_build(cfg_v, raw, out, splits, stats_md, seed=0)
    files = [*out.glob("*.jsonl"), *splits.glob("*.json"), *(raw / "images").glob("*.jpg"), stats_md]
    first = _digest(files)
    run_build(cfg_v, raw, out, splits, stats_md, seed=0)
    assert _digest(files) == first
    text = stats_md.read_text(encoding="utf-8")
    assert text.count("<!-- vnmmkg:begin -->") == 1 and text.startswith("# Thống kê\n\nPhần khác.\n")
    assert sorted(p.name for p in (raw / "images").glob("*.jpg")) == sorted(f"{i}.jpg" for i in range(11))


def test_build_rejects_changed_image(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    (raw / "candidates" / "Q2_0.jpg").write_bytes(b"not the reviewed image")
    with pytest.raises(ValueError, match="Q2_0.jpg"):
        run_build(cfg_v, raw, out, tmp_path / "splits", tmp_path / "s.md", seed=0)
    assert not (out / "entities.jsonl").exists()                             # nothing written


def test_undecided_items_are_excluded_and_counted(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    (out / "review_triples.json").unlink()
    r = build_dataset(cfg_v, raw, out, seed=0)
    assert r["stats"]["triple_review"] == {"derived_undecided": 2, "infobox_undecided": 1}
    assert all(t["source"] == "wikidata" for t in r["triples"])


def test_replace_stats_block_and_scale(tmp_path):
    md = tmp_path / "s.md"
    replace_stats_block(md, "<!-- vnmmkg:begin -->\nA\n<!-- vnmmkg:end -->\n")
    replace_stats_block(md, "<!-- vnmmkg:begin -->\nB\n<!-- vnmmkg:end -->\n")
    assert md.read_text(encoding="utf-8").count("vnmmkg:begin") == 1 and "B" in md.read_text(encoding="utf-8")
    assert scale_problems({"entities": 120, "triples": 400, "images": 350}) == []
    assert scale_problems({"entities": 6, "triples": 400, "images": 350}) == ["entities = 6 outside 50-150"]


def test_build_refuses_missing_period_qid(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    cfg_v["periods"][0]["qid"] = None
    with pytest.raises(ValueError, match="qid"):
        build_dataset(cfg_v, raw, out, seed=0)


import json

from src.data.build_vnmmkg import resolve_hand_queries


def _hand(qid="h1", group="indirect", text="Ngôi chùa ở Hà Nội có người dựng", pattern=None, gold=None):
    return {"query_id": qid, "group": group, "text": text,
            "pattern": [["P31", "Q50"], ["P131", "Q12"]] if pattern is None else pattern, "gold_qids": gold}


def test_hand_queries_gold_and_validation(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    r = build_dataset(cfg_v, raw, out, seed=0)
    args = (r["triples"], r["entities"], cfg_v)
    queries = [_hand(), _hand("d1", "direct", "Ngôi chùa có mái ngói cong và sân rộng", [], ["Q3"])]
    resolved, stats = resolve_hand_queries(queries, *args, allow_update=False)
    assert resolved[0]["gold_qids"] == ["Q1", "Q2"] and resolved[1]["gold_qids"] == ["Q3"]
    assert stats == {"direct": 1, "indirect": 1, "indirect_gold_sizes": {"2": 1}}
    assert resolve_hand_queries(resolved, *args, allow_update=False)[0] == resolved      # idempotent

    def problems(qs, **kw):
        with pytest.raises(ValueError) as err:
            resolve_hand_queries(qs, *args, allow_update=kw.get("allow_update", False))
        return str(err.value)

    assert "non-visual" in problems([_hand(pattern=[["P31", "Q50"]])])
    assert "no gold" in problems([_hand(pattern=[["P131", "Q999"]])])
    cfg_v["max_pattern_golds"] = 1                      # args holds the same dict
    assert "too many" in problems([_hand()])
    cfg_v["max_pattern_golds"] = 5
    assert "exactly one" in problems([_hand("d1", "direct", "Ngôi chùa có mái ngói cong", [], ["Q1", "Q2"])])
    assert "duplicate" in problems([_hand(), _hand()])
    assert "gold changed" in problems([_hand(gold=["Q1"])])
    assert resolve_hand_queries([_hand(gold=["Q1"])], *args, allow_update=True)[0][0]["gold_qids"] == ["Q1", "Q2"]
    assert "names its own answer" in problems([_hand("d1", "direct", "Chùa Hương có mái ngói cong", [], ["Q3"])])


def test_run_build_writes_hand_queries_and_stats(tmp_path):
    cfg_v, raw, out = make_raw_cache(tmp_path)
    (out / "queries_hand.json").write_text(json.dumps([_hand()], ensure_ascii=False), encoding="utf-8")
    stats = run_build(cfg_v, raw, out, tmp_path / "splits", tmp_path / "s.md", seed=0)
    assert json.loads((out / "queries_hand.json").read_text(encoding="utf-8"))[0]["gold_qids"] == ["Q1", "Q2"]
    assert stats["hand_queries"]["indirect"] == 1
    before = (out / "queries_hand.json").read_bytes()
    run_build(cfg_v, raw, out, tmp_path / "splits", tmp_path / "s.md", seed=0)
    assert (out / "queries_hand.json").read_bytes() == before
