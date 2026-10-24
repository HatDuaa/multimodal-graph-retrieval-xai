import json

from src.data.vnmmkg_cache import load_entities


def test_load_entities_follows_redirects(tmp_path):
    wd = tmp_path / "wikidata"
    wd.mkdir()
    (wd / "Q1.json").write_text(json.dumps({"qid": "Q1", "claims": {"P31": [{"value": "Q8", "rank": "normal"}]}}),
                                encoding="utf-8")
    (wd / "Q9.json").write_text(json.dumps({"qid": "Q9", "claims": {}}), encoding="utf-8")
    (wd / "_redirects.json").write_text(json.dumps({"Q8": "Q9"}), encoding="utf-8")
    ents = load_entities(tmp_path)
    assert set(ents) == {"Q1", "Q9"}
    assert ents["Q1"]["claims"]["P31"][0]["value"] == "Q9"
