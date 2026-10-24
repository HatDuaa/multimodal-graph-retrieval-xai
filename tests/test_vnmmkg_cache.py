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


from src.data.vnmmkg_cache import infobox_rows, load_link_qids, load_viwiki


def test_viwiki_loaders_and_infobox_rows(tmp_path):
    (tmp_path / "viwiki").mkdir()
    page = {"qid": "Q1", "title": "Chùa Một Cột", "page_title": "Chùa Một Cột", "pageid": 5, "revid": 99,
            "extract": "Chùa Một Cột là một ngôi chùa.",
            "wikitext": "{{Thông tin chùa\n| founder = [[Lý Thái Tông]]\n}}"}
    (tmp_path / "viwiki" / "Q1.json").write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "viwiki_links.json").write_text(json.dumps({"Lý Thái Tông": "Q60", "Không có": None},
                                                           ensure_ascii=False), encoding="utf-8")
    viwiki = load_viwiki(tmp_path)
    assert viwiki["Q1"]["revid"] == 99
    assert load_link_qids(tmp_path) == {"Lý Thái Tông": "Q60"}
    cfg_v = {"infobox_templates": ["Thông tin chùa"], "infobox_fields": {"founder": "P112"},
             "founded_fields": ["founded"]}
    assert infobox_rows(viwiki, cfg_v)["Q1"][0]["links"] == ["Lý Thái Tông"]
