import pytest

from src.data.vnmmkg_triples import (cap_entities, categorize, gold_from_pattern, normalize_title, parse_infobox,
                                     parse_year, period_for_year, propose_triples, province_of, triple_key,
                                     validate_periods)

PERIODS = [{"label_vi": "Nhà Lý", "start": 1009, "end": 1225, "qid": "Q200"},
           {"label_vi": "Nhà Trần", "start": 1226, "end": 1399, "qid": "Q201"}]
CFG = {"landmark_pids": ["P31", "P140", "P825", "P112"], "context_extra_pids": ["P131"],
       "province_classes": ["Q2824648"], "human_class": "Q5", "max_p131_depth": 4,
       "infobox_fields": {"founder": "P112", "thờ phụng": "P825"}, "founded_fields": ["founded"],
       "periods": PERIODS}


def ent(qid, claims=None):
    claims = claims or {}
    return {"qid": qid, "label_vi": qid, "claims": {p: [{"value": v, "rank": "normal"} for v in vs]
                                                   for p, vs in claims.items()}}


def test_normalize_title():
    assert normalize_title("thông_tin  chùa") == "Thông tin chùa"
    assert normalize_title("Lý Thái Tông#Thân thế") == "Lý Thái Tông"


WIKITEXT = """{{Thông tin chùa
| tên = Chùa Một Cột
| founder = [[Lý Thái Tông|vua Lý Thái Tông]]
| founded = năm 1049
| thờ phụng = Quán Thế Âm
| hình = [[Tập tin:Chua Mot Cot.jpg]]
}}
'''Chùa Một Cột''' là ..."""


def test_parse_infobox_reads_links_and_raw_text():
    rows = parse_infobox(WIKITEXT, ["thông tin chùa"], ["founder", "founded", "thờ phụng"])
    by_field = {r["field"]: r for r in rows}
    assert by_field["founder"]["links"] == ["Lý Thái Tông"]
    assert by_field["founder"]["raw"] == "vua Lý Thái Tông"
    assert by_field["founded"] == {"template": "Thông tin chùa", "field": "founded", "raw": "năm 1049", "links": []}
    assert by_field["thờ phụng"]["links"] == []                 # plain text: no link, counted later
    assert parse_infobox(WIKITEXT, ["Thông tin đền Việt Nam"], ["founder"]) == []


def test_parse_year_only_unambiguous():
    assert parse_year("năm 1049") == 1049
    assert parse_year("1049–1054") is None                      # two years
    assert parse_year("thế kỷ 11") is None
    assert parse_year("khoảng năm 1100") is None
    assert parse_year("") is None


def test_periods_boundaries_and_validation():
    validate_periods(PERIODS)
    assert period_for_year(1009, PERIODS) == "Q200"
    assert period_for_year(1225, PERIODS) == "Q200"
    assert period_for_year(1226, PERIODS) == "Q201"
    assert period_for_year(1008, PERIODS) is None               # before 1009: no triple
    assert period_for_year(1400, PERIODS) is None               # gap
    with pytest.raises(ValueError, match="qid"):
        validate_periods([{**PERIODS[0], "qid": None}])
    with pytest.raises(ValueError, match="overlap"):
        validate_periods([PERIODS[0], {**PERIODS[1], "start": 1200}])


def entities_with_chain():
    return {
        "Q1": ent("Q1", {"P131": ["Q10"]}),                    # landmark -> ward -> district -> province
        "Q10": ent("Q10", {"P131": ["Q11"]}),
        "Q11": ent("Q11", {"P131": ["Q12"]}),
        "Q12": ent("Q12", {"P31": ["Q2824648"]}),
        "Q2": ent("Q2", {"P131": ["Q12"]}),                    # directly in the province
        "Q3": ent("Q3", {"P131": ["Q10", "Q13"]}),             # two truthy P131 (merger) -> two provinces
        "Q13": ent("Q13", {"P131": ["Q14"]}),
        "Q14": ent("Q14", {"P31": ["Q2824648"]}),
    }


def test_province_chain():
    ents = entities_with_chain()
    assert province_of("Q1", ents, ["Q2824648"], 4) == ("Q12", 3)
    assert province_of("Q2", ents, ["Q2824648"], 4) == ("Q12", 1)
    assert province_of("Q1", ents, ["Q2824648"], 2) == (None, 0)   # depth limit


def test_province_ambiguous_chain_gives_none():
    assert province_of("Q3", entities_with_chain(), ["Q2824648"], 4) == (None, 0)


def landmark_world():
    ents = entities_with_chain()
    ents["Q1"]["claims"].update({"P31": [{"value": "Q50", "rank": "normal"}],
                                 "P571": [{"value": "+1049-00-00T00:00:00Z", "rank": "normal", "precision": 9}]})
    ents.update({"Q50": ent("Q50"), "Q60": ent("Q60", {"P31": ["Q5"], "P140": ["Q70"]}), "Q70": ent("Q70"),
                 "Q200": ent("Q200"), "Q201": ent("Q201")})
    rows = {"Q1": [{"template": "Thông tin chùa", "field": "founder", "raw": "vua Lý Thái Tông",
                    "links": ["Lý Thái Tông"]},
                   {"template": "Thông tin chùa", "field": "thờ phụng", "raw": "Quán Thế Âm", "links": []}],
            "Q2": [{"template": "Thông tin chùa", "field": "founded", "raw": "năm 1300", "links": []}]}
    return ents, rows, {"Lý Thái Tông": "Q60"}


def test_propose_triples_sources_and_counters():
    ents, rows, links = landmark_world()
    triples, counts = propose_triples(["Q1", "Q2"], ents, rows, links, CFG)
    keys = {triple_key(t) for t in triples}
    assert "Q1|P31|Q50|wikidata" in keys
    assert "Q1|P112|Q60|infobox" in keys
    assert "Q1|P131|Q12|derived" in keys                     # depth 3 -> derived
    assert "Q2|P131|Q12|wikidata" in keys                    # depth 1 is a plain Wikidata statement
    assert "Q1|P2348|Q200|derived" in keys                   # P571 = 1049 -> Nhà Lý
    assert "Q2|P2348|Q201|derived" in keys                   # infobox founded = 1300 -> Nhà Trần
    assert "Q60|P140|Q70|wikidata" not in keys               # Q70 is not a context entity
    assert counts["infobox_unlinked"] == 1
    infobox = next(t for t in triples if t["source"] == "infobox")
    assert infobox["evidence"] == "Thông tin chùa|founder=vua Lý Thái Tông"
    assert triples == sorted(triples, key=lambda t: (int(t["h"][1:]), t["r"], int(t["t"][1:])))


def test_propose_triples_prefers_wikidata_over_infobox():
    ents, rows, links = landmark_world()
    ents["Q1"]["claims"]["P112"] = [{"value": "Q60", "rank": "normal"}]
    triples, _ = propose_triples(["Q1"], ents, rows, links, CFG)
    assert [t["source"] for t in triples if t["r"] == "P112"] == ["wikidata"]


def test_context_one_hop_triples():
    ents, rows, links = landmark_world()
    ents["Q1"]["claims"]["P140"] = [{"value": "Q70", "rank": "normal"}]     # Q70 becomes a context entity
    triples, _ = propose_triples(["Q1"], ents, rows, links, CFG)
    assert "Q60|P140|Q70|wikidata" in {triple_key(t) for t in triples}


def test_categorize():
    ents, rows, links = landmark_world()
    triples, _ = propose_triples(["Q1", "Q2"], ents, rows, links, CFG)
    assert categorize("Q1", {"Q1", "Q2"}, ents, triples, CFG) == "landmark"
    assert categorize("Q12", {"Q1", "Q2"}, ents, triples, CFG) == "province"
    assert categorize("Q60", {"Q1", "Q2"}, ents, triples, CFG) == "person"
    assert categorize("Q200", {"Q1", "Q2"}, ents, triples, CFG) == "period"
    assert categorize("Q50", {"Q1", "Q2"}, ents, triples, CFG) == "class"


def test_cap_entities_keeps_highest_degree_then_qid():
    triples = [{"h": "Q1", "r": "P31", "t": "Q50"}, {"h": "Q2", "r": "P31", "t": "Q50"},
               {"h": "Q1", "r": "P140", "t": "Q70"}, {"h": "Q2", "r": "P140", "t": "Q60"}]
    kept, kept_triples, n_ent, n_tri = cap_entities(["Q1", "Q2"], triples, 4)
    assert kept == ["Q50", "Q60"]                            # degree 2 first, then Q60 < Q70 on the tie
    assert (n_ent, n_tri) == (1, 1)
    assert {t["t"] for t in kept_triples} == {"Q50", "Q60"}
    with pytest.raises(ValueError):
        cap_entities(["Q1", "Q2"], triples, 1)


def test_gold_from_pattern():
    triples = [{"h": "Q1", "r": "P31", "t": "Q50"}, {"h": "Q1", "r": "P131", "t": "Q12"},
               {"h": "Q2", "r": "P31", "t": "Q50"}]
    assert gold_from_pattern([["P31", "Q50"]], triples, ["Q2", "Q1"]) == ["Q1", "Q2"]
    assert gold_from_pattern([["P31", "Q50"], ["P131", "Q12"]], triples, ["Q1", "Q2"]) == ["Q1"]
    assert gold_from_pattern([["P31", "Q99"]], triples, ["Q1", "Q2"]) == []
