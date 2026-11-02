from src.data.vnmmkg_wikidata import (inception_year, qid_key, rank_candidates, rejection_reason,
                                      trim_entity, truthy)

CFG = {"country": "Q881", "landmark_classes": ["Q5393308", "Q44539"], "exclude_if_has": ["P195", "P189"],
       "max_landmark_candidates": 2}


def claim(pid, value, rank="normal", kind="wikibase-entityid", precision=None):
    if kind == "wikibase-entityid":
        datavalue = {"type": kind, "value": {"id": value}}
    elif kind == "time":
        datavalue = {"type": kind, "value": {"time": value, "precision": precision}}
    else:
        datavalue = {"type": "string", "value": value}
    return {"mainsnak": {"snaktype": "value", "property": pid, "datavalue": datavalue}, "rank": rank}


def raw_entity(qid="Q1", sitelinks=3, claims=None):
    return {
        "id": qid, "lastrevid": 7,
        "labels": {"vi": {"value": "Chùa Một Cột"}, "en": {"value": "One Pillar Pagoda"}},
        "aliases": {"vi": [{"value": "Diên Hựu tự"}]},
        "descriptions": {"vi": {"value": "chùa ở Hà Nội"}},
        "sitelinks": {f"site{i}": {} for i in range(sitelinks - 1)} | {"viwiki": {"title": "Chùa Một Cột"}},
        "claims": claims if claims is not None else {
            "P17": [claim("P17", "Q881")], "P31": [claim("P31", "Q5393308")],
            "P373": [claim("P373", "One Pillar Pagoda", kind="string")],
            "P999": [claim("P999", "Q5")]},
    }


def test_trim_entity_keeps_fields_and_selected_pids():
    e = trim_entity(raw_entity(), {"P17", "P31", "P373"})
    assert e["qid"] == "Q1" and e["label_vi"] == "Chùa Một Cột" and e["label_en"] == "One Pillar Pagoda"
    assert e["aliases_vi"] == ["Diên Hựu tự"] and e["viwiki_title"] == "Chùa Một Cột" and e["sitelinks"] == 3
    assert set(e["claims"]) == {"P17", "P31", "P373"}
    assert e["claims"]["P373"] == [{"value": "One Pillar Pagoda", "rank": "normal"}]


def test_trim_entity_missing_returns_none():
    assert trim_entity({"id": "Q9", "missing": ""}, {"P31"}) is None


def test_trim_entity_skips_novalue_snaks():
    raw = raw_entity(claims={"P31": [{"mainsnak": {"snaktype": "novalue", "property": "P31"}, "rank": "normal"}]})
    assert trim_entity(raw, {"P31"})["claims"] == {}


def test_truthy_prefers_preferred_and_drops_deprecated():
    e = trim_entity(raw_entity(claims={"P131": [claim("P131", "Q10"), claim("P131", "Q11", rank="preferred"),
                                                claim("P131", "Q12", rank="deprecated")]}), {"P131"})
    assert truthy(e, "P131") == ["Q11"]
    e = trim_entity(raw_entity(claims={"P131": [claim("P131", "Q10"), claim("P131", "Q12", rank="deprecated")]}),
                    {"P131"})
    assert truthy(e, "P131") == ["Q10"]
    assert truthy(e, "P999") == []


def test_inception_year_needs_year_precision():
    def with_time(value, precision):
        return trim_entity(raw_entity(claims={"P571": [claim("P571", value, kind="time", precision=precision)]}),
                           {"P571"})
    assert inception_year(with_time("+1049-00-00T00:00:00Z", 9)) == 1049
    assert inception_year(with_time("+1100-00-00T00:00:00Z", 7)) is None     # century precision
    assert inception_year(with_time("-0200-00-00T00:00:00Z", 9)) is None     # BCE
    assert inception_year(trim_entity(raw_entity(claims={}), {"P571"})) is None


def test_rejection_reasons():
    keep = {"P17", "P31", "P373", "P1435", "P195", "P189"}
    assert rejection_reason(trim_entity(raw_entity(), keep), CFG) is None
    no_cat = raw_entity()
    del no_cat["claims"]["P373"]
    assert rejection_reason(trim_entity(no_cat, keep), CFG) == "no_commons_category"
    drum = raw_entity()
    drum["claims"]["P195"] = [claim("P195", "Q77")]
    assert rejection_reason(trim_entity(drum, keep), CFG) == "movable_object"
    other = raw_entity()
    other["claims"]["P31"] = [claim("P31", "Q3947")]
    assert rejection_reason(trim_entity(other, keep), CFG) == "not_landmark_class"
    other["claims"]["P1435"] = [claim("P1435", "Q100")]          # any heritage designation qualifies
    assert rejection_reason(trim_entity(other, keep), CFG) is None
    no_wiki = raw_entity()
    del no_wiki["sitelinks"]["viwiki"]
    assert rejection_reason(trim_entity(no_wiki, keep), CFG) == "no_viwiki"


def test_rank_candidates_sorts_by_sitelinks_then_qid_and_caps():
    keep = {"P17", "P31", "P373"}
    ents = {q: trim_entity(raw_entity(q, s), keep) for q, s in [("Q30", 5), ("Q4", 9), ("Q20", 5)]}
    ranked, rejected = rank_candidates(ents, ["Q30", "Q4", "Q20", "Q999"], CFG)
    assert ranked == ["Q4", "Q20"]                                # Q20 < Q30 on the tie, cap 2
    assert rejected == {"missing": 1}
    assert qid_key("Q20") < qid_key("Q30") and qid_key("Q4") < qid_key("Q20")
