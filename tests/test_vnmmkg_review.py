import json

from scripts.build_vnmmkg_review import render_html, review_payload


def test_review_payload_and_safe_embedding():
    entities = {"Q1": {"label_vi": "Chùa </script> Một Cột", "label_en": None},
                "Q50": {"label_vi": None, "label_en": "pagoda"}}
    commons = {"Q1": {"candidates": [{"file_title": "File:A.jpg", "local_file": "abc.jpg", "license": "CC0",
                                      "artist": "X", "source_url": "https://commons/A"}]}}
    triples = [{"h": "Q1", "r": "P31", "t": "Q50", "source": "wikidata", "evidence": "wikidata:P31"}]
    payload = review_payload(["Q1", "Q2"], entities, commons, triples, {"P31": "là"})
    assert [lm["qid"] for lm in payload["landmarks"]] == ["Q1"]               # Q2 has no commons file
    assert payload["landmarks"][0]["images"][0] == {"file_title": "File:A.jpg", "local": "abc.jpg",
                                                    "license": "CC0", "artist": "X",
                                                    "source_url": "https://commons/A"}
    assert payload["triples"][0]["key"] == "Q1|P31|Q50|wikidata"
    assert payload["triples"][0]["t_label"] == "pagoda" and payload["triples"][0]["r_label"] == "là"
    page = render_html(payload)
    assert "Chùa <\\/script> Một Cột" in page and "__DATA__" not in page
    start = page.index("const DATA = ") + len("const DATA = ")
    embedded = page[start:page.index(";\n", start)].replace("<\\/", "</")
    assert json.loads(embedded) == payload
