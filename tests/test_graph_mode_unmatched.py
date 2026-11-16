from src.data.graph_store import GraphStore
from src.service.graph_mode import unmatched_parts

IMAGE = {"objects": [{"id": 7, "name": "player"}, {"id": 8, "name": "base"}],
         "relations": [{"subject": 7, "predicate": "on", "object": 8}]}


def parts(query, image=IMAGE):
    return GraphStore._graph_parts(query, query=True), GraphStore._graph_parts(image)


def test_pronoun_object_and_its_relation_are_reported():
    # "A baseball runner slows as he arrives at third base": the only triple hangs on the pronoun "he"
    query = {"objects": [{"id": 0, "name": "baseball runner"}, {"id": 1, "name": "he"}, {"id": 2, "name": "base"}],
             "relations": [{"subject": 1, "predicate": "at", "object": 2}]}
    q, i = parts(query)
    out = unmatched_parts(query, q, i)
    assert out["objects"] == [{"query_part": "he", "reason": "pronoun"}]
    assert out["triples"] == [{"query_part": ("he", "at", "base"), "reason": "pronoun"}]


def test_scorable_parts_have_no_entry_and_empty_label_is_masked():
    query = {"objects": [{"id": 0, "name": "man"}, {"id": 1, "name": ""}, {"id": 2, "name": "horse"}],
             "relations": [{"subject": 0, "predicate": "Riding", "object": 2}]}
    q, i = parts(query)
    out = unmatched_parts(query, q, i)
    assert out["objects"] == [{"query_part": "", "reason": "empty_label"}]
    assert out["triples"] == []


def test_image_without_triples_reports_every_query_triple():
    query = {"objects": [{"id": 0, "name": "man"}, {"id": 1, "name": "horse"}],
             "relations": [{"subject": 0, "predicate": "riding", "object": 1}]}
    image = {"objects": [{"id": 3, "name": "horse"}], "relations": []}
    q, i = parts(query, image)
    out = unmatched_parts(query, q, i)
    assert out["objects"] == []
    assert out["triples"] == [{"query_part": ("man", "riding", "horse"), "reason": "image_empty"}]
