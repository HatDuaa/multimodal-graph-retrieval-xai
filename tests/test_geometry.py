"""Rule-based count/spatial checks and the LLM JSON reader of the parser comparison."""
import pytest

from src.graph.caption_parsers import parse_llm_json
from src.graph.geometry import (caption_counts, count_score, iou, merge_duplicates, satisfies, spatial_kind,
                                spatial_score)


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    return spacy.load("en_core_web_sm")


def test_caption_counts_reads_number_words_digits_and_pairs(nlp):
    assert caption_counts(nlp("Two brown dogs chase three cats.")) == [(2, "dog"), (3, "cat")]
    assert caption_counts(nlp("A pair of skis next to 4 poles")) == [(2, "ski"), (4, "pole")]
    assert caption_counts(nlp("A dog with one ball.")) == []


def test_duplicate_boxes_are_merged_before_counting():
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 10, 10]]
    assert iou(boxes[0], boxes[1]) > .5
    assert len(merge_duplicates(boxes)) == 2
    assert count_score(2, boxes) == 1.
    assert count_score(4, boxes) == .5
    assert count_score(2, []) is None


def test_spatial_predicates_follow_box_geometry():
    left, right = [0, 40, 10, 10], [30, 40, 10, 10]
    top, bottom = [30, 0, 10, 10], [30, 80, 10, 10]
    assert spatial_kind("Next To ") == "beside" and spatial_kind("underneath") == "below"
    assert spatial_kind("in front of") is None and spatial_kind("behind") is None
    assert satisfies("left", left, right) and not satisfies("right", left, right)
    assert satisfies("above", top, bottom) and satisfies("below", bottom, top)
    assert satisfies("on_top", top, bottom) and not satisfies("on_top", left, bottom)
    assert satisfies("beside", left, [15, 42, 10, 10]) and not satisfies("beside", left, right)
    assert not satisfies("beside", left, [15, 80, 10, 10])
    assert spatial_score("left", [left], [right, [-50, 40, 10, 10]]) == .5
    assert spatial_score("left", [], [right]) is None


def test_llm_json_reader_drops_invalid_entities_and_relations():
    text = ('Sure: {"entities": [{"name": "dog", "count": 2}, {"name": ""}, {"name": "ball", "attributes": ["red", 3]}],'
            ' "relations": [{"subject": 0, "predicate": "with", "object": 2}, {"subject": 0, "predicate": "on",'
            ' "object": 1}, {"subject": 2, "predicate": "x", "object": 2}]}')
    graph = parse_llm_json(text)
    assert [e["name"] for e in graph["entities"]] == ["dog", "ball"]
    assert graph["entities"][0]["count"] == 2 and graph["entities"][1]["attributes"] == ["red"]
    assert graph["relations"] == [{"subject": 0, "predicate": "with", "object": 1}]
    assert parse_llm_json("no json here") is None
