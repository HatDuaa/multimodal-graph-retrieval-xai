"""Duplicate merging and role grouping of Visual Genome image graphs (graph cleaning audit)."""
import pytest

pytest.importorskip("nltk")
from src.graph.dedup import (geometry_class, graph_stats, group_by_role, merge_duplicates, role_pair_counts,  # noqa: E402
                             same_type)


def node(i, name, synset, box):
    return {"id": i, "name": name, "synset": synset, "box": box, "count": 1, "members": [i]}


def wordnet_ready():
    try:
        from nltk.corpus import wordnet as wn
        wn.synset("dog.n.01")
        return True
    except LookupError:
        return False


pytestmark = pytest.mark.skipif(not wordnet_ready(), reason="WordNet data not installed")


def test_same_type_uses_person_and_direct_hypernyms_only():
    assert same_type("man.n.01", "skier.n.01", "man", "skier")
    assert same_type("dog.n.01", "animal.n.01", "dog", "animal")
    assert not same_type("dog.n.01", "cat.n.01", "dog", "cat")
    assert same_type(None, None, "sky", "sky") and same_type(None, "sky.n.01", "sky", "sky")
    assert not same_type(None, "sky.n.01", "sky", "cloud")  # without a synset on one side, names decide


def test_merge_joins_overlapping_boxes_of_one_thing_and_keeps_distinct_ones():
    graph = {"objects": [node(0, "horse", "horse.n.01", [0, 0, 100, 100]), node(1, "horse", "horse.n.01", [2, 2, 100, 100]),
                         node(2, "horse", "horse.n.01", [300, 0, 100, 100]), node(3, "man", "man.n.01", [10, 0, 50, 80]),
                         node(4, "beach", "beach.n.01", [0, 100, 500, 100])],
             "relations": [{"subject": 3, "predicate": "riding", "object": 0},
                           {"subject": 1, "predicate": "on", "object": 4},
                           {"subject": 0, "predicate": "near", "object": 1}]}
    merged, mapping = merge_duplicates(graph, .7)
    assert len(merged["objects"]) == 4 and mapping[0] == mapping[1] != mapping[2]
    horse = merged["objects"][mapping[0]]
    assert horse["count"] == 1 and horse["boxes"] == 2  # two boxes of one horse are still one horse
    assert {(r["subject"], r["predicate"], r["object"]) for r in merged["relations"]} == {
        (mapping[3], "riding", mapping[0]), (mapping[0], "on", mapping[4])}
    stats = graph_stats(merged)
    assert stats["largest_component"] == 3 and stats["isolated"] == 1
    assert graph_stats(graph)["components"] == 2


def test_geometry_classes():
    anchor = [100, 100, 50, 50]
    assert geometry_class([110, 110, 20, 20], anchor) == "overlap"
    assert geometry_class([60, 100, 30, 50], anchor) == "beside"
    assert geometry_class([0, 100, 40, 50], anchor) == "left"  # gap 60 > wider box width 50
    assert geometry_class([0, 300, 40, 40], anchor) == "below"
    assert geometry_class([-300, 120, 40, 40], anchor) == "left"


def dogs_and_person():
    person = node(0, "man", "man.n.01", [200, 0, 100, 300])
    dogs = [node(1, "dog", "dog.n.01", [0, 200, 40, 40]), node(2, "dog", "dog.n.01", [50, 210, 40, 40]),
            node(3, "dog", "dog.n.01", [20, 260, 40, 40]), node(4, "dog", "dog.n.01", [900, 900, 40, 40])]
    relations = [{"subject": d, "predicate": "in front of", "object": 0} for d in (1, 2)]
    return {"objects": [person] + dogs, "relations": relations}


def test_role_grouping_by_predicate_and_by_geometry():
    graph = dogs_and_person()
    by_a, _ = group_by_role(graph, "a")
    assert sorted(o["count"] for o in by_a["objects"]) == [1, 1, 1, 2]
    by_ab, mapping = group_by_role(graph, "ab")
    assert sorted(o["count"] for o in by_ab["objects"]) == [1, 1, 3]  # dog 3 sits like dogs 1-2 but has no edge
    assert mapping[1] == mapping[3] != mapping[4]
    counts = role_pair_counts(graph)
    assert counts["pairs_a"] == 1 and counts["pairs_b"] >= 3
