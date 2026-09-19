"""Scene graph normalisation, VG150 filtering and caption coverage. No spaCy, no real data."""
import pytest

from src.graph.coverage import (canon_query, coverage, graph_stats, image_index, parser_stats, pool_selectivity,
                                summarize, top_misses)
from src.graph.scene_graph import Vocab, build_scene_graph, label_pairs, label_triples, load_alias, normalize_label


@pytest.fixture
def vocab(tmp_path):
    (tmp_path / "oa.txt").write_text("person,human,people\ncar,automobile\nball,balls\n", encoding="utf-8")
    (tmp_path / "pa.txt").write_text("on,on top of\nholding,holds,hold\n", encoding="utf-8")
    return Vocab({"person", "car", "ball"}, {"on", "holding"}, load_alias(tmp_path / "oa.txt"), load_alias(tmp_path / "pa.txt"))


def test_normalize_label():
    assert normalize_label("  ( A   B! ) ") == "a b"
    assert normalize_label("ON") == "on"


def test_alias_first_token_is_canonical_and_chains_resolve(tmp_path):
    (tmp_path / "a.txt").write_text("person,human\nhuman,guy\n", encoding="utf-8")
    alias = load_alias(tmp_path / "a.txt")
    assert alias == {"person": "person", "human": "person", "guy": "person"}


def test_vg150_filter_drops_unknown_objects_and_relations(vocab):
    objects = {"image_id": 1, "objects": [{"object_id": 1, "names": ["human"]}, {"object_id": 2, "names": ["tree"]},
                                          {"object_id": 3, "names": ["balls"]}]}
    rels = {"image_id": 1, "relationships": [
        {"predicate": "HOLDS", "subject": {"object_id": 1, "name": "human"}, "object": {"object_id": 3, "names": ["balls"]}},
        {"predicate": "near", "subject": {"object_id": 1, "name": "human"}, "object": {"object_id": 3, "names": ["balls"]}},
        {"predicate": "on", "subject": {"object_id": 3, "names": ["balls"]}, "object": {"object_id": 2, "names": ["tree"]}}]}
    g = build_scene_graph(objects, rels, vocab, vg150_only=True)
    assert [o["name"] for o in g["objects"]] == ["person", "ball"]
    assert label_triples(g) == {("person", "holding", "ball")}

    g_open = build_scene_graph(objects, rels, vocab, vg150_only=False)
    assert label_triples(g_open) == {("person", "holding", "ball"), ("person", "near", "ball"), ("ball", "on", "tree")}


def test_relation_endpoint_missing_from_objects_is_added_and_duplicates_dropped(vocab):
    objects = {"image_id": 7, "objects": [{"object_id": 1, "names": ["person"]}]}
    rel = {"predicate": "on top of", "subject": {"object_id": 1, "name": "person"}, "object": {"object_id": 9, "name": "automobile"}}
    g = build_scene_graph(objects, {"image_id": 7, "relationships": [rel, rel]}, vocab, vg150_only=True)
    assert g["image_id"] == 7
    assert [o["name"] for o in g["objects"]] == ["person", "car"]
    assert g["relations"] == [{"subject": 0, "predicate": "on", "object": 1}]
    assert label_pairs(g) == {("person", "car")}


def test_image_without_annotations_gives_empty_graph(vocab):
    g = build_scene_graph({"image_id": 5}, {"image_id": 5}, vocab, vg150_only=True)
    assert g == {"image_id": 5, "objects": [], "relations": []}


def _graph(image_id, names, relations):
    return {"image_id": image_id, "objects": [{"id": i, "name": n} for i, n in enumerate(names)],
            "relations": [{"subject": s, "predicate": p, "object": o} for s, p, o in relations]}


def _query(names, relations):
    return {"objects": [{"id": i, "name": n} for i, n in enumerate(names)],
            "relations": [{"subject": s, "predicate": p, "predicate_lemma": lemma, "object": o} for s, p, lemma, o in relations]}


def test_coverage_levels(vocab):
    index = image_index(_graph(1, ["person", "ball", "car"], [(0, "holding", 1), (1, "on", 2)]))
    # "people holds balls": alias on both ends and on the predicate -> exact triple
    exact = coverage(canon_query(_query(["people", "balls"], [(0, "holds", "hold", 1)]), vocab), index)
    assert exact == {"entity": [True, True], "pair": [True], "either_pair": [True], "triple": [True]}
    # reversed direction and an unknown predicate: only the undirected pair matches
    loose = coverage(canon_query(_query(["car", "ball", "dog"], [(0, "under", "under", 1)]), vocab), index)
    assert loose == {"entity": [True, True, False], "pair": [False], "either_pair": [True], "triple": [False]}
    # the lemma is tried when the raw predicate is unknown
    lemma = coverage(canon_query(_query(["person", "ball"], [(0, "gripping", "hold", 1)]), vocab), index)
    assert lemma["triple"] == [True]


def test_summarize_and_stats(vocab):
    results = [{"entity": [True, False], "pair": [True], "either_pair": [True], "triple": [False]},
               {"entity": [True], "pair": [], "either_pair": [], "triple": []}]
    s = summarize(results)
    assert s["entity_item_share"] == round(2 / 3, 4) and s["captions_with_entity_hit"] == 1.0
    assert s["pair_items"] == 1 and s["captions_with_pair_hit"] == 0.5 and s["captions_with_triple_hit"] == 0.0
    assert s["captions_with_all_entities_hit"] == 0.5

    g = graph_stats([_graph(1, ["person", "ball"], [(0, "holding", 1)]), _graph(2, [], [])])
    assert g["objects_mean"] == 1 and g["share_images_without_relations"] == 0.5
    p = parser_stats([_query(["a", "b"], [(0, "on", "on", 1)]), _query([], [])])
    assert p["share_with_entity"] == 0.5 and p["relations_mean"] == 0.5


def test_pool_selectivity_and_top_misses(vocab):
    graphs = [_graph(1, ["person", "ball"], [(0, "holding", 1)]), _graph(2, ["person", "car"], [(0, "on", 1)]),
              _graph(3, ["ball"], []), _graph(4, ["car"], [])]
    indexes = {g["image_id"]: image_index(g) for g in graphs}
    queries = [_query(["person", "ball", "dog"], [(0, "holds", "hold", 1)])]
    canon = [canon_query(q, vocab) for q in queries]
    sel = pool_selectivity(canon, [1], indexes)
    assert sel["any_entity"]["pool_share_mean"] == 0.75        # images 1, 2, 3
    assert sel["all_entities"]["captions_with_gold_hit"] == 0  # no image has a dog
    assert sel["any_pair"]["pool_share_mean"] == 0.25 and sel["any_triple"]["pool_share_mean"] == 0.25

    misses = top_misses(queries, canon, [coverage(canon[0], indexes[2])])
    assert misses["entities_not_in_gold_image"] == [("ball", 1), ("dog", 1)]
    assert misses["predicates_without_exact_match"] == [("holds", 1)]
