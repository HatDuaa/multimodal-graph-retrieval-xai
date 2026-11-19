"""Known-answer benchmarks of the level-3 explanations: scores, planting and the synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.explain import synthetic_graphs as synthetic
from src.models.graph_reranker import GraphReranker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xai_ground_truth as gt  # noqa: E402


def test_scores_of_a_perfect_a_reversed_and_a_half_order():
    answer = [("node", 0), ("triple", 0)]
    perfect = gt.gt_scores([("node", 0), ("triple", 0), ("node", 1), ("node", 2)], answer, decoys=[("node", 2)])
    assert perfect == {"gea": 1., "precision": 1., "auroc": 1., "first": 1, "decoy_percentile": 1.}
    reversed_ = gt.gt_scores([("node", 2), ("node", 1), ("triple", 0), ("node", 0)], answer)
    assert reversed_["gea"] == 0 and reversed_["auroc"] == 0 and reversed_["first"] == 3
    half = gt.gt_scores([("node", 0), ("node", 1), ("triple", 0), ("node", 2)], answer)
    assert half["gea"] == pytest.approx(1 / 3) and half["precision"] == .5 and half["auroc"] == .75


def test_planting_adds_the_motifs_and_keeps_the_original():
    graph = {"width": 10, "height": 10, "objects": [{"id": 0, "name": "dog", "box": [1, 1, 2, 2], "count_bucket": 0}],
             "relations": []}
    planted, placed = gt.plant(graph, [("man", "holding", "cup"), ("lamp", "near", "rock")], np.random.default_rng(0))
    assert len(graph["objects"]) == 1 and not graph["relations"]
    assert len(planted["objects"]) == 5 and len(planted["relations"]) == 2
    for i, (s, p, o) in enumerate([("man", "holding", "cup"), ("lamp", "near", "rock")]):
        objects, relation = placed[i]
        assert [planted["objects"][j]["name"] for j in objects] == [s, o]
        rel = planted["relations"][relation]
        assert (rel["subject"], rel["predicate"], rel["object"]) == (planted["objects"][objects[0]]["id"], p,
                                                                     planted["objects"][objects[1]]["id"])
    assert all("box" in obj and "count_bucket" in obj for obj in planted["objects"])


def motif_parts(graph):
    names = {o["id"]: o["name"] for o in graph["objects"]}
    return ({o["name"] for o in graph["objects"]},
            {(names[r["subject"]], r["predicate"], names[r["object"]]) for r in graph["relations"]})


def test_every_negative_misses_a_part_of_the_motif_and_every_part_is_missed_once():
    data = synthetic.make_dataset(0, n_train=20, n_val=5, candidates=12)
    for cid, query in data["queries"].items():
        q_names, q_triples = motif_parts(query)
        split, row = ("train", int(cid.split("_")[1])) if cid.startswith("train") else ("val", int(cid.split("_")[1]))
        arrays = data["arrays"][split]
        gold = int(arrays["gold"][row])
        assert arrays["candidate_ids"][row][arrays["gold_rank"][row] - 1] == gold
        names, triples = motif_parts(data["graphs"][gold])
        assert q_names <= names and q_triples <= triples
        answer = data["answers"][cid]
        assert len(answer["nodes"]) == len(query["objects"]) and len(answer["triples"]) == len(query["relations"])
        missed_names, missed_triples = set(), set()
        for image_id in arrays["candidate_ids"][row]:
            if image_id == gold:
                continue
            names, triples = motif_parts(data["graphs"][int(image_id)])
            lost = (q_names - names) | {t for t in q_triples - triples}
            assert lost, "a negative holds the whole motif"
            missed_names |= q_names - names
            missed_triples |= q_triples - triples
        assert missed_names == q_names and missed_triples == q_triples


def test_the_synthetic_store_batches_for_the_main_architecture():
    data = synthetic.make_dataset(1, n_train=4, n_val=2, candidates=6)
    keys = synthetic.part_keys(data)
    torch.manual_seed(0)
    vectors = F.normalize(torch.randn(len(keys), 512), dim=-1).numpy()
    sentences = F.normalize(torch.randn(len(data["captions"]), 512), dim=-1).numpy()
    store = synthetic.synthetic_store(data, vectors, {k: i for i, k in enumerate(keys)}, sentences)
    model = GraphReranker(dropout=0., fusion="raw", count_table=True, completion_edges=True, global_node=True,
                          edge_geometry=True).eval()
    batch = store.batch(["train_0", "val_1"])
    with torch.no_grad():
        scores, _ = model(batch)
    assert scores.shape == (2, 6) and torch.isfinite(scores).all()
    assert (batch["clip"] == 0).all()


def test_planting_into_an_image_without_objects_gives_the_parts_a_box():
    graph = {"width": 10, "height": 8, "objects": [], "relations": []}
    planted, _ = gt.plant(graph, [("man", "holding", "cup"), ("lamp", "near", "rock")], np.random.default_rng(0))
    assert [obj["box"] for obj in planted["objects"]] == [[0, 0, 10, 8]] * 4


def test_planted_parts_are_not_always_last():
    graph = {"width": 10, "height": 10, "relations": [],
             "objects": [{"id": i, "name": f"thing{i}", "box": [1, 1, 2, 2], "count_bucket": 0} for i in range(6)]}
    last = []
    for seed in range(20):
        planted, placed = gt.plant(graph, [("man", "holding", "cup"), ("lamp", "near", "rock")],
                                   np.random.default_rng(seed))
        objects, relation = placed[0]
        assert [planted["objects"][j]["name"] for j in objects] == ["man", "cup"]
        rel = planted["relations"][relation]
        assert (rel["subject"], rel["object"]) == tuple(planted["objects"][j]["id"] for j in objects)
        last.append(set(objects) | set(placed[1][0]) == set(range(6, 10)))
    assert not all(last)


class Queries:
    def __init__(self, query):
        self.queries = {"c": query}


def test_caption_motifs_skip_pronoun_ends():
    query = {"objects": [{"id": 0, "name": "who"}, {"id": 1, "name": "man"}, {"id": 2, "name": "table"}],
             "relations": [{"subject": 0, "predicate": "at", "object": 2},
                           {"subject": 1, "predicate": "at", "object": 2}]}
    motif, _ = gt.caption_motif(Queries(query), "c", np.random.default_rng(0))
    assert motif == ("man", "at", "table")
    query["relations"] = query["relations"][:1]
    assert gt.caption_motif(Queries(query), "c", np.random.default_rng(0)) is None


def test_json_items_reads_a_list_across_chunk_borders(tmp_path):
    items = [{"id": i, "text": "x" * (i % 7), "nested": [i, {"a": i}]} for i in range(50)]
    path = tmp_path / "items.json"
    path.write_text("  [\n" + " ,\n ".join(__import__("json").dumps(x) for x in items) + "\n]\n", encoding="utf-8")
    for chunk in (7, 64, 1 << 20):
        assert list(gt.json_items(path, chunk=chunk)) == items


def test_a_label_matcher_ignores_parts_appended_to_the_store_after_it():
    data = synthetic.make_dataset(1, n_train=2, n_val=1, candidates=3)
    keys = synthetic.part_keys(data)
    vectors = F.normalize(torch.randn(len(keys), 512), dim=-1).numpy()
    sentences = F.normalize(torch.randn(len(data["captions"]), 512), dim=-1).numpy()
    store = synthetic.synthetic_store(data, vectors, {k: i for i, k in enumerate(keys)}, sentences)
    matcher = gt.matcher_of(store)
    store.append_parts(["node:unicorn"], np.ones((1, 512), dtype=np.float32))
    assert store.part_row["node:unicorn"] == len(keys) and len(store.part_vectors) == len(keys) + 1
    assert matcher("unicorn", "unicorn", None) is not None
