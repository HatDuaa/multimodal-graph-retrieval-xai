"""Group nodes and counts (experiment 4): caption count rules, image counts, group naming, the count table g."""
import numpy as np
import pytest
import spacy
import torch
from torch.nn import functional as F

from src.data.graph_store import GraphStore
from src.explain.graph_explainer import score
from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS, caption_classes, compatible, image_bucket, image_counts
from src.graph.dedup import group_by_role, majority_name, same_type
from src.models.graph_reranker import GraphReranker, channel_score
from src.train_graph_reranker import build_model

NLP = spacy.load("en_core_web_sm", disable=["ner"])


def classes(text, names, attributes=None):
    objects = [{"name": n, "attributes": (attributes or {}).get(n, [])} for n in names]
    return [CAPTION_CLASSES[c] for c, _ in caption_classes(NLP(text), objects)]


def test_image_buckets_and_compatibility():
    assert [IMAGE_BUCKETS[image_bucket(n)] for n in (1, 2, 3, 4, 5, 9)] == ["1", "2", "3-4", "3-4", ">=5", ">=5"]
    two, at_least_two, at_least_three = (CAPTION_CLASSES.index(c) for c in ("2", ">=2", ">=3"))
    assert compatible(two, 1) and not compatible(two, 2)
    assert compatible(at_least_two, 1) and not compatible(at_least_two, 0)
    assert compatible(at_least_three, 2) and not compatible(at_least_three, 1)
    assert compatible(0, 3) is None


@pytest.mark.parametrize("text, names, expected", [
    ("Three young men are standing next to four bicycles.", ["man", "bicycle"], ["3-4", "3-4"]),
    ("Two dogs and a cat on a sofa.", ["dog", "cat", "sofa"], ["2", "unknown", "unknown"]),
    ("A man holding a pair of skis.", ["man", "ski"], ["unknown", "2"]),
    ("A couple of dogs play with both kids.", ["dog", "kid"], ["2", "2"]),
    ("A lone tree in a field.", ["tree", "field"], ["1", "unknown"]),
    ("A group of three men next to bikes.", ["man", "bike"], ["3-4", ">=2"]),
    ("A herd of elephants and several birds near many cars.", ["elephant", "bird", "car"], [">=3", ">=3", ">=3"]),
    ("Lots of apples in a bowl.", ["apple", "bowl"], [">=3", "unknown"]),
    ("A man in pants walks dogs.", ["man", "pant", "dog"], ["unknown", "unknown", ">=2"]),
    ("Seven cows graze.", ["cow"], [">=5"]),
])
def test_caption_count_rules(text, names, expected):
    assert classes(text, names) == expected


def test_caption_falls_back_to_factual_count_and_attributes():
    objects = [{"name": "zebra", "count": 3, "attributes": []}, {"name": "sheep", "attributes": ["group of"]},
               {"name": "crowd", "attributes": []}, {"name": "", "attributes": []}]
    out = caption_classes(NLP("Animals in a field."), objects)
    assert [CAPTION_CLASSES[c] for c, _ in out] == ["3-4", ">=3", ">=3", "unknown"]
    assert [rule for _, rule in out] == ["factual_count", "group_many", "group_many", "empty"]


def node(i, name, synset, box, **extra):
    return {"id": i, "name": name, "synset": synset, "box": box, "count": 1, "boxes": 1, "members": [i], **extra}


def riders():
    """Two men and a (bigger) skier all riding one horse: one group by the shared (riding, horse) role."""
    objects = [node(0, "man", "man.n.01", [0, 0, 10, 10]), node(1, "skier", "skier.n.01", [20, 0, 30, 30]),
               node(2, "man", "man.n.01", [60, 0, 10, 10]), node(3, "horse", "horse.n.01", [0, 40, 100, 50]),
               node(4, "dog", "dog.n.01", [200, 40, 10, 10])]
    relations = [{"subject": s, "predicate": "riding", "object": 3} for s in (0, 1, 2)]
    return {"objects": objects, "relations": relations}


def test_group_takes_the_majority_label_not_the_largest_box():
    grouped, _ = group_by_role(riders(), "ab", naming=majority_name)
    group = next(o for o in grouped["objects"] if o.get("group"))
    assert (group["name"], group["count"], group["naming"], group["members"]) == ("man", 3, "majority", [0, 1, 2])
    assert group["box"] == [0, 0, 70, 30]
    assert grouped["relations"] == [{"subject": 0, "predicate": "riding", "object": 1, "source": 0}]
    old, _ = group_by_role(riders(), "ab")
    assert next(o for o in old["objects"] if o.get("group"))["name"] == "skier"


def test_tied_labels_take_their_lowest_common_hypernym():
    objects = [node(0, "man", "man.n.01", [0, 0, 5, 5]), node(1, "woman", "woman.n.01", [9, 0, 5, 5])]
    assert majority_name(objects, [0, 1]) == ("adult", "adult.n.01", "hypernym")
    objects = [node(0, "foo", None, [0, 0, 5, 5]), node(1, "bar", None, [9, 0, 5, 5])]
    assert majority_name(objects, [0, 1]) == ("bar", None, "tie_first")


def test_image_counts_use_group_members_or_same_type_instances():
    grouped, _ = group_by_role(riders(), "ab", naming=majority_name)
    counts = dict(zip((o["name"] for o in grouped["objects"]), image_counts(grouped["objects"], same_type)))
    assert counts == {"man": 3, "horse": 1, "dog": 1}
    merged_only = riders()["objects"]
    assert image_counts(merged_only, same_type) == [3, 3, 3, 1, 1]  # every person counts the three people


def count_store():
    torch.manual_seed(2)
    store = GraphStore.__new__(GraphStore)
    store.part_dim = 512
    keys = ['node:dog', 'node:ball', 'rel:near', 'triple:dog near ball']
    store.part_row = {key: i for i, key in enumerate(keys)}
    store.part_vectors = F.normalize(torch.randn(4, 512), dim=-1).numpy()
    store.graphs = {1: {'objects': [{'id': 0, 'name': 'dog', 'count_bucket': 1}], 'relations': []},
                    2: {'objects': [{'id': 0, 'name': 'dog', 'count_bucket': 0}, {'id': 1, 'name': 'ball', 'count_bucket': 3}],
                        'relations': [{'subject': 0, 'predicate': 'near', 'object': 1}]}}
    store.queries = {'q': {'objects': [{'id': 0, 'name': 'dog', 'count_class': CAPTION_CLASSES.index("2")}],
                           'relations': []}}
    store.caption_vectors = F.normalize(torch.randn(1, 512), dim=-1).numpy()
    store.caption_row = {'q': 0}
    store.arrays = {'val': {'candidate_ids': np.array([[1, 2]]), 'clip_scores': np.array([[.5, .4]], dtype='float32'),
                            'gold_rank': np.array([1]), 'gold': np.array([1])}}
    store.caption_location = {'q': ('val', 0)}
    return store


def test_batches_carry_count_buckets_and_classes():
    store = count_store()
    store.set_structure(global_node=True)
    batch = store.batch(['q'])
    assert batch['queries'].count_index.tolist() == [CAPTION_CLASSES.index("2"), 0]  # the global node takes 0
    assert sorted(batch['images'].count_index.tolist()) == [0, 0, 0, 1, 3]


def test_zero_table_scores_like_the_model_without_it_and_g_adds_to_similarity():
    store = count_store()
    torch.manual_seed(4)
    plain = GraphReranker(dropout=0., fusion='raw').eval()
    counted = GraphReranker(dropout=0., fusion='raw', count_table=True).eval()
    counted.load_state_dict(plain.state_dict(), strict=False)
    batch = store.batch(['q'])
    _, before = plain(batch)
    _, zero = counted(batch)
    torch.testing.assert_close(before['objects'], zero['objects'])
    two, bucket_two = CAPTION_CLASSES.index("2"), IMAGE_BUCKETS.index("2")
    with torch.no_grad():
        counted.count_table[two - 1, bucket_two] = .5
    _, after = counted(batch)
    image_one = 0 if batch['image_ids'][0] == 1 else 1
    row = int(batch['candidate_index'][0, 0])  # candidate 0 is image 1: one dog, bucket "2"
    assert row == image_one
    torch.testing.assert_close(after['objects'][0, 0], before['objects'][0, 0] + .5)
    assert counted.counts_table()[0].abs().sum() == 0  # the unknown row stays 0
    counted.train()
    counted(batch)[0][0, 0].backward()
    assert counted.count_table.grad[two - 1, bucket_two] != 0


def test_channel_score_without_table_is_unchanged():
    torch.manual_seed(0)
    q, i = F.normalize(torch.randn(1, 2, 8), dim=-1), F.normalize(torch.randn(3, 2, 8), dim=-1)
    args = (q, torch.ones(1, 2, dtype=torch.bool), i, torch.ones(3, 2, dtype=torch.bool), torch.tensor([[0, 2]]),
            torch.tensor(.05), torch.zeros(8))
    zero = torch.zeros(7, 4)
    torch.testing.assert_close(channel_score(*args), channel_score(*args, torch.zeros(1, 2, dtype=torch.long),
                                                                   torch.zeros(3, 2, dtype=torch.long), zero))


def test_explanations_show_the_count_term():
    store = count_store()
    model = GraphReranker(dropout=0., fusion='raw', count_table=True).eval()
    with torch.no_grad():
        model.count_table[CAPTION_CLASSES.index("2") - 1, IMAGE_BUCKETS.index("2")] = .1
    row = score(model, store, 'q', 1)['objects'][0]
    assert (row['query_count'], row['image_count']) == ("2", "2")
    assert row['count_bias'] == pytest.approx(.1)
    assert build_model({'adaptive_weights': True, 'use_gat': True, 'use_triple_channel': True, 'use_edges': True,
                        'query_graph_encoder': True, 'count_table': True}, 'cpu').counts_table().shape == (7, 4)


def test_count_buckets_beside_the_graphs_keep_labels_and_order():
    from src.data.graph_store import attach_count_buckets
    graphs = {7: {'objects': [{'id': 0, 'name': 'man'}, {'id': 1, 'name': 'skier'}, {'id': 2, 'name': 'horse'}], 'relations': []}}
    attach_count_buckets(graphs, {'7': {'counts': [2, 2, 1], 'buckets': [1, 1, 0]}})
    assert [(o['name'], o['count'], o['count_bucket']) for o in graphs[7]['objects']] == \
        [('man', 2, 1), ('skier', 2, 1), ('horse', 1, 0)]
    assert GraphStore._graph_parts(graphs[7])['count_index'] == [1, 1, 0]
    with pytest.raises(ValueError):
        attach_count_buckets(graphs, {'7': {'counts': [1], 'buckets': [0]}})
