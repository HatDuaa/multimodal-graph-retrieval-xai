from src.graph.parse_query import parse_caption
from src.graph.scene_graph import Vocab, build_scene_graph, normalize_label


def test_attribute_normalization_and_join_by_object_id():
    vocab = Vocab({'horse'}, set(), {}, {})
    objects = {'image_id': 1, 'objects': [{'object_id': 9, 'names': ['horse']}]}
    rels = {'image_id': 1, 'relationships': []}
    graph = build_scene_graph(objects, rels, vocab, False, True)
    assert graph['objects'][0]['source_object_id'] == 9
    vals = [normalize_label(x) for x in ['White', 'white', 'horse', ''] if normalize_label(x) and normalize_label(x) != 'horse']
    assert vals == ['white', 'white']


def test_query_attributes_and_legacy_shape():
    graph = parse_caption('A large white horse beside a young man.', with_attributes=True)
    assert graph['objects'][0]['attributes'] == ['large', 'white']
    assert graph['objects'][1]['attributes'] == ['young']
    old = parse_caption('A large white horse beside a young man.')
    assert all('attributes' not in obj for obj in old['objects'])
    assert [obj['name'] for obj in old['objects']] == ['horse', 'man']


def test_d_variant_keeps_an_attribute_both_ends_share_on_each_end():
    from scripts.checks.attribute_probe import variant_parts
    graph = {'objects': [{'id': 0, 'name': 'dog', 'attributes': ['white']},
                         {'id': 1, 'name': 'cat', 'attributes': ['white', 'small']}],
             'relations': [{'subject': 0, 'predicate': 'on', 'object': 1}]}
    _, triples = variant_parts(graph, False, 'D_parts_triples')
    assert triples == ['dog on cat', 'white dog on cat', 'dog on white cat', 'dog on small cat']
    _, query = variant_parts(graph, True, 'D_parts_triples')
    assert query == ['white dog on white small cat']


def test_query_variants_drop_empty_names_like_query_instance_parts():
    from scripts.checks.attribute_probe import variant_parts
    graph = {'objects': [{'id': 0, 'name': '', 'attributes': ['red']}, {'id': 1, 'name': 'dog', 'attributes': []}],
             'relations': []}
    for variant in ('B_concat', 'C_parts', 'D_parts_triples'):
        assert variant_parts(graph, True, variant)[0] == ['dog']
