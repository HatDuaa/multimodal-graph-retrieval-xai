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
