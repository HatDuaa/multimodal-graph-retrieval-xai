import json

import numpy as np
import torch

from src.data.graph_store import GraphStore
from src.train_graph_reranker import build_parser, train

NAMES = ['man', 'horse', 'dog', 'ball', 'tree']
PREDICATES = ['riding', 'near', 'holding']


def write_tiny_dataset(root):
    rng = np.random.default_rng(0)
    for folder in ('features', 'graphs', 'processed', 'splits'):
        (root / folder).mkdir()
    graphs, parts = {}, set()
    for image_id in range(1, 9):
        objects = [{'id': k, 'name': NAMES[(image_id + k) % 5]} for k in range(3)]
        relations = [{'subject': 0, 'predicate': PREDICATES[image_id % 3], 'object': 1},
                     {'subject': 2, 'predicate': PREDICATES[(image_id + 1) % 3], 'object': 0}]
        graphs[image_id] = {'image_id': image_id, 'split': 'train' if image_id <= 5 else 'val',
                            'objects': objects, 'relations': relations}
    queries = {f'c{i}': dict(graphs[1 + i % 8], split=None) for i in range(12)}
    for graph in list(graphs.values()) + list(queries.values()):
        names = {o['id']: o['name'] for o in graph['objects']}
        parts.update(f'node:{o["name"]}' for o in graph['objects'])
        for r in graph['relations']:
            parts.update({f'rel:{r["predicate"]}', f'triple:{names[r["subject"]]} {r["predicate"]} {names[r["object"]]}'})
    def table(stem, ids):
        vectors = rng.normal(size=(len(ids), 512)).astype('float32')
        np.save(root / 'features' / f'{stem}.npy', vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        (root / 'features' / f'{stem}.ids.json').write_text(json.dumps({'ids': ids}))
    table('vg_coco_graph_parts', sorted(parts))
    table('vg_coco_caption', list(queries))
    with (root / 'graphs' / 'vg_coco_scene_graphs.jsonl').open('w') as handle:
        for graph in graphs.values():
            handle.write(json.dumps(graph) + '\n')
    for split, ids in (('train', [f'c{i}' for i in range(8)]), ('val', [f'c{i}' for i in range(8, 12)])):
        (root / 'processed' / f'query_graphs_{split}.json').write_text(json.dumps({c: queries[c] for c in ids}))
        pool = np.arange(1, 6) if split == 'train' else np.arange(4, 9)
        candidates = np.stack([np.roll(pool, i)[:4] for i in range(len(ids))])
        np.savez(root / 'processed' / f'candidates_{split}.npz', caption_ids=np.array(ids), gold=candidates[:, 1],
                 candidate_ids=candidates, clip_scores=rng.normal(size=candidates.shape).astype('float32'),
                 gold_rank=np.full(len(ids), 2), group_index=np.zeros(len(ids), dtype=np.int64))
    return {'paths': {'features': 'features', 'graphs': 'graphs', 'processed': 'processed', 'splits': 'splits'}}


def comparable(run_dir):
    lines = [json.loads(line) for line in (run_dir / 'log.jsonl').read_text().splitlines()]
    drop = ('seconds', 'rss_peak_bytes', 'gpu_memory_peak_bytes')
    return [{k: v for k, v in line.items() if k not in drop} for line in lines]


def test_resume_after_one_epoch_equals_uninterrupted_run(tmp_path):
    cfg = write_tiny_dataset(tmp_path)
    store = GraphStore(cfg, root=tmp_path)
    common = ['--run-name', 'x', '--epochs', '3', '--batch-size', '4', '--micro-batch-size', '2', '--seed', '5']
    straight = tmp_path / 'straight'
    train(build_parser().parse_args(common), store, straight, torch.device('cpu'), cfg)
    resumed = tmp_path / 'resumed'
    assert train(build_parser().parse_args(common + ['--stop-after-epoch', '1']), store, resumed,
                 torch.device('cpu'), cfg) is None
    assert not (resumed / 'metrics.json').exists()
    torch.manual_seed(123)  # a fresh process would not share the RNG position
    train(build_parser().parse_args(common + ['--resume']), store, resumed, torch.device('cpu'), cfg)
    assert comparable(straight) == comparable(resumed)
    a = torch.load(straight / 'checkpoint_last.pt', weights_only=False)
    b = torch.load(resumed / 'checkpoint_last.pt', weights_only=False)
    assert a['epoch'] == b['epoch'] == 3
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert json.loads((resumed / 'resume.jsonl').read_text().splitlines()[0])['from_epoch'] == 1


def test_existing_run_is_refused_without_resume(tmp_path):
    cfg = write_tiny_dataset(tmp_path)
    store = GraphStore(cfg, root=tmp_path)
    run = tmp_path / 'run'
    args = ['--run-name', 'x', '--epochs', '1', '--batch-size', '4', '--micro-batch-size', '4']
    train(build_parser().parse_args(args), store, run, torch.device('cpu'), cfg)
    try:
        train(build_parser().parse_args(args), store, run, torch.device('cpu'), cfg)
    except FileExistsError:
        return
    raise AssertionError('second run into the same folder must be refused')
