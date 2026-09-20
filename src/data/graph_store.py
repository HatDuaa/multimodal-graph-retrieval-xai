"""Indexed graph structures and memory-mapped frozen vectors for train/val only."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.graph.soft_match import PRONOUNS
from src.utils.config import REPO_ROOT, load_config


class GraphStore:
    def __init__(self, cfg=None, root=None, splits=("train", "val")):
        if not set(splits) <= {"train", "val"}:
            raise ValueError("GraphStore permits train and val only")
        self.cfg = cfg or load_config()
        self.base = Path(root) if root else REPO_ROOT
        self.input_paths = []
        features = self.base / self.cfg['paths']['features']
        self.part_vectors, self.part_row = self._table(features / 'vg_coco_graph_parts')
        self.part_dim = self.part_vectors.shape[1]
        self.caption_vectors, self.caption_row = self._table(features / 'vg_coco_caption')
        self.graphs, self.queries, self.arrays = {}, {}, {}
        graph_path = self.base / self.cfg['paths']['graphs'] / 'vg_coco_scene_graphs.jsonl'
        self.input_paths.append(graph_path)
        with graph_path.open(encoding='utf-8') as handle:
            for line in handle:
                graph = json.loads(line)
                if graph['split'] in splits:
                    self.graphs[int(graph['image_id'])] = graph
        processed = self.base / self.cfg['paths']['processed']
        for split in splits:
            query_path = processed / f'query_graphs_{split}.json'
            self.input_paths.append(query_path)
            self.queries.update(json.loads(query_path.read_text(encoding='utf-8')))
            path = processed / f'candidates_{split}.npz'
            self.input_paths.append(path)
            with np.load(path, allow_pickle=False) as data:
                self.arrays[split] = {key: data[key] for key in ('caption_ids', 'gold', 'candidate_ids',
                                                               'clip_scores', 'gold_rank', 'group_index')}
        self.caption_location = {str(cid): (split, row) for split, arrays in self.arrays.items()
                                 for row, cid in enumerate(arrays['caption_ids'])}
        for split in splits:
            self.input_paths.append(self.base / self.cfg['paths']['splits'] / f'vg_coco_{split}.json')
        if 'train' in splits:
            self.input_paths.append(self.base / self.cfg['paths']['splits'] / 'vg_coco_train_groups.json')

    def _table(self, stem):
        self.input_paths.extend([stem.with_suffix('.npy'), stem.with_suffix('.ids.json')])
        meta = json.loads(stem.with_suffix('.ids.json').read_text(encoding='utf-8'))
        return np.load(stem.with_suffix('.npy'), mmap_mode='r'), {key: row for row, key in enumerate(meta['ids'])}

    def _vectors(self, keys):
        return torch.from_numpy(np.array(self.part_vectors[[self.part_row[key] for key in keys]], dtype=np.float32))

    @staticmethod
    def _graph_parts(graph, query=False):
        # Keep empty image labels: instance_parts in the reference includes them.
        objects = graph['objects']
        names = {obj['id']: obj['name'] for obj in objects}
        kept = [obj for obj in objects if not query or obj['name'] not in PRONOUNS]
        remap = {obj['id']: row for row, obj in enumerate(kept)}
        edges, relations, triples, relation_indices = [], [], [], []
        for index, rel in enumerate(graph['relations']):
            s, o = rel['subject'], rel['object']
            if s not in remap or o not in remap:
                continue
            predicate = rel['predicate'].lower() if query else rel['predicate']
            edges.append((remap[s], remap[o], predicate))
            relations.append(predicate)
            triples.append(f'{names[s]} {predicate} {names[o]}')
            relation_indices.append(index)
        return dict(node_labels=[obj['name'] for obj in kept], relation_labels=relations, triple_labels=triples,
                    edges=edges, raw=graph, node_indices=[i for i, obj in enumerate(objects) if obj['id'] in remap],
                    relation_indices=relation_indices,
                    part_mask=[bool(obj['name']) if query else True for obj in kept])

    def tensor_graph(self, raw, query=False):
        parts = self._graph_parts(raw, query)
        parts['node_f'] = self._vectors([f'node:{x}' for x in parts['node_labels']])
        parts['relation_f'] = self._vectors([f'rel:{x}' for x in parts['relation_labels']])
        parts['triple_f'] = self._vectors([f'triple:{x}' for x in parts['triple_labels']])
        return parts

    def image(self, image_id):
        return self.tensor_graph(self.graphs[int(image_id)])

    def query(self, caption_id):
        return self.tensor_graph(self.queries[str(caption_id)], query=True)

    def sentence(self, caption_id):
        return torch.from_numpy(np.array(self.caption_vectors[self.caption_row[str(caption_id)]], dtype=np.float32))

    def candidates(self, split):
        if split not in self.arrays:
            raise ValueError('Only loaded train/val splits may be requested')
        return self.arrays[split]

    @staticmethod
    def batch_graphs(graphs, device=None):
        rows = []
        for graph in graphs:
            edges = torch.tensor([[e[0] for e in graph['edges']], [e[1] for e in graph['edges']]], dtype=torch.long)
            rows.append(Data(x=graph['node_f'], edge_index=edges.reshape(2, -1), edge_attr=graph['relation_f'],
                             triple_f=graph['triple_f'],
                             part_mask=torch.tensor(graph.get('part_mask', [True] * len(graph['node_f'])), dtype=torch.bool)))
        return Batch.from_data_list(rows).to(device)

    def batch(self, caption_ids, device='cpu', overrides=None):
        """Return PyG batches for unique images and all queries, plus index-only candidate maps."""
        locations = [self.caption_location[str(cid)] for cid in caption_ids]
        candidates = np.stack([self.arrays[split]['candidate_ids'][row] for split, row in locations])
        if np.any(candidates < 0):
            raise ValueError('Model training requires full unpadded candidate pools')
        unique, inverse = np.unique(candidates, return_inverse=True)
        image_graphs = [self.tensor_graph(overrides[int(i)]) if overrides and int(i) in overrides
                        else self.image(int(i)) for i in unique]
        return {'queries': self.batch_graphs([self.query(cid) for cid in caption_ids], device),
                'images': self.batch_graphs(image_graphs, device),
                'candidate_index': torch.from_numpy(inverse.reshape(candidates.shape)).to(device),
                'candidate_ids': candidates, 'image_ids': unique,
                'clip': torch.tensor(np.stack([self.arrays[s]['clip_scores'][r] for s, r in locations]), device=device),
                'sentence': torch.stack([self.sentence(cid) for cid in caption_ids]).to(device),
                'target': torch.tensor([int(self.arrays[s]['gold_rank'][r]) - 1 for s, r in locations], device=device),
                'gold': [int(self.arrays[s]['gold'][r]) for s, r in locations]}
