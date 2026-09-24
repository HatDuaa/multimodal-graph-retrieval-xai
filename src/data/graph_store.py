"""Graph structures for train/val, precomputed once as integer rows into the frozen part table."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.graph.soft_match import PRONOUNS
from src.utils.config import REPO_ROOT, load_config


@dataclass
class PackedBatch:
    """The attributes of a PyG ``Batch`` that the reranker reads, built by index arithmetic."""
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    triple_f: torch.Tensor
    part_mask: torch.Tensor
    batch: torch.Tensor
    num_graphs: int


def _flat_positions(starts, counts):
    """Concatenated positions start_i, ..., start_i + count_i - 1 for every selected graph."""
    before = np.cumsum(counts) - counts
    return np.repeat(starts - before, counts) + np.arange(int(counts.sum()), dtype=np.int64)


class GraphIndex:
    """Many small graphs as flat int arrays: part-table rows per node / relation / triple and local edges."""

    NODE_FIELDS = ('node_rows', 'part_mask')
    EDGE_FIELDS = ('edge_src', 'edge_dst', 'relation_rows', 'triple_rows')

    def __init__(self, node_counts, edge_counts, **fields):
        self.node_counts = np.asarray(node_counts, dtype=np.int64)
        self.edge_counts = np.asarray(edge_counts, dtype=np.int64)
        self.fields = fields
        self.node_start = np.cumsum(self.node_counts) - self.node_counts
        self.edge_start = np.cumsum(self.edge_counts) - self.edge_counts

    def __len__(self):
        return len(self.node_counts)

    @classmethod
    def build(cls, parts_list, part_row):
        node_rows, part_mask, edge_src, edge_dst, relation_rows, triple_rows = [], [], [], [], [], []
        node_counts, edge_counts = [], []
        for parts in parts_list:
            node_rows.extend(part_row[f'node:{x}'] for x in parts['node_labels'])
            part_mask.extend(parts['part_mask'])
            edge_src.extend(e[0] for e in parts['edges'])
            edge_dst.extend(e[1] for e in parts['edges'])
            relation_rows.extend(part_row[f'rel:{x}'] for x in parts['relation_labels'])
            triple_rows.extend(part_row[f'triple:{x}'] for x in parts['triple_labels'])
            node_counts.append(len(parts['node_labels']))
            edge_counts.append(len(parts['edges']))
        as_int = lambda values: np.asarray(values, dtype=np.int64)  # noqa: E731
        return cls(node_counts, edge_counts, node_rows=as_int(node_rows), part_mask=np.asarray(part_mask, dtype=bool),
                   edge_src=as_int(edge_src), edge_dst=as_int(edge_dst), relation_rows=as_int(relation_rows),
                   triple_rows=as_int(triple_rows))

    def select(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        node_counts, edge_counts = self.node_counts[indices], self.edge_counts[indices]
        nodes = _flat_positions(self.node_start[indices], node_counts)
        edges = _flat_positions(self.edge_start[indices], edge_counts)
        fields = {k: self.fields[k][nodes] for k in self.NODE_FIELDS}
        fields.update({k: self.fields[k][edges] for k in self.EDGE_FIELDS})
        return GraphIndex(node_counts, edge_counts, **fields)

    @staticmethod
    def concat(items):
        return GraphIndex(np.concatenate([i.node_counts for i in items]), np.concatenate([i.edge_counts for i in items]),
                          **{k: np.concatenate([i.fields[k] for i in items])
                             for k in GraphIndex.NODE_FIELDS + GraphIndex.EDGE_FIELDS})

    def to_batch(self, table, device):
        """Same tensors as ``Batch.from_data_list`` over per-graph ``Data`` objects, gathered from ``table``."""
        offsets = np.repeat(self.node_start, self.edge_counts)
        edge_index = np.stack((self.fields['edge_src'] + offsets, self.fields['edge_dst'] + offsets))
        graph_of_node = np.repeat(np.arange(len(self), dtype=np.int64), self.node_counts)
        to = lambda array: torch.from_numpy(array).to(device, non_blocking=True)  # noqa: E731
        rows = lambda key: table[to(self.fields[key])]  # noqa: E731
        return PackedBatch(x=rows('node_rows'), edge_index=to(edge_index), edge_attr=rows('relation_rows'),
                           triple_f=rows('triple_rows'), part_mask=to(self.fields['part_mask']),
                           batch=to(graph_of_node), num_graphs=len(self))


class GraphStore:
    def __init__(self, cfg=None, root=None, splits=("train", "val"), allow_test=False):
        # Test is opened only by the one-time final evaluation (scripts/evaluate_test.py), never with train.
        allowed = {"test"} if allow_test else {"train", "val"}
        if not splits or not set(splits) <= allowed:
            raise ValueError("GraphStore permits train and val only; test needs allow_test=True and splits=('test',)")
        self.cfg = cfg or load_config()
        self.base = Path(root) if root else REPO_ROOT
        self.input_paths = []
        features = self.base / self.cfg['paths']['features']
        # Both tables are read fully into RAM (1.2 GB + 0.5 GB); no memory map on the training path.
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
        vectors = np.ascontiguousarray(np.load(stem.with_suffix('.npy')), dtype=np.float32)
        return vectors, {key: row for row, key in enumerate(meta['ids'])}

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

    def precompute(self):
        """Index every loaded image and query graph once; later batches only gather rows."""
        if getattr(self, 'image_index', None) is not None:
            return
        image_ids = sorted(self.graphs)
        self.image_position = {image_id: i for i, image_id in enumerate(image_ids)}
        self.image_index = GraphIndex.build((self._graph_parts(self.graphs[i]) for i in image_ids), self.part_row)
        caption_ids = list(self.queries)
        self.query_position = {cid: i for i, cid in enumerate(caption_ids)}
        self.query_index = GraphIndex.build((self._graph_parts(self.queries[c], query=True) for c in caption_ids),
                                            self.part_row)
        self._device_tables = {}

    def replace_image_graphs(self, graphs, extra_vectors=None, extra_keys=()):
        """Swap in other image graphs (corruption controls); extra part vectors are appended to the table."""
        if extra_vectors is not None and len(extra_keys):
            missing = [key for key in extra_keys if key not in self.part_row]
            keep = [row for row, key in enumerate(extra_keys) if key not in self.part_row]
            offset = len(self.part_vectors)
            self.part_vectors = np.concatenate([self.part_vectors, np.asarray(extra_vectors, dtype=np.float32)[keep]])
            self.part_row.update({key: offset + row for row, key in enumerate(missing)})
        self.graphs = dict(graphs)
        self.image_index = None
        self._device_tables = {}

    def use_rewired_graphs(self, seed, table_suffix=''):
        """Replace every loaded image graph by its degree-preserving rewiring for ``seed``.

        Rewired triple phrases that the main table lacks come from
        ``vg_coco_graph_parts_rewire_seed<seed>`` (``scripts/build_rewired_parts.py``).
        """
        from src.graph.corruption import rewire_graphs
        rewired = rewire_graphs(self.graphs, seed)
        stem = self.base / self.cfg['paths']['features'] / f'vg_coco_graph_parts_rewire_seed{seed}{table_suffix}'
        vectors, rows = self._table(stem)
        keys = sorted(rows, key=rows.get)
        self.replace_image_graphs(rewired, vectors, keys)
        self.precompute()  # fails loudly if a rewired phrase has no vector

    def part_table(self, device):
        """The part vector table as one tensor per device, gathered with torch indexing."""
        device = torch.device(device)
        tables = self.__dict__.setdefault('_device_tables', {})
        if device not in tables:
            tables[device] = torch.as_tensor(self.part_vectors, dtype=torch.float32).to(device)
        return tables[device]

    def candidates(self, split):
        if split not in self.arrays:
            raise ValueError('Only loaded train/val splits may be requested')
        return self.arrays[split]

    def batch(self, caption_ids, device='cpu', overrides=None):
        """Query and unique-image batches plus index-only candidate maps; overrides replace stored image graphs."""
        self.precompute()
        device = torch.device(device)
        table = self.part_table(device)
        caption_ids = [str(cid) for cid in caption_ids]
        locations = [self.caption_location[cid] for cid in caption_ids]
        candidates = np.stack([self.arrays[split]['candidate_ids'][row] for split, row in locations])
        if np.any(candidates < 0):
            raise ValueError('Model training requires full unpadded candidate pools')
        unique, inverse = np.unique(candidates, return_inverse=True)
        overrides = overrides or {}
        stored = [int(i) for i in unique if int(i) not in overrides]
        replaced = [int(i) for i in unique if int(i) in overrides]
        images = self.image_index.select([self.image_position[i] for i in stored])
        if replaced:
            images = GraphIndex.concat([images, GraphIndex.build([self._graph_parts(overrides[i]) for i in replaced],
                                                                  self.part_row)])
        order = {image_id: row for row, image_id in enumerate(stored + replaced)}
        candidate_index = np.array([order[int(i)] for i in unique], dtype=np.int64)[inverse.reshape(candidates.shape)]
        queries = self.query_index.select([self.query_position[cid] for cid in caption_ids])
        sentence_rows = [self.caption_row[cid] for cid in caption_ids]
        return {'queries': queries.to_batch(table, device),
                'images': images.to_batch(table, device),
                'candidate_index': torch.from_numpy(candidate_index).to(device),
                'candidate_ids': candidates, 'image_ids': np.array(stored + replaced, dtype=np.int64),
                'clip': torch.from_numpy(np.stack([self.arrays[s]['clip_scores'][r] for s, r in locations])).to(device),
                'sentence': torch.from_numpy(self.caption_vectors[sentence_rows]).to(device),
                'target': torch.tensor([int(self.arrays[s]['gold_rank'][r]) - 1 for s, r in locations], device=device),
                'gold': [int(self.arrays[s]['gold'][r]) for s, r in locations]}
