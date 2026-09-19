"""Per-image Visual Genome scene graphs in the docs/interfaces.md section 3 shape.

Labels are normalised with the alias files and, optionally, restricted to VG150 (150 object classes and
50 predicates of Xu et al. 2017, the usual scene graph benchmark vocabulary). A graph only ever holds the
annotations of its own image, so nothing is aggregated across images or splits.
"""
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from src.utils.config import resolve


def normalize_label(text: str) -> str:
    text = str(text).lower().strip(" 	
.,;:!?()[]{}\"'")
    return re.sub(r"\s+", " ", text)


def load_alias(path: str | Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        names = [normalize_label(x) for x in line.split(",") if normalize_label(x)]
        if names:
            target = aliases.get(names[0], names[0])
            for name in names:
                aliases[name] = target
    return aliases


class Vocab:
    def __init__(self, objects: set[str], predicates: set[str], object_alias: dict[str, str], predicate_alias: dict[str, str]):
        self.objects, self.predicates = objects, predicates
        self.object_alias, self.predicate_alias = object_alias, predicate_alias

    @classmethod
    def from_config(cls, cfg: dict) -> "Vocab":
        folder = resolve(cfg, "raw") / "vg150"
        if not (folder / "object_list.txt").exists():
            raise SystemExit(f"missing {folder}; run: python scripts/download_data.py")
        objects = {normalize_label(x) for x in (folder / "object_list.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
        predicates = {normalize_label(x) for x in (folder / "predicate_list.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
        return cls(objects, predicates, load_alias(folder / "object_alias.txt"), load_alias(folder / "predicate_alias.txt"))

    def canon_object(self, name: str) -> str:
        value = normalize_label(name)
        return self.object_alias.get(value, value)

    def canon_predicate(self, name: str) -> str:
        value = normalize_label(name)
        return self.predicate_alias.get(value, value)

    def has_object(self, name: str) -> bool:
        return self.canon_object(name) in self.objects

    def has_predicate(self, name: str) -> bool:
        return self.canon_predicate(name) in self.predicates


def _name(item: dict) -> str:
    names = item.get("names")
    return str(item.get("name", names[0] if names else ""))


def build_scene_graph(objects_entry: dict, relationships_entry: dict, vocab: Vocab, vg150_only: bool = True) -> dict:
    objects: list[dict] = []
    by_id: dict[int, int] = {}
    for obj in objects_entry.get("objects", []):
        name = vocab.canon_object(_name(obj))
        if vg150_only and name not in vocab.objects:
            continue
        by_id[obj["object_id"]] = len(objects)
        objects.append({"id": len(objects), "name": name})
    relations = []
    seen = set()
    for rel in relationships_entry.get("relationships", []):
        pred = vocab.canon_predicate(rel.get("predicate", ""))
        if vg150_only and pred not in vocab.predicates:
            continue
        endpoints = []
        for endpoint in (rel.get("subject", {}), rel.get("object", {})):
            oid = endpoint.get("object_id")
            if oid not in by_id:
                name = vocab.canon_object(_name(endpoint))
                if vg150_only and name not in vocab.objects:
                    endpoints.append(None)
                    continue
                by_id[oid] = len(objects)
                objects.append({"id": len(objects), "name": name})
            endpoints.append(by_id[oid])
        if None in endpoints:
            continue
        triple = (endpoints[0], pred, endpoints[1])
        if triple not in seen:
            seen.add(triple)
            relations.append({"subject": triple[0], "predicate": triple[1], "object": triple[2]})
    return {"image_id": objects_entry.get("image_id", relationships_entry.get("image_id")), "objects": objects, "relations": relations}


def label_triples(graph: dict) -> set[tuple[str, str, str]]:
    labels = {o["id"]: o["name"] for o in graph["objects"]}
    return {(labels[r["subject"]], r["predicate"], labels[r["object"]]) for r in graph["relations"]}


def label_pairs(graph: dict) -> set[tuple[str, str]]:
    labels = {o["id"]: o["name"] for o in graph["objects"]}
    return {(labels[r["subject"]], labels[r["object"]]) for r in graph["relations"]}


def load_raw(cfg: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    """Read objects.json and relationships.json once (about 1 GB of JSON); both are keyed by Visual Genome image_id."""
    folder = resolve(cfg, "raw") / "visual_genome"
    out = []
    for name in ("objects.json", "relationships.json"):
        if not (folder / name).exists():
            raise SystemExit(f"missing {folder / name}; run: python scripts/download_data.py")
        out.append({x["image_id"]: x for x in json.loads((folder / name).read_text(encoding="utf-8"))})
    return out[0], out[1]


def iter_scene_graphs(raw: tuple[dict[int, dict], dict[int, dict]], vocab: Vocab, image_ids: Iterable[int],
                      vg150_only: bool) -> Iterator[dict]:
    objects, relationships = raw
    for image_id in image_ids:
        empty = {"image_id": image_id}
        yield build_scene_graph(objects.get(image_id, empty), relationships.get(image_id, empty), vocab, vg150_only)
