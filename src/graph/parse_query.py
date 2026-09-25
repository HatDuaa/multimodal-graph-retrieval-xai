"""Caption scene graph parsing with an optional lazy spaCy dependency."""
from multiprocessing import Pool


def parse_caption(text: str, with_attributes: bool = False) -> dict:
    from sng_parser import parse
    parsed = parse(text)
    objects = []
    for i, entity in enumerate(parsed.get("entities", [])):
        obj = {"id": i, "name": entity.get("lemma_head", entity.get("head", "")).lower()}
        if with_attributes:
            modifiers = entity.get("modifiers", [])
            attrs = [m.get("span", "") for m in modifiers
                     if m.get("dep") == "amod"]
            obj["attributes"] = [str(a).lower().strip() for a in attrs if str(a).strip()]
        objects.append(obj)
    relations = [{"subject": r["subject"], "predicate": r.get("relation", ""),
                  "predicate_lemma": r.get("lemma_relation", r.get("relation", "")), "object": r["object"]}
                 for r in parsed.get("relations", [])]
    return {"objects": objects, "relations": relations}


def parse_many(texts: list[str], workers: int, with_attributes: bool = False) -> list[dict]:
    if workers <= 1:
        return [parse_caption(text, with_attributes=with_attributes) for text in texts]
    with Pool(workers) as pool:
        if with_attributes:
            return pool.starmap(parse_caption, [(text, True) for text in texts], chunksize=max(1, len(texts) // (workers * 4)))
        return pool.map(parse_caption, texts, chunksize=max(1, len(texts) // (workers * 4)))
