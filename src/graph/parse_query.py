"""Caption scene graph parsing with an optional lazy spaCy dependency."""
from multiprocessing import Pool


def parse_caption(text: str) -> dict:
    from sng_parser import parse
    parsed = parse(text)
    objects = [{"id": i, "name": e.get("lemma_head", e.get("head", "")).lower()} for i, e in enumerate(parsed.get("entities", []))]
    relations = [{"subject": r["subject"], "predicate": r.get("relation", ""),
                  "predicate_lemma": r.get("lemma_relation", r.get("relation", "")), "object": r["object"]}
                 for r in parsed.get("relations", [])]
    return {"objects": objects, "relations": relations}


def parse_many(texts: list[str], workers: int) -> list[dict]:
    if workers <= 1:
        return [parse_caption(text) for text in texts]
    with Pool(workers) as pool:
        return pool.map(parse_caption, texts, chunksize=max(1, len(texts) // (workers * 4)))
