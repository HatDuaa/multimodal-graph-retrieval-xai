"""Alternative caption parsers, all returning the query graph schema of src/graph/parse_query.py.

Schema: {"objects": [{"id", "name", "attributes", "count"}], "relations": [{"subject", "predicate",
"predicate_lemma", "object"}]}; "count" is an int only when the parser states a quantity, else None.
- sng: the level-1 parser (sng_parser via parse_query.parse_caption), unchanged.
- factual: FACTUAL-T5 (Li et al., ACL Findings 2023) through the FactualSceneGraph package.
- llm: an instruction-tuned LLM prompted few-shot for JSON, asked to replace pronouns by their referents.
Object names of factual and llm are lower-cased and their last word lemmatised (spaCy), like sng's lemma_head.
"""
from __future__ import annotations

import json
import re

PROMPT = """You turn an image caption into a scene graph.
Return only JSON: {"entities": [{"name": ..., "attributes": [...], "count": ...}], "relations": [{"subject": i, "predicate": ..., "object": j}]}.
- An entity is a physical thing or person in the image. "name" is its noun in singular form, lower case, without
  articles, numbers or adjectives (keep compound nouns such as "tennis racket"). List one entity per distinct thing.
- "attributes" are the adjectives describing it (colour, size, material, state); [] if none.
- "count" is the number stated in the caption ("two dogs" -> 2, "a pair of shoes" -> 2); null when no number is stated.
- Never output pronouns (it, they, he, she, that, who, ...) as entities: replace each pronoun by the entity it refers to.
- A relation links two entities by their 0-based index; "predicate" is the verb or preposition from the caption,
  lower case, e.g. "riding", "on", "next to", "on top of", "holding". Add a relation only if the caption states it."""

EXAMPLES = [
    ("Two brown dogs are chasing a red frisbee on the grass.",
     {"entities": [{"name": "dog", "attributes": ["brown"], "count": 2},
                   {"name": "frisbee", "attributes": ["red"], "count": None},
                   {"name": "grass", "attributes": [], "count": None}],
      "relations": [{"subject": 0, "predicate": "chasing", "object": 1},
                    {"subject": 0, "predicate": "on", "object": 2}]}),
    ("A man holding an umbrella while he walks his dog next to a building.",
     {"entities": [{"name": "man", "attributes": [], "count": None},
                   {"name": "umbrella", "attributes": [], "count": None},
                   {"name": "dog", "attributes": [], "count": None},
                   {"name": "building", "attributes": [], "count": None}],
      "relations": [{"subject": 0, "predicate": "holding", "object": 1},
                    {"subject": 0, "predicate": "walks", "object": 2},
                    {"subject": 0, "predicate": "next to", "object": 3}]}),
    ("a pair of skis leaning against a wooden fence that is covered in snow",
     {"entities": [{"name": "ski", "attributes": [], "count": 2},
                   {"name": "fence", "attributes": ["wooden"], "count": None},
                   {"name": "snow", "attributes": [], "count": None}],
      "relations": [{"subject": 0, "predicate": "leaning against", "object": 1},
                    {"subject": 1, "predicate": "covered in", "object": 2}]}),
]


def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def normalize_names(names: list[str], nlp) -> dict[str, str]:
    """Lower-case, collapse spaces, and lemmatise the last word of each distinct name."""
    out = {}
    clean = {n: re.sub(r"\s+", " ", str(n).lower().strip(" .,;:!?\"'")) for n in set(names)}
    for raw, doc in zip(clean, nlp.pipe(clean.values())):
        tokens = [t.text for t in doc]
        if tokens:
            tokens[-1] = doc[-1].lemma_.lower() if doc[-1].lemma_ else tokens[-1]
        out[raw] = " ".join(tokens)
    return out


def predicate_lemma(predicates: list[str], nlp) -> dict[str, str]:
    clean = {p: str(p).lower().strip() for p in set(predicates)}
    return {raw: " ".join(t.lemma_.lower() for t in doc) for raw, doc in zip(clean, nlp.pipe(clean.values()))}


def to_query_graph(entities: list[dict], relations: list[dict], names: dict, lemmas: dict) -> dict:
    objects = [{"id": i, "name": names[e["name"]], "attributes": sorted(e.get("attributes") or []),
                "count": e.get("count")} for i, e in enumerate(entities)]
    rels = [{"subject": r["subject"], "predicate": str(r["predicate"]).lower().strip(),
             "predicate_lemma": lemmas[r["predicate"]], "object": r["object"]} for r in relations]
    return {"objects": objects, "relations": rels}


def _count(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value) or None
    return None


def convert(raw_graphs: list[dict]) -> list[dict]:
    """Normalise names and predicates of parser-neutral graphs {"entities": [...], "relations": [...]}."""
    nlp = _nlp()
    names = normalize_names([e["name"] for g in raw_graphs for e in g["entities"]], nlp)
    lemmas = predicate_lemma([r["predicate"] for g in raw_graphs for r in g["relations"]], nlp)
    return [to_query_graph(g["entities"], g["relations"], names, lemmas) for g in raw_graphs]


class FactualParser:
    def __init__(self, checkpoint: str, device: str = "cuda", batch_size: int = 64, beam_size: int = 5):
        import contextlib

        from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
        self.parser = SceneGraphParser(checkpoint, device=device)
        # transformers 5 dropped as_target_tokenizer; FactualSceneGraph 0.7.3 only uses it to count prompt tokens,
        # which a T5 tokenizer encodes the same way on both sides.
        self.parser.tokenizer.as_target_tokenizer = contextlib.nullcontext
        self.batch_size, self.beam_size = batch_size, beam_size

    def parse(self, texts: list[str]) -> tuple[list[dict], list[str]]:
        outputs = self.parser.parse(texts, batch_size=self.batch_size, beam_size=self.beam_size, return_text=True,
                                    filter_factual_chars=True)
        raw = []
        for text in outputs:
            graph = self.parser.graph_string_to_object(text)
            entities = [{"name": e["head"], "attributes": sorted(e["attributes"]), "count": _count(e["quantity"])}
                        for e in graph["entities"]]
            relations = [{"subject": r["subject"], "predicate": r["relation"], "object": r["object"]}
                         for r in graph["relations"] if r["subject"] != r["object"]]
            raw.append({"entities": entities, "relations": relations})
        return raw, outputs


def parse_llm_json(text: str) -> dict | None:
    """The first JSON object in an LLM answer, validated and cleaned; None when unusable."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        return None
    entities = []
    for e in data["entities"]:
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].strip():
            attributes = [a for a in e.get("attributes") or [] if isinstance(a, str) and a.strip()]
            entities.append({"name": e["name"], "attributes": attributes, "count": _count(e.get("count"))})
        else:
            entities.append(None)
    keep = {i: len([x for x in entities[:i] if x is not None]) for i, x in enumerate(entities) if x is not None}
    relations = []
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        s, o, p = r.get("subject"), r.get("object"), r.get("predicate")
        if isinstance(s, int) and isinstance(o, int) and s in keep and o in keep and s != o \
                and isinstance(p, str) and p.strip():
            relations.append({"subject": keep[s], "predicate": p, "object": keep[o]})
    return {"entities": [e for e in entities if e is not None], "relations": relations}


class LLMParser:
    def __init__(self, model_id: str, device: str = "cuda", batch_size: int = 64, max_new_tokens: int = 320):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.device, self.batch_size, self.max_new_tokens = device, batch_size, max_new_tokens

    def messages(self, caption: str) -> list[dict]:
        out = [{"role": "system", "content": PROMPT}]
        for example, answer in EXAMPLES:
            out += [{"role": "user", "content": example}, {"role": "assistant", "content": json.dumps(answer)}]
        return out + [{"role": "user", "content": caption.strip()}]

    def parse(self, texts: list[str]) -> tuple[list[dict], list[str]]:
        raw, answers = [], []
        prompts = [self.tokenizer.apply_chat_template(self.messages(t), tokenize=False, add_generation_prompt=True)
                   for t in texts]
        # Similar lengths in a batch keep left padding short; results are put back in input order.
        order = sorted(range(len(texts)), key=lambda i: len(prompts[i]))
        decoded = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.pad_token_id)
            for i, row in zip(idx, out[:, batch["input_ids"].shape[1]:]):
                decoded[i] = self.tokenizer.decode(row, skip_special_tokens=True)
        for answer in decoded:
            graph = parse_llm_json(answer)
            answers.append(answer)
            raw.append(graph if graph is not None else {"entities": [], "relations": [], "failed": True})
        return raw, answers
