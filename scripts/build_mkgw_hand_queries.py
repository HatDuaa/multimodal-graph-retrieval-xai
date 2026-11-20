"""Validate the hand-written MKG-W evaluation queries and compute the gold of indirect ones.

Row 24 of the task sheet (level 2: >= 20 evaluation queries with a direct and an
indirect group). Same rules as the Vietnamese MMKG set (src/data/build_vnmmkg.py,
resolve_hand_queries):

  direct    describes what the answer entity is / looks like, without naming it or
            relying on a relation to another named entity; exactly one gold entity.
            Checked: no entity linked in the text (EntityLinker) has an edge to the gold.
  indirect  names another entity and needs a KG relation to reach the answer
            ("a commune whose region is X"); gold is computed from `pattern`, a list
            of [pid, qid] pairs: the answer must have an edge answer --pid--> qid for
            every pair. 1..MAX_GOLD answers.

Pool = images of the retrieval TEST split (8 920 entities -> 892 test). Gold is computed
on the full MKG-W triple set (all three KGC splits): it is the ground truth, not a model
input. The graph channel at inference still sees only KGC-train edges, so an indirect
query whose edge lives only in KGC valid/test is a fair, harder case.

The set is written once and locked: queries are never edited after the first scored run
(--allow-update only to fix a validation error before that run).

Usage:  python scripts/build_mkgw_hand_queries.py [--allow-update]
File:   data/queries/mkgw_queries_hand.json (edited by hand, then rewritten with gold)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402

MAX_GOLD = 5
HAND = REPO_ROOT / "data" / "queries" / "mkgw_queries_hand.json"


def names_in(text: str, names: list[str]) -> list[str]:
    return [n for n in names if n and len(n) >= 3 and re.search(rf"\b{re.escape(n)}\b", text, re.IGNORECASE)]


def gold_from_pattern(pattern: list[list[str]], out_edges: dict[str, set], pool: set[str]) -> list[str]:
    return sorted(q for q in pool if all((pid, obj) in out_edges.get(q, ()) for pid, obj in pattern))


def resolve_queries(queries: list[dict], out_edges: dict, pool: set, names_of: dict, allow_update: bool,
                    link=lambda text: {}):
    problems, out, seen = [], [], set()
    for q in queries:
        qid, group, pattern = q["query_id"], q["group"], [list(p) for p in q.get("pattern") or []]
        if qid in seen:
            problems.append(f"{qid}: duplicate query_id")
        seen.add(qid)
        gold = q.get("gold_qids")
        if group == "direct":
            if pattern:
                problems.append(f"{qid}: a direct query has no pattern")
            if not gold or len(gold) != 1 or gold[0] not in pool:
                problems.append(f"{qid}: a direct query needs exactly one test-pool entity")
            else:
                neighbours = {o for _, o in out_edges.get(gold[0], ())} | {
                    h for h, es in out_edges.items() if any(o == gold[0] for _, o in es)}
                shortcut = sorted(set(link(q["text"])) & neighbours)
                if shortcut:
                    problems.append(f"{qid}: direct query links KG neighbours of its answer {shortcut}")
        elif group == "indirect":
            computed = gold_from_pattern(pattern, out_edges, pool) if pattern else []
            if not computed:
                problems.append(f"{qid}: pattern has no gold in the test pool")
            elif len(computed) > MAX_GOLD:
                problems.append(f"{qid}: {len(computed)} golds; narrow the pattern")
            if gold is not None and gold != computed and not allow_update:
                problems.append(f"{qid}: gold changed {gold} -> {computed} (locked set)")
            gold = computed
        else:
            problems.append(f"{qid}: group must be direct or indirect")
        for g in gold or []:
            hit = names_in(q["text"], names_of.get(g, []))
            if hit:
                problems.append(f"{qid}: text names its own answer {g} ({hit})")
        out.append({"query_id": qid, "group": group, "text": q["text"], "pattern": pattern,
                    "gold_qids": gold, "note": q.get("note", "")})
    return out, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-update", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    raw = REPO_ROOT / "data" / "raw" / "mkgw"
    cache = [json.loads(line) for line in (raw / "wikidata_entities.jsonl").read_text(encoding="utf-8").splitlines()]
    id2qid = {r["id"]: r["qid"] for r in cache}
    names_of = {r["qid"]: [r.get("label_en")] + list(r.get("aliases_en") or []) for r in cache}
    rels = json.loads((raw / "wikidata_relations.json").read_text(encoding="utf-8"))
    id2pid = {v["id"]: pid for pid, v in rels.items()}
    out_edges = defaultdict(set)
    for line in (REPO_ROOT / "data" / "processed" / "mkgw_triples.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        out_edges[id2qid[t["h"]]].add((id2pid[t["r"]], id2qid[t["t"]]))
    test = json.loads((resolve(cfg, "splits") / "mkgw_test.json").read_text(encoding="utf-8"))["images"]
    pool = {im["qid"] for im in test}

    from src.retrieval.mkgw_graph_score import EntityLinker
    linker = EntityLinker.from_cache(raw / "wikidata_entities.jsonl", lang=cfg["mkgw"]["query_lang"])
    queries, problems = resolve_queries(json.loads(HAND.read_text(encoding="utf-8")), out_edges, pool,
                                        names_of, args.allow_update, link=linker.link)
    if problems:
        sys.exit("hand-written queries:\n" + "\n".join(problems))
    HAND.write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = defaultdict(int)
    for q in queries:
        if q["group"] == "indirect":
            sizes[len(q["gold_qids"])] += 1
    print(json.dumps({"direct": sum(q["group"] == "direct" for q in queries),
                      "indirect": sum(q["group"] == "indirect" for q in queries),
                      "indirect_gold_sizes": dict(sorted(sizes.items()))}))


if __name__ == "__main__":
    main()
