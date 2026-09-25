"""Pre-training check for level-2 step b: does the query graph channel cover enough?

For every train+val query (test is never read) this measures:
  - how many queries link at least one KG entity by label/alias (EntityLinker),
  - how many golds have a direct KGC-train edge to a linked entity
    (the explainable triple channel).

No training, no CLIP; entity embeddings are only loaded to prove the pipeline
runs end to end. Deterministic, so no seed. Result: experiments/checks/mkgw_link_coverage.json.

Usage:  python scripts/checks/mkgw_link_coverage.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.extract_clip import load_features  # noqa: E402
from src.retrieval.mkgw_graph_score import EntityLinker, MkgwGraphScorer, load_train_triples  # noqa: E402
from src.utils.config import REPO_ROOT, load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    raw = REPO_ROOT / "data" / "raw" / "mkgw"
    cache = [json.loads(l) for l in (raw / "wikidata_entities.jsonl").read_text(encoding="utf-8").splitlines()]
    id2qid = {r["id"]: r["qid"] for r in cache}
    rels = json.loads((raw / "wikidata_relations.json").read_text(encoding="utf-8"))
    id2pid = {v["id"]: pid for pid, v in rels.items()}

    linker = EntityLinker.from_cache(raw / "wikidata_entities.jsonl", lang=cfg["mkgw"]["query_lang"])
    feats, qids, encoder = load_features(REPO_ROOT / "data" / "features" / "mkgw_native_ent")
    triples = load_train_triples(REPO_ROOT / "data" / "processed" / "mkgw_triples.jsonl", id2qid, id2pid)
    scorer = MkgwGraphScorer(feats, {q: i for i, q in enumerate(qids)}, triples,
                             {pid: v["label"] or pid for pid, v in rels.items()})

    result = {"encoder": encoder, "n_train_triples": len(triples), "splits": {}}
    for split in ("train", "val"):        # test is never read here
        images = json.loads((REPO_ROOT / "data" / "splits" / f"mkgw_{split}.json").read_text(encoding="utf-8"))["images"]
        n_linked = n_edge = n_links = 0
        for im in images:
            linked = linker.link(im["captions"][0]["text"], exclude={im["qid"]})
            if linked:
                n_linked += 1
                n_links += len(linked)
                if scorer.score(linked, im["qid"])["triple"] > 0:
                    n_edge += 1
        result["splits"][split] = {
            "queries": len(images),
            "linked_ge_1": n_linked,
            "linked_ge_1_pct": round(100 * n_linked / len(images), 2),
            "avg_links_per_linked_query": round(n_links / max(n_linked, 1), 2),
            "gold_direct_train_edge": n_edge,
            "gold_direct_train_edge_pct": round(100 * n_edge / len(images), 2),
        }

    out = REPO_ROOT / "experiments" / "checks" / "mkgw_link_coverage.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
