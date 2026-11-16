"""Gate G2 (spec section 7): is there enough triple material before spending time on image review?

Proposes triples for the top-N landmark candidates (N = all valid candidates, 90, 80) with
the current config, applies the entity cap, and writes the counts to
experiments/checks/vnmmkg_triple_budget_<tag>.json. Run once with --tag before, and again
with --tag after if PIDs are added, so the data card can report both.

Usage:  python scripts/checks/vnmmkg_triple_budget.py --tag before
"""
import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.vnmmkg_cache import infobox_rows, landmark_candidates, load_link_qids, load_viwiki  # noqa: E402
from src.data.vnmmkg_triples import cap_entities, propose_triples  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config()
    v = cfg["vnmmkg"]
    raw_dir = REPO_ROOT / v["raw_dir"]
    ranked, entities = landmark_candidates(raw_dir, {**v, "max_landmark_candidates": 10**6})
    rows, links = infobox_rows(load_viwiki(raw_dir), v), load_link_qids(raw_dir)
    report = {"date": datetime.date.today().isoformat(), "landmark_pids": v["landmark_pids"],
              "max_entities": v["max_entities"], "valid_candidates": len(ranked), "by_n": {}}
    for n in sorted({len(ranked), 90, 80}, reverse=True):
        landmarks = ranked[:n]
        triples, counts = propose_triples(landmarks, entities, rows, links, v)
        context, kept, n_ent, n_tri = cap_entities(landmarks, triples, v["max_entities"])
        report["by_n"][str(n)] = {
            "proposed": len(triples), "after_cap": len(kept), "context_kept": len(context),
            "dropped_entities": n_ent, "dropped_triples": n_tri,
            "by_source": dict(sorted(Counter(t["source"] for t in kept).items())),
            "by_relation": dict(sorted(Counter(t["r"] for t in kept).items())),
            "counters": counts}
        print(f"N={n}: proposed {len(triples)}, after cap {len(kept)} "
              f"(context {len(context)}), by source {report['by_n'][str(n)]['by_source']}")
    out = resolve(cfg, "experiments") / "checks" / f"vnmmkg_triple_budget_{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("->", out)


if __name__ == "__main__":
    main()
