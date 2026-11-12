"""List single and paired (relation, value) constraints over the built triples with 1..max golds.

Helps write indirect hand queries (spec 4.5): every pair printed contains at least one
non-visual relation and matches between 1 and max_pattern_golds landmarks.
Usage:  python scripts/checks/vnmmkg_pattern_counts.py > local-docs/vnmmkg_patterns.txt
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import REPO_ROOT, load_config  # noqa: E402


def main() -> None:
    v = load_config()["vnmmkg"]
    out_dir = REPO_ROOT / v["out_dir"]
    entities = {json.loads(line)["qid"]: json.loads(line)
                for line in (out_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()}
    landmarks = {q for q, e in entities.items() if e["kind"] == "landmark"}
    holders = defaultdict(set)
    for line in (out_dir / "triples.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t["h"] in landmarks:
            holders[(t["r"], t["t"])].add(t["h"])
    label = lambda q: entities.get(q, {}).get("label_vi") or q  # noqa: E731
    non_visual, cap = set(v["non_visual_pids"]), v["max_pattern_golds"]
    print("# single constraints: r, t, label, #landmarks")
    for (r, t), hs in sorted(holders.items(), key=lambda kv: -len(kv[1])):
        print(r, t, label(t), len(hs))
    print(f"\n# pairs with 1..{cap} landmarks and a non-visual relation")
    for (c1, h1), (c2, h2) in itertools.combinations(sorted(holders.items()), 2):
        both = h1 & h2
        if c1[0] != c2[0] and 1 <= len(both) <= cap and {c1[0], c2[0]} & non_visual:
            print(json.dumps([list(c1), list(c2)]), label(c1[1]), "+", label(c2[1]), "->",
                  ", ".join(label(q) for q in sorted(both)))


if __name__ == "__main__":
    main()
