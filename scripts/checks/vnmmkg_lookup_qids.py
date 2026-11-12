"""Print labels of the QIDs configured in the vnmmkg block, or search QIDs by Vietnamese label.

Used at gate G1 to check landmark/province classes and to find the period QIDs.
Usage:  python scripts/checks/vnmmkg_lookup_qids.py --configured
        python scripts/checks/vnmmkg_lookup_qids.py --search "Nhà Lý" "thành phố trực thuộc trung ương"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.wikimedia_client import WIKIDATA_API, client_from_config  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configured", action="store_true")
    ap.add_argument("--search", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config()
    v = cfg["vnmmkg"]
    client = client_from_config(cfg)
    if args.configured:
        qids = [v["country"], v["human_class"], *v["landmark_classes"], *v["province_classes"]]
        qids += [p["qid"] for p in v["periods"] if p["qid"]]
        for start in range(0, len(qids), 50):
            data = client.get_json(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(qids[start:start + 50]),
                                                  "props": "labels", "languages": "vi|en"})
            for qid, ent in data["entities"].items():
                labels = ent.get("labels", {})
                print(qid, "|", labels.get("vi", {}).get("value"), "|", labels.get("en", {}).get("value"))
    for text in args.search:
        data = client.get_json(WIKIDATA_API, {"action": "wbsearchentities", "search": text, "language": "vi",
                                              "uselang": "vi", "limit": 5, "type": "item"})
        print(f"== {text}")
        for hit in data.get("search", []):
            print("  ", hit["id"], "|", hit.get("label"), "|", hit.get("description"))


if __name__ == "__main__":
    main()
