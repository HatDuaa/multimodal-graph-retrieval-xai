"""Pure triple logic for the Vietnamese MMKG (spec 4.2): no network, no file access.

Three sources, one triple per (head, relation, tail):
- wikidata: truthy statements of the configured PIDs, landmark -> context entity, plus
  one-hop statements between context entities (configured PIDs + P131);
- infobox: viwiki infobox fields mapped to PIDs; only wikilinked values count;
- derived: two fixed rules, province (walk truthy P131 up to a province-level entity)
  and period (year from P571 or the infobox `founded` field -> configured period table).
When a triple comes from several sources the most direct one is kept
(wikidata > infobox > derived). infobox and derived triples go through human review.
"""
import re
import unicodedata
from collections import Counter

import mwparserfromhell

from src.data.vnmmkg_wikidata import inception_year, qid_key, truthy

SOURCE_ORDER = {"wikidata": 0, "infobox": 1, "derived": 2}
NAMESPACES = ("Tập tin:", "File:", "Hình:", "Image:", "Thể loại:", "Category:", "Wikipedia:", "Bản mẫu:")
APPROXIMATE = re.compile(r"thế\s*k[ỷỉ]|khoảng|tcn|trước công nguyên|thập niên|đầu|cuối|giữa|\?", re.IGNORECASE)
INCOMING_CATEGORY = [("P825", "deity"), ("P2348", "period"), ("P140", "religion"), ("P1435", "designation"),
                     ("P149", "style"), ("P31", "class")]


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).split("#", 1)[0].replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]


def parse_infobox(wikitext: str, templates: list[str], fields: list[str]) -> list[dict]:
    wanted = {normalize_title(t) for t in templates}
    fields = {unicodedata.normalize("NFC", f) for f in fields}
    rows = []
    for tpl in mwparserfromhell.parse(wikitext).filter_templates(recursive=False):
        name = normalize_title(tpl.name)
        if name not in wanted:
            continue
        for param in tpl.params:
            field = unicodedata.normalize("NFC", str(param.name)).strip()
            if field not in fields:
                continue
            links = [normalize_title(link.title) for link in param.value.filter_wikilinks()]
            links = [t for t in links if t and not t.startswith(NAMESPACES)]
            raw = re.sub(r"\s+", " ", param.value.strip_code()).strip()
            if raw or links:
                rows.append({"template": name, "field": field, "raw": raw, "links": links})
    return rows


def parse_year(text: str) -> int | None:
    text = unicodedata.normalize("NFC", text or "")
    if not text or APPROXIMATE.search(text):
        return None
    years = {int(y) for y in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", text)}
    return years.pop() if len(years) == 1 else None


def validate_periods(periods: list[dict]) -> None:
    for p in periods:
        if not p.get("qid"):
            raise ValueError(f"period {p['label_vi']} has no qid (fill it at gate G1)")
        if p["start"] > p["end"]:
            raise ValueError(f"period {p['label_vi']}: start > end")
    ordered = sorted(periods, key=lambda p: p["start"])
    for a, b in zip(ordered, ordered[1:]):
        if b["start"] <= a["end"]:
            raise ValueError(f"periods overlap: {a['label_vi']} / {b['label_vi']}")


def period_for_year(year: int, periods: list[dict]) -> str | None:
    for p in periods:
        if p["start"] <= year <= p["end"]:
            return p["qid"]
    return None


def province_of(qid: str, entities: dict, province_classes: list[str], max_depth: int) -> tuple[str | None, int]:
    """Walk every truthy P131 path upwards; each path stops at its first province-level entity.

    Exactly one distinct province over all paths -> (province, depth); none or several -> (None, 0).
    """
    classes = set(province_classes)
    found: dict[str, int] = {}
    frontier, seen = {qid}, {qid}
    for depth in range(1, max_depth + 1):
        parents = {p for q in frontier if q in entities for p in truthy(entities[q], "P131")}
        for p in parents:
            if p in entities and classes & set(truthy(entities[p], "P31")):
                found.setdefault(p, depth)
        frontier = {p for p in parents if p not in found} - seen
        seen |= parents
        if not frontier:
            break
    if len(found) == 1:
        return next(iter(found.items()))
    return None, 0


def triple_key(t: dict) -> str:
    return f"{t['h']}|{t['r']}|{t['t']}|{t['source']}"


def propose_triples(landmarks: list[str], entities: dict, infobox_rows: dict[str, list[dict]],
                    link_qids: dict[str, str], cfg_v: dict) -> tuple[list[dict], dict[str, int]]:
    counts: Counter = Counter()
    found: dict[tuple, dict] = {}

    def add(h, r, t, source, evidence):
        if t not in entities:
            counts["target_not_cached"] += 1
            return
        if h == t:
            return
        old = found.get((h, r, t))
        if old is None or SOURCE_ORDER[source] < SOURCE_ORDER[old["source"]]:
            found[(h, r, t)] = {"h": h, "r": r, "t": t, "source": source, "evidence": evidence}

    for lm in landmarks:
        e = entities[lm]
        for pid in cfg_v["landmark_pids"]:
            for t in truthy(e, pid):
                if isinstance(t, str) and t.startswith("Q"):
                    add(lm, pid, t, "wikidata", f"wikidata:{pid}")
        rows = infobox_rows.get(lm, [])
        for row in rows:
            pid = cfg_v["infobox_fields"].get(row["field"])
            if pid is None:
                continue
            if not row["links"]:
                counts["infobox_unlinked"] += 1
            for title in row["links"]:
                target = link_qids.get(title)
                if target is None:
                    counts["infobox_link_unresolved"] += 1
                else:
                    add(lm, pid, target, "infobox", f"{row['template']}|{row['field']}={row['raw']}")
        province, depth = province_of(lm, entities, cfg_v["province_classes"], cfg_v["max_p131_depth"])
        if province is None:
            counts["no_province"] += 1
        elif depth == 1:
            add(lm, "P131", province, "wikidata", "wikidata:P131")
        else:
            add(lm, "P131", province, "derived", f"rule:province via truthy P131 chain, depth {depth}")
        year, origin = inception_year(e), "wikidata:P571"
        if year is None:
            row = next((r for r in rows if r["field"] in cfg_v["founded_fields"]), None)
            if row is not None:
                year, origin = parse_year(row["raw"]), f"{row['template']}|{row['field']}={row['raw']}"
        if year is not None:
            period = period_for_year(year, cfg_v["periods"])
            if period is None:
                counts["year_without_period"] += 1
            else:
                add(lm, "P2348", period, "derived", f"rule:period year {year} from {origin}")

    landmark_set = set(landmarks)
    context = {t["t"] for t in found.values()} - landmark_set
    for c in sorted(context, key=qid_key):
        for pid in [*cfg_v["landmark_pids"], *cfg_v["context_extra_pids"]]:
            for t in truthy(entities[c], pid):
                if t in context:
                    add(c, pid, t, "wikidata", f"wikidata:{pid}")
    triples = sorted(found.values(), key=lambda t: (qid_key(t["h"]), t["r"], qid_key(t["t"])))
    return triples, dict(sorted(counts.items()))


def categorize(qid: str, landmark_set: set[str], entities: dict, triples: list[dict], cfg_v: dict) -> str:
    if qid in landmark_set:
        return "landmark"
    p31 = set(truthy(entities[qid], "P31"))
    if p31 & set(cfg_v["province_classes"]):
        return "province"
    if cfg_v["human_class"] in p31:
        return "person"
    incoming = {t["r"] for t in triples if t["t"] == qid}
    for pid, category in INCOMING_CATEGORY:
        if pid in incoming:
            return category
    return "other"


def cap_entities(landmarks: list[str], triples: list[dict], max_entities: int) -> tuple[list[str], list[dict], int, int]:
    if len(landmarks) > max_entities:
        raise ValueError(f"{len(landmarks)} landmarks exceed max_entities={max_entities}")
    landmark_set = set(landmarks)
    degree: Counter = Counter()
    for t in triples:
        for end in (t["h"], t["t"]):
            if end not in landmark_set:
                degree[end] += 1
    ranked = sorted(degree, key=lambda q: (-degree[q], qid_key(q)))
    kept = ranked[: max_entities - len(landmarks)]
    keep = landmark_set | set(kept)
    kept_triples = [t for t in triples if t["h"] in keep and t["t"] in keep]
    return sorted(kept, key=qid_key), kept_triples, len(ranked) - len(kept), len(triples) - len(kept_triples)


def gold_from_pattern(pattern: list[list[str]], triples: list[dict], landmarks: list[str]) -> list[str]:
    facts = {(t["h"], t["r"], t["t"]) for t in triples}
    return sorted((lm for lm in landmarks if all((lm, r, o) in facts for r, o in pattern)), key=qid_key)
