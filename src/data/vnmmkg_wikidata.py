"""Pure helpers over Wikidata entities for the Vietnamese MMKG (no network).

`trim_entity` turns one `wbgetentities` answer into the compact cached form; the other
functions read that form. "Truthy" values follow Wikidata's best-rank rule: preferred
statements if any exist, otherwise normal ones; deprecated statements never count.
"""
KEEP_PIDS_EXTRA = {"P17", "P131", "P571", "P18", "P373"}


def qid_key(qid: str) -> int:
    return int(qid[1:])


def _snak_value(snak: dict):
    if snak.get("snaktype") != "value":
        return None
    dv = snak["datavalue"]
    if dv["type"] == "wikibase-entityid":
        return dv["value"]["id"]
    if dv["type"] == "time":
        return dv["value"]["time"], dv["value"]["precision"]
    if dv["type"] == "string":
        return dv["value"]
    return None


def trim_entity(raw: dict, keep_pids: set[str]) -> dict | None:
    if "missing" in raw:
        return None
    claims = {}
    for pid in sorted(keep_pids):
        values = []
        for st in raw.get("claims", {}).get(pid, []):
            v = _snak_value(st["mainsnak"])
            if v is None:
                continue
            if isinstance(v, tuple):
                values.append({"value": v[0], "rank": st["rank"], "precision": v[1]})
            else:
                values.append({"value": v, "rank": st["rank"]})
        if values:
            claims[pid] = values
    sitelinks = raw.get("sitelinks", {})
    return {
        "qid": raw["id"],
        "lastrevid": raw.get("lastrevid"),
        "label_vi": raw.get("labels", {}).get("vi", {}).get("value"),
        "label_en": raw.get("labels", {}).get("en", {}).get("value"),
        "aliases_vi": [a["value"] for a in raw.get("aliases", {}).get("vi", [])],
        "desc_vi": raw.get("descriptions", {}).get("vi", {}).get("value"),
        "viwiki_title": sitelinks.get("viwiki", {}).get("title"),
        "sitelinks": len(sitelinks),
        "claims": claims,
    }


def _truthy_statements(entity: dict, pid: str) -> list[dict]:
    statements = [s for s in entity["claims"].get(pid, []) if s["rank"] != "deprecated"]
    preferred = [s for s in statements if s["rank"] == "preferred"]
    return preferred or statements


def truthy(entity: dict, pid: str) -> list[str]:
    return [s["value"] for s in _truthy_statements(entity, pid)]


def inception_year(entity: dict) -> int | None:
    for s in _truthy_statements(entity, "P571"):
        if s.get("precision", 0) >= 9 and s["value"].startswith("+"):
            return int(s["value"][1:].split("-", 1)[0])
    return None


def rejection_reason(entity: dict, cfg_v: dict) -> str | None:
    if not entity["label_vi"]:
        return "no_label_vi"
    if not entity["viwiki_title"]:
        return "no_viwiki"
    if not truthy(entity, "P373"):
        return "no_commons_category"
    if any(entity["claims"].get(pid) for pid in cfg_v["exclude_if_has"]):
        return "movable_object"
    if cfg_v["country"] not in truthy(entity, "P17"):
        return "not_in_country"
    if not (truthy(entity, "P1435") or set(truthy(entity, "P31")) & set(cfg_v["landmark_classes"])):
        return "not_landmark_class"
    return None


def rank_candidates(entities: dict[str, dict], qids: list[str], cfg_v: dict) -> tuple[list[str], dict[str, int]]:
    rejected: dict[str, int] = {}
    valid = []
    for qid in qids:
        entity = entities.get(qid)
        reason = "missing" if entity is None else rejection_reason(entity, cfg_v)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            valid.append(qid)
    valid.sort(key=lambda q: (-entities[q]["sitelinks"], qid_key(q)))
    return valid[: cfg_v["max_landmark_candidates"]], dict(sorted(rejected.items()))
