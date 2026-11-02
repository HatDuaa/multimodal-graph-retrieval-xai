"""Vietnamese text handling for the Vietnamese MMKG queries (spec 4.4). Pure functions.

Auto queries are the first sentences of a landmark's viwiki lead with every known form of
its name removed: label, aliases, article title, and the same names without a leading type
word ("Chùa Một Cột" -> "Một Cột"). Matching is on Unicode NFC text, case-insensitive, on
word boundaries (Vietnamese letters with diacritics count as word characters). Parenthetical
groups are removed first because they usually hold Han characters and alternate names that
Wikidata does not list. A sentence is dropped if it is too short after masking or if any
name form is still found in it.
"""
import re
import unicodedata
from collections import Counter

# lower-cased tokens that end with a period without ending the sentence
ABBREVIATIONS = {"tp", "tt", "st", "ts", "ths", "pgs", "gs", "th", "tk", "v.v", "q", "p", "ng", "tx"}


def nfc(text: str | None) -> str:
    return unicodedata.normalize("NFC", text or "")


def _strip_disambiguation(title: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", title).strip()


def name_variants(label: str | None, aliases: list[str], title: str | None, prefixes: list[str]) -> list[str]:
    base = [nfc(n).strip() for n in [label, *aliases, _strip_disambiguation(nfc(title)) if title else None] if n]
    base = [n for n in base if n]
    names = list(base)
    ordered_prefixes = sorted((nfc(p) for p in prefixes), key=len, reverse=True)
    for name in base:
        for prefix in ordered_prefixes:
            if name.casefold().startswith(prefix.casefold() + " "):
                rest = name[len(prefix):].strip()
                if rest:
                    names.append(rest)
                break
    seen, out = set(), []
    for name in sorted(names, key=lambda s: (-len(s), s)):
        if name.casefold() not in seen:
            seen.add(name.casefold())
            out.append(name)
    return out


def _name_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(nfc(name))}(?!\w)", re.IGNORECASE)


def mask_names(text: str, names: list[str]) -> str:
    text = nfc(text)
    for name in sorted(names, key=len, reverse=True):
        text = _name_pattern(name).sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,;:.!?)])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    text = re.sub(r"^[\s,;:.'\"-]+", "", text)
    return text.strip()


def contains_name(text: str, names: list[str]) -> bool:
    text = nfc(text)
    return any(_name_pattern(name).search(text) for name in names)


def strip_parentheticals(text: str) -> str:
    text = nfc(text)
    while True:
        stripped = re.sub(r"\s*\([^()]*\)", "", text)
        if stripped == text:
            return text
        text = stripped


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", nfc(text)).strip()
    sentences, start = [], 0
    for m in re.finditer(r"[.!?…]+[\"”’)]*\s+", text):
        following = text[m.end():m.end() + 1]
        if not (following.isupper() or following.isdigit() or following in "\"“("):
            continue
        words = text[start:m.start() + 1].split()
        last = words[-1] if words else ""
        token = last.rstrip(".").lower()
        if last.endswith(".") and (token in ABBREVIATIONS or (len(token) == 1 and token.isalpha())):
            continue
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text):
        sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def make_auto_queries(qid: str, extract: str, revid: int, names: list[str], max_sentences: int,
                      min_words: int) -> tuple[list[dict], dict[str, int]]:
    drops: Counter = Counter()
    queries = []
    for i, sentence in enumerate(split_sentences(extract)[:max_sentences]):
        masked = mask_names(strip_parentheticals(sentence), names)
        if len(masked.split()) < min_words:
            drops["too_short"] += 1
        elif contains_name(masked, names):
            drops["name_left"] += 1
        else:
            queries.append({"query_id": f"{qid}_auto_{i}", "qid": qid, "text": masked,
                            "sentence_index": i, "viwiki_revid": revid})
    return queries, dict(sorted(drops.items()))
