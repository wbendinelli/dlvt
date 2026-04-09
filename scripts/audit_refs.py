#!/usr/bin/env python3
"""
DLVT R7-9 literature audit — existence pass.

Parses references.bib, queries Crossref for each entry, classifies each as:
  OK_DOI       - has DOI and DOI resolves on Crossref
  DOI_BAD      - has DOI but Crossref returns no record
  FOUND_NODOI  - no DOI, Crossref bibliographic search returns confident match
  WEAK_NODOI   - no DOI, Crossref returns a candidate but score is borderline
  MISSING      - no DOI and Crossref returns no plausible match
  BOOK_SKIP    - book/techreport without DOI (Crossref coverage thin); flagged for manual check

Outputs JSON to stdout. Polite to Crossref: 1 req/sec, User-Agent with email.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path("/sessions/great-loving-ride/mnt/dynamic-leadership-vitality-theory/references.bib")
UA = "DLVT-AuditBot/1.0 (mailto:wbendinelli@gmail.com)"
SLEEP = 0.4  # seconds between requests, polite

# ----------- bib parser (lightweight, sufficient for our flat bib) ----------

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def parse_bib(text):
    entries = []
    for m in ENTRY_RE.finditer(text):
        etype, key = m.group(1).lower(), m.group(2)
        # find matching closing brace from m.end()
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        body = text[m.end() : i - 1]
        fields = parse_fields(body)
        fields["__type__"] = etype
        fields["__key__"] = key
        entries.append(fields)
    return entries


FIELD_RE = re.compile(r"(\w+)\s*=\s*", re.MULTILINE)


def parse_fields(body):
    fields = {}
    pos = 0
    while pos < len(body):
        m = FIELD_RE.search(body, pos)
        if not m:
            break
        name = m.group(1).lower()
        v_start = m.end()
        # value is in {...}, "...", or bare digits
        c = body[v_start] if v_start < len(body) else ""
        if c == "{":
            depth = 1
            j = v_start + 1
            while j < len(body) and depth > 0:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            value = body[v_start + 1 : j - 1]
            pos = j
        elif c == '"':
            j = v_start + 1
            while j < len(body) and body[j] != '"':
                j += 1
            value = body[v_start + 1 : j]
            pos = j + 1
        else:
            j = v_start
            while j < len(body) and body[j] not in ",\n":
                j += 1
            value = body[v_start:j].strip()
            pos = j
        # strip trailing comma/whitespace
        fields[name] = clean(value)
        # advance past comma
        while pos < len(body) and body[pos] in ", \n\t":
            pos += 1
    return fields


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("{", "").replace("}", "")
    return s


# ----------- Crossref helpers ----------


def http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"__error__": str(e)}


def check_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    j = http_get_json(url)
    if "__error__" in j:
        return False, j["__error__"], None
    msg = j.get("message", {})
    title = (msg.get("title") or [""])[0]
    return True, None, title


def search_bib(entry):
    parts = []
    if entry.get("title"):
        parts.append(entry["title"])
    if entry.get("author"):
        first_author = entry["author"].split(" and ")[0]
        # last name only
        if "," in first_author:
            parts.append(first_author.split(",")[0])
        else:
            parts.append(first_author.split()[-1] if first_author else "")
    if entry.get("year"):
        parts.append(entry["year"])
    q = " ".join(parts)
    url = (
        "https://api.crossref.org/works?rows=3&query.bibliographic="
        + urllib.parse.quote(q)
    )
    j = http_get_json(url)
    if "__error__" in j:
        return None, j["__error__"]
    items = j.get("message", {}).get("items", [])
    return items, None


def title_similarity(a, b):
    """Crude jaccard on lowercased word sets, ignoring stopwords."""
    stop = {
        "the", "a", "an", "of", "and", "in", "on", "for", "to", "with",
        "from", "by", "is", "are", "as", "at",
    }
    wa = {w for w in re.findall(r"\w+", a.lower()) if w not in stop}
    wb = {w for w in re.findall(r"\w+", b.lower()) if w not in stop}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ----------- main ----------


def main():
    text = BIB.read_text(encoding="utf-8")
    entries = parse_bib(text)
    print(f"# parsed {len(entries)} entries", file=sys.stderr)
    results = []
    for i, e in enumerate(entries, 1):
        key = e["__key__"]
        etype = e["__type__"]
        title = e.get("title", "")
        doi = e.get("doi", "")
        rec = {
            "key": key,
            "type": etype,
            "title": title,
            "year": e.get("year", ""),
            "journal": e.get("journal", ""),
            "doi": doi,
            "status": None,
            "note": "",
            "suggested_doi": None,
            "suggested_title": None,
        }
        if doi:
            ok, err, ctitle = check_doi(doi)
            time.sleep(SLEEP)
            if ok:
                sim = title_similarity(title, ctitle or "")
                if sim >= 0.4:
                    rec["status"] = "OK_DOI"
                else:
                    rec["status"] = "DOI_TITLE_MISMATCH"
                    rec["note"] = f"Crossref title: {ctitle!r} (sim={sim:.2f})"
            else:
                rec["status"] = "DOI_BAD"
                rec["note"] = err or "no record"
        else:
            if etype in ("book", "techreport", "incollection", "misc"):
                rec["status"] = "BOOK_SKIP"
                rec["note"] = f"{etype} without DOI; manual check needed"
            else:
                items, err = search_bib(e)
                time.sleep(SLEEP)
                if err:
                    rec["status"] = "SEARCH_ERROR"
                    rec["note"] = err
                elif not items:
                    rec["status"] = "MISSING"
                else:
                    best = items[0]
                    btitle = (best.get("title") or [""])[0]
                    sim = title_similarity(title, btitle)
                    rec["suggested_doi"] = best.get("DOI")
                    rec["suggested_title"] = btitle
                    if sim >= 0.6:
                        rec["status"] = "FOUND_NODOI"
                    elif sim >= 0.35:
                        rec["status"] = "WEAK_NODOI"
                    else:
                        rec["status"] = "MISSING"
                    rec["note"] = f"sim={sim:.2f}"
        print(f"[{i:>3}/{len(entries)}] {rec['status']:<22} {key}", file=sys.stderr)
        results.append(rec)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
