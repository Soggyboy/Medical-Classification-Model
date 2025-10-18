"""
rules.py — Rule-based text classification for medication dose frequency.

Detects labels like QD, BID, TID, QID, PRN, QWK, etc.
Used to generate "Primary", "Secondary", and "Tertiary" class predictions
before training the neural model.

All regex finders return (label, span) tuples if matched.
"""

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# Vocabulary and mappings
# ---------------------------------------------------------------------

_NUM_WORDS = {
    "one": 1, "once": 1, "single": 1,
    "two": 2, "twice": 2, "double": 2,
    "three": 3, "thrice": 3,
    "four": 4
}

_NUM_TO_LABEL = {1: "QD", 2: "BID", 3: "TID", 4: "QID"}


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize string: lowercase, strip punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("\n", " ").replace("\\n", " ")
    s = re.sub(r"[()\[\],;:+\-]", " ", s)  # keep digits/letters; remove clutter
    s = s.replace(".", "")                  # remove periods ("p.r.n." → "prn")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_to_int(tok: str) -> Optional[int]:
    """Convert a word or numeric token to integer 1–4 if possible."""
    tok = tok.lower()
    if tok.isdigit():
        n = int(tok)
        return n if n in (1, 2, 3, 4) else None
    return _NUM_WORDS.get(tok)


# ---------------------------------------------------------------------
# Regex finders
# ---------------------------------------------------------------------

def _find_direct(s: str) -> Optional[Tuple[str, tuple]]:
    """Direct matches for standard abbreviations (QD/BID/TID/QID/PRN)."""
    patterns = [
        (r"\b2\s*qhs\b", "BID"),  # override special case
        (r"\bqid\b|\bq\s*i\s*d\b", "QID"),
        (r"\btid\b|\bt\s*i\s*d\b", "TID"),
        (r"\bbid\b|\bb\s*i\s*d\b", "BID"),
        (r"\bqwk\b|\b(once|every)\s+(a|per)?\s*week(ly)?\b|\b1\s*(x|time)\s*(a|per|/)\s*week(ly)?\b", "QWK"),
        (r"\bqd\b|\bq\s*d\b|\bdaily\b|\bonce\s+(a|per)?\s*day\b|\b1\s*(x|time)?\s*(per|/)?\s*(day|daily)\b", "QD"),
        (r"\bq\s*24\s*h\b|\bevery\s*24\s*h(ou)?rs?\b", "QD"),
        (r"\bqam\b|\bqpm\b|\bqhs\b|\b(at\s*)?bedtime\b", "QD"),
        (r"\bprn\b|\b(as|when|if)\s+needed\b|\bas\s+(necessary|required)\b", "PRN"),
    ]
    for pat, label in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return label, m.span(0)
    return None


def _find_every_hours(s: str) -> Optional[Tuple[str, tuple]]:
    """Detect 'every X hours' patterns and convert to daily frequency."""
    m = re.search(r"\bq\s*(\d{1,2})\s*h\b|\bevery\s*(\d{1,2})\s*(h|hr|hrs|hour|hours)\b", s, flags=re.I)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    try:
        h = int(val)
        if h:
            per_day = round(24 / h)
            if per_day in _NUM_TO_LABEL:
                return _NUM_TO_LABEL[per_day], m.span(0)
    except ValueError:
        pass
    return None


def _find_numeric_daybased(s: str) -> Optional[Tuple[str, tuple]]:
    """Find numeric per-day dosage counts, e.g., '3x/day', '2 per day'."""
    patterns = [
        r"x\s*(\d)\s*/\s*(day|daily)\b",
        r"x\s*(\d)\s*(?:per\s*)?(day|daily)\b",
        r"\b(\d)\s*x\s*(?:per\s*)?(day|daily)\b",
        r"\b(\d)\s*(times?|time)\s*(?:a|per|/)?\s*(day|daily)\b",
        r"\b(\d)\s*/\s*(day|daily)\b",
        r"\b(\d)\s*(?:x|×)\b.*\b(day|daily)\b",
        r"x\s*(\d)\s*/",  # fallback: "x3/"
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            n = int(m.group(1))
            if n in _NUM_TO_LABEL:
                return _NUM_TO_LABEL[n], m.span(0)
    return None


def _find_worded_counts_or_range(s: str) -> Optional[Tuple[str, tuple]]:
    """Handle spelled-out quantities like 'twice a day', 'one to two times daily'."""
    # Range form: "one or two times daily"
    rng = re.search(
        r"\b(one|once|single|two|twice|double|three|thrice|four|1|2|3|4)\b"
        r".{0,20}?\b(or|to|/|-)\b.{0,20}?\b(one|once|single|two|twice|double|three|thrice|four|1|2|3|4)\b"
        r".{0,20}?\b(day|daily|times?)\b",
        s, flags=re.I,
    )
    if rng:
        a = _token_to_int(rng.group(1))
        b = _token_to_int(rng.group(3))
        if a in _NUM_TO_LABEL and b in _NUM_TO_LABEL:
            return _NUM_TO_LABEL[min(a, b)], rng.span(1)

    # Single form: "twice daily"
    m = re.search(
        r"\b(once|one|single|twice|two|double|three|thrice|four)\b"
        r".{0,20}?\b(day|daily|times?)\b",
        s, flags=re.I,
    )
    if m:
        n = _token_to_int(m.group(1))
        if n in _NUM_TO_LABEL:
            return _NUM_TO_LABEL[n], m.span(0)
    return None


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def classify_all(text: str, max_labels: int = 3):
    """
    Extract up to `max_labels` dose frequency labels (primary→secondary→tertiary).

    Args:
        text (str): The input dose text.
        max_labels (int): Maximum number of frequency labels to extract.

    Returns:
        list[str | None]: List of found labels (QD/BID/etc), or [None, None, None].
    """
    s = _norm(text)
    out = []
    for _ in range(max_labels):
        found = None
        for finder in (_find_worded_counts_or_range, _find_numeric_daybased, _find_every_hours, _find_direct):
            res = finder(s)
            if res:
                label, span = res
                out.append(label)
                # remove matched chunk before next search
                s = (s[:span[0]] + " " + s[span[1]:]).strip()
                s = re.sub(r"\s+", " ", s)
                found = True
                break
        if not found:
            break
    return out if out else [None, None, None]
