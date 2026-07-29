import re
from pathlib import Path

import ahocorasick

from ngword.normalize import fold

_WL_DIR = Path(__file__).resolve().parents[2] / "wordlists"

# Names have no word boundaries, so unbounded substring matching of short keys
# over-blocks ("hoe" -> "Phoenix", "nig" -> "Enigma", "ass" -> "Cassandra").
# Tier by length: short slurs match only as whole tokens; long slurs (rare inside
# real words) match as substrings so separator/leet evasion is still caught.
_LONG_MIN_LEN = 5

# Split on whitespace/punctuation/underscore but KEEP Unicode letters (incl. homoglyphs
# like Cyrillic 'а') and digits in the token, so fold() can canonicalize them.
_SEP = re.compile(r"[\W_]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _load_lines(name: str) -> list[str]:
    path = _WL_DIR / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            out.append(w)
    return out


def _raw_tokens(text: str) -> set[str]:
    # Split on separators and camelCase; return RAW pieces (unfolded) so the caller
    # can both fold them and strip trailing digits before folding.
    out = set()
    for chunk in _SEP.split(text):
        for piece in _CAMEL.sub(" ", chunk).split():
            if piece:
                out.add(piece)
    return out


def _token_forms(raw: str) -> set[str]:
    # Folded token, plus (for "puta123"/"admin1") the fold of the digit-stripped token.
    # Stripping digits BEFORE folding avoids fold()'s leet map turning "123" into "i2e".
    forms = {fold(raw)}
    stripped = raw.rstrip("0123456789")
    if stripped and stripped != raw:
        forms.add(fold(stripped))
    forms.discard("")
    return forms


_ALL_PROFANITY = {
    fold(w) for w in
    _load_lines("en.txt") + _load_lines("zh.txt") + _load_lines("ja.txt") + _load_lines("tl.txt")
}
_ALL_PROFANITY = {w for w in _ALL_PROFANITY if len(w) >= 3}  # <3 is pure noise even as a token
_SHORT_PROFANITY = {w for w in _ALL_PROFANITY if len(w) < _LONG_MIN_LEN}
_IMPERSONATION_TERMS = {fold(w) for w in _load_lines("impersonation.txt")} - {""}


def _build_automaton(words) -> ahocorasick.Automaton:
    a = ahocorasick.Automaton()
    for w in words:
        a.add_word(w, w)
    a.make_automaton()
    return a


_LONG_PROFANITY = _build_automaton([w for w in _ALL_PROFANITY if len(w) >= _LONG_MIN_LEN])
_ALLOWLIST = [fold(w) for w in _load_lines("allowlist.txt")]


def _covered_by_allowlist(shadow: str, start: int, end: int) -> bool:
    # ponytail: allow a long-term hit that sits inside a benign word present in the shadow.
    # Ceiling: purely substring-based; upgrade to span-aware token check if FPs appear.
    for good in _ALLOWLIST:
        idx = shadow.find(good)
        while idx != -1:
            if idx <= start and end <= idx + len(good):
                return True
            idx = shadow.find(good, idx + 1)
    return False


def check_name(text: str) -> dict:
    for raw in _raw_tokens(text):
        forms = _token_forms(raw)
        if forms & _IMPERSONATION_TERMS:
            return {"decision": "block", "reason": "impersonation"}
        hit = forms & _SHORT_PROFANITY
        if hit:
            return {"decision": "block", "reason": sorted(hit)[0]}

    shadow = fold(text)
    if not shadow:
        return {"decision": "allow", "reason": "clean"}

    for end_idx, term in _LONG_PROFANITY.iter(shadow):
        start_idx = end_idx - len(term) + 1
        if _covered_by_allowlist(shadow, start_idx, end_idx):
            continue
        return {"decision": "block", "reason": term}

    return {"decision": "allow", "reason": "clean"}
