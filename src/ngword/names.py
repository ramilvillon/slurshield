from pathlib import Path

import ahocorasick

from ngword.normalize import fold

_WL_DIR = Path(__file__).resolve().parents[2] / "wordlists"


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


def _build_automaton(words: list[str]) -> ahocorasick.Automaton:
    a = ahocorasick.Automaton()
    for w in words:
        folded = fold(w)
        if folded:
            a.add_word(folded, folded)
    a.make_automaton()
    return a


_PROFANITY = _build_automaton(
    _load_lines("en.txt") + _load_lines("zh.txt") + _load_lines("ja.txt") + _load_lines("tl.txt")
)
_ALLOWLIST = [fold(w) for w in _load_lines("allowlist.txt")]
_IMPERSONATION = _build_automaton(_load_lines("impersonation.txt"))


def _covered_by_allowlist(shadow: str, start: int, end: int) -> bool:
    # ponytail: allow a hit if it sits inside a benign word present in the shadow.
    # Ceiling: purely substring-based; upgrade to span-aware token check if FPs appear.
    for good in _ALLOWLIST:
        idx = shadow.find(good)
        while idx != -1:
            if idx <= start and end <= idx + len(good):
                return True
            idx = shadow.find(good, idx + 1)
    return False


def check_name(text: str) -> dict:
    shadow = fold(text)
    if not shadow:
        return {"decision": "allow", "reason": "clean"}

    for end_idx, term in _IMPERSONATION.iter(shadow):
        return {"decision": "block", "reason": "impersonation"}

    for end_idx, term in _PROFANITY.iter(shadow):
        start_idx = end_idx - len(term) + 1
        if _covered_by_allowlist(shadow, start_idx, end_idx):
            continue
        return {"decision": "block", "reason": term}

    return {"decision": "allow", "reason": "clean"}
