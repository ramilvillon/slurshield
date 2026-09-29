import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿⁠"), None)

# Latin-lookalike homoglyphs NFKC does not cover (Cyrillic/Greek).
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "ѕ": "s", "і": "i", "ј": "j", "к": "k", "н": "h", "т": "t", "м": "m",
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i",
})

_LEET = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "+": "t", "|": "i",
})

_KEEP = re.compile(r"[0-9a-z぀-ヿ一-鿿가-힯]")  # every other char is a separator, dropped
_RUN_MIN = 3  # ponytail: collapses runs of >=3 only; 3+ repeats is the evasion signal


def fold_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """fold(), plus the source span in `text` that produced each shadow char.

    Folding is length-lossy ('f  u  c  k' -> 'fuck'), so a matcher hit on the shadow
    cannot be located in the original without this map. Spans, not single indices:
    one shadow char can come from a run ('iiii' -> 'i') or a ligature ('ﬁ' -> 'fi').
    """
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        # NFKC composes a base char with its combining marks, so fold whole clusters —
        # char-by-char would leave marks behind that the whole-string pass absorbs.
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        piece = unicodedata.normalize("NFKC", text[i:j].translate(_ZERO_WIDTH))
        for ch in piece.lower().translate(_HOMOGLYPHS).translate(_LEET):
            if _KEEP.match(ch):
                chars.append(ch)
                spans.append((i, j))
        i = j

    shadow: list[str] = []
    src: list[tuple[int, int]] = []
    k = 0
    while k < len(chars):
        end = k
        while end < len(chars) and chars[end] == chars[k]:
            end += 1
        if end - k >= _RUN_MIN:  # collapsed run: one char covering the whole run
            shadow.append(chars[k])
            src.append((spans[k][0], spans[end - 1][1]))
        else:
            shadow += chars[k:end]
            src += spans[k:end]
        k = end
    return "".join(shadow), src


def fold(text: str) -> str:
    """Return a lowercase match-shadow for the matcher path only. Never applied to model input."""
    return fold_map(text)[0]
