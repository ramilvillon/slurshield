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

_SEP = re.compile(r"[^0-9a-z぀-ヿ一-鿿가-힯]+")
_RUN = re.compile(r"(.)\1{2,}")  # ponytail: collapses runs of >=3 only; 3+ repeats is the evasion signal


def fold(text: str) -> str:
    """Return a lowercase match-shadow for the matcher path only. Never applied to model input."""
    if not text:
        return ""
    text = text.translate(_ZERO_WIDTH)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(_HOMOGLYPHS)
    text = text.translate(_LEET)
    text = _SEP.sub("", text)
    text = _RUN.sub(r"\1", text)
    return text
