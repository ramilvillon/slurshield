import re

_WS = re.compile(r"\s+")

# Each source maps its native label -> 4-class index (0 clean,1 explicit,2 implicit,3 action).
_MAPS = {
    "conda":     {"O": 0, "E": 1, "I": 2, "A": 3},
    "mginoben":  {0: 0, 1: 1, "0": 0, "1": 1},
    "cold":      {0: 0, 1: 1, "0": 0, "1": 1},
    "toxicn":    {0: 0, 1: 1, 2: 2, 3: 3, "0": 0, "1": 1, "2": 2, "3": 3},
    "textdetox": {"neutral": 0, "toxic": 1},
    "llmjp":     {"non": 0, "toxic": 1},
}


def harmonize(source: str, native_label):
    return _MAPS[source].get(native_label)


def dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = _WS.sub(" ", r["text"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
