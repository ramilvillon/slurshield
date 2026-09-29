import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from training.labels import dedup, harmonize

_DATA = Path(__file__).resolve().parents[1] / "data"
_RAW = _DATA / "raw"


def stratified_split(rows: list[dict], seed: int = 42):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["lang"], r["label"])].append(r)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for key in sorted(buckets):
        group = buckets[key]
        rng.shuffle(group)
        n = len(group)
        n_test = int(n * 0.1)
        n_val = int(n * 0.1)
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    return train, val, test


def _load_conda() -> list[dict]:
    url = "https://raw.githubusercontent.com/usydnlp/CONDA/main/data/CONDA_train.csv"
    df = pd.read_csv(url)
    out = []
    for _, r in df.iterrows():
        lab = harmonize("conda", str(r["intentClass"]).strip())
        if lab is not None and isinstance(r["utterance"], str):
            out.append({"text": r["utterance"], "label": lab, "lang": "en", "source": "conda"})
    return out


def _load_hf_mginoben() -> list[dict]:
    from datasets import load_dataset
    out = []
    for split in ("train", "validation"):
        ds = load_dataset("mginoben/tagalog-profanity-dataset", split=split)
        for r in ds:
            lab = harmonize("mginoben", r["label"])
            if lab is not None:
                out.append({"text": r["text"], "label": lab, "lang": "tl", "source": "mginoben"})
    return out


def _load_textdetox(lang: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("textdetox/multilingual_toxicity_dataset", split=lang)
    out = []
    for r in ds:
        native = "toxic" if int(r["toxic"]) == 1 else "neutral"
        lab = harmonize("textdetox", native)
        if lab is not None:
            out.append({"text": r["text"], "label": lab, "lang": lang, "source": "textdetox"})
    return out


def _load_raw_csv(name: str, source: str, lang: str, text_col: str, label_col: str,
                  segment: bool = False) -> list[dict]:
    path = _RAW / name
    if not path.exists():
        print(f"SKIP {name} (not found — see data/SOURCES.md)")
        return []
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        text = r[text_col]
        if not isinstance(text, str):
            continue
        lab = harmonize(source, r[label_col])
        if lab is None:
            continue
        chunks = _segment(text) if segment else [text]
        for c in chunks:
            out.append({"text": c, "label": lab, "lang": lang, "source": source})
    return out


def _toxicn_class(toxic, expression) -> int | None:
    # ToxiCN's raw `expression` (0=non-hate,1=explicit,2=implicit,3=reporting) mislabels
    # 816 genuinely-toxic rows as expression=0. Collapse toxic-first: clean only when
    # toxic==0; untyped-toxic folds to explicit. Handles the raw file (has `toxic`) and a
    # pre-collapsed file (expression already 0-3, no `toxic` column).
    try:
        expr = int(expression)
    except (TypeError, ValueError):
        return None
    if toxic is None or (isinstance(toxic, float) and pd.isna(toxic)):
        return harmonize("toxicn", expr)  # pre-collapsed: identity map
    return 0 if int(toxic) == 0 else (expr if expr in (1, 2, 3) else 1)


def _load_toxicn() -> list[dict]:
    path = _RAW / "toxicn.csv"
    if not path.exists():
        print("SKIP toxicn.csv (not found — see data/SOURCES.md)")
        return []
    df = pd.read_csv(path)
    has_toxic = "toxic" in df.columns
    out = []
    for _, r in df.iterrows():
        text = r.get("content")
        if not isinstance(text, str):
            continue
        lab = _toxicn_class(r["toxic"] if has_toxic else None, r.get("expression"))
        if lab is not None:
            out.append({"text": text, "label": lab, "lang": "zh", "source": "toxicn"})
    return out


def _segment(text: str) -> list[str]:
    # ja long-doc handling: split on sentence enders, keep chat-length chunks only.
    import re
    parts = re.split(r"[。！？\n]", text)
    return [p.strip() for p in parts if 1 <= len(p.strip()) <= 120]


def _balance(rows: list[dict], seed: int = 42, cap_per_lang: int = 20000) -> list[dict]:
    # Downsample so no single language dominates; keep class ratio within each language.
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    out = []
    for lang, group in by_lang.items():
        if len(group) > cap_per_lang:
            rng.shuffle(group)
            group = group[:cap_per_lang]
        out += group
    return out


def build() -> None:
    rows = []
    rows += _load_conda()
    rows += _load_hf_mginoben()
    rows += _load_textdetox("zh")
    rows += _load_textdetox("ja")
    rows += _load_raw_csv("cold.csv", "cold", "zh", "TEXT", "label")
    rows += _load_toxicn()
    rows += _load_raw_csv("llmjp.csv", "llmjp", "ja", "text", "label", segment=True)

    rows = dedup(rows)
    rows = _balance(rows)
    train, val, test = stratified_split(rows)

    _DATA.mkdir(exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame(split).to_parquet(_DATA / f"{name}.parquet", index=False)
        print(f"{name}: {len(split)} rows")


if __name__ == "__main__":
    build()
