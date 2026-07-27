# ngword Profanity & Toxicity Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a name/title matcher (no ML) and a fine-tuned multilingual chat-toxicity model, exposed behind one `classify(text, kind)` inference function.

**Architecture:** Two independent components. Names/titles go through deterministic normalization + Aho-Corasick substring matching + allowlist + impersonation rules. Chat goes through a fine-tuned `mmbert-small` transformer producing a 4-class intent label. Normalization touches only the matcher path; the model always sees raw text.

**Tech Stack:** Python 3.11, PyTorch (MPS), HuggingFace `transformers` + `datasets`, `optimum[onnxruntime]`, `onnxruntime`, `pyahocorasick`, `scikit-learn`, `pandas`/`pyarrow`.

## Global Constraints

- Python 3.11 (host has 3.11.5).
- Backbone: `jhu-clsp/mmbert-small`; fallback `distilbert-base-multilingual-cased` (Task 3 decides which; all later tasks read it from `src/ngword/config.py:BACKBONE`).
- `max_length = 48`, dynamic padding.
- 4 classes, fixed index order: `0=clean, 1=explicit, 2=implicit, 3=action`.
- Chat model receives **raw** text. Normalization is applied **only** on the matcher path — never to model input.
- `title` is routed identically to `name`.
- Training runs on `torch.device("mps")` with `bf16=True`; inference runs int8 ONNX on **CPU**.
- Per-language metrics only — never a single aggregate F1.
- Conservative licensing: `mginoben` (unknown) and CONDA (no license) are dev-only, flagged as commercial-blockers.

---

### Task 1: Project scaffold + normalization

**Files:**
- Create: `pyproject.toml`
- Create: `src/ngword/__init__.py`
- Create: `src/ngword/config.py`
- Create: `src/ngword/normalize.py`
- Create: `tests/test_normalize.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: `normalize.fold(text: str) -> str` — returns a lowercase match-shadow with zero-width chars stripped, NFKC-normalized (fullwidth→ascii), homoglyphs folded, leetspeak folded, separators removed, and runs of ≥3 identical chars collapsed to one. Used by the matcher only.
- Produces: `config.BACKBONE: str`, `config.CLASS_NAMES: list[str]`, `config.MAX_LENGTH: int`.

- [ ] **Step 1: Write `.gitignore`**

```
data/
*.onnx
__pycache__/
.pytest_cache/
final_model/
checkpoints/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "ngword"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "transformers>=4.44",
    "datasets>=2.20",
    "optimum[onnxruntime]>=1.21",
    "onnxruntime>=1.18",
    "pyahocorasick>=2.1",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "pyarrow>=16",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `src/ngword/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/ngword/config.py`**

```python
# Backbone is provisional; Task 3 (backbone smoke test) confirms or flips it to the fallback.
BACKBONE = "jhu-clsp/mmbert-small"
FALLBACK_BACKBONE = "distilbert-base-multilingual-cased"
MAX_LENGTH = 48
CLASS_NAMES = ["clean", "explicit", "implicit", "action"]  # index order is fixed
```

- [ ] **Step 5: Write the failing test `tests/test_normalize.py`**

```python
from ngword.normalize import fold


def test_lowercases_and_strips_zero_width():
    assert fold("F​U​CK") == "fuck"

def test_fullwidth_folds_to_ascii():
    assert fold("ＦＵＣＫ") == "fuck"

def test_leetspeak_folds():
    assert fold("f00l") == "fool"
    assert fold("@ss") == "ass"

def test_separators_removed():
    assert fold("f.u.c.k") == "fuck"
    assert fold("p_u_t_a") == "puta"

def test_long_runs_collapsed():
    assert fold("fuuuuck") == "fuck"

def test_clean_word_unchanged():
    assert fold("hello") == "hello"

def test_cjk_passes_through():
    # NFKC + zero-width strip only; CJK chars survive for substring matching
    assert fold("操​你") == "操你"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ngword.normalize'`

- [ ] **Step 7: Write `src/ngword/normalize.py`**

```python
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
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS (7 passed)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/ngword/__init__.py src/ngword/config.py src/ngword/normalize.py tests/test_normalize.py .gitignore
git commit -m "feat: project scaffold + matcher-path normalization"
```

---

### Task 2: Name/title matcher

**Files:**
- Create: `wordlists/en.txt`, `wordlists/zh.txt`, `wordlists/ja.txt`, `wordlists/tl.txt`
- Create: `wordlists/allowlist.txt`
- Create: `wordlists/impersonation.txt`
- Create: `src/ngword/names.py`
- Create: `tests/test_names.py`

**Interfaces:**
- Consumes: `normalize.fold`.
- Produces: `names.check_name(text: str) -> dict` returning `{"decision": "block"|"allow", "reason": str}`. `reason` is the matched term, `"impersonation"`, or `"clean"`.

- [ ] **Step 1: Seed wordlists**

Fetch the public-domain lists and place the raw words (one per line) into `wordlists/{en,zh,ja}.txt`:

```bash
mkdir -p wordlists
curl -sL https://raw.githubusercontent.com/censor-text/profanity-list/main/list/en.txt -o wordlists/en.txt
curl -sL https://raw.githubusercontent.com/censor-text/profanity-list/main/list/zh.txt -o wordlists/zh.txt
curl -sL https://raw.githubusercontent.com/censor-text/profanity-list/main/list/ja.txt -o wordlists/ja.txt
```

Write `wordlists/tl.txt` by hand (the public tl list is 9 words). Seed:

```
puta
putangina
tangina
gago
gaga
tanga
bobo
ulol
punyeta
pakyu
leche
tarantado
hayop
kupal
bwisit
```

- [ ] **Step 2: Write `wordlists/allowlist.txt`** (benign strings that contain profane substrings)

```
scunthorpe
assassin
assassination
assess
assessment
bass
class
classic
grass
pass
password
analysis
analyst
cockburn
penistone
shitake
```

- [ ] **Step 3: Write `wordlists/impersonation.txt`**

```
admin
administrator
moderator
mod
gm
gamemaster
staff
support
official
system
```

- [ ] **Step 4: Write the failing test `tests/test_names.py`**

```python
from ngword.names import check_name


def test_obvious_profanity_blocks():
    assert check_name("putangina")["decision"] == "block"

def test_leetspeak_evasion_blocks():
    assert check_name("f00l_you")["reason"] == "fool" or check_name("sh1t")["decision"] == "block"

def test_separator_evasion_blocks():
    assert check_name("p.u.t.a")["decision"] == "block"

def test_homoglyph_evasion_blocks():
    # Cyrillic 'а' in place of Latin 'a'
    assert check_name("putа")["decision"] == "block"

def test_scunthorpe_passes():
    assert check_name("scunthorpe")["decision"] == "allow"

def test_assassin_passes():
    assert check_name("assassin123")["decision"] == "allow"

def test_impersonation_blocks():
    assert check_name("admin_official")["reason"] == "impersonation"

def test_clean_name_allows():
    r = check_name("shadowhunter")
    assert r["decision"] == "allow" and r["reason"] == "clean"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_names.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ngword.names'`

- [ ] **Step 6: Write `src/ngword/names.py`**

```python
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
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_names.py -v`
Expected: PASS (8 passed). If `test_assassin_passes` fails because `ass` matches inside `assassin`, confirm `assassin` is in `allowlist.txt` and the coverage check runs.

- [ ] **Step 8: Commit**

```bash
git add wordlists/ src/ngword/names.py tests/test_names.py
git commit -m "feat: name/title matcher with allowlist + impersonation"
```

---

### Task 3: Backbone smoke test (gate)

**Files:**
- Create: `src/ngword/smoke_backbone.py`
- Modify: `src/ngword/config.py` (set `BACKBONE` to the confirmed value)

**Interfaces:**
- Produces: a confirmed `config.BACKBONE` value that Tasks 5–7 depend on.

- [ ] **Step 1: Write `src/ngword/smoke_backbone.py`**

```python
"""Confirm the primary backbone fine-tunes on MPS. Flip config.BACKBONE to the fallback if it does not."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ngword import config


def try_backbone(name: str) -> bool:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)
        model.to(device)
        batch = tok(["you are trash", "have a nice day"], return_tensors="pt",
                    padding=True, truncation=True, max_length=config.MAX_LENGTH).to(device)
        labels = torch.tensor([1, 0]).to(device)
        out = model(**batch, labels=labels)
        out.loss.backward()  # exercises the backward pass on-device
        print(f"OK   {name} on {device}: loss={out.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"FAIL {name} on {device}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    if try_backbone(config.BACKBONE):
        print(f"\nUse BACKBONE = {config.BACKBONE!r}")
    elif try_backbone(config.FALLBACK_BACKBONE):
        print(f"\nPrimary failed. Set config.BACKBONE = {config.FALLBACK_BACKBONE!r}")
    else:
        print("\nBoth backbones failed — investigate the MPS/torch install before proceeding.")
```

- [ ] **Step 2: Run the smoke test**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m ngword.smoke_backbone`
Expected: one `OK ...` line. Note which backbone passed.

- [ ] **Step 3: Confirm `config.BACKBONE`**

If the primary failed and the fallback passed, edit `src/ngword/config.py` and set `BACKBONE = "distilbert-base-multilingual-cased"`. If the primary passed, leave it.

- [ ] **Step 4: Commit**

```bash
git add src/ngword/smoke_backbone.py src/ngword/config.py
git commit -m "chore: confirm training backbone on MPS"
```

---

### Task 4: Label harmonization + dedup (pure functions)

**Files:**
- Create: `src/ngword/labels.py`
- Create: `tests/test_labels.py`

**Interfaces:**
- Produces: `labels.harmonize(source: str, native_label) -> int | None` — maps a source-specific label to a 4-class index (0–3), or `None` to drop the row.
- Produces: `labels.dedup(rows: list[dict]) -> list[dict]` — removes rows with duplicate `text` (whitespace-normalized, case-folded key).

- [ ] **Step 1: Write the failing test `tests/test_labels.py`**

```python
from ngword.labels import harmonize, dedup


def test_conda_direct_mapping():
    assert harmonize("conda", "O") == 0
    assert harmonize("conda", "E") == 1
    assert harmonize("conda", "I") == 2
    assert harmonize("conda", "A") == 3

def test_binary_sources_map_toxic_to_explicit():
    assert harmonize("mginoben", 0) == 0
    assert harmonize("mginoben", 1) == 1
    assert harmonize("cold", 1) == 1
    assert harmonize("textdetox", "toxic") == 1
    assert harmonize("textdetox", "neutral") == 0

def test_toxicn_keeps_implicit():
    assert harmonize("toxicn", "implicit") == 2
    assert harmonize("toxicn", "explicit") == 1
    assert harmonize("toxicn", "non") == 0

def test_unknown_label_drops_row():
    assert harmonize("cold", 99) is None

def test_dedup_removes_case_and_space_duplicates():
    rows = [
        {"text": "GG ez", "label": 1},
        {"text": "gg  ez", "label": 1},
        {"text": "different", "label": 0},
    ]
    assert len(dedup(rows)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ngword.labels'`

- [ ] **Step 3: Write `src/ngword/labels.py`**

```python
import re

_WS = re.compile(r"\s+")

# Each source maps its native label -> 4-class index (0 clean,1 explicit,2 implicit,3 action).
_MAPS = {
    "conda":     {"O": 0, "E": 1, "I": 2, "A": 3},
    "mginoben":  {0: 0, 1: 1, "0": 0, "1": 1},
    "cold":      {0: 0, 1: 1, "0": 0, "1": 1},
    "toxicn":    {"non": 0, "explicit": 1, "implicit": 2},
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ngword/labels.py tests/test_labels.py
git commit -m "feat: label harmonization + dedup (approach A)"
```

---

### Task 5: Dataset assembly + split

**Files:**
- Create: `src/ngword/dataprep.py`
- Create: `data/SOURCES.md` (manual-download instructions)
- Create: `tests/test_dataprep.py`

**Interfaces:**
- Consumes: `labels.harmonize`, `labels.dedup`.
- Produces: `dataprep.build() -> None` writing `data/{train,val,test}.parquet` with columns `text: str, label: int, lang: str, source: str`.
- Produces: `dataprep.stratified_split(rows, seed=42) -> tuple[list, list, list]` — 80/10/10 split stratified by `(lang, label)`, no `text` shared across splits.

- [ ] **Step 1: Write `data/SOURCES.md`**

```markdown
# Data sources

Auto-downloaded by dataprep.build():
- CONDA (en): https://raw.githubusercontent.com/usydnlp/CONDA/main/data/CONDA_train.csv (cols: utterance, intentClass)
- mginoben (tl): HF `mginoben/tagalog-profanity-dataset` (cols: text, label)
- textdetox (zh, ja): HF `textdetox/multilingual_toxicity_dataset` (per-language split, cols: text, toxic)

Manual download required (place CSVs under data/raw/):
- COLD (zh): https://github.com/thu-coai/COLDataset -> data/raw/cold.csv (cols: TEXT, label)
- ToxiCN (zh): https://github.com/DUT-lujunyu/ToxiCN -> data/raw/toxicn.csv (cols: content, toxic_type in {non,explicit,implicit})
- LLM-jp v2 (ja): https://llm-jp.nii.ac.jp/... -> data/raw/llmjp.csv (cols: text, label)

Held-out adversarial eval (do NOT train on):
- PCR-ToxiCN / ToxiCloakCN (zh) -> data/adversarial/zh.csv
- hand-built leetspeak (en) -> data/adversarial/en.csv

LICENSING: mginoben = unknown, CONDA = no license file. Dev-only. Clear or replace before any commercial ship.
```

- [ ] **Step 2: Write the failing test `tests/test_dataprep.py`**

```python
from ngword.dataprep import stratified_split


def _rows():
    rows = []
    for lang in ("en", "zh", "tl", "ja"):
        for label in (0, 1):
            for i in range(50):
                rows.append({"text": f"{lang}-{label}-{i}", "label": label, "lang": lang, "source": "t"})
    return rows


def test_split_ratios_and_no_leakage():
    train, val, test = stratified_split(_rows(), seed=42)
    total = len(train) + len(val) + len(test)
    assert total == 400
    # ~80/10/10
    assert 0.75 < len(train) / total < 0.85
    # no text appears in more than one split
    texts_train = {r["text"] for r in train}
    texts_val = {r["text"] for r in val}
    texts_test = {r["text"] for r in test}
    assert not (texts_train & texts_val)
    assert not (texts_train & texts_test)
    assert not (texts_val & texts_test)

def test_split_is_deterministic():
    a = stratified_split(_rows(), seed=42)[0]
    b = stratified_split(_rows(), seed=42)[0]
    assert [r["text"] for r in a] == [r["text"] for r in b]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_dataprep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ngword.dataprep'`

- [ ] **Step 4: Write `src/ngword/dataprep.py`**

```python
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ngword.labels import dedup, harmonize

_DATA = Path(__file__).resolve().parents[2] / "data"
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
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
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
    ds = load_dataset("mginoben/tagalog-profanity-dataset", split="train")
    out = []
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
    rows += _load_raw_csv("toxicn.csv", "toxicn", "zh", "content", "toxic_type")
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_dataprep.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Build the real dataset** (downloads + any manual CSVs in place)

Run: `python -m ngword.dataprep`
Expected: three lines `train/val/test: N rows`. Sources not yet downloaded print `SKIP` — fill them per `data/SOURCES.md` and re-run. ja rows are lower-confidence (documented in spec §7).

- [ ] **Step 7: Commit**

```bash
git add src/ngword/dataprep.py data/SOURCES.md tests/test_dataprep.py
git commit -m "feat: dataset assembly, segmentation, balancing, stratified split"
```

---

### Task 6: Fine-tune + per-language evaluation

**Files:**
- Create: `src/ngword/train.py`
- Create: `src/ngword/evaluate_model.py`

**Interfaces:**
- Consumes: `config.BACKBONE`, `config.MAX_LENGTH`, `config.CLASS_NAMES`, `data/{train,val,test}.parquet`.
- Produces: a saved model directory `final_model/` (torch + tokenizer).
- Produces: `evaluate_model.report(model_dir: str) -> dict` — per-language macro-F1 and per-class F1, printed and returned.

- [ ] **Step 1: Write `src/ngword/train.py`**

```python
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from ngword import config

_DATA = Path(__file__).resolve().parents[2] / "data"


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self._cw.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def _load(split):
    import pandas as pd
    return Dataset.from_pandas(pd.read_parquet(_DATA / f"{split}.parquet"), preserve_index=False)


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(config.BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(config.BACKBONE, num_labels=4).to(device)

    train_ds, val_ds = _load("train"), _load("val")
    weights = compute_class_weight("balanced", classes=np.arange(4),
                                   y=np.array(train_ds["label"]))
    class_weights = torch.tensor(weights, dtype=torch.float)

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=config.MAX_LENGTH)

    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)

    args = TrainingArguments(
        output_dir="checkpoints",
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=3, weight_decay=0.01, bf16=True,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=50, report_to="none",
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics, class_weights=class_weights,
    )
    trainer.train()
    model.save_pretrained("final_model")
    tok.save_pretrained("final_model")
    print("saved final_model/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `src/ngword/evaluate_model.py`**

```python
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ngword import config

_DATA = Path(__file__).resolve().parents[2] / "data"


def report(model_dir: str = "final_model") -> dict:
    df = pd.read_parquet(_DATA / "test.parquet")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            batch = tok(list(df["text"][i:i + 64]), return_tensors="pt",
                        padding=True, truncation=True, max_length=config.MAX_LENGTH)
            preds += model(**batch).logits.argmax(-1).tolist()
    df = df.assign(pred=preds)

    out = {}
    print("\n=== Per-language results ===")
    for lang in sorted(df["lang"].unique()):
        sub = df[df["lang"] == lang]
        macro = f1_score(sub["label"], sub["pred"], average="macro")
        out[lang] = macro
        flag = "  [LOWER-CONFIDENCE]" if lang == "ja" else ""
        print(f"\n{lang}: macro-F1 = {macro:.3f}{flag}")
        print(classification_report(sub["label"], sub["pred"],
                                    labels=list(range(4)), target_names=config.CLASS_NAMES,
                                    zero_division=0))
    return out


if __name__ == "__main__":
    report()
```

- [ ] **Step 3: Train**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m ngword.train`
Expected: training runs (overnight-scale on M2), then `saved final_model/`.

- [ ] **Step 4: Evaluate**

Run: `python -m ngword.evaluate_model`
Expected: per-language macro-F1 + per-class breakdown. `ja` flagged lower-confidence. Confirm `action`/`implicit` are weak/absent outside en/zh (expected per Approach A).

- [ ] **Step 5: Commit**

```bash
git add src/ngword/train.py src/ngword/evaluate_model.py
git commit -m "feat: class-weighted fine-tune + per-language eval"
```

---

### Task 7: Export to int8 ONNX

**Files:**
- Create: `src/ngword/export.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: `final_model/`.
- Produces: `onnx_model/` containing a quantized int8 ONNX model + tokenizer, loadable via `optimum.onnxruntime.ORTModelForSequenceClassification`.

- [ ] **Step 1: Write `src/ngword/export.py`**

```python
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def export(src: str = "final_model", dst: str = "onnx_model") -> None:
    model = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    tok = AutoTokenizer.from_pretrained(src)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)

    quantizer = ORTQuantizer.from_pretrained(dst)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=dst, quantization_config=qconfig)
    print(f"exported int8 ONNX to {dst}/")


if __name__ == "__main__":
    export()
```

- [ ] **Step 2: Write the failing test `tests/test_export.py`**

```python
import pytest


def test_quantized_model_loads_and_predicts():
    pytest.importorskip("optimum")
    import os
    if not os.path.isdir("onnx_model"):
        pytest.skip("run `python -m ngword.export` first")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    m = ORTModelForSequenceClassification.from_pretrained("onnx_model", file_name="model_quantized.onnx")
    tok = AutoTokenizer.from_pretrained("onnx_model")
    inp = tok(["you are trash"], return_tensors="pt")
    logits = m(**inp).logits
    assert logits.shape[-1] == 4
```

- [ ] **Step 3: Export**

Run: `python -m ngword.export`
Expected: `exported int8 ONNX to onnx_model/` (contains `model_quantized.onnx`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_export.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ngword/export.py tests/test_export.py
git commit -m "feat: int8 ONNX export"
```

---

### Task 8: Unified inference entry point

**Files:**
- Create: `src/ngword/infer.py`
- Create: `tests/test_infer.py`

**Interfaces:**
- Consumes: `names.check_name`, `config.CLASS_NAMES`, `config.MAX_LENGTH`, `onnx_model/` (with `final_model/` torch fallback).
- Produces: `infer.classify(text: str, kind: str) -> dict` returning `{"kind", "decision", "label", "score", "reason"}`. For `name`/`title`: matcher result (`score=None`). For `chat`: model result (`label` ∈ CLASS_NAMES, `decision="block"` unless `label=="clean"`).

- [ ] **Step 1: Write the failing test `tests/test_infer.py`**

```python
import pytest

from ngword.infer import classify


def test_name_routes_to_matcher():
    r = classify("putangina", "name")
    assert r["kind"] == "name" and r["decision"] == "block"

def test_title_uses_matcher_path():
    r = classify("scunthorpe united", "title")
    assert r["decision"] == "allow"

def test_empty_input_does_not_crash():
    r = classify("", "chat")
    assert r["decision"] in ("allow", "block")

def test_chat_returns_four_class_label():
    pytest.importorskip("optimum")
    import os
    if not (os.path.isdir("onnx_model") or os.path.isdir("final_model")):
        pytest.skip("train + export first")
    r = classify("you are absolute trash, kill yourself", "chat")
    assert r["label"] in ("clean", "explicit", "implicit", "action")
    assert 0.0 <= r["score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ngword.infer'`

- [ ] **Step 3: Write `src/ngword/infer.py`**

```python
import os
from functools import lru_cache

from ngword import config
from ngword.names import check_name

_MATCHER_KINDS = {"name", "title"}


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoTokenizer
    if os.path.isdir("onnx_model"):
        from optimum.onnxruntime import ORTModelForSequenceClassification
        m = ORTModelForSequenceClassification.from_pretrained(
            "onnx_model", file_name="model_quantized.onnx")
        tok = AutoTokenizer.from_pretrained("onnx_model")
    else:  # torch fallback
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained("final_model").eval()
        tok = AutoTokenizer.from_pretrained("final_model")
    return m, tok


def classify(text: str, kind: str) -> dict:
    if not isinstance(text, str):
        text = str(text or "")

    if kind in _MATCHER_KINDS:
        r = check_name(text)
        return {"kind": kind, "decision": r["decision"], "label": None,
                "score": None, "reason": r["reason"]}

    # chat -> model
    if not text.strip():
        return {"kind": "chat", "decision": "allow", "label": "clean",
                "score": 1.0, "reason": "empty"}

    import torch
    m, tok = _model()
    inp = tok([text], return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)
    with torch.no_grad():
        probs = m(**inp).logits.softmax(-1)[0]
    idx = int(probs.argmax())
    label = config.CLASS_NAMES[idx]
    return {"kind": "chat", "decision": "allow" if label == "clean" else "block",
            "label": label, "score": float(probs[idx]), "reason": label}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_infer.py -v`
Expected: PASS (matcher tests pass; chat test passes if model exists, else skips).

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all non-skipped tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ngword/infer.py tests/test_infer.py
git commit -m "feat: unified classify(text, kind) inference"
```

---

## Self-Review Notes

- **Spec coverage:** matcher (§3) → Tasks 1–2; backbone gate (§4.1) → Task 3; harmonization approach A (§5) → Task 4; data pipeline incl. ja segmentation + balancing + stratified split (§6) → Task 5; class-weighted training + per-language metrics (§4.2, §8, §11) → Task 6; int8 ONNX export (§4.4) → Task 7; `classify(text, kind)` routing + error handling (§2, §9) → Task 8; per-language + ja-lower-confidence reporting (§7, §11) → Task 6 `evaluate_model`. Adversarial eval sets (§6) are documented in `data/SOURCES.md` and consumed by `evaluate_model.report` when present in `data/adversarial/` — extend the report loop to score them once collected.
- **Deferred/out of scope (spec §12):** Approach B masked loss, deployment/serving/caching, in-domain ja labeling — correctly absent.
- **Type consistency:** `check_name -> {"decision","reason"}` used identically in Tasks 2 and 8; `classify` return schema fixed in Task 8 interface; class index order `0=clean,1=explicit,2=implicit,3=action` consistent across `config`, `labels`, `train`, `evaluate_model`, `infer`.
- **Known ceilings (ponytail):** allowlist coverage is substring-based (Task 2); repeat-collapse fires at ≥3 (Task 1); balancing is a hard per-language cap (Task 5). Each is a documented v1 simplification with an upgrade path.
```
