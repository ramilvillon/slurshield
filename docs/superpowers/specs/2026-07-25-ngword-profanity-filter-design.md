# ngword — Profanity & Toxicity Filter (Design Spec)

**Date:** 2026-07-25
**Scope:** Train a chat-toxicity model and build a name/title matcher, plus a single inference entry point. **Deployment, serving, caching, and containers are explicitly out of scope.**

---

## 1. Goal

Filter three kinds of game user-generated text — **names**, **titles**, and **chat messages** — across four languages: English (en), Tagalog (tl), Chinese (zh), Japanese (ja).

Two independent components behind one inference function:

- **Name/title matcher** — deterministic string matching, no ML.
- **Chat model** — fine-tuned multilingual transformer, 4-class intent output.

Deliverables: a training pipeline that produces a model artifact, and an inference module exposing `classify(text, kind)`.

---

## 2. Architecture

```
classify(text, kind)                       kind in {name, title, chat}
   |
   +-- name / title --> MATCHER (no ML) --> {block|allow} + matched term / reason
   |                    normalize -> Aho-Corasick -> allowlist -> impersonation
   |
   +-- chat ----------> MODEL (transformer) --> {clean|explicit|implicit|action} + score
                        raw text -> tokenize -> 4-class head
```

Rules:

- The matcher and model share **nothing at runtime** except the `classify` entry point.
- **Normalization is applied only on the matcher path.** The chat model always receives **raw** text, so training and inference input distributions match.
- `title` is treated identically to `name` (same matcher path). If titles later need contextual scoring, they can be routed to the model; not in v1.

---

## 3. Component: Name/Title Matcher (no training)

Pipeline, in order:

1. **Unicode confusables fold** — UTS #39 skeleton algorithm (via `confusable_homoglyphs` or an equivalent skeleton map). Folds Cyrillic/fullwidth/math-styled homoglyphs to canonical ASCII/CJK. Mandatory for zh/ja where fullwidth input is everyday.
2. **Strip separators & collapse repeats** — `p_u_t_a` -> `puta`, `fuuuck` -> `fuck`.
3. **Leetspeak fold** — digit/symbol -> letter map (`0->o`, `@->a`, `3->e`, `$->s`, etc.).
4. **Substring match** — Aho-Corasick (`pyahocorasick`) over per-language wordlists. **Substring, not word-boundary** (names have no whitespace, so `\b` is invalid and the Scunthorpe problem is at maximum severity).
5. **Allowlist** — cross-reference matches against a benign list (`scunthorpe`, `assassin`, `bass`, `analysis`, plus in-game hero/item/place names) to suppress false positives.
6. **Impersonation rules** — block reserved patterns: `admin`, `gm_`, `[staff]`, `support`, `mod`, `moderator`. Independent of profanity; required for a game backend.

**Output:** `{decision: block|allow, reason: <matched-term|impersonation|clean>}`.

**Wordlists:**

- en / zh / ja: seeded from `censor-text/profanity-list` (Unlicense / public domain).
- tl: **hand-authored** (the public Tagalog list is 9 words). Seed from tl profanity observed in the training datasets.

**Policy:** names are permanent and publicly visible -> bias toward **block**, always return a `reason` to support an appeal flow (appeal flow itself is out of scope).

Size: ~80-120 lines. No model, no GPU, microsecond latency.

---

## 4. Component: Chat Model

### 4.1 Backbone

- **Primary:** `jhu-clsp/mmbert-small` (140M params, 22x384, 256k Gemma-2 tokenizer — strong CJK tokenization, MIT license).
- **Fallback:** `distilbert-base-multilingual-cased` (134M, 6x768).
- **Day-0 gate:** a smoke test MUST confirm the primary backbone fine-tunes on the Apple Silicon MPS backend. FlashAttention2 is CUDA-only and ModernBERT-family MPS support has been historically rough. If it fails on MPS after trying `attn_implementation="sdpa"` / `"eager"`, switch to the fallback and proceed. This is the **first** implementation step; everything downstream assumes a working backbone.

### 4.2 Training config

- `max_length = 48`, **dynamic padding** (median chat message is 11 characters; 128 would be ~10x oversized).
- **Class-weighted cross-entropy** (corpus is ~19% toxic; classes are imbalanced).
- `bf16 = True` (host is macOS 26 / Darwin 25, which supports it), HuggingFace `Trainer` on `torch.device("mps")`.
- Learning rate 2e-5, weight decay 0.01, ~3 epochs (tune on validation).
- Training runs on the M2 (16GB). Expected several hours; overnight-scale, not minutes.

### 4.3 Output

4-class single-label head: `clean | explicit | implicit | action` (CONDA intent scheme). Inference returns the argmax class + softmax confidence.

### 4.4 Export

torch checkpoint -> ONNX (`optimum`) -> **int8 dynamic quantization**. Inference loads the int8 ONNX on **CPU** (portable, runs on the Mac and anywhere else). torch checkpoint retained as fallback.

---

## 5. Label Harmonization (Approach A: best-effort mapping)

The model always predicts 4 classes. Source datasets carry different label schemas; map each to the 4-class scheme as follows:

| Dataset | Lang | Native labels | Mapping to {clean, explicit, implicit, action} |
|---|---|---|---|
| CONDA | en | O / E / I / A | **direct** — clean / explicit / implicit / action |
| mginoben | tl | binary directed-abuse (0/1) | 0 -> clean, 1 -> explicit |
| COLD | zh | binary offensive (0/1) | 0 -> clean, 1 -> explicit |
| ToxiCN | zh | explicit / implicit / non-toxic | non -> clean, explicit -> explicit, implicit -> implicit |
| textdetox (zh, ja) | zh, ja | toxic / neutral | neutral -> clean, toxic -> explicit |
| LLM-jp v2 | ja | toxicity + category | non -> clean, toxic -> explicit (long text -> segment/truncate, see 6) |
| inspection-ai | ja | toxicity levels | **eval-only** (436 rows), not used for training |

**Documented consequences of Approach A (accepted for v1):**

- `implicit` has real training signal only for **en** (CONDA) and **zh** (ToxiCN).
- `action` is **English-only** (only CONDA carries it).
- For tl and ja, the model learns **clean vs explicit** well; `implicit`/`action` are weak-to-absent.
- These per-language class-support gaps MUST be stated in the eval report so downstream policy does not over-trust `implicit`/`action` for tl/ja.

Approach B (masked loss so binary rows only supervise clean-vs-toxic) is explicitly **deferred to v2**.

---

## 6. Data Pipeline

Steps:

1. **Download / load** each dataset. Use `mginoben/tagalog-profanity-dataset` **directly** — do NOT load `SEACrowd/tgl_profanity` or `kornwtp/profanity-fil-dataset` (both are byte-identical copies of mginoben; loading multiple causes duplicate/leak). SEACrowd script-based loaders are skipped (disabled-by-default remote code).
2. **Dedupe** — exact and near-duplicate removal within and across sources.
3. **Harmonize labels** per Section 5.
4. **Japanese long-text handling** — LLM-jp v2 averages 2,567 chars (Common Crawl documents), but the target is short chat. Segment long documents into sentence-ish units and/or truncate to the model window; keep only segments that plausibly resemble chat length. Document how many rows survive.
5. **Balance** — downsample the dominant `clean` class and English dominance so no single language/class overwhelms training. Target a roughly comparable per-language contribution rather than raw counts.
6. **Stratified split** — train/val/test stratified by (language x class). No row appears in more than one split.

**Adversarial eval sets** (held out, never trained on):

- zh: PCR-ToxiCN / ToxiCloakCN (phonetic-cloaking / homophone / pinyin obfuscation).
- en: a small hand-built leetspeak/unicode-obfuscation set (~100 cases).

**Licensing note (conservative default):** the commercial-vs-research decision is deferred, so treat as commercial. `mginoben` (license: unknown) and CONDA (no license file) are **quality-blocking for a commercial ship** and must be cleared or replaced with permissively-licensed equivalents before production. They are used for development; this risk is tracked, not resolved here.

---

## 7. Japanese Confidence Caveat

v1 Japanese trains on ~9k public rows (textdetox ja 5k + LLM-jp v2 3.8k, post-segmentation). It is evaluated separately on the 436 well-annotated `inspection-ai` rows. The eval report MUST flag ja as **lower-confidence**. A labeling hook is left in the pipeline so ~3-5k in-domain labeled messages can be added later **without pipeline changes**. Not blocking for v1.

---

## 8. Project Layout

```
ngword/
  data/                     # downloaded + processed (gitignored)
  wordlists/
    en.txt zh.txt ja.txt tl.txt
    allowlist.txt
    impersonation.txt
  src/ngword/
    normalize.py            # unicode fold + leet fold + separator/repeat collapse (matcher path only)
    names.py                # name/title matcher: Aho-Corasick + allowlist + impersonation
    dataprep.py             # download, dedupe, harmonize, segment, balance, split
    train.py                # backbone smoke test + fine-tune
    export.py               # ONNX + int8 quant
    infer.py                # classify(text, kind): routes name/title -> matcher, chat -> model
  tests/
  pyproject.toml
```

Keep the file count minimal. `normalize.py` is shared by `names.py` and the wordlist path **only** — it is never applied to chat model input.

---

## 9. Error Handling

- **Inference:** handle empty string, non-string input, over-length input (truncate), and unknown/mixed scripts without crashing. If the int8 ONNX artifact is missing, fall back to the torch checkpoint.
- **Data pipeline:** tolerate individual dataset download failures (log + continue with what loaded); skip script-based SEACrowd loaders; assert post-conditions (no cross-split duplicates, valid label mapping).

---

## 10. Testing (one runnable check per non-trivial unit)

- `tests/test_names.py` — Scunthorpe/`assassin` **pass**; `p_u_t_a`, homoglyph-obfuscated, and `admin`/impersonation **block**; leetspeak folds correctly.
- `tests/test_dataprep.py` — no duplicate rows across splits; label mapping matches Section 5; no train/test leakage.
- `tests/test_train.py` — tiny-subset training smoke test completes and saves an artifact.
- `tests/test_infer.py` — `classify` returns the 4-class shape for chat and block/allow for name; known toxic/clean cases classify correctly.
- **Per-language eval report** — macro-F1 per language AND per-class F1; adversarial-set scores; ja flagged lower-confidence.

No frameworks beyond a plain test runner. Assert-based checks.

---

## 11. Metrics

- **Per-language macro-F1** and **per-class F1**. Never report a single aggregate score (it hides a broken per-language head).
- Decision threshold tuned **per language** on validation.
- Track adversarial-set performance (Section 6) as a separate, first-class metric.

---

## 12. Out of Scope (v1)

- Deployment, serving, containers, sidecar/microservice topology, caching (Redis/LRU).
- Approach B masked-loss harmonization.
- In-domain Japanese labeling (hook left, work deferred).
- Contextual (multi-message history) toxicity — CONDA supports it, v1 is single-utterance.
- Appeal-flow / reserved-name registry beyond the impersonation wordlist.

---

## 13. First Implementation Step (gate)

Backbone smoke test: confirm `mmbert-small` fine-tunes on MPS. If not, fall back to `distilbert-base-multilingual-cased`. Do this before building the data pipeline — it decides the backbone the rest of the training code targets.
