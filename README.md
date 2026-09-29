# slurshield

A multilingual profanity & toxicity filter for game backends — **usernames, titles, and chat** across **English, Tagalog, Chinese, and Japanese**. Ships a trained model in the repo and serves it from a ~400 MB CPU-only container.

Two independent systems behind one call:

| Input | Handler | How |
|-------|---------|-----|
| **name / title** | deterministic matcher | Unicode-fold → leetspeak → Aho-Corasick + allowlist + impersonation. No ML, microseconds. |
| **chat** | fine-tuned transformer | mmBERT-small → int8 ONNX, 4-class intent, with a deobfuscation pre-filter in front. |

Inference depends on **onnxruntime + tokenizers + numpy** only — no PyTorch, no `transformers` at serve time.

---

## Quick start

```bash
git clone <repo> && cd slurshield
git lfs pull            # fetch the 135 MB int8 model (requires git-lfs)

# --- run the server (Docker) ---
docker build -t slurshield .
docker run -p 8000:8000 slurshield

curl -s -X POST localhost:8000/classify -d '{"text":"you are sh1t","kind":"chat"}'
# {"kind":"chat","decision":"block","label":"explicit","score":1.0,"reason":"prefilter:shit",
#  "masked":"you are ****"}
```

Or as a library:

```bash
pip install .            # runtime deps only
```
```python
from slurshield.infer import classify, redact

classify("gg wp everyone", "chat")     # {'decision':'allow','label':'clean', ...}
classify("you are tr@sh", "chat")      # {'decision':'block','label':'explicit', ...}
classify("xX_admin_Xx", "name")        # {'decision':'block','reason':'impersonation', ...}
classify("Scunthorpe", "name")         # {'decision':'allow','reason':'clean', ...}

redact("you are sh1t")                 # 'you are ****'
redact("p.u.t.a.n.g.i.n.a stop")       # '***************** stop'
redact("you are absolute trash")       # unchanged — model-flagged, no word to mask
```

---

## API

**`classify(text: str, kind: str) -> dict`** — `kind` ∈ `{"name","title","chat"}`

```jsonc
{ "kind": "chat", "decision": "block", "label": "explicit", "score": 0.94, "reason": "explicit" }
// name/title: label & score are null; reason is the matched term / "impersonation" / "clean"
// chat labels: clean | explicit | implicit | action   (decision = block only for explicit | implicit)
```

**`redact(text: str, mask: str = "*") -> str`** — mask profanity, preserving length.

Word-list hits only. The chat model returns a sentence label with no spans, so text it
flags with no lexical hit (`"you are trash"`) is returned unchanged — `classify()` still
blocks it. Span-level redaction of contextual toxicity needs a token-classification head,
i.e. a retrain. Separator evasion (`p.u.t.a`) is masked for slurs of 5+ chars only, the
same length tier the matcher uses.

**HTTP** (`slurshield.serve`, stdlib only):

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/classify` | `{"text": "...", "kind": "chat"}` | the verdict JSON above, plus `"masked"` |
| `GET`  | `/health` | — | `{"status":"ok"}` |

---

## How well it works

Per-language on the held-out test set (macro-F1 over the classes each language actually has):

| Lang | clean | explicit | implicit | action | Notes |
|------|:-----:|:--------:|:--------:|:------:|-------|
| **en** | 0.93 | 0.80 | 0.64 | 0.83 | all four classes |
| **tl** | 0.89 | 0.86 | — | — | clean/explicit only (by data) |
| **ja** | 0.79 | 0.57 | — | — | weakest — lower-confidence data |
| **zh** | 0.85 | 0.82 | 0.04 | 0.00 | strong binary; fine-grained classes starved |

**Block/allow — the real product decision — is strong in en/tl/zh, usable in ja.** The fine-grained `implicit`/`action` labels are English-reliable, weak elsewhere (a documented consequence of the data; see [Limitations](#limitations)).

**Obfuscation robustness** (block/allow on a leetspeak/unicode evasion set), model alone vs. with the pre-filter:

| Evasion type | model only | + pre-filter |
|--------------|:----------:|:------------:|
| obfuscated **profanity** (`sh1t`, `b1tch`, `a$$hole`, `ＦＵＣＫ`) | 2/10 | **9/10** |
| obfuscated **contextual** toxicity (`tr4sh`, `kys`) | 0/6 | 0/6 |
| clean incl. Scunthorpe traps (`class`, `assemble`, `pass`) | 0 false blocks | **0 false blocks** |

The pre-filter recovers obfuscated *profanity* for free. Obfuscated *contextual* toxicity remains open (see roadmap).

---

## Architecture

```
classify(text, kind)
   ├─ name / title ─► fold() → Aho-Corasick (length-tiered) + allowlist + impersonation
   └─ chat ─────────► profanity pre-filter (fold + wordlist) ──hit──► block
                                    │ miss
                                    ▼
                      tokenizers → int8 ONNX (mmBERT-small) → 4-class softmax
```

- **Matcher** never touches the model; the **model** always sees raw text (train/serve consistency). Normalization lives only on the matcher/pre-filter path.
- **Length-tiered matching**: short slurs match as whole tokens, long slurs as substrings — so `Phoenix`/`class`/`Model` don't false-block while evasions still get caught.

---

## Project layout

```
src/slurshield/          runtime (deploys): config, normalize, names, infer, serve
  model/             int8 ONNX + tokenizer            (Git LFS)
wordlists/           per-language slur lists + allowlist + impersonation
training/            data pipeline + train + export   (not shipped; tracked)
tests/               35 tests
data/                datasets, parquet, adversarial   (gitignored — local only, not distributed)
Dockerfile           ~400 MB CPU runtime image
```

---

## Reproducing the model

```bash
pip install ".[training]"                      # torch, transformers, datasets, optimum, ...

# COLD + ToxiCN need manual download from the upstream repos linked below (rest auto-fetches)
python -m training.smoke_backbone              # confirm backbone runs on this machine
python -m training.dataprep                    # build train/val/test.parquet
python -m training.train                       # fine-tune (overnight-scale on Apple M2 MPS)
python -m training.evaluate_model              # per-language + adversarial report
python -m training.export                      # -> src/slurshield/model/ (int8 ONNX)
```

Backbone: [`jhu-clsp/mmbert-small`](https://huggingface.co/jhu-clsp/mmbert-small) (140M, MIT). Fine-tuned with class-weighted loss, `max_length=48`, 4-class head. Trains in a couple hours on an M2; overfits after ~1 epoch (checkpoint selection keeps the best).

---

## Limitations

- **Japanese is the weakest language** — its toxic training data is segmented long-form web text, so labels are noisier. Real fix: label in-domain short messages (a hook is left in the pipeline).
- **`implicit` / `action` are English-reliable only.** zh has too few examples (`implicit` F1 0.04); tl/ja have none. Treat those languages as clean-vs-explicit.
- **Obfuscated *contextual* toxicity evades** (`tr4sh`, `k y s`) — it isn't lexical, so no wordlist catches it, and the raw-text model can't read the obfuscation. Needs training-time augmentation.
- **Vowel-swapped profanity** (`f@ck` → folds to `fack`) can slip; most obfuscation preserves vowels and is caught.

## Roadmap

- Training-time obfuscation augmentation to close the contextual-evasion gap.
- Chinese `implicit`/`action` data to revive the fine-grained classes.
- Adversarial `zh` set (PCR-ToxiCN / ToxiCloakCN) — loader is wired, data pending.

---

## Data & licensing

Trained on [CONDA](https://github.com/usydnlp/CONDA) (en), [mginoben](https://huggingface.co/datasets/mginoben/tagalog-profanity-dataset) (tl), [COLD](https://github.com/thu-coai/COLDataset) + [ToxiCN](https://github.com/DUT-lujunyu/ToxiCN) + [textdetox](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) (zh), textdetox + [LLM-jp Toxicity](https://huggingface.co/datasets/p1atdev/LLM-jp-Toxicity-Dataset) (ja).
The datasets and the exact prep recipes are not redistributed here — they carry their own
academic licenses; fetch them from the links above.

> **Note:** some training datasets (`mginoben`, CONDA) have unclear/absent licenses. Fine for research/dev; **clear or replace them before any commercial deployment.**
