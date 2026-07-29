# Data sources

Auto-downloaded by dataprep.build():
- CONDA (en): https://raw.githubusercontent.com/usydnlp/CONDA/main/data/CONDA_train.csv (cols: utterance, intentClass)
- mginoben (tl): HF `mginoben/tagalog-profanity-dataset` (cols: text, label)
- textdetox (zh, ja): HF `textdetox/multilingual_toxicity_dataset` (per-language split, cols: text, toxic)

Manual download required (place CSVs under data/raw/). Exact prep recipe used (reproducible):

- COLD (zh) -> data/raw/cold.csv (cols: TEXT, label). Source repo: https://github.com/thu-coai/COLDataset
  Recipe: concat COLDataset/{train,dev,test}.csv, keep columns [TEXT, label]. ~37,480 rows.

- ToxiCN (zh) -> data/raw/toxicn.csv. Source: https://raw.githubusercontent.com/DUT-lujunyu/ToxiCN/main/ToxiCN_1.0.csv
  Just save the raw file (keep columns `content, toxic, expression`). `dataprep._load_toxicn` applies the
  toxic-first collapse IN CODE: clean only when toxic==0; the 816 toxic-but-untyped (expression=0) rows fold
  to explicit; expression 1/2/3 -> explicit/implicit/action. A pre-collapsed 2-col file (content, expression)
  also loads correctly. ~12,011 rows -> clean 5550 / explicit 3553 / implicit 1995 / action 913.

- LLM-jp v2 (ja) -> data/raw/llmjp.csv (cols: text, label as {non|toxic}). Source: HF `p1atdev/LLM-jp-Toxicity-Dataset`.
  Recipe: label = "non" if raw label=="nontoxic" else "toxic" (folds "toxic" + "has_toxic_expression" -> toxic).
  1,847 long docs; dataprep segments them into short chunks (segment=True). WARNING: chunks inherit the
  document-level label, so ja is lower-confidence (many "explicit" chunks are neutral sentences). See spec §7.

Held-out adversarial eval (do NOT train on):
- PCR-ToxiCN / ToxiCloakCN (zh) -> data/adversarial/zh.csv
- hand-built leetspeak (en) -> data/adversarial/en.csv

LICENSING: mginoben = unknown, CONDA = no license file. Dev-only. Clear or replace before any commercial ship.
