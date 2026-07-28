# Data sources

Auto-downloaded by dataprep.build():
- CONDA (en): https://raw.githubusercontent.com/usydnlp/CONDA/main/data/CONDA_train.csv (cols: utterance, intentClass)
- mginoben (tl): HF `mginoben/tagalog-profanity-dataset` (cols: text, label)
- textdetox (zh, ja): HF `textdetox/multilingual_toxicity_dataset` (per-language split, cols: text, toxic)

Manual download required (place CSVs under data/raw/). Exact prep recipe used (reproducible):

- COLD (zh) -> data/raw/cold.csv (cols: TEXT, label). Source repo: https://github.com/thu-coai/COLDataset
  Recipe: concat COLDataset/{train,dev,test}.csv, keep columns [TEXT, label]. ~37,480 rows.

- ToxiCN (zh) -> data/raw/toxicn.csv (cols: content, expression as int 0-3 = clean/explicit/implicit/action).
  Source: https://raw.githubusercontent.com/DUT-lujunyu/ToxiCN/main/ToxiCN_1.0.csv
  NOTE: the raw `expression` column is NOT directly usable — 816 genuinely toxic rows carry expression=0.
  Recipe (toxic-first collapse): class = 0 if toxic==0 else (expression if expression in {1,2,3} else 1).
  This keeps clean only when toxic==0 and folds untyped-toxic rows into explicit. ~12,011 rows.
  Resulting dist: clean 5550 / explicit 3553 / implicit 1995 / action 913.

- LLM-jp v2 (ja) -> data/raw/llmjp.csv (cols: text, label as {non|toxic}). Source: HF `p1atdev/LLM-jp-Toxicity-Dataset`.
  Recipe: label = "non" if raw label=="nontoxic" else "toxic" (folds "toxic" + "has_toxic_expression" -> toxic).
  1,847 long docs; dataprep segments them into short chunks (segment=True). WARNING: chunks inherit the
  document-level label, so ja is lower-confidence (many "explicit" chunks are neutral sentences). See spec §7.

Held-out adversarial eval (do NOT train on):
- PCR-ToxiCN / ToxiCloakCN (zh) -> data/adversarial/zh.csv
- hand-built leetspeak (en) -> data/adversarial/en.csv

LICENSING: mginoben = unknown, CONDA = no license file. Dev-only. Clear or replace before any commercial ship.
