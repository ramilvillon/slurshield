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
