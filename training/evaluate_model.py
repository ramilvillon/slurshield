from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slurshield import config

_DATA = Path(__file__).resolve().parents[1] / "data"


def _load(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    return tok, model


def _predict(texts, tok, model) -> list[int]:
    texts = list(texts)
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = tok(texts[i:i + 64], return_tensors="pt",
                        padding=True, truncation=True, max_length=config.MAX_LENGTH)
            preds += model(**batch).logits.argmax(-1).tolist()
    return preds


def report(model_dir: str = "final_model") -> dict:
    df = pd.read_parquet(_DATA / "test.parquet")
    tok, model = _load(model_dir)
    df = df.assign(pred=_predict(df["text"], tok, model))

    out = {}
    print("\n=== Per-language results ===")
    for lang in sorted(df["lang"].unique()):
        sub = df[df["lang"] == lang]
        macro = f1_score(sub["label"], sub["pred"], average="macro")
        class_report = classification_report(
            sub["label"], sub["pred"],
            labels=list(range(len(config.CLASS_NAMES))), target_names=config.CLASS_NAMES,
            zero_division=0, output_dict=True)
        per_class_f1 = {name: class_report[name]["f1-score"] for name in config.CLASS_NAMES}
        out[lang] = {"macro_f1": macro, "per_class_f1": per_class_f1}
        flag = "  [LOWER-CONFIDENCE]" if lang == "ja" else ""
        print(f"\n{lang}: macro-F1 = {macro:.3f}{flag}")
        print(classification_report(sub["label"], sub["pred"],
                                    labels=list(range(len(config.CLASS_NAMES))), target_names=config.CLASS_NAMES,
                                    zero_division=0))
    return out


def adversarial_report(model_dir: str = "final_model") -> dict:
    # Robustness to obfuscation, scored at the block/allow level (clean vs not-clean).
    # The chat model sees RAW text, so leetspeak/spacing/unicode evasion is a real risk.
    adv_dir = _DATA / "adversarial"
    files = sorted(adv_dir.glob("*.csv")) if adv_dir.exists() else []
    if not files:
        print("\nNo adversarial sets in data/adversarial/ — skipping.")
        return {}

    from slurshield.infer import classify  # measure the SHIPPING pipeline (pre-filter + model)
    out = {}
    print("\n=== Adversarial robustness (block = predicted not-clean) ===")
    for f in files:
        df = pd.read_csv(f)
        pred_toxic = [classify(t, "chat")["decision"] == "block" for t in df["text"]]
        true_toxic = [lab != 0 for lab in df["label"]]

        toxic = sum(true_toxic)
        clean = len(df) - toxic
        evaded = sum(1 for t, p in zip(true_toxic, pred_toxic) if t and not p)
        false_block = sum(1 for t, p in zip(true_toxic, pred_toxic) if not t and p)
        recall = (toxic - evaded) / toxic if toxic else float("nan")

        out[f.stem] = {"n": len(df), "toxic_recall": recall,
                       "evaded": evaded, "false_blocks": false_block}
        print(f"\n{f.stem}: {len(df)} rows ({toxic} toxic / {clean} clean)")
        print(f"  toxic caught: {toxic - evaded}/{toxic}  (recall {recall:.2f}) — {evaded} evaded detection")
        print(f"  clean kept:   {clean - false_block}/{clean}  ({false_block} false blocks)")
    return out


if __name__ == "__main__":
    report()
    adversarial_report()
