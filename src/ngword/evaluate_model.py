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
