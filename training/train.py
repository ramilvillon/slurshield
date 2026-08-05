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

from slurshield import config

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
    weights = compute_class_weight("balanced", classes=np.arange(len(config.CLASS_NAMES)),
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
