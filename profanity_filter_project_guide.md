# Profanity & Toxicity Filter Model: Project Guide & Architecture Brief

## 1. Executive Summary & Project Strategy

Building a robust bad/curse word filter requires balancing low latency, context awareness, and resistance to user evasion (leetspeak, zero-width spaces, character substitution).

### Architecture Recommendation: Hybrid Pipeline

```
[ Incoming Text ]
       │
       ▼
[ 1. Preprocessor / Leetspeak Normalizer ]
       │
       ▼
[ 2. Fast Regex / Wordlist Filter ]  ──► (Match found) ──► [ Block / Mask Fast ]
       │ (No match)
       ▼
[ 3. Fine-tuned DistilBERT (MPS) ]  ──► [ Contextual Toxicity Score ] ──► [ Clean / Ban ]
```

1. **Rule-Based Pre-filter (Regex / Lookup):** Catches explicit, obvious profanity and direct matches in sub-millisecond time.
2. **ML Classifier (DistilBERT):** Evaluates context, masked/obfuscated toxicity, threats, and implicit harassment.

---

## 2. Hardware Compatibility: Mac M2 (16GB RAM)

Your Apple Silicon M2 Mac with 16GB Unified Memory Architecture (UMA) is fully capable of training and running fine-tuned lightweight Transformers.

* **Acceleration Engine:** Metal Performance Shaders (`torch.device("mps")`)
* **Optimal Batch Size:** `16` to `32`
* **VRAM Allocation:** Up to ~10-12 GB accessible by GPU via UMA
* **Training Time (DistilBERT):** ~10–20 minutes for standard datasets

---

## 3. Dataset Recommendations

To train an effective model, combine multi-class toxicity datasets:

* **Jigsaw Toxic Comment Classification (Kaggle):** Multi-label dataset covering toxicity, severe toxicity, insults, and identity attacks.
* **Davidson Hate Speech & Offensive Language Dataset:** Helps distinguish between actual hate speech vs. offensive casual language.
* **Bad Words Lists / Test Suites:** Open-source wordlists (`profanity-check`, `badwords`) for rule-based matching and validation.

---

## 4. Key Challenges & Preprocessing

1. **The Scunthorpe Problem:** Avoid false positives on benign words containing swear-word substrings (e.g., *Scunthorpe*, *assassination*, *password*).
2. **Leetspeak Normalization:** Map character variations before tokenization (e.g., `f00l` $\rightarrow$ `fool`, `@` $\rightarrow$ `a`, `f.u.c.k` $\rightarrow$ `fuck`).
3. **Unicode Cleaning:** Strip zero-width spaces (`\u200B`), homoglyphs, and non-standard character paddings.

---

## 5. Complete Python Fine-Tuning Script (`train_profanity_filter.py`)

Below is the complete PyTorch + Hugging Face Transformers script optimized for Apple Silicon MPS.

```python
import os
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# 1. Fallback for unsupported PyTorch MPS operators
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def check_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 MPS Device Detected: Using Apple Silicon M2 GPU.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not available: Falling back to CPU.")
    return device

def prepare_data(tokenizer):
    train_texts = [
        "I really love this product, it works great!",
        "You are an absolute idiot and a loser.",
        "That presentation was super informative, thanks.",
        "Shut up you complete piece of trash.",
        "Have a wonderful weekend everyone!",
        "Get out of here, nobody likes you f3ckhead.",
        "The food at this restaurant was delicious.",
        "What a terrible and ugly mistake."
    ]
    train_labels = [0, 1, 0, 1, 0, 1, 0, 1]  # 0: Clean, 1: Profane/Toxic

    val_texts = [
        "Great job on the project team!",
        "You are disgusting and useless.",
        "This is an amazing experience.",
        "I hope you fail miserably."
    ]
    val_labels = [0, 1, 0, 1]

    raw_train = Dataset.from_dict({"text": train_texts, "label": train_labels})
    raw_val = Dataset.from_dict({"text": val_texts, "label": val_labels})

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    train_dataset = raw_train.map(tokenize_func, batched=True)
    val_dataset = raw_val.map(tokenize_func, batched=True)

    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    device = check_device()
    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_dataset, val_dataset = prepare_data(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./profanity_model_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting Training on M2 MPS GPU...")
    trainer.train()

    save_path = "./final_profanity_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    main()
```

---

## 6. AI Agent Prompt Context (Copy-Paste Ready)

```text
PROJECT BRIEF FOR LOCAL AI AGENT:
Goal: Build a profanity and toxicity filtering system for text processing.
Hardware Target: Apple Silicon Mac M2 with 16GB Unified Memory.
Model Choice: DistilBERT fine-tuned via PyTorch MPS backend (`torch.device('mps')`).
Environment Setup: `PYTORCH_ENABLE_MPS_FALLBACK=1`, batch_size=16, max_length=128.
Data Strategy: Combine Jigsaw Toxic Comment dataset with leetspeak/unicode preprocessing.
System Architecture: Hybrid pipeline (Regex pre-filter + Transformer classifier).
```
