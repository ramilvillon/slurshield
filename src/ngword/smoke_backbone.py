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
