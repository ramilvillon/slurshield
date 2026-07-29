import os
from functools import lru_cache

from ngword import config
from ngword.names import check_name, profanity_hit

_MATCHER_KINDS = {"name", "title"}


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoTokenizer
    if os.path.isdir("onnx_model"):
        from optimum.onnxruntime import ORTModelForSequenceClassification
        m = ORTModelForSequenceClassification.from_pretrained(
            "onnx_model", file_name="model_quantized.onnx")
        tok = AutoTokenizer.from_pretrained("onnx_model")
    else:  # torch fallback
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained("final_model").eval()
        tok = AutoTokenizer.from_pretrained("final_model")
    return m, tok


def classify(text: str, kind: str) -> dict:
    if not isinstance(text, str):
        text = str(text or "")

    if kind in _MATCHER_KINDS:
        r = check_name(text)
        return {"kind": kind, "decision": r["decision"], "label": None,
                "score": None, "reason": r["reason"]}

    # chat -> model
    if not text.strip():
        return {"kind": "chat", "decision": "allow", "label": "clean",
                "score": 1.0, "reason": "empty"}

    # Deobfuscation pre-filter: the model reads raw text and misses leetspeak/unicode
    # evasion, so catch explicit slurs on the normalized shadow before the model runs.
    hit = profanity_hit(text)
    if hit:
        return {"kind": "chat", "decision": "block", "label": "explicit",
                "score": 1.0, "reason": f"prefilter:{hit}"}

    import torch
    m, tok = _model()
    inp = tok([text], return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH)
    with torch.no_grad():
        probs = m(**inp).logits.softmax(-1)[0]
    idx = int(probs.argmax())
    label = config.CLASS_NAMES[idx]
    return {"kind": "chat", "decision": "allow" if label == "clean" else "block",
            "label": label, "score": float(probs[idx]), "reason": label}
