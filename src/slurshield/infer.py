from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from slurshield import config
from slurshield.names import check_name, profanity_hit

_MATCHER_KINDS = {"name", "title"}
_MODEL_DIR = Path(__file__).resolve().parent / "model"


@lru_cache(maxsize=1)
def _session():
    sess = ort.InferenceSession(
        str(_MODEL_DIR / "model_quantized.onnx"), providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(_MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=config.MAX_LENGTH)
    return sess, tok


def _chat_model(text: str) -> dict:
    sess, tok = _session()
    enc = tok.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(probs.argmax())
    label = config.CLASS_NAMES[idx]
    return {"kind": "chat", "decision": "allow" if label == "clean" else "block",
            "label": label, "score": float(probs[idx]), "reason": label}


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

    return _chat_model(text)
