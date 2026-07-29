import pytest

from ngword.infer import classify


def test_name_routes_to_matcher():
    r = classify("putangina", "name")
    assert r["kind"] == "name" and r["decision"] == "block"

def test_title_uses_matcher_path():
    r = classify("scunthorpe united", "title")
    assert r["decision"] == "allow"

def test_chat_prefilter_blocks_obfuscated_profanity_without_model():
    # The deobfuscation pre-filter fires before the model loads, so no model needed.
    r = classify("sh1t player uninstall", "chat")
    assert r["decision"] == "block"
    assert r["reason"].startswith("prefilter:")

def test_empty_input_does_not_crash():
    r = classify("", "chat")
    assert r["decision"] in ("allow", "block")

def test_chat_returns_four_class_label():
    pytest.importorskip("optimum")
    import os
    if not (os.path.isdir("onnx_model") or os.path.isdir("final_model")):
        pytest.skip("train + export first")
    r = classify("you are absolute trash, kill yourself", "chat")
    assert r["label"] in ("clean", "explicit", "implicit", "action")
    assert 0.0 <= r["score"] <= 1.0
