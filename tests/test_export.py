import pytest


def test_quantized_model_loads_and_predicts():
    pytest.importorskip("optimum")
    import os
    if not os.path.isdir("onnx_model"):
        pytest.skip("run `python -m ngword.export` first")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    m = ORTModelForSequenceClassification.from_pretrained("onnx_model", file_name="model_quantized.onnx")
    tok = AutoTokenizer.from_pretrained("onnx_model")
    inp = tok(["you are trash"], return_tensors="pt")
    logits = m(**inp).logits
    assert logits.shape[-1] == 4
