from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

_MODEL = Path(__file__).resolve().parents[1] / "src" / "ngword" / "model" / "model_quantized.onnx"


def test_shipped_int8_model_loads_and_outputs_four_classes():
    if not _MODEL.exists():
        pytest.skip("shipped model not present (git lfs pull)")
    sess = ort.InferenceSession(str(_MODEL), providers=["CPUExecutionProvider"])
    assert {i.name for i in sess.get_inputs()} == {"input_ids", "attention_mask"}
    ids = np.array([[2, 708, 943, 1]], dtype=np.int64)
    mask = np.ones_like(ids)
    logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    assert logits.shape[-1] == 4
