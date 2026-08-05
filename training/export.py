from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def export(src: str = "final_model", dst: str = "src/slurshield/model") -> None:
    model = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    tok = AutoTokenizer.from_pretrained(src)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)

    quantizer = ORTQuantizer.from_pretrained(dst)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=dst, quantization_config=qconfig)
    print(f"exported int8 ONNX to {dst}/")


if __name__ == "__main__":
    export()
