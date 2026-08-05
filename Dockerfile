# Small runtime image: no torch/transformers/optimum — just onnxruntime + tokenizers.
FROM python:3.11-slim

WORKDIR /app

# Runtime deps only (must match pyproject [project].dependencies).
RUN pip install --no-cache-dir onnxruntime tokenizers numpy pyahocorasick

# Runtime code + matcher data + the int8 model (pulled via git lfs before build).
COPY src/ ./src/
COPY wordlists/ ./wordlists/

ENV PYTHONPATH=/app/src PORT=8000
EXPOSE 8000
CMD ["python", "-m", "slurshield.serve"]
