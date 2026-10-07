# syntax=docker/dockerfile:1.7
# Demo image: web page + API, both models and the chunk vectors baked in,
# so the container runs offline. Build context = repository root:
#   docker build -f demo/Dockerfile -t semantic-search-demo .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Optional: a corporate proxy's CA certificate for the build
#   docker build --secret id=extra_ca,src=proxy-ca.crt ...
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then cat /run/secrets/extra_ca >> /etc/ssl/certs/ca-certificates.crt; fi

RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install fastapi==0.142.2 uvicorn==0.54.0 requests==2.33.1 numpy==2.4.6 \
        sentence-transformers==6.1.0 transformers==5.17.0 tokenizers==0.23.2 \
        sentencepiece==0.2.2 huggingface_hub==1.33.0 pydantic==2.13.5

# Models (~4.5 GB): bge-m3 for vectors, bge-reranker-v2-m3 for the decision.
# bge-m3's ONNX copy and sparse/colbert heads are not used.
RUN python -c "\
from huggingface_hub import snapshot_download as d; \
d('BAAI/bge-m3', local_dir='/models/bge-m3', \
  ignore_patterns=['onnx/*', 'imgs/*', '*.jpg', '*.md', 'colbert_linear.pt', 'sparse_linear.pt']); \
d('BAAI/bge-reranker-v2-m3', local_dir='/models/bge-reranker-v2-m3', \
  allow_patterns=['*.json', '*.safetensors', '*.model'])"

ENV BGE_M3_DIR=/models/bge-m3 \
    RERANKER_DIR=/models/bge-reranker-v2-m3 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ES_URL=http://elasticsearch:9200 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
COPY experiment/textnorm.py experiment/testset.json experiment/
COPY experiment/data/chunks.jsonl experiment/data/live_run_2026-10-07.json experiment/data/
COPY demo/engine.py demo/server.py demo/qa_demo.py demo/validate_live_sim.py demo/index.html demo/

# Chunk vectors for both pipelines, computed once at build time.
RUN cd demo && python -c "from engine import Engine; Engine(with_reranker=False).prepare()"

EXPOSE 8000
WORKDIR /app/demo
CMD ["python", "server.py"]
