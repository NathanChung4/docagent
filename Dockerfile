# Multi-stage build. Stage 1 installs deps + pre-downloads the HF models so
# the runtime image (and CI cold starts) never pull weights at request time.
# Stage 2 is a slim runtime that copies the venv + the model cache + app code.
#
# One image, two services: docker-compose overrides CMD to switch between
# uvicorn (api) and streamlit (ui).

# --- Stage 1: builder --------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps for any Python wheels that need to compile from source. The default
# wheels for psycopg[binary], torch, sentence-transformers all ship binaries on
# linux/amd64 — build-essential is here defensively for any transitive dep that
# misses a wheel on the runner's arch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Layer 1: dependencies (cached as long as requirements.txt is unchanged).
# Install CPU-only torch BEFORE requirements.txt — sentence-transformers pulls
# torch transitively, and `pip install torch` defaults to the CUDA wheel
# (~1.1 GB + ~3 GB of bundled nvidia-cudnn/cublas) which is dead weight on a
# CPU-only inference image. Pinning the CPU wheel here cuts the runtime
# image from ~12 GB to ~5 GB. The remaining 5 GB is mostly the bge-reranker
# weights (1.1 GB) + torch CPU (~1.5 GB) + transformers/sentence-transformers
# (~400 MB) — all load-bearing for the rerank step that drove Phase 9's
# +21pp MRR delta. Going smaller would mean dropping the reranker.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install -r requirements.txt

# Layer 2: pre-download HF models into /opt/hf-cache. Saves ~30s of first-request
# latency for both the API (embedder) and the eval pipeline (reranker).
ENV HF_HOME=/opt/hf-cache \
    PATH="/opt/venv/bin:$PATH"
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('BAAI/bge-reranker-base')"

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/opt/hf-cache \
    KNOWLEDGE_DOMAIN=sample

# libgomp1 is required by the torch wheel that sentence-transformers pulls in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
# --chown on the COPY itself avoids a separate `RUN chown -R` layer (~280 MB
# of model files would otherwise be re-written into a new layer).
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app domains ./domains
COPY --chown=app:app data ./data
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app pyproject.toml requirements.txt ./

USER app

EXPOSE 8000
# docker-compose's `ui` service overrides this with the streamlit command.
CMD ["uvicorn", "knowledge_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
