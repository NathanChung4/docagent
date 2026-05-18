# HF Spaces deploy image. Single container runs everything: pgvector Postgres,
# FastAPI backend on :8000 (internal), Streamlit UI on $PORT (exposed). The
# normal docker-compose dev flow keeps Postgres as a separate service; this
# image bundles it because HF free-tier Spaces give you exactly one container
# and one exposed port.
#
# Source-of-truth for the app code lives at github.com/NathanChung4/docagent.
# This image clones it at build time so the Space repo only needs three
# files (Dockerfile + entrypoint.sh + README.md). Pushing to docagent's
# main branch + re-syncing the Space rebuilds against fresh code.

# --- Stage 1: builder --------------------------------------------------------
# Same base as the runtime stage so the venv's python binary and symlinks
# stay valid when we COPY it across.
FROM pgvector/pgvector:pg16 AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the public source so we have requirements.txt to install against.
# `--depth=1` keeps the build cache small. Pin to main; HF Space rebuilds
# pick up new commits automatically.
WORKDIR /build
ARG DOCAGENT_REF=main
RUN git clone --depth=1 --branch ${DOCAGENT_REF} \
    https://github.com/NathanChung4/docagent.git /build/docagent

# Layer: CPU-only torch first (same trick as docagent's own Dockerfile —
# defaulting to the CUDA wheel adds ~3 GB of dead weight on a CPU image).
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install -r /build/docagent/requirements.txt

# Pre-download HF models so cold start doesn't pay for them. ~30s saved per
# Space restart (HF cold-starts a Space after idle).
ENV HF_HOME=/opt/hf-cache \
    PATH="/opt/venv/bin:$PATH"
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('BAAI/bge-reranker-base')"

# --- Stage 2: runtime --------------------------------------------------------
FROM pgvector/pgvector:pg16 AS runtime

# HF Spaces convention: run as UID 1000. The base image (postgres:16) doesn't
# define this user, so add it. Postgres itself will also run as this UID —
# initdb just needs PGDATA to be owned by the running user.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 1000 -m -s /bin/bash user

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=user:user /opt/hf-cache /opt/hf-cache
COPY --from=builder --chown=user:user /build/docagent /app

COPY --chown=user:user entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/opt/venv/bin:/usr/lib/postgresql/16/bin:$PATH" \
    PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    HOME=/home/user \
    KNOWLEDGE_DOMAIN=sample \
    KNOWLEDGE_DB_DSN=postgresql://postgres:postgres@localhost:5432/postgres \
    KNOWLEDGE_RAG_API_URL=http://localhost:8000 \
    PGDATA=/tmp/pgdata \
    PORT=7860

USER user
WORKDIR /app

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
