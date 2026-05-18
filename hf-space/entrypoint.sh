#!/bin/bash
# HF Space entrypoint: start postgres -> ingest corpus -> start API -> start UI.
#
# Lifecycle (single container, ephemeral disk on HF free tier):
#   1. initdb if PGDATA is empty (first start after each cold boot)
#   2. start postgres in the background, wait for pg_isready
#   3. create the vector extension (idempotent)
#   4. ingest the sample corpus (~30s, runs every cold start since disk is wiped)
#   5. start uvicorn on :8000 in the background (internal, not exposed)
#   6. exec streamlit on $PORT in the foreground (HF Space watches PID 1)
#
# Streamlit's heartbeat keeps the container alive; if either pg or uvicorn
# crashes after startup we'd silently lose backend functionality. Acceptable
# for a demo; a production deploy would use a supervisor.

set -e

DATA_INIT_MARKER="${PGDATA}/PG_VERSION"

if [ ! -f "$DATA_INIT_MARKER" ]; then
    echo "[entrypoint] Initializing postgres data dir at $PGDATA..."
    initdb -D "$PGDATA" -U postgres --auth=trust >/dev/null
    # Local-only auth; the container's network is its own loopback.
    echo "listen_addresses = 'localhost'" >> "$PGDATA/postgresql.conf"
    echo "unix_socket_directories = '/tmp'" >> "$PGDATA/postgresql.conf"
fi

echo "[entrypoint] Starting postgres..."
postgres -D "$PGDATA" -p 5432 -k /tmp &
PG_PID=$!

echo "[entrypoint] Waiting for postgres to accept connections..."
for i in $(seq 1 30); do
    if pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
        echo "[entrypoint] Postgres ready (pid $PG_PID, ${i}s)."
        break
    fi
    sleep 1
done

if ! pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
    echo "[entrypoint] FATAL: postgres did not become ready within 30s." >&2
    exit 1
fi

# pgvector extension is provided by the base image; CREATE EXTENSION just
# registers it in the postgres catalog. Idempotent across restarts (which
# don't happen here because PGDATA is ephemeral, but keep the IF NOT EXISTS
# in case a future change persists the volume).
psql -h localhost -p 5432 -U postgres -d postgres \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo "[entrypoint] Ingesting sample corpus..."
python /app/scripts/ingest_sample.py

echo "[entrypoint] Starting FastAPI on :8000 (internal)..."
uvicorn knowledge_rag.api:app --host 127.0.0.1 --port 8000 --log-level warning &
API_PID=$!

# Give uvicorn a moment to bind before streamlit's reachability check fires.
sleep 2

echo "[entrypoint] Starting Streamlit on :${PORT}..."
exec streamlit run /app/src/knowledge_rag/dashboard.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT}" \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
