"""One-shot ingest: load + chunk + embed + index the active domain's corpus.

Used inside Docker after `docker compose up`:
    docker compose run --rm api python scripts/ingest_sample.py

Idempotent — `ensure_index` reuses an existing populated table by default. Pass
`--reset` to drop the table first (forces re-embed; needed if the embedding
model or chunking strategy changed).

Reads KNOWLEDGE_DOMAIN (default: 'sample') and KNOWLEDGE_DB_DSN from env so
the same script works locally and inside the compose network.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from knowledge_rag.eval_pipeline import ensure_index
from knowledge_rag.vectorstore import DEFAULT_DSN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", default=None, help="Domain pack (default: $KNOWLEDGE_DOMAIN or 'sample')."
    )
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: $KNOWLEDGE_DB_DSN).")
    parser.add_argument("--table", default=None, help="Table name (default: chunks).")
    parser.add_argument(
        "--reset", action="store_true", help="Drop the table first; force re-embed."
    )
    args = parser.parse_args()

    domain = args.domain or os.environ.get("KNOWLEDGE_DOMAIN", "sample")
    dsn = args.dsn or os.environ.get("KNOWLEDGE_DB_DSN", DEFAULT_DSN)
    table = args.table or "chunks"

    print(f"Ingesting domain '{domain}' into table {table!r}...")
    _store, _bm25, stats = ensure_index(domain, dsn, table, reset=args.reset)
    print(f"  -> {stats.n_documents} documents, {stats.n_chunks} chunks indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
