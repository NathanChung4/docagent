"""Postgres + pgvector backed vector store.

Single table per VectorStore instance. Embeddings stored as `vector(384)` with
an HNSW cosine index; everything else (source_type, title, uri, user metadata)
lives in a `metadata jsonb` column with a GIN index for fast filter lookups.

Re-indexing model: chunks belong to a parent doc via the `doc_id` column.
`reindex_document` deletes by doc_id then re-inserts — the same atomic-from-the-
caller's-perspective contract as the previous ChromaDB store.

The `where=` filter on `query()` accepts a flat equality dict like
`{"source_type": "wiki"}` and translates it to `metadata @> %s::jsonb`.
Anything fancier (`$or`, `$in`, nested operators) raises `NotImplementedError`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from knowledge_rag.embeddings import Embedder
from knowledge_rag.models import Chunk, Document, RetrievalResult, SourceType

__all__ = [
    "RetrievalResult",
    "VectorStore",
    "DEFAULT_DSN",
    "DEFAULT_TABLE",
    "EMBEDDING_DIM",
]

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
DEFAULT_TABLE = "chunks"
EMBEDDING_DIM = 384


class VectorStore:
    """Postgres + pgvector backed collection of Chunks.

    Stores each chunk's text, embedding, and metadata. Queries use the cosine
    distance operator (`<=>`) over an HNSW index and return RetrievalResult
    objects with a normalized similarity score in [0, 1].

    Attributes:
        dsn: Postgres connection string.
        table_name: Table holding chunks. Each VectorStore instance owns one.
        embedder: Embedder used when add_chunks gets un-embedded chunks.
    """

    # Only field still split into a real column. Everything else is serialized
    # into the metadata jsonb dict; Chunk reconstructs source_type/title/uri
    # from there on read.
    _RESERVED = {"doc_id"}

    def __init__(
        self,
        dsn: str = DEFAULT_DSN,
        table_name: str = DEFAULT_TABLE,
        embedder: Embedder | None = None,
    ) -> None:
        # Validate the table identifier — we interpolate it into DDL/DML, never
        # bind it. Anything outside [a-zA-Z0-9_] is a SQL-injection risk.
        if not table_name or not table_name.replace("_", "").isalnum():
            raise ValueError(f"table_name must be alphanumeric/underscore, got: {table_name!r}")
        self.dsn = dsn
        self.table_name = table_name
        self.embedder = embedder or Embedder()
        self._conn: Any = None
        self._schema_ready = False

    # --- connection / schema setup ------------------------------------------

    def _get_conn(self):
        """Lazy psycopg connection; registers the pgvector adapter once."""
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self.dsn, autocommit=True)
            # Extension must exist before register_vector can find the type.
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(self._conn)
            self._schema_ready = False
        if not self._schema_ready:
            self._ensure_schema()
            self._schema_ready = True
        return self._conn

    def _ensure_schema(self) -> None:
        """Create the chunks table and its indexes if they don't exist."""
        t = self.table_name
        ddl = f"""
            CREATE TABLE IF NOT EXISTS {t} (
                id        text PRIMARY KEY,
                doc_id    text NOT NULL,
                content   text NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                metadata  jsonb NOT NULL DEFAULT '{{}}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS {t}_doc_id_idx ON {t} (doc_id);
            CREATE INDEX IF NOT EXISTS {t}_embedding_idx
              ON {t} USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS {t}_metadata_idx
              ON {t} USING gin (metadata jsonb_path_ops);
        """
        with self._conn.cursor() as cur:
            cur.execute(ddl)

    # --- write path ---------------------------------------------------------

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed (if needed) and upsert chunks into the table.

        Existing rows with the same chunk_id are overwritten. Returns the count.
        """
        if not chunks:
            return 0
        unembedded = [c for c in chunks if c.embedding is None]
        if unembedded:
            self.embedder.embed_chunks(unembedded)

        rows: list[tuple[Any, ...]] = []
        for c in chunks:
            assert c.embedding is not None  # set by embed_chunks above
            user_meta = {k: v for k, v in c.metadata.items() if k not in self._RESERVED}
            metadata = {
                "source_type": c.source_type.value,
                "title": c.title,
                "uri": c.uri,
                **user_meta,
            }
            rows.append((c.chunk_id, c.doc_id, c.content, c.embedding, json.dumps(metadata)))

        conn = self._get_conn()
        sql = f"""
            INSERT INTO {self.table_name} (id, doc_id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                doc_id    = EXCLUDED.doc_id,
                content   = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata  = EXCLUDED.metadata;
        """
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        return len(chunks)

    # --- query path ---------------------------------------------------------

    def query(
        self,
        text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search via cosine distance. Returns up to `k` chunks.

        Args:
            text: Natural-language query string.
            k: Max number of hits to return.
            where: Optional flat equality filter, e.g. {"source_type": "wiki"}.
        """
        if not text:
            return []

        query_vec = list(self.embedder.embed_texts([text])[0])
        params: list[Any] = [query_vec]
        where_clause = ""
        if where:
            self._validate_where(where)
            where_clause = "WHERE metadata @> %s::jsonb"
            params.append(json.dumps(where))
        params.append(query_vec)
        params.append(int(k))

        sql = f"""
            SELECT id, doc_id, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[RetrievalResult] = []
        for chunk_id, doc_id, content, metadata, similarity in rows:
            meta = dict(metadata or {})
            source_type = SourceType(meta.pop("source_type", SourceType.WIKI.value))
            title = meta.pop("title", "")
            uri = meta.pop("uri", "")
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                source_type=source_type,
                title=title,
                uri=uri,
                metadata=meta,
            )
            # Cosine similarity ∈ [-1, 1] for arbitrary vectors; normalized
            # sentence-transformers embeddings keep it in [0, 1] in practice.
            # Clamp defensively so downstream consumers can treat it as a score.
            results.append(RetrievalResult(chunk=chunk, score=max(0.0, float(similarity))))
        return results

    @staticmethod
    def _validate_where(where: dict[str, Any]) -> None:
        """Reject filter shapes we don't translate."""
        for key, value in where.items():
            if key.startswith("$"):
                raise NotImplementedError(
                    f"Operator filter {key!r} is not supported; "
                    "only flat key=value equality (e.g. {'source_type': 'wiki'}) works."
                )
            if isinstance(value, dict):
                raise NotImplementedError(
                    f"Nested filter on {key!r} is not supported; use a flat equality dict."
                )

    # --- maintenance --------------------------------------------------------

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove every chunk that belongs to the given parent document."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE doc_id = %s;", (doc_id,))

    def reindex_document(self, doc: Document, chunks: Iterable[Chunk]) -> int:
        """Replace all chunks for `doc` with the supplied new chunks."""
        self.delete_by_doc_id(doc.doc_id)
        return self.add_chunks(list(chunks))

    def count(self) -> int:
        """Total number of chunks currently in the table."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            return cur.fetchone()[0]

    def list_documents(self) -> list[dict[str, Any]]:
        """One row per parent document with its chunk count and metadata."""
        conn = self._get_conn()
        sql = f"""
            SELECT
                doc_id,
                MAX(metadata->>'title')       AS title,
                MAX(metadata->>'source_type') AS source_type,
                MAX(metadata->>'uri')         AS uri,
                COUNT(*)                      AS chunk_count
            FROM {self.table_name}
            GROUP BY doc_id
            ORDER BY MAX(metadata->>'title') NULLS LAST;
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            {
                "doc_id": doc_id,
                "title": title,
                "source_type": source_type,
                "uri": uri,
                "chunk_count": chunk_count,
            }
            for doc_id, title, source_type, uri, chunk_count in rows
        ]

    def reset(self) -> None:
        """Drop the table entirely. Next operation recreates the schema."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
        self._schema_ready = False

    def close(self) -> None:
        """Close the underlying connection. Safe to call multiple times."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
        self._schema_ready = False
