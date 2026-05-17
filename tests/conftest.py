"""Pytest fixtures and global config.

Forces KNOWLEDGE_DOMAIN=sample for every test so we never accidentally exercise
the private pack from the test suite. CI relies on this.

Shared fixtures here are session-scoped where construction is cheap and the
returned object is read-only — avoids per-test re-loading of the sample corpus.

The `pg_container` / `pg_dsn` fixtures spin up an ephemeral pgvector container
once per pytest session via testcontainers. Tests that touch the vector store
require Docker Desktop to be running; they will fail loudly otherwise.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Force the selector loop on Windows.

    Why: psycopg3's async pool refuses to run on ProactorEventLoop (Windows
    default since Py3.8). Without this fixture every async DB test fails with
    PoolTimeout from `Psycopg cannot use the 'ProactorEventLoop'`.
    """
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# Make the repo root importable so `import knowledge_rag` and `import domains`
# work without requiring `pip install -e .` first. This keeps a fresh checkout
# runnable in CI before the install step finishes.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("KNOWLEDGE_DOMAIN", "sample")


@pytest.fixture(scope="session")
def domain():
    """Active sample-domain instance. Cheap to construct; safe to share."""
    from knowledge_rag.domain import get_domain

    return get_domain()


@pytest.fixture(scope="session")
def paths(domain):
    """DataSourcePaths for the sample domain."""
    return domain.paths()


@pytest.fixture(scope="session")
def sample_docs(domain):
    """Run the full ingestion pipeline once per test session.

    Loaders are cheap individually but several test files exercise the same
    corpus; this avoids repeated disk + parse passes.
    """
    from knowledge_rag.loaders.ingestion import ingest_all

    return ingest_all(domain)


@pytest.fixture(scope="session")
def pg_container():
    """Session-scoped pgvector container.

    Spins up `pgvector/pgvector:pg16` once per pytest invocation (~3s startup,
    amortized across all vectorstore tests) and tears it down at session end.
    Requires Docker Desktop to be running.

    Skipped when PG_DSN_OVERRIDE is set — CI runs Postgres as a workflow
    service container and points tests at it via that env var, avoiding the
    Docker-in-Docker dance that testcontainers needs.
    """
    if os.environ.get("PG_DSN_OVERRIDE"):
        yield None
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError as exc:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for pgvector tests. "
            "Install with: pip install 'testcontainers[postgres]'\n"
            f"Underlying error: {exc}"
        )
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_dsn(pg_container) -> str:
    """Plain psycopg DSN built from the testcontainer's exposed port.

    `PostgresContainer.get_connection_url()` returns a SQLAlchemy-style URL
    (`postgresql+psycopg2://...`); psycopg3 wants the `postgresql://` prefix.

    When PG_DSN_OVERRIDE is set, return that DSN verbatim — used in CI to
    talk to a service-container Postgres without spinning up testcontainers.
    """
    override = os.environ.get("PG_DSN_OVERRIDE")
    if override:
        return override
    return (
        f"postgresql://{pg_container.username}:{pg_container.password}"
        f"@{pg_container.get_container_host_ip()}"
        f":{pg_container.get_exposed_port(5432)}"
        f"/{pg_container.dbname}"
    )


@pytest.fixture(scope="module")
def shared_embedder():
    """One Embedder instance shared across all tests in a module.

    Loading the sentence-transformers weights costs ~3s; sharing avoids paying
    that per test. Module scope (not session) keeps tests of different shapes
    isolated if a future test ever needs a different model.
    """
    from knowledge_rag.embeddings import Embedder

    return Embedder()


@pytest.fixture
def store(pg_dsn, shared_embedder):
    """Fresh pgvector table per test; teardown drops it.

    Used by `test_vectorstore.py` and `test_retrieval.py`. Each test gets a
    unique `chunks_<uuid>` table so they can run in parallel without bleed.
    """
    from uuid import uuid4

    from knowledge_rag.vectorstore import VectorStore

    table = f"test_{uuid4().hex[:12]}"
    s = VectorStore(dsn=pg_dsn, table_name=table, embedder=shared_embedder)
    try:
        yield s
    finally:
        s.reset()
        s.close()
