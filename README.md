# DocAgent

**A retrieval-augmented agent for technical documentation.** Streams grounded answers with source citations, dispatches validated tools to produce real artifacts (config files, sweep summaries, status lookups), and ships with a domain-pluggable architecture so swapping the corpus is an env var, not a fork.

[![ci](https://github.com/USER/docagent/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/docagent/actions/workflows/ci.yml)
[![eval](https://github.com/USER/docagent/actions/workflows/eval.yml/badge.svg)](https://github.com/USER/docagent/actions/workflows/eval.yml)

> **Live demo:** _coming soon — Hugging Face Spaces deploy in progress._
>
> _Screenshots will be added once the live demo is up._

## What it does

- **Grounded Q&A.** Asks Claude with a curated context window: the top retrieved chunks from your corpus. Every answer cites the documents it used; says "I don't know" rather than hallucinate when context is insufficient.
- **Agentic tool use.** When a prompt is an imperative ("generate a config file for X"), the agent dispatches to a validated tool that produces a real artifact. Validation runs against the same spec the retriever just surfaced — failed validations come back as tool errors the agent can recover from.
- **Streaming chat UI.** Tokens render as they arrive, with a live "Calling tool…" indicator. Per-message footer shows latency, first-token time, cost, cache hit rate.

## Quick start (Docker)

```bash
cp .env.example .env                 # fill in ANTHROPIC_API_KEY
docker compose up -d                 # postgres + api + ui
docker compose run --rm api python scripts/ingest_sample.py
open http://localhost:8501           # Streamlit UI
# API on http://localhost:8000 — Swagger UI at /docs
```

The image is multi-stage and pre-bakes the sentence-transformers + cross-encoder weights at build time, so the first request doesn't pull from Hugging Face.

## Quick start (dev mode)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

docker compose up -d postgres        # pgvector on localhost:5432 only

# Run the retrieval eval suite against the sample domain
KNOWLEDGE_DOMAIN=sample python scripts/run_eval.py --reset --mode retrieval

# Run the answer + tool eval (requires ANTHROPIC_API_KEY)
KNOWLEDGE_DOMAIN=sample python scripts/run_eval.py --reset --mode all
```

`pytest` requires Docker Desktop running locally — `tests/conftest.py` spins up an ephemeral pgvector container per session via `testcontainers-python`. CI sidesteps that by running pgvector as a service container.

## Metrics that matter

The retrieval pipeline is **evaluated, not vibe-tested.** Hand-labeled eval sets drive every claim below. The numbers in the table come from the included sample-domain suite; a larger 30-question domain-specific suite (private) confirms the same conclusions at higher statistical power.

| Metric | Value | What it means |
|---|---|---|
| Recall@5 (sample) | 1.000 | The right document made it into the top-5 on every question |
| MRR (sample) | 0.646 | Average rank of the first correct doc; ≥0.5 means the correct doc usually lands in the top-2 |
| Reranker MRR lift | **+21 pp** | Cross-encoder reranker vs hybrid-only at the best (alpha, candidate_k) cell |
| Answer pass rate | 93% | Haiku-as-judge grading on the domain suite; pass = score ≥ 0.7 |
| Per-query cost | ~$0.003 | With Anthropic prompt caching enabled (>70% cache hit rate on stable prefix) |
| Eval-as-CI gate | MRR ≥ 0.50, recall@5 ≥ 0.75 | Floors enforced on every PR; build fails if breached |

The CI gate is intentionally a smoke test, not a benchmark — the sample suite is small enough that each question is 25% of any aggregate. The point is to catch *regressions*, not certify perfection.

## Architecture

```
Wiki / Code / CSV / Excel  →  loader  →  chunker  →  embeddings  →  pgvector
                                                                       ↓
                User question  →  hybrid retrieval (semantic + BM25, top-20)
                                                                       ↓
                                                cross-encoder reranker  (top-5)
                                                                       ↓
                                      Claude (streaming + prompt caching + tools)
                                                                       ↓
                                              SSE → Streamlit chat UI
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design, data model, API spec, and the rationale behind each pick.

**Design decisions worth defending in a code review:**

- **Hybrid retrieval + cross-encoder reranker** — BM25 catches exact-term matches (parameter names, identifiers) that semantic search smooths over. The reranker reorders the top-20 into a precise top-5 and earns +21pp MRR.
- **Streaming responses (SSE)** — first-token latency drops from ~2s to <500ms; the UX difference is the difference between "responsive" and "loading spinner".
- **Anthropic prompt caching on system prompt + tool schemas + retrieved context** — ~90% cost reduction and ~80% latency reduction on the cached prefix.
- **Tool input validation against retrieved spec** — the agent's tool calls are checked against the same spec the retriever just surfaced. Failed validations come back as `is_error=true` tool_results so the agent can self-correct in the next turn rather than silently writing bad files.
- **pgvector over standalone vector DBs** — one database for both relational metadata and embeddings; SQL joins between vector similarity and metadata predicates that single-purpose stores can't natively express.
- **Eval-as-CI** — retrieval quality is treated as a build artifact, not a vibe. CI fails on regression.
- **Domain isolation** — `src/knowledge_rag/` contains zero references to any specific corpus. New domains plug in via the `Domain` interface; swap corpora with `KNOWLEDGE_DOMAIN=...`.

## Tech stack

| Layer | Choice |
|---|---|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim, local) |
| Vector store | pgvector on Postgres 16 (HNSW, cosine) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Keyword search | BM25 (rank-bm25) |
| Generation | Claude via Anthropic SDK (streaming + prompt caching + tool use) |
| API | FastAPI + uvicorn + sse-starlette |
| UI | Streamlit |
| Persistence | Postgres (sessions, query logs, chunks all in one DB) |
| Testing | pytest + testcontainers (pgvector per-session) |
| Containers | Docker + docker-compose |
| CI | GitHub Actions (lint, test, Docker build, retrieval eval gate) |

## Domain isolation

Switching corpora is one env var, not a fork. `src/knowledge_rag/` is generic — every domain-specific concern (where the corpus lives, which tools to register, what the eval questions are) lives in a separate domain pack under `domains/`. The shipped sample pack is the worked example; private packs slot in alongside it.

To add your own domain:

```python
# domains/your_pack/config.py
from knowledge_rag.domain import DataSourcePaths, Domain

class YourDomain(Domain):
    name = "your_pack"

    def paths(self) -> DataSourcePaths: ...
    def tools(self) -> list[Tool]: ...
    def eval_dataset(self) -> list[dict]: ...
```

Then `KNOWLEDGE_DOMAIN=your_pack docker compose up`.

## Development

```bash
pytest                                          # full test suite (requires Docker)
pytest -m "not slow"                            # skip slow/integration tests
ruff check . && ruff format --check .           # lint
python scripts/run_eval.py --mode retrieval     # retrieval eval only (free)
python scripts/run_eval.py --mode all           # +answer +tool (uses API)
python scripts/run_sweep.py                     # alpha × candidate_k × rerank sweep
python scripts/build_report.py results/...      # render HTML/Markdown eval report
```

## License

MIT — see `LICENSE`.
