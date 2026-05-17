# DocAgent — Architecture

## Overview

DocAgent is a retrieval-augmented generation (RAG) system with an agentic capability layer. It ingests technical documentation from heterogeneous sources — wiki pages, code repositories, structured CSV reports, and spreadsheet checklists — chunks each according to its structure, indexes the chunks in Postgres + pgvector, and answers natural-language questions over them. Every answer is grounded in retrieved documents with source citations; the system says "I don't know" rather than hallucinate when context is insufficient. On prompts that ask for action ("generate a config file for X"), an agent loop dispatches to validated tools that produce real artifacts.

The core (`src/knowledge_rag/`) is **domain-pluggable**: any specific corpus, set of tools, and evaluation dataset live in a separate domain pack under `domains/`. The default `sample` pack ships components such as `clock_divider`, `flow_controller`, `pressure_sensor`, `signal_buffer`, `temp_regulator`, and `voltage_monitor` for demo purposes; private corpora can be slotted in by writing a new `Domain` implementation.

## Problem statement

Engineering knowledge ends up scattered:

- **Wiki / Confluence-style pages** — component specifications, parameters, validation criteria
- **Code repositories** — argument parsers, default values, the actual implementation logic
- **Run / sweep reports** — CSV results from parameter exploration
- **Checklists** — ownership and readiness state

Answering a question like *"what parameters does `flow_controller` support and what's a sane setting?"* manually means opening the wiki, finding the page, cross-referencing the code, and checking the most recent sweep results. That's 15–30 minutes of context-switching per question. DocAgent compresses it to a single chat turn.

## Solution

A chat-based web application that:

1. Ingests documents from wiki pages, code, CSV reports, and Excel spreadsheets
2. Chunks each document by its natural structure (sections for prose, functions/classes for code, rows for structured data)
3. Converts chunks to vector embeddings for semantic search
4. Stores embeddings in Postgres + pgvector with a GIN index on metadata
5. On query, retrieves the most relevant chunks across all sources via hybrid search (semantic + BM25) and a cross-encoder reranker
6. Generates an answer using Claude, grounded in the retrieved context, streamed token-by-token
7. Cites every source so the user can verify
8. Decides between answering directly and invoking a validated tool when the prompt is an imperative
9. Tracks per-query cost, latency, retrieved sources, tool calls, and prompt-cache hit rate

## User stories

### Core
- **As a user**, I want to ask *"what parameters does `clock_divider` support?"* and get an accurate answer pulled from the wiki page and the code.
- **As a user**, I want to ask *"what was the best `temp_regulator` config from the latest sweep?"* and get an answer from the sweep reports.
- **As a user**, I want to ask *"who owns `voltage_monitor` and what's its status?"* and get a row from the checklist.
- **As a user**, I want every answer to cite which document it came from, so I can verify it.

### Multi-source
- **As a user**, I want answers that combine wiki + code when both are relevant — e.g. *"what does the `--thresh` parameter do?"* should pull from both the spec page and the implementation.
- **As a user**, I want to re-ingest a single document (a refreshed wiki page, a new sweep) without rebuilding the entire index.

### Agentic actions
- **As a user**, I want to say *"generate a config file for `clock_divider` with `divisor=4` and `jitter_budget_ps=300`"* and have the agent create the actual file — but only after validating the parameters against the retrieved spec.
- **As a user**, I want to say *"summarize the latest `temp_regulator` sweep"* and have the agent locate and summarize the most recent report.
- **As a user**, I want the agent to decide on its own whether a query is text-only or requires a tool call.

### Conversational
- **As a user**, I want follow-up questions in the same chat session to remember context from earlier turns — e.g. after asking about `clock_divider`, *"what about its `--jitter_budget_ps` param?"* should work without naming the component again.
- **As a user**, I want responses to stream token-by-token so I see progress immediately rather than waiting for a wall of text.

### Quality
- **As a user**, I want the system to say *"I don't know"* when the answer isn't in the documents, rather than hallucinate.
- **As a user**, I want to see which chunks were retrieved so I can judge relevance.
- **As a user**, I want to see per-query cost (tokens + dollars) so I can audit spend.

## Architecture

### Pipeline overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                                                              │
│  Wiki pages  ─┐                                              │
│  Code repo  ─┤─→ Document Loader ─→ Chunker ─→ Embedder ─→ │
│  Sweep CSV  ─┤                      (smart)    (sentence-   │
│  Excel      ─┘                                 transformers)│
│                                                     │        │
│                                              Vector Store    │
│                                              (pgvector)      │
└──────────────────────────────────────┬──────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────┐
│                  QUERY + AGENT PIPELINE                      │
│                                                              │
│  User Question + Session History ─→ Embed Query             │
│                                          │                   │
│                                          ↓                   │
│                Hybrid Retrieval (semantic + BM25, top-20)    │
│                                          │                   │
│                                          ↓                   │
│                Cross-Encoder Reranker  (top-20 → top-5)      │
│                                          │                   │
│                                          ↓                   │
│           Claude API (streaming + prompt caching)            │
│           system prompt + tool schemas + retrieved context   │
│                                          │                   │
│                            ┌─────────────┴─────────────┐     │
│                            ↓                           ↓     │
│                    Answer directly             Call tool     │
│                    (stream tokens)        ┌────────┴───────┐ │
│                            │              │ generate_      │ │
│                            │              │   config_file  │ │
│                            │              │ summarize_     │ │
│                            │              │   report       │ │
│                            │              │ lookup_item_   │ │
│                            │              │   status       │ │
│                            │              └────────┬───────┘ │
│                            │                       │         │
│                            │           tool_result back to   │
│                            │           agent → final answer  │
│                            │                       │         │
│                            └───────────┬───────────┘         │
│                                        ↓                     │
│         SSE stream ─→ Streamlit Chat UI                      │
│         (tokens + sources + tool calls + cost)               │
└─────────────────────────────────────────────────────────────┘
```

### Design decisions worth defending

1. **Multi-source ingestion** — wiki, code, structured CSV, Excel — not just PDFs. Each source type gets its own chunker.
2. **Code-aware chunking** — Python scripts are chunked by function/class boundaries (AST-based), not arbitrary character limits. A retrieved chunk is a whole function with its docstring intact, not the middle of an `if` block.
3. **Hybrid retrieval + cross-encoder reranker** — BM25 catches exact-term matches (a parameter name, a register identifier) that semantic search smooths over. The cross-encoder reranks the top-20 into a precise top-5; on benchmark runs it moved MRR by **+21pp** at a modest latency cost.
4. **Streaming responses** — Server-Sent Events stream tokens as they arrive; first-token latency drops from multiple seconds to <500 ms.
5. **Anthropic prompt caching** — System prompt + tool schemas + retrieved context are all marked cacheable. Cuts cost by ~90% and latency by ~80% on cached tokens.
6. **Agentic tool use** — Anthropic native tool-use loop (structured JSON schemas) rather than ReAct-style prose parsing. Less brittle; easier to validate; the SDK handles tool_use/tool_result round-trips.
7. **Tool input validation against retrieved spec** — tool calls are checked against the same spec the retriever just surfaced (e.g. reject `divisor=99` if the spec says max is 8). Failed validations come back as `is_error=true` tool_results so the agent can self-correct in the next turn rather than silently writing a bad file.
8. **Multi-turn conversation memory** — sessions persist conversation history server-side so follow-ups work naturally.
9. **Source attribution** — every answer cites the exact document(s).
10. **Eval-as-CI** — a retrieval-quality gate runs on every PR; the build fails if MRR or recall@5 drops below the threshold.
11. **Domain isolation** — the entire `src/knowledge_rag/` package contains zero references to any specific corpus. Domains plug in via the `Domain` interface; the active pack is selected by an env var.

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| RAG framework | LangChain (where it adds value), custom where it doesn't | Industry standard; easy to opt out of for hot paths |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, local, fast, 384-dim, good quality for technical docs |
| Vector store | pgvector on Postgres | Single database for embeddings + relational metadata; SQL joins between vector similarity and metadata predicates; production-realistic |
| Reranker | sentence-transformers cross-encoder `BAAI/bge-reranker-base` | High-precision second pass; standard in production RAG stacks |
| Generation | Claude (Anthropic SDK) | Best reasoning for technical Q&A; native streaming, prompt caching, and tool use |
| Agent pattern | Anthropic native tool use | Structured tool_use/tool_result loop; not parsed from prose |
| Streaming transport | Server-Sent Events (FastAPI StreamingResponse) | One-way LLM token streams; simpler than WebSockets, works through proxies |
| Keyword search | BM25 (rank-bm25) | Classic IR algorithm; complements semantic search for exact-term matching |
| Document loaders | Custom + LangChain | Wiki HTML, Python source, pandas for CSV/Excel |
| UI | Streamlit | Chat interface built-in; fastest path to a demo |
| Testing | pytest | Unit + integration; testcontainers spins up pgvector per test session |
| Containerization | Docker + docker-compose | One-command deployment |
| CI/CD | GitHub Actions | Lint + tests + Docker build + retrieval-quality gate on every PR |

## Data sources & ingestion

The system is multi-source by design. The included sample domain pack ships representative data for each:

### 1. Wiki pages (HTML)
- **What:** Component specification pages (sample: `clock_divider`, `flow_controller`, `pressure_sensor`, `signal_buffer`, `temp_regulator`, `voltage_monitor`)
- **Content:** Description, parameters, defaults, validation criteria
- **Chunking:** Split by section headers (h2/h3), tables kept intact
- **Metadata:** page title, source path, last-modified date

### 2. Code repository (Python)
- **What:** Component scripts with argument parsers and implementation logic
- **Chunking:** AST-aware split by function/class
- **Metadata:** file path, function name, component name

### 3. Sweep / run reports (CSV)
- **What:** Parameter sweep output — run configurations + per-run metrics
- **Chunking:** One chunk per sweep summary, one per run detail
- **Metadata:** component name, date, sweep name

### 4. Component checklist (Excel)
- **What:** A spreadsheet listing component names, owners, and status
- **Chunking:** One chunk per row (small, structured)
- **Metadata:** component name, owner, status

## Domain isolation

The repo is organized so that a new corpus is a new domain pack, not a code rewrite:

```
src/knowledge_rag/        # generic core — zero references to any specific domain
domains/sample/           # public sample pack (this repo)
domains/<your_domain>/    # plug in your own — same interface
```

The `Domain` interface (`src/knowledge_rag/domain.py`) exposes:

- **`paths()`** — where to find the source documents
- **`tools()`** — the list of `Tool` instances the agent can call
- **`eval_dataset()`** — Q&A pairs and tool-call expectations for evaluation

The agent loop, retrieval, generation, and evaluation all consume a `Domain` instance via dependency injection — they never name a specific tool or component. Switching domain is one env var: `KNOWLEDGE_DOMAIN=sample` vs `KNOWLEDGE_DOMAIN=your_pack`.

## Data model

### Document
- `id` (auto-generated), `source_type` (enum), `source_path`, `title`, `content`, `metadata` (JSON), `ingested_at`, `chunk_count`

### Chunk
- `id`, `document_id`, `content`, `embedding` (vector(384), pgvector), `metadata` (JSON — section header, function name, etc.), `chunk_index`

### Session
- `id`, `created_at`, `last_activity_at`, `messages` (ordered list of `{role, content, tool_calls?}` turns)

### QueryLog
- `id`, `session_id` (nullable), `question`, `answer`, `sources` (JSON), `retrieved_chunks` (chunk IDs + scores pre- and post-rerank), `tool_calls` (list of `{tool_name, args, result, error?}`), `first_token_ms`, `response_time_ms`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `created_at`, `user_rating` (optional)

## pgvector schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  id        text PRIMARY KEY,
  doc_id    text NOT NULL,
  content   text NOT NULL,
  embedding vector(384) NOT NULL,
  metadata  jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX chunks_doc_id_idx     ON chunks (doc_id);
CREATE INDEX chunks_embedding_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_metadata_idx   ON chunks USING gin (metadata jsonb_path_ops);
```

Why:

- **`doc_id` as a real column** because `delete_by_doc_id` is the hot path during re-ingestion; a btree on a column beats `metadata->>'doc_id'` lookups.
- **`source_type`, `title`, `uri`, and any user metadata in `metadata` jsonb** — schema stays fixed at five columns regardless of domain. Filters use the GIN index (`metadata @> %s::jsonb`).
- **HNSW over IVFFlat** at this corpus size (~700 chunks today) build time isn't a concern and HNSW's higher recall at low `ef_search` is worth it. The trade-off flips for million-vector corpora where IVFFlat's lower memory footprint wins.
- **`vector_cosine_ops` (`<=>`)** because sentence-transformers embeddings are L2-normalized and the scoring math expects cosine similarity.

## API endpoints

### Ingestion
- `POST /api/ingest` — run loaders → chunker → vector upsert → BM25 rebuild for the active domain
- `GET /api/documents` — list indexed documents
- `DELETE /api/documents/{id}` — remove a document and its chunks

### Query (streaming)
- `POST /api/query` — ask a question, get a streamed Server-Sent Events response
  - Request: `{ "question": "...", "session_id": "optional-uuid" }`
  - Response: SSE stream of typed events:
    - `event: token` `data: { "text": "..." }` (one per generated token)
    - `event: tool_call` `data: { "tool": "generate_config_file", "args": {...} }`
    - `event: tool_result` `data: { "tool": "generate_config_file", "result": "...", "error": null }`
    - `event: done` `data: { "sources": [...], "chunks_retrieved": 5, "first_token_ms": 320, "response_time_ms": 1800, "cost_usd": 0.0028 }`

### Sessions (multi-turn)
- `POST /api/sessions` — create a new chat session, returns `session_id`
- `GET /api/sessions/{id}` — get a session's conversation history
- `DELETE /api/sessions/{id}` — end a session

### Analytics
- `GET /api/queries` — list past queries with answers
- `GET /api/stats` — system stats (document count, chunk count, p50/p95 latency, cache hit rate, tool-call frequency by tool, total spend)
- `POST /api/queries/{id}/rate` — rate an answer (1–5)

## Dashboard views

### Chat (main)
- Message-bubble chat with streaming responses
- Expandable "Sources" section under each answer
- Live tool-call indicator (`Calling tool…` while in flight, completed status block with args + result after)
- Per-message footer: latency / first-token / cost / iterations / token counts / cache hit rate

### Knowledge base browser
- All indexed documents grouped by source type
- Per-document chunk count
- Re-ingest button
- Per-document delete

### Analytics dashboard
- Top-line metrics (query count, average / p50 / p95 latency, total spend)
- Tool-call frequency bar chart
- Recent queries table

## Quantifiable metrics

### System
- Documents indexed: ~26 (sample pack); scales to thousands
- Chunks indexed: ~64 (sample pack); ~700 in private benchmark
- First-token latency: <500 ms (streaming)
- Full-response latency: p95 <3s for the generation step; pgvector lookup is sub-millisecond at this scale
- Retrieval precision@5: see README for the live benchmark numbers

### Engineering / cost
- Embedding dimensions: 384 (all-MiniLM-L6-v2)
- Hybrid retrieval alpha: tuned per-domain (sweep over `{0.3, 0.5, 0.7}`)
- Reranker precision lift: +21pp MRR on benchmark (cross-encoder vs hybrid-only)
- Prompt-cache hit rate: >70% target; reduces input-token cost by ~90% on cached prefix
- Average cost per query: target <$0.005

## Non-goals

- No fine-tuning of models — pre-trained embeddings + Claude API
- No user authentication (single-user; rate-limited public endpoint for the demo)
- No real-time document syncing (re-ingest is manual)
- No multi-tenant support

## Success criteria

- Answers cite the correct source document(s)
- Retrieval recall@5 > 0.80, MRR > 0.85 on the domain eval set
- First-token latency < 500 ms; full response p95 < 3 s for the generation step
- `docker compose up` brings the entire stack live with zero manual setup
- Test coverage > 80% for the core pipeline
