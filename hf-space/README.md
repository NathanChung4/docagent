---
title: DocAgent
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Retrieval-augmented agent — streaming, grounded, tool-use
---

# DocAgent — Live Demo

A retrieval-augmented agent for technical documentation: streams grounded answers with source citations, dispatches validated tools to produce real artifacts, ships with a domain-pluggable architecture.

This Space runs the full stack — pgvector Postgres + FastAPI + Streamlit — in a single container against a small sample corpus. The same image can be pointed at any domain by swapping the `KNOWLEDGE_DOMAIN` env var.

**Source:** [github.com/NathanChung4/docagent](https://github.com/NathanChung4/docagent)

## What you can try

- **Q&A:** "What are the input parameters for the clock divider?" — watch tokens stream in with citations.
- **Tool use:** "Generate a config file for the flow controller with thresh=5" — the agent dispatches the `generate_config_file` tool, the result drops into chat as a downloadable artifact.
- **Status lookup:** "What's the status of the clock divider feature?" — structured lookup tool returns owner + state from the checklist.
- **Browse the index:** sidebar nav → "Knowledge base" shows the ingested chunks; "Analytics" shows cost/latency per query.

## Notes

- Cold start takes ~60s while the corpus re-ingests (HF Spaces' disk is ephemeral). Subsequent queries return in <1s.
- Set the `ANTHROPIC_API_KEY` secret in the Space settings before first use — generation and tool dispatch both require it. Retrieval-only paths (Knowledge base browse) work without it.
- Built from `main` of the source repo. Restart the Space to pick up new commits.
