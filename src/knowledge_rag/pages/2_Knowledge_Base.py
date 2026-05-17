"""Knowledge Base page — list documents, re-ingest, delete.

Documents come from GET /api/documents — one row per parent doc with
chunk count, source type, and uri. Re-ingest hits POST /api/ingest which
re-runs loaders → chunker → vectorstore upsert → BM25 rebuild on the
backend. Delete hits DELETE /api/documents/{id} which removes chunks
from pgvector and rebuilds BM25.
"""

from __future__ import annotations

import streamlit as st

from knowledge_rag.ui import get_client
from knowledge_rag.ui.api_client import APIClient

st.set_page_config(page_title="Knowledge Base — Knowledge RAG", layout="wide")
st.title("Knowledge Base")

client: APIClient = get_client()


# --- ingest control --------------------------------------------------------

with st.sidebar:
    st.markdown("### Ingestion")
    if st.button("Re-ingest all sources", type="primary"):
        with st.spinner("Re-running loaders → chunker → vectorstore → BM25…"):
            try:
                result = client.ingest()
                st.success(f"Indexed {result['documents']} documents ({result['chunks']} chunks).")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingest failed: {exc}")


# --- document table --------------------------------------------------------

try:
    docs = client.list_documents()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load documents: {exc}")
    st.stop()

if not docs:
    st.info("No documents indexed. Click **Re-ingest all sources** to load them.")
    st.stop()

st.write(f"**{len(docs)} document(s)** — total {sum(d['chunk_count'] for d in docs)} chunks")

# Group counts by source_type for a quick overview.
type_counts: dict[str, int] = {}
for d in docs:
    type_counts[d.get("source_type") or "unknown"] = (
        type_counts.get(d.get("source_type") or "unknown", 0) + 1
    )
cols = st.columns(len(type_counts))
for col, (src_type, count) in zip(cols, sorted(type_counts.items()), strict=True):
    col.metric(src_type, count)

st.divider()

for doc in docs:
    with st.container(border=True):
        head, action = st.columns([6, 1])
        with head:
            st.markdown(f"**{doc.get('title') or doc['doc_id']}**")
            meta_parts = [
                f"`{doc.get('source_type', '?')}`",
                f"{doc['chunk_count']} chunks",
                f"`{doc['doc_id']}`",
            ]
            if doc.get("uri"):
                meta_parts.append(doc["uri"])
            st.caption(" · ".join(meta_parts))
        with action:
            if st.button("Delete", key=f"del_{doc['doc_id']}"):
                try:
                    client.delete_document(doc["doc_id"])
                    st.toast(f"Deleted {doc['doc_id']}")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Delete failed: {exc}")
