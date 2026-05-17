"""Analytics page — query stats, latency, cost, tool-call frequency.

Stats come from GET /api/stats (a single Postgres aggregation: percentile_cont
for p50/p95, JSONB unnest for tool-call counts). Recent queries come from
GET /api/queries.

Charts use st.bar_chart for tool frequency. Recent queries render as a
dataframe — st.column_config formats cost/latency cleanly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from knowledge_rag.ui import get_client
from knowledge_rag.ui.api_client import APIClient

st.set_page_config(page_title="Analytics — Knowledge RAG", layout="wide")
st.title("Analytics")

client: APIClient = get_client()


# Cache `stats` separately from `list_queries` so dragging the recent-queries
# slider only re-fetches the queries — not the aggregate stats above.
@st.cache_data(ttl=30)
def _load_stats() -> dict:
    return get_client().stats()


try:
    stats = _load_stats()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load stats: {exc}")
    st.stop()

if stats["query_count"] == 0:
    st.info("No queries logged yet. Ask something on the Chat page.")
    st.stop()


# --- top-line metrics ------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Queries", f"{stats['query_count']:,}")
c2.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
c3.metric("Total spend", f"${stats['total_cost_usd']:.4f}")
c4.metric("Cache hit rate", f"{stats['cache_hit_rate'] * 100:.1f}%")

c5, c6, c7 = st.columns(3)
c5.metric("p50 latency", f"{stats['p50_latency_ms']:.0f} ms")
c6.metric("p95 latency", f"{stats['p95_latency_ms']:.0f} ms")
c7.metric("Avg first-token", f"{stats['avg_first_token_ms']:.0f} ms")

st.divider()


# --- tool-call frequency ---------------------------------------------------

st.subheader("Tool-call frequency")
tool_counts = stats.get("tool_call_counts") or {}
if tool_counts:
    df_tools = pd.DataFrame(
        sorted(tool_counts.items(), key=lambda kv: kv[1], reverse=True),
        columns=["tool", "calls"],
    ).set_index("tool")
    st.bar_chart(df_tools)
else:
    st.caption("No tool calls logged yet.")

st.divider()


# --- recent queries --------------------------------------------------------

st.subheader("Recent queries")
limit = st.slider("Show last N queries", min_value=10, max_value=200, value=50, step=10)
try:
    queries = client.list_queries(limit=limit)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load queries: {exc}")
    st.stop()

if not queries:
    st.caption("No queries to show.")
else:
    rows = []
    for q in queries:
        rows.append(
            {
                "timestamp": q["timestamp"],
                "query": q["query"],
                "tools_used": ", ".join(tc["tool_name"] for tc in q.get("tool_calls", [])) or "—",
                "latency_ms": q["latency_ms"],
                "first_token_ms": q["first_token_ms"],
                "cost_usd": q["cost_usd"],
                "input_tokens": q["input_tokens"],
                "output_tokens": q["output_tokens"],
                "cached_tokens": q["cached_tokens"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss"),
            "query": st.column_config.TextColumn("Query", width="large"),
            "tools_used": st.column_config.TextColumn("Tools"),
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
            "first_token_ms": st.column_config.NumberColumn("First token (ms)", format="%.0f"),
            "cost_usd": st.column_config.NumberColumn("Cost", format="$%.4f"),
            "input_tokens": st.column_config.NumberColumn("In tok"),
            "output_tokens": st.column_config.NumberColumn("Out tok"),
            "cached_tokens": st.column_config.NumberColumn("Cached tok"),
        },
    )
