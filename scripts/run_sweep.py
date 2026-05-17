"""Sweep retrieval hyperparameters and pick a winner.

Iterates a 3x3x2 = 18-cell grid (alpha x candidate_k x rerank-on/off),
runs the retrieval eval at each cell, and emits a comparison report
plus a JSON dump of the full grid. The rerank A/B falls out of the same
grid: pick the best (alpha, candidate_k) cell with rerank on, compare to
the same cell with rerank off — that's the "rerank earns its complexity"
delta.

Usage:
    python scripts/run_sweep.py                                  # active domain
    python scripts/run_sweep.py --domain sample --reset          # rebuild index first
    python scripts/run_sweep.py --alphas 0.3,0.5,0.7,0.9         # custom grid

Output:
    Writes results/sweep_<domain>_<timestamp>.json with one row per cell:
        {alpha, candidate_k, use_rerank, recall@5, recall@10, mrr, p50_ms, p95_ms}
    Plus the picked winner and the rerank A/B summary.

The sweep is retrieval-only — answer/tool eval is orthogonal to retrieval
tuning and would multiply runtime by ~30x without changing which cell wins.
Re-run `run_eval.py --mode all` separately with the winning config.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the repo importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from knowledge_rag.domain import get_domain
from knowledge_rag.embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from knowledge_rag.eval_pipeline import (
    DEFAULT_RERANKER_MODEL,
    TimedRetriever,
    build_retriever,
    ensure_index,
    latency_summary,
)
from knowledge_rag.evaluation import evaluate_retriever
from knowledge_rag.reranker import CrossEncoderReranker
from knowledge_rag.vectorstore import DEFAULT_DSN

DEFAULT_ALPHAS = (0.3, 0.5, 0.7)
DEFAULT_CANDIDATE_KS = (10, 20, 30)
DEFAULT_RERANK_VALUES = (True, False)


@dataclass
class SweepCell:
    """One row of the sweep grid: hyperparams + measured aggregates."""

    alpha: float
    candidate_k: int
    use_rerank: bool
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    p50_ms: float
    p95_ms: float
    n_questions: int
    seconds: float


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default=None,
        help="Domain pack (default: $KNOWLEDGE_DOMAIN or 'sample').",
    )
    parser.add_argument(
        "--alphas",
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help=f"Comma-separated alpha values (default '{','.join(str(a) for a in DEFAULT_ALPHAS)}').",
    )
    parser.add_argument(
        "--candidate-ks",
        default=",".join(str(k) for k in DEFAULT_CANDIDATE_KS),
        help=f"Comma-separated candidate_k values (default '{','.join(str(k) for k in DEFAULT_CANDIDATE_KS)}').",
    )
    parser.add_argument(
        "--rerank",
        default="on,off",
        help="Comma-separated subset of {on,off} (default 'on,off').",
    )
    parser.add_argument(
        "--k-values",
        default="5,10",
        help="Comma-separated k values for recall@k (default '5,10').",
    )
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--table", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


# --- Grid expansion ----------------------------------------------------------


def expand_grid(
    alphas: list[float], candidate_ks: list[int], reranks: list[bool]
) -> list[tuple[float, int, bool]]:
    """All combinations of (alpha, candidate_k, use_rerank). Stable order."""
    return list(itertools.product(alphas, candidate_ks, reranks))


def parse_rerank_flag(raw: str) -> list[bool]:
    """'on,off' -> [True, False]. Case-insensitive; rejects unknown values."""
    out: list[bool] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if tok == "on":
            out.append(True)
        elif tok == "off":
            out.append(False)
        elif tok:
            raise ValueError(f"Unknown rerank value {tok!r}; use 'on' or 'off'.")
    return out or [True, False]


# --- Cell runner -------------------------------------------------------------


def run_cell(
    *,
    store: Any,
    bm25: Any,
    dataset: list[dict],
    alpha: float,
    candidate_k: int,
    use_rerank: bool,
    k_values: tuple[int, ...],
    reranker: CrossEncoderReranker | None = None,
) -> SweepCell:
    """Build a retriever for this cell, run the eval, capture metrics.

    `reranker` is the shared instance loaded once at sweep startup — passing it
    in avoids re-loading ~200MB of model weights for every rerank-on cell.
    Ignored when `use_rerank=False`.
    """
    inner = build_retriever(
        store,
        bm25,
        alpha=alpha,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
        reranker=reranker if use_rerank else None,
    )
    timed = TimedRetriever(inner)

    t0 = time.perf_counter()
    report = evaluate_retriever(timed, dataset, k_values=k_values)
    elapsed = time.perf_counter() - t0
    lat = latency_summary(timed.latencies_ms)

    return SweepCell(
        alpha=alpha,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
        recall_at_5=report.aggregate.mean_recall_at_k.get(5, 0.0),
        recall_at_10=report.aggregate.mean_recall_at_k.get(10, 0.0),
        mrr=report.aggregate.mrr,
        precision_at_5=report.aggregate.mean_precision_at_k.get(5, 0.0),
        p50_ms=lat["p50_ms"],
        p95_ms=lat["p95_ms"],
        n_questions=report.aggregate.n_questions,
        seconds=round(elapsed, 2),
    )


# --- Winner + A/B ------------------------------------------------------------


def pick_winner(cells: list[SweepCell]) -> SweepCell:
    """Highest MRR; ties broken by lowest p95 latency."""
    return max(cells, key=lambda c: (c.mrr, -c.p95_ms))


def rerank_ab(cells: list[SweepCell]) -> dict[str, Any] | None:
    """Pick best rerank-on cell; report delta vs same (alpha, candidate_k) rerank-off.

    Returns None if either side is missing from the grid.
    """
    on_cells = [c for c in cells if c.use_rerank]
    off_cells = [c for c in cells if not c.use_rerank]
    if not on_cells or not off_cells:
        return None
    best_on = max(on_cells, key=lambda c: c.mrr)
    matched_off = next(
        (c for c in off_cells if c.alpha == best_on.alpha and c.candidate_k == best_on.candidate_k),
        None,
    )
    if matched_off is None:
        return None
    return {
        "alpha": best_on.alpha,
        "candidate_k": best_on.candidate_k,
        "with_rerank": {
            "recall_at_5": best_on.recall_at_5,
            "mrr": best_on.mrr,
            "p95_ms": best_on.p95_ms,
        },
        "without_rerank": {
            "recall_at_5": matched_off.recall_at_5,
            "mrr": matched_off.mrr,
            "p95_ms": matched_off.p95_ms,
        },
        "delta": {
            "recall_at_5": round(best_on.recall_at_5 - matched_off.recall_at_5, 4),
            "mrr": round(best_on.mrr - matched_off.mrr, 4),
            "p95_ms": round(best_on.p95_ms - matched_off.p95_ms, 2),
        },
    }


# --- Printing ----------------------------------------------------------------


def print_grid(cells: list[SweepCell], winner: SweepCell) -> None:
    print(
        f"{'alpha':>6} {'cand_k':>7} {'rerank':>7} {'recall@5':>9} {'recall@10':>10} {'MRR':>6} {'p95(ms)':>9} {'sec':>6}"
    )
    print("-" * 70)
    for c in cells:
        marker = (
            " *"
            if (
                c.alpha == winner.alpha
                and c.candidate_k == winner.candidate_k
                and c.use_rerank == winner.use_rerank
            )
            else ""
        )
        print(
            f"{c.alpha:>6.2f} {c.candidate_k:>7d} {('on' if c.use_rerank else 'off'):>7} "
            f"{c.recall_at_5:>9.3f} {c.recall_at_10:>10.3f} {c.mrr:>6.3f} "
            f"{c.p95_ms:>9.1f} {c.seconds:>6.1f}{marker}"
        )


# --- Main --------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    domain_name = args.domain or os.environ.get("KNOWLEDGE_DOMAIN", "sample")
    dsn = args.dsn or os.environ.get("KNOWLEDGE_DB_DSN", DEFAULT_DSN)
    table = args.table or f"eval_{domain_name}"

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    candidate_ks = [int(x) for x in args.candidate_ks.split(",") if x.strip()]
    reranks = parse_rerank_flag(args.rerank)
    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())

    domain = get_domain(domain_name)
    dataset = domain.eval_dataset()
    if not dataset:
        print(f"ERROR: no eval dataset for domain '{domain_name}'.", file=sys.stderr)
        return 2

    n_cells = len(alphas) * len(candidate_ks) * len(reranks)
    print(
        f"Sweep: {len(alphas)} x {len(candidate_ks)} x {len(reranks)} = "
        f"{n_cells} cells on domain '{domain_name}'"
    )
    print(f"Building eval index in table {table!r}...")
    t0 = time.perf_counter()
    store, bm25, corpus_stats = ensure_index(domain_name, dsn, table, reset=args.reset)
    print(
        f"  -> {corpus_stats.n_documents} docs, {corpus_stats.n_chunks} chunks ({time.perf_counter() - t0:.1f}s)"
    )

    # Load the cross-encoder once and share it across every rerank-on cell.
    # The model is ~200MB; reloading per cell would add ~5–30s × #rerank cells.
    shared_reranker = (
        CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL) if any(reranks) else None
    )

    cells: list[SweepCell] = []
    sweep_t0 = time.perf_counter()
    for i, (alpha, candidate_k, use_rerank) in enumerate(
        expand_grid(alphas, candidate_ks, reranks), start=1
    ):
        print(
            f"  [{i}/{n_cells}] "
            f"alpha={alpha} candidate_k={candidate_k} rerank={'on' if use_rerank else 'off'} ...",
            end=" ",
            flush=True,
        )
        cell = run_cell(
            store=store,
            bm25=bm25,
            dataset=dataset,
            alpha=alpha,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
            k_values=k_values,
            reranker=shared_reranker,
        )
        cells.append(cell)
        print(f"MRR={cell.mrr:.3f} recall@5={cell.recall_at_5:.3f} ({cell.seconds:.1f}s)")

    sweep_elapsed = time.perf_counter() - sweep_t0
    winner = pick_winner(cells)
    ab = rerank_ab(cells)

    timestamp = datetime.now(UTC).isoformat()
    payload = {
        "timestamp": timestamp,
        "domain": domain_name,
        "config": {
            "alphas": alphas,
            "candidate_ks": candidate_ks,
            "reranks": reranks,
            "k_values": list(k_values),
            "vector_store": "pgvector",
            "vector_store_table": table,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": DEFAULT_RERANKER_MODEL,
        },
        "corpus": asdict(corpus_stats),
        "cells": [asdict(c) for c in cells],
        "winner": asdict(winner),
        "rerank_ab": ab,
        "total_seconds": round(sweep_elapsed, 1),
    }

    timestamp_slug = timestamp.replace(":", "").replace("-", "").split(".")[0]
    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"sweep_{domain_name}_{timestamp_slug}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- Print summary ---
    print()
    print("=" * 70)
    print(f"Sweep results ({domain_name}, {len(cells)} cells, {sweep_elapsed:.1f}s)")
    print("=" * 70)
    print_grid(cells, winner)
    print("-" * 70)
    print(
        f"Winner: alpha={winner.alpha} candidate_k={winner.candidate_k} "
        f"rerank={'on' if winner.use_rerank else 'off'} "
        f"-> MRR={winner.mrr:.3f} recall@5={winner.recall_at_5:.3f} "
        f"recall@10={winner.recall_at_10:.3f} p95={winner.p95_ms:.0f}ms"
    )
    if ab:
        d = ab["delta"]
        sign_r = "+" if d["recall_at_5"] >= 0 else ""
        sign_m = "+" if d["mrr"] >= 0 else ""
        sign_p = "+" if d["p95_ms"] >= 0 else ""
        print(
            f"Rerank A/B (alpha={ab['alpha']}, candidate_k={ab['candidate_k']}): "
            f"recall@5 {sign_r}{d['recall_at_5']:.3f}  "
            f"MRR {sign_m}{d['mrr']:.3f}  "
            f"p95 {sign_p}{d['p95_ms']:.0f}ms"
        )
    print(f"Wrote: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
