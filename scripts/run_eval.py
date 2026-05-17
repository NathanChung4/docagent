"""Run the eval suite against the active domain.

Usage:
    python scripts/run_eval.py                                # retrieval only (default)
    python scripts/run_eval.py --mode all                     # retrieval + answer + tool
    python scripts/run_eval.py --mode answer                  # answer judge only
    python scripts/run_eval.py --mode tool                    # tool-call accuracy only
    python scripts/run_eval.py --domain sample
    python scripts/run_eval.py --no-rerank                    # ablation: rerank off
    python scripts/run_eval.py --alpha 1.0                    # ablation: semantic only
    python scripts/run_eval.py --reset                        # force re-ingest
    python scripts/run_eval.py --compare-with results/eval_sample_<earlier-ts>.json

Output:
    Writes a self-contained JSON to `results/eval_<domain>_<timestamp>.json`
    capturing the config snapshot, corpus stats, per-question metrics, query
    latency percentiles, and macro-averaged aggregates. When --mode is "answer"
    or "all", the JSON also has an "answer_eval" block (judge verdicts + cost).
    When --mode is "tool" or "all", it also has a "tool_eval" block.
    Prints a one-screen summary to stdout.

Phase 4.7 added pgvector + per-query latency tracking. Phase 9 added the
--mode flag for answer + tool eval. The output JSON is the contract that
the sweep runner and report builder consume — don't add fields without
thinking about what those scripts would do with them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the repo importable when run as a script (mirrors tests/conftest.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from knowledge_rag.answer_eval import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PASS_THRESHOLD,
    AnswerEvalReport,
    evaluate_answers,
)
from knowledge_rag.domain import get_domain
from knowledge_rag.embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from knowledge_rag.eval_pipeline import (
    DEFAULT_RERANKER_MODEL,
    TimedRetriever,
    build_retriever,
    ensure_index,
    latency_summary,
)
from knowledge_rag.evaluation import EvalReport, evaluate_retriever
from knowledge_rag.generation import Generator
from knowledge_rag.retrieval import DEFAULT_ALPHA, DEFAULT_CANDIDATE_K
from knowledge_rag.tool_eval import ToolEvalReport, evaluate_tool_calls
from knowledge_rag.vectorstore import DEFAULT_DSN

VALID_MODES = ("retrieval", "answer", "tool", "all")


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default=None,
        help="Domain pack to evaluate (default: $KNOWLEDGE_DOMAIN or 'sample').",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="retrieval",
        help=(
            "What to evaluate: retrieval (default; offline), answer (LLM-as-judge "
            "over generated answers), tool (agent tool-call accuracy), or all."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Hybrid blend weight; 1.0=semantic-only, 0.0=BM25-only (default {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help=f"Candidates fetched from each leg before blending (default {DEFAULT_CANDIDATE_K}).",
    )
    parser.add_argument(
        "--k-values",
        default="5,10",
        help="Comma-separated k values for recall@k and precision@k (default '5,10').",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the cross-encoder reranker (for the rerank A/B test).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model for answer eval (default {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help=f"Judge pass threshold (default {DEFAULT_PASS_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the answer judge even in --mode answer/all (useful for offline tests).",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help=(
            "Postgres DSN (default: $KNOWLEDGE_DB_DSN, then "
            "postgresql://postgres:postgres@localhost:5432/postgres). "
            "Bring up the local server with `docker compose up -d postgres`."
        ),
    )
    parser.add_argument(
        "--table",
        default=None,
        help="pgvector table name (default: eval_<domain>).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the eval table; re-ingest before running.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/eval_<domain>_<timestamp>.json).",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help="Path to a prior results JSON to print a side-by-side delta against.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-question results to stdout in addition to writing JSON.",
    )
    return parser.parse_args()


# --- Result serialization ----------------------------------------------------


def _stringify_int_keys(d: dict[int, float]) -> dict[str, float]:
    """JSON object keys must be strings — convert {5: 0.7} to {"5": 0.7}."""
    return {str(k): v for k, v in d.items()}


def _retrieval_report_to_dict(report: EvalReport) -> dict:
    """Serialize a retrieval EvalReport to a JSON-safe dict."""
    per_q = []
    for qr in report.per_question:
        d = asdict(qr)
        d["recall_at_k"] = _stringify_int_keys(qr.recall_at_k)
        d["precision_at_k"] = _stringify_int_keys(qr.precision_at_k)
        per_q.append(d)
    agg = asdict(report.aggregate)
    agg["mean_recall_at_k"] = _stringify_int_keys(report.aggregate.mean_recall_at_k)
    agg["mean_precision_at_k"] = _stringify_int_keys(report.aggregate.mean_precision_at_k)
    return {
        "k_values": list(report.k_values),
        "aggregate": agg,
        "per_question": per_q,
        "skipped": list(report.skipped),
    }


def _answer_report_to_dict(report: AnswerEvalReport) -> dict:
    return {
        "pass_rate": report.pass_rate,
        "mean_score": report.mean_score,
        "total_cost_usd": report.total_cost_usd,
        "n_parse_errors": report.n_parse_errors,
        "per_question": [asdict(v) for v in report.per_question],
    }


def _tool_report_to_dict(report: ToolEvalReport) -> dict:
    return {
        "n_evaluated": report.n_evaluated,
        "tool_choice_accuracy": report.tool_choice_accuracy,
        "mean_args_match": report.mean_args_match,
        "full_match_rate": report.full_match_rate,
        "validation_error_accuracy": report.validation_error_accuracy,
        "per_question": [asdict(v) for v in report.per_question],
    }


# --- Mode runners ------------------------------------------------------------


def _run_answer_mode(
    *,
    dataset: list[dict],
    retriever: TimedRetriever,
    judge_model: str,
    pass_threshold: float,
    no_judge: bool,
) -> tuple[AnswerEvalReport | None, list[dict]]:
    """Generate answers for every Q&A entry, then grade them.

    Returns (report, generated_pairs). `report` is None when `no_judge=True` —
    the caller then writes the generated answers but skips the answer_eval block.
    """
    generator = Generator()
    answer_pairs: list[tuple[dict, str]] = []
    serialized_pairs: list[dict] = []

    qa_entries = [e for e in dataset if e.get("kind", "qa") == "qa"]
    for entry in qa_entries:
        results = retriever.retrieve(entry["question"], k=5)
        gen_result = generator.generate(entry["question"], results)
        answer_pairs.append((entry, gen_result.answer))
        serialized_pairs.append(
            {
                "id": entry.get("id"),
                "question": entry["question"],
                "answer": gen_result.answer,
                "cost_usd": gen_result.query_log.cost_usd,
                "input_tokens": gen_result.query_log.input_tokens,
                "output_tokens": gen_result.query_log.output_tokens,
                "cached_tokens": gen_result.query_log.cached_tokens,
            }
        )

    if no_judge:
        return None, serialized_pairs

    report = evaluate_answers(answer_pairs, model=judge_model, pass_threshold=pass_threshold)
    return report, serialized_pairs


def _run_tool_mode(
    *,
    dataset: list[dict],
    retriever: TimedRetriever,
    domain_name: str,
) -> tuple[ToolEvalReport, list[dict]]:
    """Run the agent on every tool entry, then evaluate tool-call accuracy."""
    # Imported here so retrieval-only runs don't pay the import cost.
    from knowledge_rag.agent import Agent

    domain = get_domain(domain_name)
    agent = Agent(tools=domain.tools())
    pairs: list[tuple[dict, Any]] = []
    serialized: list[dict] = []

    tool_entries = [e for e in dataset if e.get("kind") == "tool"]
    for entry in tool_entries:
        results = retriever.retrieve(entry["question"], k=5)
        agent_result = agent.run(entry["question"], context=results)
        pairs.append((entry, agent_result))
        serialized.append(
            {
                "id": entry.get("id"),
                "question": entry["question"],
                "answer": agent_result.answer,
                "tool_calls": [asdict(tc) for tc in agent_result.query_log.tool_calls],
                "iterations": agent_result.iterations,
                "cost_usd": agent_result.query_log.cost_usd,
            }
        )

    return evaluate_tool_calls(pairs), serialized


# --- Summary printing --------------------------------------------------------


def _print_retrieval_summary(report: EvalReport, *, latency: dict) -> None:
    agg = report.aggregate
    print("-" * 60)
    print(f"MRR:          {agg.mrr:.3f}")
    for k in report.k_values:
        print(
            f"recall@{k:<3}    {agg.mean_recall_at_k[k]:.3f}"
            f"    precision@{k:<3} {agg.mean_precision_at_k[k]:.3f}"
        )
    print(
        f"Latency:      p50={latency['p50_ms']:.1f}ms  "
        f"p95={latency['p95_ms']:.1f}ms  mean={latency['mean_ms']:.1f}ms  "
        f"(n={latency['n_queries']})"
    )


def _print_answer_summary(report: AnswerEvalReport) -> None:
    print("-" * 60)
    print(
        f"Answer eval:  pass_rate={report.pass_rate:.3f}  "
        f"mean_score={report.mean_score:.3f}  "
        f"judge_cost=${report.total_cost_usd:.4f}  "
        f"parse_errors={report.n_parse_errors}"
    )


def _print_tool_summary(report: ToolEvalReport) -> None:
    print("-" * 60)
    print(
        f"Tool eval:    choice_acc={report.tool_choice_accuracy:.3f}  "
        f"args_match={report.mean_args_match:.3f}  "
        f"full_match={report.full_match_rate:.3f}  "
        f"val_err_acc={report.validation_error_accuracy:.3f}  "
        f"(n={report.n_evaluated})"
    )


def _walk(d: dict, path: tuple[str, ...]) -> Any:
    """Safely walk a nested dict by a tuple path. Returns None on any miss."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _print_compare(prior_path: Path, current_payload: dict) -> None:
    """Side-by-side delta of MRR / recall@5 / recall@10 vs a prior run."""
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARN: could not load comparison file {prior_path}: {e}", file=sys.stderr)
        return
    prior_agg = prior.get("aggregate", {})
    cur_agg = current_payload.get("aggregate", {})
    if not prior_agg or not cur_agg:
        print("WARN: comparison aggregate missing from one of the files.", file=sys.stderr)
        return
    print("-" * 60)
    print(f"Comparison vs {prior_path.name}:")
    rows = [
        ("MRR", ("mrr",)),
        ("recall@5", ("mean_recall_at_k", "5")),
        ("recall@10", ("mean_recall_at_k", "10")),
    ]
    for label, path in rows:
        cur = _walk(cur_agg, path)
        old = _walk(prior_agg, path)
        if cur is None or old is None:
            continue
        delta = cur - old
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<10} {old:.3f} -> {cur:.3f}   ({sign}{delta:.3f})")


# --- Main --------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    domain_name = args.domain or os.environ.get("KNOWLEDGE_DOMAIN", "sample")
    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    use_rerank = not args.no_rerank
    dsn = args.dsn or os.environ.get("KNOWLEDGE_DB_DSN", DEFAULT_DSN)
    table = args.table or f"eval_{domain_name}"

    domain = get_domain(domain_name)
    dataset = domain.eval_dataset()
    if not dataset:
        print(
            f"ERROR: no eval dataset for domain '{domain_name}'. "
            f"Populate domains/{domain_name}/eval_dataset.json first.",
            file=sys.stderr,
        )
        return 2

    print(f"Building eval index for domain '{domain_name}' in table {table!r}...")
    t0 = time.perf_counter()
    store, bm25, corpus_stats = ensure_index(domain_name, dsn, table, reset=args.reset)
    t_index = time.perf_counter() - t0
    print(f"  -> {corpus_stats.n_documents} docs, {corpus_stats.n_chunks} chunks ({t_index:.1f}s)")

    inner = build_retriever(
        store,
        bm25,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        use_rerank=use_rerank,
    )
    retriever = TimedRetriever(inner)

    timestamp = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "domain": domain_name,
        "mode": args.mode,
        "config": {
            "alpha": args.alpha,
            "candidate_k": args.candidate_k,
            "k_values": list(k_values),
            "use_rerank": use_rerank,
            "vector_store": "pgvector",
            "vector_store_table": table,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": DEFAULT_RERANKER_MODEL if use_rerank else None,
            "judge_model": args.judge_model
            if args.mode in ("answer", "all") and not args.no_judge
            else None,
        },
        "corpus": asdict(corpus_stats),
    }

    # --- Retrieval (always — answer/tool modes also need the retriever warmed) ---
    print(f"Running retrieval eval over {len(dataset)} dataset entries...")
    t0 = time.perf_counter()
    retrieval_report = evaluate_retriever(retriever, dataset, k_values=k_values)
    t_retrieval = time.perf_counter() - t0
    payload["timing_s"] = {"index_or_reuse": round(t_index, 3), "retrieval": round(t_retrieval, 3)}
    payload["latency_ms"] = latency_summary(retriever.latencies_ms)
    payload.update(_retrieval_report_to_dict(retrieval_report))

    # --- Answer mode ---
    ans_report: AnswerEvalReport | None = None
    if args.mode in ("answer", "all"):
        print(f"Running answer eval (judge={args.judge_model})...")
        t0 = time.perf_counter()
        ans_report, generated = _run_answer_mode(
            dataset=dataset,
            retriever=retriever,
            judge_model=args.judge_model,
            pass_threshold=args.pass_threshold,
            no_judge=args.no_judge,
        )
        payload["timing_s"]["answer"] = round(time.perf_counter() - t0, 3)
        if ans_report is not None:
            payload["answer_eval"] = _answer_report_to_dict(ans_report)
        payload["generated_answers"] = generated

    # --- Tool mode ---
    tool_report: ToolEvalReport | None = None
    if args.mode in ("tool", "all"):
        print("Running tool-call eval...")
        t0 = time.perf_counter()
        tool_report, agent_traces = _run_tool_mode(
            dataset=dataset,
            retriever=retriever,
            domain_name=domain_name,
        )
        payload["timing_s"]["tool"] = round(time.perf_counter() - t0, 3)
        payload["tool_eval"] = _tool_report_to_dict(tool_report)
        payload["agent_traces"] = agent_traces

    # --- Write JSON ---
    timestamp_slug = timestamp.replace(":", "").replace("-", "").split(".")[0]
    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"eval_{domain_name}_{timestamp_slug}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- Print summary ---
    print("=" * 60)
    print(f"Eval Run @ {timestamp}")
    print("=" * 60)
    print(f"Domain:       {domain_name}")
    print(f"Mode:         {args.mode}")
    print(f"Vector store: pgvector (table {table!r})")
    print(f"Embedder:     {EMBEDDING_MODEL}")
    print(f"Reranker:     {DEFAULT_RERANKER_MODEL if use_rerank else '(disabled)'}")
    print(f"Hybrid:       alpha={args.alpha}, candidate_k={args.candidate_k}")
    print(f"Corpus:       {corpus_stats.n_documents} documents, {corpus_stats.n_chunks} chunks")
    print(
        f"Eval set:     {retrieval_report.aggregate.n_questions} questions"
        + (f", {len(retrieval_report.skipped)} skipped" if retrieval_report.skipped else "")
    )
    _print_retrieval_summary(retrieval_report, latency=payload["latency_ms"])
    if ans_report is not None:
        _print_answer_summary(ans_report)
    if tool_report is not None:
        _print_tool_summary(tool_report)
    if args.compare_with:
        _print_compare(Path(args.compare_with), payload)
    print("=" * 60)
    print(f"Wrote: {output_path}")

    if args.verbose:
        print("\nPer-question (retrieval):")
        for qr in payload["per_question"]:
            print(
                f"  [{qr['question_id']}] RR={qr['reciprocal_rank']:.2f}  "
                f"recall@5={qr['recall_at_k']['5']:.2f}  "
                f"q={qr['question'][:60]}..."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
