"""Build a Markdown + HTML eval report from `run_eval.py` and `run_sweep.py` outputs.

Usage:
    python scripts/build_report.py \\
        --eval results/eval_sample_<ts>.json \\
        --sweep results/sweep_sample_<ts>.json \\
        --baseline results/eval_sample_<earlier-ts>.json \\
        --output results/report_sample_<ts>

The report assembles whichever inputs are provided:
  --eval     latest eval JSON (any --mode); answer/tool sections appear iff present
  --sweep    sweep JSON; renders heatmap + rerank A/B
  --baseline second eval JSON to diff against (delta in headline metrics)

Outputs:
  <output>.md   — Markdown with embedded PNG references (renders on GitHub)
  <output>.html — self-contained Plotly HTML (open in browser)
  <output>_charts/ — PNG charts referenced by the .md file

Charts:
  1. Sweep heatmap (recall@5 over alpha x candidate_k, faceted by rerank on/off)
  2. Rerank A/B bar (with vs without, on best cell)
  3. Per-question reciprocal rank, sorted, with regression highlights vs baseline
  4. Answer eval pass-rate summary (if answer_eval present)
  5. Tool eval bars (if tool_eval present)
  6. Cost breakdown (judge cost vs query cost, if present)

This module is domain-agnostic — feed it any domain's JSONs, it doesn't care.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the repo importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, help="Path to eval JSON (run_eval.py output).")
    parser.add_argument(
        "--sweep", default=None, help="Optional path to sweep JSON (run_sweep.py output)."
    )
    parser.add_argument(
        "--baseline", default=None, help="Optional prior eval JSON to compute deltas against."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output stem (without extension); writes <stem>.md, <stem>.html, <stem>_charts/.",
    )
    return parser.parse_args()


# --- IO helpers --------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs(stem: Path) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    charts_dir = stem.parent / f"{stem.name}_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


# --- Chart builders (matplotlib for PNG, plotly for HTML) -------------------


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_plotly():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


def chart_sweep_heatmap_png(sweep: dict, out_path: Path) -> Path:
    """Two side-by-side heatmaps (rerank on / off) of recall@5 over (alpha, candidate_k)."""
    plt = _import_matplotlib()
    cells = sweep["cells"]
    alphas = sorted({c["alpha"] for c in cells})
    cks = sorted({c["candidate_k"] for c in cells})

    def _grid(rerank_value: bool) -> list[list[float]]:
        return [
            [
                next(
                    (
                        c["recall_at_5"]
                        for c in cells
                        if c["alpha"] == a
                        and c["candidate_k"] == ck
                        and c["use_rerank"] == rerank_value
                    ),
                    float("nan"),
                )
                for ck in cks
            ]
            for a in alphas
        ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rerank_value, title in [(axes[0], True, "rerank ON"), (axes[1], False, "rerank OFF")]:
        grid = _grid(rerank_value)
        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(cks)), [str(k) for k in cks])
        ax.set_yticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
        ax.set_xlabel("candidate_k")
        ax.set_ylabel("alpha")
        ax.set_title(title)
        for i, _ in enumerate(alphas):
            for j, _ in enumerate(cks):
                v = grid[i][j]
                if not math.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color="white" if v < 0.5 else "black",
                        fontsize=8,
                    )
        fig.colorbar(im, ax=ax, label="recall@5")
    fig.suptitle(f"Sweep: recall@5 over (alpha, candidate_k) — domain '{sweep['domain']}'")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def chart_rerank_ab_png(sweep: dict, out_path: Path) -> Path | None:
    """Bar chart: recall@5, MRR with vs without rerank on the best (alpha, candidate_k) cell."""
    ab = sweep.get("rerank_ab")
    if ab is None:
        return None
    plt = _import_matplotlib()
    metrics = ["recall@5", "MRR"]
    on = [ab["with_rerank"]["recall_at_5"], ab["with_rerank"]["mrr"]]
    off = [ab["without_rerank"]["recall_at_5"], ab["without_rerank"]["mrr"]]
    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], on, width, label="rerank ON", color="#2563eb")
    ax.bar([i + width / 2 for i in x], off, width, label="rerank OFF", color="#94a3b8")
    ax.set_xticks(list(x), metrics)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(
        f"Rerank A/B at alpha={ab['alpha']}, candidate_k={ab['candidate_k']} "
        f"(p95: ON={ab['with_rerank']['p95_ms']:.0f}ms vs OFF={ab['without_rerank']['p95_ms']:.0f}ms)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def chart_per_question_rr_png(eval_data: dict, baseline: dict | None, out_path: Path) -> Path:
    """Bar of per-question reciprocal rank, sorted desc; baseline overlay if provided."""
    plt = _import_matplotlib()
    rows = sorted(eval_data["per_question"], key=lambda r: r["reciprocal_rank"], reverse=True)
    ids = [r["question_id"] for r in rows]
    rr = [r["reciprocal_rank"] for r in rows]

    base_rr: list[float] | None = None
    if baseline is not None:
        base_lookup = {
            r["question_id"]: r["reciprocal_rank"] for r in baseline.get("per_question", [])
        }
        base_rr = [base_lookup.get(qid, 0.0) for qid in ids]

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.3), 4))
    x = range(len(ids))
    ax.bar(x, rr, color="#2563eb", label="current")
    if base_rr is not None:
        ax.scatter(x, base_rr, color="#dc2626", marker="x", s=40, label="baseline", zorder=5)
    ax.set_xticks(list(x), ids, rotation=80, fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("reciprocal rank")
    ax.set_title("Per-question reciprocal rank (sorted desc)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def chart_answer_eval_png(answer_eval: dict, out_path: Path) -> Path:
    """Single-bar pass-rate chart with mean-score overlay; cost annotated below."""
    plt = _import_matplotlib()
    labels = ["pass_rate", "mean_score"]
    values = [answer_eval["pass_rate"], answer_eval["mean_score"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values, color=["#16a34a", "#2563eb"])
    ax.set_ylim(0.0, 1.05)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_title(
        f"Answer eval — judge cost ${answer_eval['total_cost_usd']:.4f}, "
        f"parse_errors={answer_eval['n_parse_errors']}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def chart_tool_eval_png(tool_eval: dict, out_path: Path) -> Path:
    """Bar chart of the four tool-eval headline metrics."""
    plt = _import_matplotlib()
    labels = ["choice_acc", "args_match", "full_match", "val_err_acc"]
    values = [
        tool_eval["tool_choice_accuracy"],
        tool_eval["mean_args_match"],
        tool_eval["full_match_rate"],
        tool_eval["validation_error_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, values, color=["#2563eb", "#16a34a", "#9333ea", "#ea580c"])
    ax.set_ylim(0.0, 1.05)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_title(f"Tool eval — n={tool_eval['n_evaluated']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --- Plotly HTML -------------------------------------------------------------


def build_html_report(
    eval_data: dict, sweep: dict | None, baseline: dict | None, out_path: Path
) -> Path:
    """One self-contained HTML page using plotly. CDN-hosted plotly.js to keep size down."""
    go, make_subplots = _import_plotly()
    figs: list[Any] = []

    # 1. Per-question RR with baseline overlay
    rows = sorted(eval_data["per_question"], key=lambda r: r["reciprocal_rank"], reverse=True)
    ids = [r["question_id"] for r in rows]
    rr = [r["reciprocal_rank"] for r in rows]
    fig = go.Figure()
    fig.add_bar(x=ids, y=rr, name="current")
    if baseline is not None:
        base_lookup = {
            r["question_id"]: r["reciprocal_rank"] for r in baseline.get("per_question", [])
        }
        fig.add_scatter(
            x=ids,
            y=[base_lookup.get(qid, 0.0) for qid in ids],
            mode="markers",
            name="baseline",
            marker=dict(symbol="x", size=10, color="red"),
        )
    fig.update_layout(
        title="Per-question reciprocal rank (sorted desc)", yaxis=dict(range=[0, 1.05]), height=420
    )
    figs.append(fig)

    # 2. Sweep heatmaps
    if sweep:
        cells = sweep["cells"]
        alphas = sorted({c["alpha"] for c in cells})
        cks = sorted({c["candidate_k"] for c in cells})
        fig = make_subplots(rows=1, cols=2, subplot_titles=("rerank ON", "rerank OFF"))
        for col, rerank_value in enumerate([True, False], start=1):
            z = [
                [
                    next(
                        (
                            c["recall_at_5"]
                            for c in cells
                            if c["alpha"] == a
                            and c["candidate_k"] == ck
                            and c["use_rerank"] == rerank_value
                        ),
                        None,
                    )
                    for ck in cks
                ]
                for a in alphas
            ]
            fig.add_heatmap(
                z=z,
                x=[str(k) for k in cks],
                y=[f"{a:.2f}" for a in alphas],
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z],
                texttemplate="%{text}",
                row=1,
                col=col,
                showscale=(col == 2),
            )
        fig.update_layout(
            title=f"Sweep: recall@5 over (alpha, candidate_k) — domain '{sweep['domain']}'",
            height=420,
        )
        fig.update_xaxes(title_text="candidate_k")
        fig.update_yaxes(title_text="alpha")
        figs.append(fig)

    # 3. Rerank A/B
    if sweep and sweep.get("rerank_ab"):
        ab = sweep["rerank_ab"]
        fig = go.Figure()
        fig.add_bar(
            name="rerank ON",
            x=["recall@5", "MRR"],
            y=[ab["with_rerank"]["recall_at_5"], ab["with_rerank"]["mrr"]],
        )
        fig.add_bar(
            name="rerank OFF",
            x=["recall@5", "MRR"],
            y=[ab["without_rerank"]["recall_at_5"], ab["without_rerank"]["mrr"]],
        )
        fig.update_layout(
            title=f"Rerank A/B at alpha={ab['alpha']}, candidate_k={ab['candidate_k']}",
            yaxis=dict(range=[0, 1.0]),
            barmode="group",
            height=380,
        )
        figs.append(fig)

    # 4. Answer eval bars
    if "answer_eval" in eval_data:
        ans = eval_data["answer_eval"]
        fig = go.Figure()
        fig.add_bar(x=["pass_rate", "mean_score"], y=[ans["pass_rate"], ans["mean_score"]])
        fig.update_layout(
            title=f"Answer eval — judge cost ${ans['total_cost_usd']:.4f}",
            yaxis=dict(range=[0, 1.05]),
            height=380,
        )
        figs.append(fig)

    # 5. Tool eval bars
    if "tool_eval" in eval_data:
        te = eval_data["tool_eval"]
        fig = go.Figure()
        fig.add_bar(
            x=["choice_acc", "args_match", "full_match", "val_err_acc"],
            y=[
                te["tool_choice_accuracy"],
                te["mean_args_match"],
                te["full_match_rate"],
                te["validation_error_accuracy"],
            ],
        )
        fig.update_layout(
            title=f"Tool eval — n={te['n_evaluated']}", yaxis=dict(range=[0, 1.05]), height=380
        )
        figs.append(fig)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Eval report — {eval_data['domain']}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;}"
        "h1{margin-bottom:4px;} .meta{color:#64748b;font-size:13px;margin-bottom:24px;}"
        ".chart{margin:18px 0;}</style>",
        "</head><body>",
        f"<h1>Eval report — {eval_data['domain']}</h1>",
        f"<div class='meta'>Generated {datetime.now(UTC).isoformat(timespec='seconds')} "
        f"from {eval_data.get('mode', 'retrieval')} eval at {eval_data['timestamp']}</div>",
    ]
    for i, fig in enumerate(figs):
        # First chart embeds plotly.js inline; the rest use the same already-loaded copy.
        include_js = "cdn" if i == 0 else False
        html_parts.append(
            f"<div class='chart'>{fig.to_html(full_html=False, include_plotlyjs=include_js)}</div>"
        )
    html_parts.append("</body></html>")

    out_path.write_text("".join(html_parts), encoding="utf-8")
    return out_path


# --- Markdown body -----------------------------------------------------------


def build_markdown(
    eval_data: dict,
    sweep: dict | None,
    baseline: dict | None,
    chart_paths: dict[str, Path],
    md_path: Path,
) -> Path:
    """Render the Markdown report. Image paths are relative to md_path's parent."""
    lines: list[str] = []
    md_dir = md_path.parent

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(md_dir)).replace("\\", "/")
        except ValueError:
            return str(p)

    agg = eval_data["aggregate"]
    lines += [
        f"# Eval report — domain `{eval_data['domain']}`",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} from "
        f"`{eval_data.get('mode', 'retrieval')}` eval at {eval_data['timestamp']}_",
        "",
        "## Headline metrics",
        "",
        "| metric | value | baseline | delta |",
        "|---|---|---|---|",
    ]

    def _bget(*path: str) -> Any:
        cur: Any = baseline.get("aggregate") if baseline else None
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    rows = [
        ("MRR", agg.get("mrr"), _bget("mrr")),
        ("recall@5", agg.get("mean_recall_at_k", {}).get("5"), _bget("mean_recall_at_k", "5")),
        ("recall@10", agg.get("mean_recall_at_k", {}).get("10"), _bget("mean_recall_at_k", "10")),
        (
            "precision@5",
            agg.get("mean_precision_at_k", {}).get("5"),
            _bget("mean_precision_at_k", "5"),
        ),
    ]
    for label, cur, old in rows:
        if cur is None:
            continue
        if old is None:
            lines.append(f"| {label} | {cur:.3f} | — | — |")
        else:
            delta = cur - old
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {label} | {cur:.3f} | {old:.3f} | {sign}{delta:.3f} |")
    lat = eval_data.get("latency_ms", {})
    if lat:
        lines += [
            "",
            f"Query latency: p50 = {lat.get('p50_ms', 0):.0f}ms, "
            f"p95 = {lat.get('p95_ms', 0):.0f}ms, "
            f"mean = {lat.get('mean_ms', 0):.0f}ms (n={lat.get('n_queries', 0)})",
        ]

    if "per_question_rr" in chart_paths:
        lines += [
            "",
            "### Per-question reciprocal rank",
            "",
            f"![per-question RR]({_rel(chart_paths['per_question_rr'])})",
        ]

    if sweep:
        lines += [
            "",
            "## Sweep results",
            "",
            f"Cells evaluated: **{len(sweep['cells'])}**, "
            f"total runtime: {sweep.get('total_seconds', 0):.0f}s",
        ]
        winner = sweep["winner"]
        lines += [
            "",
            f"**Winner:** alpha={winner['alpha']}, candidate_k={winner['candidate_k']}, "
            f"rerank={'ON' if winner['use_rerank'] else 'OFF'} "
            f"→ MRR={winner['mrr']:.3f}, recall@5={winner['recall_at_5']:.3f}, "
            f"recall@10={winner['recall_at_10']:.3f}, p95={winner['p95_ms']:.0f}ms",
        ]
        if "sweep_heatmap" in chart_paths:
            lines += ["", f"![sweep heatmap]({_rel(chart_paths['sweep_heatmap'])})"]

        ab = sweep.get("rerank_ab")
        if ab:
            d = ab["delta"]
            sign_r = "+" if d["recall_at_5"] >= 0 else ""
            sign_m = "+" if d["mrr"] >= 0 else ""
            sign_p = "+" if d["p95_ms"] >= 0 else ""
            lines += [
                "",
                "### Rerank A/B",
                "",
                f"At the best (alpha={ab['alpha']}, candidate_k={ab['candidate_k']}) cell:",
                "",
                "| | recall@5 | MRR | p95 (ms) |",
                "|---|---|---|---|",
                f"| rerank ON | {ab['with_rerank']['recall_at_5']:.3f} | {ab['with_rerank']['mrr']:.3f} | {ab['with_rerank']['p95_ms']:.0f} |",
                f"| rerank OFF | {ab['without_rerank']['recall_at_5']:.3f} | {ab['without_rerank']['mrr']:.3f} | {ab['without_rerank']['p95_ms']:.0f} |",
                f"| **delta** | **{sign_r}{d['recall_at_5']:.3f}** | **{sign_m}{d['mrr']:.3f}** | **{sign_p}{d['p95_ms']:.0f}** |",
            ]
            if "rerank_ab" in chart_paths:
                lines += ["", f"![rerank A/B]({_rel(chart_paths['rerank_ab'])})"]

        # Full sweep table
        lines += [
            "",
            "### Full sweep grid",
            "",
            "| alpha | candidate_k | rerank | recall@5 | recall@10 | MRR | p95 (ms) | sec |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in sweep["cells"]:
            lines.append(
                f"| {c['alpha']:.2f} | {c['candidate_k']} | "
                f"{'on' if c['use_rerank'] else 'off'} | "
                f"{c['recall_at_5']:.3f} | {c['recall_at_10']:.3f} | "
                f"{c['mrr']:.3f} | {c['p95_ms']:.0f} | {c['seconds']:.1f} |"
            )

    if "answer_eval" in eval_data:
        ae = eval_data["answer_eval"]
        lines += [
            "",
            "## Answer eval (LLM-as-judge)",
            "",
            f"Judge model: `{eval_data['config'].get('judge_model', 'unknown')}`",
            "",
            f"- pass_rate: **{ae['pass_rate']:.3f}**",
            f"- mean_score: **{ae['mean_score']:.3f}**",
            f"- judge cost: **${ae['total_cost_usd']:.4f}**",
            f"- parse_errors: {ae['n_parse_errors']}",
        ]
        if "answer" in chart_paths:
            lines += ["", f"![answer eval]({_rel(chart_paths['answer'])})"]

    if "tool_eval" in eval_data:
        te = eval_data["tool_eval"]
        lines += [
            "",
            "## Tool-call accuracy",
            "",
            f"- tool_choice_accuracy: **{te['tool_choice_accuracy']:.3f}**",
            f"- mean_args_match: **{te['mean_args_match']:.3f}**",
            f"- full_match_rate: **{te['full_match_rate']:.3f}**",
            f"- validation_error_accuracy: **{te['validation_error_accuracy']:.3f}**",
            f"- n evaluated: {te['n_evaluated']}",
        ]
        if "tool" in chart_paths:
            lines += ["", f"![tool eval]({_rel(chart_paths['tool'])})"]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# --- Main --------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    eval_path = Path(args.eval)
    eval_data = _load_json(eval_path)
    sweep = _load_json(Path(args.sweep)) if args.sweep else None
    baseline = _load_json(Path(args.baseline)) if args.baseline else None

    out_stem = Path(args.output)
    charts_dir = _ensure_dirs(out_stem)
    chart_paths: dict[str, Path] = {}

    chart_paths["per_question_rr"] = chart_per_question_rr_png(
        eval_data, baseline, charts_dir / "per_question_rr.png"
    )
    if sweep:
        chart_paths["sweep_heatmap"] = chart_sweep_heatmap_png(
            sweep, charts_dir / "sweep_heatmap.png"
        )
        ab_path = chart_rerank_ab_png(sweep, charts_dir / "rerank_ab.png")
        if ab_path is not None:
            chart_paths["rerank_ab"] = ab_path
    if "answer_eval" in eval_data:
        chart_paths["answer"] = chart_answer_eval_png(
            eval_data["answer_eval"], charts_dir / "answer_eval.png"
        )
    if "tool_eval" in eval_data:
        chart_paths["tool"] = chart_tool_eval_png(
            eval_data["tool_eval"], charts_dir / "tool_eval.png"
        )

    md_path = build_markdown(eval_data, sweep, baseline, chart_paths, Path(f"{out_stem}.md"))
    html_path = build_html_report(eval_data, sweep, baseline, Path(f"{out_stem}.html"))

    print(f"Wrote: {md_path}")
    print(f"Wrote: {html_path}")
    print(f"Charts: {charts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
