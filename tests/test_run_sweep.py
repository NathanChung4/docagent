"""Tests for the parameter sweep runner's pure logic (grid expansion, winner pick, A/B).

The full end-to-end run requires a pgvector container — that's exercised
through `scripts/run_sweep.py` itself, not here. This module covers the
deterministic logic so it stays in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/run_sweep.py isn't a package; load it directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_SWEEP = _REPO_ROOT / "scripts" / "run_sweep.py"

spec = importlib.util.spec_from_file_location("run_sweep", _RUN_SWEEP)
run_sweep = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["run_sweep"] = run_sweep
spec.loader.exec_module(run_sweep)  # type: ignore[union-attr]

SweepCell = run_sweep.SweepCell
expand_grid = run_sweep.expand_grid
parse_rerank_flag = run_sweep.parse_rerank_flag
pick_winner = run_sweep.pick_winner
rerank_ab = run_sweep.rerank_ab


def _cell(alpha=0.5, ck=20, rerank=True, recall5=0.8, mrr=0.85, p95=12000.0) -> SweepCell:
    return SweepCell(
        alpha=alpha,
        candidate_k=ck,
        use_rerank=rerank,
        recall_at_5=recall5,
        recall_at_10=recall5 + 0.05,
        mrr=mrr,
        precision_at_5=recall5 * 0.6,
        p50_ms=p95 - 1500,
        p95_ms=p95,
        n_questions=28,
        seconds=10.0,
    )


# --- expand_grid -------------------------------------------------------------


def test_expand_grid_full_product():
    grid = expand_grid([0.3, 0.5], [10, 20], [True, False])
    assert len(grid) == 8
    assert (0.3, 10, True) in grid
    assert (0.5, 20, False) in grid


def test_expand_grid_stable_order():
    grid = expand_grid([0.3, 0.7], [20], [True])
    assert grid == [(0.3, 20, True), (0.7, 20, True)]


# --- parse_rerank_flag -------------------------------------------------------


def test_parse_rerank_flag_on_off():
    assert parse_rerank_flag("on,off") == [True, False]
    assert parse_rerank_flag("off,on") == [False, True]


def test_parse_rerank_flag_single():
    assert parse_rerank_flag("on") == [True]
    assert parse_rerank_flag("off") == [False]


def test_parse_rerank_flag_case_insensitive():
    assert parse_rerank_flag("ON,OFF") == [True, False]


def test_parse_rerank_flag_rejects_unknown():
    with pytest.raises(ValueError):
        parse_rerank_flag("on,maybe")


# --- pick_winner -------------------------------------------------------------


def test_pick_winner_highest_mrr():
    cells = [
        _cell(alpha=0.3, mrr=0.80),
        _cell(alpha=0.5, mrr=0.90),
        _cell(alpha=0.7, mrr=0.85),
    ]
    winner = pick_winner(cells)
    assert winner.alpha == 0.5


def test_pick_winner_breaks_mrr_tie_by_lower_p95():
    cells = [
        _cell(alpha=0.3, mrr=0.90, p95=15000.0),
        _cell(alpha=0.5, mrr=0.90, p95=10000.0),  # faster
        _cell(alpha=0.7, mrr=0.90, p95=20000.0),
    ]
    winner = pick_winner(cells)
    assert winner.alpha == 0.5
    assert winner.p95_ms == 10000.0


# --- rerank_ab ---------------------------------------------------------------


def test_rerank_ab_picks_best_on_and_compares_same_cell_off():
    cells = [
        _cell(alpha=0.3, ck=20, rerank=True, mrr=0.80, recall5=0.78),
        _cell(alpha=0.5, ck=20, rerank=True, mrr=0.90, recall5=0.85),  # best on
        _cell(alpha=0.5, ck=20, rerank=False, mrr=0.75, recall5=0.70),  # matched off
        _cell(alpha=0.5, ck=10, rerank=False, mrr=0.60, recall5=0.55),  # not the match
    ]
    ab = rerank_ab(cells)
    assert ab is not None
    assert ab["alpha"] == 0.5
    assert ab["candidate_k"] == 20
    assert ab["with_rerank"]["mrr"] == 0.90
    assert ab["without_rerank"]["mrr"] == 0.75
    assert ab["delta"]["mrr"] == pytest.approx(0.15)
    assert ab["delta"]["recall_at_5"] == pytest.approx(0.15)


def test_rerank_ab_returns_none_when_one_side_missing():
    only_on = [_cell(rerank=True), _cell(alpha=0.7, rerank=True)]
    assert rerank_ab(only_on) is None


def test_rerank_ab_returns_none_when_no_matched_pair():
    """Best-on cell has no rerank-off counterpart at the same (alpha, candidate_k)."""
    cells = [
        _cell(alpha=0.5, ck=20, rerank=True, mrr=0.90),
        _cell(alpha=0.7, ck=10, rerank=False, mrr=0.70),
    ]
    assert rerank_ab(cells) is None
