"""Compare an eval results JSON against pass/fail thresholds for CI.

Used by .github/workflows/eval.yml after `scripts/run_eval.py` writes a results
JSON. Reads the aggregate MRR and recall@5, compares against floors, exits
nonzero if any metric is below its floor. Stdlib only.

Usage:
    python scripts/check_eval_thresholds.py results/eval_sample_*.json \
        --min-mrr 0.50 --min-recall-5 0.75

Exit codes:
    0 — all metrics at or above floors
    1 — at least one metric below its floor
    2 — usage error or unreadable input
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _resolve_input(pattern: str) -> Path:
    """Expand a glob and pick the lexicographically last match (newest by timestamp).

    Lets CI pass `results/eval_sample_*.json` without knowing the exact filename.
    """
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"ERROR: no files matched {pattern!r}", file=sys.stderr)
        sys.exit(2)
    return Path(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Path or glob to a results JSON written by run_eval.py.")
    parser.add_argument("--min-mrr", type=float, required=True, help="Floor for aggregate MRR.")
    parser.add_argument(
        "--min-recall-5",
        type=float,
        required=True,
        help="Floor for aggregate mean recall@5.",
    )
    args = parser.parse_args()

    path = _resolve_input(args.results)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    agg = payload.get("aggregate") or {}
    mrr = agg.get("mrr")
    recall5 = (agg.get("mean_recall_at_k") or {}).get("5")
    if mrr is None or recall5 is None:
        print(
            f"ERROR: results JSON missing aggregate.mrr or mean_recall_at_k['5']: {path}",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    print(f"Eval gate vs {path.name}:")
    print(f"  MRR        {mrr:.3f}   floor={args.min_mrr:.3f}", end="")
    if mrr < args.min_mrr:
        failures.append(f"MRR {mrr:.3f} < floor {args.min_mrr:.3f}")
        print("   FAIL")
    else:
        print("   ok")
    print(f"  recall@5   {recall5:.3f}   floor={args.min_recall_5:.3f}", end="")
    if recall5 < args.min_recall_5:
        failures.append(f"recall@5 {recall5:.3f} < floor {args.min_recall_5:.3f}")
        print("   FAIL")
    else:
        print("   ok")

    if failures:
        print("\nEval gate FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nEval gate PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
