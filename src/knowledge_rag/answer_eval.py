"""LLM-as-judge: score generated answers against expected fragments.

Pairs with `evaluation.py` (retrieval metrics) — that module says whether the
right *chunks* were retrieved; this one says whether the model wrote the right
*answer*. Together they bracket the RAG quality story end-to-end.

The judge (Claude Haiku 4.5 by default, ~$0.0003 / question) reads
(question, expected_contains, generated_answer) and returns a float 0.0–1.0
plus a one-sentence rationale. Scores ≥ pass_threshold count as passes.

`expected_contains_hits` / `_misses` on the verdict are forensic only — a
deterministic substring view of which expected fragments appear in the answer.
They're kept on the verdict so per-question failure inspection doesn't need
to re-derive them, but they don't influence `score` or `passed`.

Special case: when `expected_contains` is empty, the question is an "I don't
know" entry — the judge is told to score 1.0 iff the assistant explicitly
disclaims and 0.0 if it confidently answers anyway. Catches hallucination
on out-of-corpus questions.

Robustness:
  - Judge response is parsed strict-JSON. If parsing fails, the judge is
    re-prompted once with an explicit "respond with JSON only" reminder. If
    that also fails, the verdict is recorded with score=0.0 and a `parse_error`
    rationale rather than crashing the whole eval run.
  - Cost is computed per call from a model-keyed pricing map so `--judge-model`
    can swap to Sonnet/Opus without silently mis-pricing.
  - System prompt is sent in a cache-control block so repeated judge calls
    in one run share a single cache write — same pattern as `generation.py`.

Generic — zero domain-specific strings. Consumes (question, expected_contains,
generated_answer) tuples; doesn't know or care which domain produced them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
DEFAULT_PASS_THRESHOLD = 0.7
DEFAULT_MAX_TOKENS = 256

# Per-1M-token pricing (USD), keyed by model. Cached: 2026-04-15.
# Unknown models fall back to Haiku rates with a soft warning at call time.
_JUDGE_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (5.00, 25.00),
}
_DEFAULT_PRICING = _JUDGE_PRICING_PER_M["claude-haiku-4-5"]


JUDGE_SYSTEM_PROMPT = """You grade an AI assistant's answer to a technical question.

You will see: the question, a list of expected answer fragments (substrings the \
correct answer should contain), and the assistant's full answer.

Score 0.0–1.0 on correctness:
  1.0 — all expected fragments present (or paraphrased), no contradictions.
  0.7 — most fragments present, no major errors. Minor omissions OK.
  0.3 — some fragments missing OR contains a partial factual error.
  0.0 — most/all fragments missing OR the answer contradicts known facts.

Special case — if `expected_contains` is empty, the assistant should disclaim \
(say "I don't know", "the documents don't say", or similar). Score 1.0 iff \
the assistant explicitly disclaims; 0.0 if it confidently answers anyway.

Respond with ONLY a JSON object on a single line, no markdown, no preamble:
{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}"""


@dataclass
class JudgeVerdict:
    """Per-question grading output. Keep small — collected into reports."""

    question_id: str
    question: str
    score: float
    passed: bool
    rationale: str
    expected_contains_hits: list[str] = field(default_factory=list)
    expected_contains_misses: list[str] = field(default_factory=list)
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost_usd: float = 0.0
    parse_error: bool = False


@dataclass
class AnswerEvalReport:
    """Aggregate output across the dataset."""

    per_question: list[JudgeVerdict]
    pass_rate: float
    mean_score: float
    total_cost_usd: float
    n_parse_errors: int


# --- helpers -----------------------------------------------------------------


def _local_hits_misses(
    expected_contains: Sequence[str], answer: str
) -> tuple[list[str], list[str]]:
    """Case-insensitive substring split of expected fragments into hits/misses."""
    lower = answer.lower()
    hits: list[str] = []
    misses: list[str] = []
    for frag in expected_contains:
        if frag.lower() in lower:
            hits.append(frag)
        else:
            misses.append(frag)
    return hits, misses


def _judge_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one judge call. Looks up per-M pricing by model name."""
    in_per_m, out_per_m = _JUDGE_PRICING_PER_M.get(model, _DEFAULT_PRICING)
    return (input_tokens * in_per_m + output_tokens * out_per_m) / 1_000_000


def _build_judge_user_message(
    question: str, expected_contains: Sequence[str], generated_answer: str
) -> str:
    """Single user message; system prompt has the rubric so we don't repeat it."""
    expected_repr = (
        json.dumps(list(expected_contains))
        if expected_contains
        else "[] (the answer should disclaim — see special case)"
    )
    return (
        f"Question: {question}\n"
        f"Expected fragments: {expected_repr}\n"
        f"Assistant's answer: {generated_answer}"
    )


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Pull (score, rationale) out of the judge's response.

    Tolerant of leading/trailing whitespace and of the model wrapping the JSON
    in ```json fences (a common Haiku failure mode). Raises ValueError if
    nothing parseable comes back.
    """
    cleaned = text.strip()
    # Strip ``` fences if the model added them.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    # Find first { ... } if there's preamble.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    payload = json.loads(cleaned[start : end + 1])
    score = float(payload["score"])
    rationale = str(payload.get("rationale", ""))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score {score} outside [0, 1]")
    return score, rationale


def _call_judge(
    client: Any,
    model: str,
    user_message: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Single Anthropic call. Returns (text, input_tokens, output_tokens).

    The system prompt sits in a cache_control block — across the ~30-50 judge
    calls per eval run it's identical, so the second call onwards reads from
    cache instead of re-paying for the prompt.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {"type": "text", "text": JUDGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    return text, response.usage.input_tokens, response.usage.output_tokens


# --- public API --------------------------------------------------------------


def judge_answer(
    question: str,
    expected_contains: Sequence[str],
    generated_answer: str,
    *,
    question_id: str = "",
    client: Any | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> JudgeVerdict:
    """Grade one (question, expected, answer) triple. One Anthropic call (or two on parse retry)."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    hits, misses = _local_hits_misses(expected_contains, generated_answer)
    user_message = _build_judge_user_message(question, expected_contains, generated_answer)

    text, in_tokens, out_tokens = _call_judge(client, model, user_message, max_tokens)
    parse_error = False
    try:
        score, rationale = _parse_judge_response(text)
    except (ValueError, KeyError, json.JSONDecodeError):
        # One retry with a stricter user reminder. Keeps the eval moving when
        # Haiku occasionally returns prose; if both fail, record a 0 and a
        # parse_error flag rather than crashing the whole run.
        retry_msg = user_message + "\n\nReminder: respond with ONLY a JSON object on a single line."
        text2, in2, out2 = _call_judge(client, model, retry_msg, max_tokens)
        in_tokens += in2
        out_tokens += out2
        try:
            score, rationale = _parse_judge_response(text2)
        except (ValueError, KeyError, json.JSONDecodeError):
            score = 0.0
            rationale = f"parse_error: judge returned non-JSON response: {text2[:120]!r}"
            parse_error = True

    return JudgeVerdict(
        question_id=question_id,
        question=question,
        score=score,
        passed=score >= pass_threshold,
        rationale=rationale,
        expected_contains_hits=hits,
        expected_contains_misses=misses,
        judge_input_tokens=in_tokens,
        judge_output_tokens=out_tokens,
        judge_cost_usd=_judge_cost(model, in_tokens, out_tokens),
        parse_error=parse_error,
    )


def evaluate_answers(
    answer_pairs: Sequence[tuple[dict, str]],
    *,
    client: Any | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> AnswerEvalReport:
    """Grade many (dataset_entry, generated_answer) pairs. One Anthropic call per pair.

    Skips entries whose `kind` isn't `"qa"` (tool entries are graded by
    `tool_eval.py`). Entries are always graded even if `expected_contains` is
    empty — those are the IDK adversarial cases.

    Args:
        answer_pairs: Each is (dataset_entry_dict, model_generated_answer_str).
        client: Anthropic SDK client. Constructed lazily if None.
        model: Judge model. Default Haiku 4.5 — cheap + reliable for grading.
        pass_threshold: Score ≥ this counts as a pass.

    Returns:
        AnswerEvalReport with per-question verdicts and aggregates.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    verdicts: list[JudgeVerdict] = []
    for entry, generated in answer_pairs:
        if entry.get("kind", "qa") != "qa":
            continue
        verdicts.append(
            judge_answer(
                question=entry["question"],
                expected_contains=entry.get("expected_answer_contains", []),
                generated_answer=generated,
                question_id=entry.get("id", ""),
                client=client,
                model=model,
                pass_threshold=pass_threshold,
            )
        )

    n = len(verdicts)
    pass_rate = sum(1 for v in verdicts if v.passed) / n if n else 0.0
    mean_score = sum(v.score for v in verdicts) / n if n else 0.0
    total_cost = sum(v.judge_cost_usd for v in verdicts)
    n_parse_errors = sum(1 for v in verdicts if v.parse_error)

    return AnswerEvalReport(
        per_question=verdicts,
        pass_rate=pass_rate,
        mean_score=mean_score,
        total_cost_usd=total_cost,
        n_parse_errors=n_parse_errors,
    )
