"""Evaluation case generation: corner-case-biased from task, trace, or both."""

from __future__ import annotations

from typing import List

import chak
from pydantic import BaseModel, Field

from kraft import prompts as P
from kraft.skill import EvalCase


class _GeneratedCase(BaseModel):
    """Schema for LLM-generated evaluation cases (chak structured output)."""

    input: str = Field(description="The question/prompt to give the model under test")
    reference: str = Field(description="The correct/expected output")
    criterion: str = Field(description="One sentence describing what 'equivalent' means for this case")
    why_corner: str = Field(default="", description="Brief note on why this is a corner case")


class _CaseList(BaseModel):
    """Container model — OpenAI structured output requires top-level object."""

    cases: list[_GeneratedCase] = Field(description="List of generated evaluation cases")


def _to_eval_cases(items: list[_GeneratedCase], source: str = "synthetic") -> list[EvalCase]:
    return [
        EvalCase(
            input=item.input,
            reference=item.reference,
            criterion=item.criterion,
            source=source,
        )
        for item in items
    ]


class CaseGenerationError(RuntimeError):
    """Raised when the model fails to produce structured evaluation cases.

    chak returns ``None`` from ``asend(..., returns=...)`` when structured
    output is unsupported or the provider rejected the schema (e.g.
    DeepSeek thinking mode). We fail fast rather than silently degrading
    to an empty case list — the downstream authoring + evaluation phases
    would otherwise burn tokens producing a skill no one can score.
    """


def _unwrap_cases(result: _CaseList | None, *, phase: str, model: str) -> list[_GeneratedCase]:
    if result is None:
        raise CaseGenerationError(
            f"Structured output returned None during '{phase}' with model={model!r}. "
            "The provider likely rejected the schema (e.g. unsupported tool_choice). "
            "Aborting before downstream phases waste tokens."
        )
    if not result.cases:
        raise CaseGenerationError(
            f"Model produced zero cases during '{phase}' with model={model!r}. "
            "Aborting before downstream phases waste tokens."
        )
    return result.cases


async def from_natural_language(
    raw_cases: list[str], model: str, api_key: str
) -> tuple[list[EvalCase], chak.Conversation]:
    """Convert user natural-language case descriptions to structured EvalCase via LLM.

    Uses chak structured output to extract (input, reference, criterion) from
    each raw string.  Cases where ``reference`` is empty after structuring are
    silently discarded — the user didn't specify an expectation so the case
    cannot be evaluated.

    Returns ``(cases, conv)`` for token accounting.
    """
    prompt = P.structure_cases(raw_cases=raw_cases)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    if result is None:
        raise CaseGenerationError(
            f"Structured output returned None during 'from_natural_language' with model={model!r}. "
            "The provider likely rejected the schema."
        )
    # Filter out cases with empty reference — user didn't say what to expect.
    valid = [c for c in result.cases if c.reference.strip()]
    return _to_eval_cases(valid, source="user"), conv


async def from_task(task: str, model: str, api_key: str, n: int = 8) -> tuple[list[EvalCase], chak.Conversation]:
    """Generate corner-case-biased evaluation cases from a task description.

    Returns ``(cases, conv)`` so the caller can both fold the conversation's
    input/output token usage into run-level accounting *and* dump the full
    message log via ``conv.dump()`` for cost auditing.

    Raises:
        CaseGenerationError: if the model fails to produce any structured
            cases (chak returned None, or an empty list).
    """
    prompt = P.tests_corner_case(task=task, n=n)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    return _to_eval_cases(_unwrap_cases(result, phase="from_task", model=model)), conv


async def from_trace(
    trace_summary: str, real_cases: list[EvalCase], model: str, api_key: str, n: int = 4
) -> tuple[list[EvalCase], chak.Conversation]:
    """Generate adversarial variants based on a trace summary.

    real_cases are already extracted from the trace (ground truth).
    This function generates additional synthetic corner cases.

    Raises:
        CaseGenerationError: if structured generation of the synthetic
            corner cases fails. We fail even when ``real_cases`` is
            non-empty because a broken structured-output path is a strong
            signal the rest of the pipeline will also misbehave.
    """
    prompt = P.tests_from_trace(trace=trace_summary, n=n)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    return real_cases + _to_eval_cases(_unwrap_cases(result, phase="from_trace", model=model)), conv


async def from_task_and_trace(
    task: str, trace_summary: str, real_cases: list[EvalCase], model: str, api_key: str, n: int = 8
) -> tuple[list[EvalCase], chak.Conversation]:
    """Generate evaluation cases combining task description and trace data.

    Raises:
        CaseGenerationError: see :func:`from_trace`.
    """
    synthetic_n = max(1, n - len(real_cases))
    prompt = P.tests_corner_case(task=task, n=synthetic_n)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    return real_cases + _to_eval_cases(_unwrap_cases(result, phase="from_task_and_trace", model=model)), conv
