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


async def from_task(task: str, model: str, api_key: str, n: int = 8) -> tuple[list[EvalCase], int]:
    """Generate corner-case-biased evaluation cases from a task description.

    Returns ``(cases, tokens_used)`` so the caller can fold token cost into
    its run-level usage accounting.
    """
    prompt = P.tests_corner_case(task=task, n=n)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    return _to_eval_cases(result.cases if result else []), conv.stats()["total_tokens"]


async def from_trace(
    trace_summary: str, real_cases: list[EvalCase], model: str, api_key: str, n: int = 4
) -> tuple[list[EvalCase], int]:
    """Generate adversarial variants based on a trace summary.

    real_cases are already extracted from the trace (ground truth).
    This function generates additional synthetic corner cases.
    """
    prompt = P.tests_from_trace(trace=trace_summary, n=n)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    return real_cases + _to_eval_cases(result.cases if result else []), conv.stats()["total_tokens"]


async def from_task_and_trace(
    task: str, trace_summary: str, real_cases: list[EvalCase], model: str, api_key: str, n: int = 8
) -> tuple[list[EvalCase], int]:
    """Generate evaluation cases combining task description and trace data."""
    synthetic_n = max(1, n - len(real_cases))
    prompt = P.tests_corner_case(task=task, n=synthetic_n)
    conv = chak.Conversation(model, api_key)
    result = await conv.asend(prompt, returns=_CaseList)
    return real_cases + _to_eval_cases(result.cases if result else []), conv.stats()["total_tokens"]
