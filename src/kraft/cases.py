"""Test case generation: corner-case-biased from task, trace, or both."""

from __future__ import annotations

from pathlib import Path
from typing import List

import chak
from pydantic import BaseModel, Field

from kraft.skill import TestCase

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


class _GeneratedCase(BaseModel):
    """Schema for LLM-generated test cases (chak structured output)."""

    input: str = Field(description="The question/prompt to give the model under test")
    reference: str = Field(description="The correct/expected output")
    criterion: str = Field(description="One sentence describing what 'equivalent' means for this case")
    why_corner: str = Field(default="", description="Brief note on why this is a corner case")


class _CaseList(BaseModel):
    """Container model — OpenAI structured output requires top-level object."""

    cases: list[_GeneratedCase] = Field(description="List of generated test cases")


def _to_test_cases(items: list[_GeneratedCase], source: str = "synthetic") -> list[TestCase]:
    return [
        TestCase(
            input=item.input,
            reference=item.reference,
            criterion=item.criterion,
            source=source,
        )
        for item in items
    ]


async def from_task(task: str, model: str, api_key: str, n: int = 8) -> list[TestCase]:
    """Generate corner-case-biased test cases from a task description."""
    prompt = _load_prompt("tests_corner_case.md").format(task=task, n=n)
    result = await chak.Conversation(model, api_key).asend(prompt, returns=_CaseList)
    return _to_test_cases(result.cases if result else [])


async def from_trace(
    trace_summary: str, real_cases: list[TestCase], model: str, api_key: str, n: int = 4
) -> list[TestCase]:
    """Generate adversarial variants based on a trace summary.

    real_cases are already extracted from the trace (ground truth).
    This function generates additional synthetic corner cases.
    """
    prompt = _load_prompt("tests_from_trace.md").format(trace=trace_summary, n=n)
    result = await chak.Conversation(model, api_key).asend(prompt, returns=_CaseList)
    return real_cases + _to_test_cases(result.cases if result else [])


async def from_task_and_trace(
    task: str, trace_summary: str, real_cases: list[TestCase], model: str, api_key: str, n: int = 8
) -> list[TestCase]:
    """Generate tests combining task description and trace data."""
    synthetic_n = max(1, n - len(real_cases))
    prompt = _load_prompt("tests_corner_case.md").format(task=task, n=synthetic_n)
    result = await chak.Conversation(model, api_key).asend(prompt, returns=_CaseList)
    return real_cases + _to_test_cases(result.cases if result else [])
