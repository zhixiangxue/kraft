"""LLM-as-judge verifier: grounded by reference, binary yes/no."""

from __future__ import annotations

import chak
from kraft.skill import TestCase

JUDGE_PROMPT = """\
You are a strict equivalence judge. Given a question, a reference answer,
and a candidate answer, decide whether the candidate is equivalent to the
reference under the stated criterion.

Question:
{input}

Reference answer:
{reference}

Equivalence criterion:
{criterion}

Candidate answer:
{candidate}

Reply with exactly one word: 'yes' or 'no'.
"""


async def llm_judge(output: str, case: TestCase, judge_model: str, api_key: str | None = None) -> bool:
    """Judge whether output is equivalent to case.reference under case.criterion.

    Uses a separate LLM call with compressed yes/no output space for consistency.
    Same judge evaluates both baseline and with_skill — bias cancels out.
    """
    import os
    if api_key is None:
        provider = judge_model.split("/")[0] if "/" in judge_model else judge_model
        env_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        env_var = env_map.get(provider, f"{provider.upper()}_API_KEY")
        api_key = os.environ.get(env_var, "")
    prompt = JUDGE_PROMPT.format(
        input=case.input,
        reference=case.reference,
        criterion=case.criterion,
        candidate=output,
    )
    resp = await chak.Conversation(judge_model, api_key).asend(prompt)
    return resp.content.strip().lower().startswith("yes")
