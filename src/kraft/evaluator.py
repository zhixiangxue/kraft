"""Evaluator: judges whether a candidate output is equivalent to a reference.

The default ``llm_judge`` uses an LLM as the equivalence judge — a standard
"LLM-as-judge" pattern. ``Kraft`` accepts any async callable with the same
signature via the ``evaluator=`` parameter, so deterministic judges
(string match, regex, sandboxed exec) can be plugged in for tasks where
reference comparison is mechanical.
"""

from __future__ import annotations

import os

import chak

from kraft.skill import EvalCase

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


async def llm_judge(
    output: str,
    case: EvalCase,
    judge_model: str,
    api_key: str | None = None,
) -> tuple[bool, int]:
    """Default evaluator: LLM-as-judge with binary yes/no output.

    Uses a separate LLM call with a compressed yes/no output space for
    consistency. The same judge evaluates both baseline and with-skill arms
    so judge bias cancels out in the lift signal.

    Returns ``(passed, tokens_used)`` so the caller can fold judge cost into
    run-level usage accounting (separately from the candidate-arm cost).
    """
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
    conv = chak.Conversation(judge_model, api_key)
    resp = await conv.asend(prompt)
    passed = resp.content.strip().lower().startswith("yes")
    return passed, conv.stats()["total_tokens"]
