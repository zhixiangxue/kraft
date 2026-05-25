"""Kraft engine: skill generation, evaluation, and refinement loop."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Callable, Awaitable

import chak
from chak.tools.skills import ClaudeSkill

from kraft.skill import Skill, TestCase, RunResult, Report, total_tokens
from kraft.decision import Thresholds, Verdict, decide
from kraft.verifier import llm_judge
from kraft import cases

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _extract_skill_md(text: str) -> str:
    """Extract the skill markdown (with frontmatter) from LLM output.

    The LLM may wrap it in a code block — strip that.
    """
    # Try to find content between ``` markers
    match = re.search(r"```(?:markdown|md)?\s*\n(---.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code block, look for raw frontmatter
    if text.strip().startswith("---"):
        return text.strip()
    return text.strip()


async def kraft(*, task: str | None = None, trace: str | None = None, model: str, api_key: str | None = None, **kw) -> Skill:
    """Top-level one-liner: generate a skill, verify it, return it."""
    return await Kraft(model=model, api_key=api_key, **kw).run(task=task, trace=trace)


class Kraft:
    """Mid-level API: configurable skill generation and verification engine."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        judge_model: str | None = None,
        max_iter: int = 3,
        runs_dir: str | Path = "./runs",
        thresholds: Thresholds | None = None,
        verifier: Callable[[str, TestCase, str], Awaitable[bool]] | None = None,
        n_tests: int = 8,
    ):
        self.model = model
        self.api_key = api_key or self._resolve_api_key(model)
        self.judge_model = judge_model or model
        self.thresholds = thresholds or Thresholds(max_iter=max_iter)
        self.runs_dir = Path(runs_dir)
        self._verifier = verifier or llm_judge
        self.n_tests = n_tests

        # Lazy import to avoid circular dep
        from kraft.store import Store
        self.store = Store(self.runs_dir)

    @staticmethod
    def _resolve_api_key(model: str) -> str:
        """Resolve API key from environment based on model provider."""
        import os
        provider = model.split("/")[0] if "/" in model else model
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = env_map.get(provider, f"{provider.upper()}_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(f"No api_key provided and {env_var} not set in environment")
        return key

    async def run(self, *, task: str | None = None, trace: str | None = None) -> Skill:
        """Full loop: generate → test → evaluate → refine → decide."""
        assert task or trace, "At least one of task or trace must be provided"
        run_dir = self.store.create_run_dir()
        self._run_dir = run_dir

        skill = await self.generate(task=task, trace=trace)
        tests = await self.make_tests(task=task, trace=trace)
        self.store.save_tests(run_dir, tests)

        skill_history: list[Skill] = []
        report_history: list[Report] = []

        while True:
            skill = await self.evaluate(skill, tests, iter=len(report_history))
            skill_history.append(skill)
            report_history.append(skill.report)

            verdict = decide(report_history, self.thresholds)
            self.store.save_iter(run_dir, len(report_history) - 1, skill, verdict)

            if verdict.state == "REFINE":
                skill = await self.refine(skill)
                continue

            # Terminal state
            final_skill = (
                skill_history[verdict.kept_iter]
                if verdict.kept_iter >= 0
                else skill_history[-1]
            )
            final_skill.report.recommendation = self._verdict_to_recommendation(verdict)
            self.store.finalize(run_dir, final_skill, verdict)
            return final_skill

    async def generate(self, *, task: str | None = None, trace: str | None = None) -> Skill:
        """Generate a skill from task description, trace, or both."""
        if task and trace:
            prompt = _load_prompt("skill_from_task_and_trace.md").format(task=task, trace=trace)
        elif trace:
            prompt = _load_prompt("skill_from_trace.md").format(trace=trace)
        else:
            prompt = _load_prompt("skill_from_task.md").format(task=task)

        resp = await chak.Conversation(self.model, self.api_key).asend(prompt)
        raw = _extract_skill_md(resp.content)
        return Skill.parse(raw)

    async def make_tests(self, *, task: str | None = None, trace: str | None = None) -> list[TestCase]:
        """Generate test cases (corner-case-biased)."""
        if task and not trace:
            return await cases.from_task(task, self.model, self.api_key, n=self.n_tests)
        # trace and task+trace modes would go through trace.py (Phase 4)
        # For now, fall back to task-based generation
        if task:
            return await cases.from_task(task, self.model, self.api_key, n=self.n_tests)
        raise NotImplementedError("Trace-only test generation requires Phase 4 (trace.py)")

    async def evaluate(self, skill: Skill, tests: list[TestCase], iter: int = 0) -> Skill:
        """Run baseline vs with-skill comparison on all test cases."""
        # Materialize skill to a directory for ClaudeSkill
        skill_dir = self.store.materialize_skill(self._run_dir, iter, skill)

        # Run both arms concurrently
        with_skill_coros = [self._run_one(t, skill_dir=skill_dir) for t in tests]
        baseline_coros = [self._run_one(t, skill_dir=None) for t in tests]

        with_skill, baseline = await asyncio.gather(
            asyncio.gather(*with_skill_coros),
            asyncio.gather(*baseline_coros),
        )

        skill.report = Report(
            with_skill=list(with_skill),
            baseline=list(baseline),
            iter=iter,
        )
        return skill

    async def refine(self, skill: Skill) -> Skill:
        """Refine skill based on failed cases with full raw output."""
        payload = "\n\n".join(
            f"### Failed case ({r.case.source})\n"
            f"INPUT: {r.case.input}\n"
            f"REFERENCE: {r.case.reference}\n"
            f"CRITERION: {r.case.criterion}\n"
            f"ACTUAL OUTPUT:\n{r.output}\n"
            for r in skill.report.failures
        )
        prompt = _load_prompt("refine.md").format(skill=skill.render(), failures=payload)
        resp = await chak.Conversation(self.model, self.api_key).asend(prompt)
        raw = _extract_skill_md(resp.content)
        return Skill.parse(raw)

    async def _run_one(self, case: TestCase, *, skill_dir: Path | None) -> RunResult:
        """Run a single test case with or without skill."""
        tools = [ClaudeSkill(str(skill_dir))] if skill_dir is not None else []
        conv = chak.Conversation(self.model, self.api_key, tools=tools)
        t0 = time.time()
        resp = await conv.asend(case.input)
        latency_ms = int((time.time() - t0) * 1000)
        passed = await self._verifier(resp.content, case, self.judge_model, self.api_key)
        return RunResult(
            case=case,
            output=resp.content,
            passed=passed,
            tokens=total_tokens(resp),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _verdict_to_recommendation(v: Verdict) -> str:
        # decide() collapses all KEEP-eligible cases (including PLATEAU/MAX_ITER
        # where best iter passes the gate) into state="KEEP". So PLATEAU and
        # MAX_ITER reaching this branch always mean: best iter still inadequate.
        if v.state == "KEEP":
            return "keep"
        if v.state == "NO_SKILL_NEEDED":
            return "no-skill-needed"
        return "discard"
