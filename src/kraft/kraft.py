"""Kraft engine: skill generation, evaluation, and refinement loop.

Directory layout (from the developer's point of view):

    <skill_dir>/                    ← the developer's chosen path; final
      SKILL.md                        SKILL.md (+ scripts/ if scripts_enabled)
      scripts/                        sits here — ready to use, no digging.
    <skill_dir>.kraft/              ← sibling, automatic; iteration history
      <timestamp>/
        cases.json                    generated evaluation cases for this run
        evaluation.json               run-level evaluation (chosen iter)
        iter_0/skill_dir/             agent-authored SKILL.md (+ scripts/)
        iter_0/evaluation.json        per-iter evaluation (verdict embedded)
        iter_1/skill_dir/
        ...

``<skill_dir>.kraft`` lives **outside** ``<skill_dir>`` because ClaudeSkill
scans the skill root recursively (including dotfiles) and would otherwise
expose iteration leftovers as Layer-3 supporting files.

Skills are authored in **agentic mode**: the LLM is given a filesystem tool
(rooted at the iteration's working directory) and writes SKILL.md directly.
Whether the author agent is also given **python + bash** tools — and is
allowed to emit supporting files (scripts/, references/, examples/) — is
controlled by the ``scripts_enabled`` flag (default ``False``: pure-prose
SKILL.md only).

For single-variable attribution, the evaluation arms get the **same tool
set implied by ``scripts_enabled``** — both either with Python+Bash (when
``scripts_enabled=True``) or both without (when ``scripts_enabled=False``,
the pure-doc case). The only difference between baseline and with-skill is
whether ClaudeSkill(skill_dir) is also in the tools list. This keeps the
evaluation runtime aligned with the skill's intended deployment runtime.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Awaitable, Callable

import chak
from chak.tools.skills import ClaudeSkill
from chak.tools.std import Bash, FileSystem, Python

from kraft import cases
from kraft import prompts as P
from kraft.decision import Thresholds, Verdict, decide
from kraft.skill import EvalCase, EvalResult, Evaluation, Skill, TokenUsage, total_tokens
from kraft.evaluator import llm_judge

# Note on tool-call iteration cap:
# We do **not** set max_iterations explicitly. chak's default (50) is the
# framework's considered upper bound for "agent stuck in a loop" and applies
# uniformly to author/refine and per-case evaluation. Capping more aggressively
# based on "this run feels long" is mistaking symptom for cause: an author
# legitimately needs many rounds to read_file + edit_file iteratively, and a
# truncated agent ships a half-baked skill. If the default cap is hit, that's
# a real signal the agent went off the rails — we surface it as a swallowed
# exception in _run_author_agent so the on-disk SKILL.md still flows into
# evaluation, where the metrics decide whether the partial work is keepable.


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy all files under src into dst, preserving relative layout."""
    dst.mkdir(parents=True, exist_ok=True)
    for s in src.rglob("*"):
        if s.is_file():
            rel = s.relative_to(src)
            d = dst / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)


async def kraft(
    *,
    skill_dir: str | Path,
    task: str | None = None,
    trace: str | None = None,
    model: str,
    api_key: str | None = None,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    overwrite: bool = False,
    scripts_enabled: bool = False,
    **kw,
) -> Skill:
    """Top-level one-liner: generate a skill, verify it, return it.

    ``skill_dir``       — required. Final SKILL.md (+ scripts/) lands here.
    ``overwrite``       — if False (default) and ``skill_dir`` is non-empty,
                          raises FileExistsError. Pass True to clobber.
    ``judge_api_key``   — only needed when ``judge_model`` is on a different
                          provider than ``model``. Otherwise auto-resolved.
    ``scripts_enabled`` — False (default): pure-prose SKILL.md only.
                          True: author may add scripts/ references/ examples/.

    The returned ``Skill`` carries both ``evaluation`` (per-run verdict and
    pass rates) and ``usage`` (TokenUsage partitioned into generation /
    evaluation), so callers can inspect cost and quality in one object.
    """
    return await Kraft(
        model=model,
        skill_dir=skill_dir,
        api_key=api_key,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        overwrite=overwrite,
        scripts_enabled=scripts_enabled,
        **kw,
    ).run(task=task, trace=trace)


class Kraft:
    """Mid-level API: configurable skill generation and verification engine."""

    def __init__(
        self,
        model: str,
        *,
        skill_dir: str | Path,
        api_key: str | None = None,
        judge_model: str | None = None,
        judge_api_key: str | None = None,
        overwrite: bool = False,
        max_iter: int = 1,
        thresholds: Thresholds | None = None,
        evaluator: Callable[..., Awaitable[bool]] | None = None,
        n_cases: int = 2,
        scripts_enabled: bool = False,
    ):
        self.model = model
        self.api_key = api_key or self._resolve_api_key(model)
        self.judge_model = judge_model or model

        # Judge api_key: three-tier fallback so cross-provider judges work.
        #   1. explicit `judge_api_key`        → use it
        #   2. judge_model == model            → reuse self.api_key
        #   3. cross-provider                  → resolve from environment
        if judge_api_key is not None:
            self.judge_api_key = judge_api_key
        elif self.judge_model == self.model:
            self.judge_api_key = self.api_key
        else:
            self.judge_api_key = self._resolve_api_key(self.judge_model)

        self.thresholds = thresholds or Thresholds(max_iter=max_iter)
        self._evaluator = evaluator or llm_judge
        self.n_cases = n_cases
        self.scripts_enabled = scripts_enabled

        # Run-level token accounting, partitioned by phase. Populated by
        # _run_author_agent / make_cases / _run_one / the judge call inside
        # _run_one. Read by callers after run() returns to print a summary.
        self.usage = TokenUsage()

        # Lazy import to avoid circular dep
        from kraft.store import Store
        self.store = Store(Path(skill_dir), overwrite=overwrite)
        self.skill_dir = self.store.skill_dir

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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, *, task: str | None = None, trace: str | None = None) -> Skill:
        """Full loop: generate → evaluate → refine → decide."""
        assert task or trace, "At least one of task or trace must be provided"
        run_dir = self.store.create_run_dir()
        self._run_dir = run_dir

        eval_cases = await self.make_cases(task=task, trace=trace)
        self.store.save_cases(run_dir, eval_cases)

        # Iter 0: agent authors the skill from scratch into iter_0/skill_dir/
        iter_idx = 0
        skill_dir = self.store.iter_skill_dir(run_dir, iter_idx)
        skill = await self.generate(task=task, trace=trace, out_dir=skill_dir)

        skill_history: list[Skill] = []
        eval_history: list[Evaluation] = []

        while True:
            skill = await self.evaluate(skill, eval_cases, iter=iter_idx)
            skill_history.append(skill)
            eval_history.append(skill.evaluation)

            verdict = decide(eval_history, self.thresholds)
            # Make every iter's evaluation.json self-explanatory: embed the
            # decision context so a reader doesn't need to guess at threshold
            # defaults or cross-reference anything else (verdict.json is gone).
            skill.evaluation.verdict = asdict(verdict)
            skill.evaluation.thresholds = asdict(self.thresholds)
            self.store.save_iter(run_dir, iter_idx, skill, verdict)

            if verdict.state == "REFINE":
                iter_idx += 1
                next_dir = self.store.iter_skill_dir(run_dir, iter_idx)
                skill = await self.refine(skill, out_dir=next_dir)
                continue

            # Terminal state — pick best skill, attach recommendation, finalize
            final_skill = (
                skill_history[verdict.kept_iter]
                if verdict.kept_iter >= 0
                else skill_history[-1]
            )
            final_skill.evaluation.recommendation = self._verdict_to_recommendation(verdict)
            # Overwrite the verdict on the chosen iter's evaluation with the
            # *terminal* verdict (e.g. MAX_ITER), not the REFINE verdict it
            # carried at the time it was created. Without this, final
            # evaluation.json would say "will refine" while the recommendation
            # says "discard" — contradictory, confusing.
            final_skill.evaluation.verdict = asdict(verdict)
            final_skill.evaluation.thresholds = asdict(self.thresholds)
            self.store.finalize(run_dir, final_skill, verdict)
            # Re-bind the returned Skill to the user's <skill_dir> (the
            # primary, non-historical location). final_skill currently points
            # at iter_<n>/skill_dir/ inside history; rebinding gives the
            # caller a Skill whose `.dir` is the path they asked for.
            rebound = Skill.from_dir(self.skill_dir)
            rebound.evaluation = final_skill.evaluation
            rebound.usage = self.usage
            return rebound

    # ------------------------------------------------------------------
    # Authoring (agentic): generate / refine
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        task: str | None = None,
        trace: str | None = None,
        out_dir: Path,
    ) -> Skill:
        """Author a skill into out_dir using filesystem + python + bash tools."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if task and trace:
            prompt = P.from_task_and_trace(
                task=task,
                trace=trace,
                skill_dir=str(out_dir),
                scripts_enabled=self.scripts_enabled,
            )
        elif trace:
            prompt = P.from_trace(
                trace=trace,
                skill_dir=str(out_dir),
                scripts_enabled=self.scripts_enabled,
            )
        else:
            prompt = P.from_task(
                task=task,
                skill_dir=str(out_dir),
                scripts_enabled=self.scripts_enabled,
            )

        await self._run_author_agent(prompt, out_dir)
        return Skill.from_dir(out_dir)

    async def refine(self, skill: Skill, *, out_dir: Path) -> Skill:
        """Refine a skill: copy prev → out_dir, then let the agent edit in place."""
        out_dir = Path(out_dir)
        # Seed out_dir with the previous skill so the agent can edit in place.
        if out_dir.exists():
            for child in out_dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        _copy_tree(skill.dir, out_dir)

        payload = "\n\n".join(
            f"### Failed case ({r.case.source})\n"
            f"INPUT: {r.case.input}\n"
            f"REFERENCE: {r.case.reference}\n"
            f"CRITERION: {r.case.criterion}\n"
            f"ACTUAL OUTPUT:\n{r.output}\n"
            for r in skill.evaluation.failures
        )
        prompt = P.refine(
            failures=payload,
            skill_dir=str(out_dir),
            scripts_enabled=self.scripts_enabled,
        )

        await self._run_author_agent(prompt, out_dir)
        return Skill.from_dir(out_dir)

    async def _run_author_agent(self, prompt: str, work_dir: Path) -> None:
        """Drive the author/refiner agent.

        Tool set is gated by ``scripts_enabled``:
          - False (default): filesystem only.
          - True: filesystem + python + bash (install / prototype / self-test).

        Iteration cap is **not** set here — chak's default applies. See module
        header for rationale. If the cap is reached, the partially-written
        SKILL.md on disk still flows into evaluation; we swallow the exception
        rather than crash the run.
        """
        tools: list = [FileSystem(workdir=str(work_dir))]
        if self.scripts_enabled:
            tools.extend([Python(), Bash()])
        conv = chak.Conversation(self.model, self.api_key, tools=tools)
        try:
            await conv.asend(prompt)
        except Exception as e:
            # chak raises a plain Exception when the cap is hit; identify
            # by message to avoid swallowing genuine errors.
            if "Max tool call iterations" not in str(e):
                raise
        # Fold authoring cost into run-level usage. conv.stats() works even
        # when the cap was hit — we still paid for those tokens.
        self.usage.authoring += conv.stats()["total_tokens"]

    # ------------------------------------------------------------------
    # Test case generation
    # ------------------------------------------------------------------

    async def make_cases(
        self, *, task: str | None = None, trace: str | None = None
    ) -> list[EvalCase]:
        """Generate evaluation cases (corner-case-biased)."""
        if task:
            cases_, tokens = await cases.from_task(task, self.model, self.api_key, n=self.n_cases)
            self.usage.case_gen += tokens
            return cases_
        # trace-only path requires Phase 4 (trace.py)
        raise NotImplementedError("Trace-only case generation requires Phase 4 (trace.py)")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, skill: Skill, eval_cases: list[EvalCase], iter: int = 0) -> Skill:
        """Run baseline vs with-skill comparison on all evaluation cases."""
        # skill.dir is already on disk (generate/refine wrote it there).
        skill_dir = skill.dir

        with_skill_coros = [self._run_one(c, skill_dir=skill_dir) for c in eval_cases]
        baseline_coros = [self._run_one(c, skill_dir=None) for c in eval_cases]

        with_skill, baseline = await asyncio.gather(
            asyncio.gather(*with_skill_coros),
            asyncio.gather(*baseline_coros),
        )

        skill.evaluation = Evaluation(
            with_skill=list(with_skill),
            baseline=list(baseline),
            iter=iter,
        )
        return skill

    async def _run_one(self, case: EvalCase, *, skill_dir: Path | None) -> EvalResult:
        """Run a single evaluation case.

        Tool injection is gated by ``self.scripts_enabled`` so the evaluation
        runtime mirrors the *deployment* runtime implied by skill authoring:

          - ``scripts_enabled=False`` (pure-doc skill): neither arm gets
            Python/Bash. The skill is evaluated as it will actually be
            consumed — a naked LLM reading SKILL.md.
          - ``scripts_enabled=True``: both arms get Python/Bash. The only
            experimental variable remains ClaudeSkill(skill_dir).

        This keeps lift attribution clean and aligned with how the developer
        intends to ship the skill.
        """
        tools: list = []
        if self.scripts_enabled:
            tools.extend([Python(), Bash()])
        if skill_dir is not None:
            tools.append(ClaudeSkill(str(skill_dir)))
        conv = chak.Conversation(self.model, self.api_key, tools=tools)
        # No explicit iteration cap — chak default applies (see module header).
        t0 = time.time()
        # Cap exceptions are swallowed: an arm getting stuck in a tool loop is
        # a property of *that arm on that case*, not a fatal run-level failure.
        # We mark the case failed, skip judging (output is garbage), and let
        # the rest of the evaluation continue. Tokens already burned are still
        # accounted under eval_arms so cost reporting stays honest.
        try:
            resp = await conv.asend(case.input)
            output = resp.content
            tokens = total_tokens(resp)
            cap_hit = False
        except Exception as e:
            if "Max tool call iterations" not in str(e):
                raise
            output = f"[cap-exceeded] {e}"
            tokens = 0
            cap_hit = True
        latency_ms = int((time.time() - t0) * 1000)
        # Account candidate-arm tokens (with_skill or baseline) under eval_arms,
        # including partial usage from a cap-aborted run.
        self.usage.eval_arms += conv.stats()["total_tokens"]
        if cap_hit:
            return EvalResult(
                case=case,
                output=output,
                passed=False,
                tokens=tokens,
                latency_ms=latency_ms,
            )
        passed, judge_tokens = await self._evaluator(
            output, case, self.judge_model, self.judge_api_key
        )
        self.usage.judging += judge_tokens
        return EvalResult(
            case=case,
            output=output,
            passed=passed,
            tokens=tokens,
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
