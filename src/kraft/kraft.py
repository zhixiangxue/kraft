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
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Awaitable, Callable

import chak
from chak.tools.skills import ClaudeSkill
from chak.tools.std import Bash, FileSystem, Python

from kraft import cases as cases_mod
from kraft import prompts as P
from kraft.decision import Thresholds, Verdict, decide
from kraft.skill import EvalCase, EvalResult, Evaluation, PhaseTokens, Skill, TokenUsage, io_tokens
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


def _dump_conv(conv: chak.Conversation, path: Path) -> None:
    """Persist a conversation's full message log as JSON.

    Uses chak's ``Conversation.dump()`` which serialises every message
    (system / user / assistant / tool) into a list of plain dicts. The file
    sits under the run's history directory so a reader can audit exactly
    what was sent and received for every billable token.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = conv.dump()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def kraft(
    *,
    skill_dir: str | Path,
    task: str | None = None,
    trace: str | None = None,
    model: str,
    api_key: str,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    overwrite: bool = False,
    scripts_enabled: bool = False,
    **kw,
) -> Skill:
    """Top-level one-liner: generate a skill, verify it, return it.

    ``skill_dir``       — required. Final SKILL.md (+ scripts/) lands here.
    ``api_key``         — **required**. The library does not read environment
                          variables on your behalf — the caller is
                          responsible for sourcing the credential (e.g. via
                          ``os.environ["OPENAI_API_KEY"]``). This keeps the
                          provider → env-var mapping explicit at the boundary
                          rather than buried in library internals.
    ``overwrite``       — if False (default) and ``skill_dir`` is non-empty,
                          raises FileExistsError. Pass True to clobber.
    ``judge_api_key``   — required when ``judge_model`` is on a different
                          provider than ``model``; reuses ``api_key`` when
                          judge and candidate share a provider.
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


async def evaluate(
    *,
    skill_dir: str | Path,
    model: str,
    api_key: str,
    cases: list[str] | list[EvalCase] | None = None,
    n_cases: int = 10,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    scripts_enabled: bool = False,
) -> Skill:
    """Evaluate an existing skill without authoring or refinement.

    Use this to quickly assess whether a skill (e.g. downloaded from a
    marketplace or generated externally) is effective. The function runs
    a single baseline-vs-with-skill comparison and returns the Skill with
    evaluation metrics attached.

    ``skill_dir``       — must exist and contain SKILL.md.
    ``cases``           — evaluation cases. Three modes:
                          * ``None`` (default): auto-generate from SKILL.md
                            content, producing ``n_cases`` synthetic cases.
                          * ``list[str]``: natural-language descriptions;
                            kraft structures them via LLM. Cases where the
                            expected output cannot be inferred are discarded.
                          * ``list[EvalCase]``: pre-structured; used as-is.
    ``n_cases``         — number of cases to auto-generate (ignored when
                          ``cases`` is provided).
    ``scripts_enabled`` — controls tool injection symmetry during evaluation.
    """
    skill = Skill.from_dir(skill_dir)
    usage = TokenUsage()

    # Resolve judge credentials.
    _judge_model = judge_model or model
    if judge_api_key is not None:
        _judge_api_key = judge_api_key
    elif _judge_model == model:
        _judge_api_key = api_key
    else:
        raise ValueError(
            f"judge_model={_judge_model!r} differs from model={model!r}; "
            "pass judge_api_key explicitly."
        )

    # --- Resolve cases --------------------------------------------------------
    eval_cases: list[EvalCase]

    if cases is None:
        # Auto-generate from SKILL.md full text.
        task_text = skill.render()
        eval_cases, conv = await cases_mod.from_task(task_text, model, api_key, n=n_cases)
        usage.case_gen.add_conv(conv)
    elif cases and isinstance(cases[0], str):
        # Natural language → structured via LLM.
        eval_cases, conv = await cases_mod.from_natural_language(
            cases, model, api_key  # type: ignore[arg-type]
        )
        usage.case_gen.add_conv(conv)
    else:
        # Already structured EvalCase objects.
        eval_cases = list(cases)  # type: ignore[arg-type]

    if not eval_cases:
        raise ValueError(
            "No valid evaluation cases available. If you provided natural-language "
            "cases, ensure each describes an expected output."
        )

    # --- Run evaluation (no Store, no authoring) ------------------------------
    evaluator = _EvalRunner(
        model=model,
        api_key=api_key,
        judge_model=_judge_model,
        judge_api_key=_judge_api_key,
        scripts_enabled=scripts_enabled,
        usage=usage,
    )
    skill = await evaluator.run(skill, eval_cases)
    skill.usage = usage
    return skill


class _EvalRunner:
    """Lightweight evaluation-only runner (no Store, no authoring)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        judge_model: str,
        judge_api_key: str,
        scripts_enabled: bool,
        usage: TokenUsage,
    ):
        self.model = model
        self.api_key = api_key
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key
        self.scripts_enabled = scripts_enabled
        self.usage = usage

    async def run(self, skill: Skill, eval_cases: list[EvalCase]) -> Skill:
        """Run baseline vs with-skill comparison."""
        skill_dir = skill.dir
        with_skill_coros = [
            self._run_one(c, skill_dir=skill_dir)
            for c in eval_cases
        ]
        baseline_coros = [
            self._run_one(c, skill_dir=None)
            for c in eval_cases
        ]
        with_skill_results, baseline_results = await asyncio.gather(
            asyncio.gather(*with_skill_coros),
            asyncio.gather(*baseline_coros),
        )
        skill.evaluation = Evaluation(
            with_skill=list(with_skill_results),
            baseline=list(baseline_results),
            iter=0,
        )
        return skill

    async def _run_one(self, case: EvalCase, *, skill_dir: Path | None) -> EvalResult:
        """Run a single evaluation case."""
        tools: list = []
        if self.scripts_enabled:
            tools.extend([Python(), Bash()])
        if skill_dir is not None:
            tools.append(ClaudeSkill(str(skill_dir)))
        conv = chak.Conversation(self.model, self.api_key, tools=tools)
        t0 = time.time()
        try:
            resp = await conv.asend(case.input)
            output = resp.content
            in_tok, out_tok = io_tokens(resp)
            cap_hit = False
        except Exception as e:
            if "Max tool call iterations" not in str(e):
                raise
            output = f"[cap-exceeded] {e}"
            in_tok, out_tok = 0, 0
            cap_hit = True
        latency_ms = int((time.time() - t0) * 1000)
        self.usage.eval_arms.add_conv(conv)
        if cap_hit:
            return EvalResult(
                case=case, output=output, passed=False,
                input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency_ms,
            )
        passed, judge_conv = await llm_judge(
            output, case, self.judge_model, self.judge_api_key
        )
        self.usage.judging.add_conv(judge_conv)
        return EvalResult(
            case=case, output=output, passed=passed,
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency_ms,
        )


class Kraft:
    """Mid-level API: configurable skill generation and verification engine."""

    def __init__(
        self,
        model: str,
        *,
        skill_dir: str | Path,
        api_key: str,
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
        self.api_key = api_key
        self.judge_model = judge_model or model

        # Judge api_key: caller must be explicit about cross-provider judges.
        # Same-provider reuse is the only implicit path — a deliberate
        # convenience; cross-provider must come in via ``judge_api_key``.
        if judge_api_key is not None:
            self.judge_api_key = judge_api_key
        elif self.judge_model == self.model:
            self.judge_api_key = self.api_key
        else:
            raise ValueError(
                f"judge_model={self.judge_model!r} differs from model={self.model!r}; "
                "pass judge_api_key explicitly (the library does not resolve "
                "keys from the environment)."
            )

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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, *, task: str | None = None, trace: str | None = None) -> Skill:
        """Full loop: elaborate → generate → evaluate → refine → decide."""
        assert task or trace, "At least one of task or trace must be provided"
        run_dir = self.store.create_run_dir()
        self._run_dir = run_dir
        # All conversation logs land here, mirroring the iter_<n>/ layout so
        # case_gen / authoring / eval_arms / judging can be cross-referenced
        # against per-iter artifacts. See _dump_conv().
        self._messages_dir = run_dir / "messages"
        self._messages_dir.mkdir(parents=True, exist_ok=True)

        # Step 0: elaborate the user's brief task into a structured spec.
        # This is always run — if the task is already detailed, elaboration
        # normalises format and fills minor gaps at negligible token cost.
        if task:
            task = await self._elaborate_task(task)

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
            rebound = Skill.from_dir(self.skill_dir, strict=False)
            rebound.evaluation = final_skill.evaluation
            rebound.usage = self.usage
            return rebound

    # ------------------------------------------------------------------
    # Spec elaboration (pipeline step 0)
    # ------------------------------------------------------------------

    async def _elaborate_task(self, task: str) -> str:
        """Expand a brief task into a structured spec via a single LLM call.

        The elaborated spec replaces the raw user task for all downstream
        phases (case_gen, author, refine).  It is persisted to
        ``<run_dir>/spec.md`` and its conversation log to
        ``<run_dir>/messages/elaboration.json`` for audit.
        """
        prompt = P.elaborate_task(task=task, scripts_enabled=self.scripts_enabled)
        conv = chak.Conversation(self.model, self.api_key)
        resp = await conv.asend(prompt)
        elaborated = resp.content
        # Persist the elaborated spec as a readable Markdown file.
        (self._run_dir / "spec.md").write_text(elaborated, encoding="utf-8")
        # Token accounting + message dump.
        self.usage.spec_elab.add_conv(conv)
        _dump_conv(conv, self._messages_dir / "elaboration.json")
        return elaborated

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

        await self._run_author_agent(prompt, out_dir, dump_label="authoring")
        return Skill.from_dir(out_dir, strict=False)

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

        await self._run_author_agent(prompt, out_dir, dump_label="authoring")
        return Skill.from_dir(out_dir, strict=False)

    async def _run_author_agent(self, prompt: str, work_dir: Path, *, dump_label: str = "authoring") -> None:
        """Drive the author/refiner agent.

        Tool set is gated by ``scripts_enabled``:
          - False (default): filesystem only.
          - True: filesystem + python + bash (install / prototype / self-test).

        Iteration cap is **not** set here — chak's default applies. See module
        header for rationale. If the cap is reached, the partially-written
        SKILL.md on disk still flows into evaluation; we swallow the exception
        rather than crash the run.

        ``work_dir`` doubles as the iter-anchored location for the message dump
        — we drop ``messages/iter_<n>/<dump_label>.json`` next to the iter's
        skill_dir so the audit trail tracks the iteration tree.
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
        # Fold authoring cost into run-level usage with input/output split.
        # conv.stats() works even when the cap was hit — we still paid for
        # those tokens.
        self.usage.authoring.add_conv(conv)
        # Persist the full message log under <run>/messages/iter_<n>/<label>.json
        iter_name = work_dir.parent.name  # e.g. "iter_0"
        _dump_conv(conv, self._messages_dir / iter_name / f"{dump_label}.json")

    # ------------------------------------------------------------------
    # Test case generation
    # ------------------------------------------------------------------

    async def make_cases(
        self, *, task: str | None = None, trace: str | None = None
    ) -> list[EvalCase]:
        """Generate evaluation cases (corner-case-biased)."""
        if task:
            cases_, conv = await cases_mod.from_task(task, self.model, self.api_key, n=self.n_cases)
            self.usage.case_gen.add_conv(conv)
            _dump_conv(conv, self._messages_dir / "case_gen.json")
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

        with_skill_coros = [
            self._run_one(c, idx=i, arm="with_skill", iter=iter, skill_dir=skill_dir)
            for i, c in enumerate(eval_cases)
        ]
        baseline_coros = [
            self._run_one(c, idx=i, arm="baseline", iter=iter, skill_dir=None)
            for i, c in enumerate(eval_cases)
        ]

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

    async def _run_one(
        self,
        case: EvalCase,
        *,
        idx: int,
        arm: str,
        iter: int,
        skill_dir: Path | None,
    ) -> EvalResult:
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

        ``arm`` ("with_skill" | "baseline") and ``idx`` are used purely to
        name the dumped message logs; both candidate-arm and judge
        conversations land under ``messages/iter_<n>/`` for audit.
        """
        msg_dir = self._messages_dir / f"iter_{iter}"
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
            in_tok, out_tok = io_tokens(resp)
            cap_hit = False
        except Exception as e:
            if "Max tool call iterations" not in str(e):
                raise
            output = f"[cap-exceeded] {e}"
            in_tok, out_tok = 0, 0
            cap_hit = True
        latency_ms = int((time.time() - t0) * 1000)
        # Account candidate-arm tokens (with_skill or baseline) under eval_arms,
        # with input/output split for accurate cost reporting. Includes partial
        # usage from a cap-aborted run.
        self.usage.eval_arms.add_conv(conv)
        _dump_conv(conv, msg_dir / f"eval_{arm}_case_{idx}.json")
        if cap_hit:
            return EvalResult(
                case=case,
                output=output,
                passed=False,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
            )
        passed, judge_conv = await self._evaluator(
            output, case, self.judge_model, self.judge_api_key
        )
        self.usage.judging.add_conv(judge_conv)
        _dump_conv(judge_conv, msg_dir / f"judge_{arm}_case_{idx}.json")
        return EvalResult(
            case=case,
            output=output,
            passed=passed,
            input_tokens=in_tok,
            output_tokens=out_tok,
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
