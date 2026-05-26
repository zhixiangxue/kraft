"""Persistence: ``<skill_dir>`` (terminal artifact) + ``<skill_dir>.kraft`` (history).

Layout:

    <skill_dir>/                       primary artifact, exposed to runtime
      SKILL.md                           (+ scripts/, references/, examples/
                                          when scripts_enabled=True)
    <skill_dir>.kraft/                 sibling, automatic, holds all history
      <YYYY_MM_DD_HH_MM_SS>/
        cases.json                       generated evaluation cases for this run
        evaluation.json                  run-level evaluation (chosen iter)
        iter_0/skill_dir/                author wrote SKILL.md here
        iter_0/evaluation.json           per-iter evaluation (with verdict embedded)
        iter_1/skill_dir/
        ...

Each ``evaluation.json`` is **self-contained**: it carries the with-skill /
baseline runs *and* the verdict + thresholds that drove the decision. There is
no longer a separate ``verdict.json`` — single source of truth.

The history directory is **outside** the skill root because ClaudeSkill scans
the skill root recursively (including dotfiles) when registering Layer-3
supporting files; nesting history inside would leak iteration leftovers into
the runtime view of the skill.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from kraft.decision import Verdict
from kraft.skill import Skill, EvalCase


class Store:
    """Manages a single skill's lifetime artifacts.

    A Store is bound to one ``skill_dir`` for its whole life. The constructor
    enforces the "no silent clobber" contract: a non-empty ``skill_dir``
    raises ``FileExistsError`` unless ``overwrite=True`` is passed. The
    sibling ``<skill_dir>.kraft`` history directory is created/extended
    unconditionally — multiple runs append timestamped subdirs.
    """

    def __init__(self, skill_dir: Path, *, overwrite: bool = False):
        self.skill_dir = Path(skill_dir).resolve()
        self.history_dir = self.skill_dir.with_name(self.skill_dir.name + ".kraft")

        # Guard against accidental overwrite. An empty skill_dir is fine
        # (developers may have pre-created it); a populated one is not.
        if self.skill_dir.exists() and any(self.skill_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"skill_dir already exists and is non-empty: {self.skill_dir}\n"
                    f"Pass overwrite=True to replace, or pick a fresh path."
                )
            shutil.rmtree(self.skill_dir)

        self.skill_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # History (timestamped run inside <skill_dir>.kraft)
    # ------------------------------------------------------------------

    def create_run_dir(self) -> Path:
        """Create a timestamped run directory under the history root."""
        ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = self.history_dir / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_cases(self, run_dir: Path, cases: list[EvalCase]) -> None:
        """Save generated evaluation cases to ``<run_dir>/cases.json``."""
        data = [c.model_dump() for c in cases]
        (run_dir / "cases.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def iter_skill_dir(self, run_dir: Path, iter: int) -> Path:
        """Allocate (and create) the working directory for iteration ``iter``.

        Author/refiner agents write SKILL.md (and any supporting files) here.
        Evaluation also reads from here. Path: ``<run_dir>/iter_<n>/skill_dir/``.
        """
        skill_dir = run_dir / f"iter_{iter}" / "skill_dir"
        skill_dir.mkdir(parents=True, exist_ok=True)
        return skill_dir

    def save_iter(self, run_dir: Path, iter: int, skill: Skill, verdict: Verdict) -> None:
        """Persist iteration metadata as ``iter_<n>/evaluation.json``.

        SKILL.md and any supporting files were written by the author agent
        directly into ``iter_<n>/skill_dir/``; no extra dump needed here.
        The evaluation already carries verdict + thresholds (kraft sets them
        before calling), so this file is the single source of truth for the
        iteration's decision context.
        """
        iter_dir = run_dir / f"iter_{iter}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        if skill.evaluation:
            (iter_dir / "evaluation.json").write_text(
                skill.evaluation.model_dump_json(indent=2), encoding="utf-8"
            )

    # ------------------------------------------------------------------
    # Finalize: promote the chosen skill_dir to the user's <skill_dir>
    # ------------------------------------------------------------------

    def finalize(self, run_dir: Path, skill: Skill, verdict: Verdict) -> None:
        """Copy the chosen iteration's skill files into the user's ``<skill_dir>``.

        Also writes the run-level ``evaluation.json`` into the history's
        ``run_dir`` (NOT into ``skill_dir``, which must stay clean for
        ClaudeSkill). The evaluation is self-contained — verdict and
        thresholds are embedded fields.
        """
        # Wipe the user-facing skill_dir so it ends up reflecting *only* the
        # chosen iteration. (We already verified at construction time that
        # the developer is OK with overwriting.)
        for child in self.skill_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

        for src in skill.dir.rglob("*"):
            if src.is_file():
                rel = src.relative_to(skill.dir)
                dst = self.skill_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        if skill.evaluation:
            (run_dir / "evaluation.json").write_text(
                skill.evaluation.model_dump_json(indent=2), encoding="utf-8"
            )
