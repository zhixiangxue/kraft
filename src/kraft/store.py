"""Persistence: runs/<timestamp>/ directory structure."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from kraft.skill import Skill, TestCase, Report
from kraft.decision import Verdict


class Store:
    """Manages the runs/ directory for persisting evaluation history."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self) -> Path:
        """Create a timestamped run directory."""
        ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = self.runs_dir / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_tests(self, run_dir: Path, tests: list[TestCase]) -> None:
        """Save generated test cases to tests.json."""
        data = [t.model_dump() for t in tests]
        (run_dir / "tests.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def materialize_skill(self, run_dir: Path, iter: int, skill: Skill) -> Path:
        """Write SKILL.md to iter directory and return the skill_dir path."""
        skill_dir = run_dir / f"iter_{iter}" / "skill_dir"
        skill.materialize(skill_dir)
        return skill_dir

    def save_iter(self, run_dir: Path, iter: int, skill: Skill, verdict: Verdict) -> None:
        """Save iteration artifacts: skill.md, report.json, verdict.json."""
        iter_dir = run_dir / f"iter_{iter}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Skill markdown
        (iter_dir / "skill.md").write_text(skill.render(), encoding="utf-8")

        # Report
        if skill.report:
            (iter_dir / "report.json").write_text(
                skill.report.model_dump_json(indent=2), encoding="utf-8"
            )

        # Verdict
        (iter_dir / "verdict.json").write_text(
            json.dumps(asdict(verdict), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def finalize(self, run_dir: Path, skill: Skill, verdict: Verdict) -> None:
        """Create final/ directory with the chosen skill and verdict."""
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        (final_dir / "skill.md").write_text(skill.render(), encoding="utf-8")
        (final_dir / "verdict.json").write_text(
            json.dumps(asdict(verdict), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if skill.report:
            (final_dir / "report.json").write_text(
                skill.report.model_dump_json(indent=2), encoding="utf-8"
            )
