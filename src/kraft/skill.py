"""Core data models: Skill, TestCase, RunResult, Report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case with reference answer for LLM-as-judge verification."""

    input: str
    reference: str
    criterion: str
    source: str  # "synthetic" | "trace:line42"


class RunResult(BaseModel):
    """Result of running a single test case."""

    case: TestCase
    output: str  # full raw output, never truncated
    passed: bool
    tokens: int  # cumulative input + output tokens for the conversation
    latency_ms: int


class Report(BaseModel):
    """Evaluation report comparing baseline vs with-skill runs."""

    with_skill: list[RunResult]
    baseline: list[RunResult]
    iter: int
    recommendation: str = ""  # filled after decision: "keep" | "discard" | "no-skill-needed"

    @property
    def pass_rate_with_skill(self) -> float:
        if not self.with_skill:
            return 0.0
        return sum(1 for r in self.with_skill if r.passed) / len(self.with_skill)

    @property
    def pass_rate_baseline(self) -> float:
        if not self.baseline:
            return 0.0
        return sum(1 for r in self.baseline if r.passed) / len(self.baseline)

    @property
    def lift(self) -> float:
        return self.pass_rate_with_skill - self.pass_rate_baseline

    @property
    def failures(self) -> list[RunResult]:
        """Failed cases from with_skill runs (used by refine)."""
        return [r for r in self.with_skill if not r.passed]

    @property
    def cost_per_correct_with_skill(self) -> float:
        passed = sum(1 for r in self.with_skill if r.passed)
        total_tokens = sum(r.tokens for r in self.with_skill)
        return total_tokens / passed if passed > 0 else float("inf")

    @property
    def cost_per_correct_baseline(self) -> float:
        passed = sum(1 for r in self.baseline if r.passed)
        total_tokens = sum(r.tokens for r in self.baseline)
        return total_tokens / passed if passed > 0 else float("inf")

    @property
    def cost_ratio(self) -> float:
        """Token cost ratio: with_skill / baseline. Falls back to cpc comparison."""
        baseline_tokens = sum(r.tokens for r in self.baseline)
        skill_tokens = sum(r.tokens for r in self.with_skill)
        if baseline_tokens == 0:
            return float("inf")
        return skill_tokens / baseline_tokens

    def summary(self) -> str:
        base = self.pass_rate_baseline
        skill = self.pass_rate_with_skill
        ratio = self.cost_ratio
        # cost_ratio is a health flag, not a gate. Tag visually so users
        # can decide whether the token spend is worth it.
        if ratio > 10.0:
            cost_tag = "⚠️ insane"
        elif ratio > 3.0:
            cost_tag = "⚠️ high"
        else:
            cost_tag = "ok"
        return (
            f"baseline {base:.0%} → {skill:.0%} ({self.lift:+.0%}) | "
            f"cost_ratio {ratio:.2f}× [{cost_tag}] | "
            f"recommendation: {self.recommendation}"
        )


class Skill(BaseModel):
    """A skill with YAML frontmatter metadata and markdown body."""

    name: str  # kebab-case, e.g. "cron-expressions"
    description: str = Field(max_length=200)
    body: str  # markdown content
    report: Report | None = None

    @classmethod
    def parse(cls, text: str) -> Skill:
        """Parse YAML frontmatter + markdown body into a Skill."""
        post = frontmatter.loads(text)
        desc = str(post.metadata.get("description", ""))
        if len(desc) > 200:
            desc = desc[:197] + "..."
        return cls(
            name=post.metadata.get("name", ""),
            description=desc,
            body=post.content,
        )

    def render(self) -> str:
        """Serialize to SKILL.md text (YAML frontmatter + body)."""
        post = frontmatter.Post(self.body)
        post.metadata["name"] = self.name
        post.metadata["description"] = self.description
        return frontmatter.dumps(post)

    def materialize(self, dir: Path) -> Path:
        """Write SKILL.md to dir/SKILL.md and return dir.

        Used before evaluate — ClaudeSkill(str(dir)) needs a directory.
        """
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "SKILL.md").write_text(self.render(), encoding="utf-8")
        return dir

    def save(self, dir: str | Path) -> None:
        """Write SKILL.md + report.json to target directory for production use."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "SKILL.md").write_text(self.render(), encoding="utf-8")
        if self.report:
            (dir / "report.json").write_text(
                self.report.model_dump_json(indent=2), encoding="utf-8"
            )


def total_tokens(resp: Any) -> int:
    """Extract cumulative input+output tokens from a chak response.

    chak accumulates usage across multi-turn tool round-trips.
    Access via resp.metadata.usage.prompt_tokens / completion_tokens.
    """
    usage = getattr(getattr(resp, "metadata", None), "usage", None)
    if usage is None:
        return 0
    return (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
