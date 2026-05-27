"""Core data models: Skill, EvalCase, EvalResult, Evaluation.

Skill is **directory-native**: its on-disk form is a directory containing
SKILL.md (required) plus arbitrary supporting files (scripts/, references/, …).
The in-memory Skill object is just a typed view over that directory — there is
no separate "string" representation. This aligns with chak's ClaudeSkill, which
also takes a directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel, Field

# Validation constants for skill directories.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_ALLOWED_ROOT_DIRS = {"scripts", "references", "examples", "assets"}
_PLACEHOLDER_MARKERS = ("[todo", "todo:")


class PhaseTokens(BaseModel):
    """Input / output token split for a single accounting bucket.

    Stored separately because input and output are billed at different
    unit prices (typically output is 3-5x more expensive). Merging into a
    single ``total`` would silently lose the cost basis.
    """

    input: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input += int(input_tokens or 0)
        self.output += int(output_tokens or 0)

    def add_conv(self, conv: Any) -> None:
        """Fold a chak Conversation's cumulative usage into this bucket."""
        s = conv.stats()
        self.add(s.get("input_tokens", 0), s.get("output_tokens", 0))


class TokenUsage(BaseModel):
    """Run-level token accounting, partitioned by phase, with input/output split.

    Three top-level buckets matching the user-facing mental model:
      - elaboration = spec_elab  (expanding brief task → structured spec)
      - generation  = case_gen + authoring  (everything that *produces* the skill)
      - evaluation  = eval_arms + judging   (everything that *measures* it)

    Each bucket is a :class:`PhaseTokens` carrying ``input`` and ``output``
    counts so callers can apply the correct per-token price (output is
    typically 3-5x the input rate). Aggregate ``input`` / ``output`` /
    ``total`` are exposed as properties for reporting.
    """

    spec_elab: PhaseTokens = Field(default_factory=PhaseTokens)
    case_gen: PhaseTokens = Field(default_factory=PhaseTokens)
    authoring: PhaseTokens = Field(default_factory=PhaseTokens)
    eval_arms: PhaseTokens = Field(default_factory=PhaseTokens)
    judging: PhaseTokens = Field(default_factory=PhaseTokens)

    @property
    def elaboration(self) -> PhaseTokens:
        return self.spec_elab

    @property
    def generation(self) -> PhaseTokens:
        return PhaseTokens(
            input=self.case_gen.input + self.authoring.input,
            output=self.case_gen.output + self.authoring.output,
        )

    @property
    def evaluation(self) -> PhaseTokens:
        return PhaseTokens(
            input=self.eval_arms.input + self.judging.input,
            output=self.eval_arms.output + self.judging.output,
        )

    @property
    def input(self) -> int:
        return (
            self.spec_elab.input
            + self.case_gen.input + self.authoring.input
            + self.eval_arms.input + self.judging.input
        )

    @property
    def output(self) -> int:
        return (
            self.spec_elab.output
            + self.case_gen.output + self.authoring.output
            + self.eval_arms.output + self.judging.output
        )

    @property
    def total(self) -> int:
        return self.input + self.output


class EvalCase(BaseModel):
    """A single evaluation case with reference answer for LLM-as-judge verification."""

    input: str
    reference: str
    criterion: str
    source: str  # "synthetic" | "trace:line42"


class EvalResult(BaseModel):
    """Result of running a single evaluation case.

    Tokens are split into ``input_tokens`` and ``output_tokens`` so callers
    can compute cost at distinct provider rates. ``tokens`` (the sum) is
    exposed as a derived property for legacy aggregations.
    """

    case: EvalCase
    output: str  # full raw output, never truncated
    passed: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Evaluation(BaseModel):
    """Result of one evaluation round: cases + with-skill results + baseline
    results + decision context. Persisted as ``evaluation.json``."""

    with_skill: list[EvalResult]
    baseline: list[EvalResult]
    iter: int
    recommendation: str = ""  # filled after decision: "keep" | "discard" | "no-skill-needed"
    # Decision context — makes evaluation.json self-explanatory. ``verdict``
    # carries state/reason/kept_iter from kraft.decision.Verdict; ``thresholds``
    # records the gate values that produced the verdict, so a reader can
    # reproduce the decision from a single file (no separate verdict.json).
    verdict: dict | None = None
    thresholds: dict | None = None

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
    def failures(self) -> list[EvalResult]:
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
        head = (
            f"baseline {base:.0%} → {skill:.0%} ({self.lift:+.0%}) | "
            f"cost_ratio {ratio:.2f}× [{cost_tag}] | "
            f"recommendation: {self.recommendation}"
        )
        # Surface the decision reason so a glance at summary() answers
        # "why this recommendation?" — no need to inspect evaluation.json.
        if self.verdict and self.verdict.get("reason"):
            head += f"\n  reason: {self.verdict['reason']}"
        return head


class Skill(BaseModel):
    """A skill rooted at a directory.

    The directory MUST contain SKILL.md with YAML frontmatter (name + description
    are required). It MAY also contain supporting files — scripts/, references/,
    examples/ — that the runtime exposes via Layer 3 progressive disclosure.

    There is no `parse(text)` / `render()` API anymore: the directory IS the
    skill. Construct via `Skill.from_dir(path)`. Persistence is handled by
    Kraft's Store, which writes the chosen iteration's files directly into
    the developer-supplied ``skill_dir`` — callers no longer need to copy.
    """

    dir: Path
    evaluation: Evaluation | None = None
    usage: TokenUsage | None = None  # run-level token cost; populated by Kraft.run()

    # ---- factories ---------------------------------------------------------

    @classmethod
    def from_dir(cls, dir: str | Path, *, strict: bool = True) -> "Skill":
        """Bind a Skill to an existing directory containing SKILL.md.

        When ``strict=True`` (default), full validation is applied:
          - name must be hyphen-case, ≤64 chars
          - description must be non-empty and not contain TODO placeholders
          - root directory may only contain SKILL.md + allowed subdirs
          - symlinks are rejected

        Pass ``strict=False`` to skip structural checks (e.g. during
        authoring iterations where the skill is still being written).
        """
        d = Path(dir).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"Skill directory not found: {d}")
        if not (d / "SKILL.md").is_file():
            raise FileNotFoundError(f"SKILL.md not found in {d}")
        skill = cls(dir=d)
        # Validate frontmatter eagerly — fail fast on malformed authoring.
        meta = skill._frontmatter()
        if not meta.get("name"):
            raise ValueError(f"SKILL.md missing 'name' in frontmatter: {d}")
        if not meta.get("description"):
            raise ValueError(f"SKILL.md missing 'description' in frontmatter: {d}")

        if strict:
            cls._validate_strict(d, meta)
        return skill

    @staticmethod
    def _validate_strict(d: Path, meta: dict) -> None:
        """Extended validation: naming, structure, symlinks."""
        name = str(meta["name"]).strip()
        description = str(meta["description"]).strip()

        # Name format: hyphen-case, letters/digits/hyphens only.
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                f"Skill name '{name}' must be hyphen-case "
                "(lowercase letters, digits, and single hyphens only)"
            )
        if len(name) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"Skill name '{name}' is too long ({len(name)} chars, max {_MAX_NAME_LENGTH})"
            )

        # Description quality: no TODO placeholders.
        desc_lower = description.lower()
        if any(m in desc_lower for m in _PLACEHOLDER_MARKERS):
            raise ValueError(
                f"Skill description still contains TODO placeholder text: {d}"
            )
        # Root directory: only SKILL.md + allowed subdirectories.
        for child in d.iterdir():
            if child.name == "SKILL.md":
                continue
            if child.is_symlink():
                raise ValueError(
                    f"Symlinks not allowed in skill directory: {child}"
                )
            if child.is_dir() and child.name in _ALLOWED_ROOT_DIRS:
                continue
            raise ValueError(
                f"Unexpected item in skill root: '{child.name}'. "
                f"Allowed: SKILL.md + directories: {sorted(_ALLOWED_ROOT_DIRS)}"
            )

    # ---- views over disk ---------------------------------------------------

    def _frontmatter(self) -> dict[str, Any]:
        post = frontmatter.loads(
            (self.dir / "SKILL.md").read_text(encoding="utf-8")
        )
        return dict(post.metadata)

    @property
    def name(self) -> str:
        return str(self._frontmatter().get("name", ""))

    @property
    def description(self) -> str:
        return str(self._frontmatter().get("description", ""))

    @property
    def body(self) -> str:
        """Markdown body of SKILL.md (frontmatter stripped)."""
        post = frontmatter.loads(
            (self.dir / "SKILL.md").read_text(encoding="utf-8")
        )
        return post.content

    def render(self) -> str:
        """Full SKILL.md text (frontmatter + body), as written on disk."""
        return (self.dir / "SKILL.md").read_text(encoding="utf-8")

    def files(self) -> list[Path]:
        """All non-SKILL.md files, as paths relative to self.dir."""
        out: list[Path] = []
        skill_md = self.dir / "SKILL.md"
        for p in self.dir.rglob("*"):
            if p.is_file() and p.resolve() != skill_md.resolve():
                out.append(p.relative_to(self.dir))
        return sorted(out)


def io_tokens(resp: Any) -> tuple[int, int]:
    """Extract (input, output) token counts from a chak response.

    chak accumulates usage across multi-turn tool round-trips. Access via
    ``resp.metadata.usage.prompt_tokens`` / ``completion_tokens``.
    """
    usage = getattr(getattr(resp, "metadata", None), "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )
