"""Rich-formatted run report — example-only helper.

Lives under ``examples/`` rather than ``src/kraft/`` on purpose: pretty
printing is a *presentation* concern, not a library concern. The kraft
library produces a ``Skill`` carrying ``evaluation`` and ``usage``; how
callers display them (rich, plain print, JSON, logging, a web UI…) is up
to them. Keeping rich out of the library means library users don't pay
the dependency cost for a UX choice they may not want.

If multiple examples want the same nicely-formatted summary, they share
this helper via ``from _report import print_run_report``.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kraft.skill import Evaluation, Skill, TokenUsage


def _fmt(n: int) -> str:
    """Thousands-separated integer; tokens get into 6+ digits fast."""
    return f"{n:,}"


def usage_table(usage: TokenUsage, *, title: str = "Token Usage") -> Table:
    """Build a rich.Table breaking tokens down by phase and sub-bucket.

    Layout: 4 leaf rows + 1 grand total. No per-phase subtotals — with only
    2 buckets per phase, the eye sums them faster than reading a dedicated
    aggregate row, and dropping the subtotal also frees the % column from
    the misleading "sum-of-column" reading. Phase column at the far left,
    with ``end_section`` between phases for visual grouping.
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Phase", style="cyan")
    table.add_column("Bucket")
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("% of total", justify="right")

    total = usage.total or 1  # avoid div-by-zero on dry runs

    def pct(n: int) -> str:
        return f"{n / total:.1%}"

    table.add_row("generation", "case_gen", _fmt(usage.case_gen), pct(usage.case_gen))
    table.add_row(
        "generation", "authoring",
        _fmt(usage.authoring), pct(usage.authoring),
        end_section=True,
    )
    table.add_row("evaluation", "eval_arms", _fmt(usage.eval_arms), pct(usage.eval_arms))
    table.add_row(
        "evaluation", "judging",
        _fmt(usage.judging), pct(usage.judging),
        end_section=True,
    )
    table.add_row(
        "[bold magenta]TOTAL[/bold magenta]",
        "",
        f"[bold magenta]{_fmt(usage.total)}[/bold magenta]",
        "[bold magenta]100.0%[/bold magenta]",
    )
    return table


def print_run_report(skill: Skill, console: Console | None = None) -> None:
    """One-call summary of a kraft run: evaluation verdict + token usage table.

    Reads everything from ``skill`` itself — ``skill.evaluation`` carries the
    verdict, ``skill.usage`` carries token cost.
    """
    console = console or Console()
    ev: Evaluation | None = skill.evaluation
    console.rule(f"[bold]{skill.name}[/bold]")
    console.print(f"[dim]{skill.description}[/dim]\n")
    if ev is not None:
        console.print(ev.summary())
        console.print()
    if skill.usage is not None:
        console.print(usage_table(skill.usage))
    console.print(f"\nSkill dir: {skill.dir}")
    extras = skill.files()
    if extras:
        console.print(f"Supporting files ({len(extras)}):")
        for p in extras:
            console.print(f"  - {p}")
    else:
        console.print("Supporting files: [dim](none — pure SKILL.md)[/dim]")
    console.print(f"History: {skill.dir.with_name(skill.dir.name + '.kraft')}")
