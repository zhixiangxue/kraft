"""Rich-formatted run reports — example-only helpers.

Lives under ``examples/`` rather than ``src/kraft/`` on purpose: pretty
printing is a *presentation* concern, not a library concern. The kraft
library produces a ``Skill`` carrying ``evaluation`` and ``usage``; how
callers display them (rich, plain print, JSON, logging, a web UI…) is up
to them. Keeping rich out of the library means library users don't pay
the dependency cost for a UX choice they may not want.

Two report styles:
  - ``print_run_report``  — authoring flow (kraft()), token-focused
  - ``print_eval_report`` — evaluate-only flow, case-breakdown-focused
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kraft.skill import Evaluation, PhaseTokens, Skill, TokenUsage


def _fmt(n: int) -> str:
    """Thousands-separated integer; tokens get into 6+ digits fast."""
    return f"{n:,}"


def usage_table(usage: TokenUsage, *, title: str = "Token Usage") -> Table:
    """Build a rich.Table breaking tokens down by phase, with input/output split.

    Layout: Phase | Bucket | Input | Output | Total | % of total. Splitting
    input vs output is non-cosmetic — providers price them at different
    rates (output typically 3-5x input), so a single ``Tokens`` column would
    erase the cost basis. Sub-buckets (case_gen, authoring, eval_arms,
    judging) stay as leaf rows; phase column groups them visually with an
    ``end_section`` between phases.
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Phase", style="cyan")
    table.add_column("Bucket")
    table.add_column("Input", justify="right", style="blue")
    table.add_column("Output", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="green")
    table.add_column("% of total", justify="right")

    total = usage.total or 1  # avoid div-by-zero on dry runs

    def _row(phase: str, bucket: str, t: PhaseTokens, *, end_section: bool = False) -> None:
        table.add_row(
            phase,
            bucket,
            _fmt(t.input),
            _fmt(t.output),
            _fmt(t.total),
            f"{t.total / total:.1%}",
            end_section=end_section,
        )

    _row("elaboration", "spec_elab", usage.spec_elab, end_section=True)
    _row("generation", "case_gen", usage.case_gen)
    _row("generation", "authoring", usage.authoring, end_section=True)
    _row("evaluation", "eval_arms", usage.eval_arms)
    _row("evaluation", "judging", usage.judging, end_section=True)

    table.add_row(
        "[bold magenta]TOTAL[/bold magenta]",
        "",
        f"[bold magenta]{_fmt(usage.input)}[/bold magenta]",
        f"[bold magenta]{_fmt(usage.output)}[/bold magenta]",
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


def print_eval_report(skill: Skill, console: Console | None = None) -> None:
    """Evaluation-focused report: verdict upfront, per-case breakdown, failures.

    Designed for the ``evaluate()`` flow where the user's question is simply
    'is this skill worth using?' — not 'how much did authoring cost?'.
    """
    console = console or Console()
    ev: Evaluation | None = skill.evaluation
    assert ev is not None, "No evaluation attached to skill"

    n_cases = len(ev.with_skill)
    skill_pass = sum(1 for r in ev.with_skill if r.passed)
    base_pass = sum(1 for r in ev.baseline if r.passed)
    lift = ev.lift

    # ---- Hero verdict --------------------------------------------------------
    console.print()
    console.rule(f"[bold]Evaluation: {skill.name}[/bold]")
    console.print(f"[dim]{skill.description}[/dim]\n")

    if lift > 0.15:
        verdict_str = "[bold green]✅ EFFECTIVE[/bold green] — this skill adds clear value"
    elif lift > 0:
        verdict_str = "[bold yellow]⚠️  MARGINAL[/bold yellow] — slight improvement, may not justify cost"
    elif lift == 0:
        verdict_str = "[bold dim]➖ NO DIFFERENCE[/bold dim] — skill didn't help"
    else:
        verdict_str = "[bold red]❌ HARMFUL[/bold red] — skill made things worse"

    console.print(f"  Verdict:   {verdict_str}")
    console.print(
        f"  Accuracy:  [bold]{skill_pass}/{n_cases}[/bold] with skill vs "
        f"[dim]{base_pass}/{n_cases}[/dim] baseline  "
        f"([bold green]{lift:+.0%} lift[/bold green])" if lift > 0 else
        f"  Accuracy:  [bold]{skill_pass}/{n_cases}[/bold] with skill vs "
        f"[dim]{base_pass}/{n_cases}[/dim] baseline  "
        f"({lift:+.0%} lift)"
    )
    console.print(f"  Cost:      {ev.cost_ratio:.1f}× tokens vs baseline")
    console.print()

    # ---- Per-case table ------------------------------------------------------
    table = Table(title="Per-Case Breakdown", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Input (truncated)", max_width=50)
    table.add_column("Baseline", justify="center", width=8)
    table.add_column("With Skill", justify="center", width=10)
    table.add_column("Δ", justify="center", width=5)

    for i, (ws, bl) in enumerate(zip(ev.with_skill, ev.baseline), 1):
        # Show the last meaningful line of input (skip schema preambles).
        inp_lines = ws.case.input.strip().splitlines()
        inp_display = inp_lines[-1] if inp_lines else ws.case.input
        inp_short = inp_display[:47] + "..." if len(inp_display) > 50 else inp_display
        bl_icon = "[green]✓[/green]" if bl.passed else "[red]✗[/red]"
        ws_icon = "[green]✓[/green]" if ws.passed else "[red]✗[/red]"
        if ws.passed and not bl.passed:
            delta = "[green]+1[/green]"
        elif not ws.passed and bl.passed:
            delta = "[red]-1[/red]"
        else:
            delta = "[dim]=[/dim]"
        table.add_row(str(i), inp_short, bl_icon, ws_icon, delta)

    console.print(table)

    # ---- Failed cases detail -------------------------------------------------
    failures = [r for r in ev.with_skill if not r.passed]
    if failures:
        console.print(f"\n[bold red]Failed cases ({len(failures)}):[/bold red]")
        for r in failures:
            console.print(f"\n  [bold]Input:[/bold] {r.case.input[:100]}")
            console.print(f"  [bold]Expected:[/bold] {r.case.reference[:100]}")
            console.print(f"  [bold]Got:[/bold] {r.output[:150]}")
            console.print(f"  [dim]Criterion: {r.case.criterion}[/dim]")
    else:
        console.print("\n[bold green]All cases passed![/bold green]")

    # ---- Token summary (compact) ---------------------------------------------
    if skill.usage is not None:
        total = skill.usage.total
        console.print(f"\n[dim]Total tokens: {_fmt(total)} "
                      f"(eval: {_fmt(skill.usage.eval_arms.total)} + "
                      f"judge: {_fmt(skill.usage.judging.total)} + "
                      f"case_gen: {_fmt(skill.usage.case_gen.total)})[/dim]")
    console.print()
