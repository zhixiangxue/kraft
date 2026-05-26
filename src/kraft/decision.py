"""Decision logic: 5-state machine determining whether a skill is effective."""

from __future__ import annotations

from dataclasses import dataclass

from kraft.skill import Evaluation


@dataclass
class Thresholds:
    """Tunable thresholds for the decision function."""

    alpha_baseline_ok: float = 0.95
    beta_lift_min: float = 0.10
    gamma_skill_min: float = 0.80
    # cost_ratio is a health warning, not a KEEP gate. Only intervenes when
    # the ratio is wildly out of range (e.g. runaway tool loops, oversized
    # references). pass_rate is the bottom line; users decide whether the
    # token spend is worth it.
    delta_cost_ratio_sane: float = 10.0
    epsilon_plateau: float = 0.02
    max_iter: int = 3


@dataclass
class Verdict:
    """Decision output from a single evaluation round."""

    state: str  # "KEEP" | "NO_SKILL_NEEDED" | "PLATEAU" | "MAX_ITER" | "REFINE"
    reason: str  # human-readable explanation
    kept_iter: int  # which iteration's skill to use; -1 when REFINE


def decide(history: list[Evaluation], th: Thresholds = Thresholds()) -> Verdict:
    """Determine whether to keep, refine, or discard the skill.

    Args:
        history: Chronological list of Evaluations. Latest is history[-1].
                 First call has len(history) == 1.
        th: Tunable thresholds.

    Returns:
        Verdict with one of 5 terminal/continuation states.
    """
    current = history[-1]
    base = current.pass_rate_baseline
    skill = current.pass_rate_with_skill
    lift = skill - base
    ratio = current.cost_ratio

    # 1. Baseline already good enough — no skill needed
    if base >= th.alpha_baseline_ok:
        return Verdict(
            state="NO_SKILL_NEEDED",
            reason=f"baseline={base:.0%} ≥ α={th.alpha_baseline_ok:.0%}",
            kept_iter=-1,
        )

    # 2. pass_rate gate (lift + absolute pass rate)
    pass_rate_ok = lift >= th.beta_lift_min and skill >= th.gamma_skill_min
    cost_sane = ratio <= th.delta_cost_ratio_sane

    # 3. KEEP: pass_rate gate satisfied AND cost not wildly out of range
    if pass_rate_ok and cost_sane:
        return Verdict(
            state="KEEP",
            reason=(
                f"lift={lift:+.0%} ≥ β={th.beta_lift_min:.0%}, "
                f"skill={skill:.0%} ≥ γ={th.gamma_skill_min:.0%}, "
                f"cost_ratio={ratio:.2f}× ≤ δ_sane={th.delta_cost_ratio_sane}×"
            ),
            kept_iter=len(history) - 1,
        )

    # 4. Hit max_iter — decide based on the BEST historical iteration.
    #    If best iter's pass_rate is satisfied (cost is health-only), KEEP.
    #    Otherwise MAX_ITER (→ discard at recommendation layer).
    if len(history) >= th.max_iter:
        best = _best_iter(history)
        best_eval = history[best]
        best_lift = best_eval.lift
        best_skill = best_eval.pass_rate_with_skill
        best_ratio = best_eval.cost_ratio
        best_pass_ok = (
            best_lift >= th.beta_lift_min
            and best_skill >= th.gamma_skill_min
        )
        if best_pass_ok:
            cost_note = (
                f"cost_ratio={best_ratio:.2f}× above sane ({th.delta_cost_ratio_sane}×), accepted"
                if best_ratio > th.delta_cost_ratio_sane
                else f"cost_ratio={best_ratio:.2f}× ok"
            )
            return Verdict(
                state="KEEP",
                reason=(
                    f"max_iter reached; best iter {best}: "
                    f"lift={best_lift:+.0%} ≥ β, skill={best_skill:.0%} ≥ γ, {cost_note}"
                ),
                kept_iter=best,
            )
        return Verdict(
            state="MAX_ITER",
            reason=(
                f"max_iter={th.max_iter} reached; best iter {best} still inadequate "
                f"(lift={best_lift:+.0%}, skill={best_skill:.0%})"
            ),
            kept_iter=best,
        )

    # 5. Plateau: need at least 2 rounds of history.
    #    Same logic as max_iter: judge by best iter.
    if len(history) >= 2:
        prev_lift = history[-2].lift
        if abs(lift - prev_lift) < th.epsilon_plateau:
            best = _best_iter(history)
            best_eval = history[best]
            best_lift = best_eval.lift
            best_skill = best_eval.pass_rate_with_skill
            best_pass_ok = (
                best_lift >= th.beta_lift_min
                and best_skill >= th.gamma_skill_min
            )
            if best_pass_ok:
                return Verdict(
                    state="KEEP",
                    reason=(
                        f"plateau detected (Δlift={lift - prev_lift:+.0%} < ε); "
                        f"best iter {best}: lift={best_lift:+.0%}, skill={best_skill:.0%}"
                    ),
                    kept_iter=best,
                )
            return Verdict(
                state="PLATEAU",
                reason=(
                    f"lift converged: {prev_lift:+.0%} → {lift:+.0%} (Δ < ε={th.epsilon_plateau:.0%}); "
                    f"best iter {best} inadequate (lift={best_lift:+.0%}, skill={best_skill:.0%})"
                ),
                kept_iter=best,
            )

    # 6. Otherwise continue refining.
    #    Triggered by: pass_rate insufficient / cost_ratio insane / both.
    return Verdict(
        state="REFINE",
        reason=(
            f"not yet good enough (lift={lift:+.0%}, "
            f"skill={skill:.0%}, ratio={ratio:.2f}×), will refine"
        ),
        kept_iter=-1,
    )


def _best_iter(history: list[Evaluation]) -> int:
    """Pick iteration with highest with_skill pass_rate; ties go to earlier (cheaper)."""
    return max(
        range(len(history)),
        key=lambda i: (history[i].pass_rate_with_skill, -i),
    )
