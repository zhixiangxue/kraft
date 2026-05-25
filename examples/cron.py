"""Smoke test: generate and verify a skill for cron expression conversion."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import kraft


async def main():
    print("Starting kraft run...")
    print("Task: convert natural language to cron expressions")
    print("-" * 60)

    skill = await kraft(
        task="convert natural language to cron expressions",
        model="openai/gpt-4o-mini",
    )

    print(f"\n{'=' * 60}")
    print(f"Skill: {skill.name}")
    print(f"Description: {skill.description}")
    print(f"{'=' * 60}")
    print(f"\n{skill.report.summary()}")
    print(f"\nBaseline pass rate: {skill.report.pass_rate_baseline:.0%}")
    print(f"With-skill pass rate: {skill.report.pass_rate_with_skill:.0%}")
    print(f"Lift: {skill.report.lift:+.0%}")
    print(f"Cost ratio: {skill.report.cost_ratio:.2f}×")
    print(f"Recommendation: {skill.report.recommendation}")
    print(f"\n{'=' * 60}")
    print("Skill body:")
    print(skill.body[:500])

    # Save to examples/output/
    out_dir = Path(__file__).parent / "output" / "cron"
    skill.save(out_dir)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
