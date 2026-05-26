"""Smoke test: generate and verify a skill for cron expression conversion."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so sibling examples can `import _report`

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import kraft

from _report import print_run_report


async def main():
    # The developer picks where the skill lives. Final SKILL.md (+ scripts/
    # if scripts_enabled=True) lands directly here. Iteration history is
    # written automatically to a sibling `<skill_dir>.kraft/` directory.
    skill_dir = Path(__file__).parent / "output" / "cron"

    print("Starting kraft run...")
    print("Task: convert natural language to cron expressions")
    print(f"Skill dir: {skill_dir}")
    print("-" * 60)

    skill = await kraft(
        task="convert natural language to cron expressions",
        skill_dir=skill_dir,
        model="openai/gpt-4o-mini",
        overwrite=True,  # smoke test — always start fresh
    )

    print_run_report(skill)


if __name__ == "__main__":
    asyncio.run(main())
