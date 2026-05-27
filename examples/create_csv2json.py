"""Offline example: build a CSV-to-typed-JSON converter skill.

Demonstrates ``scripts_enabled=True`` with a completely offline task.
The user passes a ONE-LINE task; kraft's elaboration phase expands it
into a full structured spec internally. No network, no API keys for
the *task itself* (only the LLM provider key is needed).

The lift signal here comes from **type-inference determinism**: without
the skill, a baseline LLM picks ad-hoc rules per call and produces
inconsistent results. The skill pins rules in SKILL.md so every
invocation yields identical outputs for identical inputs.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so sibling examples can `import _report`

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import kraft

from _report import print_run_report


TASK = (
    "Build a CSV-to-JSON converter CLI tool (stdlib only) with "
    "deterministic type inference: empty/null, bool, int (no leading "
    "zeros), float, ISO-8601 dates kept as strings, everything else "
    "verbatim. Honor RFC 4180 quoting. Ship as scripts/csv_to_json.py."
)


async def main():
    skill_dir = Path(__file__).parent / "output" / "csv_to_json"

    print("Starting kraft run...")
    print(f"Task: {TASK}")
    print(f"Skill dir: {skill_dir}")
    print("-" * 60)

    skill = await kraft(
        task=TASK,
        skill_dir=skill_dir,
        model="openai/gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
        scripts_enabled=True,
        n_cases=10,
        max_iter=4,
        overwrite=True,
    )

    print_run_report(skill)


if __name__ == "__main__":
    asyncio.run(main())
