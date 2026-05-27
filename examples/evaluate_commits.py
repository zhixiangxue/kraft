"""Evaluate a Conventional Commits skill with natural-language cases.

Demonstrates the simplest ``evaluate()`` flow: pass a list of plain strings
describing test scenarios. kraft converts them to structured EvalCase via LLM,
discards any where expected output can't be inferred, then runs the comparison.

No schema, no domain knowledge — just a widely-used dev convention that
models often get subtly wrong (capitalization, tense, type choice, breaking
change syntax).
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import evaluate

from _report import print_eval_report

SKILL_DIR = Path(__file__).parent / "skills" / "conventional-commits"


async def main():
    api_key = os.environ["OPENAI_API_KEY"]

    skill = await evaluate(
        skill_dir=SKILL_DIR,
        model="openai/gpt-4o",
        api_key=api_key,
        cases=[
            # Type selection edge cases
            "I replaced the MySQL driver with PostgreSQL but didn't change any "
            "business logic. Generate a commit message. "
            "Expected: 'refactor(db): replace MySQL driver with PostgreSQL' — "
            "type must be 'refactor' (not 'feat'), imperative tense, no period.",

            # Breaking change syntax
            "I removed the /users/v1 endpoint entirely, existing clients must "
            "migrate to /users/v2. Generate a commit message. "
            "Expected: must include BREAKING CHANGE indicator — either '!' after "
            "type/scope or a 'BREAKING CHANGE:' footer. Type should be 'feat' or 'refactor'.",

            # Description formatting rules
            "I added unit tests for the payment service retry logic. "
            "Generate a commit message. "
            "Expected: 'test(payment): add retry logic unit tests' — "
            "type must be 'test', description starts lowercase, no trailing period, imperative mood.",

            # Scope inference
            "Fixed a race condition in the WebSocket connection handler that "
            "caused duplicate messages. Generate a commit message. "
            "Expected: 'fix(websocket): resolve race condition causing duplicate messages' — "
            "type 'fix', scope inferred from context, imperative tense.",

            # chore vs build distinction
            "Updated the .gitignore to exclude the new build artifacts folder. "
            "Generate a commit message. "
            "Expected: 'chore: update .gitignore for build artifacts' — "
            "type 'chore' (not 'build', since .gitignore isn't a build tool config).",
        ],
    )

    print_eval_report(skill)


if __name__ == "__main__":
    asyncio.run(main())
