"""Evaluate a CSV-to-JSON skill that ships with executable scripts.

Demonstrates ``scripts_enabled=True`` in the evaluate flow: both arms get
Python + Bash tools. The skill's edge comes from its bundled script
(``scripts/csv_to_json.py``) and precise type-inference rules — the baseline
model can also write Python, but tends to get edge cases wrong (leading zeros,
null detection, RFC 4180 quoting).

Key design:
  - scripts_enabled=True → both arms can execute Python
  - case.input contains inline CSV (self-contained, no file paths)
  - reference is the exact JSON output the script produces
  - Lift comes from the skill's deterministic rules, not tool access asymmetry
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import EvalCase, evaluate

from _report import print_eval_report

SKILL_DIR = Path(__file__).parent / "skills" / "csv-converter"


async def main():
    api_key = os.environ["OPENAI_API_KEY"]

    skill = await evaluate(
        skill_dir=SKILL_DIR,
        model="openai/gpt-4o",
        api_key=api_key,
        scripts_enabled=True,
        cases=[
            # Leading zeros → must stay as string
            EvalCase(
                input=(
                    "Convert this CSV to typed JSON:\n"
                    "id,zip,score\n"
                    "1,007,95\n"
                    "2,10001,100"
                ),
                reference=(
                    '[{"id": 1, "zip": "007", "score": 95}, '
                    '{"id": 2, "zip": "10001", "score": 100}]'
                ),
                criterion=(
                    "zip '007' must be string (leading zero). "
                    "zip '10001' and scores must be integers. "
                    "id must be integer."
                ),
                source="user",
            ),
            # Null/empty detection
            EvalCase(
                input=(
                    "Convert this CSV to typed JSON:\n"
                    "name,age,email\n"
                    "Alice,30,alice@test.com\n"
                    "Bob,,null\n"
                    "Charlie,NULL,charlie@test.com"
                ),
                reference=(
                    '[{"name": "Alice", "age": 30, "email": "alice@test.com"}, '
                    '{"name": "Bob", "age": null, "email": null}, '
                    '{"name": "Charlie", "age": null, "email": "charlie@test.com"}]'
                ),
                criterion=(
                    "Empty string and literal 'null'/'NULL' must become JSON null. "
                    "Integers must be numbers. Strings stay as strings."
                ),
                source="user",
            ),
            # Boolean + float
            EvalCase(
                input=(
                    "Convert this CSV to typed JSON:\n"
                    "feature,enabled,weight\n"
                    "dark_mode,True,0.85\n"
                    "beta_flag,FALSE,1.0\n"
                    "legacy,true,0"
                ),
                reference=(
                    '[{"feature": "dark_mode", "enabled": true, "weight": 0.85}, '
                    '{"feature": "beta_flag", "enabled": false, "weight": 1.0}, '
                    '{"feature": "legacy", "enabled": true, "weight": 0}]'
                ),
                criterion=(
                    "Boolean: True/TRUE/true → true, FALSE/false → false. "
                    "Float: 0.85, 1.0. Integer: 0 (single zero is valid int). "
                    "feature names stay as strings."
                ),
                source="user",
            ),
            # RFC 4180 quoting with embedded commas
            EvalCase(
                input=(
                    'Convert this CSV to typed JSON:\n'
                    'name,address,age\n'
                    '"Smith, Jr.",\"123 Main St\",42\n'
                    'Jones,456 Oak Ave,28'
                ),
                reference=(
                    '[{"name": "Smith, Jr.", "address": "123 Main St", "age": 42}, '
                    '{"name": "Jones", "address": "456 Oak Ave", "age": 28}]'
                ),
                criterion=(
                    "Quoted field 'Smith, Jr.' must preserve the comma as part of the value. "
                    "Ages must be integers. Must follow RFC 4180 quoting rules."
                ),
                source="user",
            ),
            # Mixed edge: negative int, whitespace trimming
            EvalCase(
                input=(
                    "Convert this CSV to typed JSON:\n"
                    "item, qty , price\n"
                    "Widget, -5 , 3.99\n"
                    "Gadget, 10 ,  None "
                ),
                reference=(
                    '[{"item": "Widget", "qty": -5, "price": 3.99}, '
                    '{"item": "Gadget", "qty": 10, "price": null}]'
                ),
                criterion=(
                    "Headers and values must be trimmed (no leading/trailing spaces). "
                    "Negative integer -5 must be a number. 'None' must become null. "
                    "3.99 must be float."
                ),
                source="user",
            ),
        ],
    )

    print_eval_report(skill)


if __name__ == "__main__":
    asyncio.run(main())
