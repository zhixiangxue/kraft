"""CSV to typed-JSON converter — stdlib only.

Type inference follows a strict priority:
  empty/null → null, bool → bool, int (no leading zeros) → number,
  float → number, else → string.

RFC 4180 quoting is honored via the csv module.
"""

import csv
import io
import json
import re
import sys

_INT_RE = re.compile(r"^-?[1-9]\d*$|^0$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_NULL_VALS = {"", "null", "NULL", "None"}


def _infer(value: str):
    """Infer typed value from a raw CSV cell string."""
    value = value.strip()
    if value in _NULL_VALS:
        return None
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    return value


def convert(csv_text: str) -> list[dict]:
    """Convert CSV text to a list of typed dicts."""
    reader = csv.reader(io.StringIO(csv_text))
    headers = [h.strip() for h in next(reader)]
    rows = []
    for row in reader:
        obj = {}
        for i, cell in enumerate(row):
            key = headers[i] if i < len(headers) else f"col_{i}"
            obj[key] = _infer(cell)
        rows.append(obj)
    return rows


if __name__ == "__main__":
    text = sys.stdin.read()
    print(json.dumps(convert(text), indent=2, ensure_ascii=False))
