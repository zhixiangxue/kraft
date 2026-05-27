---
name: csv-typed-json
description: Convert CSV text to JSON with deterministic type inference (stdlib only).
---

# CSV → Typed JSON Converter

Convert raw CSV text into a JSON array of objects with **deterministic type
inference**. Use the bundled script at `scripts/csv_to_json.py` for
guaranteed-consistent results.

## Usage

```bash
echo "name,age,active\nAlice,30,true" | python scripts/csv_to_json.py
```

Or from Python:

```python
from scripts.csv_to_json import convert
result = convert("name,age,active\nAlice,30,true")
```

## Type Inference Rules (in priority order)

1. **Empty / null**: cell is empty or literally `null`/`NULL`/`None` → JSON `null`
2. **Boolean**: case-insensitive `true`/`false` → JSON `true`/`false`
3. **Integer**: matches `^-?[1-9]\d*$` or `^0$` (NO leading zeros) → JSON number
   - `007` → string `"007"` (leading zero = not an integer)
   - `-42` → number `-42`
4. **Float**: matches `^-?\d+\.\d+$` → JSON number
   - `3.14` → number `3.14`
5. **Everything else**: kept as JSON string verbatim

## Edge Cases

- RFC 4180 quoting: fields wrapped in double-quotes; embedded commas and
  newlines inside quotes are preserved.
- Whitespace: leading/trailing spaces in unquoted fields are **trimmed**
  before type inference.
- Header row: the first line is ALWAYS treated as column headers.
