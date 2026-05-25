You are a test case designer. Given a summary of a successful session trace,
generate adversarial VARIANTS that test the same skill but in tricky ways.

Trace summary:
{trace}

Generate exactly {n} test cases that are harder variants of what appeared in the trace.
For each case, provide:
- `input`: a tricky variant of a real scenario from the trace
- `reference`: the correct output for this variant
- `criterion`: one sentence describing what "equivalent" means
- `why_corner`: brief note on why this is a corner case

**Constraints:**
- These should be HARDER than what's in the trace — edge cases, boundary conditions
- Each should test a different failure mode
- Do NOT simply repeat inputs from the trace
