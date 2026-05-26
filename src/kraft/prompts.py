"""Prompt builders for kraft.

Two orthogonal dimensions for the author/refiner agent:
- **source**: ``task`` / ``trace`` / ``task+trace`` / ``refine``
- **form**:   ``doc`` (SKILL.md only) vs ``scripts`` (SKILL.md + scripts/refs/examples)

Each combination is exposed as a typed builder function returning a finished
prompt string. Common fragments (Working environment + Required output +
Authoring guidelines + Done criteria) are factored into private module-level
constants and selected via ``scripts_enabled``.

Two additional builders — :func:`tests_corner_case` and :func:`tests_from_trace`
— cover the test-case-generation path used by ``cases.py``.
"""

from __future__ import annotations

# =====================================================================
# Form fragments — selected by ``scripts_enabled``
#
# Each template uses ``{skill_dir}`` as its sole ``str.format`` placeholder;
# nothing else inside these strings looks like ``{...}``, so the format call
# is unambiguous.
# =====================================================================

_FORM_DOC = """\
You have one tool: **filesystem** — write SKILL.md into the directory.

This skill is **methodology-only** — it must teach the LLM through prose
rules, decision trees, and worked examples. No supporting files are
available at runtime.

# Required output

Create exactly one file at `{skill_dir}/SKILL.md`:

```
---
name: <kebab-case-name>
description: <one line, ≤ 200 chars — when to invoke this skill>
---

<methodology body — decision rules, edge cases, worked examples>
```

**Do NOT create any other files** — no `scripts/`, `references/`, or
`examples/` subdirectories.

# Authoring guidelines

- Anticipate failure modes the model would hit naively. Counter each one
  with an explicit rule or a worked example in the prose.
- Inline any small lookup data (mappings, vocabulary, format specs) directly
  into SKILL.md.
- For tricky cases, write 2-3 concrete input/output examples.
- Keep it tight — methodology, not exhaustive reference.
"""

_FORM_SCRIPTS = """\
You have these tools at your disposal:

- **filesystem** — read/write/edit files inside the working directory
- **python** — run Python snippets in the active venv
- **bash** — run shell commands

Use them freely to author, prototype, and self-check. Install packages via
pip if you need them.

# What "a skill" looks like

A skill is a *directory*. The runtime loads it as an Anthropic Agent Skill
with three-layer progressive disclosure:

- **Layer 1 (always visible)** — `name` + `description` from SKILL.md
  frontmatter. The runtime decides whether to invoke the skill based purely
  on this.
- **Layer 2 (on activation)** — the SKILL.md *body* (the methodology).
- **Layer 3 (on demand)** — supporting files (`scripts/`, `references/`,
  `examples/`). Read only when SKILL.md tells the runtime to.

# Required output

Create `{skill_dir}/SKILL.md` with this frontmatter shape:

```
---
name: <kebab-case-name>
description: <one line, ≤ 200 chars — when to invoke this skill>
---

<methodology body>
```

# Optional supporting files

Decide **consciously** whether the task benefits from extras:

- `scripts/*.py` — when the task has a deterministic checker or normalizer
  (e.g. cron syntax check, JSON-schema validation, unit conversion).
  Reference them explicitly from SKILL.md.
- `references/*.md` — when the task needs domain knowledge too long to
  inline (lookup tables, spec excerpts, vocabulary).
- `examples/*.md` — when concrete worked examples teach better than rules.

If the task is pure prose methodology, **just SKILL.md is fine** — don't pad.

# Authoring guidelines

- Anticipate failure modes; counter each with a rule, an example, or a script.
- For deterministic transformations, a small Python validator beats prose
  rules. Write it. Test it with python before declaring done.
- Keep SKILL.md focused on *methodology*. Push raw data into references/.
- Cross-reference supporting files from SKILL.md by relative path.
"""

_DONE_DOC = """\
# Done criteria

`{skill_dir}` contains exactly `SKILL.md` and nothing else. **Do not echo
the skill content back in your final message** — write it to disk and stop.
"""

_DONE_SCRIPTS = """\
# Done criteria

`{skill_dir}` contains a working SKILL.md plus any supporting files you
decided to include. **Do not echo the skill content back in your final
message** — write it to disk and stop.
"""


def _form_block(scripts_enabled: bool, skill_dir: str) -> str:
    template = _FORM_SCRIPTS if scripts_enabled else _FORM_DOC
    return template.format(skill_dir=skill_dir)


def _done_block(scripts_enabled: bool, skill_dir: str) -> str:
    template = _DONE_SCRIPTS if scripts_enabled else _DONE_DOC
    return template.format(skill_dir=skill_dir)


# =====================================================================
# Skill-authoring builders (4 source variants × 2 forms = 8 outputs)
# =====================================================================


def from_task(*, task: str, skill_dir: str, scripts_enabled: bool) -> str:
    """Author a skill from a task description."""
    form = _form_block(scripts_enabled, skill_dir)
    done = _done_block(scripts_enabled, skill_dir)
    return f"""\
You are a skill author. Your job: write a SKILL.md that captures the
methodology an LLM should follow to perform this task more reliably than it
would naively.

Task:

{task}

# Working environment

Working directory: `{skill_dir}` (already exists, possibly empty).

{form}
{done}"""


def from_trace(*, trace: str, skill_dir: str, scripts_enabled: bool) -> str:
    """Author a skill from a successful session trace."""
    form = _form_block(scripts_enabled, skill_dir)
    done = _done_block(scripts_enabled, skill_dir)
    return f"""\
You are a skill author. Your job: distill a successful session trace into
a SKILL.md that teaches an LLM to repeat the same kind of success on
similar tasks.

Trace summary:

{trace}

# Working environment

Working directory: `{skill_dir}` (already exists, possibly empty).

{form}
# Trace-specific guidance

- Extract the **why** behind the trace's actions, not just the **what**.
- Identify recurring patterns and decision points; encode them as rules.
- Note edge cases the trace handled well that a naive approach would miss.

{done}"""


def from_task_and_trace(
    *, task: str, trace: str, skill_dir: str, scripts_enabled: bool
) -> str:
    """Author a skill from both a task description and a successful trace."""
    form = _form_block(scripts_enabled, skill_dir)
    done = _done_block(scripts_enabled, skill_dir)
    return f"""\
You are a skill author. You have BOTH a task description (the goal) AND a
session trace (proven patterns). Combine them into a single SKILL.md.

Task:

{task}

Trace summary (showing known-good patterns for this task):

{trace}

# Working environment

Working directory: `{skill_dir}` (already exists, possibly empty).

{form}
# Combination guidance

- Use the **task description** to define scope and goals.
- Use the **trace** to ground the methodology in proven moves.
- Highlight edge cases from the trace that a naive reading of the task would
  miss.

{done}"""


# Refine has a different working-environment + constraints shape — the agent
# is editing in place rather than authoring from scratch — so it carries its
# own form-specific blocks instead of reusing _FORM_*.

_REFINE_ENV_DOC = """\
Working directory: `{skill_dir}` — the previous SKILL.md is already copied
here. Read it first to understand what's there, then edit in place.

You have one tool: **filesystem** — read/edit SKILL.md.

This skill is methodology-only. **Do not create `scripts/`, `references/`,
or `examples/`** — refinement happens entirely inside SKILL.md.
"""

_REFINE_ENV_SCRIPTS = """\
Working directory: `{skill_dir}` — the previous skill is already copied
here. Read it first, then modify in place. You may:

- Edit `SKILL.md` (body or frontmatter description if scope shifted)
- Add/edit/delete supporting files (`scripts/*.py`, `references/*.md`,
  `examples/*.md`)

Tools: **filesystem**, **python**, **bash**. Run scripts to verify changes.
"""

_REFINE_CONSTRAINTS_DOC = """\
- **Do NOT remove rules that currently work** — only ADD or REFINE.
- Keep `name` (kebab-case) the same; update `description` only if scope
  genuinely shifted.
- Output must remain a single SKILL.md file. No supporting files.
- Avoid bloat — prefer sharpening existing rules over piling on new ones.
"""

_REFINE_CONSTRAINTS_SCRIPTS = """\
- **Do NOT remove rules that currently work** — only ADD or REFINE.
- Keep `name` (kebab-case) the same; update `description` only if scope
  genuinely shifted.
- Prefer encoding deterministic checks as scripts rather than prose rules.
- If you add a script, reference it from SKILL.md.
"""

_REFINE_DONE_DOC = """\
`{skill_dir}/SKILL.md` is the improved skill, and there are no other files.
**Do not echo the content back** — write to disk and stop.
"""

_REFINE_DONE_SCRIPTS = """\
`{skill_dir}` contains the improved skill — SKILL.md plus any supporting
files. **Do not echo the content back** — write to disk and stop.
"""


def refine(*, failures: str, skill_dir: str, scripts_enabled: bool) -> str:
    """Refine an existing skill in place based on failed evaluation cases."""
    if scripts_enabled:
        env = _REFINE_ENV_SCRIPTS.format(skill_dir=skill_dir)
        constraints = _REFINE_CONSTRAINTS_SCRIPTS
        done = _REFINE_DONE_SCRIPTS.format(skill_dir=skill_dir)
    else:
        env = _REFINE_ENV_DOC.format(skill_dir=skill_dir)
        constraints = _REFINE_CONSTRAINTS_DOC
        done = _REFINE_DONE_DOC.format(skill_dir=skill_dir)
    return f"""\
You are a skill refiner. The previous skill was tested; some cases failed.
Your job: improve the skill so the failures are handled while keeping
existing strengths.

# Working environment

{env}
# Failed cases (full raw model output)

{failures}

# Refinement procedure

For each failure, think:

1. What went wrong in the actual output? (Wrong format? Hallucinated value?
   Missed edge case?)
2. What gap in the current skill caused it? (Missing rule? Vague guidance?
   No example for this case shape?)
3. What's the smallest change that fixes it without breaking others?

Apply the change.

# Constraints

{constraints}
# Done criteria

{done}"""


# =====================================================================
# Test-case generation builders (used by cases.py)
# =====================================================================


def tests_corner_case(*, task: str, n: int) -> str:
    """Prompt a designer to produce a balanced mix of happy-path + corner-case tests.

    Why a mix and not pure adversarial: a skill's primary job is to handle the
    *typical* request well; corner cases tell us about robustness, but if the
    happy path is broken the skill has zero value. An adversarial-only test set
    also pathologically biases certain task shapes — e.g. for "build a client
    library" tasks the only obvious corners are "caller passes invalid params,
    function should reject", which conflates client responsibility with server
    responsibility and produces a useless eval. Mixing in explicit happy-path
    cases keeps the eval grounded in the skill's actual purpose.
    """
    return f"""\
You are a test case designer. Generate evaluation cases for the task below.

Task under test:

{task}

The model under test is a *consumer* of the skill (not its author). At
evaluation time it has the skill's SKILL.md and ``scripts/`` available
and can execute Python tools to actually call into them. Your `input`
must be a real-world usage request — phrased as something an end-user
would ask a capable assistant — NOT a code-writing assignment.

  GOOD inputs (consumer-style usage requests):
    "Get the pet with id=1 from Petstore. Return its name and status."
    "Find all pets currently marked as 'available'."
    "Add a new pet named Rex with status 'available' and report the new id."

  BAD inputs (DO NOT generate these — they conflate the model with the
  skill author and force code-text answers, which can't be judged):
    "Create a function get_pet_by_id(...) that sends a GET request to
     /pet/{{id}}"
    "Implement a POST /pet endpoint wrapper using httpx"

Similarly, ``reference`` should describe the expected real result of
performing that request ("Returns a pet object containing id=1 and a
status field"), not the expected source code.

Generate exactly {n} evaluation cases. For each case, provide:

- `input`: the question/prompt to give the model under test
- `reference`: the correct/expected output
- `criterion`: one sentence describing what "equivalent" means for this case
- `why_corner`: brief note on why this case matters (happy-path realism
  OR a specific corner / failure mode it probes)

**Mix requirements:**

- **Roughly half must be HAPPY-PATH cases**: realistic, well-formed inputs
  representative of how the skill will actually be used day-to-day. These
  must NOT be malformed, adversarial, or tricky. They are the bar the
  skill must clear before robustness even becomes interesting.
- **The other roughly half must be CORNER / EDGE cases**: boundary
  conditions, ambiguous input, empty/null, overflow, format edge cases,
  or other inputs likely to trip a capable but imperfect model.
- Cover diverse failure modes — don't repeat the same edge across cases.
- Be careful not to invent corner cases that test responsibilities the
  skill doesn't own (e.g. don't test input validation against a skill
  whose job is to forward calls to a server that already validates).

Think carefully about both: what "normal usage" looks like, AND what
would trip up a capable but imperfect model on the edges.
"""


def tests_from_trace(*, trace: str, n: int) -> str:
    """Prompt a designer to produce tricky variants from a session trace."""
    return f"""\
You are a test case designer. Given a summary of a successful session trace,
generate adversarial VARIANTS that test the same skill but in tricky ways.

Trace summary:

{trace}

Generate exactly {n} evaluation cases that are harder variants of what appeared
in the trace. For each case, provide:

- `input`: a tricky variant of a real scenario from the trace
- `reference`: the correct output for this variant
- `criterion`: one sentence describing what "equivalent" means
- `why_corner`: brief note on why this is a corner case

**Constraints:**

- These should be HARDER than what's in the trace — edge cases, boundary
  conditions.
- Each should test a different failure mode.
- Do NOT simply repeat inputs from the trace.
"""
