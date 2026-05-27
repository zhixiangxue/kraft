---
name: conventional-commits
description: Generate Conventional Commits messages from natural-language change descriptions.
---

# Conventional Commits Generator

Given a description of code changes, produce a commit message that strictly
follows the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/)
specification.

## Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Rules

1. **type** must be one of:
   - `feat` — a new feature (correlates with MINOR in SemVer)
   - `fix` — a bug fix (correlates with PATCH)
   - `docs` — documentation only
   - `style` — formatting, missing semicolons, no code change
   - `refactor` — code change that neither fixes a bug nor adds a feature
   - `perf` — performance improvement
   - `test` — adding or correcting tests
   - `build` — changes to build system or external dependencies
   - `ci` — CI configuration files and scripts
   - `chore` — other changes that don't modify src or test files

2. **scope** is optional but MUST be a noun in parentheses describing the
   section of the codebase: `feat(parser):`, `fix(auth):`.

3. **description** MUST:
   - Be imperative, present tense ("add" not "added" or "adds")
   - Not capitalize the first letter
   - Not end with a period

4. **body** is optional. Use it when the "what" needs context on "why".
   Separate from description with a blank line.

5. **BREAKING CHANGE**: either:
   - Append `!` after type/scope: `feat(api)!: remove /v1 endpoints`
   - OR add a footer: `BREAKING CHANGE: /v1 endpoints removed`
   - A breaking change correlates with MAJOR in SemVer.

6. **Multiple changes**: if the diff touches unrelated concerns, pick the
   most significant change for the commit message. Do NOT combine types.

## Examples

| Change description | Commit message |
|---|---|
| Added dark mode toggle to settings page | `feat(settings): add dark mode toggle` |
| Fixed crash when user email is null | `fix(auth): handle null email gracefully` |
| Upgraded webpack from v4 to v5 | `build: upgrade webpack to v5` |
| Removed deprecated /api/v1 routes | `feat(api)!: remove deprecated /v1 routes` |
| Reformatted all files with prettier | `style: reformat codebase with prettier` |
