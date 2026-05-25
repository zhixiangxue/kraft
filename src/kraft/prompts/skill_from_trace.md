You are a skill author. Given a session trace showing successful task completions,
distill the underlying methodology into a reusable skill document.

Trace summary:
{trace}

Output a single markdown document with YAML frontmatter. Format:

```
---
name: <kebab-case-name>
description: <one line, max 100 chars>
---

<skill body: patterns extracted from trace, decision rules, edge cases>
```

Requirements:
- Extract the *why* behind the actions, not just the *what*
- Identify recurring patterns and decision points
- Note edge cases that were handled well in the trace
- Keep it concise — the skill should fit in ~500 tokens
