You are a skill author. Given a task description AND a session trace showing
successful completions of that task, write a reusable skill document.

Task:
{task}

Trace summary (showing known successful patterns):
{trace}

Output a single markdown document with YAML frontmatter. Format:

```
---
name: <kebab-case-name>
description: <one line, max 100 chars>
---

<skill body: methodology combining task goals with proven patterns from trace>
```

Requirements:
- Use the task description to define scope and goals
- Use the trace to ground the methodology in proven patterns
- Highlight edge cases from the trace that a naive approach would miss
- Keep it concise — the skill should fit in ~500 tokens
