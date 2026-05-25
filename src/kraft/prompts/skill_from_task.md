You are a skill author. Given a task description, write a reusable skill document
that helps an LLM perform this task more reliably.

Task:
{task}

Output a single markdown document with YAML frontmatter. Format:

```
---
name: <kebab-case-name>
description: <one line, max 100 chars>
---

<skill body: methodology, steps, edge cases, examples>
```

Requirements:
- Focus on methodology and decision rules, not surface-level instructions
- Include concrete examples of tricky cases and how to handle them
- Anticipate common failure modes and provide explicit countermeasures
- Keep it concise — the skill should fit in ~500 tokens
