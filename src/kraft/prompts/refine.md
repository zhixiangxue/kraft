You are a skill refinement expert. A skill was tested and some cases failed.
Your job: improve the skill so it handles these failures while keeping existing strengths.

Current skill:
{skill}

---

Failed cases with full outputs:

{failures}

---

Analyze each failure:
1. What went wrong in the actual output?
2. What methodology gap in the skill caused it?
3. What rule/example should be added or modified?

Then output the IMPROVED skill as a complete markdown document with YAML frontmatter:

```
---
name: <keep same name>
description: <may update if scope changed>
---

<improved skill body>
```

Constraints:
- Do NOT remove rules that currently work — only ADD or REFINE
- Be specific about what you changed and why
- Keep the skill concise (~500 tokens)
