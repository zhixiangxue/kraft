"""Smoke test: build a Swagger Petstore client skill.

The target skill is *domain-specific*: given the Petstore v2 Swagger spec
(https://petstore.swagger.io/v2/swagger.json) as a kraft-time input,
the author agent should produce a skill whose ``scripts/`` directory ships
a ready-to-use Python httpx client for Petstore, and whose SKILL.md tells
a downstream consumer "to call addPet, import and call X(...)".

We use **v2** rather than v3 because the v3 demo backend is half-broken
(only ``GET /pet/{id}`` works; POST/PUT and findByStatus all return 500),
which would make any real-call evaluation meaningless. The v2 demo at
petstore.swagger.io is fully functional and key-free across all 20 ops,
so eval-arm runs can actually hit the API and return real responses
— the only honest way to demonstrate that the generated skill *works*.

The spec is Swagger 2.0 (not OpenAPI 3.0); the field-name differences
(host+basePath+schemes vs servers, in:body params vs requestBody) are
well within what gpt-4o-mini handles.

This exercises ``scripts_enabled=True``: the author needs filesystem +
python + bash to fetch the spec, emit a client module, and run a smoke
import / ast.parse check before declaring done. The spec URL is the
kraft-time input — it is *not* something the skill consumer has to
provide later. Consumers only see Python functions like ``add_pet(...)``.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so sibling examples can `import _report`

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from kraft import kraft

from _report import print_run_report


TASK = """\
Build a Python client skill for the Swagger Petstore v2 API.

Spec source (fetch this at authoring time):
  https://petstore.swagger.io/v2/swagger.json
Server base URL:
  https://petstore.swagger.io/v2

Note: the spec is Swagger 2.0 (uses host + basePath + schemes; body
parameters use ``in: body`` rather than OpenAPI 3.0's ``requestBody``).
The v2 demo backend is fully functional and accepts all calls without
an API key, including operations the spec marks as requiring api_key
(``deletePet`` accepts any string). Treat api_key as an optional kwarg.

The skill you produce must let a downstream consumer call Petstore
endpoints from Python without ever touching the spec themselves.
Concretely:

  - ``scripts/`` ships a Python module exposing one function per
    operation, using ``httpx`` and synchronous calls. Each function
    takes typed kwargs for path / query / header / body params and
    returns the parsed JSON.
  - SKILL.md documents which functions exist, their signatures, and a
    minimal usage example per function. The consumer reads SKILL.md and
    imports from ``scripts/`` — they should never need to read the spec.

Must cover at least these operations:
  - addPet              POST   /pet              (body: Pet)
  - getPetById          GET    /pet/{petId}      (path param)
  - findPetsByStatus    GET    /pet/findByStatus (query param)
  - updatePet           PUT    /pet              (body: Pet)
  - deletePet           DELETE /pet/{petId}      (path + optional api_key)
  - placeOrder          POST   /store/order      (body: Order)

The authoring agent has python + bash + filesystem tools — use them to
fetch the spec, write the client, and at minimum ``ast.parse`` / import
the generated module before finishing. A live curl against
``https://petstore.swagger.io/v2/pet/1`` is a fine extra smoke check.
"""


async def main():
    skill_dir = Path(__file__).parent / "output" / "openapi_client"

    print("Starting kraft run...")
    print("Task: Petstore v2 Swagger spec → Python httpx client skill")
    print(f"Skill dir: {skill_dir}")
    print("scripts_enabled: True (author may write helper scripts)")
    print("-" * 60)

    skill = await kraft(
        task=TASK,
        skill_dir=skill_dir,
        model="anthropic/claude-sonnet-4-6",
        scripts_enabled=True,  # the whole point of this example
        overwrite=True,
    )

    print_run_report(skill)


if __name__ == "__main__":
    asyncio.run(main())
