"""Debug: run kraft-produced openapi_client skill end-to-end with verbose logs.

Goal: see *exactly* what gpt-4o-mini does when handed the produced skill —
which tools it picks, in what order, what each tool returns, and where the
tokens go. Output is event-by-event so we can decide whether the 1.5M-token
explosion in kraft evaluation is (a) the model genuinely lacking agentic
ability, (b) a SKILL.md authoring bug, or (c) something else entirely.

Usage:
    python examples/debug_openapi_skill.py
    python examples/debug_openapi_skill.py openai/gpt-4o-mini
    python examples/debug_openapi_skill.py anthropic/claude-sonnet-4-6
"""

import argparse
import asyncio
import os
from pathlib import Path

import dotenv

import chak
from chak.tools.std import Bash, Python
from chak.tools.skills import ClaudeSkill

dotenv.load_dotenv()

_ROOT = Path(__file__).parent.parent
SKILL_DIR = _ROOT / "examples" / "output" / "openapi_client"

# A single consumer-style request — same shape as kraft cases.
USER_REQUEST = (
    "Get the pet with id=1 from Petstore. Return its name and status."
)


async def main(model_uri: str):
    provider = model_uri.split("/")[0].upper()
    api_key = os.getenv(f"{provider}_API_KEY", "")
    if not api_key:
        raise ValueError(f"Please set {provider}_API_KEY in .env")

    if not SKILL_DIR.exists():
        raise FileNotFoundError(
            f"Skill not found at {SKILL_DIR}. Run examples/openapi_client.py first."
        )

    skill = ClaudeSkill(str(SKILL_DIR))
    bash = Bash()
    python = Python()

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        tools=[skill, bash, python],
    )

    print(f"Model:  {model_uri}")
    print(f"Skill:  {SKILL_DIR}")
    print(f"User:   {USER_REQUEST}")
    print("-" * 70)

    tool_call_count = 0
    tool_call_histogram: dict[str, int] = {}

    async for event in await conv.asend(USER_REQUEST, event=True, timeout=180):
        if isinstance(event, chak.MessageChunk):
            if event.content:
                print(event.content, end="", flush=True)
            if event.is_final:
                print()
        elif isinstance(event, chak.ToolCallStartEvent):
            tool_call_count += 1
            tool_call_histogram[event.tool_name] = (
                tool_call_histogram.get(event.tool_name, 0) + 1
            )
            print(f"\n[#{tool_call_count} call] {event.tool_name}")
            if event.arguments:
                args_str = str(event.arguments)
                if len(args_str) > 400:
                    args_str = args_str[:400] + "...(truncated)"
                print(f"  args: {args_str}")
        elif isinstance(event, chak.ToolCallSuccessEvent):
            result = event.result if isinstance(event.result, str) else str(event.result)
            preview = result[:400] + "...(truncated)" if len(result) > 400 else result
            print(f"  -> {preview}")
        elif isinstance(event, chak.ToolCallErrorEvent):
            print(f"  -> [error] {event.error}")

    print("-" * 70)
    stats = conv.stats()
    print(f"Total tool calls : {tool_call_count}")
    print(f"Tool histogram   : {tool_call_histogram}")
    print(f"Total tokens     : {stats.get('total_tokens', 'n/a')}")
    print(f"Full stats       : {stats}")


def main_entry():
    parser = argparse.ArgumentParser(description="Debug openapi_client skill")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="openai/gpt-4o-mini",
        help="Model URI (default: openai/gpt-4o-mini)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model_uri))


if __name__ == "__main__":
    main_entry()
