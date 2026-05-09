# -*- coding: utf-8 -*-
"""Regex-based structured extraction with ``PatternSchema``.

Sometimes the cheapest, most robust "structured output" is to let the
model speak naturally and then pull the fields you want out with a
single regex. This is exactly what ClassicReAct / ReWOO do internally.

Two demos:

* a hand-rolled schema applied to a freeform assistant reply,
* a ClassicReAct run (which uses the same helper under the hood) that
  exposes its ``Thought / Action / Observation`` trajectory as typed
  ``ReActStep`` records.

Run::

    export DS_API_KEY=sk-...
    python examples/03_pattern_schema.py
"""

from __future__ import annotations

import asyncio
import os

from agentkits import (
    ClassicReActAgent,
    OpenAIChatModel,
    PatternSchema,
    Toolkit,
    ToolResponse,
)


# A minimal extraction schema: "- name: John Doe, age: 33" style lines.
PERSON_SCHEMA = PatternSchema.build(
    r"""
    (?:^|\n)\s*-\s*name\s*:\s*(?P<name>[^,\n]+),\s*
    age\s*:\s*(?P<age>\d+)
    """,
    verbose=True,
    multiline=True,
)


def build_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def add(a: int, b: int) -> ToolResponse:
        """Add two integers.

        Args:
            a: First.
            b: Second.
        """
        return ToolResponse.from_value(str(a + b))

    return tk


async def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("DS_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DS_API_KEY first.")

    # --- Demo 1: regex extraction on freeform text --------------------
    sample = (
        "Here are the attendees:\n"
        "- name: Alice, age: 30\n"
        "- name: Bob, age: 27\n"
        "- name: Carol Zhou, age: 41\n"
    )
    print("[regex schema] records:")
    for row in PERSON_SCHEMA.match_all(sample):
        print(" ", row)

    # --- Demo 2: ClassicReAct trajectory parsed via PatternSchema ------
    async with OpenAIChatModel(
        model_name="deepseek-v4-pro",
        api_key=api_key,
        base_url="https://api.deepseek.com",
    ) as model:
        agent = ClassicReActAgent(
            model=model,
            toolkit=build_toolkit(),
            max_steps=6,
        )
        result = await agent.run(
            "Use add to compute 21 + 21, then finish with the number.",
        )

    print("\n[classic_react] final:", result.final_answer)
    for step in result.steps:
        print(
            f"  step {step.index}: {step.action_name}[{step.action_arg!s:.40}] "
            f"-> {step.observation!s:.40}",
        )


if __name__ == "__main__":
    asyncio.run(main())
