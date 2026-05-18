from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import (
    ClassicReActAgent,
    PatternSchema,
    Toolkit,
    ToolResponse,
)
from _shared import ali_model, print_result


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
    sample = (
        "Here are the attendees:\n"
        "- name: Alice, age: 30\n"
        "- name: Bob, age: 27\n"
        "- name: Carol Zhou, age: 41\n"
    )
    print("[regex schema] records:")
    for row in PERSON_SCHEMA.match_all(sample):
        print(" ", row)

    async with ali_model() as model:
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
    print_result(result, final_text=result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
