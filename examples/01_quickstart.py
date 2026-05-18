from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import (
    ReActAgent,
    Toolkit,
    ToolResponse,
)
from _shared import RunPrinter, ali_model, print_result


def build_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def add(a: int, b: int) -> ToolResponse:
        """Add two numbers.

        Args:
            a: The first number.
            b: The second number.
        """
        return ToolResponse.from_value(str(a + b))

    @tk.tool()
    def multiply(a: int, b: int) -> ToolResponse:
        """Multiply two numbers.

        Args:
            a: The first number.
            b: The second number.
        """
        return ToolResponse.from_value(str(a * b))

    return tk


async def main() -> None:
    user_input = (
        "Use tools to compute the launch-room capacity score: "
        "(12 reserved seats + 30 open seats) * 2."
    )
    printer = RunPrinter("01 quickstart: ReAct tool loop")
    printer.start(user_input)

    async with ali_model() as model:
        agent = ReActAgent(
            model=model,
            toolkit=build_toolkit(),
            system_prompt=(
                "You are a concise arithmetic assistant. Use tools for every "
                "calculation. Reply with the final number and one short label."
            ),
            max_iterations=6,
        )

        result = await agent.run(
            user_input,
            on_message=printer.on_message,
        )

    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
