from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from agentkits import (
    ChatMessageBase,
    ReActAgent,
    Toolkit,
    ToolResponse,
)
from _shared import RunPrinter, ali_model, print_result


class Answer(BaseModel):
    number: int
    explanation: str


class Receipt(BaseModel):
    result: int
    lhs: int
    rhs: int
    note: str


def build_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def add(a: int, b: int) -> ToolResponse:
        """Sum two integers.

        Args:
            a: First.
            b: Second.
        """
        return ToolResponse.from_value(str(a + b))

    return tk


async def main() -> None:
    async with ali_model() as model:
        resp = await model.chat(
            [
                ChatMessageBase.user(
                    "What is 21 + 21? Explain in one short sentence.",
                ),
            ],
            structured_model=Answer,
        )
        print("[structured_model]", resp.parsed)

        user_input = "Use add to compute 20 plus 22, then return a receipt."
        printer = RunPrinter("02 structured output: agent output_type")
        printer.start(user_input)
        agent = ReActAgent(
            model=model,
            toolkit=build_toolkit(),
            system_prompt=(
                "Use the add tool. Reply with the numeric result before the "
                "structured-output pass runs."
            ),
            max_iterations=4,
        )
        result = await agent.run(
            user_input,
            on_message=printer.on_message,
            output_type=Receipt,
        )
        print("[agent.output_type]", result.parsed)
        print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
