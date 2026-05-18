from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import (
    ReActAgent,
    Toolkit,
    ToolResponse,
    agent_as_tool,
    handoff,
)
from _shared import RunPrinter, ali_model, print_result


def math_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def add(a: int, b: int) -> ToolResponse:
        """Add.

        Args:
            a: First.
            b: Second.
        """
        return ToolResponse.from_value(str(a + b))

    @tk.tool()
    def multiply(a: int, b: int) -> ToolResponse:
        """Multiply.

        Args:
            a: First.
            b: Second.
        """
        return ToolResponse.from_value(str(a * b))

    return tk


async def demo_agent_as_tool(model) -> None:
    user_input = (
        "Ask the math_agent to solve this subtask: add 21 and 21, then "
        "multiply the result by 2. Return the final number with context."
    )
    printer = RunPrinter("05a composition: agent_as_tool")
    printer.start(user_input)

    math_agent = ReActAgent(
        name="math_agent",
        description="Solves arithmetic questions using add / multiply.",
        model=model,
        toolkit=math_toolkit(),
        system_prompt="Use the tools. Reply with the final number only.",
        max_iterations=4,
    )

    parent_tk = Toolkit()
    parent_tk.register_tool_function(agent_as_tool(math_agent))

    parent = ReActAgent(
        name="parent",
        model=model,
        toolkit=parent_tk,
        system_prompt=(
            "You answer user questions. For arithmetic, delegate to the "
            "`math_agent` tool with the exact sub-question as `request`. "
            "After getting its answer, reply to the user."
        ),
        max_iterations=4,
    )

    result = await parent.run(
        user_input,
        on_message=printer.on_message,
    )
    print_result(result)


async def demo_handoff(model) -> None:
    user_input = (
        "This is an arithmetic request. Route it to the specialist and "
        "compute: (19 + 23) * 3."
    )
    printer = RunPrinter("05b composition: runtime handoff")
    printer.start(user_input)

    math_agent = ReActAgent(
        name="math_agent",
        description="Handles arithmetic by calling add / multiply.",
        model=model,
        toolkit=math_toolkit(),
        system_prompt="Use the tools. Reply with the final number only.",
        max_iterations=4,
    )

    router_tk = Toolkit()
    router_tk.register_tool_function(handoff(math_agent))

    router = ReActAgent(
        name="router",
        model=model,
        toolkit=router_tk,
        system_prompt=(
            "You route questions to specialists. For any arithmetic "
            "question call `transfer_to_math_agent`. Never answer math "
            "yourself."
        ),
        max_iterations=3,
    )

    result = await router.run(
        user_input,
        on_message=printer.on_message,
    )
    print_result(result)


async def main() -> None:
    async with ali_model() as model:
        await demo_agent_as_tool(model)
        await demo_handoff(model)


if __name__ == "__main__":
    asyncio.run(main())
