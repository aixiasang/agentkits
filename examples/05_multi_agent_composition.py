# -*- coding: utf-8 -*-
"""Hierarchical composition: sub-agents as tools, and runtime handoff.

Two patterns side by side:

1. ``agent_as_tool(child)`` exposes any agent as a regular tool. The
   parent's ReAct loop calls it like any other function; the child's
   loop runs to completion and returns its final answer to the parent.

2. ``handoff(target)`` hands the conversation off mid-run. When the
   main agent calls the handoff tool, ``ReActAgent`` detects the
   marker, stops its own loop, and the *target* agent takes over with
   the full (optionally filtered) history. The final ``AgentResult``
   carries ``metadata={"handoff_to": <target name>}``.

Run::

    export DS_API_KEY=sk-...
    python examples/05_multi_agent_composition.py
"""

from __future__ import annotations

import asyncio
import os

from agentkits import (
    OpenAIChatModel,
    ReActAgent,
    Toolkit,
    ToolResponse,
    agent_as_tool,
    handoff,
)


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
        "I need 21 + 21, then double that. What's the final number?",
    )
    print(
        f"[agent_as_tool] answer={result.text()!r}  "
        f"tool_calls={result.tool_calls}  "
        f"usage={result.usage.to_dict() if result.usage else None}",
    )


async def demo_handoff(model) -> None:
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

    result = await router.run("What is 21 + 21?")
    print(
        f"[handoff] answer={result.text()!r}  "
        f"handoff_to={(result.metadata or {}).get('handoff_to')!r}  "
        f"tool_calls={result.tool_calls}  "
        f"usage={result.usage.to_dict() if result.usage else None}",
    )


async def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("DS_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DS_API_KEY first.")

    async with OpenAIChatModel(
        model_name="deepseek-v4-pro",
        api_key=api_key,
        base_url="https://api.deepseek.com",
    ) as model:
        await demo_agent_as_tool(model)
        await demo_handoff(model)


if __name__ == "__main__":
    asyncio.run(main())
