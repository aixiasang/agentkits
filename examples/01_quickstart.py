# -*- coding: utf-8 -*-
"""ReAct quickstart.

Run::

    export DS_API_KEY=sk-...
    python examples/01_quickstart.py
"""

from __future__ import annotations

import asyncio
import os

from agentkits import (
    ChatMessageBase,
    ChatStreamChunk,
    OpenAIChatModel,
    ReActAgent,
    Toolkit,
    ToolResponse,
)


def build_toolkit() -> Toolkit:
    tk = Toolkit()

    @tk.tool()
    def add(a: int, b: int) -> ToolResponse:
        """Add two integers.

        Args:
            a: The first integer.
            b: The second integer.
        """
        return ToolResponse.from_value(str(a + b))

    @tk.tool()
    def multiply(a: int, b: int) -> ToolResponse:
        """Multiply two integers.

        Args:
            a: The first integer.
            b: The second integer.
        """
        return ToolResponse.from_value(str(a * b))

    return tk


async def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("DS_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DS_API_KEY in your environment first.")

    async with OpenAIChatModel(
        model_name="deepseek-v4-pro",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        stream=True,
    ) as model:
        agent = ReActAgent(
            model=model,
            toolkit=build_toolkit(),
            system_prompt=(
                "You are a concise arithmetic assistant. Use the provided "
                "tools; reply with just the final number."
            ),
            max_iterations=6,
        )

        def on_chunk(chunk: ChatStreamChunk) -> None:
            if chunk.delta_text:
                print(chunk.delta_text, end="", flush=True)

        def on_message(msg: ChatMessageBase) -> None:
            if msg.role == "assistant" and msg.tool_calls:
                print()
            elif msg.role == "tool":
                for res in msg.tool_results:
                    print(f"[tool:{res.name} -> {res.text}]")

        result = await agent.run(
            "Compute (12 + 30) * 2. Only output the result.",
            on_chunk=on_chunk,
            on_message=on_message,
        )

    print()
    print(
        f"[done] iterations={result.iterations} "
        f"tool_calls={result.tool_calls} "
        f"usage={result.usage.to_dict() if result.usage else None}",
    )
    print(f"answer: {result.text()}")


if __name__ == "__main__":
    asyncio.run(main())
