# -*- coding: utf-8 -*-
"""Structured output two ways.

1. ``model.chat(messages, structured_model=Model)``  -- one-shot parse.
2. ``agent.run(task, output_type=Model)``            -- run the agent
   loop normally, then coerce the final answer into ``Model``.

Run::

    export DS_API_KEY=sk-...
    python examples/02_structured_output.py
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel

from agentkits import (
    ChatMessageBase,
    OpenAIChatModel,
    ReActAgent,
    Toolkit,
    ToolResponse,
)


class Answer(BaseModel):
    number: int
    explanation: str


class Receipt(BaseModel):
    result: int
    lhs: int
    rhs: int


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
        # (1) Plain structured call.
        resp = await model.chat(
            [ChatMessageBase.user("What is 21 + 21? Explain in one sentence.")],
            structured_model=Answer,
        )
        print("[structured_model]", resp.parsed)

        # (2) Agent + output_type.
        agent = ReActAgent(
            model=model,
            toolkit=build_toolkit(),
            system_prompt="Use the add tool. Reply with only the number.",
            max_iterations=4,
        )
        result = await agent.run(
            "What is 20 plus 22?", output_type=Receipt,
        )
        print("[agent.output_type]", result.parsed)
        print(f"  (usage={result.usage.to_dict() if result.usage else None})")


if __name__ == "__main__":
    asyncio.run(main())
