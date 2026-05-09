# -*- coding: utf-8 -*-
"""Compare every built-in reasoning agent on the same task.

Covers:

* ReActAgent           - modern tool-calling loop.
* ClassicReActAgent    - paper-faithful Thought/Action/Observation.
* PlanAgent            - planner -> ReAct executor.
* ReWOOAgent           - plan-then-execute-then-solve (fewer round trips).
* ReflexionAgent       - retry with verbal self-reflection.
* SelfRefineAgent      - draft -> critique -> revise.

Run::

    export DS_API_KEY=sk-...
    python examples/04_multi_agent.py
"""

from __future__ import annotations

import asyncio
import os

from agentkits import (
    ClassicReActAgent,
    OpenAIChatModel,
    PlanAgent,
    ReActAgent,
    ReflexionAgent,
    ReWOOAgent,
    SelfRefineAgent,
    Toolkit,
    ToolResponse,
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

    @tk.tool()
    def multiply(a: int, b: int) -> ToolResponse:
        """Multiply two integers.

        Args:
            a: First.
            b: Second.
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
        raise SystemExit("Set DS_API_KEY first.")

    task = "Use the tools to compute (12 + 30) * 2. Reply only with the number."

    async with OpenAIChatModel(
        model_name="deepseek-v4-pro",
        api_key=api_key,
        base_url="https://api.deepseek.com",
    ) as model:
        tk = build_toolkit()

        agents = [
            ("react", ReActAgent(model=model, toolkit=tk, max_iterations=6)),
            ("classic", ClassicReActAgent(model=model, toolkit=tk, max_steps=8)),
            ("plan", PlanAgent(model=model, toolkit=tk, max_steps=4, max_iterations=6)),
            ("rewoo", ReWOOAgent(model=model, toolkit=tk, max_steps=4)),
            (
                "reflexion",
                ReflexionAgent(
                    model=model,
                    toolkit=tk,
                    max_trials=2,
                    max_iterations=4,
                    evaluator=lambda _, ans: "84" in ans,
                ),
            ),
            (
                "self_refine",
                SelfRefineAgent(
                    model=model, toolkit=tk, max_rounds=1, max_iterations=4,
                ),
            ),
        ]

        for name, agent in agents:
            try:
                res = await agent.run(task)
            except Exception as e:
                print(f"[{name:>11}] ERROR {type(e).__name__}: {e}")
                continue
            usage = res.usage.to_dict() if res.usage else None
            answer = res.text()
            if name == "classic":
                answer = getattr(res, "final_answer", answer)
            print(
                f"[{name:>11}] answer={answer!r:<14}  "
                f"iterations={res.iterations}  tool_calls={res.tool_calls}  "
                f"usage={usage}",
            )


if __name__ == "__main__":
    asyncio.run(main())
