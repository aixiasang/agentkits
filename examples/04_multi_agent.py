from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkits import (
    ClassicReActAgent,
    PlanAgent,
    ReActAgent,
    ReflexionAgent,
    ReWOOAgent,
    SelfRefineAgent,
    Toolkit,
    ToolResponse,
)
from _shared import ali_model


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
    task = "Use the tools to compute (12 + 30) * 2. Reply only with the number."

    async with ali_model() as model:
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
            if name in {"plan", "rewoo"} and getattr(res, "plan", None):
                print(f"[{name:>11}] plan={getattr(res, 'plan')}")
            print(
                f"[{name:>11}] answer={answer!r:<14}  "
                f"iterations={res.iterations}  tool_calls={res.tool_calls}  "
                f"usage={usage}",
            )


if __name__ == "__main__":
    asyncio.run(main())
