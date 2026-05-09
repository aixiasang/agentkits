# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Awaitable, Callable

from ..tool import ToolResponse
from ._base import AgentBase


def agent_as_tool(
    agent: AgentBase,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[str], "Awaitable[ToolResponse]"]:
    """Expose ``agent`` as a tool callable taking a ``request: str``.

    Use with ``toolkit.register_tool_function`` to build hierarchical /
    multi-agent setups.
    """
    tool_name = name or agent.name
    tool_description = description or agent.description

    async def _invoke(request: str) -> ToolResponse:
        result = await agent.run(request)
        final = result.final_message
        if final is None:
            return ToolResponse()
        return ToolResponse(content=list(final.content))

    _invoke.__name__ = tool_name  # type: ignore[attr-defined]
    _invoke.__doc__ = (
        f"{tool_description}\n\n"
        f"Args:\n"
        f"    request: Natural-language task description to delegate."
    )
    return _invoke
