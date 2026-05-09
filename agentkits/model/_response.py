# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from ..message import ChatMessageBase, ChatUsage
from ..utils._id import make_resp_id


@dataclass
class ChatResponse:
    message: ChatMessageBase
    id: str = field(default_factory=make_resp_id)
    created: int = 0
    type: Literal["chat"] = field(default="chat")
    usage: ChatUsage | None = None
    metadata: dict[str, Any] | None = None
    finish_reason: str | None = None
    parsed: Any = None
    """Populated when a structured output was requested and parsed."""

    @classmethod
    def from_chunks(cls, chunks: Iterable["ChatStreamChunk"]) -> "ChatResponse":
        last: ChatStreamChunk | None = None
        for c in chunks:
            last = c
        if last is None:
            raise ValueError("from_chunks() received no chunks")
        return last.to_response()


@dataclass
class ChatStreamChunk:
    """One chunk of a streaming chat response.

    ``message`` is an accumulated snapshot up to this chunk. Tool calls
    are populated only on the terminal chunk (``is_last=True``); mid-
    stream snapshots carry text/reasoning only.
    """

    message: ChatMessageBase
    delta_text: str = ""
    delta_reasoning: str = ""
    is_last: bool = False
    id: str = field(default_factory=make_resp_id)
    created: int = 0
    usage: ChatUsage | None = None
    metadata: dict[str, Any] | None = None
    finish_reason: str | None = None

    def to_response(self) -> ChatResponse:
        return ChatResponse(
            message=self.message,
            id=self.id,
            created=self.created,
            usage=self.usage,
            metadata=self.metadata,
            finish_reason=self.finish_reason,
        )
