# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..message import ChatMessageBase


class SessionBase(ABC):
    """Abstract conversation store keyed by session id."""

    @abstractmethod
    async def load(self, session_id: str) -> list[ChatMessageBase]: ...

    @abstractmethod
    async def append(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None: ...

    @abstractmethod
    async def save(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None: ...

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...

    @abstractmethod
    async def list_sessions(self) -> AsyncIterator[str]:
        if False:
            yield ""
