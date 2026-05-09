# -*- coding: utf-8 -*-
"""Regex-based lightweight schema extraction.

Often the cheapest way to get "structured output" from an LLM is to let
it speak naturally and then pluck the fields you need with a regex.
This is how the classic ReAct / ReWOO papers format their traces
(``Thought / Action / Observation``, ``#E1 = tool[args]``). Compared to
JSON-schema structured output, the regex path costs zero extra tokens
and works on any model.

:class:`PatternSchema` wraps a compiled regex so callers don't have to
manage flags, groups, or multi-match iteration. Use :meth:`match_one`
for a single record, :meth:`match_all` for a list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PatternSchema:
    """A named-group regex you can apply to LLM output.

    Example::

        STEP = PatternSchema.build(
            r'''
            Thought\\s*(?P<index>\\d+)?\\s*:\\s*(?P<thought>.*?)
            (?=\\n\\s*Action\\s*\\d*\\s*:)
            \\n\\s*Action\\s*\\d*\\s*:\\s*
            (?P<tool>[A-Za-z_][\\w-]*)\\s*\\[(?P<arg>.*?)\\]
            ''',
            verbose=True, dotall=True,
        )

        STEP.match_all(assistant_text)
        # [{"index": "1", "thought": "…", "tool": "add", "arg": "…"}, ...]
    """

    pattern: re.Pattern
    fields: tuple[str, ...] | None = None
    """Optional subset of named groups to keep (in this order)."""

    _defaults: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        pattern: str,
        *,
        verbose: bool = False,
        dotall: bool = False,
        ignorecase: bool = False,
        multiline: bool = False,
        fields: Iterable[str] | None = None,
        defaults: dict[str, str] | None = None,
    ) -> "PatternSchema":
        """Compile ``pattern`` with the common flags pre-wired."""
        flags = 0
        if verbose:
            flags |= re.VERBOSE
        if dotall:
            flags |= re.DOTALL
        if ignorecase:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        compiled = re.compile(pattern, flags)
        return cls(
            pattern=compiled,
            fields=tuple(fields) if fields else None,
            _defaults=dict(defaults or {}),
        )

    @property
    def groups(self) -> tuple[str, ...]:
        """All named groups declared by the pattern, in order."""
        return tuple(self.pattern.groupindex.keys())

    def match_one(self, text: str) -> dict[str, str] | None:
        """Return the first non-overlapping match as a dict, or ``None``."""
        m = self.pattern.search(text or "")
        if m is None:
            return None
        return self._to_dict(m)

    def match_all(self, text: str) -> list[dict[str, str]]:
        """Return every non-overlapping match as a list of dicts."""
        if not text:
            return []
        return [self._to_dict(m) for m in self.pattern.finditer(text)]

    def _to_dict(self, m: re.Match) -> dict[str, str]:
        gd = m.groupdict()
        raw: dict[str, str] = {}
        keys = self.fields or tuple(self.pattern.groupindex.keys())
        for k in keys:
            v = gd.get(k)
            if v is None:
                v = self._defaults.get(k, "")
            raw[k] = v.strip() if isinstance(v, str) else v
        return raw
