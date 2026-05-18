from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Sequence

from agentkits import ChatMessageBase, OpenAIChatModel, PlanAgent


ALI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_MODEL = "qwen3-max"


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def require_ali_api_key() -> str:
    load_env()
    api_key = os.environ.get("ALI_API_KEY") or os.environ.get("ali_api_key")
    if not api_key:
        raise SystemExit("Set ALI_API_KEY or ali_api_key in your environment first.")
    return api_key


@asynccontextmanager
async def ali_model(*, stream: bool = False) -> AsyncIterator[OpenAIChatModel]:
    require_ali_api_key()
    async with OpenAIChatModel.from_ali_env(stream=stream) as model:
        yield model


class RunPrinter:
    def __init__(self, title: str) -> None:
        self.title = title
        self._assistant_header_open = False
        self._plan_steps: list[str] = []
        self._plan_status: list[str] = []
        self._active_plan_steps: dict[str, list[int]] = {}

    def start(self, user_input: str) -> None:
        print(f"\n=== {self.title} ===")
        print("[model]", ALI_MODEL)
        print("[user input]")
        print(user_input)

    def plan(self, steps: Sequence[str]) -> None:
        self._plan_steps = list(steps)
        self._plan_status = ["pending" for _ in self._plan_steps]
        self._active_plan_steps.clear()
        print("[plan generated]")
        for i, step in enumerate(self._plan_steps, start=1):
            print(f"  [{self._plan_status[i - 1]}] {i}. {step}")

    def on_chunk(self, chunk: Any) -> None:
        if chunk.delta_text:
            if not self._assistant_header_open:
                print("[assistant stream]")
                self._assistant_header_open = True
            print(chunk.delta_text, end="", flush=True)
        if chunk.is_last and self._assistant_header_open:
            print()
            self._assistant_header_open = False

    def on_message(self, msg: ChatMessageBase) -> None:
        if msg.role == "assistant":
            if msg.tool_calls:
                for call in msg.tool_calls:
                    self._mark_plan_running(call)
                print("[tool calls]")
                for call in msg.tool_calls:
                    print(f"  - {call.name}({format_json(call.input or {})})")
            text = msg.text.strip()
            if text and not msg.tool_calls:
                self.finish_plan()
                print("[assistant]")
                print(text)
            return

        if msg.role == "tool":
            print("[tool results]")
            for result in msg.tool_results:
                print(f"  - {result.name}: {one_line(result.text)}")
                self._mark_plan_finished(
                    result.id,
                    result.name,
                    failed=result.is_error,
                )

    def finish_plan(self) -> None:
        if not self._plan_steps:
            return
        for i, status in enumerate(list(self._plan_status)):
            if status == "completed" or status == "failed":
                continue
            if status == "running" or _looks_like_final_step(self._plan_steps[i]):
                self._update_plan(i, "completed")
            else:
                self._update_plan(i, "skipped")

    def _mark_plan_running(self, call: Any) -> None:
        idx = self._find_plan_step(
            call.name,
            args=call.input or {},
            allow_running=True,
        )
        if idx is None:
            return
        self._active_plan_steps.setdefault(call.id, []).append(idx)
        self._update_plan(idx, "running")

    def _mark_plan_finished(
        self,
        call_id: str,
        tool_name: str,
        *,
        failed: bool,
    ) -> None:
        indexes = self._active_plan_steps.pop(call_id, [])
        if not indexes:
            idx = self._find_plan_step(tool_name, allow_running=True)
            indexes = [] if idx is None else [idx]
        for idx in indexes:
            self._update_plan(idx, "failed" if failed else "completed")

    def _find_plan_step(
        self,
        tool_name: str,
        *,
        args: dict[str, Any] | None = None,
        allow_running: bool = False,
    ) -> int | None:
        if not self._plan_steps:
            return None

        status_groups = [{"pending"}]
        if allow_running:
            status_groups.append({"running"})

        if args:
            for statuses in status_groups:
                for i, step in enumerate(self._plan_steps):
                    if (
                        self._plan_status[i] in statuses
                        and _matches_tool(step, tool_name)
                        and _matches_args(step, args)
                    ):
                        return i

            for statuses in status_groups:
                for i, step in enumerate(self._plan_steps):
                    if self._plan_status[i] in statuses and _matches_args(step, args):
                        return i

        for statuses in status_groups:
            for i, step in enumerate(self._plan_steps):
                if self._plan_status[i] in statuses and _matches_tool(step, tool_name):
                    return i
        return None

    def _update_plan(self, index: int, status: str) -> None:
        if not self._plan_steps:
            return
        current = self._plan_status[index]
        if current == status:
            return
        self._plan_status[index] = status
        print(
            f"[plan update] [{status}] {index + 1}. "
            f"{self._plan_steps[index]}",
        )


def print_result(result: Any, *, final_text: str | None = None) -> None:
    handoff_to = (result.metadata or {}).get("handoff_to")
    if handoff_to:
        print(f"[handoff] target={handoff_to}")

    answer = final_text if final_text is not None else result.text()
    print("[final output]")
    print(answer.strip())

    usage = result.usage.to_dict() if result.usage else None
    print(
        f"[stats] iterations={result.iterations} "
        f"tool_calls={result.tool_calls} usage={usage}",
    )


def format_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def one_line(text: str, *, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _matches_tool(step: str, tool_name: str) -> bool:
    text = _normalize(step)
    tool_text = _normalize(tool_name)
    if tool_text in text:
        return True

    terms = [part for part in tool_text.split() if len(part) > 2]
    if terms and all(part in text for part in terms):
        return True

    long_terms = [part for part in terms if len(part) >= 5]
    return bool(long_terms and any(part in text for part in long_terms))


def _looks_like_final_step(step: str) -> bool:
    text = _normalize(step)
    return any(
        word in text
        for word in (
            "answer",
            "compose",
            "final",
            "produce",
            "report",
            "return",
            "summary",
            "summarize",
        )
    )


def _matches_args(step: str, args: dict[str, Any]) -> bool:
    text = _normalize(step)
    terms = _arg_terms(args)
    if not terms:
        return False
    return any(term in text for term in terms)


def _arg_terms(value: Any) -> list[str]:
    terms: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            terms.extend(_arg_terms(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            terms.extend(_arg_terms(item))
    elif isinstance(value, str):
        normalized = _normalize(value)
        if len(normalized) >= 4:
            terms.append(normalized)
        terms.extend(part for part in normalized.split() if len(part) >= 3)
    return terms


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


class PlanProgressAgent(PlanAgent):
    def __init__(
        self,
        *,
        on_plan: Callable[[Sequence[str]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._on_plan = on_plan

    async def _plan(
        self,
        user_text: str,
    ) -> list[str]:
        steps = await super()._plan(user_text)
        if self._on_plan is not None:
            self._on_plan(steps)
        return steps
