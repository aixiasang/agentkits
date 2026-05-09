# agentkits

A minimal agent kit over OpenAI Chat Completions, OpenAI Responses, and
Anthropic. References
[openai-agents-python](https://github.com/openai/openai-agents-python)
and [agentscope](https://github.com/modelscope/agentscope) for API
shape; the rest is compressed to four orthogonal layers (message /
model / tool / agent).

## Quick start

```python
import asyncio
from agentkits import OpenAIChatModel, ReActAgent, Toolkit, ToolResponse

tk = Toolkit()

@tk.tool()
def add(a: int, b: int) -> ToolResponse:
    """Add two integers.

    Args:
        a: first.
        b: second.
    """
    return ToolResponse.from_value(str(a + b))

async def main():
    async with OpenAIChatModel(
        model_name="deepseek-v4-pro",
        api_key="sk-...",
        base_url="https://api.deepseek.com",
    ) as model:
        res = await ReActAgent(model=model, toolkit=tk).run("21 + 21?")
        print(res.text())  # "42"

asyncio.run(main())
```

## Agents

All live under `agentkits.agent`, share one `AgentBase.run()` contract,
and return an `AgentResult` with aggregated `usage`.

| Agent | Pattern / paper |
|---|---|
| `ReActAgent` | Modern tool-calling ReAct loop with streaming. |
| `ClassicReActAgent` | Yao et al. 2023 — plain-text `Thought / Action / Observation`, ends on `finish[...]`. |
| `PlanAgent` | Plan-and-execute: planner outputs steps, a child ReAct executes them. |
| `ReflexionAgent` | Shinn et al. 2023 — Actor → Evaluator → verbal Reflector, retry with reflection. |
| `SelfRefineAgent` | Madaan et al. 2023 — draft → feedback → revise. |
| `ReWOOAgent` | Xu et al. 2023 — planner emits the full tool DAG up front; worker executes; solver composes. |

`agent_as_tool(child)` exposes any agent as a regular tool callable.
`handoff(target)` transfers the run (optionally with a history filter)
to another agent mid-conversation.

## Structured output

Pass `structured_model=MyPydanticModel` to any `chat()` / `chat_cb()`;
the response's `parsed` attribute holds a validated instance.

```python
class Answer(BaseModel):
    number: int
    explanation: str

resp = await model.chat(
    [ChatMessageBase.user("What is 21 + 21?")],
    structured_model=Answer,
)
print(resp.parsed)    # Answer(number=42, explanation='…')
```

Provider paths (try native first, fall back gracefully):

* **OpenAI Chat Completions** — `beta.chat.completions.parse` → forced
  tool call → prompt.
* **OpenAI Responses** — `text.format=json_schema` → forced tool call.
* **Anthropic** — forced `tool_use`.
* **Others** — prompt-based JSON + `json_repair` + `TypeAdapter`.
  Subclass and override `_structured` to plug in provider-native APIs.

At the agent level, pass `output_type=MyModel` to `run()`; the agent
runs its loop, then coerces the final answer into `MyModel` and places
it on `result.parsed`.

## Regex-based extraction

For patterns like ReAct's `Thought / Action / Observation` or ReWOO's
`#E1 = tool[args]`, a regex is cheaper than a JSON round-trip and works
on any model:

```python
from agentkits import PatternSchema

step = PatternSchema.build(
    r"""
    Thought\s*(?P<index>\d+)?\s*:\s*(?P<thought>.*?)
    (?=\n\s*Action\s*\d*\s*:)
    \n\s*Action\s*\d*\s*:\s*(?P<tool>[A-Za-z_][\w-]*)\s*\[(?P<arg>.*?)\]
    """,
    verbose=True, dotall=True,
)

for hit in step.match_all(assistant_reply):
    print(hit)  # {'index': '1', 'thought': '…', 'tool': 'add', 'arg': '…'}
```

`ClassicReActAgent` and `ReWOOAgent` use it internally.

## Examples

```bash
export DS_API_KEY=sk-...

python examples/01_quickstart.py              # streaming ReAct + tools
python examples/02_structured_output.py       # structured_model + output_type
python examples/03_pattern_schema.py          # PatternSchema + ClassicReAct
python examples/04_multi_agent.py             # every built-in agent side-by-side
python examples/05_multi_agent_composition.py # agent_as_tool + handoff
```


## License

[Apache-2.0](./LICENSE).
