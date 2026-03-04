from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    func: Callable[[dict[str, Any]], str]
    description: str = ""
    json_schema: dict[str, Any] | None = None


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    raw: Any | None = None


@dataclass
class LoopMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def add_usage(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if value is None:
                continue
            current = getattr(self, key)
            setattr(self, key, value if current is None else current + value)


def _json_tool_payload(tool_calls: list[dict[str, Any]]) -> str:
    return json.dumps({"tool_calls": tool_calls})


def _parse_tool_payload(text: str) -> tuple[list[dict[str, Any]], str]:
    stripped = text.strip()
    if not stripped:
        return [], ""
    if not stripped.startswith("{"):
        return [], stripped
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return [], stripped
    if isinstance(payload, dict):
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            return payload["tool_calls"], ""
        if "tool_call" in payload and isinstance(payload["tool_call"], dict):
            return [payload["tool_call"]], ""
        if "final" in payload and isinstance(payload["final"], str):
            return [], payload["final"]
    return [], stripped


def _render_tools(tools: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    rendered = []
    for tool in tools.values():
        if tool.json_schema is None:
            schema = {"type": "object", "properties": {}}  # minimal
        else:
            schema = tool.json_schema
        rendered.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            }
        )
    return rendered


def _call_openai(messages: list[dict[str, str]], tools: dict[str, ToolSpec], model: str, temperature: float) -> ModelResponse:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed") from exc

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=_render_tools(tools) if tools else None,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    message = choice.message

    tool_calls: list[dict[str, Any]] = []
    if message.tool_calls:
        for call in message.tool_calls:
            tool_calls.append({"name": call.function.name, "arguments": json.loads(call.function.arguments)})

    content = message.content or ""
    usage = response.usage.model_dump() if response.usage is not None else None
    return ModelResponse(content=content, tool_calls=tool_calls, usage=usage, raw=response)


def _call_stub(messages: list[dict[str, str]], tools: dict[str, ToolSpec], policy: Callable[[list[dict[str, str]]], str] | None) -> ModelResponse:
    if policy is None:
        # Default stub: if a preference exists in BRAIN_CONTEXT, answer directly.
        brain_context = "\n".join(msg["content"] for msg in messages if "BRAIN_CONTEXT" in msg["content"])
        last_user = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        if "preferred" in last_user.lower() or "preference" in last_user.lower():
            if brain_context:
                return ModelResponse(content=json.dumps({"final": "Preference remembered from context."}))
            if "get_preference" in tools:
                return ModelResponse(
                    content=_json_tool_payload(
                        [
                            {"name": "get_preference", "arguments": {"key": "generic"}},
                        ]
                    )
                )
        return ModelResponse(content=json.dumps({"final": "OK."}))

    return ModelResponse(content=policy(messages))


def run_agent_loop(
    messages: list[dict[str, str]],
    tools: dict[str, ToolSpec] | None = None,
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_steps: int = 8,
    tool_result_max_chars: int = 2000,
    stub_policy: Callable[[list[dict[str, str]]], str] | None = None,
    use_openai: bool | None = None,
    execute_tools: bool = True,
    stop_on_tool_call: bool = False,
    tool_call_trace: list[dict[str, Any]] | None = None,
) -> tuple[str, LoopMetrics]:
    tools = tools or {}
    metrics = LoopMetrics()
    if use_openai is None:
        use_openai = bool(os.environ.get("OPENAI_API_KEY"))

    for step_idx in range(max_steps):
        metrics.llm_calls += 1
        if use_openai:
            try:
                response = _call_openai(messages, tools, model, temperature)
            except Exception:
                response = _call_stub(messages, tools, stub_policy)
        else:
            response = _call_stub(messages, tools, stub_policy)

        tool_calls = response.tool_calls
        if not tool_calls:
            tool_calls, final_text = _parse_tool_payload(response.content)
        else:
            final_text = ""

        metrics.add_usage(response.usage)

        if tool_calls:
            if tool_call_trace is not None:
                for call in tool_calls:
                    tool_call_trace.append(
                        {
                            "step": step_idx + 1,
                            "name": call.get("name"),
                            "arguments": call.get("arguments"),
                        }
                    )
            if not execute_tools:
                return "", metrics
            for call in tool_calls:
                name = call.get("name")
                args = call.get("arguments") or {}
                tool = tools.get(name)
                if tool is None:
                    output = f"Tool '{name}' not found."
                else:
                    metrics.tool_calls += 1
                    output = tool.func(args)
                output = str(output)
                if tool_result_max_chars is not None and tool_result_max_chars > 0:
                    output = output[:tool_result_max_chars]
                messages.append({"role": "tool", "name": name or "", "content": output})
            if stop_on_tool_call:
                return "", metrics
            continue

        if final_text:
            return final_text, metrics

    return "", metrics
