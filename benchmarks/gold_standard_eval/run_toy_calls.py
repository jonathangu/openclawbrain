from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

from benchmarks.gold_standard_eval.agent_loop import ToolSpec, run_agent_loop
from benchmarks.gold_standard_eval.state_utils import build_state_from_messages, resolve_embedder, retrieve_prompt_context


PREFERENCES = {
    "unit": "miles",
    "timezone": "America/Los_Angeles",
    "currency": "USD",
    "temperature": "Fahrenheit",
    "date_format": "MM/DD/YYYY",
}


@dataclass
class ToyTask:
    task_id: str
    question: str
    preference_key: str


def build_tasks() -> list[ToyTask]:
    tasks: list[ToyTask] = []
    for idx in range(1, 5):
        tasks.append(ToyTask(f"unit-{idx}", f"What is my preferred unit for distance? ({idx})", "unit"))
    for idx in range(1, 5):
        tasks.append(ToyTask(f"tz-{idx}", f"Use my default timezone for scheduling ({idx}).", "timezone"))
    for idx in range(1, 5):
        tasks.append(ToyTask(f"currency-{idx}", f"Quote prices in my default currency ({idx}).", "currency"))
    for idx in range(1, 5):
        tasks.append(ToyTask(f"temp-{idx}", f"Report the temperature in my preferred scale ({idx}).", "temperature"))
    for idx in range(1, 5):
        tasks.append(ToyTask(f"date-{idx}", f"Format dates using my preferred format ({idx}).", "date_format"))
    return tasks


def _preference_text() -> str:
    return (
        "User preferences:\n"
        f"- unit={PREFERENCES['unit']}\n"
        f"- timezone={PREFERENCES['timezone']}\n"
        f"- currency={PREFERENCES['currency']}\n"
        f"- temperature={PREFERENCES['temperature']}\n"
        f"- date_format={PREFERENCES['date_format']}\n"
    )


def _policy_factory(preferences: dict[str, str]):
    def policy(messages: list[dict[str, str]]) -> str:
        last_user = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
        tool_msg = next((msg["content"] for msg in reversed(messages) if msg["role"] == "tool"), "")
        if tool_msg:
            return json.dumps({"final": f"Preference: {tool_msg}"})

        brain_context = "\n".join(msg["content"] for msg in messages if "BRAIN_CONTEXT" in msg["content"])
        key = None
        lowered = last_user.lower()
        if "unit" in lowered or "distance" in lowered:
            key = "unit"
        elif "timezone" in lowered or "schedule" in lowered:
            key = "timezone"
        elif "currency" in lowered or "prices" in lowered:
            key = "currency"
        elif "temperature" in lowered or "scale" in lowered:
            key = "temperature"
        elif "date" in lowered and "format" in lowered:
            key = "date_format"

        if key and key in brain_context:
            return json.dumps({"final": f"Preference: {preferences[key]}"})
        if key:
            return json.dumps({"tool_calls": [{"name": "get_preference", "arguments": {"key": key}}]})
        return json.dumps({"final": "OK."})

    return policy


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = build_tasks()
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    embedder = resolve_embedder()
    preference_message = {"role": "system", "content": _preference_text()}
    graph, index = build_state_from_messages([preference_message], embedder)

    tool = ToolSpec(
        name="get_preference",
        description="Fetch the user's saved preference for a key.",
        json_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        func=lambda args_dict: preferences_lookup(args_dict),
    )

    policy = _policy_factory(PREFERENCES)

    baseline_metrics = []
    brain_metrics = []

    for task in tasks:
        base_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": task.question},
        ]
        _final, metrics = run_agent_loop(
            list(base_messages),
            tools={"get_preference": tool},
            stub_policy=policy,
            tool_result_max_chars=args.tool_result_max_chars,
        )
        baseline_metrics.append(metrics)

        brain_context, _stats = retrieve_prompt_context(
            graph,
            index,
            embedder,
            task.question,
            top_k=3,
            max_prompt_context_chars=args.max_prompt_context_chars,
        )
        brain_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": brain_context},
            {"role": "user", "content": task.question},
        ]
        _final, metrics = run_agent_loop(
            list(brain_messages),
            tools={"get_preference": tool},
            stub_policy=policy,
            tool_result_max_chars=args.tool_result_max_chars,
        )
        brain_metrics.append(metrics)

    summary = {
        "tasks": len(tasks),
        "baseline_avg_calls": _avg([m.llm_calls for m in baseline_metrics]),
        "baseline_avg_tool_calls": _avg([m.tool_calls for m in baseline_metrics]),
        "brain_avg_calls": _avg([m.llm_calls for m in brain_metrics]),
        "brain_avg_tool_calls": _avg([m.tool_calls for m in brain_metrics]),
        "tool_result_max_chars": args.tool_result_max_chars,
        "brain_context_max_chars": args.max_prompt_context_chars,
        "embedder_mode": embedder.mode,
        "embedder_name": embedder.name,
    }
    summary["baseline_avg_llm_calls"] = summary["baseline_avg_calls"]
    summary["brain_avg_llm_calls"] = summary["brain_avg_calls"]
    return summary


def preferences_lookup(args_dict: dict[str, Any]) -> str:
    key = str(args_dict.get("key", ""))
    return PREFERENCES.get(key, "unknown")


def _avg(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy call-counting evaluation.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--max-prompt-context-chars", type=int, default=20000, help="Brain context cap")
    parser.add_argument("--tool-result-max-chars", type=int, default=2000, help="Tool result cap")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    summary = run(args)
    payload = json.dumps(summary, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
