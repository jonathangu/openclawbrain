from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from benchmarks.gold_standard_eval.agent_loop import ToolSpec, run_agent_loop
from benchmarks.gold_standard_eval.state_utils import build_state_from_messages, resolve_embedder, retrieve_prompt_context


API_CODE_RE = re.compile(r'"apiCode"\s*:\s*"([^"]+)"')
API_REQUEST_RE = re.compile(r"api-?request\s*:", re.IGNORECASE)


@dataclass
class ApiBankExample:
    example_id: str
    task: str
    output_text: str
    tool_calls: list[dict[str, Any]]
    tool_names: list[str]


@dataclass
class EvalStats:
    success: bool
    calls_to_first_correct: int
    redundant_calls: int
    predicted_calls: int


DRY_RUN_DATA = {
    "train": [
        {
            "id": "dry-train-1",
            "instruction": "Schedule a yoga session.",
            "input": "Create a Yin yoga session on 2023-06-03 at 19:30 for 90 minutes.",
            "output": (
                "API-Request: [Create_New_Session(session_name='Yin yoga', "
                "session_date='2023-06-03', session_time='19:30', duration_in_minutes=90)]"
            ),
        },
        {
            "id": "dry-train-2",
            "instruction": "Add a medication.",
            "input": "Log aspirin 75mg for every morning.",
            "output": (
                "API-Request: [Add_Medication(medication_name='aspirin 75mg', "
                "dosage='take one tablet every morning')]"
            ),
        },
    ],
    "test": [
        {
            "id": "dry-test-1",
            "instruction": "Schedule a yoga session.",
            "input": "Create a Yin yoga session on 2023-06-03 at 19:30 for 90 minutes.",
            "output": (
                "API-Request: [Create_New_Session(session_name='Yin yoga', "
                "session_date='2023-06-03', session_time='19:30', duration_in_minutes=90)]"
            ),
        },
        {
            "id": "dry-test-2",
            "instruction": "Add a medication.",
            "input": "Log aspirin 75mg for every morning.",
            "output": (
                "API-Request: [Add_Medication(medication_name='aspirin 75mg', "
                "dosage='take one tablet every morning')]"
            ),
        },
    ],
}

API_BANK_LEVEL_FILES: dict[int, dict[str, str]] = {
    1: {
        "train": "training-data/lv1-api-train.json",
        "test": "test-data/level-1-api.json",
    },
    2: {
        "train": "training-data/lv2-api-train.json",
        "test": "test-data/level-2-api.json",
    },
}


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_task_text(example: dict[str, Any]) -> str:
    instruction = _as_str(
        example.get("instruction")
        or example.get("prompt")
        or example.get("question")
        or example.get("query")
        or example.get("task")
    )
    input_text = _as_str(example.get("input") or example.get("context") or example.get("input_text"))
    if instruction and input_text:
        return f"{instruction}\nInput:\n{input_text}"
    return instruction or input_text


def _extract_output_text(example: dict[str, Any]) -> str:
    return _as_str(
        example.get("output")
        or example.get("expected_output")
        or example.get("answer")
        or example.get("target")
        or example.get("response")
    )


def _extract_tool_names(text: str, example: dict[str, Any]) -> list[str]:
    names = set(API_CODE_RE.findall(text))
    for key in ("api_name", "api", "tool_name", "tool"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            names.add(value.strip())
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    names.add(item.strip())
    return sorted(names)


def _extract_bracketed(text: str, start: int) -> tuple[str | None, int | None]:
    depth = 0
    in_quote: str | None = None
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx
    return None, None


def _split_calls(segment: str) -> list[str]:
    calls: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_quote: str | None = None
    escape = False
    start = 0
    for idx, ch in enumerate(segment):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            continue
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            piece = segment[start:idx].strip()
            if piece:
                calls.append(piece)
            start = idx + 1
    tail = segment[start:].strip()
    if tail:
        calls.append(tail)
    return calls


def _ast_to_obj(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_to_obj(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_to_obj(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _ast_to_obj(key): _ast_to_obj(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_to_obj(node.operand)
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.Name):
        return node.id
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    return ""


def _parse_call_text(call_text: str) -> dict[str, Any] | None:
    cleaned = call_text.strip()
    if not cleaned:
        return None
    if "->" in cleaned:
        cleaned = cleaned.split("->", 1)[0].strip()
    try:
        parsed = ast.parse(cleaned, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(parsed, ast.Call):
        return None
    func = parsed.func
    if isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    args: dict[str, Any] = {}
    for kw in parsed.keywords:
        if kw.arg is None:
            continue
        args[kw.arg] = _ast_to_obj(kw.value)
    return {"name": name, "arguments": args}


def _extract_tool_calls_from_output(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    matches: list[str] = []
    for match in API_REQUEST_RE.finditer(text):
        bracket_start = text.find("[", match.end())
        if bracket_start == -1:
            continue
        segment, end_idx = _extract_bracketed(text, bracket_start)
        if segment:
            matches.append(segment)
            if end_idx is not None:
                continue
    if not matches:
        bracket_start = text.find("[")
        if bracket_start != -1:
            segment, _end_idx = _extract_bracketed(text, bracket_start)
            if segment:
                matches.append(segment)
    if not matches:
        call_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        if call_match:
            matches.append(text[call_match.start() :].strip())

    calls: list[dict[str, Any]] = []
    for segment in matches:
        for call_text in _split_calls(segment):
            parsed = _parse_call_text(call_text)
            if parsed:
                calls.append(parsed)
    return calls


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(call.get("name") or call.get("tool") or call.get("api") or ""),
        "arguments": _normalize_value(call.get("arguments") or {}),
    }


def _build_stub_policy():
    def policy(messages: list[dict[str, str]]) -> str:
        brain_context = "\n".join(msg["content"] for msg in messages if "BRAIN_CONTEXT" in msg["content"])
        if brain_context:
            for line in brain_context.splitlines():
                if "TOOL_CALL_JSON:" in line:
                    payload = line.split("TOOL_CALL_JSON:", 1)[-1].strip()
                    try:
                        tool_call = json.loads(payload)
                    except json.JSONDecodeError:
                        tool_call = None
                    if isinstance(tool_call, dict):
                        return json.dumps({"tool_calls": [tool_call]})
        return json.dumps({"final": "OK."})

    return policy


def _build_examples(split: Iterable[dict[str, Any]]) -> list[ApiBankExample]:
    examples: list[ApiBankExample] = []
    for idx, raw in enumerate(split):
        if not isinstance(raw, dict):
            continue
        task = _extract_task_text(raw)
        output_text = _extract_output_text(raw)
        if not task or not output_text:
            continue
        tool_calls = _extract_tool_calls_from_output(output_text)
        tool_names = _extract_tool_names(task + "\n" + output_text, raw)
        example_id = _as_str(raw.get("id") or raw.get("uid") or f"example-{idx}")
        examples.append(
            ApiBankExample(
                example_id=example_id,
                task=task,
                output_text=output_text,
                tool_calls=tool_calls,
                tool_names=tool_names,
            )
        )
    return examples


def _pick_examples(examples: list[ApiBankExample], max_examples: int | None, seed: int) -> list[ApiBankExample]:
    if max_examples is None or max_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), k=max_examples)
    return [examples[idx] for idx in indices]


def _tool_specs(tool_names: list[str]) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for name in tool_names:
        specs[name] = ToolSpec(
            name=name,
            description="API-Bank tool stub.",
            json_schema={"type": "object", "properties": {}},
            func=lambda _args, tool_name=name: f"stubbed:{tool_name}",
        )
    return specs


def _score_example(
    predicted: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    max_calls_cap: int,
) -> EvalStats:
    normalized_pred = [_normalize_call(call) for call in predicted]
    normalized_targets = [_normalize_call(call) for call in targets]

    def is_match(call: dict[str, Any]) -> bool:
        return any(call["name"] == target["name"] and call["arguments"] == target["arguments"] for target in normalized_targets)

    first_correct = None
    for idx, call in enumerate(normalized_pred, start=1):
        if is_match(call):
            first_correct = idx
            break

    success = first_correct is not None
    calls_to_first = first_correct if success else max_calls_cap + 1
    redundant = sum(1 for call in normalized_pred if not is_match(call))
    return EvalStats(
        success=success,
        calls_to_first_correct=calls_to_first,
        redundant_calls=redundant,
        predicted_calls=len(normalized_pred),
    )


def _aggregate(stats: list[EvalStats]) -> dict[str, float]:
    total = len(stats)
    if total == 0:
        return {
            "success_rate": 0.0,
            "avg_calls_to_first_correct": 0.0,
            "avg_redundant_calls": 0.0,
            "avg_predicted_calls": 0.0,
        }
    return {
        "success_rate": sum(1 for item in stats if item.success) / total,
        "avg_calls_to_first_correct": sum(item.calls_to_first_correct for item in stats) / total,
        "avg_redundant_calls": sum(item.redundant_calls for item in stats) / total,
        "avg_predicted_calls": sum(item.predicted_calls for item in stats) / total,
    }


def _build_brain_state(
    examples: list[ApiBankExample],
    max_examples: int,
    embedder: object,
) -> tuple[object, object, list[ApiBankExample]]:
    memories: list[dict[str, str]] = []
    selected = examples[:max_examples]
    for example in selected:
        if not example.tool_calls:
            continue
        tool_call = _normalize_call(example.tool_calls[0])
        content = f"Task: {example.task}\nTOOL_CALL_JSON: {json.dumps(tool_call, sort_keys=True)}"
        memories.append({"role": "system", "content": content})
    graph, index = build_state_from_messages(memories, embedder)
    return graph, index, selected


def _build_messages(task: str, brain_context: str | None) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": "Return JSON with a tool_calls array. Do not execute real APIs.",
        },
    ]
    if brain_context:
        messages.append({"role": "system", "content": f"[BRAIN_CONTEXT]\n{brain_context}"})
    messages.append({"role": "user", "content": task})
    return messages


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        train_split = DRY_RUN_DATA.get(args.train_split, [])
        test_split = DRY_RUN_DATA.get(args.split, [])
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("datasets is required; install with: pip install -e .[eval]") from exc
        data_files = None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            hf_hub_download = None

        if args.level == 3:
            raise SystemExit("API-Bank level 3 is not wired yet; use --level 1 or 2.")

        if hf_hub_download is None:
            sys.stderr.write(
                "hf_hub_download is unavailable; falling back to load_dataset(repo). "
                "This may fail if API-Bank files have mismatched columns.\n"
            )
            dataset = load_dataset(args.dataset)
        else:
            level_files = API_BANK_LEVEL_FILES.get(args.level)
            if level_files is None:
                raise SystemExit(f"Unsupported API-Bank level: {args.level}")
            data_files = {
                "train": hf_hub_download(
                    repo_id=args.dataset,
                    repo_type="dataset",
                    filename=level_files["train"],
                ),
                "test": hf_hub_download(
                    repo_id=args.dataset,
                    repo_type="dataset",
                    filename=level_files["test"],
                ),
            }
            dataset = load_dataset("json", data_files=data_files)
        if args.train_split not in dataset or args.split not in dataset:
            raise SystemExit(f"split '{args.split}' not in dataset; splits={list(dataset.keys())}")
        train_split = dataset[args.train_split]
        test_split = dataset[args.split]

    train_examples = _build_examples(train_split)
    test_examples = _build_examples(test_split)
    test_examples = _pick_examples(test_examples, args.max_examples, args.seed)

    brain_graph = None
    brain_index = None
    embedder = None
    brain_selected: list[ApiBankExample] = []
    if args.brain_examples > 0:
        embedder = resolve_embedder()
        brain_graph, brain_index, brain_selected = _build_brain_state(train_examples, args.brain_examples, embedder)

    policy = _build_stub_policy() if args.stub_model else None
    use_openai = args.allow_openai and not args.stub_model

    baseline_stats: list[EvalStats] = []
    brain_stats: list[EvalStats] = []
    max_calls_cap = max(1, args.max_steps)

    for example in test_examples:
        tool_call_trace: list[dict[str, Any]] = []
        tools = _tool_specs(example.tool_names or [call["name"] for call in example.tool_calls])
        messages = _build_messages(example.task, None)
        run_agent_loop(
            messages,
            tools=tools,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            stub_policy=policy,
            use_openai=use_openai,
            execute_tools=False,
            stop_on_tool_call=True,
            tool_call_trace=tool_call_trace,
        )
        baseline_stats.append(_score_example(tool_call_trace, example.tool_calls, max_calls_cap=max_calls_cap))

        if brain_graph is None or brain_index is None or embedder is None:
            continue

        brain_context, _stats = retrieve_prompt_context(
            brain_graph,
            brain_index,
            embedder,
            example.task,
            top_k=3,
            max_prompt_context_chars=args.max_prompt_context_chars,
        )
        tool_call_trace = []
        messages = _build_messages(example.task, brain_context)
        run_agent_loop(
            messages,
            tools=tools,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            stub_policy=policy,
            use_openai=use_openai,
            execute_tools=False,
            stop_on_tool_call=True,
            tool_call_trace=tool_call_trace,
        )
        brain_stats.append(_score_example(tool_call_trace, example.tool_calls, max_calls_cap=max_calls_cap))

    baseline_summary = _aggregate(baseline_stats)
    brain_summary = _aggregate(brain_stats) if brain_stats else {}

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "level": args.level,
        "split": args.split,
        "train_split": args.train_split,
        "examples_evaluated": len(test_examples),
        "brain_examples": args.brain_examples,
        "max_steps": args.max_steps,
        "max_prompt_context_chars": args.max_prompt_context_chars,
        "stub_model": args.stub_model,
        "allow_openai": args.allow_openai,
        "baseline_success_rate": baseline_summary["success_rate"],
        "baseline_avg_calls_to_first_correct": baseline_summary["avg_calls_to_first_correct"],
        "baseline_avg_redundant_calls": baseline_summary["avg_redundant_calls"],
        "baseline_avg_predicted_calls": baseline_summary["avg_predicted_calls"],
    }

    if brain_summary:
        summary.update(
            {
                "brain_success_rate": brain_summary["success_rate"],
                "brain_avg_calls_to_first_correct": brain_summary["avg_calls_to_first_correct"],
                "brain_avg_redundant_calls": brain_summary["avg_redundant_calls"],
                "brain_avg_predicted_calls": brain_summary["avg_predicted_calls"],
                "brain_vs_baseline_calls_to_first_correct_delta": (
                    brain_summary["avg_calls_to_first_correct"] - baseline_summary["avg_calls_to_first_correct"]
                ),
                "brain_selected_examples": len(brain_selected),
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API-Bank tool-use evaluation.")
    parser.add_argument("--dataset", default="liminghao1630/API-Bank", help="HuggingFace dataset name")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1, help="API-Bank level subset")
    parser.add_argument("--split", default="test", help="Evaluation split")
    parser.add_argument("--train-split", default="train", help="Training split for brain context")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling")
    parser.add_argument("--max-steps", type=int, default=4, help="Max model calls per example")
    parser.add_argument("--max-prompt-context-chars", type=int, default=20000, help="Brain context cap")
    parser.add_argument("--brain-examples", type=int, default=64, help="Number of training examples to store in memory")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name for OpenAI mode")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--allow-openai", action="store_true", help="Allow OpenAI calls if API key is set")
    parser.add_argument("--stub-model", action="store_true", help="Force stub model (no external API calls)")
    parser.add_argument("--dry-run", action="store_true", help="Use bundled synthetic examples (offline)")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if not args.allow_openai:
        args.stub_model = True

    summary = run(args)
    payload = json.dumps(summary, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
