from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from openclawbrain.store import save_state
from benchmarks.gold_standard_eval.state_utils import (
    build_state_from_messages,
    resolve_embedder,
    retrieve_prompt_context,
)


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "is", "are",
    "was", "were", "be", "this", "that", "these", "those", "it", "its", "as", "at", "from",
    "what", "which", "who", "whom", "when", "where", "why", "how", "i", "you", "he", "she",
    "they", "we", "my", "your", "his", "her", "their", "our", "me", "him", "them",
}


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, dict):
        for key in ("text", "answer", "value", "span"):
            if key in value:
                return _as_str_list(value.get(key))
        return []
    if isinstance(value, (list, tuple)):
        results: list[str] = []
        for item in value:
            results.extend(_as_str_list(item))
        return [item for item in results if item]
    return [str(value).strip()]


def _extract_messages(example: dict) -> list[object]:
    for key in ("messages", "conversation", "dialogue", "chat", "history", "turns"):
        if key in example:
            value = example.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)
    for key in ("context", "story", "source", "document"):
        if key in example:
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
    return []


def _extract_query(example: dict) -> str:
    for key in ("question", "query", "prompt", "instruction", "ask", "q"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_answers(example: dict) -> list[str]:
    for key in (
        "answer",
        "answers",
        "gold",
        "label",
        "target",
        "response",
        "expected",
        "memory",
        "memory_key",
        "memory_value",
    ):
        if key in example:
            answers = _as_str_list(example.get(key))
            if answers:
                return answers
    return []


def _flatten_locomo_sessions(conversation: dict) -> list[dict[str, str]]:
    if not isinstance(conversation, dict):
        return []
    session_keys: list[tuple[int, str]] = []
    for key, value in conversation.items():
        match = re.match(r"^session_(\d+)$", str(key))
        if match and isinstance(value, list):
            session_keys.append((int(match.group(1)), str(key)))
    messages: list[dict[str, str]] = []
    for _idx, key in sorted(session_keys, key=lambda item: item[0]):
        turns = conversation.get(key, [])
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker") or turn.get("role") or turn.get("from") or "user")
            content = (
                turn.get("text")
                or turn.get("utterance")
                or turn.get("blip_caption")
                or ""
            )
            content = str(content).strip()
            if not content:
                continue
            messages.append({"speaker": speaker, "text": content})
    return messages


def _load_locomo10_json(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit("locomo10.json must contain a top-level list of conversations")
    examples: list[dict[str, object]] = []
    for sample in data:
        if not isinstance(sample, dict):
            continue
        conversation = sample.get("conversation", {})
        messages = _flatten_locomo_sessions(conversation)
        qas = sample.get("qa") or []
        if not isinstance(qas, list):
            continue
        for qa in qas:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            answers = _as_str_list(
                qa.get("answer") or qa.get("answers") or qa.get("adversarial_answer")
            )
            examples.append(
                {
                    "messages": messages,
                    "question": question,
                    "answers": answers,
                    "sample_id": sample.get("sample_id"),
                }
            )
    return examples


def _proxy_overlap(question: str, context: str) -> float:
    if not question or not context:
        return 0.0
    tokens = [tok.lower() for tok in question.replace("?", " ").split()]
    tokens = [tok.strip(".,:;!()[]{}\"'") for tok in tokens]
    tokens = [tok for tok in tokens if tok and tok not in STOPWORDS and len(tok) > 2]
    if not tokens:
        return 0.0
    context_lower = context.lower()
    hits = sum(1 for tok in tokens if tok in context_lower)
    return hits / max(1, len(tokens))


def _pick_indices(total: int, max_examples: int | None, seed: int) -> list[int]:
    if max_examples is None or max_examples >= total:
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(range(total), k=max_examples)


def run(args: argparse.Namespace) -> dict[str, object]:
    examples: list[dict[str, object]] = []
    dataset_name = args.dataset if not args.path else None
    if args.path:
        if not Path(args.path).is_file():
            raise SystemExit(f"--path does not exist: {args.path}")
        examples = _load_locomo10_json(args.path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("datasets is required; install with: pip install -e .[eval]") from exc

        dataset_kwargs = {}
        if args.config:
            dataset_kwargs["name"] = args.config

        if not args.dataset:
            raise SystemExit("Provide --path or --dataset for HuggingFace loading.")

        dataset = load_dataset(args.dataset, **dataset_kwargs)
        if args.split not in dataset:
            raise SystemExit(f"split '{args.split}' not in dataset; splits={list(dataset.keys())}")

        split = dataset[args.split]
        for item in split:
            if isinstance(item, dict):
                examples.append(item)

    indices = _pick_indices(len(examples), args.max_examples, args.seed)
    embedder = resolve_embedder()

    hits = 0
    proxy_hits = 0
    proxy_examples = 0
    total = 0
    retrieved_chars: list[int] = []
    prompt_chars: list[int] = []
    sample_keys: list[str] = []

    for idx in indices:
        example = examples[int(idx)]
        if not sample_keys and isinstance(example, dict):
            sample_keys = sorted(example.keys())

        if not isinstance(example, dict):
            continue

        messages = _extract_messages(example)
        query = _extract_query(example)
        answers = _extract_answers(example)

        if not messages or not query:
            continue

        graph, index = build_state_from_messages(messages, embedder)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = f"{tmpdir}/state.json"
            save_state(
                graph=graph,
                index=index,
                path=state_path,
                embedder_name=embedder.name,
                embedder_dim=embedder.dim,
                meta={"source": "locomo"},
            )
            prompt_context, stats = retrieve_prompt_context(
                graph,
                index,
                embedder,
                query,
                top_k=args.top_k,
                max_prompt_context_chars=args.max_prompt_context_chars,
            )
        retrieved_chars.append(stats.get("retrieved_raw_chars", 0))
        prompt_chars.append(stats.get("prompt_context_len", len(prompt_context)))

        total += 1
        context_lower = prompt_context.lower()
        if answers:
            if any(answer.lower() in context_lower for answer in answers if answer):
                hits += 1
        else:
            proxy_examples += 1
            overlap = _proxy_overlap(query, prompt_context)
            if overlap >= args.proxy_overlap_threshold:
                proxy_hits += 1

    recall = hits / total if total else 0.0
    proxy_recall = proxy_hits / proxy_examples if proxy_examples else 0.0

    summary = {
        "dataset": dataset_name,
        "path": args.path,
        "config": args.config,
        "split": args.split,
        "examples_evaluated": total,
        "hits": hits,
        "recall_at_k": recall,
        "recall_at_k_proxy": recall,
        "avg_retrieved_chars": sum(retrieved_chars) / len(retrieved_chars) if retrieved_chars else 0.0,
        "avg_prompt_context_chars": sum(prompt_chars) / len(prompt_chars) if prompt_chars else 0.0,
        "proxy_examples": proxy_examples,
        "proxy_overlap_threshold": args.proxy_overlap_threshold,
        "proxy_recall": proxy_recall,
        "embedder_mode": embedder.mode,
        "embedder_name": embedder.name,
        "sample_keys": sample_keys,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo memory eval against OpenClawBrain retrieval.")
    parser.add_argument(
        "--path",
        default="",
        help="Path to locomo10.json (preferred; uses HuggingFace only when omitted)",
    )
    parser.add_argument(
        "--dataset",
        default="desire2020/locomo-serialized",
        help="HuggingFace dataset name (fallback when --path is omitted)",
    )
    parser.add_argument("--config", default=None, help="Optional dataset config name")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling")
    parser.add_argument("--top-k", type=int, default=6, help="Top-K nodes to retrieve")
    parser.add_argument(
        "--max-prompt-context-chars",
        type=int,
        default=20000,
        help="Fairness cap for brain context size",
    )
    parser.add_argument(
        "--proxy-overlap-threshold",
        type=float,
        default=0.3,
        help="Overlap threshold for proxy metric when no gold answer spans",
    )
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
