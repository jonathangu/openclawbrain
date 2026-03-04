from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
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
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required; install with: pip install -e .[eval]") from exc

    dataset_kwargs = {}
    if args.config:
        dataset_kwargs["name"] = args.config

    dataset = load_dataset(args.dataset, **dataset_kwargs)
    if args.split not in dataset:
        raise SystemExit(f"split '{args.split}' not in dataset; splits={list(dataset.keys())}")

    split = dataset[args.split]
    indices = _pick_indices(len(split), args.max_examples, args.seed)
    embedder = resolve_embedder()

    hits = 0
    proxy_hits = 0
    proxy_examples = 0
    total = 0
    retrieved_chars: list[int] = []
    prompt_chars: list[int] = []
    sample_keys: list[str] = []

    for idx in indices:
        example = split[int(idx)]
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
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "examples_evaluated": total,
        "hits": hits,
        "recall_at_k": recall,
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
    parser.add_argument("--dataset", default="locomo", help="HuggingFace dataset name (default: locomo)")
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
