"""Ollama chat callbacks for optional LLM-assisted workflows."""

from __future__ import annotations

import json
import os
import sys
from urllib import request


_DEFAULT_OLLAMA_TIMEOUT_SECONDS = 600
_DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
_DEFAULT_OLLAMA_THINK = False


def _resolve_ollama_host() -> str:
    host = (
        os.environ.get("OPENCLAWBRAIN_OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST")
        or _DEFAULT_OLLAMA_HOST
    )
    host = host.strip() if isinstance(host, str) else _DEFAULT_OLLAMA_HOST
    if not host:
        host = _DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _resolve_ollama_model(model: str | None) -> str:
    if isinstance(model, str) and model.strip():
        return model
    env_model = os.environ.get("OPENCLAWBRAIN_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL")
    if isinstance(env_model, str) and env_model.strip():
        return env_model.strip()
    return _DEFAULT_OLLAMA_MODEL


def _resolve_ollama_timeout_seconds() -> int:
    env_timeout = (
        os.environ.get("OPENCLAWBRAIN_OLLAMA_TIMEOUT_SECONDS")
        or os.environ.get("OLLAMA_TIMEOUT_SECONDS")
    )
    if not isinstance(env_timeout, str):
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS
    env_timeout = env_timeout.strip()
    if not env_timeout:
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS
    try:
        timeout = int(env_timeout)
    except ValueError:
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS
    if timeout < 1:
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS
    return timeout


def _resolve_ollama_think() -> bool:
    v = os.environ.get("OPENCLAWBRAIN_OLLAMA_THINK")
    if v is None or v == "":
        return _DEFAULT_OLLAMA_THINK
    v = str(v).strip().lower()
    if v in {"0", "false", "no", "n", "off"}:
        return False
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    return _DEFAULT_OLLAMA_THINK


def _ollama_chat(system: str, user: str, *, model: str) -> str:
    host = _resolve_ollama_host()
    timeout_seconds = _resolve_ollama_timeout_seconds()
    payload = {
        "model": model,
        "stream": False,
        "think": _resolve_ollama_think(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{host}/api/chat",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        raw_bytes = resp.read()
    raw_text = raw_bytes.decode("utf-8") if raw_bytes else ""
    if not raw_text:
        return ""
    response = json.loads(raw_text)
    message = response.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return content


def ollama_llm_fn(system: str, user: str, *, model: str | None = None) -> str:
    """Run a single Ollama chat request."""
    resolved_model = _resolve_ollama_model(model)
    return _ollama_chat(system, user, model=resolved_model)


def ollama_llm_batch_fn(requests: list[dict], *, max_workers: int = 8) -> list[dict]:
    """Run Ollama chat requests concurrently.

    Args:
        requests: list of {id, system, user}
        max_workers: thread pool size
    """
    if not requests:
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(requests)
    results: list[dict] = []

    def _call(req: dict) -> dict:
        system = str(req.get("system", ""))
        user = str(req.get("user", ""))
        model = _resolve_ollama_model(req.get("model"))
        request_id = req.get("id")
        try:
            response = _ollama_chat(system, user, model=model)
        except Exception as exc:  # noqa: BLE001
            return {"id": request_id, "response": "", "error": str(exc)}
        return {"id": request_id, "response": response}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, req) for req in requests]
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"LLM batch progress: processing {completed}/{total}", file=sys.stderr)
            results.append(fut.result())

    return results
