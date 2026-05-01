# OpenClawBrain

**OpenClawBrain helps OpenClaw agents stop repeating themselves.**

It is a native [OpenClaw](https://docs.openclaw.ai) plugin that remembers useful corrections, preferences, and past wins, then brings back only the small piece that matters for the current turn.

In plain English: if you already taught your agent "use pnpm here," "run the tests first," or "this is how I want this project handled," OpenClawBrain helps that lesson stick.

## What it does

- **Remembers the useful stuff.** Corrections, preferences, and prior work can become local memory instead of disappearing at the end of the session.
- **Keeps prompts small.** It does not dump your whole history into every turn. It retrieves a few likely memories and injects only a bounded slice.
- **Stays local-first.** Memory is stored in local SQLite, with redaction and proof surfaces built in.
- **Lets you inspect it.** You can check status, run health checks, inspect proof events, search memory, and view the graph.
- **Learns on the standard local path.** OpenClawBrain points at local Ollama by default, so corrections can turn into structured memory automatically. If you deliberately turn that path off, the plugin is still live and inspectable, but automatic learning is not active.

## Why it exists

Most agents are smart but forgetful. They can do good work inside one turn, then make the same mistake again tomorrow.

The usual fix is to keep stuffing more text into the prompt. That works badly. Prompts get bloated, latency goes up, and the agent still lacks a real memory system.

OpenClawBrain takes a different approach:

1. keep memory local
2. retrieve candidates fast
3. inject only a small useful slice
4. show proof instead of asking for blind trust

## Current release

- **Current package release:** `0.2.10`
- **Recommended mode:** `balanced`
- **Requires:** OpenClaw `2026.4.29` or later

## Install

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

## Default local setup

The default OpenClawBrain setup is already aimed at the full local path: balanced mode, conversation/tool hooks on, and local Ollama on `127.0.0.1`.

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
openclaw config validate
openclaw gateway restart
```

## Default local learning models

The default local learning path uses a local OpenAI-compatible endpoint such as local Ollama.

```bash
ollama list

openclaw config set plugins.entries.openclawbrain.config.llm '{
  "enabled": true,
  "baseUrl": "http://127.0.0.1:11434/v1",
  "routeModel": "qwen2.5:32b-instruct",
  "plannerModel": "qwen2.5:32b-instruct",
  "feedbackModel": "qwen2.5:32b-instruct",
  "learningModel": "qwen2.5:32b-instruct"
}' --strict-json
openclaw config validate
openclaw gateway restart
```

If you deliberately disable this path, OpenClawBrain still runs its local memory, search, proof, and health surfaces. It just will not auto-distill fresh corrections.

## Check that it is live

```bash
openclaw plugins inspect openclawbrain --json
curl http://127.0.0.1:18789/plugins/openclawbrain/status
curl http://127.0.0.1:18789/plugins/openclawbrain/doctor
curl http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=10
curl 'http://127.0.0.1:18789/plugins/openclawbrain/search?query=pnpm&limit=10'
```

## What you can inspect

| Endpoint | What it shows |
|---|---|
| `/plugins/openclawbrain/status` | whether the plugin is enabled, loaded, and how the runtime is behaving |
| `/plugins/openclawbrain/doctor` | SQLite + FTS health under the current Node runtime |
| `/plugins/openclawbrain/proof?limit=20` | recent redacted proof and route events |
| `/plugins/openclawbrain/graph?limit=50` | redacted memory nodes and edges |
| `/plugins/openclawbrain/learn?limit=50` | route examples and current learning state |
| `/plugins/openclawbrain/search?query=...&limit=20` | local memory search |

## Privacy and safety

- Local learning defaults to on when local Ollama is available
- Raw transcript upload is hard-disabled
- Redaction happens before storage and before model use
- Plugin failure does not block the main agent
- Local-first by default

## More

- [Getting started](docs/GETTING_STARTED.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Vision](VISION.md)
- [Final plan](FINAL_PLAN.md)

## License

MIT
