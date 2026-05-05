# OpenClawBrain

**Evidence, not vibes, for agent memory.**

OpenClawBrain is local, accountable memory for [OpenClaw](https://docs.openclaw.ai) agents. It remembers durable corrections, preferences, workflows, and context, then retrieves only the small slice that matters for the current turn.

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**

![OpenClawBrain memory graph showing LLM update pulses, SQLite memory, learned route_fn paths, and injected context.](docs/assets/openclawbrain-memory-graph.jpg)

Core capture/store/retrieve/inject works today. Route-learning quality, status polish, and long-term organic runtime still need more mileage.

## Install

Requires OpenClaw `2026.5.2` or later.

```bash
openclaw plugins install clawhub:openclawbrain@0.2.16
openclaw plugins enable openclawbrain
openclaw gateway restart
```

If ClawHub is rate-limited or package metadata is still propagating, install the release archive instead:

```bash
curl -L -o /tmp/openclawbrain-0.2.16.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.16/openclawbrain-0.2.16.tgz
openclaw plugins install /tmp/openclawbrain-0.2.16.tgz --force
openclaw plugins enable openclawbrain
openclaw gateway restart
```

## Verify it is live

Use runtime inspection, not just package metadata.

```bash
openclaw plugins inspect openclawbrain --runtime
openclaw doctor
# /plugins/openclawbrain/proof?limit=10
# /plugins/openclawbrain/search?query=pnpm&limit=10
```

You want to see:

- plugin loaded
- hooks and routes registered
- SQLite + FTS healthy
- no `No active memory plugin` warning from `openclaw doctor`

HTTP plugin routes are authenticated on normal OpenClaw installs. Use the authenticated dashboard/client, or pass your gateway auth header when using curl.

## Five-minute proof example

Teach one small repo rule:

```text
Use pnpm instead of npm in this repo.
```

Then check whether memory captured and can retrieve it:

```bash
openclaw plugins inspect openclawbrain --runtime
openclaw doctor
# /plugins/openclawbrain/proof?limit=10
# /plugins/openclawbrain/search?query=pnpm&limit=10
# /plugins/openclawbrain/graph?agentId=main&limit=10
# /plugins/openclawbrain/explain-last
```

A later test/build turn should receive a small bounded context block, not a transcript dump:

```xml
<openclawbrain_context>
Relevant memory:
- Must follow: Use pnpm instead of npm in this repo.
</openclawbrain_context>
```

That is the product claim: capture a durable correction, retrieve it later, inject it inside budget, and leave proof behind.

## Configuration

Recommended default mode is `balanced`.

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"balanced"' --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowPromptContext true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowConversationAccess true --strict-json
openclaw config set plugins.entries.openclawbrain.config.hooks.allowToolObservation true --strict-json
openclaw config validate
openclaw gateway restart
```

### Local learning model

OpenClawBrain uses the local LLM path for semantic updates: feedback distillation, route examples, and learning. The model proposes structured JSON. Code validates, redacts, scopes, dedupes, thresholds, and writes.

Default local path is Ollama through an OpenAI-compatible endpoint:

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

If local model calls are unavailable, the main agent keeps working. Known memories can still be searched and injected; background learning gets quieter or retries later.

### Multiple agents

If you run multiple named agents/profiles, scope OpenClawBrain to all of them so each gets its own local graph:

```bash
openclaw config set plugins.entries.openclawbrain.config.scopes.agents '["main","pelican","bountiful"]' --strict-json
openclaw gateway restart
```

Use your real agent ids. Single-agent installs can skip this.

## Inspectable endpoints

| Endpoint | What it shows |
|---|---|
| `/plugins/openclawbrain/status` | whether the plugin is enabled, loaded, and how the runtime is behaving |
| `/plugins/openclawbrain/doctor` | SQLite + FTS health under the current Node runtime |
| `/plugins/openclawbrain/proof?limit=20` | recent redacted proof, route, and injection events |
| `/plugins/openclawbrain/search?query=...&limit=20` | local memory search |
| `/plugins/openclawbrain/graph?limit=50` | redacted memory nodes and memory edges |
| `/plugins/openclawbrain/learn?limit=50` | route examples and current learning state |
| `/plugins/openclawbrain/audit?limit=20` | recent capture/store/reject decisions and rejection distribution |
| `/plugins/openclawbrain/explain-last` | compact postmortem for the latest memory decision |

## How it works

OpenClawBrain sits beside the normal OpenClaw run. It does not replace the main model; it gives the model better working memory.

```text
before_prompt_build
  → redact current turn
  → route_fn decides whether memory should participate
  → SQLite FTS + graph search finds candidates
  → context selector chooses a small set
  → inject bounded prompt context

agent_end / after_tool_call
  → distill durable feedback
  → validate and store memory updates
  → update outcomes and learned route pointers
```

The graph stores scoped memory nodes and edges: corrections, preferences, workflows, context, route examples, outcomes, and superseded facts.

## Privacy and safety

- Local learning defaults to the local Ollama path
- Raw transcript upload is hard-disabled
- Redaction happens before storage and before model use
- The model does not write directly to memory
- SQLite stores the graph and evidence locally
- Plugin failure should not block the main agent

## Links

- [Install](https://openclawbrain.ai/install/)
- [Proof](https://openclawbrain.ai/proof/)
- [How it works](https://openclawbrain.ai/how-it-works/)
- [Architecture](docs/ARCHITECTURE.md)
- [Vision](VISION.md)
- [Final plan](FINAL_PLAN.md)
- [Memory graph image](docs/assets/openclawbrain-memory-graph.jpg)
- [Getting started](docs/GETTING_STARTED.md)
- [Copy-paste install note](docs/FRIEND_INSTALL.md)

## License

MIT
