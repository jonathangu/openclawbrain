# OpenClaw Integration (OpenClawBrain)

OpenClawBrain is built to be the long-term memory layer for **OpenClaw agents**.
Canonical docs and examples: https://openclawbrain.ai
Primary operator runbook: [docs/operator-guide.md](operator-guide.md)
Operator recipes (cutover, parallel replay, prompt caching, media memory): [docs/ops-recipes.md](ops-recipes.md)
New-agent canonical SOP (workspace + dedicated brain + launchd + routing): [docs/new-agent-sop.md](new-agent-sop.md)
Packaged adapter CLIs (no repo clone required): `python3 -m openclawbrain.openclaw_adapter.query_brain ...`, `python3 -m openclawbrain.openclaw_adapter.capture_feedback ...`, `python3 -m openclawbrain.openclaw_adapter.learn_by_chat_id ...`, and `python3 -m openclawbrain.openclaw_adapter.learn_correction ...`
OpenClaw hook pack (recommended): [docs/openclawbrain-openclaw-hooks.md](openclawbrain-openclaw-hooks.md)
Glossary: [docs/glossary.md](glossary.md)
State directory lifecycle: [docs/state-directory.md](state-directory.md)
Guardrails and rollback: [docs/guardrails-and-rollback.md](guardrails-and-rollback.md)
Benchmarks: [docs/benchmarks.md](benchmarks.md)
End-to-end trace: [docs/end-to-end-trace.md](end-to-end-trace.md)

If you’re already running OpenClaw, this guide shows the fastest path to:

- Build a brain (`state.json`) from your OpenClaw workspace
- Run the **persistent daemon service** (`openclawbrain serve`) so queries stay fast
- Enable **brain-first** OpenClaw integration (no AGENTS wiring required)
- Verify it’s working, and roll back cleanly

---

## Brain-first OpenClaw integration (recommended)

Brain-first mode injects OpenClawBrain context automatically at the hook layer. This keeps your existing prompts intact and avoids manual AGENTS wiring.

**BEFORE (no hook):**
- OpenClaw uses only its base prompt (`AGENTS.md`, `SOUL.md`, etc.).
- You must manually run `query_brain` or wire commands into AGENTS.

**AFTER (hook enabled):**
- Each non-slash user message gets a `[BRAIN_CONTEXT ...]` block prepended to `bodyForAgent`.
- Retrieval is automatic and fail-open (if anything breaks, OpenClaw proceeds unchanged).
- Corrections and recall language get a deeper context budget.

### 5-minute quickstart (copy/paste)

```bash
openclawbrain openclaw install --agent main --yes
```

Equivalent manual steps:

```bash
openclawbrain serve install --state ~/.openclawbrain/main/state.json
openclaw hooks install /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector
openclaw hooks enable openclawbrain-context-injector
openclawbrain loop install --state ~/.openclawbrain/main/state.json
openclaw gateway restart
```

Use `/path/to` as a placeholder for your local `openclawbrain` repo path.
If the hook pack is not bundled with your install, pass `--hooks-path /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector`.

### Step-by-step install (managed)

1. Build a brain if you do not already have one:

```bash
openclawbrain init --workspace ~/.openclaw/workspace --output ~/.openclawbrain/main
```

2. Start the daemon (hot socket):

```bash
openclawbrain serve install --state ~/.openclawbrain/main/state.json
```

macOS launchd is the default. For Linux, use `openclawbrain serve --systemd` to print a unit template.
For a foreground daemon, run `openclawbrain serve start --state ~/.openclawbrain/main/state.json`.

3. Install + enable the hook:

```bash
openclaw hooks install /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector
openclaw hooks enable openclawbrain-context-injector
```

4. Install the always-learning loop (recommended default):

```bash
openclawbrain loop install --state ~/.openclawbrain/main/state.json
```

On Linux/systemd hosts, run `openclawbrain loop --systemd` to print a unit template.
When the serve daemon is managed by launchd on macOS, the loop will briefly pause it to acquire the state lock while applying updates (short downtime). Disable with `--no-pause-serve-when-locked` if needed.

Default schedule:
- Hourly cheap job: edges-only replay + harvest (LLM-free)
- Nightly heavier job: full replay (fast-learning + harvest) + maintenance

5. Restart the gateway so hooks load:

```bash
openclaw gateway restart
```

### Verification (expect “Ready”)

```bash
openclaw hooks check
openclaw hooks list
openclaw hooks info openclawbrain-context-injector
```

Convenience status check:

```bash
openclawbrain openclaw status --agent main
```

**“Ready”** means the hook is discovered, eligible, and enabled for the gateway (no missing requirements like `python3` or `workspace.dir`).

### Rollback (clean and fast)

```bash
openclawbrain openclaw uninstall --agent main --yes
```

Manual rollback steps:

```bash
openclaw hooks disable openclawbrain-context-injector
openclaw gateway restart
```

If you want to stop the daemon too:

```bash
openclawbrain serve stop --state ~/.openclawbrain/main/state.json
```

If you want to stop the loop too:

```bash
openclawbrain loop uninstall --state ~/.openclawbrain/main/state.json
```

### Why the gateway restart is required

OpenClaw loads hook manifests on gateway start. Restarting ensures the new hook is discovered and activated. This is a normal, quick reload and does not change your agent data.

### Budgets and “remember” behavior

- Default context budget: **20,000** chars.
- Recall/correction language (for example: “remember”, “last time”, “earlier”, “correction”, “audit”) raises the budget to **80,000** chars.

### Context budget slices (why tool caps matter)

Context budgets are shared across multiple prompt slices, so comparisons must cap tool outputs as well as brain context.

| Slice | What it includes | Why it matters |
| --- | --- | --- |
| System/boot | AGENTS/SOUL/USER/TOOLS/bootstrap files loaded by OpenClaw | These are fixed and already present; avoid re-injecting them. |
| Transcript | Recent user/assistant turns from OpenClaw | Grows quickly; can crowd out retrieval if left unchecked. |
| Tool outputs | `tool_result` payloads, logs, JSON blobs | Large tool payloads can dwarf brain context if not capped. |
| Brain context | `[BRAIN_CONTEXT ...]` from OpenClawBrain | This is the budget you tune for recall depth. |

If you raise brain budgets without capping tool outputs, the prompt can still be dominated by tools. Always keep `--tool-result-max-chars` (or equivalent OpenClaw limits) aligned with your context target.

### Security defaults and guardrails

- **Exclude paths**: use `--exclude-paths` with manual wiring (or a custom hook) to drop sensitive directories from retrieval.
- **Redaction**: prompt context is redacted for common secret patterns before injection.
- **Data-only delimiter**: injected context is marked as `[BRAIN_CONTEXT ...]` (context, not instructions).
- **Fail-open**: if the hook can’t run or times out, OpenClaw continues with the original prompt.

Troubleshooting guide: [docs/openclaw-integration-troubleshooting.md](openclaw-integration-troubleshooting.md)
Guardrails and rollback: [docs/guardrails-and-rollback.md](guardrails-and-rollback.md)
Benchmarks: [docs/benchmarks.md](benchmarks.md)

---

## What OpenClawBrain does (and why you want it)

OpenClawBrain is a Python memory graph library that turns your workspace into a **learned retrieval policy**.
Instead of “top-k similarity every time”, it learns which context routes actually helped.

**What changes in practice:**

- Your agent stops resurfacing the same wrong chunks over and over.
- Corrections create **inhibitory edges** that actively suppress bad retrieval routes.
- You can add knowledge without rebuilding the whole index (`inject --type TEACHING`).
- With the socket server, the brain stays **hot in memory**, avoiding per-call reload overhead.

**Always-on learning (recommended default experience):**

- Clear user **corrections** and **teachings** should be captured automatically in the same turn via `capture_feedback`.
- Operators should *not* need to say “inject teaching” / “log this” / special phrasing.
- Use `--message-id` or `--dedup-key` so retries cannot double-inject.

OpenClawBrain stores everything in a single portable file:

- `state.json` — graph + embeddings + metadata

---

## Prerequisites

- Python **3.10+**
- `pip install openclawbrain`
- Daemon query embedding mode defaults to `--embed-model auto`:
  - `local:*` states use local query embeddings (fastembed, offline by default).
  - `hash-v1` states use hash query embeddings (offline, no OpenAI call).
  - OpenAI states do not auto-call OpenAI; use `--embed-model openai:<model>` explicitly when needed.
  - Legacy hash-v1 states can force hash query embeddings with `--embed-model hash` (legacy only).
  - Force local mode: `--embed-model local`.
  - Force explicit OpenAI model: `--embed-model openai:text-embedding-3-small`.

Install:

```bash
pip install openclawbrain
```

Sanity check:

```bash
openclawbrain --help
openclawbrain info --help
```

---

## Secrets registry

Use pointer-only secret handling. Keep capability names, required key names, and local secret pointers, but never secret values.

- Canonical policy: [docs/secrets-and-capabilities.md](secrets-and-capabilities.md)
- Harvest pointers: `python3 -m openclawbrain.ops.harvest_secret_pointers --workspace ~/.openclaw/workspace`
- Leak audit (path + line only): `python3 -m openclawbrain.ops.audit_secret_leaks --workspace ~/.openclaw/workspace --strict`

---

## Architecture (how it fits into OpenClaw)

OpenClaw today is file-and-instructions driven: the agent reads `AGENTS.md`, then runs whatever you tell it to.
OpenClawBrain plugs into that contract.

### Data flow

```text
User message
   ↓
OpenClaw agent
   ↓ (reads AGENTS.md)
OpenClawBrain query (daemon)
   ↓
Prompt appendix (`[BRAIN_CONTEXT]`) + internal fired-node tracking
   ↓
Agent response
   ↓
Outcome (+1 good / -1 bad / correction)
   ↓
OpenClawBrain learn / inject
```

### Why the daemon matters

Without the socket service, every query path tends to do:

- start Python
- load `state.json`
- initialize index
- run query
- exit

With `openclawbrain serve`, you pay the load cost once and queries become:

- NDJSON request → response

This is the “production shape” for OpenClaw integration.

Adapter CLIs (`python3 -m openclawbrain.openclaw_adapter.query_brain` and `python3 -m openclawbrain.openclaw_adapter.capture_feedback`) now auto-detect the daemon socket:

- If `~/.openclawbrain/<agent>/daemon.sock` exists, they use the in-memory socket transport (fast path).
- If the socket is missing, they fall back to loading `state.json` from disk directly (slower but still works).
- Both scripts also accept `--socket` when you want to force an explicit socket path.

---

## Step 1 — Build your first brain (`openclawbrain init`)

Pick where you want brains to live. Recommended layout:

```text
~/.openclawbrain/
  main/
    state.json
    journal.jsonl
    fired_log.jsonl
    injected_corrections.jsonl
```

Build from an OpenClaw workspace:

```bash
# Example workspace path (adjust)
WORKSPACE=~/.openclaw/workspace

# Where the brain lives
BRAIN_DIR=~/.openclawbrain/main

mkdir -p "$BRAIN_DIR"

# Create/overwrite state.json inside the output directory
# By default, init uses embedder auto -> local fastembed.
openclawbrain init --workspace "$WORKSPACE" --output "$BRAIN_DIR"

# Quick health signal
openclawbrain doctor --state "$BRAIN_DIR/state.json"
openclawbrain health  --state "$BRAIN_DIR/state.json"
```

Notes:

- First init is the “embed everything once” step.
- Subsequent edits should use `openclawbrain sync` (incremental) rather than full rebuild.

---

## Step 2 — Install the recommended OpenClaw hook (if not already done)

The recommended path is the OpenClaw hook pack:

- `integrations/openclaw/hooks/openclawbrain-context-injector/`
- [docs/openclawbrain-openclaw-hooks.md](openclawbrain-openclaw-hooks.md)

Install and enable:

```bash
openclaw hooks install /path/to/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector
openclaw hooks enable openclawbrain-context-injector
openclaw gateway restart
```

Note: `--link` is dev-only. If you use it, set `hooks.internal.load.extraDirs` to the parent hooks directory (the directory that contains `openclawbrain-context-injector/`), then restart the gateway.

Troubleshooting:

- If `openclaw hooks info openclawbrain-context-injector` is not found, run `openclaw hooks list` and restart the gateway. Ensure the hook exists under `~/.openclaw/hooks/`.

The hook behavior is:

- Keep OpenClaw working as-is (fail-open).
- Append a retrieved `[BRAIN_CONTEXT ...]` block to `bodyForAgent` for normal messages.
- Use 20,000-char budget by default, 80,000-char budget for recall/correction language (e.g., “remember”, “last time”, “correction”).
- Keep `--exclude-bootstrap` and `--redact` enabled.
- Skip slash commands.

What changes when enabled:

- OpenClaw still reads all its normal `AGENTS.md` files.
- The hook adds only data context to prompts (it does not replace or reorder existing instructions).
- Retrieval is automatic each qualifying message (no manual copy/paste per-agent edits required).
- Failures are non-blocking: if query context cannot be loaded within 2s, OpenClaw proceeds with the original message.

## Step 2 fallback — manual AGENTS.md integration (legacy/optional)

If you do not want hook-based integration, you can keep the legacy manual path in `AGENTS.md`.
This still works and remains fully supported, but is not required for brain-first mode:

```bash
python3 -m openclawbrain.openclaw_adapter.query_brain ~/.openclawbrain/AGENT/state.json '<summary of user message>' --chat-id '<chat_id from inbound metadata>' --format prompt --exclude-bootstrap --max-prompt-context-chars 20000
```

For same-turn learning:

```bash
python3 -m openclawbrain.openclaw_adapter.capture_feedback \
  --state ~/.openclawbrain/AGENT/state.json \
  --chat-id '<chat_id>' \
  --kind CORRECTION \
  --content "The correction text here" \
  --lookback 1 \
  --message-id '<stable-message-id>' \
  --json
```

### Always-on self-learning (default)

Use this as the default operator policy; no special user phrasing like "log this" should be required.

- If the user clearly corrects the agent, run `capture_feedback --kind CORRECTION --chat-id '<chat_id>' --lookback 1`.
- If the user teaches a durable rule/fact, run `capture_feedback --kind TEACHING` (optionally with `--outcome`).
- If intent is ambiguous (correction vs preference vs new teaching), ask one clarifying question before writing memory.
- Never log secrets, tokens, passwords, private keys, or other sensitive values.
- Always pass a stable `--dedup-key` (or `--message-id`) when available.

This remains compatible with tight prompts:
- Keep retrieval on `query_brain --format prompt` so prompt payload contains only `[BRAIN_CONTEXT]`.
- Learning/injection uses `capture_feedback` keyed by `chat_id`, with optional `--outcome`.
- Do not pass `fired_nodes` in prompt/tool payloads; the brain tracks them internally by `chat_id`.

### Prompt-context duplication control (recommended)

OpenClaw already loads bootstrap files (`AGENTS.md`, `SOUL.md`, `USER.md`, `MEMORY.md`, `active-tasks.md`) into the base prompt. If you also include them again from brain retrieval, token usage grows quickly with little value.

Use prompt format and exclusions to keep context “tight and right”:

- Prefer `--format prompt` so only `[BRAIN_CONTEXT]` is appended to the model prompt.
- Keep `--exclude-bootstrap` enabled (default in the adapter).
- Start with `--max-prompt-context-chars 20000` when matching hook defaults; allow up to `80000` for deep recall only when needed.
- Use `--exclude-recent-memory ...` only for explicit daily notes already injected into the same OpenClaw turn.

This aligns retrieval output with OpenClawBrain’s context-efficiency goal: preserve high-value retrieved nodes while minimizing repeated bootstrap content.

---

## Step 3 — Run the daemon in production (launchd + systemd)

### Option A: run it manually (smoke test)

```bash
openclawbrain serve start --state ~/.openclawbrain/main/state.json
```

The daemon speaks NDJSON over `stdin`/`stdout`.

- Protocol: each line is one JSON request with `id`, `method`, and `params`.
- Response: one JSON object with matching `id` and either `result` or `error`.
- Start-up cost is paid once (state loaded at process start), so steady-state query/learn calls avoid repeated reload overhead.

Supported methods: `query`, `learn`, `last_fired`, `learn_by_chat_id`, `capture_feedback`, `maintain`, `health`, `info`, `save`, `reload`, `shutdown`, `inject`, `correction`.

Daemon embed-model default and overrides:
- Default is `--embed-model auto`.
- `local:*` state metadata => local query embeddings.
- `hash-v1` state metadata => hash query embeddings (offline).
- OpenAI state metadata => no OpenAI call in auto; use `--embed-model openai:<model>` explicitly.
- Force modes with `--embed-model local`. Legacy hash-v1 states can force hash query embeddings with `--embed-model hash` (legacy only).

Example request and reply:

```bash
echo '{"id":"req-1","method":"query","params":{"query":"how to deploy","top_k":4}}' | openclawbrain daemon --state ~/.openclawbrain/main/state.json
```

```json
{"id":"req-1","result":{"fired_nodes":["a"],"context":"...","embed_query_ms":1.1,"traverse_ms":2.4,"total_ms":3.5}}
```

- `inject` and `correction` are now available and are the preferred path for same-turn updates.
- The daemon is still NDJSON over stdio internally, with production transport now provided by `openclawbrain serve`.

### Option B: launchd (macOS)

Use the first-class lifecycle helpers for most local installs:

```bash
openclawbrain serve install --state ~/.openclawbrain/main/state.json
```

To remove:

```bash
openclawbrain serve uninstall --state ~/.openclawbrain/main/state.json
```

You can inspect the generated plist in dry-run mode and apply custom options:

```bash
openclawbrain serve install --state ~/.openclawbrain/main/state.json --dry-run --label com.openclawbrain.main
```

The command writes `~/Library/LaunchAgents/com.openclawbrain.main.plist` by default and runs:

- `openclawbrain serve start --state ... --socket-path ...` under the resolved Python executable path via `sys.executable`.

If you need a manually customized template, see the legacy plist approach below.

### Option C: legacy launchd plist template

Create `~/Library/LaunchAgents/com.openclawbrain.daemon.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.openclawbrain.daemon</string>

  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string>
    <string>openclawbrain</string>
    <string>serve</string>
    <string>start</string>
    <string>--state</string>
    <string>/Users/YOU/.openclawbrain/main/state.json</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/YOU/.openclawbrain/main/daemon.stdout.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/YOU/.openclawbrain/main/daemon.stderr.log</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>OPENAI_API_KEY</key>
    <string>YOUR_KEY_HERE</string>
  </dict>
</dict>
</plist>
```

Load it:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.openclawbrain.daemon.plist 2>/dev/null || true
launchctl load -w ~/Library/LaunchAgents/com.openclawbrain.daemon.plist
launchctl list | rg openclawbrain
```

Notes:

- Prefer injecting `OPENAI_API_KEY` via your own secure mechanism (1Password, Keychain, etc.).
- The daemon is a stdio worker. You typically run it behind a small supervisor that manages pipes.

### Option D: systemd (Linux)

Create `/etc/systemd/system/openclawbrain-daemon.service`:

```ini
[Unit]
Description=OpenClawBrain daemon (hot state.json worker)
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER
Environment=OPENAI_API_KEY=YOUR_KEY_HERE
ExecStart=/usr/bin/env openclawbrain serve start --state /home/YOUR_USER/.openclawbrain/main/state.json
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openclawbrain-daemon
sudo systemctl status openclawbrain-daemon --no-pager
```

---

## Step 6 — Wire up the learning loop (query → feedback → learn)

In OpenClaw terms, you want a stable ritual:

1. **Query** before answering
2. Extract `fired` node IDs
3. After answering, apply a reward:
   - `+1.0` if the context helped
   - `-1.0` if it hurt (or user corrected you)
4. When corrected, also inject the correction text as a durable inhibitory node

### Minimal CLI loop (no daemon)

```bash
# Query
Q=$(openclawbrain query "how do we deploy" --state ~/.openclawbrain/main/state.json --top 4 --json)

# Pull fired IDs out (jq recommended)
FIRED=$(echo "$Q" | jq -r '.fired | join(",")')

# Learn (reward)
openclawbrain learn --state ~/.openclawbrain/main/state.json --outcome 1.0 --fired-ids "$FIRED"
```

### Correction flow (the one you actually want)

OpenClawBrain ships an OpenClaw adapter that logs fired IDs per chat.

- Query: `query_brain.py --chat-id ...` writes `fired_log.jsonl`
- Correction: `correction` daemon method (`method:"correction"`) penalizes recent fired IDs *and* injects a correction node

That’s the ergonomic way to do “same-turn correction” inside OpenClaw.

If you use the hook pack, user messages starting with `Correction:`, `Fix:`, `Teaching:`, or `Note:` automatically call `capture_feedback` in a fail-open way (deduped by message-id when available).

For full-history rebuilds, replay your sessions. Default mode is `full` and the recommended operator experience (the "best brain") is the full, bells-and-whistles pipeline.

See `examples/ops/default_experience.sh` for the recommended sequence: local BGE-large reembed + `replay --mode full` (fast-learning + edge replay + harvest) + `maintain`. Optional async teacher traces + `train-route-model` are off by default and can be enabled after the brain is running.

Use `--mode edges-only` for cheap edge-only replay when you explicitly want a minimal, low-cost pass.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions
```

Single-writer reminder:
- Rebuild/replay and daemon learning are both writers.
- If lock checks fail on LIVE state, prefer rebuild-then-cutover from [docs/operator-guide.md](operator-guide.md).
- Expert override: `--force` or `OPENCLAWBRAIN_STATE_LOCK_FORCE=1` (only when no conflicting writer is active).

Media note for OpenClaw logs:

- User-uploaded image/audio is often stored as a stub like `[media attached: ...]`.
- The meaningful OCR/transcript text usually appears later as `toolResult`.
- OpenClawBrain replay can attach allowlisted `toolResult` text back onto the
  media-stub user query and also expose it to fast-learning extraction windows.

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --include-tool-results \
  --tool-result-allowlist image,openai-whisper,openai-whisper-api,openai-whisper-local,summarize \
  --tool-result-max-chars 20000
```

This does:

- replay query edges from session history (`--mode edges-only`)
- optional LLM transcript mining into `learning::` nodes (`--mode fast-learning`)
- optional full pass (`--mode full`) adding harvest tasks (`decay,scale,split,merge,prune,connect`)

For cheap edge-only replay (no LLM, no harvest), use `--mode edges-only`:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --mode edges-only
```

To enable decay during an edges-only replay, add `--decay-during-replay`:

```bash
openclawbrain replay \
  --state ~/.openclawbrain/main/state.json \
  --sessions /path/to/sessions \
  --mode edges-only \
  --decay-during-replay \
  --decay-interval 10
```

`--decay-interval N` (default 10) controls how many learning steps between each decay pass.

`learning_events.jsonl` is an append-only sidecar under the brain directory used by harvest:

`~/.openclawbrain/main/learning_events.jsonl`

---

## Step 5 — Maintenance cron (keep the graph healthy)

The fast loop changes weights and adds nodes.
The slow loop keeps the graph compact and sane.

### What to run

Recommended maintenance command:

```bash
openclawbrain maintain --state ~/.openclawbrain/main/state.json --tasks health,decay,prune,merge
```

The explicit slow-learning path is:

```bash
openclawbrain harvest \
  --state ~/.openclawbrain/main/state.json \
  --events ~/.openclawbrain/main/learning_events.jsonl \
  --tasks split,merge,prune,connect,scale
```

Start conservative:

- First week: run `--tasks health,decay` only
- Then enable `prune,merge` when you’re comfortable with the behavior

### cron example

```cron
# Every 30 minutes: small hygiene pass
*/30 * * * * /usr/bin/env openclawbrain maintain --state ~/.openclawbrain/main/state.json --tasks health,decay,prune,merge >> ~/.openclawbrain/main/maintenance.log 2>&1

# Weekly: compact old daily notes into the brain (optional)
0 4 * * 0 /usr/bin/env openclawbrain compact --state ~/.openclawbrain/main/state.json --memory-dir ~/.openclaw/memory >> ~/.openclawbrain/main/compact.log 2>&1
```

---

## Troubleshooting

### “`openclawbrain: command not found`”

You installed into a different Python environment.

- Verify: `which openclawbrain`
- Fix: install into the same interpreter your agent uses:

```bash
python3 -m pip install --upgrade openclawbrain
python3 -m pip show openclawbrain
```

### “Embedder mismatch / dimension mismatch”

Dimension mismatch usually means query embedding dimensions do not match the vectors stored in `state.json` (often from forcing the wrong daemon embed model).

Fixes:
- Run daemon in default auto mode (`--embed-model auto`), which follows state metadata safely.
- Rebuild state if needed with the embedder you intend to keep.
- Only force `--embed-model local` or `--embed-model openai:<model>` when you intentionally want that behavior and dimensions are known to match. Legacy hash-v1 states can use `--embed-model hash` if required.

```bash
openclawbrain init --workspace ~/.openclaw/workspace --output ~/.openclawbrain/main
```

### “My queries are slow”

Common causes:

- You aren’t using the daemon (so `state.json` reload happens every call).
- You’re using legacy hash-v1 embeddings and still want production-grade retrieval (or `embed_query_ms` is high from OpenAI calls).

Fixes:

- Use the canonical brain-on command (`openclawbrain serve start --state ...`).
- Use OpenAI or local embeddings for production routing/scoring behavior; hash embeddings are legacy-only for existing hash-v1 states.

### “Daemon starts but OpenClaw can’t talk to it”

The daemon worker is an NDJSON stdio process, wrapped for production by `openclawbrain serve` (implemented in `socket_server`).
If your integration layer cannot reach the socket, fall back to disk-path operation.

Two practical options:

1. Keep using the adapter scripts; they will auto-detect and use the socket when available.
2. For custom integrations, call `openclawbrain.socket_client` against `daemon.sock`.

### “Corrections aren’t sticking”

- Ensure you pass `--chat-id` on every query so fired nodes are logged.
- On correction/teaching, run daemon `capture_feedback` in the same turn with `dedup_key`/`message_id`.

### “state.json is getting big”

- That’s normal with real embeddings.
- Run `maintain` (prune/merge) and optionally `compact` old notes.

---

## Native OpenClaw Tool (opencormorant fork)

If you run the **opencormorant** fork of OpenClaw (`github.com/jonathangu/opencormorant`), there is a built-in `openclawbrain` tool that agents can call directly — no shell exec needed.

### What it provides
The tool connects to the daemon Unix socket (`~/.openclawbrain/<agent>/daemon.sock`) and routes to the correct agent automatically (main/pelican/bountiful).

It supports **all OpenClawBrain daemon methods** (plus a safe generic passthrough):

| Action | Description |
|---|---|
| `query` | Retrieve context + prompt appendix (`prompt_context`) |
| `learn` | Apply numeric outcome to fired IDs (weight updates) |
| `last_fired` | Return recent fired node IDs for `chat_id` |
| `learn_by_chat_id` | Apply numeric outcome using recent fired nodes for `chat_id` |
| `capture_feedback` | Canonical real-time CORRECTION/TEACHING/DIRECTIVE with optional `outcome` + dedup |
| `inject` | Inject TEACHING/CORRECTION/DIRECTIVE nodes |
| `maintain` | Run structural maintenance tasks |
| `health` | Health summary |
| `info` | Node/edge counts + embedder |
| `save` | Persist state immediately |
| `reload` | Reload state from disk |
| `correction` | Penalize recent fired path for `chat_id` + inject correction node |
| `self_learn` / `self_correct` | Autonomous learning/correction helpers |
| `shutdown` | Stop the daemon (**requires** `confirm="shutdown"`) |
| `call` | Generic validated passthrough: provide `method` + `params` |

### How to use it (agent perspective)
The tool is registered automatically. Agents can call it like any other tool.

Query:
```
openclawbrain(action="query", query="how do we deploy", chat_id="telegram:123", top_k=4)
```

Feedback capture (canonical):
```
openclawbrain(action="capture_feedback", chat_id="telegram:123", kind="CORRECTION", content="Actually we use blue-green deploys, not rolling", message_id="telegram:123:456")
```

Legacy correction:
```
openclawbrain(action="correction", chat_id="telegram:123", content="Actually we use blue-green deploys, not rolling")
```

Generic call (validated passthrough):
```
openclawbrain(action="call", method="maintain", params={"tasks":["health","decay"]})
```

Shutdown (explicit guard):
```
openclawbrain(action="shutdown", confirm="shutdown")
```

### Requirements
- OpenClawBrain daemon must be running (`openclawbrain serve start --state ...`)
- The `daemon.sock` file must be accessible from the OpenClaw process

### Media understanding synergy
OpenClaw's built-in `tools.media` pipeline (audio transcription, image description) runs *before* the agent responds. When enabled, audio/image content is extracted to text and stored in session logs, so OpenClawBrain replay/full-learning can learn from media messages naturally.

To enable in your OpenClaw config:
```json
{
  "tools": {
    "media": {
      "audio": { "enabled": true },
      "image": { "enabled": true }
    }
  }
}
```

---

## Appendix: ASCII architecture diagram

```text
                         ┌──────────────────────────────────┐
                         │           OpenClaw Agent          │
                         │  (reads AGENTS.md instructions)   │
                         └───────────────┬───────────────────┘
                                         │
                                         │ query (summary)
                                         ▼
                         ┌──────────────────────────────────┐
                         │        OpenClawBrain daemon       │
                         │ openclawbrain serve start --state ... │
                         │  (NDJSON over stdin/stdout)       │
                         └───────────────┬───────────────────┘
                                         │
                          context + fired│ ids + timings
                                         ▼
                         ┌──────────────────────────────────┐
                         │        OpenClaw Agent reply       │
                         └───────────────┬───────────────────┘
                                         │
                          outcome (+/-)  │  correction text
                                         ▼
                         ┌──────────────────────────────────┐
                         │     learn / inject / maintain     │
                         │  (updates weights, adds nodes)    │
                         └──────────────────────────────────┘
```
