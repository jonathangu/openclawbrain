# OpenClawBrain — Your Agent's Second Brain

> New installs and reinstalls should use the split `0.4.0` lane: `@openclawbrain/openclaw@0.4.0` for the plugin/runtime payload and `@openclawbrain/cli@0.4.0` for the operator CLI. `@jonathangu/openclawbrain@0.3.5` remains compatibility-only for older installs.

<p align="center">
  <strong>Your OpenClaw agent should not relearn the same lesson twice.</strong>
</p>

<p align="center">
  <a href="https://openclawbrain.ai">🌐 Site</a> ·
  <a href="https://github.com/jonathangu/openclawbrain">📦 GitHub</a> ·
  <a href="https://openclawbrain.ai/jonathan-gu-2016-reinforcement-learning-paper.pdf">📄 2016 RL Paper</a> ·
  <a href="mailto:asianendowment@gmail.com">✉️ Contact</a>
</p>

---

## The Problem

Your AI assistant keeps making the same mistakes. You correct it, it forgets. You teach it a preference, it's gone by next session. You tell it to keep answers concise and end with the next action, and by tomorrow it's rambling again.

**That's not intelligence — that's forgetfulness wearing a fancy hat.**

## The Solution

**OpenClawBrain** gives your OpenClaw agent a second brain that actually learns. It combines:

- ✅ **Lossless transcript memory** — never lose a conversation
- ✅ **Learned routing graph** — knows what context to surface
- ✅ **Immediate corrections** — explicit user corrections can commit immediately and win on the next turn
- ✅ **Paper-faithful RL** — REINFORCE over full trajectories
- ✅ **Replay-gated packs** — no regressions, ever
- ✅ **Decision traces** — every choice is inspectable

## What It Does For You

| Before OpenClawBrain | After OpenClawBrain |
|---------------------|---------------------|
| Corrections vanish after session | Corrections persist forever |
| Agent relearns same lessons | Agent remembers what worked |
| Blind context selection | Learned retrieval from graph |
| No visibility into decisions | Full decision traces |

## Canonical Operator Path

Use the split `0.4.0` lane across GitHub and the site.

Current public packages:
- plugin/runtime payload: `@openclawbrain/openclaw@0.4.0`
- operator CLI: `@openclawbrain/cli@0.4.0`
- compatibility holdover for older installs: `@jonathangu/openclawbrain@0.3.5`

The now-proven public-registry flow on the real host `redogfood` is:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

`openclawbrain install` is still the activation-root pinning step after the plugin payload is present. `openclaw gateway restart` makes the new hook live immediately. `status --detailed` is the first verification read.

Current host/plugin caveat: some hosts still warn about a plugin id mismatch because the plugin manifest uses `openclawbrain` while the package/entry hint uses `openclaw`. The install still works; treat that warning as currently cosmetic rather than a failed attach.

### Install

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

This is the primary operator story now. The plugin payload is installed through OpenClaw, and the CLI runs through the published `@openclawbrain/cli` package rather than an older combined global package.

### Upgrade

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

Upgrade uses the same lane: refresh the plugin package inside OpenClaw, rerun `install` for the target OpenClaw home, restart the gateway, then verify.

### Verify

Use the detailed status view as the human check and JSON when you want the canonical machine-readable answer:

```bash
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --json
```

### Remove

Remove only the OpenClaw profile hook and keep OpenClawBrain data:

```bash
npx @openclawbrain/cli@0.4.0 openclawbrain detach --openclaw-home ~/.openclaw
openclaw gateway restart
```

Remove the hook and keep data explicitly:

```bash
npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart
```

Remove the hook and purge OpenClawBrain data for that install:

```bash
npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
```

`detach` and `uninstall` act on one OpenClaw home. The data semantics stay explicit: `detach` always keeps data, `uninstall --keep-data` keeps it explicitly, and `uninstall --purge-data` removes it. The plugin payload itself was installed through OpenClaw's plugin manager, so remove `@openclawbrain/openclaw` there separately only if you want the package files gone too.

### Compatibility Path

The older plugin/wrapper package still exists for compatibility with existing installs:

```bash
openclaw plugins install @jonathangu/openclawbrain@0.3.5
```

Treat that as a holdover lane, not the main operator story.

Decision and migration note: see [`docs/lifecycle.md`](docs/lifecycle.md).

> **Source checkout?** Clone the [GitHub repo](https://github.com/jonathangu/openclawbrain) only if you want to develop or contribute. Normal usage should use the published split packages above.

### Optional Workspace Bootstrap

If you want to prebuild a workspace snapshot after the lifecycle attach above, run:

```bash
npx @openclawbrain/cli@0.4.0 openclawbrain init /path/to/your/workspace
```

### Correct (in any conversation)

Just tell the agent it's wrong — the correction can commit on the next turn:

```text
User says:  How should you answer me?
Agent says: I'll answer in long freeform paragraphs.

User says:  Wrong — keep it concise, use bullets, and end with the next action.
Agent says: Got it. I'll keep answers concise, use bullets, and end with the next action.

User says:  How should you answer me?
Agent says: Concisely, with bullets, and ending with the next action.
```

The brain commits the correction immediately. Verify with `brain_trace`:

```bash
brain_trace        # shows the correction node fired and the old answer vetoed
```

For bulk or structured teaching, use `brain_teach` directly:

```
brain_teach instruction="Keep answers concise, use bullets, and end with the next action" kind="correction" tags=["style","preferences"]
```

### Inspect

```bash
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --json
brain_trace        # See recent routing decisions in conversation
```

## How It Works

```
You ask a question → Brain finds seed nodes → Traverses graph → 
Chooses best context → Surfaces as priority context → Learns from outcome
```

Deep dives:
- [`docs/lifecycle.md`](docs/lifecycle.md) — canonical install, upgrade, verify, detach, uninstall, and migration decision
- [`docs/configuration.md`](docs/configuration.md) — practical operator install, upgrade, remove, and config guide
- [`docs/release-notes-0.4.0.md`](docs/release-notes-0.4.0.md) — split packages and the proven public-registry dogfood flow
- [`docs/routing-prior.md`](docs/routing-prior.md) — why summaries are a routing/search prior rather than the truth layer
- [`docs/corrections.md`](docs/corrections.md) — how explicit user corrections become durable current truth
- [`docs/architecture.md`](docs/architecture.md) — full system architecture

### Two Layers, One Plugin

**Layer 1 — Lossless Transcript Memory (LCM)**
- Every conversation persisted in SQLite
- Older turns summarized into a DAG without throwing history away
- Grep, describe, expand any past conversation

**Layer 2 — Learned Routing Brain**
- Knowledge graph with corrections, toolcards, workflows
- Reinforcement learning from human/self/scanner/teacher signals
- Only serves from immutable promoted packs
- Replay gates block regressions

### The Learning Loop

1. **Seed** — Find candidate start nodes (embedding similarity)
2. **Expand** — Follow outgoing edges to candidate nodes  
3. **Route** — Softmax policy over candidates + STOP
4. **Fire** — Add chosen nodes to context
5. **Learn** — REINFORCE update from outcome (full trajectory)

## Real Examples

### Correcting the agent in plain language
```text
User says:  How should you answer me?
Agent says: I'll answer in long freeform paragraphs.

User says:  Wrong — keep it concise, use bullets, and end with the next action.
Agent says: Got it. I'll keep answers concise, use bullets, and end with the next action.

User says:  How should you answer me?
Agent says: Concisely, with bullets, and ending with the next action.
```

Behind the scenes:
- the user's correction is committed as durable memory
- on the next similar question, that correction is retrieved as priority context before the agent answers
- the stale default answer loses because the retrieved correction outranks it
- `brain_trace` shows the retrieved context, the fired correction, and the losing path

### Explicit teaching with brain_teach
```
brain_teach instruction="Keep answers concise, use bullets, and end with the next action" kind="correction" tags=["style","preferences"]

Brain: Creates a correction node. Next time the agent answers,
       it surfaces that preference as priority context.
```

### Learning from outcomes
```
Session 1: Agent uses approach A → Deploy fails → z = -0.5
Session 2: Agent uses approach B → Deploy succeeds → z = +0.5
Session 3+: Brain strengthens edges leading to approach B
```

## Why It's Different

### Paper-Faithful
Built on [Gu 2016](https://openclawbrain.ai/jonathan-gu-2016-reinforcement-learning-paper.pdf):
- Finite-horizon traversal (max 8 hops)
- Terminal reward signals only
- Full-trajectory REINFORCE (not one-step)
- Stochastic policy (softmax, never argmax)

### Safe by Design
- **Immutable packs** — serving graph never changes during queries
- **Replay gates** — mutations evaluated on clone first
- **Fail-open** — serving continues from last promoted pack if worker dies
- **Child-worker mode** — learning runs out-of-process

### Transparent
- Every decision traced with episode/trace IDs
- Shadow mode records without injecting
- `brain_trace` shows seed choice, hops, fired nodes, vetos

## Configuration

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "enabled": true,
        "config": {
          "brainEmbeddingProvider": "ollama",
          "brainEmbeddingModel": "bge-large:latest",
          "brainWorkerMode": "child",
          "brainRoot": "~/.openclaw/openclawbrain"
        }
      }
    }
  }
}
```

### With Ollama (local, free)

```json
{
  "brainEmbeddingProvider": "ollama",
  "brainEmbeddingModel": "bge-large:latest"
}
```

Defaults to `http://127.0.0.1:11434/v1`.

### With OpenAI or compatible

```json
{
  "brainEmbeddingProvider": "openai",
  "brainEmbeddingModel": "text-embedding-3-large",
  "brainEmbeddingBaseUrl": "https://your-endpoint.com/v1"
}
```

Set `OPENCLAWBRAIN_EMBEDDING_API_KEY` if needed.

## Operator Commands

Use the published CLI package directly for clean-host operator commands:

| Command | What it does |
|---------|-------------|
| `npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw` | Attach OpenClawBrain to one OpenClaw home after the plugin payload is installed |
| `openclaw gateway restart` | Reload the gateway after install, detach, or uninstall |
| `npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed` | Human verification for lifecycle, worker, and pack truth |
| `npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --json` | Canonical machine-readable verification for one installed target |
| `npx @openclawbrain/cli@0.4.0 openclawbrain init [workspace]` | Optional workspace bootstrap after attach |
| `npx @openclawbrain/cli@0.4.0 openclawbrain detach --openclaw-home ~/.openclaw` | Remove only the profile hook and keep data |
| `npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data|--purge-data` | Remove the hook and choose the data outcome explicitly |
| `brain_trace` | Inspect routing decisions inside the agent/tool lane |
| `brain_teach` | Explicitly teach corrections or instructions in conversation |

## The 2016 Paper

The routing policy implements **Lemma 6.1** from:

> **Jonathan Gu — Reinforcement Learning** (Econometrics Field, July 2016)

```
∂/∂ρ v_ρ(s_t) = E[ z · Σ_{l=t}^{T} ∂log P_ρ(a_l|s_l) / ∂ρ ]
```

Key insight: REINFORCE assigns credit to **every routing decision** in the episode, not just the last one. A good outcome strengthens the entire path.

[Read the paper →](https://openclawbrain.ai/jonathan-gu-2016-reinforcement-learning-paper.pdf)

## Current Status

| Metric | Status |
|--------|--------|
| Tests | ✅ 340 passing |
| Type check | ✅ Clean (`tsc --noEmit`) |
| Runtime proofs | ✅ Deterministic |
| Evidence bundles | ✅ Frozen |

## Where to Find It

| Resource | URL |
|----------|-----|
| 🌐 **Product site** | https://openclawbrain.ai |
| 📦 **GitHub repo** | https://github.com/jonathangu/openclawbrain |
| 📄 **2016 RL paper** | https://openclawbrain.ai/jonathan-gu-2016-reinforcement-learning-paper.pdf |
| 💬 **Discord** | https://discord.com/invite/clawd |
| 👤 **Jonathan's site** | https://jonathangu.com |

## Lifecycle Summary

```bash
# Install
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed

# Upgrade or repair
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed

# Verify
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --json

# Remove but keep data
npx @openclawbrain/cli@0.4.0 openclawbrain detach --openclaw-home ~/.openclaw
openclaw gateway restart

# Uninstall but keep data
npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart

# Remove and purge data
npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
```

Compatibility path for older installs:

```bash
openclaw plugins install @jonathangu/openclawbrain@0.3.5
```

Canonical lifecycle and migration note: [`docs/lifecycle.md`](docs/lifecycle.md)

For development or contributing, clone the repo and link locally instead of treating that checkout as the normal operator lane.

**Prerequisites:** OpenClaw, Node.js 22+, embeddings provider (Ollama or OpenAI-compatible)

---

*MIT License. Built by [Jonathan Gu](https://jonathangu.com).*
