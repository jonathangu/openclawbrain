# OpenClawBrain — Your Agent's Second Brain

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

## Quick Start

Current public packages:
- plugin/runtime payload: `@openclawbrain/openclaw@0.4.0`
- operator CLI: `@openclawbrain/cli@0.4.1`
- compatibility holdover for older installs: `@jonathangu/openclawbrain@0.3.5`

### Install

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

The plugin payload is installed through OpenClaw's plugin manager. The CLI runs through the published `@openclawbrain/cli` package. Upgrade uses the same lane: refresh the plugin package, rerun `install`, restart the gateway, then verify.

Current host/plugin caveat: some hosts still warn about a plugin id mismatch because the plugin manifest uses `openclawbrain` while the package/entry hint uses `openclaw`. The install still works; treat that warning as currently cosmetic rather than a failed attach.

For the full lifecycle reference — including verify, detach, uninstall, and migration — see [`docs/lifecycle.md`](docs/lifecycle.md).

### Compatibility Path

The older combined package still exists for compatibility with existing installs:

```bash
openclaw plugins install @jonathangu/openclawbrain@0.3.5
```

Treat that as a holdover lane, not the main operator story.

> **Source checkout?** Clone the [GitHub repo](https://github.com/jonathangu/openclawbrain) only if you want to develop or contribute. Normal usage should use the published split packages above.

### Optional One-Shot Learning Pass

After the lifecycle attach above, you can run one explicit local learning pass:

```bash
npx @openclawbrain/cli@0.4.1 openclawbrain learn --openclaw-home ~/.openclaw --json
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

Behind the scenes:
- the user's correction is committed as durable memory
- on the next similar question, that correction is retrieved as priority context before the agent answers
- the stale default answer loses because the retrieved correction outranks it
- `brain_trace` shows the retrieved context, the fired correction, and the losing path

For bulk or structured teaching, use `brain_teach` directly:

```
brain_teach instruction="Keep answers concise, use bullets, and end with the next action" kind="correction" tags=["style","preferences"]
```

### Inspect

```bash
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --json
brain_trace        # See recent routing decisions in conversation
```

## How It Works

```
You ask a question → Brain finds seed nodes → Traverses graph →
Chooses best context → Surfaces as priority context → Learns from outcome
```

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

Deep dives:
- [`docs/lifecycle.md`](docs/lifecycle.md) — canonical install, upgrade, verify, detach, uninstall, and migration
- [`docs/configuration.md`](docs/configuration.md) — practical operator setup and config guide
- [`docs/release-notes-0.4.1.md`](docs/release-notes-0.4.1.md) — CLI-only shared-home idempotence patch
- [`docs/release-notes-0.4.0.md`](docs/release-notes-0.4.0.md) — split packages and the proven public-registry dogfood flow
- [`docs/routing-prior.md`](docs/routing-prior.md) — why summaries are a routing/search prior rather than the truth layer
- [`docs/corrections.md`](docs/corrections.md) — how explicit user corrections become durable current truth
- [`docs/architecture.md`](docs/architecture.md) — full system architecture

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

Use the published CLI package for operator commands:

| Command | What it does |
|---------|-------------|
| `npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home ~/.openclaw` | Attach OpenClawBrain to one OpenClaw home after the plugin payload is installed |
| `openclaw gateway restart` | Reload the gateway after install, detach, or uninstall |
| `npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed` | Human verification for lifecycle, worker, and pack truth |
| `npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --json` | Canonical machine-readable verification |
| `npx @openclawbrain/cli@0.4.1 openclawbrain learn --openclaw-home ~/.openclaw --json` | Run one explicit local learning pass and inspect the result |
| `npx @openclawbrain/cli@0.4.1 openclawbrain detach --openclaw-home ~/.openclaw` | Remove only the profile hook and keep data |
| `npx @openclawbrain/cli@0.4.1 openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data\|--purge-data` | Remove the hook and choose the data outcome explicitly |
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
| Tests | ✅ Passing (`vitest run --dir test`) |
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

**Prerequisites:** OpenClaw, Node.js 22+, embeddings provider (Ollama or OpenAI-compatible)

---

*MIT License. Built by [Jonathan Gu](https://jonathangu.com).*
