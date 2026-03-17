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

Your AI coding assistant keeps making the same mistakes. You correct it, it forgets. You teach it a pattern, it's gone by next session. You figure out that `gh pr create` works better than `hub`, but the agent keeps suggesting `hub`.

**That's not intelligence — that's forgetfulness wearing a fancy hat.**

## The Solution

**OpenClawBrain** gives your OpenClaw agent a second brain that actually learns. It combines:

- ✅ **Lossless transcript memory** — never lose a conversation
- ✅ **Learned routing graph** — knows what context to surface
- ✅ **Immediate corrections** — `brain_teach` works instantly
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

### 1. Install (one command)

```bash
openclaw plugins install @jonathangu/openclawbrain
```

### 2. Initialize (one command)

```bash
openclawbrain init /path/to/your/workspace
```

That's it! The brain will discover your files, compute embeddings, and promote the first pack.

### 3. Teach (in any conversation)

```
brain_teach instruction="For PRs, use gh not hub"
brain_teach kind="correction" tags=["git", "github"]
```

### 4. Inspect

```bash
openclawbrain status     # Health check
openclawbrain trace     # See recent decisions
openclawbrain doctor    # Diagnose issues
```

## How It Works

```
You ask a question → Brain finds seed nodes → Traverses graph → 
Chooses best context → Surfaces as priority context → Learns from outcome
```

### Two Layers, One Plugin

**Layer 1 — Lossless Transcript Memory (LCM)**
- Every conversation persisted in SQLite
- Older turns summarized into DAG (never丢弃)
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

### Teaching a correction
```
You: "Don't use hub for PRs, use gh pr create"

Brain: Creates high-trust correction node.
Next time: Agent surfaces "Use gh pr create, not hub" as priority context.
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

| Command | What it does |
|---------|-------------|
| `openclawbrain init [workspace]` | Initialize brain on workspace |
| `openclawbrain status` | Health check, worker status, pack version |
| `openclawbrain trace [id]` | Inspect routing decisions |
| `openclawbrain replay` | Run replay gate on recent episodes |
| `openclawbrain promote` | Force promotion (if replay passes) |
| `openclawbrain doctor` | Diagnose issues |
| `brain_teach` | Teach corrections (in conversation) |

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
| Tests | ✅ 335 passing |
| Type check | ✅ Clean (source) |
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

## Install

```bash
# One command
openclaw plugins install @jonathangu/openclawbrain

# Or link local development
openclaw plugins install --link /path/to/openclawbrain

# Initialize once
openclawbrain init /path/to/workspace
```

**Prerequisites:** OpenClaw, Node.js 22+, embeddings provider (Ollama or OpenAI-compatible)

---

*MIT License. Built by [Jonathan Gu](https://jonathangu.com).*