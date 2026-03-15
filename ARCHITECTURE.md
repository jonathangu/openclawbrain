# OpenClawBrain: Architecture

## Two Layers, One Plugin

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenClaw Gateway                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          OpenClawBrain Plugin (contextEngine)         │   │
│  │                                                      │   │
│  │  ┌─────────────────┐    ┌──────────────────────┐    │   │
│  │  │  Layer 1: LCM   │    │  Layer 2: Brain      │    │   │
│  │  │  Transcript DAG  │    │  Learned Graph       │    │   │
│  │  │                 │    │                      │    │   │
│  │  │  messages       │    │  brain_nodes         │    │   │
│  │  │  summaries      │    │  brain_edges         │    │   │
│  │  │  context_items  │    │  brain_episodes      │    │   │
│  │  │  compaction     │    │  brain_labels        │    │   │
│  │  │                 │    │  brain_packs         │    │   │
│  │  │  grep/describe/ │    │  brain_teach/status/ │    │   │
│  │  │  expand tools   │    │  trace tools         │    │   │
│  │  └─────────────────┘    └──────────────────────┘    │   │
│  │                                                      │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │           Hybrid Context Assembly             │    │   │
│  │  │  1. Correction cards (brain, highest priority)│    │   │
│  │  │  2. Route-selected evidence (brain)           │    │   │
│  │  │  3. Toolcards and playbooks (brain)           │    │   │
│  │  │  4. Transcript support (LCM fresh tail + DAG) │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  │                                                      │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │          Background Learner Service           │    │   │
│  │  │  processLabels → teacher → REINFORCE → decay  │    │   │
│  │  │  → mutations → pack promotion                 │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Serving Isolation

- Gateway reads only the **promoted pack** (immutable)
- Learner daemon writes only to **mutable training state**
- Pack promotion requires passing the **replay gate**
- Live serving graph is never mutated during queries

## Data Flow

```
User message → Engine.ingestSingle() → LCM persistence + Brain label harvesting
                                          ↓
Model turn → Assembler.assemble() → LCM context items + Brain traversal
                                          ↓
                                    Hybrid prompt (4 sections)
                                          ↓
Background → Trainer.tick() → labels → REINFORCE → decay → mutations → promotion
```

## Sacred Rules

1. LCM's summary DAG ≠ brain's learned graph
2. Teacher sees only what the router saw
3. Human corrections outrank everything
4. Mutations: proposal → replay gate → promotion
5. Brain is additive — fail open, never break the session
