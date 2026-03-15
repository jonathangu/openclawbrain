# OpenClawBrain v2

OpenClawBrain v2 is a clean rebuild on top of [lossless-claw](https://github.com/Martian-Engineering/lossless-claw). The goal is a production-ready OpenClaw plugin that keeps lossless transcript memory while adding a correctly wired learning layer for retrieval and correction routing.

This repo is the active v2 codebase.

The earlier spike is archived at [jonathangu/openclawbrain-v1-spike-archive](https://github.com/jonathangu/openclawbrain-v1-spike-archive).

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Operator Commands](#operator-commands)
- [Fallback Behavior](#fallback-behavior)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## What it does today

When a conversation grows beyond the model's context window, OpenClaw (just like all of the other agents) normally truncates older messages. LCM instead:

1. **Persists every message** in a SQLite database, organized by conversation
2. **Summarizes chunks** of older messages into summaries using your configured LLM
3. **Condenses summaries** into higher-level nodes as they accumulate, forming a DAG (directed acyclic graph)
4. **Assembles context** each turn by combining summaries + recent raw messages
5. **Provides tools** (`lcm_grep`, `lcm_describe`, `lcm_expand`) so agents can search and recall details from compacted history

Nothing is lost. Raw messages stay in the database. Summaries link back to their source messages. Agents can drill into any summary to recover the original detail.

Today this repo ships the proven lossless-claw baseline while the learning layer is being rebuilt behind explicit module boundaries, tests, and operator commands. The README will only claim learned retrieval once it is wired end to end.

## Quick start

### Prerequisites

- OpenClaw with plugin context engine support
- Node.js 22+
- An LLM provider configured in OpenClaw (used for summarization)

### Install the plugin

Use OpenClaw's plugin installer once the package is published:

```bash
openclaw plugins install @jonathangu/openclawbrain
```

If you're running from a local OpenClaw checkout, use:

```bash
pnpm openclaw plugins install @jonathangu/openclawbrain
```

For local plugin development, link your working copy instead of copying files:

```bash
openclaw plugins install --link /path/to/openclawbrain
# or from a local OpenClaw checkout:
# pnpm openclaw plugins install --link /path/to/openclawbrain
```

The install command records the plugin, enables it, and applies compatible slot selection (including `contextEngine` when applicable).

### Configure OpenClaw

In most cases, no manual JSON edits are needed after `openclaw plugins install`.

If you need to set it manually, ensure the context engine slot points at `openclawbrain`:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "openclawbrain"
    }
  }
}
```

Restart OpenClaw after configuration changes.

### Initialize the brain index

The lossless transcript path works immediately. Learned retrieval needs an explicit init pass:

```bash
openclawbrain init /path/to/your/workspace
```

`openclawbrain init` scans the workspace, chunks source material, computes embeddings, builds the initial graph, writes `state.db`, creates pack `v000001`, and promotes it.

## Configuration

LCM is configured through a combination of plugin config and environment variables. Environment variables take precedence for backward compatibility.

### Plugin config

Add an `openclawbrain` entry under `plugins.entries` in your OpenClaw config:

```json
{
  "plugins": {
    "entries": {
      "openclawbrain": {
        "enabled": true,
        "config": {
          "freshTailCount": 32,
          "contextThreshold": 0.75,
          "incrementalMaxDepth": -1,
          "brainRoot": "~/.openclaw/openclawbrain",
          "brainEmbeddingProvider": "openai",
          "brainEmbeddingModel": "text-embedding-3-large"
        }
      }
    }
  }
}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LCM_ENABLED` | `true` | Enable/disable the plugin |
| `LCM_DATABASE_PATH` | `~/.openclaw/lcm.db` | Path to the SQLite database |
| `LCM_CONTEXT_THRESHOLD` | `0.75` | Fraction of context window that triggers compaction (0.0–1.0) |
| `LCM_FRESH_TAIL_COUNT` | `32` | Number of recent messages protected from compaction |
| `LCM_LEAF_MIN_FANOUT` | `8` | Minimum raw messages per leaf summary |
| `LCM_CONDENSED_MIN_FANOUT` | `4` | Minimum summaries per condensed node |
| `LCM_CONDENSED_MIN_FANOUT_HARD` | `2` | Relaxed fanout for forced compaction sweeps |
| `LCM_INCREMENTAL_MAX_DEPTH` | `0` | How deep incremental compaction goes (0 = leaf only, -1 = unlimited) |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Max source tokens per leaf compaction chunk |
| `LCM_LEAF_TARGET_TOKENS` | `1200` | Target token count for leaf summaries |
| `LCM_CONDENSED_TARGET_TOKENS` | `2000` | Target token count for condensed summaries |
| `LCM_MAX_EXPAND_TOKENS` | `4000` | Token cap for sub-agent expansion queries |
| `LCM_LARGE_FILE_TOKEN_THRESHOLD` | `25000` | File blocks above this size are intercepted and stored separately |
| `LCM_LARGE_FILE_SUMMARY_PROVIDER` | `""` | Provider override for large-file summarization |
| `LCM_LARGE_FILE_SUMMARY_MODEL` | `""` | Model override for large-file summarization |
| `LCM_SUMMARY_MODEL` | *(from OpenClaw)* | Model for summarization (e.g. `anthropic/claude-sonnet-4-20250514`) |
| `LCM_SUMMARY_PROVIDER` | *(from OpenClaw)* | Provider override for summarization |
| `LCM_AUTOCOMPACT_DISABLED` | `false` | Disable automatic compaction after turns |
| `LCM_PRUNE_HEARTBEAT_OK` | `false` | Retroactively delete `HEARTBEAT_OK` turn cycles from LCM storage |
| `OPENCLAWBRAIN_ENABLED` | `true` | Enable/disable the learning layer |
| `OPENCLAWBRAIN_ROOT` | `~/.openclaw/openclawbrain` | Root directory for `state.db` and immutable packs |
| `OPENCLAWBRAIN_EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `OPENCLAWBRAIN_EMBEDDING_MODEL` | `""` | Embedding model required for `init`, retrieval, and `brain_teach` |
| `OPENCLAWBRAIN_EMBEDDING_BASE_URL` | `""` | Optional embeddings API base URL override |
| `OPENCLAWBRAIN_MAX_HOPS` | `8` | Hard traversal cap |
| `OPENCLAWBRAIN_MAX_SEEDS` | `10` | Max seed nodes per query |
| `OPENCLAWBRAIN_SEMANTIC_THRESHOLD` | `0.7` | Minimum seed similarity |
| `OPENCLAWBRAIN_TRAINER_INTERVAL_MS` | `30000` | Background worker interval |

## Operator Commands

```bash
openclawbrain init [workspace]
openclawbrain status
openclawbrain trace [traceId]
openclawbrain replay
openclawbrain promote
openclawbrain rollback [version]
openclawbrain disable
openclawbrain enable
openclawbrain doctor
```

## Fallback Behavior

- If the brain has not been initialized, the plugin serves LCM-only context.
- If embeddings are not configured, learned retrieval and `brain_teach` stay disabled.
- If the background worker is unavailable, serving still uses the last promoted pack.
- `brain_teach` now binds taught corrections to the active conversation when invoked from a live tool session.

## Limitations

- Embedding support currently targets OpenAI-compatible `/v1/embeddings` APIs.
- The worker records structural mutation proposals but does not auto-promote them yet.
- Upstream `openclaw/plugin-sdk` type drift still affects full-repo `tsc --noEmit`.

## Roadmap

- Finish mutation replay-gate application for connect/prune/inject.
- Extend embedding adapters beyond OpenAI-compatible providers.
- Expand end-to-end runtime tests against a full OpenClaw install.

### Recommended starting configuration

```
LCM_FRESH_TAIL_COUNT=32
LCM_INCREMENTAL_MAX_DEPTH=-1
LCM_CONTEXT_THRESHOLD=0.75
```

- **freshTailCount=32** protects the last 32 messages from compaction, giving the model enough recent context for continuity.
- **incrementalMaxDepth=-1** enables unlimited automatic condensation after each compaction pass — the DAG cascades as deep as needed. Set to `0` (default) for leaf-only, or a positive integer for a specific depth cap.
- **contextThreshold=0.75** triggers compaction when context reaches 75% of the model's window, leaving headroom for the model's response.

### OpenClaw session reset settings

LCM preserves history through compaction, but it does **not** change OpenClaw's core session reset policy. If sessions are resetting sooner than you want, increase OpenClaw's `session.reset.idleMinutes` or use a channel/type-specific override.

```json
{
  "session": {
    "reset": {
      "mode": "idle",
      "idleMinutes": 10080
    }
  }
}
```

- `session.reset.mode: "idle"` keeps a session alive until the idle window expires.
- `session.reset.idleMinutes` is the actual reset interval in minutes.
- OpenClaw does **not** currently enforce a maximum `idleMinutes`; in source it is validated only as a positive integer.
- If you also use daily reset mode, `idleMinutes` acts as a secondary guard and the session resets when **either** the daily boundary or the idle window is reached first.
- Legacy `session.idleMinutes` still works, but OpenClaw prefers `session.reset.idleMinutes`.

Useful values:

- `1440` = 1 day
- `10080` = 7 days
- `43200` = 30 days
- `525600` = 365 days

For most long-lived LCM setups, a good starting point is:

```json
{
  "session": {
    "reset": {
      "mode": "idle",
      "idleMinutes": 10080
    }
  }
}
```

## Documentation

- [Configuration guide](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Agent tools](docs/agent-tools.md)
- [TUI Reference](docs/tui.md)
- [lcm-tui](tui/README.md)
- [Optional: enable FTS5 for fast full-text search](docs/fts5.md)

## Development

```bash
# Run tests
npx vitest

# Type check
npx tsc --noEmit

# Run a specific test file
npx vitest test/engine.test.ts
```

### Project structure

```
index.ts                    # Plugin entry point and registration
src/
  engine.ts                 # LcmContextEngine — implements ContextEngine interface
  assembler.ts              # Context assembly (summaries + messages → model context)
  compaction.ts             # CompactionEngine — leaf passes, condensation, sweeps
  summarize.ts              # Depth-aware prompt generation and LLM summarization
  retrieval.ts              # RetrievalEngine — grep, describe, expand operations
  expansion.ts              # DAG expansion logic for lcm_expand_query
  expansion-auth.ts         # Delegation grants for sub-agent expansion
  expansion-policy.ts       # Depth/token policy for expansion
  large-files.ts            # File interception, storage, and exploration summaries
  integrity.ts              # DAG integrity checks and repair utilities
  transcript-repair.ts      # Tool-use/result pairing sanitization
  types.ts                  # Core type definitions (dependency injection contracts)
  openclaw-bridge.ts        # Bridge utilities
  db/
    config.ts               # LcmConfig resolution from env vars
    connection.ts           # SQLite connection management
    migration.ts            # Schema migrations
  store/
    conversation-store.ts   # Message persistence and retrieval
    summary-store.ts        # Summary DAG persistence and context item management
    fts5-sanitize.ts        # FTS5 query sanitization
  tools/
    lcm-grep-tool.ts        # lcm_grep tool implementation
    lcm-describe-tool.ts    # lcm_describe tool implementation
    lcm-expand-tool.ts      # lcm_expand tool (sub-agent only)
    lcm-expand-query-tool.ts # lcm_expand_query tool (main agent wrapper)
    lcm-conversation-scope.ts # Conversation scoping utilities
    common.ts               # Shared tool utilities
test/                       # Vitest test suite
specs/                      # Design specifications
openclaw.plugin.json        # Plugin manifest with config schema and UI hints
tui/                        # Interactive terminal UI (Go)
  main.go                   # Entry point and bubbletea app
  data.go                   # Data loading and SQLite queries
  dissolve.go               # Summary dissolution
  repair.go                 # Corrupted summary repair
  rewrite.go                # Summary re-summarization
  transplant.go             # Cross-conversation DAG copy
  prompts/                  # Depth-aware prompt templates
.goreleaser.yml             # GoReleaser config for TUI binary releases
```

## License

MIT
