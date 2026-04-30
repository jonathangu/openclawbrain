# Getting Started

This guide walks you through installing OpenClawBrain and using it for the first time.

## Prerequisites

- [OpenClaw](https://docs.openclaw.ai) 2026.4.29 or later
- An OpenClaw agent you use regularly (the `main` agent works)

## Step 1: Install

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

## Step 2: Configure

```bash
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config validate
openclaw gateway restart
```

Conservative mode is the safest starting point. It only injects context on turns where a correction or continuation is clearly relevant. On direct answers, it stays silent.

## Step 3: Create your activation files

```bash
mkdir -p ~/.openclawbrain/activation/main
```

Create one or more of these files:

**`~/.openclawbrain/activation/main/corrections.md`** — Things you've corrected the agent on:
```markdown
- Use "family inbox" not "work inbox" when checking email.
- Prefer `pnpm` over `npm` for this project.
- The timezone is America/Los_Angeles, not UTC.
```

**`~/.openclawbrain/activation/main/context.md`** — Ongoing context that helps across sessions:
```markdown
- Working on the OpenClawBrain plugin.
- Main repo is at /Users/jon/openclawbrain.
- Current priority: ship v0.1 to ClawHub.
```

**`~/.openclawbrain/activation/main/tool-guidance.md`** — Hints for tool-heavy turns:
```markdown
- Run tests before claiming something works.
- Check git status before assuming clean state.
```

You don't need all three. Create whichever are useful. The plugin reads them lazily only when it decides to inject.

## Step 4: Verify

```bash
# Check plugin status
curl http://127.0.0.1:18789/plugins/openclawbrain/status

# Check proof events (will be empty until the agent runs a turn)
curl http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=5
```

## Step 5: Use your agent normally

That's it. Use your agent as usual. OpenClawBrain runs in the background:
- On **correction follow-ups** (you correcting something), it injects `corrections.md`.
- On **continuations** or **retrieval-heavy** turns, it injects `context.md`.
- On **tool-heavy** turns, it injects `tool-guidance.md` plus a verification hint.
- On **direct answers**, it stays silent.

Check proof events periodically to see what it's doing:

```bash
curl http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=10
```

## Next steps

- Read [Architecture](ARCHITECTURE.md) for how the plugin decides what to do.
- Try `active` mode if you want more aggressive injection.
- Edit your activation files as you discover what context helps.
