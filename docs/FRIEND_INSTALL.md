# Copy-paste notes

## Project blurb

```text
Hey friends — I just shipped OpenClawBrain, my local-first memory system for AI agents.

The core idea: an agent should not just "remember everything." It should learn when memory actually matters. OpenClawBrain turns corrections, outcomes, misses, and handoffs into local evidence, then learns a small routing policy that decides when to bring the right memory into a future turn — or abstain when it is not confident.

The trust boundary is the important part: the LLM proposes meaning, but code owns validation, storage, calibration, promotion, and rollback. SQLite keeps the graph and evidence local and inspectable.

How it works: https://openclawbrain.ai/how-it-works/
Install or upgrade: https://openclawbrain.ai/install/
Project page: https://jonathangu.com/openclawbrain/

Install/upgrade if you already run OpenClaw:
openclaw plugins install clawhub:openclawbrain@0.2.21 --force
openclaw plugins enable openclawbrain
openclaw gateway restart
```

## Install note

Send this to someone who already has OpenClaw `2026.5.2` or later installed.

```text
Hey — if you want to try OpenClawBrain, install the latest native OpenClaw memory plugin.

What it does: local-first memory for OpenClaw agents. It stores durable corrections/workflows in SQLite, learns when those memories should route into a future turn, injects only a bounded relevant context slice, and exposes proof/status/search/graph/route-policy routes so you can verify what happened.

Install or upgrade:

openclaw plugins install clawhub:openclawbrain@0.2.21 --force
openclaw plugins enable openclawbrain
openclaw gateway restart

If ClawHub is still rate-limited or metadata is propagating, use the release archive fallback:

curl -L -o /tmp/openclawbrain-0.2.21.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.21/openclawbrain-0.2.21.tgz
openclaw plugins install /tmp/openclawbrain-0.2.21.tgz --force
openclaw plugins enable openclawbrain
openclaw gateway restart

Verify:

openclaw --version
openclaw plugins inspect openclawbrain --runtime
openclaw doctor

If you use multiple OpenClaw agents/profiles, scope it explicitly so each gets its own local graph:

openclaw config set plugins.entries.openclawbrain.config.scopes.agents '["main","pelican","bountiful"]' --strict-json
openclaw gateway restart

Then teach it a tiny durable rule, like “Use pnpm instead of npm in this repo.” Inspect with the plugin routes if your local gateway auth allows it, or use the OpenClaw dashboard/routes with your normal gateway auth:

/plugins/openclawbrain/status
/plugins/openclawbrain/doctor
/plugins/openclawbrain/proof?limit=10
/plugins/openclawbrain/search?query=pnpm&limit=10
/plugins/openclawbrain/graph?agentId=main&limit=10
/plugins/openclawbrain/route-policy

Docs: https://openclawbrain.ai/install/
Repo: https://github.com/jonathangu/openclawbrain
```
