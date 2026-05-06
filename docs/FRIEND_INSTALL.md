# Copy-paste install note

Send this to someone who already has OpenClaw `2026.5.2` or later installed.

```text
Hey — if you want to try OpenClawBrain, install the latest native OpenClaw memory plugin.

What it does: local-first memory for OpenClaw agents. It stores durable corrections/workflows in SQLite, retrieves only the small relevant slice when useful, injects a bounded context block, and exposes proof/status/search/graph routes so you can verify what happened.

Install:

openclaw plugins install clawhub:openclawbrain@0.2.20
openclaw plugins enable openclawbrain
openclaw gateway restart

If ClawHub is still rate-limited or metadata is propagating, use the release archive fallback:

curl -L -o /tmp/openclawbrain-0.2.20.tgz \
  https://github.com/jonathangu/openclawbrain/releases/download/v0.2.20/openclawbrain-0.2.20.tgz
openclaw plugins install /tmp/openclawbrain-0.2.20.tgz --force
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

Docs: https://openclawbrain.ai/install/
Repo: https://github.com/jonathangu/openclawbrain
```
