# Copy-paste install note

Send this to someone who already has OpenClaw `2026.4.29` or later installed.

```text
Hey — if you want to try OpenClawBrain, it is now a native OpenClaw plugin.

What it does: local-first memory for OpenClaw agents. It stores durable corrections/workflows in SQLite, retrieves only the small relevant slice when useful, injects a bounded context block, and exposes proof/status/search/graph routes so you can verify what happened.

Install:

openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
openclaw gateway restart

Verify:

openclaw plugins inspect openclawbrain --json
curl http://127.0.0.1:18789/plugins/openclawbrain/status
curl http://127.0.0.1:18789/plugins/openclawbrain/doctor

If you use multiple OpenClaw agents/profiles, scope it explicitly so each gets its own local graph:

openclaw config set plugins.entries.openclawbrain.config.scopes.agents '["main","pelican","bountiful"]' --strict-json
openclaw gateway restart

Then teach it a tiny durable rule, like “Use pnpm instead of npm in this repo,” and inspect:

curl 'http://127.0.0.1:18789/plugins/openclawbrain/proof?limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/search?query=pnpm&limit=10'
curl 'http://127.0.0.1:18789/plugins/openclawbrain/graph?agentId=main&limit=10'

Docs: https://openclawbrain.ai/install/
Repo: https://github.com/jonathangu/openclawbrain
```
