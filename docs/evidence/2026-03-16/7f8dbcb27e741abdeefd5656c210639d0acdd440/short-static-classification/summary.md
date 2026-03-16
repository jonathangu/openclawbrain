# Short-static host-path classification summary

- commit: `7f8dbcb27e741abdeefd5656c210639d0acdd440`
- workspace: `/Users/cormorantai/.openclaw/workspace-ocbphase1`
- validation root: `/Users/cormorantai/.openclaw-ocbphase1-short-static`
- model: `ollama/qwen2.5:7b-instruct`
- embedding: `ollama/bge-large:latest`
- classification bucket: **upstream host-agent/profile interaction**
- blocked by stale host seam: true
- same local session key across different --to values: false
- same local session id across different --to values: false

## Host preflight
- preflight root: `/Users/cormorantai/.openclaw-ocbphase1-short-static/preflight`
- preflight config: `/Users/cormorantai/.openclaw-ocbphase1-short-static/preflight/openclaw.json`
- setup-only exit: 0
- doctor exit: 0
- sdk probe exit: 0

## Why this bucket
- OpenClaw config schema no longer accepts `plugins.slots.contextEngine` in the sterile host lane.
- The current OpenClaw plugin API no longer exposes `registerContextEngine`, so the host harness seam used by this repo is stale.
- That means the current raw host validation lane cannot honestly reach the short-static semantic question yet; the host/plugin integration boundary moved underneath the Phase 1 harness.
- Freeze this as upstream host-agent/profile interaction for now, then adapt the plugin + config seam before claiming host-path short-static behavior is classified.

## Scenario matrix
- Skipped turn-level host probes because the host/plugin seam is stale before the agent path becomes meaningful.

## Honest release implication
- Do not treat the current raw host lane as a valid short-static proof surface. First adapt the plugin/config seam to the current OpenClaw host, then rerun turn-level classification on top of that repaired boundary.
