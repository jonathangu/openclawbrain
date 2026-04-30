# OpenClawBrain Build Plan — Native OpenClaw Plugin

Date: 2026-04-30
Status: greenlit after OpenClaw-native review
Decision state: **BUILD A PURE NATIVE OPENCLAW PLUGIN**

## 0. The one-sentence goal

OpenClawBrain should become the smallest trustworthy native OpenClaw plugin that helps an OpenClaw agent use corrections, continuation context, and verification hints only when useful, while staying local, agent-scoped, conservative, and inspectable.

## 1. Product shape

Ship one thing through OpenClaw's plugin lifecycle:

```bash
openclaw plugins install openclawbrain
openclaw plugins enable openclawbrain

openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json

openclaw config validate
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
```

No separate installer package for v1.
No external runtime adapter.
No root `openclawbrain` config key.
No context-engine replacement.
No CLI-first product requirement.

The product lives inside OpenClaw’s plugin lifecycle and is ultimately used through OpenClaw.

## 2. User-facing promise

OpenClawBrain is a local, agent-scoped selective intervention layer for OpenClaw.

It helps an OpenClaw agent:

- remember corrections,
- continue bounded work,
- use relevant local context,
- verify before claiming when the turn is tool-heavy,
- stay silent on direct answers,
- show local proof of what it did.

It is not:

- generic agent memory,
- a cloud memory service,
- a smarter-brain marketing layer,
- a replacement for OpenClaw memory/search/context engine,
- something that should fire every turn.

## 3. Canonical package layout

```text
packages/openclaw-plugin/
  package.json
  tsconfig.json
  openclaw.plugin.json
  src/
    index.ts
    config.ts
    redact.ts
    policy.ts
    context-files.ts
    proof-store.ts
    status.ts
```

This package is the publishable package: `openclawbrain`.

Package metadata must declare:

```json
{
  "name": "openclawbrain",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "files": ["dist", "openclaw.plugin.json"],
  "openclaw": {
    "extensions": ["./dist/index.js"]
  }
}
```

The old multi-package scaffold can remain as research/reference temporarily, but the v1 product path is the native plugin package above.

## 4. Native OpenClaw integration contract

### Manifest

`openclaw.plugin.json` owns the schema.

Config lives only at:

```text
plugins.entries.openclawbrain.config
```

Required config fields:

- `enabled`: default `false`
- `mode`: `off | proof-only | conservative | active`, default `conservative`
- `activationRoot`: default `~/.openclawbrain/activation/${agentId}`
- `proofEvents`: default `true`
- `proofRetentionEvents`: default `1000`
- `maxContextChars`: default `3000`
- `includeActivationContext`: default `true`
- `rawTranscriptUpload`: const `false`
- `scopes.agents`: default `["main"]`; empty array means all agents

### Hook permissions

Prompt mutation requires explicit plugin hook permission:

```bash
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
```

Do not require conversation access by default.

Only set this if a later feature truly needs final-message/full-conversation observation:

```bash
openclaw config set plugins.entries.openclawbrain.hooks.allowConversationAccess true --strict-json
```

### Hook usage

Use prompt hooks, not the context engine:

- `before_prompt_build` — primary v1 injection seam.
- `agent_turn_prepare` — use only if current OpenClaw runtime exposes it; otherwise treat as future-compatible.
- `model_call_started` / `model_call_ended` — sanitized model-call telemetry only.
- `gateway_start` / `gateway_stop` — service lifecycle and health/status surface.

Optional only:

- `agent_end` — disabled by default; use only when `plugins.entries.openclawbrain.hooks.allowConversationAccess=true` and the feature truly needs it.

Do not use `before_tool_call` for “tool-heavy verification” in v1. That hook happens after the model chooses a tool. For v1, inject a bounded verification hint before prompt build.

### First-class surfaces beyond hooks

Register first-class plugin capabilities:

- `api.registerService(...)` for OpenClawBrain status/health/proof-store service.
- `api.registerHttpRoute(...)` for local plugin status inspection: `/plugins/openclawbrain/status`.
- `api.registerHttpRoute(...)` for local bounded proof inspection: `/plugins/openclawbrain/proof?limit=20`.

The proof route must return recent redacted proof events only, never raw prompts or raw transcripts.

This avoids being merely hook-only and aligns with OpenClaw plugin expectations.

## 5. Runtime policy

Keep policy pure and deterministic inside `policy.ts`.

Input:

- mode
- redacted prompt
- prompt hash
- timestamp
- OpenClaw config/profile identity when available
- agent id
- session key hash
- run id hash

Output:

- `stay_silent`
- `proof_only`
- `correction_only`
- `full_context`

Selected product policy:

| Turn slice | Decision |
| --- | --- |
| direct-answer | stay_silent |
| unknown / low confidence | stay_silent |
| correction-follow-up | correction_only |
| stale-memory-conflict | correction_only |
| continuation | full_context |
| retrieval-heavy | full_context |
| tool-heavy | full_context with verification hint |

Silence is a success state, not a miss.

## 6. Context source

Do not duplicate OpenClaw memory.

OpenClawBrain reads only small local activation files under `activationRoot`:

- `context.md`
- `corrections.md`
- `tool-guidance.md`

Rules:

- read only fixed filenames, never arbitrary user-provided paths,
- read only when the policy fires,
- reject symlinks,
- reject files over the byte cap before reading,
- treat `activationRoot` as local-only and never fetch remote URLs,
- create proof/context directories with owner-only permissions where possible,
- redact before use,
- clip to `maxContextChars`,
- record only redacted memory/file ids in proof,
- never upload raw transcripts.

Injection behavior:

- `stay_silent`: return nothing; write proof.
- `proof_only`: return nothing; write proof.
- `correction_only`: return bounded `prependContext` with correction-only guidance.
- `full_context`: return bounded `prependContext` or `appendContext` with selected context summary.
- prompt injection disabled: return nothing; write fail-closed `stay_silent` proof.

## 7. Proof store

Proof is the trust surface.

Store local JSONL under activation root:

```text
~/.openclawbrain/activation/<agentId>/proof-events.jsonl
~/.openclawbrain/activation/<agentId>/status.json
```

Proof event minimum:

```json
{
  "schemaVersion": "ocb.proof.event.v1",
  "pluginVersion": "0.1.0",
  "profileId": "...",
  "agentId": "...",
  "sessionKeyHash": "...",
  "runIdHash": "...",
  "promptHash": "...",
  "eventId": "...",
  "timestamp": "...",
  "slice": "direct-answer | continuation | correction-follow-up | retrieval-heavy | tool-heavy | stale-memory-conflict | unknown",
  "mode": "conservative",
  "decisionKind": "stay_silent | proof_only | correction_only | full_context",
  "reasonCode": "...",
  "usedMemoryIdsRedacted": [],
  "rawTranscriptStored": false,
  "rawUserTextStored": false,
  "redactionApplied": true,
  "hashesOnlyForUserText": true
}
```

Avoid broad claims like `containsRealUserData=false`. Proof metadata is derived from user activity, even when raw user text is not stored. The defensible claim is: no raw transcript storage, redaction applied, and hashes only for user text.

Retention:

- keep last `proofRetentionEvents`, default `1000`, bounded `50..50000`.

## 8. Status/proof UX

Primary status should be available through OpenClaw plugin inspection and plugin HTTP routes.

Target operator outputs:

```bash
openclaw plugins inspect openclawbrain --json
```

Status route:

```text
/plugins/openclawbrain/status
```

Proof route:

```text
/plugins/openclawbrain/proof?limit=20
```

Example status payload:

```json
{
  "ok": true,
  "enabled": true,
  "mode": "conservative",
  "agentId": "main",
  "activationRoot": "~/.openclawbrain/activation/main",
  "proofEvents": "writing",
  "rawTranscriptUpload": false,
  "lastDecisionKind": "stay_silent",
  "lastDecisionAt": "..."
}
```

A standalone `openclawbrain` CLI is optional after v1 proves native plugin usage. It should not be required for install/enable/status/proof.

## 9. Implementation phases

### Phase 1 — Replace scaffold with native plugin package

- Convert `packages/openclaw-plugin` to TypeScript.
- Add `package.json`, `tsconfig.json`, manifest, and `src/*.ts` files.
- Build to `dist/index.js`.
- Make package publishable as `openclawbrain`.
- Remove v1 dependency on `packages/installer`, `packages/cli`, and `packages/openclaw-integration`.

Gate:

```bash
pnpm --dir packages/openclaw-plugin check
pnpm --dir packages/openclaw-plugin build
npm pack --dry-run --workspace packages/openclaw-plugin
```

### Phase 2 — Policy and redaction

- Implement turn classifier.
- Implement selected policy mapping.
- Implement prompt/session/run hashing.
- Implement redaction for secrets, emails, phones, URLs, and secret-like blobs.
- Ensure `rawTranscriptUpload=true` fails closed.

Gate:

- unit tests for all turn slices,
- unit tests for `proof-only`, `off`, `conservative`, and fail-closed raw-upload config,
- proof tests assert `rawTranscriptStored=false`, `rawUserTextStored=false`, `redactionApplied=true`, and `hashesOnlyForUserText=true`.

### Phase 3 — Context files and injection

- Read `context.md`, `corrections.md`, `tool-guidance.md` only from activation root.
- Reject symlinks and oversized files before reading.
- Enforce context character limits.
- Redact and clip before injection.
- Return bounded `prependContext` from `before_prompt_build`.
- No context-engine slot usage.

Gate:

- direct-answer returns no prompt mutation,
- correction-only injects only correction guidance,
- continuation/tool-heavy/retrieval-heavy inject bounded context,
- injection disabled returns no mutation and writes proof,
- arbitrary paths and symlinked activation files are rejected.

### Phase 4 — Proof and status service

- Append proof events locally.
- Retain last N events.
- Write `status.json`.
- Register service/status route.
- Register bounded proof route.
- Observe model-call hooks without storing prompt/response content.
- Do not enable `agent_end` unless `allowConversationAccess=true` is explicitly configured.

Gate:

- status route returns JSON,
- proof route returns bounded redacted recent events,
- proof log contains no raw prompt,
- proof contains `agentId`, `sessionKeyHash`, `promptHash`, `rawUserTextStored=false`, and `hashesOnlyForUserText=true`.

### Phase 5 — Native OpenClaw dogfood

Use a disposable/local profile first:

```bash
openclaw plugins install -l ./packages/openclaw-plugin
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config validate
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
```

Run four real turns:

1. direct answer → no injection, proof says `stay_silent`
2. correction/stale-memory turn → bounded correction injection
3. continue → bounded context injection
4. tool-heavy verification → verification hint/context

Gate:

- all four proof events written,
- status route healthy,
- proof route healthy,
- no raw transcript stored,
- no cross-agent activation-root leakage.

### Phase 6 — Fresh package install

Test from packed artifact, not repo source:

```bash
cd packages/openclaw-plugin
npm pack
openclaw plugins install ./openclawbrain-0.1.0.tgz
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config validate
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
```

Gate:

- no repo-relative assumptions,
- manifest discovered,
- `dist/index.js` loaded,
- status/proof routes work.

### Phase 7 — Evidence regression

Keep existing evidence spine as regression proof, not as the product itself:

```bash
pnpm ocb:traces:production-status
pnpm ocb:e2e:smoke
pnpm ocb:e2e:production
```

Gate:

- `40` admitted real traces,
- `evidence_e2e_complete=true`,
- selected policy still scores `40/40` wins and `0/40` harms.

### Phase 8 — Public release readiness

Before publishing:

- package name locked: `openclawbrain`,
- license/repository metadata added,
- README rewritten around OpenClaw plugin lifecycle,
- provenance-ready CI path created,
- fresh machine/install path tested,
- uninstall/disable path verified.

Publish only after:

```bash
openclaw plugins install openclawbrain
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.enabled true --strict-json
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
openclaw config validate
openclaw gateway restart
openclaw plugins inspect openclawbrain --json
# run real OpenClaw turn
# inspect status/proof routes
openclaw plugins disable openclawbrain
```

## 10. Non-goals for v1

Do not build yet:

- separate installer package,
- separate CLI-first product,
- remote service,
- cloud sync,
- context-engine replacement,
- broad memory backend,
- dashboard,
- marketplace launch,
- more benchmark polish.

## 11. Definition of done

OpenClawBrain v0.1 is done when a normal OpenClaw user can install the plugin through OpenClaw, enable it for one agent, run real turns, see conservative bounded interventions, and inspect local status/proof routes showing exactly what happened without raw transcript or raw user-text storage.
