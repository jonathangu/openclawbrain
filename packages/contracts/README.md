# `@openclawbrain/contracts`

Canonical TypeScript contracts for the TypeScript-first OpenClawBrain build.

This package is the starting point if you need the public artifact and runtime payload shapes without pulling in pack loading or compilation logic.

If you want narrower public surfaces, the workspace also publishes `@openclawbrain/events`, `@openclawbrain/event-export`, `@openclawbrain/workspace-metadata`, and `@openclawbrain/provenance`.

## Install

```bash
pnpm add @openclawbrain/contracts
```

## Includes

- contract ids for runtime compile, interaction events, feedback events, manifests, and activation pointers
- workspace metadata and pack provenance shapes
- payload validators for the current `v1` public shapes
- canonical JSON and checksum helpers for immutable artifact payloads

## Example

```ts
import {
  buildNormalizedEventExport,
  CONTRACT_IDS,
  checksumJsonPayload,
  createFeedbackEvent,
  createInteractionEvent,
  validateRuntimeCompileRequest
} from "@openclawbrain/contracts";

const request = {
  contract: CONTRACT_IDS.runtimeCompile,
  agentId: "agent-1",
  userMessage: "compile feedback context",
  maxContextBlocks: 2,
  modeRequested: "heuristic"
} as const;

const errors = validateRuntimeCompileRequest(request);
const checksum = checksumJsonPayload(request);

const eventExport = buildNormalizedEventExport({
  interactionEvents: [
    createInteractionEvent({
      eventId: "evt-1",
      agentId: "agent-1",
      sessionId: "session-1",
      channel: "cli",
      sequence: 10,
      kind: "memory_compiled",
      createdAt: "2026-03-06T00:10:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "openclaw/runtime/demo" }
    })
  ],
  feedbackEvents: [
    createFeedbackEvent({
      eventId: "evt-2",
      agentId: "agent-1",
      sessionId: "session-1",
      channel: "cli",
      sequence: 11,
      kind: "teaching",
      createdAt: "2026-03-06T00:11:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "openclaw/runtime/demo" },
      content: "Promote the pack only after validation."
    })
  ]
});
```
