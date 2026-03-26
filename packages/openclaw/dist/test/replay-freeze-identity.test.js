import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildRecordedSessionReplayFixture, runRecordedSessionReplay } from "../../../cli/dist/src/index.js";
import { compileRuntimeContext } from "../src/index.js";

function buildRecordedSessionTrace() {
  return {
    contract: "recorded_session_trace.v1",
    traceId: "trace-openclaw-replay-freeze-identity",
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-25T10:00:00.000Z",
    bundleBuiltAt: "2026-03-25T10:30:00.000Z",
    sessionId: "session-openclaw-replay-freeze-identity",
    channel: "chat",
    sourceStream: "recorded/session",
    privacy: { sanitized: true, notes: ["test"] },
    workspace: {
      workspaceId: "workspace-openclaw-replay-freeze-identity",
      snapshotId: "snapshot-openclaw-replay-freeze-identity",
      capturedAt: "2026-03-25T09:55:00.000Z",
      rootDir: "/tmp/workspace-openclaw-replay-freeze-identity",
      revision: "rev-openclaw-replay-freeze-identity",
    },
    seedBuiltAt: "2026-03-25T09:56:00.000Z",
    seedActivatedAt: "2026-03-25T09:57:00.000Z",
    evalTurnCount: 2,
    seedCues: [
      {
        cueId: "cue-1",
        createdAt: "2026-03-25T09:50:00.000Z",
        content: "Always read README before editing code.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: "turn-1",
        createdAt: "2026-03-25T10:01:00.000Z",
        deliveredAt: "2026-03-25T10:01:30.000Z",
        userMessage: "What should I read before editing?",
        feedback: [
          {
            createdAt: "2026-03-25T10:01:45.000Z",
            content: "Correct: read README before editing code.",
            kind: "approval",
          },
        ],
        expectedContextPhrases: ["readme before editing"],
      },
      {
        turnId: "turn-2",
        createdAt: "2026-03-25T10:05:00.000Z",
        deliveredAt: "2026-03-25T10:05:30.000Z",
        userMessage: "Before changing files, what is the rule?",
        expectedContextPhrases: ["readme before editing"],
      },
      {
        turnId: "turn-3",
        createdAt: "2026-03-25T10:10:00.000Z",
        deliveredAt: "2026-03-25T10:10:30.000Z",
        userMessage: "What must happen before editing code?",
        expectedContextPhrases: ["readme before editing"],
      },
    ],
  };
}

test("compileRuntimeContext enforces frozen replay eval identity", () => {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-openclaw-replay-freeze-identity-"));

  try {
    const fixture = buildRecordedSessionReplayFixture(buildRecordedSessionTrace());
    const bundle = runRecordedSessionReplay(rootDir, fixture);
    const learned = bundle.modes.find((mode) => mode.mode === "learned_replay");

    assert.ok(learned);
    assert.notEqual(learned.summary.frozenEvalPackId, null);
    assert.notEqual(learned.summary.frozenEvalRouterIdentity, null);

    const activationRoot = path.join(rootDir, "learned_replay", "activation");
    const frozenIdentity = {
      packId: learned.summary.frozenEvalPackId,
      routerIdentity: learned.summary.frozenEvalRouterIdentity,
    };
    const compileInput = {
      activationRoot,
      message: "Before changing files, what is the rule?",
      _suppressServeLog: true,
    };

    const ok = compileRuntimeContext({
      ...compileInput,
      _frozenReplayEvalIdentity: frozenIdentity,
    });
    assert.equal(ok.ok, true);

    const packMismatch = compileRuntimeContext({
      ...compileInput,
      _frozenReplayEvalIdentity: {
        packId: "pack-mismatch",
        routerIdentity: frozenIdentity.routerIdentity,
      },
    });
    assert.equal(packMismatch.ok, false);
    assert.equal(packMismatch.fallbackToStaticContext, false);
    assert.equal(packMismatch.hardRequirementViolated, true);
    assert.match(packMismatch.error, /Frozen replay eval identity mismatch/);

    const routerMismatch = compileRuntimeContext({
      ...compileInput,
      _frozenReplayEvalIdentity: {
        packId: frozenIdentity.packId,
        routerIdentity: "router-mismatch",
      },
    });
    assert.equal(routerMismatch.ok, false);
    assert.equal(routerMismatch.fallbackToStaticContext, false);
    assert.equal(routerMismatch.hardRequirementViolated, true);
    assert.match(routerMismatch.error, /Frozen replay eval identity mismatch/);
  } finally {
    rmSync(rootDir, { recursive: true, force: true });
  }
});
