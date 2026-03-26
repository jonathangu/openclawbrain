import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildRecordedSessionReplayFixture, runRecordedSessionReplay } from "../src/index.js";

function buildRecordedSessionTrace(overrides = {}) {
  return {
    contract: "recorded_session_trace.v1",
    traceId: "trace-train-freeze-eval",
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-25T10:00:00.000Z",
    bundleBuiltAt: "2026-03-25T10:30:00.000Z",
    sessionId: "session-train-freeze-eval",
    channel: "chat",
    sourceStream: "recorded/session",
    privacy: { sanitized: true, notes: ["test"] },
    workspace: {
      workspaceId: "workspace-train-freeze-eval",
      snapshotId: "snapshot-train-freeze-eval",
      capturedAt: "2026-03-25T09:55:00.000Z",
      rootDir: "/tmp/workspace-train-freeze-eval",
      revision: "rev-train-freeze-eval",
    },
    seedBuiltAt: "2026-03-25T09:56:00.000Z",
    seedActivatedAt: "2026-03-25T09:57:00.000Z",
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
    ...overrides,
  };
}

function createReplayRoot(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-replay-train-freeze-eval-"));
  t.after(() => {
    rmSync(root, { recursive: true, force: true });
  });
  return root;
}

function findLearnedReplayMode(bundle) {
  const learned = bundle.modes.find((mode) => mode.mode === "learned_replay");
  assert.ok(learned);
  return learned;
}

test("learned replay freezes the trained pack and router before eval scoring", (t) => {
  const rootDir = createReplayRoot(t);
  const fixture = buildRecordedSessionReplayFixture(
    buildRecordedSessionTrace({
      evalTurnCount: 2,
    }),
  );
  const learned = findLearnedReplayMode(runRecordedSessionReplay(rootDir, fixture));
  const evalTurns = learned.turns.filter((turn) => turn.phase === "eval");

  assert.deepEqual(
    learned.turns.map((turn) => turn.phase),
    ["train", "eval", "eval"],
  );
  assert.equal(learned.summary.trainTurnCount, 1);
  assert.equal(learned.summary.evalTurnCount, 2);
  assert.equal(learned.summary.promotionCount, 1);
  assert.equal(learned.summary.scannerEvidence.activePackChangeCount, 1);
  assert.notEqual(learned.summary.frozenEvalPackId, null);
  assert.notEqual(learned.summary.frozenEvalRouterIdentity, null);
  assert.equal(evalTurns.length, 2);
  assert.ok(evalTurns.every((turn) => turn.compileOk));
  assert.ok(evalTurns.every((turn) => turn.promoted === false));
  assert.ok(evalTurns.every((turn) => turn.usedLearnedRouteFn === true));
  assert.ok(evalTurns.every((turn) => turn.activePackId === learned.summary.frozenEvalPackId));
  assert.ok(evalTurns.every((turn) => turn.routerIdentity === learned.summary.frozenEvalRouterIdentity));
  assert.equal(new Set(evalTurns.map((turn) => turn.activePackId)).size, 1);
  assert.equal(new Set(evalTurns.map((turn) => turn.routerIdentity)).size, 1);
  assert.equal(new Set(evalTurns.map((turn) => turn.compileActiveVersion)).size, 1);
});

test("learned replay defaults to holding out the final turn for eval", (t) => {
  const rootDir = createReplayRoot(t);
  const fixture = buildRecordedSessionReplayFixture(buildRecordedSessionTrace());
  const learned = findLearnedReplayMode(runRecordedSessionReplay(rootDir, fixture));
  const finalTurn = learned.turns.at(-1);

  assert.equal(learned.summary.trainTurnCount, 2);
  assert.equal(learned.summary.evalTurnCount, 1);
  assert.equal(learned.summary.promotionCount, 2);
  assert.equal(finalTurn?.phase, "eval");
  assert.equal(finalTurn?.promoted, false);
  assert.equal(finalTurn?.activePackId, learned.summary.frozenEvalPackId);
  assert.equal(finalTurn?.routerIdentity, learned.summary.frozenEvalRouterIdentity);
});
