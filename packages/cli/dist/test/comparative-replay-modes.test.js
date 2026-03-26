import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildRecordedSessionReplayFixture, runRecordedSessionReplay } from "../src/index.js";

function createWorkspace(t, label) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), `${label}-workspace-`));
  writeFileSync(
    path.join(rootDir, "README.md"),
    "# Recorded session workspace\nThe routing guide lives here.\n",
    "utf8",
  );
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-25T00:00:00.000Z",
    rootDir,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["test"],
  };
}

function createTrace(t) {
  const workspace = createWorkspace(t, "comparative-replay");
  return {
    contract: "recorded_session_trace.v1",
    traceId: "trace-comparative-replay",
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-25T00:00:00.000Z",
    bundleBuiltAt: "2026-03-25T00:10:00.000Z",
    agentId: "agent",
    sessionId: "session-comparative-replay",
    channel: "cli",
    sourceStream: "recorded/session",
    privacy: {
      sanitized: true,
      notes: ["test fixture"],
    },
    workspace,
    seedBuiltAt: "2026-03-25T00:01:00.000Z",
    seedActivatedAt: "2026-03-25T00:02:00.000Z",
    seedCues: [
      {
        cueId: "cue-routing-guide",
        createdAt: "2026-03-25T00:00:30.000Z",
        content: "The routing guide lives here.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: "turn-1",
        createdAt: "2026-03-25T00:03:00.000Z",
        deliveredAt: "2026-03-25T00:03:30.000Z",
        userMessage: "show the routing guide",
        runtimeHints: ["routing", "guide"],
        feedback: [
          {
            createdAt: "2026-03-25T00:03:45.000Z",
            content: "Keep the routing guide easy to find.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["routing guide"],
      },
      {
        turnId: "turn-2",
        createdAt: "2026-03-25T00:05:00.000Z",
        deliveredAt: "2026-03-25T00:05:30.000Z",
        userMessage: "show the routing guide again",
        runtimeHints: ["routing", "guide", "again"],
        feedback: [
          {
            createdAt: "2026-03-25T00:05:45.000Z",
            content: "The routing guide is still the right answer.",
            kind: "approval",
          },
        ],
        expectedContextPhrases: ["routing guide"],
      },
    ],
  };
}

test("recorded session replay exposes explicit comparative modes with real runtime evidence", (t) => {
  const trace = createTrace(t);
  const fixture = buildRecordedSessionReplayFixture(trace);
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "comparative-replay-root-"));
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });

  const bundle = runRecordedSessionReplay(rootDir, fixture);

  assert.deepEqual(bundle.modes.map((mode) => mode.mode), [
    "no_brain",
    "vector_only",
    "graph_prior_only",
    "learned_route",
  ]);

  const noBrain = bundle.modes.find((mode) => mode.mode === "no_brain");
  const vectorOnly = bundle.modes.find((mode) => mode.mode === "vector_only");
  const graphPriorOnly = bundle.modes.find((mode) => mode.mode === "graph_prior_only");
  const learnedRoute = bundle.modes.find((mode) => mode.mode === "learned_route");

  assert.ok(noBrain);
  assert.ok(vectorOnly);
  assert.ok(graphPriorOnly);
  assert.ok(learnedRoute);

  assert.equal(noBrain.summary.activationStrategy, "no_brain");
  assert.equal(noBrain.summary.modeRequested, null);
  assert.equal(noBrain.summary.selectionEngine, null);

  assert.equal(vectorOnly.summary.activationStrategy, "seed_pack");
  assert.equal(vectorOnly.summary.modeRequested, "heuristic");
  assert.equal(vectorOnly.summary.selectionEngine, "flat_rank_v1");
  assert.ok(vectorOnly.turns.every((turn) => turn.replayMode === "vector_only"));
  assert.ok(vectorOnly.turns.every((turn) => turn.modeRequested === "heuristic"));
  assert.ok(vectorOnly.turns.every((turn) => turn.selectionEngine === "flat_rank_v1"));

  assert.equal(graphPriorOnly.summary.activationStrategy, "seed_pack");
  assert.equal(graphPriorOnly.summary.modeRequested, "heuristic");
  assert.equal(graphPriorOnly.summary.selectionEngine, "graph_walk_v1");
  assert.ok(graphPriorOnly.turns.every((turn) => turn.replayMode === "graph_prior_only"));
  assert.ok(graphPriorOnly.turns.every((turn) => turn.modeRequested === "heuristic"));
  assert.ok(graphPriorOnly.turns.every((turn) => turn.selectionEngine === "graph_walk_v1"));

  assert.equal(learnedRoute.summary.activationStrategy, "continuous_learned_loop");
  assert.equal(learnedRoute.summary.modeRequested, "learned");
  assert.equal(learnedRoute.summary.selectionEngine, "graph_walk_v1");
  assert.ok(learnedRoute.turns.every((turn) => turn.replayMode === "learned_route"));
  assert.ok(learnedRoute.turns.every((turn) => turn.selectionEngine === "graph_walk_v1"));
  assert.ok(learnedRoute.turns.some((turn) => turn.modeEffective === "learned"));
  assert.ok(learnedRoute.turns.some((turn) => turn.usedLearnedRouteFn === true));
  assert.ok(learnedRoute.summary.promotionCount >= 1);
});
