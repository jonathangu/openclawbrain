import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  buildRecordedSessionReplayFixture,
  runRecordedSessionReplay,
  validateRecordedSessionReplayProofBundle,
  writeRecordedSessionReplayProofBundle,
} from "../src/index.js";

function createWorkspace(t, label) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), `${label}-workspace-`));
  writeFileSync(
    path.join(rootDir, "README.md"),
    "# Score resolution workspace\nReplay scoring should preserve multi-phrase turn weight.\n",
    "utf8",
  );
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-28T17:26:18.111Z",
    rootDir,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["proof-plan"],
  };
}

function createResolutionTrace(t) {
  return {
    contract: "recorded_session_trace.v1",
    traceId: "trace-score-resolution",
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-28T17:26:18.111Z",
    bundleBuiltAt: "2026-03-28T20:02:41.227Z",
    agentId: "main",
    sessionId: "sanitized-session-score-resolution",
    channel: "telegram",
    sourceStream: "telegram/direct/proof-plan",
    privacy: {
      sanitized: true,
      notes: ["sanitized recorded session"],
    },
    workspace: createWorkspace(t, "score-resolution"),
    evalTurnCount: 1,
    seedBuiltAt: "2026-03-28T17:26:18.111Z",
    seedActivatedAt: "2026-03-28T17:27:18.111Z",
    seedCues: [
      {
        cueId: "cue-proof-run",
        createdAt: "2026-03-28T17:26:18.111Z",
        content: "T-20260328-040 is the real-trace learned-route proof run for OpenClawBrain.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: "plan-turn-1",
        createdAt: "2026-03-28T19:54:42.389Z",
        deliveredAt: "2026-03-28T19:54:57.546Z",
        userMessage: "So what's next for us to work on?",
        runtimeHints: ["proof", "next", "plan"],
        feedback: [
          {
            createdAt: "2026-03-28T19:54:57.546Z",
            content: "Next steps: commit the rollout-evaluator scaffold, export a sanitized non-test real-trace replay corpus, and rerun the rollout evaluator.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["learned-route proof run"],
      },
      {
        turnId: "plan-turn-2",
        createdAt: "2026-03-28T19:56:06.301Z",
        deliveredAt: "2026-03-28T19:58:48.383Z",
        userMessage: "Please make an end-to-end master plan for me.",
        runtimeHints: ["master-plan", "proof", "roadmap"],
        feedback: [
          {
            createdAt: "2026-03-28T19:58:48.383Z",
            content: "The master plan ends with a rollout verdict: ready, limited, or blocked.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["real-trace learned-route proof run"],
      },
      {
        turnId: "plan-turn-3",
        createdAt: "2026-03-28T20:01:14.399Z",
        deliveredAt: "2026-03-28T20:01:41.227Z",
        userMessage: "Please do this entire plan end to end.",
        runtimeHints: ["execute", "proof", "end-to-end"],
        expectedContextPhrases: [
          "sanitized non-test real-trace replay corpus",
          "ready, limited, or blocked",
        ],
      },
    ],
  };
}

function findMode(bundle, mode) {
  const report = bundle.modes.find((candidate) => candidate.mode === mode);
  assert.ok(report, `missing mode ${mode}`);
  return report;
}

function findTurn(modeReport, turnId) {
  const turn = modeReport.turns.find((candidate) => candidate.turnId === turnId);
  assert.ok(turn, `missing turn ${turnId}`);
  return turn;
}

test("replay mode quality preserves aggregate phrase coverage across multi-phrase turns", (t) => {
  const trace = createResolutionTrace(t);
  const fixture = buildRecordedSessionReplayFixture(trace);
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "replay-score-resolution-root-"));
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });

  const bundle = runRecordedSessionReplay(rootDir, fixture);
  const vectorOnly = findMode(bundle, "vector_only");
  const graphPriorOnly = findMode(bundle, "graph_prior_only");
  const learnedRoute = findMode(bundle, "learned_route");

  assert.equal(vectorOnly.summary.phraseHitCount, 2);
  assert.equal(graphPriorOnly.summary.phraseHitCount, 2);
  assert.equal(learnedRoute.summary.phraseHitCount, 4);
  assert.equal(vectorOnly.summary.qualityScore, 70);
  assert.equal(graphPriorOnly.summary.qualityScore, 70);
  assert.equal(learnedRoute.summary.qualityScore, 100);
  assert.equal(bundle.summary.winnerMode, "learned_route");

  const vectorTurn3 = findTurn(vectorOnly, "plan-turn-3");
  const learnedTurn3 = findTurn(learnedRoute, "plan-turn-3");
  assert.deepEqual(vectorTurn3.phraseHits, []);
  assert.deepEqual(learnedTurn3.phraseHits, [
    "sanitized non-test real-trace replay corpus",
    "ready, limited, or blocked",
  ]);
  assert.notEqual(learnedTurn3.selectionDigest, vectorTurn3.selectionDigest);
});

test("proof bundle summaries keep the sharper learned-route winner", (t) => {
  const trace = createResolutionTrace(t);
  const tempRoot = mkdtempSync(path.join(os.tmpdir(), "replay-score-resolution-proof-"));
  const bundleRoot = path.join(tempRoot, "bundle");
  t.after(() => {
    rmSync(tempRoot, { recursive: true, force: true });
  });

  const descriptor = writeRecordedSessionReplayProofBundle({
    rootDir: bundleRoot,
    trace,
    scratchRootDir: tempRoot,
  });
  const validation = validateRecordedSessionReplayProofBundle(bundleRoot);
  const learnedRow = descriptor.summaryTables.modes.find((row) => row.mode === "learned_route");
  const vectorRow = descriptor.summaryTables.modes.find((row) => row.mode === "vector_only");

  assert.equal(validation.ok, true);
  assert.equal(descriptor.summaryTables.winnerMode, "learned_route");
  assert.deepEqual(descriptor.summaryTables.ranking.slice(0, 3), [
    { mode: "learned_route", qualityScore: 100 },
    { mode: "graph_prior_only", qualityScore: 70 },
    { mode: "vector_only", qualityScore: 70 },
  ]);
  assert.equal(learnedRow?.qualityScore, 100);
  assert.equal(learnedRow?.phraseHitCount, 4);
  assert.equal(vectorRow?.qualityScore, 70);
  assert.equal(vectorRow?.phraseHitCount, 2);
  assert.match(descriptor.summaryText, /winner mode: `learned_route`/);
});
