import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildRecordedSessionReplayFixture, runRecordedSessionReplay } from "../src/index.js";

function createWorkspace(t, label) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), `${label}-workspace-`));
  writeFileSync(path.join(rootDir, "README.md"), "# Seed carry-forward workspace\n", "utf8");
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-29T00:00:00.000Z",
    rootDir,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["real"],
  };
}

test("learned route carries forward seed cue blocks across the first promotion", (t) => {
  const workspace = createWorkspace(t, "seed-carry-forward");
  const trace = {
    contract: "recorded_session_trace.v1",
    traceId: "trace-seed-carry-forward",
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-29T00:00:00.000Z",
    bundleBuiltAt: "2026-03-29T00:10:00.000Z",
    agentId: "agent",
    sessionId: "session-seed-carry-forward",
    channel: "cli",
    sourceStream: "recorded/session",
    privacy: { sanitized: true, notes: ["real"] },
    workspace,
    seedBuiltAt: "2026-03-29T00:01:00.000Z",
    seedActivatedAt: "2026-03-29T00:02:00.000Z",
    seedCues: [
      {
        cueId: "cue-proof-run",
        createdAt: "2026-03-29T00:00:30.000Z",
        content: "T-20260329-048 is the real-trace learned-route proof run for OpenClawBrain.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: "turn-1",
        createdAt: "2026-03-29T00:03:00.000Z",
        deliveredAt: "2026-03-29T00:03:30.000Z",
        userMessage: "start the learned-route proof run",
        runtimeHints: ["proof", "run"],
        feedback: [
          {
            createdAt: "2026-03-29T00:03:45.000Z",
            content: "Next steps: commit the rollout evaluator scaffold and rerun the rollout evaluator.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["learned-route proof run"],
      },
      {
        turnId: "turn-2",
        createdAt: "2026-03-29T00:05:00.000Z",
        deliveredAt: "2026-03-29T00:05:30.000Z",
        userMessage: "what is the real-trace learned-route proof run?",
        runtimeHints: ["real-trace", "proof", "run"],
        expectedContextPhrases: ["real-trace learned-route proof run"],
      },
    ],
  };
  const fixture = buildRecordedSessionReplayFixture(trace);
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "seed-carry-forward-replay-"));
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });

  const bundle = runRecordedSessionReplay(rootDir, fixture);
  const learned = bundle.modes.find((mode) => mode.mode === "learned_route");
  assert.ok(learned);

  const replayedTurn = learned.turns[1];
  assert.equal(replayedTurn?.usedLearnedRouteFn, true);
  assert.equal(replayedTurn?.qualityScore, 100);
  assert.deepEqual(replayedTurn?.phraseHits, ["real-trace learned-route proof run"]);
  assert.ok(
    replayedTurn?.selectedContextTexts.some((text) => text.includes("real-trace learned-route proof run")),
    "expected the promoted learned pack to retain the seed cue phrase instead of dropping it on first promotion",
  );
});

test("learned route does not duplicate prior runtime-turn feedback into held-out eval context", (t) => {
  const workspace = createWorkspace(t, "seed-carry-forward-eval");
  const trace = {
    contract: "recorded_session_trace.v1",
    traceId: "trace-seed-carry-forward-eval-dedup",
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-29T00:00:00.000Z",
    bundleBuiltAt: "2026-03-29T00:10:00.000Z",
    agentId: "agent",
    sessionId: "session-seed-carry-forward-eval",
    channel: "telegram",
    sourceStream: "telegram/direct/live-proof-story",
    privacy: { sanitized: true, notes: ["real"] },
    workspace,
    evalTurnCount: 1,
    seedBuiltAt: "2026-03-29T00:01:00.000Z",
    seedActivatedAt: "2026-03-29T00:02:00.000Z",
    seedCues: [
      {
        cueId: "cue-empty-memory-scaffold",
        createdAt: "2026-03-29T00:00:30.000Z",
        content: "OpenClawBrain starts as a correctly attached but mostly empty memory scaffold.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: "story-turn-1",
        createdAt: "2026-03-29T00:03:00.000Z",
        deliveredAt: "2026-03-29T00:03:30.000Z",
        userMessage: "Use our current brain install to make it concrete.",
        runtimeHints: ["install", "story", "proof"],
        feedback: [
          {
            createdAt: "2026-03-29T00:03:45.000Z",
            content: "Use the current shared ~/.openclaw install as the canonical example.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["memory scaffold"],
      },
      {
        turnId: "story-turn-2",
        createdAt: "2026-03-29T00:05:00.000Z",
        deliveredAt: "2026-03-29T00:05:30.000Z",
        userMessage: "This is not good enough. Please make a deep detailed plan.",
        runtimeHints: ["rewrite", "proof-signals", "detail"],
        feedback: [
          {
            createdAt: "2026-03-29T00:05:45.000Z",
            content: "Ground the story in real host proof signals: BRAIN LOADED, loadProof=status_probe_ready, serveState=serving_active_pack, routeFn available=yes.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["memory scaffold"],
      },
      {
        turnId: "story-turn-3",
        createdAt: "2026-03-29T00:07:00.000Z",
        deliveredAt: "2026-03-29T00:07:30.000Z",
        userMessage: "Make it good and push it live now!",
        runtimeHints: ["publish", "live", "proof"],
        expectedContextPhrases: ["BRAIN LOADED", "routeFn available=yes"],
      },
    ],
  };
  const fixture = buildRecordedSessionReplayFixture(trace);
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "seed-carry-forward-eval-replay-"));
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });

  const bundle = runRecordedSessionReplay(rootDir, fixture);
  const learned = bundle.modes.find((mode) => mode.mode === "learned_route");
  assert.ok(learned);

  const evalTurn = learned.turns.find((turn) => turn.turnId === "story-turn-3");
  assert.equal(evalTurn?.usedLearnedRouteFn, true);
  assert.deepEqual(evalTurn?.phraseHits, ["BRAIN LOADED", "routeFn available=yes"]);
  assert.equal(
    evalTurn?.selectedContextTexts.filter((text) => text.includes("Use the current shared ~/.openclaw install")).length,
    1,
    "expected the held-out eval turn to avoid duplicate carried-forward runtime feedback",
  );
  assert.ok(
    evalTurn?.selectedContextTexts.some((text) => text.includes("BRAIN LOADED")),
    "expected the held-out eval turn to surface the newer host-proof teaching",
  );
});
