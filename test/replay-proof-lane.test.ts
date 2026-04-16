import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { RecordedSessionTraceV1 } from "../packages/cli/dist/src/index.js";
import { RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT, writeRecordedSessionReplayProofLane } from "../src/replay-proof-lane.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function createWorkspace(label: string): {
  workspaceId: string;
  snapshotId: string;
  capturedAt: string;
  rootDir: string;
  branch: string;
  revision: string;
  labels: string[];
} {
  const rootDir = makeTempDir(`${label}-workspace-`);
  writeFileSync(
    path.join(rootDir, "README.md"),
    `# ${label}\nThis workspace is part of the deterministic replay proof lane fixture.\n`,
    "utf8",
  );
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-28T17:26:18.111Z",
    rootDir,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["proof-lane"],
  };
}

function createComparativeTrace(): RecordedSessionTraceV1 {
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
    workspace: createWorkspace("comparative-replay"),
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

function createScoreResolutionTrace(): RecordedSessionTraceV1 {
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
    workspace: createWorkspace("score-resolution"),
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

function readText(root: string, relativePath: string): string {
  return readFileSync(path.join(root, relativePath), "utf8");
}

function normalizeReplayLaneVolatileText(text: string): string {
  return text
    .replace(
      /(success-adjusted economics: .*? estimated prompt USD, and )([0-9.]+)( ms serve-path latency per incremental win vs .*?, [0-9.]+, and )([0-9.]+)/g,
      "$1<volatile-latency>$3<volatile-latency>",
    )
    .replace(
      /(used [0-9.]+ estimated prompt tokens, [0-9.]+ estimated prompt USD, and )([0-9.]+)( ms serve-path latency per incremental win vs [^,]+ [0-9.]+, [0-9.]+, and )([0-9.]+)/g,
      "$1<volatile-latency>$3<volatile-latency>",
    )
    .replace(
      /(\| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| )([^|]+)( \| [^|]+ \| [^|]+ \| .* \|)/g,
      "$1<volatile-latency>$3",
    );
}

function scrubReplayLaneVolatileJson(value: unknown): unknown {
  if (typeof value === "string") {
    return normalizeReplayLaneVolatileText(value);
  }
  if (Array.isArray(value)) {
    return value.map((entry) => scrubReplayLaneVolatileJson(entry));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => {
        if ([
          "totalLatencyMs",
          "totalRouteSelectionLatencyMs",
          "totalPromptAssemblyLatencyMs",
          "candidateServePathLatencyMsPerSuccess",
          "baselineServePathLatencyMsPerSuccess",
          "servePathLatencyMsDeltaCandidateMinusBaseline",
          "totalMs",
          "routeSelectionMs",
          "promptAssemblyMs",
          "bundleHash",
        ].includes(key)) {
          return [key, "<volatile-latency>"];
        }
        return [key, scrubReplayLaneVolatileJson(entry)];
      }),
    );
  }
  return value;
}

function normalizeReplayLaneMarkdown(text: string): string {
  return normalizeReplayLaneVolatileText(text);
}

function readNormalizedReplayLaneArtifact(root: string, relativePath: string): string {
  const text = readText(root, relativePath);
  if (relativePath.endsWith(".json")) {
    return JSON.stringify(scrubReplayLaneVolatileJson(JSON.parse(text)), null, 2);
  }
  if (relativePath.endsWith(".md")) {
    return normalizeReplayLaneMarkdown(text);
  }
  return text;
}

describe("recorded session replay proof lane", () => {
  it("writes stable aggregate artifacts under _lane with pairwise deltas and win-rate matrices", () => {
    const artifactRoot = makeTempDir("replay-proof-lane-");
    const descriptor = writeRecordedSessionReplayProofLane({
      artifactRoot,
      traces: [
        { trace: createComparativeTrace(), tracePath: "/fixtures/comparative.json" },
        { trace: createScoreResolutionTrace(), tracePath: "/fixtures/score-resolution.json" },
      ],
      assumptions: ["canonical frozen set is equivalent-only, not first-party real-trace-backed"],
    });

    expect(descriptor.index.requestedTraceCount).toBe(2);
    expect(descriptor.index.successfulTraceCount).toBe(2);
    expect(descriptor.index.failedTraceCount).toBe(0);
    expect(descriptor.summaryTables.traces.map((row) => row.traceId)).toEqual([
      "trace-comparative-replay",
      "trace-score-resolution",
    ]);
    const learnedMode = descriptor.summaryTables.modes.find((row) => row.mode === "learned_route");
    const graphMode = descriptor.summaryTables.modes.find((row) => row.mode === "graph_prior_only");
    expect(learnedMode?.rankedWinnerCount).toBe(1);
    expect(learnedMode?.sharedTopScoreTraceCount).toBe(2);
    expect(graphMode?.rankedWinnerCount).toBe(1);
    expect(graphMode?.sharedTopScoreTraceCount).toBe(1);
    expect(descriptor.summaryTables.scorecard.traceTieOrBetterVsBaseline).toMatchObject({
      count: 2,
      rate: 1,
      totalCount: 2,
    });
    expect(descriptor.summaryTables.scorecard.regressionVsFloor).toMatchObject({
      count: 0,
      rate: 0,
      totalCount: 2,
    });
    expect(descriptor.summaryTables.scorecard.activationPrecision).toMatchObject({
      available: true,
      observedTurnCount: 5,
      activationCount: 5,
      beneficialActivationCount: 1,
      precision: expect.any(Number),
    });
    expect(descriptor.summaryTables.scorecard.successAdjustedEconomics).toMatchObject({
      available: true,
      candidateEstimatedPromptCostUsdPerSuccess: expect.any(Number),
      baselineEstimatedPromptCostUsdPerSuccess: expect.any(Number),
      candidateServePathLatencyMsPerSuccess: expect.any(Number),
      baselineServePathLatencyMsPerSuccess: expect.any(Number),
    });

    const learnedVsGraph = descriptor.pairwiseDeltas.pairs.find(
      (row) => row.leftMode === "graph_prior_only" && row.rightMode === "learned_route",
    );
    expect(learnedVsGraph).toBeDefined();
    expect(learnedVsGraph?.traceWins.left).toBe(0);
    expect(learnedVsGraph?.traceWins.right).toBe(1);
    expect(learnedVsGraph?.traceWins.ties).toBe(1);

    const traceMatrixRow = descriptor.winRateMatrix.traceMatrix.find((row) => row.mode === "learned_route");
    const versusGraph = traceMatrixRow?.cells.find((cell) => cell.mode === "graph_prior_only");
    expect(versusGraph?.wins).toBe(1);
    expect(versusGraph?.losses).toBe(0);
    expect(versusGraph?.ties).toBe(1);
    expect(versusGraph?.winRate).toBe(0.5);
    expect(versusGraph?.tieRate).toBe(0.5);

    const laneRoot = path.join(artifactRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.laneDir);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/Explainable Scorecard/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/diagnostic top-rank/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/Diagnostic Pairwise Deltas/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/internal deterministic replay diagnostics/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/trace-score-resolution/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/learned_route vs approved prior/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/Please do this entire plan end to end/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/trace-comparative-replay/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/diagnostic winner/);
  });

  it("keeps the core _lane artifacts reproducible across different output roots", () => {
    const firstRoot = makeTempDir("replay-proof-lane-first-");
    const secondRoot = makeTempDir("replay-proof-lane-second-");
    const comparativeTrace = createComparativeTrace();
    const scoreResolutionTrace = createScoreResolutionTrace();
    const first = writeRecordedSessionReplayProofLane({
      artifactRoot: firstRoot,
      traces: [
        { trace: structuredClone(comparativeTrace), tracePath: "/fixtures/comparative.json" },
        { trace: structuredClone(scoreResolutionTrace), tracePath: "/fixtures/score-resolution.json" },
      ],
      assumptions: ["canonical frozen set is equivalent-only, not first-party real-trace-backed"],
    });
    const second = writeRecordedSessionReplayProofLane({
      artifactRoot: secondRoot,
      traces: [
        { trace: structuredClone(comparativeTrace), tracePath: "/fixtures/comparative.json" },
        { trace: structuredClone(scoreResolutionTrace), tracePath: "/fixtures/score-resolution.json" },
      ],
      assumptions: ["canonical frozen set is equivalent-only, not first-party real-trace-backed"],
    });

    const stableFiles = [
      RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme,
      RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.index,
      RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summaryTables,
      RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.pairwiseDeltas,
      RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.winRateMatrix,
      RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces,
    ];
    for (const relativePath of stableFiles) {
      expect(readNormalizedReplayLaneArtifact(first.laneDir, relativePath)).toBe(
        readNormalizedReplayLaneArtifact(second.laneDir, relativePath),
      );
    }
  });
});
