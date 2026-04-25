import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  loadRecordedSessionReplayProofBundle,
  type RecordedSessionTraceV1,
} from "../packages/cli/dist/src/index.js";
import type { DataRegistryEntryV1, RouteDecisionRowV1 } from "../src/brain-core/cold-start-router-contracts.js";
import { trainColdStartRouterArtifactV1 } from "../src/brain-core/cold-start-router-trainer.js";
import {
  buildReplayLearnedRouteDecisionRowV1,
  RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT,
  writeRecordedSessionReplayProofLane,
} from "../src/replay-proof-lane.js";

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

function createReplayOverrideRegistryEntry(datasetId: string): DataRegistryEntryV1 {
  return {
    dataset_id: datasetId,
    source_family: "agent_traces",
    upstream_url: "https://example.org/replay-override",
    original_creator: "OpenClaw",
    license: "internal_local_only",
    commercial_use_status: "allowed",
    redistribution_status: "allowed",
    pii_risk: "none",
    benchmark_split_status: "train",
    approval_status: "approved_train",
    reviewer: "operator",
    immutable_snapshot_ref: `snapshot:${datasetId}@sha256:replay-override`,
    exact_files: ["recorded-session-replay.json"],
    file_hashes: {
      "recorded-session-replay.json": "sha256:replay-override",
    },
    allowed_uses: ["replay proof override"],
    disallowed_uses: ["redistribution"],
    notes: ["replay override test fixture"],
    created_at: "2026-04-17T00:00:00.000Z",
    updated_at: "2026-04-17T00:00:00.000Z",
  };
}

function createReplayStopRouteRows(datasetId: string): RouteDecisionRowV1[] {
  return [
    {
      row_id: "replay-override-stop-1",
      dataset_id: datasetId,
      query: "Stay stopped during replay candidate override verification.",
      cursor_path: ["recorded_session_replay"],
      candidate_set: [
        { candidate_id: "candidate:a", candidate_type: "graph_node", score_hint: 0.3 },
        { candidate_id: "candidate:b", candidate_type: "graph_node", score_hint: 0.2 },
        { candidate_id: "candidate:c", candidate_type: "graph_node", score_hint: 0.1 },
        { candidate_id: "candidate:d", candidate_type: "graph_node", score_hint: 0.05 },
        { candidate_id: "candidate:e", candidate_type: "graph_node", score_hint: 0.01 },
      ],
      teacher_action: { kind: "tool", tool_name: "__recorded_session_replay_candidate_override__" },
      stop_label: "STOP_LOCAL",
      evidence_spans: [
        { source_ref: "replay:evidence:0", start: 0, end: 19, excerpt: "Stay stopped." },
        { source_ref: "replay:evidence:1", start: 0, end: 21, excerpt: "Replay verification." },
        { source_ref: "replay:evidence:2", start: 0, end: 22, excerpt: "Candidate override." },
      ],
      hard_negatives: ["candidate:b"],
      outcome_gain: 1,
      provenance: {
        dataset: datasetId,
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: `snapshot:${datasetId}@sha256:replay-override`,
        recorded_by: "test",
        recorded_at: "2026-04-17T00:00:00.000Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-17T00:00:00.000Z",
    },
    {
      row_id: "replay-override-stop-2",
      dataset_id: datasetId,
      query: "Replay candidate override should abstain instead of traversing.",
      cursor_path: ["recorded_session_replay"],
      candidate_set: [
        { candidate_id: "candidate:f", candidate_type: "graph_node", score_hint: 0.28 },
        { candidate_id: "candidate:g", candidate_type: "graph_node", score_hint: 0.18 },
        { candidate_id: "candidate:h", candidate_type: "graph_node", score_hint: 0.08 },
        { candidate_id: "candidate:i", candidate_type: "graph_node", score_hint: 0.04 },
        { candidate_id: "candidate:j", candidate_type: "graph_node", score_hint: 0.02 },
      ],
      teacher_action: { kind: "tool", tool_name: "__recorded_session_replay_candidate_override__" },
      stop_label: "STOP_LOCAL",
      evidence_spans: [
        { source_ref: "replay:evidence:3", start: 0, end: 24, excerpt: "Replay abstention." },
        { source_ref: "replay:evidence:4", start: 0, end: 24, excerpt: "Broad live override." },
        { source_ref: "replay:evidence:5", start: 0, end: 18, excerpt: "Stay local." },
      ],
      hard_negatives: ["candidate:g"],
      outcome_gain: 1,
      provenance: {
        dataset: datasetId,
        source_license: "internal_local_only",
        source_family: "agent_traces",
        source_snapshot_ref: `snapshot:${datasetId}@sha256:replay-override`,
        recorded_by: "test",
        recorded_at: "2026-04-17T00:01:00.000Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-17T00:01:00.000Z",
    },
  ];
}

function trainReplayStopArtifact(outputDir: string, routerIdentity: string): void {
  const datasetId = `dataset:${routerIdentity}`;
  trainColdStartRouterArtifactV1({
    artifactId: `artifact:${routerIdentity}`,
    artifactVersion: "0.0.1",
    packType: "base",
    compatibleRuntimeVersion: "openclawbrain-runtime@0.4.44",
    registryEntries: [createReplayOverrideRegistryEntry(datasetId)],
    routeRows: createReplayStopRouteRows(datasetId),
    outputDir,
    routerIdentity,
    createdAt: "2026-04-17T00:10:00.000Z",
    trainingDataRefs: [datasetId],
    replayGateRefs: [`replay:${datasetId}`],
  });
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
  it("drops raw replay ranking score hints from override rows", () => {
    const input = {
      request: {
        userMessage: "continue the interrupted task",
      },
      ranked: [
        {
          blockId: "pack:event:alpha:feedback",
          source: "graph",
          text: "event alpha feedback",
          score: 98.4,
          channelScores: { graph: 1, shortTerm: 0, vector: 0 },
          routingChannels: ["graph"],
          priority: 0,
          matchedTokens: ["task"],
          tokenCount: 64,
          packOrder: 0,
          candidateSemanticClass: "answer_bearing",
          candidateSemanticEvidence: [],
        },
        {
          blockId: "pack:event:alpha:interaction",
          source: "graph",
          text: "event alpha interaction",
          score: 92.1,
          channelScores: { graph: 1, shortTerm: 0, vector: 0 },
          routingChannels: ["graph"],
          priority: 1,
          matchedTokens: ["task"],
          tokenCount: 56,
          packOrder: 0,
          candidateSemanticClass: "answer_bearing",
          candidateSemanticEvidence: [],
        },
        {
          blockId: "pack:event:alpha",
          source: "graph",
          text: "event alpha",
          score: 88.4,
          channelScores: { graph: 1, shortTerm: 0, vector: 0 },
          routingChannels: ["graph"],
          priority: 1,
          matchedTokens: ["task"],
          tokenCount: 48,
          packOrder: 0,
          candidateSemanticClass: "answer_bearing",
          candidateSemanticEvidence: [],
        },
        {
          blockId: "pack:pointer-aware-init",
          source: "graph",
          text: "pointer aware init",
          score: 288.7,
          channelScores: { graph: 1, shortTerm: 0, vector: 0 },
          routingChannels: ["graph"],
          priority: 2,
          matchedTokens: ["task"],
          tokenCount: 48,
          packOrder: 0,
          candidateSemanticClass: "answer_bearing",
          candidateSemanticEvidence: [],
        },
      ],
      maxBlocks: 1,
    } as Parameters<typeof buildReplayLearnedRouteDecisionRowV1>[0]["input"];

    const row = buildReplayLearnedRouteDecisionRowV1({
      artifactDatasetId: "dataset:test-replay-override-row",
      input,
    });

    expect(row).not.toBeNull();
    expect(row?.candidate_set).toHaveLength(4);
    expect(row?.candidate_set[0]).toMatchObject({
      candidate_id: "pack:event:alpha:feedback",
      semantic_class: "feedback_context",
      token_cost: 64,
      score_hint: 0.95,
    });
    expect(row?.candidate_set[1]).toMatchObject({
      candidate_id: "pack:event:alpha:interaction",
      semantic_class: "interaction_context",
      token_cost: 56,
      score_hint: 0.55,
    });
    expect(row?.candidate_set[2]).toMatchObject({
      candidate_id: "pack:event:alpha",
      semantic_class: "event_context",
      token_cost: 48,
      score_hint: 0.9,
    });
    expect(row?.candidate_set[3]).toMatchObject({
      candidate_id: "pack:pointer-aware-init",
      semantic_class: "init_context",
      token_cost: 48,
      score_hint: 0.2,
    });
  });

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
    expect(descriptor.summaryTables.scorecard.optimizeOver).toMatchObject({
      candidateMode: "learned_route",
      baselineMode: "graph_prior_only",
      objective: "maximize_learned_route_value_vs_graph_prior_only",
      traceDenominator: {
        requestedTraceCount: 2,
        successfulTraceCount: 2,
        failedTraceCount: 0,
        comparableTraceCount: 2,
        comparableTraceCoverageRate: 1,
      },
      turnDenominator: {
        comparableTurnCount: 5,
      },
      labels: {
        beneficial: "learned_route_better_than_graph_prior_only",
        neutral: "learned_route_tied_graph_prior_only",
        regression: "learned_route_worse_than_graph_prior_only",
      },
      traceCounts: {
        beneficial: 1,
        neutral: 1,
        regression: 0,
      },
    });
    expect(descriptor.summaryTables.scorecard.activationPrecision).toMatchObject({
      available: true,
      observedTurnCount: 5,
      activationCount: 5,
      beneficialActivationCount: 1,
      precision: expect.any(Number),
    });
    expect(descriptor.summaryTables.scorecard.activationUsefulness).toMatchObject({
      available: true,
      observedTurnCount: 5,
      firedTurnCount: 5,
      shouldHaveFiredTurnCount: 1,
      uniqueBeneficialWinCount: 1,
      harmfulActivationCount: 0,
      neutralActivationCount: 4,
      noOpTieCount: 4,
      missedBeneficialOpportunityCount: 0,
      promptTokenDeltaCandidateMinusBaseline: 93,
      labels: {
        beneficial: "fired_and_learned_route_better_than_graph_prior_only",
        harmful: "fired_and_learned_route_worse_than_graph_prior_only",
        neutral: "fired_and_learned_route_tied_graph_prior_only",
        missedBeneficialOpportunity: "did_not_fire_but_learned_route_better_than_graph_prior_only",
      },
    });
    const beneficialTurn = descriptor.summaryTables.turns.find((turn) => turn.turnId === "plan-turn-3");
    expect(beneficialTurn?.activationUsefulness).toMatchObject({
      didLearnedRoutingFire: true,
      shouldHaveFired: true,
      usefulness: "beneficial",
      relationVsBaseline: "better",
      costDelta: {
        promptTokensCandidateMinusBaseline: expect.any(Number),
        contextCharsCandidateMinusBaseline: expect.any(Number),
      },
    });
    const tieTurns = descriptor.summaryTables.turns.filter((turn) => turn.activationUsefulness.usefulness === "neutral");
    expect(tieTurns).toHaveLength(4);
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
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/optimize-over objective: maximize_learned_route_value_vs_graph_prior_only/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/activation usefulness: 1 unique beneficial win\(s\), 0 harmful activation\(s\), 4 neutral activation tie\(s\)/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.summary)).toMatch(/optimize-over labels: beneficial=learned_route_better_than_graph_prior_only/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/diagnostic top-rank/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/Diagnostic Pairwise Deltas/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.readme)).toMatch(/internal deterministic replay diagnostics/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/trace-score-resolution/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/learned_route vs approved prior/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/Please do this entire plan end to end/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/trace-comparative-replay/);
    expect(readText(laneRoot, RECORDED_SESSION_REPLAY_PROOF_LANE_LAYOUT.workedTraces)).toMatch(/diagnostic winner/);
  });

  it("uses the supplied learned-route candidate artifact instead of replay-trained selection", () => {
    const trace = createComparativeTrace();
    const baselineRoot = makeTempDir("replay-proof-lane-baseline-");
    const overrideRoot = makeTempDir("replay-proof-lane-override-");
    const candidateArtifactDir = makeTempDir("replay-proof-lane-candidate-");
    const candidateRouterIdentity = "router:replay-proof-lane-stop";

    trainReplayStopArtifact(candidateArtifactDir, candidateRouterIdentity);

    writeRecordedSessionReplayProofLane({
      artifactRoot: baselineRoot,
      traces: [{ trace: structuredClone(trace) }],
    });
    writeRecordedSessionReplayProofLane({
      artifactRoot: overrideRoot,
      traces: [{ trace: structuredClone(trace) }],
      learnedRouteCandidateArtifact: {
        artifactDir: candidateArtifactDir,
      },
    });

    const baselineBundle = loadRecordedSessionReplayProofBundle(path.join(baselineRoot, trace.traceId));
    const overrideBundle = loadRecordedSessionReplayProofBundle(path.join(overrideRoot, trace.traceId));
    const baselineLearnedRoute = baselineBundle.bundle.modes.find((mode) => mode.mode === "learned_route");
    const baselineGraphPrior = baselineBundle.bundle.modes.find((mode) => mode.mode === "graph_prior_only");
    const overrideLearnedRoute = overrideBundle.bundle.modes.find((mode) => mode.mode === "learned_route");

    expect(baselineLearnedRoute).toBeDefined();
    expect(baselineGraphPrior).toBeDefined();
    expect(overrideLearnedRoute).toBeDefined();
    expect(baselineLearnedRoute?.turns.some((turn) => turn.selectedContextIds.length > 0)).toBe(true);
    expect(overrideLearnedRoute?.summary.usedLearnedRouteTurnCount).toBe(0);
    expect(overrideLearnedRoute?.turns.every((turn) => turn.usedLearnedRouteFn === false)).toBe(true);
    expect(overrideLearnedRoute?.turns.every((turn) => turn.activationTaken === true)).toBe(true);
    expect(overrideLearnedRoute?.turns.every((turn) => turn.routerIdentity === candidateRouterIdentity)).toBe(true);
    expect(overrideLearnedRoute?.turns.map((turn) => turn.selectedContextIds)).toEqual([[], []]);
    expect(overrideLearnedRoute?.turns.every((turn) => turn.activationSource?.startsWith("learned_route_artifact:candidate_override:"))).toBe(true);
    expect(overrideLearnedRoute?.turns.map((turn) => turn.activePackId)).toEqual(
      baselineGraphPrior?.turns.map((turn) => turn.activePackId),
    );
  });

  it("records a negative served-live-policy spike result explicitly when learned-route usage still stays false", () => {
    const trace = createComparativeTrace();
    const spikeRoot = makeTempDir("replay-proof-lane-spike-");
    const candidateArtifactDir = makeTempDir("replay-proof-lane-spike-candidate-");
    const candidateRouterIdentity = "router:replay-proof-lane-spike-stop";

    trainReplayStopArtifact(candidateArtifactDir, candidateRouterIdentity);

    writeRecordedSessionReplayProofLane({
      artifactRoot: spikeRoot,
      traces: [{ trace: structuredClone(trace) }],
      learnedRouteCandidateArtifact: {
        artifactDir: candidateArtifactDir,
        mode: "served_live_policy_spike",
      },
    });

    const spikeBundle = loadRecordedSessionReplayProofBundle(path.join(spikeRoot, trace.traceId));
    const spikeLearnedRoute = spikeBundle.bundle.modes.find((mode) => mode.mode === "learned_route");

    expect(spikeLearnedRoute).toBeDefined();
    expect(spikeLearnedRoute?.summary.usedLearnedRouteTurnCount).toBe(0);
    expect(spikeLearnedRoute?.turns.every((turn) => turn.usedLearnedRouteFn === false)).toBe(true);
    expect(spikeLearnedRoute?.turns.every((turn) => turn.activationTaken === true)).toBe(true);
    expect(spikeLearnedRoute?.turns.every((turn) => turn.routerIdentity === candidateRouterIdentity)).toBe(true);
    expect(spikeLearnedRoute?.turns.map((turn) => turn.selectedContextIds)).toEqual([[], []]);
    expect(spikeLearnedRoute?.turns.every((turn) => turn.activationSource?.startsWith("learned_route_artifact:candidate_override_live_policy:"))).toBe(true);
    expect(spikeLearnedRoute?.turns.every((turn) => turn.activationReason?.includes("usedLearnedRouteFn=false"))).toBe(true);
  });

  it("can adapt a cold-start candidate artifact onto the served pack path for authoritative replay", () => {
    const trace = createComparativeTrace();
    const adapterRoot = makeTempDir("replay-proof-lane-served-pack-adapter-");
    const candidateArtifactDir = makeTempDir("replay-proof-lane-served-pack-candidate-");
    const candidateRouterIdentity = "router:replay-proof-lane-served-pack-stop";

    trainReplayStopArtifact(candidateArtifactDir, candidateRouterIdentity);

    writeRecordedSessionReplayProofLane({
      artifactRoot: adapterRoot,
      traces: [{ trace: structuredClone(trace) }],
      learnedRouteCandidateArtifact: {
        artifactDir: candidateArtifactDir,
        mode: "served_pack_adapter",
      },
    });

    const adapterBundle = loadRecordedSessionReplayProofBundle(path.join(adapterRoot, trace.traceId));
    const adapterLearnedRoute = adapterBundle.bundle.modes.find((mode) => mode.mode === "learned_route");

    expect(adapterLearnedRoute).toBeDefined();
    expect(adapterLearnedRoute?.summary.usedLearnedRouteTurnCount).toBe(2);
    expect(adapterLearnedRoute?.turns.every((turn) => turn.usedLearnedRouteFn === true)).toBe(true);
    expect(adapterLearnedRoute?.turns.every((turn) => turn.routerIdentity === candidateRouterIdentity)).toBe(true);
    expect(adapterLearnedRoute?.turns.every((turn) => turn.activationSource === "learned_route_fn")).toBe(true);
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
