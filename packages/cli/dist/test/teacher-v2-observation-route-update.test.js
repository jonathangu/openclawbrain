import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { buildNormalizedEventExport } from "@openclawbrain/event-export";
import { buildCandidatePackFromNormalizedEventExport } from "../src/local-learner.js";
import { CONTRACT_IDS, ROUTER_PG_PROFILE_V2 } from "@openclawbrain/contracts";

function createWorkspace(t) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "ocb-teacher-v2-"));
  writeFileSync(path.join(rootDir, "README.md"), "# Teacher V2 Workspace\nObservation-backed learning lives here.\n", "utf8");
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: "ws-teacher-v2",
    snapshotId: "snapshot-teacher-v2",
    capturedAt: "2026-03-23T17:00:00.000Z",
    rootDir,
    branch: "main",
    revision: "teacher-v2",
    dirty: false,
    labels: ["test"],
    files: ["README.md"]
  };
}

function createActivationRoot(t) {
  const activationRoot = mkdtempSync(path.join(os.tmpdir(), "ocb-activation-"));
  t.after(() => {
    rmSync(activationRoot, { recursive: true, force: true });
  });
  return activationRoot;
}

function seedObservationDb(activationRoot, observation, options = {}) {
  const db = new DatabaseSync(path.join(activationRoot, "state.db"));
  try {
    const routeMetadata = {
      ...(observation.serveDecisionRecordId !== undefined ? { serveDecisionRecordId: observation.serveDecisionRecordId } : {}),
      ...(observation.selectionDigest !== undefined ? { selectionDigest: observation.selectionDigest } : {}),
      ...(observation.turnCompileEventId !== undefined ? { turnCompileEventId: observation.turnCompileEventId } : {}),
      ...(observation.activePackId !== undefined ? { activePackId: observation.activePackId } : {}),
      ...(observation.activePackGraphChecksum !== undefined ? { activePackGraphChecksum: observation.activePackGraphChecksum } : {}),
      selectedNodeIds: observation.selectedNodeIds,
      selectedPathNodeIds: observation.selectedPathNodeIds
    };
    const legacySchema = options.legacySchema === true;
    db.exec(legacySchema ? `
      CREATE TABLE IF NOT EXISTS brain_observations (
        id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL UNIQUE,
        conversation_id INTEGER,
        trace_id TEXT,
        query_text TEXT NOT NULL,
        retrieved_context_json TEXT NOT NULL DEFAULT '[]',
        route_metadata_json TEXT NOT NULL DEFAULT '{}',
        assistant_response TEXT NOT NULL DEFAULT '',
        tool_results_json TEXT NOT NULL DEFAULT '[]',
        follow_up_text TEXT,
        phase1_score REAL,
        phase2_score REAL,
        final_score REAL,
        confidence REAL,
        reason TEXT,
        status TEXT NOT NULL DEFAULT 'pending_followup',
        teacher_evaluation_json TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        evaluated_at INTEGER
      );
    ` : `
      CREATE TABLE IF NOT EXISTS brain_observations (
        id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL UNIQUE,
        conversation_id INTEGER,
        trace_id TEXT,
        serve_decision_record_id TEXT,
        selection_digest TEXT,
        turn_compile_event_id TEXT,
        active_pack_id TEXT,
        active_pack_graph_checksum TEXT,
        query_text TEXT NOT NULL,
        retrieved_context_json TEXT NOT NULL DEFAULT '[]',
        route_metadata_json TEXT NOT NULL DEFAULT '{}',
        assistant_response TEXT NOT NULL DEFAULT '',
        tool_results_json TEXT NOT NULL DEFAULT '[]',
        follow_up_text TEXT,
        phase1_score REAL,
        phase2_score REAL,
        final_score REAL,
        confidence REAL,
        reason TEXT,
        status TEXT NOT NULL DEFAULT 'pending_followup',
        teacher_evaluation_json TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        evaluated_at INTEGER
      );
    `);
    if (legacySchema) {
      db.prepare(`
      INSERT INTO brain_observations (
        id,
        episode_id,
        conversation_id,
        trace_id,
        query_text,
        retrieved_context_json,
        route_metadata_json,
        assistant_response,
        tool_results_json,
        follow_up_text,
        phase1_score,
        phase2_score,
        final_score,
        confidence,
        reason,
        status,
        teacher_evaluation_json,
        created_at,
        updated_at,
        evaluated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
        observation.id,
        observation.episodeId,
        null,
        null,
        observation.queryText,
        "[]",
        JSON.stringify(routeMetadata),
        "Here is the observed answer.",
        "[]",
        "That helped.",
        0.9,
        0.8,
        observation.finalScore,
        0.9,
        "teacher_v2_test",
        "completed",
        JSON.stringify({ finalScore: observation.finalScore, confidence: 0.9 }),
        observation.createdAt,
        observation.createdAt,
        observation.createdAt + 60_000
      );
      return;
    }
    db.prepare(`
      INSERT INTO brain_observations (
        id,
        episode_id,
        conversation_id,
        trace_id,
        serve_decision_record_id,
        selection_digest,
        turn_compile_event_id,
        active_pack_id,
        active_pack_graph_checksum,
        query_text,
        retrieved_context_json,
        route_metadata_json,
        assistant_response,
        tool_results_json,
        follow_up_text,
        phase1_score,
        phase2_score,
        final_score,
        confidence,
        reason,
        status,
        teacher_evaluation_json,
        created_at,
        updated_at,
        evaluated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      observation.id,
      observation.episodeId,
      null,
      null,
      observation.serveDecisionRecordId ?? null,
      observation.selectionDigest ?? null,
      observation.turnCompileEventId ?? null,
      observation.activePackId ?? null,
      observation.activePackGraphChecksum ?? null,
      observation.queryText,
      "[]",
      JSON.stringify(routeMetadata),
      "Here is the observed answer.",
      "[]",
      "That helped.",
      0.9,
      0.8,
      observation.finalScore,
      0.9,
      "teacher_v2_test",
      "completed",
      JSON.stringify({ finalScore: observation.finalScore, confidence: 0.9 }),
      observation.createdAt,
      observation.createdAt,
      observation.createdAt + 60_000
    );
  } finally {
    db.close();
  }
}

test("native V2 can learn from teacher-v2 observation outcomes in activation state.db", (t) => {
  const workspace = createWorkspace(t);
  const activationRoot = createActivationRoot(t);
  const interactionEvents = [
    {
      contract: CONTRACT_IDS.interactionEvents,
      eventId: "evt-i1",
      agentId: "agent",
      sessionId: "sess-1",
      channel: "cli",
      sequence: 1,
      kind: "memory_compiled",
      createdAt: "2026-03-23T17:00:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "session.tail" },
      messageId: "msg-1"
    },
    {
      contract: CONTRACT_IDS.interactionEvents,
      eventId: "evt-i2",
      agentId: "agent",
      sessionId: "sess-1",
      channel: "cli",
      sequence: 2,
      kind: "memory_compiled",
      createdAt: "2026-03-23T17:00:01.000Z",
      source: { runtimeOwner: "openclaw", stream: "session.tail" },
      messageId: "msg-2"
    }
  ];
  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents: []
  });
  const structuralOps = { connect: 3, split: 1, merge: 1, prune: 1 };
  const servedPack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-served",
    workspace,
    normalizedEventExport,
    learnedRouting: false,
    builtAt: "2026-03-23T17:01:00.000Z",
    structuralOps
  });
  const packId = servedPack.summary.packId;
  const chosenContextIds = [`${packId}:event:evt-i2`, `${packId}:merge:1`];
  seedObservationDb(activationRoot, {
    id: "bo_test_1",
    episodeId: "ep_test_1",
    queryText: "show me the learned routing context",
    serveDecisionRecordId: "decision-1",
    selectionDigest: "selection-1",
    turnCompileEventId: "evt-i1",
    activePackId: packId,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    selectedNodeIds: chosenContextIds,
    selectedPathNodeIds: chosenContextIds,
    finalScore: 0.8,
    createdAt: Date.parse("2026-03-23T17:00:05.000Z")
  });
  const serveTimeDecision = {
    recordType: "serve_time_route_decision",
    recordId: "decision-1",
    recordedAt: "2026-03-23T17:00:06.000Z",
    activationRoot,
    breadcrumbs: {
      entrypoint: "compileRuntimeContext",
      invocationSurface: "direct_compile_call",
      hostEvent: null,
      installedEntryPath: null,
      syntheticTurn: false
    },
    sessionId: "sess-1",
    channel: "cli",
    userMessage: "show me the learned routing context",
    turnSequenceStart: 1,
    turnCompileEventId: "evt-i1",
    turnCreatedAt: "2026-03-23T17:00:00.000Z",
    activePackId: packId,
    activePackBuiltAt: "2026-03-23T17:01:00.000Z",
    activePackEventExportDigest: servedPack.summary.eventExportDigest,
    activePackRouterChecksum: null,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    routerIdentity: null,
    usedLearnedRouteFn: true,
    servedArtifact: null,
    selectionDigest: "selection-1",
    requestedBudget: {
      modeRequested: "learned_required",
      maxContextBlocks: 1,
      maxContextChars: null
    },
    actualBudget: {
      modeEffective: "learned_required",
      selectedCount: 2,
      selectedCharCount: 120,
      selectedTokenCount: 24
    },
    candidateSetIds: [`${packId}:event:evt-i1`, `${packId}:event:evt-i2`, `${packId}:merge:1`],
    chosenContextIds,
    candidateScores: [
      {
        blockId: `${packId}:event:evt-i1`,
        source: "event:i1",
        selected: false,
        compactedFrom: [],
        matchedTokens: ["learned"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.2 },
        routeFnScore: 0.2,
        actionScore: 0.2,
        actionProbability: 0.2,
        actionLogProbability: Math.log(0.2),
        traversalScore: 0.2,
        priority: 1
      },
      {
        blockId: `${packId}:event:evt-i2`,
        source: "event:i2",
        selected: true,
        compactedFrom: [],
        matchedTokens: ["learned", "routing"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.9 },
        routeFnScore: 0.9,
        actionScore: 0.9,
        actionProbability: 0.5,
        actionLogProbability: Math.log(0.5),
        traversalScore: 0.9,
        priority: 2
      },
      {
        blockId: `${packId}:merge:1`,
        source: "merge",
        selected: true,
        compactedFrom: [`${packId}:event:evt-i1`, `${packId}:event:evt-i2`],
        matchedTokens: ["learned", "context"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.8 },
        routeFnScore: 0.8,
        actionScore: 0.8,
        actionProbability: 0.3,
        actionLogProbability: Math.log(0.3),
        traversalScore: 0.8,
        priority: 2
      }
    ],
    structuralSignals: null,
    fallbackReason: null,
    hotPathTiming: { totalMs: 1 },
    kernelContextCount: 2,
    brainContextCount: 0,
    selectedKernelContextIds: chosenContextIds,
    selectedBrainContextIds: [],
    promotionLink: null
  };
  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-candidate",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    builtAt: "2026-03-23T17:02:00.000Z",
    structuralOps,
    pgVersion: "v2",
    serveTimeDecisions: [serveTimeDecision],
    baselineState: {
      movingAverage: 0,
      count: 0,
      alpha: 0.1,
      lastUpdatedAt: "2026-03-23T17:00:00.000Z"
    },
    activationRoot
  });
  const router = result.payloads.router;
  assert.ok(router, "expected learned router artifact");
  assert.equal(result.routingBuild.learnedRoutingPath, "policy_gradient_v2");
  assert.equal(result.routingBuild.pgVersionUsed, "v2");
  assert.deepEqual(result.routingBuild.observationBindingStats, {
    totalObservationCount: 1,
    nonZeroObservationCount: 1,
    skippedZeroRewardCount: 0,
    accounting: {
      exact: 1,
      heuristic: 0,
      unmatched: 0,
      ambiguous: 0
    },
    matched: {
      exactDecisionId: 1,
      exactSelectionDigest: 0,
      turnCompileEventId: 0,
      legacyHeuristic: 0
    },
    unmatched: {
      exactDecisionId: 0,
      exactSelectionDigest: 0,
      turnCompileEventId: 0,
      legacyHeuristic: 0
    },
    ambiguous: {
      exactDecisionId: 0,
      exactSelectionDigest: 0,
      turnCompileEventId: 0,
      legacyHeuristic: 0
    }
  });
  assert.deepEqual(router.training.objective.profile, ROUTER_PG_PROFILE_V2);
  assert.equal(router.training.method, "policy_gradient_v2");
  assert.equal(router.training.noOpReason, null);
  assert.equal(router.training.routeTraceCount, 1);
  assert.equal(router.training.supervisionCount, 1);
  assert.ok(router.policyUpdates.length > 0, "expected observation-backed V2 updates");
  assert.ok(router.policyUpdates.some((update) => update.delta !== 0), "expected at least one nonzero policy update");
});

test("native V2 falls back to exact selection digest when decision ids are absent", (t) => {
  const workspace = createWorkspace(t);
  const activationRoot = createActivationRoot(t);
  const interactionEvents = [
    {
      contract: CONTRACT_IDS.interactionEvents,
      eventId: "evt-i1",
      agentId: "agent",
      sessionId: "sess-1",
      channel: "cli",
      sequence: 1,
      kind: "memory_compiled",
      createdAt: "2026-03-23T17:00:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "session.tail" },
      messageId: "msg-1"
    }
  ];
  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents: []
  });
  const structuralOps = { connect: 3, split: 1, merge: 1, prune: 1 };
  const servedPack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-digest-served",
    workspace,
    normalizedEventExport,
    learnedRouting: false,
    builtAt: "2026-03-23T17:01:00.000Z",
    structuralOps
  });
  const packId = servedPack.summary.packId;
  const chosenContextIds = [`${packId}:event:evt-i1`];
  seedObservationDb(activationRoot, {
    id: "bo_test_digest",
    episodeId: "ep_test_digest",
    queryText: "show me the learned routing context",
    selectionDigest: "selection-digest-only",
    activePackId: packId,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    selectedNodeIds: chosenContextIds,
    selectedPathNodeIds: chosenContextIds,
    finalScore: 0.7,
    createdAt: Date.parse("2026-03-23T17:00:05.000Z")
  });
  const serveTimeDecision = {
    recordType: "serve_time_route_decision",
    recordId: "decision-digest",
    recordedAt: "2026-03-23T17:00:06.000Z",
    activationRoot,
    breadcrumbs: {
      entrypoint: "compileRuntimeContext",
      invocationSurface: "direct_compile_call",
      hostEvent: null,
      installedEntryPath: null,
      syntheticTurn: false
    },
    sessionId: "sess-1",
    channel: "cli",
    userMessage: "show me the learned routing context",
    turnSequenceStart: 1,
    turnCompileEventId: "evt-i1",
    turnCreatedAt: "2026-03-23T17:00:00.000Z",
    activePackId: packId,
    activePackBuiltAt: "2026-03-23T17:01:00.000Z",
    activePackEventExportDigest: servedPack.summary.eventExportDigest,
    activePackRouterChecksum: null,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    routerIdentity: null,
    usedLearnedRouteFn: true,
    servedArtifact: null,
    selectionDigest: "selection-digest-only",
    requestedBudget: {
      modeRequested: "learned_required",
      maxContextBlocks: 1,
      maxContextChars: null
    },
    actualBudget: {
      modeEffective: "learned_required",
      selectedCount: 1,
      selectedCharCount: 80,
      selectedTokenCount: 16
    },
    candidateSetIds: chosenContextIds,
    chosenContextIds,
    candidateScores: [
      {
        blockId: `${packId}:event:evt-i1`,
        source: "event:i1",
        selected: true,
        compactedFrom: [],
        matchedTokens: ["learned"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.9 },
        routeFnScore: 0.9,
        actionScore: 0.9,
        actionProbability: 1,
        actionLogProbability: 0,
        traversalScore: 0.9,
        priority: 1
      }
    ],
    structuralSignals: null,
    fallbackReason: null,
    hotPathTiming: { totalMs: 1 },
    kernelContextCount: 1,
    brainContextCount: 0,
    selectedKernelContextIds: chosenContextIds,
    selectedBrainContextIds: [],
    promotionLink: null
  };
  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-digest-candidate",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    builtAt: "2026-03-23T17:02:00.000Z",
    structuralOps,
    pgVersion: "v2",
    serveTimeDecisions: [serveTimeDecision],
    baselineState: {
      movingAverage: 0,
      count: 0,
      alpha: 0.1,
      lastUpdatedAt: "2026-03-23T17:00:00.000Z"
    },
    activationRoot
  });
  assert.equal(result.routingBuild.observationBindingStats?.accounting.exact, 1);
  assert.equal(result.routingBuild.observationBindingStats?.accounting.heuristic, 0);
  assert.equal(result.routingBuild.observationBindingStats?.matched.exactSelectionDigest, 1);
  assert.equal(result.routingBuild.observationBindingStats?.matched.exactDecisionId, 0);
  assert.equal(result.payloads.router?.training.noOpReason, null);
});

test("native V2 uses heuristic fallback only for legacy observation rows", (t) => {
  const workspace = createWorkspace(t);
  const activationRoot = createActivationRoot(t);
  const interactionEvents = [
    {
      contract: CONTRACT_IDS.interactionEvents,
      eventId: "evt-i1",
      agentId: "agent",
      sessionId: "sess-legacy",
      channel: "cli",
      sequence: 1,
      kind: "memory_compiled",
      createdAt: "2026-03-23T17:00:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "session.tail" },
      messageId: "msg-1"
    }
  ];
  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents: []
  });
  const structuralOps = { connect: 3, split: 1, merge: 1, prune: 1 };
  const servedPack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-legacy-served",
    workspace,
    normalizedEventExport,
    learnedRouting: false,
    builtAt: "2026-03-23T17:01:00.000Z",
    structuralOps
  });
  const packId = servedPack.summary.packId;
  const chosenContextIds = [`${packId}:event:evt-i1`];
  seedObservationDb(activationRoot, {
    id: "bo_test_legacy",
    episodeId: "ep_test_legacy",
    queryText: "show me the learned routing context",
    selectedNodeIds: chosenContextIds,
    selectedPathNodeIds: chosenContextIds,
    finalScore: 0.65,
    createdAt: Date.parse("2026-03-23T17:00:05.000Z")
  }, { legacySchema: true });
  const serveTimeDecision = {
    recordType: "serve_time_route_decision",
    recordId: "decision-legacy",
    recordedAt: "2026-03-23T17:00:06.000Z",
    activationRoot,
    breadcrumbs: {
      entrypoint: "compileRuntimeContext",
      invocationSurface: "direct_compile_call",
      hostEvent: null,
      installedEntryPath: null,
      syntheticTurn: false
    },
    sessionId: "sess-legacy",
    channel: "cli",
    userMessage: "show me the learned routing context",
    turnSequenceStart: 1,
    turnCompileEventId: "evt-i1",
    turnCreatedAt: "2026-03-23T17:00:00.000Z",
    activePackId: packId,
    activePackBuiltAt: "2026-03-23T17:01:00.000Z",
    activePackEventExportDigest: servedPack.summary.eventExportDigest,
    activePackRouterChecksum: null,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    routerIdentity: null,
    usedLearnedRouteFn: true,
    servedArtifact: null,
    selectionDigest: "selection-legacy",
    requestedBudget: {
      modeRequested: "learned_required",
      maxContextBlocks: 1,
      maxContextChars: null
    },
    actualBudget: {
      modeEffective: "learned_required",
      selectedCount: 1,
      selectedCharCount: 80,
      selectedTokenCount: 16
    },
    candidateSetIds: chosenContextIds,
    chosenContextIds,
    candidateScores: [
      {
        blockId: `${packId}:event:evt-i1`,
        source: "event:i1",
        selected: true,
        compactedFrom: [],
        matchedTokens: ["learned"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.9 },
        routeFnScore: 0.9,
        actionScore: 0.9,
        actionProbability: 1,
        actionLogProbability: 0,
        traversalScore: 0.9,
        priority: 1
      }
    ],
    structuralSignals: null,
    fallbackReason: null,
    hotPathTiming: { totalMs: 1 },
    kernelContextCount: 1,
    brainContextCount: 0,
    selectedKernelContextIds: chosenContextIds,
    selectedBrainContextIds: [],
    promotionLink: null
  };
  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-legacy-candidate",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    builtAt: "2026-03-23T17:02:00.000Z",
    structuralOps,
    pgVersion: "v2",
    serveTimeDecisions: [serveTimeDecision],
    baselineState: {
      movingAverage: 0,
      count: 0,
      alpha: 0.1,
      lastUpdatedAt: "2026-03-23T17:00:00.000Z"
    },
    activationRoot
  });
  assert.equal(result.routingBuild.observationBindingStats?.accounting.heuristic, 1);
  assert.equal(result.routingBuild.observationBindingStats?.accounting.exact, 0);
  assert.equal(result.routingBuild.observationBindingStats?.matched.legacyHeuristic, 1);
  assert.equal(result.routingBuild.observationBindingStats?.matched.exactDecisionId, 0);
  assert.ok((result.payloads.router?.policyUpdates.length ?? 0) > 0);
});

test("native V2 records unmatched exact provenance instead of silently falling back", (t) => {
  const workspace = createWorkspace(t);
  const activationRoot = createActivationRoot(t);
  const interactionEvents = [
    {
      contract: CONTRACT_IDS.interactionEvents,
      eventId: "evt-i1",
      agentId: "agent",
      sessionId: "sess-unmatched",
      channel: "cli",
      sequence: 1,
      kind: "memory_compiled",
      createdAt: "2026-03-23T17:00:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "session.tail" },
      messageId: "msg-1"
    }
  ];
  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents: []
  });
  const structuralOps = { connect: 3, split: 1, merge: 1, prune: 1 };
  const servedPack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-unmatched-served",
    workspace,
    normalizedEventExport,
    learnedRouting: false,
    builtAt: "2026-03-23T17:01:00.000Z",
    structuralOps
  });
  const packId = servedPack.summary.packId;
  const chosenContextIds = [`${packId}:event:evt-i1`];
  seedObservationDb(activationRoot, {
    id: "bo_test_unmatched",
    episodeId: "ep_test_unmatched",
    queryText: "show me the learned routing context",
    serveDecisionRecordId: "missing-decision-id",
    selectionDigest: "selection-unmatched",
    turnCompileEventId: "evt-i1",
    activePackId: packId,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    selectedNodeIds: chosenContextIds,
    selectedPathNodeIds: chosenContextIds,
    finalScore: 0.75,
    createdAt: Date.parse("2026-03-23T17:00:05.000Z")
  });
  const serveTimeDecision = {
    recordType: "serve_time_route_decision",
    recordId: "decision-unmatched",
    recordedAt: "2026-03-23T17:00:06.000Z",
    activationRoot,
    breadcrumbs: {
      entrypoint: "compileRuntimeContext",
      invocationSurface: "direct_compile_call",
      hostEvent: null,
      installedEntryPath: null,
      syntheticTurn: false
    },
    sessionId: "sess-unmatched",
    channel: "cli",
    userMessage: "show me the learned routing context",
    turnSequenceStart: 1,
    turnCompileEventId: "evt-i1",
    turnCreatedAt: "2026-03-23T17:00:00.000Z",
    activePackId: packId,
    activePackBuiltAt: "2026-03-23T17:01:00.000Z",
    activePackEventExportDigest: servedPack.summary.eventExportDigest,
    activePackRouterChecksum: null,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    routerIdentity: null,
    usedLearnedRouteFn: true,
    servedArtifact: null,
    selectionDigest: "selection-unmatched",
    requestedBudget: {
      modeRequested: "learned_required",
      maxContextBlocks: 1,
      maxContextChars: null
    },
    actualBudget: {
      modeEffective: "learned_required",
      selectedCount: 1,
      selectedCharCount: 80,
      selectedTokenCount: 16
    },
    candidateSetIds: chosenContextIds,
    chosenContextIds,
    candidateScores: [
      {
        blockId: `${packId}:event:evt-i1`,
        source: "event:i1",
        selected: true,
        compactedFrom: [],
        matchedTokens: ["learned"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.9 },
        routeFnScore: 0.9,
        actionScore: 0.9,
        actionProbability: 1,
        actionLogProbability: 0,
        traversalScore: 0.9,
        priority: 1
      }
    ],
    structuralSignals: null,
    fallbackReason: null,
    hotPathTiming: { totalMs: 1 },
    kernelContextCount: 1,
    brainContextCount: 0,
    selectedKernelContextIds: chosenContextIds,
    selectedBrainContextIds: [],
    promotionLink: null
  };
  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-unmatched-candidate",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    builtAt: "2026-03-23T17:02:00.000Z",
    structuralOps,
    pgVersion: "v2",
    serveTimeDecisions: [serveTimeDecision],
    baselineState: {
      movingAverage: 0,
      count: 0,
      alpha: 0.1,
      lastUpdatedAt: "2026-03-23T17:00:00.000Z"
    },
    activationRoot
  });
  const router = result.payloads.router;
  assert.ok(router, "expected learned router artifact");
  assert.equal(result.routingBuild.observationBindingStats?.accounting.unmatched, 1);
  assert.equal(result.routingBuild.observationBindingStats?.accounting.exact, 0);
  assert.equal(result.routingBuild.observationBindingStats?.unmatched.exactDecisionId, 1);
  assert.equal(result.routingBuild.observationBindingStats?.matched.legacyHeuristic, 0);
  assert.equal(router.training.status, "no_supervision");
  assert.match(router.training.noOpReason ?? "", /unmatched\(exact_decision_id=1/);
  assert.equal(router.policyUpdates.length, 0);
});

test("native V2 records ambiguous turn compile ids instead of training on timestamp fallback", (t) => {
  const workspace = createWorkspace(t);
  const activationRoot = createActivationRoot(t);
  const interactionEvents = [
    {
      contract: CONTRACT_IDS.interactionEvents,
      eventId: "evt-dup",
      agentId: "agent",
      sessionId: "sess-ambiguous",
      channel: "cli",
      sequence: 1,
      kind: "memory_compiled",
      createdAt: "2026-03-23T17:00:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "session.tail" },
      messageId: "msg-1"
    }
  ];
  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents: []
  });
  const structuralOps = { connect: 3, split: 1, merge: 1, prune: 1 };
  const servedPack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-ambiguous-served",
    workspace,
    normalizedEventExport,
    learnedRouting: false,
    builtAt: "2026-03-23T17:01:00.000Z",
    structuralOps
  });
  const packId = servedPack.summary.packId;
  const chosenContextIds = [`${packId}:event:evt-dup`];
  seedObservationDb(activationRoot, {
    id: "bo_test_ambiguous",
    episodeId: "ep_test_ambiguous",
    queryText: "show me the learned routing context",
    turnCompileEventId: "evt-dup",
    activePackId: packId,
    selectedNodeIds: chosenContextIds,
    selectedPathNodeIds: chosenContextIds,
    finalScore: 0.7,
    createdAt: Date.parse("2026-03-23T17:00:05.000Z")
  });
  const baseDecision = {
    recordType: "serve_time_route_decision",
    recordedAt: "2026-03-23T17:00:06.000Z",
    activationRoot,
    breadcrumbs: {
      entrypoint: "compileRuntimeContext",
      invocationSurface: "direct_compile_call",
      hostEvent: null,
      installedEntryPath: null,
      syntheticTurn: false
    },
    sessionId: "sess-ambiguous",
    channel: "cli",
    userMessage: "show me the learned routing context",
    turnSequenceStart: 1,
    turnCompileEventId: "evt-dup",
    turnCreatedAt: "2026-03-23T17:00:00.000Z",
    activePackId: packId,
    activePackBuiltAt: "2026-03-23T17:01:00.000Z",
    activePackEventExportDigest: servedPack.summary.eventExportDigest,
    activePackRouterChecksum: null,
    activePackGraphChecksum: servedPack.manifest.payloadChecksums.graph,
    routerIdentity: null,
    usedLearnedRouteFn: true,
    servedArtifact: null,
    selectionDigest: null,
    requestedBudget: {
      modeRequested: "learned_required",
      maxContextBlocks: 1,
      maxContextChars: null
    },
    actualBudget: {
      modeEffective: "learned_required",
      selectedCount: 1,
      selectedCharCount: 80,
      selectedTokenCount: 16
    },
    candidateSetIds: chosenContextIds,
    chosenContextIds,
    candidateScores: [
      {
        blockId: `${packId}:event:evt-dup`,
        source: "event:dup",
        selected: true,
        compactedFrom: [],
        matchedTokens: ["learned"],
        routingChannels: ["graph"],
        channelScores: { graph: 0.9 },
        routeFnScore: 0.9,
        actionScore: 0.9,
        actionProbability: 1,
        actionLogProbability: 0,
        traversalScore: 0.9,
        priority: 1
      }
    ],
    structuralSignals: null,
    fallbackReason: null,
    hotPathTiming: { totalMs: 1 },
    kernelContextCount: 1,
    brainContextCount: 0,
    selectedKernelContextIds: chosenContextIds,
    selectedBrainContextIds: [],
    promotionLink: null
  };
  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "teacher-v2-observation-ambiguous-candidate",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    builtAt: "2026-03-23T17:02:00.000Z",
    structuralOps,
    pgVersion: "v2",
    serveTimeDecisions: [
      {
        ...baseDecision,
        recordId: "decision-ambiguous-a",
      },
      {
        ...baseDecision,
        recordId: "decision-ambiguous-b",
        recordedAt: "2026-03-23T17:00:06.500Z",
      }
    ],
    baselineState: {
      movingAverage: 0,
      count: 0,
      alpha: 0.1,
      lastUpdatedAt: "2026-03-23T17:00:00.000Z"
    },
    activationRoot
  });
  const router = result.payloads.router;
  assert.ok(router, "expected learned router artifact");
  assert.equal(result.routingBuild.observationBindingStats?.accounting.ambiguous, 1);
  assert.equal(result.routingBuild.observationBindingStats?.ambiguous.turnCompileEventId, 1);
  assert.equal(result.routingBuild.observationBindingStats?.matched.legacyHeuristic, 0);
  assert.equal(router.training.status, "no_supervision");
  assert.match(router.training.noOpReason ?? "", /ambiguous\(exact_decision_id=0,exact_selection_digest=0,turn_compile_event_id=1/);
  assert.equal(router.policyUpdates.length, 0);
});
