import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";

import { summarizeAttributionTruth } from "../src/live-runtime-audit.js";
import { BrainStore } from "../src/brain-store/store.js";
import { runBrainMigrations } from "../src/brain-store/migrations.js";
import {
  buildProvenanceAuditChainV1,
  PROVENANCE_AUDIT_CHAIN_CONTRACT,
  PROVENANCE_AUDIT_CHAIN_MAX_ROUTE_ROWS,
  PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
  renderProvenanceAuditChainMarkdownV1,
} from "../src/brain-runtime/provenance-audit-chain.js";
import type { BrainObservationTeacherEvaluation, DecisionTrace, DecisionTraceSelectionMetadataV4 } from "../src/brain-core/types.js";

const tempDirs: string[] = [];

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function cleanupTempDirs(): void {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
}

function makeDecisionPointSnapshot(traceId: string, decisionPointId: string, selectionIndex: number) {
  return {
    schemaVersion: 1,
    decisionPointId,
    traceId,
    episodeId: `episode-${traceId}`,
    conversationId: 42,
    sourceNodeId: selectionIndex === 0 ? null : `node-${selectionIndex - 1}`,
    expansionIndex: selectionIndex,
    selectionIndex,
    decisionPointKind: selectionIndex === 0 ? "seed" : "local",
    localActionSet: [
      {
        action_id: `${decisionPointId}-a`,
        action_kind: "traverse",
        node_id: `node-${selectionIndex}`,
        tool_name: null,
        tool_capability_id: null,
        tool_instance_id: null,
        tool_args_shape: null,
        prior_score: 0.41,
        probability: 0.73,
        retrieval_features: { mode: "seed" },
      },
      {
        action_id: `${decisionPointId}-b`,
        action_kind: "stop_local",
        node_id: null,
        tool_name: null,
        tool_capability_id: null,
        tool_instance_id: null,
        tool_args_shape: null,
        prior_score: 0.18,
        probability: 0.27,
        retrieval_features: { mode: "fallback" },
      },
    ],
    chosenActionId: `${decisionPointId}-a`,
    chosenActionKind: "traverse",
    chosenNodeId: `node-${selectionIndex}`,
    chosenToolName: null,
    chosenToolCapabilityId: null,
    chosenToolInstanceId: null,
    chosenActionProbability: 0.73,
    stopProbability: 0.27,
    stopTruth: "chosen",
    stopReason: null,
    budgetContext: {
      budgetRemaining: 1200,
      initialBudget: 1600,
      reservedTokenCost: 64,
      budgetUsed: 400,
      budgetUsedFraction: 0.25,
      maxHops: 3,
      maxFrontierSize: 8,
      frontierSize: 2,
      visitedCount: 2,
      firedCount: 1,
      pendingSelectionCount: 0,
      pressureLevel: 0.2,
      frontierPressure: 0.15,
      budgetPressure: 0.1,
      budgetFraction: 0.75,
      queryBudgetChars: 240,
      maxContextChars: 512,
      injectedChars: 160,
      droppedChars: 0,
      contextClipped: false,
      routeSelectionMs: 11,
      totalQueryMs: 18,
      compileDeadlineMs: null,
      compileDeadlineHit: false,
    },
    routeContext: {
      requestDigest: `request-${traceId}`,
      activePackId: "pack-42",
      routerIdentity: "route_fn.v1",
      candidateNodeIds: [`node-${selectionIndex}`, `node-${selectionIndex + 1}`],
      selectedNodeIds: [`node-${selectionIndex}`],
      selectedTraversalNodeIds: [`node-${selectionIndex}`],
      selectedPathNodeIds: [`node-${selectionIndex}`],
      selectedSeedNodeIds: [`seed-${selectionIndex}`],
    },
  };
}

function makeSelectionMetadata(traceId: string, snapshotCount: number): DecisionTraceSelectionMetadataV4 {
  const decisionPointSnapshots = Array.from({ length: snapshotCount }, (_unused, index) =>
    makeDecisionPointSnapshot(traceId, `dp-${index + 1}`, index),
  );

  return {
    traceSliceVersion: 4,
    queryChars: 168,
    budgetChars: 512,
    maxHops: 3,
    maxFanoutPerNode: 2,
    maxFrontierSize: 8,
    seedCount: 1,
    seedSelectionCount: 1,
    candidateCount: snapshotCount * 2,
    hopCount: snapshotCount,
    expansionCount: snapshotCount,
    selectionSubstepCount: snapshotCount,
    firedCount: snapshotCount,
    vetoedCount: 0,
    chosenSeedNodeId: "seed-0",
    selectedSeedNodeIds: ["seed-0"],
    routeSelectionMs: 24,
    embeddingMs: 4,
    totalQueryMs: 28,
    queryEmbeddingSource: "provided",
    chosenStopCount: 0,
    forcedStopCount: 0,
    branchOutcomeSummary: null,
    droppedProposalCount: 0,
    droppedProposalReasons: null,
    interruption: null,
    queryInterrupted: false,
    interruptionStage: null,
    interruptionReason: null,
    servedPartial: false,
    compileElapsedMs: 17,
    compileDeadlineMs: null,
    compileDeadlineHit: false,
    brainDropReason: null,
    brainDropStage: null,
    budgetFraction: 0.75,
    maxContextChars: 512,
    queryBudgetChars: 240,
    injectedChars: 160,
    droppedChars: 0,
    contextClipped: false,
    fitStrategy: "structured_node_budget",
    retrievedNodeCount: snapshotCount * 2,
    fittedNodeCount: snapshotCount + 1,
    droppedNodeCount: 0,
    fittingDropReasons: null,
    interruptionAccounting: null,
    decisionPointSnapshots,
    decisionPointSummary: `${snapshotCount} decision point snapshots`,
    compileReportSummary: `bounded compile report for ${traceId}`,
  } as DecisionTraceSelectionMetadataV4;
}

function makeTrace(traceId: string, snapshotCount: number): DecisionTrace {
  const selectionMetadata = makeSelectionMetadata(traceId, snapshotCount);
  const firstSnapshot = selectionMetadata.decisionPointSnapshots?.[0];
  return {
    id: traceId,
    episodeId: `episode-${traceId}`,
    packVersion: 42,
    queryText: `route query for ${traceId}`,
    seedScores: [],
    trajectory: [],
    firedNodes: [],
    vetoedNodes: [],
    contextChars: 168,
    footer: "route footer",
    routeTrace: {
      requestDigest: firstSnapshot?.routeContext.requestDigest ?? traceId,
      conversationId: firstSnapshot?.conversationId ?? null,
      agentIdentity: null,
      activePackId: firstSnapshot?.routeContext.activePackId ?? "pack-42",
      routerIdentity: firstSnapshot?.routeContext.routerIdentity ?? "route_fn.v1",
      candidateNodeIds: firstSnapshot?.routeContext.candidateNodeIds ?? [],
      selectedNodeIds: firstSnapshot?.routeContext.selectedNodeIds ?? [],
      selectedTraversalNodeIds: firstSnapshot?.routeContext.selectedTraversalNodeIds ?? [],
      selectedPathNodeIds: firstSnapshot?.routeContext.selectedPathNodeIds ?? [],
      selectedSeedNodeIds: firstSnapshot?.routeContext.selectedSeedNodeIds ?? [],
      branchOutcomes: [],
      injectedNodeSummaries: [],
      sourceSummary: {
        injectedCount: 0,
        kinds: {},
        trusts: {},
        sourceUris: [],
        sourceRefs: [],
      },
      selectionMetadata,
    },
    createdAt: 1_711_000_000_000,
  };
}

function buildRuntimeStatus(traceId: string, snapshotCount: number, updateDecisionCount: number) {
  const observationAttribution = summarizeAttributionTruth({
    observationAttribution: {
      totalObservationCount: 3,
      teacherEvaluationCount: 2,
      latestAmbiguous: {
        observationId: "bo_ambiguous",
        episodeId: `episode-${traceId}`,
        traceId,
        bindingMode: "trace_id",
        attributionQuality: "fallback",
        feedbackRichness: "followup_only",
        confidence: 0.61,
        reason: "fallback binding stayed visible in the audit chain",
        evaluatedAt: 1_711_000_000_500,
      },
      latestUnmatched: {
        observationId: "bo_unmatched",
        episodeId: `episode-${traceId}`,
        traceId,
        bindingMode: "unbound",
        attributionQuality: "unbound",
        feedbackRichness: "sparse",
        confidence: 0.22,
        reason: "unbound route evidence stayed visible",
        evaluatedAt: 1_711_000_000_600,
      },
      attributionQuality: {
        exact: 1,
        fallback: 1,
        unbound: 1,
      },
    },
    teacherTruth: {
      queue: {
        budgetPerTick: 20,
        delayMs: 10_000,
        pendingCount: 1,
        pendingFollowupCount: 1,
        pendingTeacherCount: 0,
        readyCount: 1,
        delayedCount: 0,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
        richReadyCount: 1,
        sample: [
          {
            observationId: "bo_pending",
            episodeId: `episode-${traceId}`,
            traceId,
            status: "pending_followup",
            gate: "ready",
            reason: "follow-up keeps the teacher queue visible",
            feedbackRichness: "followup_only",
            createdAt: 1_711_000_000_700,
          },
        ],
        detail: "teacher queue stays visible for the audit chain",
      },
    },
  });

  const decisionTrace = makeTrace(traceId, snapshotCount);

  return {
    recentDecisionSummary: {
      windowSize: snapshotCount,
      sampleSize: snapshotCount,
      histograms: {
        decisionOutcome: {
          served_full: snapshotCount,
          served_clipped: 0,
          partial_fail_open: 0,
          partial_fail_open_clipped: 0,
          interrupted_without_partial: 0,
        },
        brainDropReason: {},
        interruptionStage: {},
        fitStrategy: {
          structured_node_budget: snapshotCount,
        },
        queryEmbeddingSource: {
          provided: snapshotCount,
        },
      },
      branchBehavior: {
        branchCount: snapshotCount,
        continuingBranchCount: 0,
        histograms: {
          stopTruth: { chosen: snapshotCount },
          terminationReason: {},
        },
        detail: `${snapshotCount} branches served without fail-open`,
      },
      clipRate: { count: 0, rate: 0 },
      failOpenRate: { count: 0, rate: 0 },
      detail: `trace ${traceId} served with ${snapshotCount} route rows`,
    },
    lastTraceSelectionMetadata: decisionTrace.routeTrace?.selectionMetadata ?? null,
    attributionTruth: observationAttribution,
    teacherTruth: {
      queue: {
        budgetPerTick: 20,
        delayMs: 10_000,
        pendingCount: 1,
        pendingFollowupCount: 1,
        pendingTeacherCount: 0,
        readyCount: 1,
        delayedCount: 0,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
        richReadyCount: 1,
        sample: [
          {
            observationId: "bo_pending",
            episodeId: `episode-${traceId}`,
            traceId,
            status: "pending_followup",
            gate: "ready",
            reason: "follow-up keeps the teacher queue visible",
            feedbackRichness: "followup_only",
            createdAt: 1_711_000_000_700,
          },
        ],
        detail: "teacher queue stays visible for the audit chain",
      },
      lastEvaluationCycle: {
        version: 1,
        generatedAt: "2026-04-07T18:00:00Z",
        eligibleObservationCount: 1,
        evaluatedObservationCount: 1,
        detail: "teacher evaluation cycle kept the attribution link alive",
      },
      lastUpdateCycle: {
        version: 1,
        generatedAt: "2026-04-07T18:01:00Z",
        eligibleEpisodeCount: 2,
        appliedEpisodeCount: 1,
        skippedEpisodeCount: 1,
        skippedReasons: {
          missing_supervision: 1,
          zero_policy_delta: 0,
        },
        decisions: Array.from({ length: updateDecisionCount }, (_unused, index) => ({
          episodeId: index === 0 ? `episode-${traceId}` : `episode-${traceId}-alt`,
          status: index === 0 ? "applied" : "skipped",
          reason: index === 0 ? "recorded route update" : "no-op kept the update bounded",
          routeUpdateCount: index === 0 ? 2 : 0,
          traceIds: index === 0 ? [traceId] : [traceId],
          observationIds: index === 0 ? ["bo_exact"] : [],
          supervisionIds: index === 0 ? ["ts_exact"] : [],
          baselineBefore: 0.42,
          baselineAfter: index === 0 ? 0.54 : 0.42,
          summary: index === 0 ? "applied update preserved the route linkage" : "skipped update stayed within the audit cap",
        })),
        detail: `${Math.min(updateDecisionCount, 3)} learning update decision(s) kept the chain bounded`,
      },
    },
    learningHealth: {
      status: "healthy",
      detail: "learning updates are audit-stubbed only",
    },
    continuousLearning: {
      retrain: {
        lineage: {
          priorBaseArtifactId: "router-base-prior-v0",
          priorBaseArtifactVersion: "v0",
          priorBaseArtifactChecksum: "sha256:prior-base-router-checksum",
          candidateArtifactId: "router-artifact-periodic-retrain-v1",
          candidateArtifactVersion: "v1",
          candidateArtifactChecksum: "sha256:candidate-router-checksum",
          priorRooted: true,
          promotionValid: true,
          residualUpdateCount: 7,
          summary: "seeded by router-base-prior-v0@v0; seed checksum=sha256:prior-base-router-checksum; current router=router-artifact-periodic-retrain-v1@v1; router checksum=sha256:candidate-router-checksum; prior-rooted=yes; promotion-valid=yes; residual updates=7",
        },
      },
    },
    promotionStory: {
      summary: {
        currentPackVersion: 42,
        mutationBacklog: {
          pending: 0,
          validated: 0,
          promoted: 0,
          rejected: 0,
        },
        lastPromotionReason: "route row promotion retained the proof path",
        lastReplayFailureReason: null,
      },
      latestActivity: {
        type: "candidate_promoted",
        at: 1_711_000_000_900,
        summary: "candidate promoted with proof linkage intact",
        packVersion: 42,
        candidateId: "candidate-42",
      },
      currentPack: {
        version: 42,
        createdAt: 1_711_000_000_000,
        promotedAt: 1_711_000_000_800,
        rolledBack: false,
        nodeCount: 3,
        edgeCount: 2,
        reason: "promoted for the audit chain test",
        metadata: { reason: "promoted for the audit chain test" },
        health: null,
      },
      recentPromotions: [],
      candidates: {
        pending: [],
        promoted: [],
        rejected: [],
      },
      integrations: {
        structuredVerdict: null,
        learningJournal: null,
      },
    },
    lastPromotionReason: "route row promotion retained the proof path",
    lastPromotionVerdict: {
      verdict: "promoted",
      summary: "promotion stayed bound to the captured route rows",
    },
    lastReplayGateVerdict: {
      verdict: "pass",
      summary: "replay gate stayed bounded",
    },
    lastCompileReportSummary: `bounded compile report for ${traceId}`,
  };
}

function buildProofTruth(traceId: string) {
  return {
    bundleDir: `artifacts/operator-proof-${traceId}`,
    command: "openclawbrain proof --openclaw-home ~/.openclaw",
    summary: `operator proof summary for ${traceId}`,
    verdict: {
      verdict: "success_and_proven",
      severity: "info",
      why: "gateway and runtime truth stayed aligned",
      missingProofs: [],
    },
    runtimeLoadProofPath: "~/.openclaw/activation/attachment-truth/runtime-load-proofs.json",
    runtimeLoadProofExists: true,
    stepCount: 5,
    postBundleCount: 2,
  };
}

afterEach(() => {
  cleanupTempDirs();
});

describe("provenance audit chain", () => {
  it("builds a bounded chain with explicit correction/raw-authority precedence labels", () => {
    const traceId = "trace-audit-01";
    const runtimeStatus = buildRuntimeStatus(traceId, PROVENANCE_AUDIT_CHAIN_MAX_ROUTE_ROWS + 2, 2);
    const chain = buildProvenanceAuditChainV1({
      bundleId: "bundle-audit-01",
      generatedAt: "2026-04-07T18:04:00Z",
      runtimeStatus,
      proofTruth: buildProofTruth(traceId),
    });

    expect(chain.contract).toBe(PROVENANCE_AUDIT_CHAIN_CONTRACT);
    expect(chain.precedence.label).toBe("user_explicit correction > raw_source > teacher_inference");
    expect(chain.serveDecision.routeRows).toHaveLength(PROVENANCE_AUDIT_CHAIN_MAX_ROUTE_ROWS);
    expect(chain.serveDecision.routeRows[0].precedenceLabel).toBe(chain.precedence.label);
    expect(chain.serveDecision.routeRows[0].detail.length).toBeLessThanOrEqual(PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS);
    expect(chain.attributionTruth.latest.ambiguous?.precedenceLabel).toBe(chain.precedence.label);
    expect(chain.attributionTruth.latest.unmatched?.precedenceLabel).toBe(chain.precedence.label);
    expect(chain.learningUpdate.summary).toContain("learning update decision");
    expect(chain.learningUpdate.precedenceLabel).toBe(chain.precedence.label);
    expect(chain.promotionProofTruth.proofTruth?.verdict).toBe("success_and_proven");
    expect(chain.promotionProofTruth.retrainLineage).toMatchObject({
      priorBaseArtifactId: "router-base-prior-v0",
      priorBaseArtifactChecksum: "sha256:prior-base-router-checksum",
      candidateArtifactId: "router-artifact-periodic-retrain-v1",
      candidateArtifactChecksum: "sha256:candidate-router-checksum",
      priorRooted: true,
      promotionValid: true,
      residualUpdateCount: 7,
    });
    expect(chain.linkages.restartSafe).toBe(true);

    const markdown = renderProvenanceAuditChainMarkdownV1(chain);
    expect(markdown).toContain("# Provenance audit chain");
    expect(markdown).toContain("## Serve decision / route rows");
    expect(markdown).toContain("## Attribution truth");
    expect(markdown).toContain("## Learning update");
    expect(markdown).toContain("## Promotion / proof truth");
    expect(markdown).toContain("retrain lineage");
    expect(markdown).toContain("current router checksum: sha256:candidate-router-checksum");
    expect(markdown).toContain(chain.precedence.label);
    expect(markdown.length).toBeLessThan(14_000);
  });

  it("keeps route-row, attribution, update, and proof linkage stable across a store restart", () => {
    const dir = makeTempDir("openclawbrain-provenance-audit-chain-");
    const dbPath = join(dir, "test.db");
    const traceId = "trace-audit-restart";
    const createdAt = 1_711_000_010_000;
    const evaluation: BrainObservationTeacherEvaluation = {
      version: 2,
      observationId: "bo_exact",
      episodeId: `episode-${traceId}`,
      traceId,
      serveDecisionRecordId: "serve-dec-1",
      selectionDigest: "sel-digest-1",
      turnCompileEventId: "turn-compile-1",
      decisionRecordedAt: "2026-04-07T18:01:00Z",
      activePackId: "pack-42",
      activePackEventExportDigest: "digest-1",
      activePackGraphChecksum: "graph-1",
      activePackRouterChecksum: "router-1",
      activePackBuiltAt: "2026-04-07T17:59:00Z",
      bindingMode: "exact_decision_id",
      retrievalRelevance: 0.9,
      agentUsage: 0.4,
      outcomeSupport: 0.6,
      finalScore: 0.88,
      confidence: 0.92,
      reason: "exact-bound route row stayed visible after restart",
    };

    const makeStore = () => {
      const db = new DatabaseSync(dbPath);
      runBrainMigrations(db);
      return { db, store: new BrainStore(db) };
    };

    let storeBundle = makeStore();
    let { db, store } = storeBundle;
    store.insertTrace(makeTrace(traceId, 2));
    const observation = store.insertObservation({
      episodeId: `episode-${traceId}`,
      conversationId: 42,
      traceId,
      queryText: "why did the route row stay linked?",
      retrievedContext: [],
      routeMetadata: {
        requestDigest: `request-${traceId}`,
        activePackId: "pack-42",
        routerIdentity: "route_fn.v1",
        bindingMode: "exact_decision_id",
        serveDecisionRecordId: "serve-dec-1",
        selectionDigest: "sel-digest-1",
        turnCompileEventId: "turn-compile-1",
        decisionRecordedAt: "2026-04-07T18:01:00Z",
        activePackEventExportDigest: "digest-1",
        activePackGraphChecksum: "graph-1",
        activePackRouterChecksum: "router-1",
        activePackBuiltAt: "2026-04-07T17:59:00Z",
        servedArtifact: {
          summary: "served artifact",
          compileReportSummary: "bounded compile report for restart test",
        },
        candidateNodeIds: ["node-0", "node-1"],
        selectedNodeIds: ["node-0"],
        selectedTraversalNodeIds: ["node-0"],
        selectedPathNodeIds: ["node-0"],
        selectedSeedNodeIds: ["seed-0"],
        sourceSummary: {
          injectedCount: 0,
          kinds: {},
          trusts: {},
          sourceUris: [],
          sourceRefs: [],
        },
        operatorAudit: null,
        selectionMetadata: makeSelectionMetadata(traceId, 2),
      },
      assistantResponse: "ack",
      toolResults: [],
      followUpText: "follow up",
      status: "completed",
    });
    store.completeObservationEvaluation({
      observationId: observation.id,
      status: "completed",
      confidence: 0.92,
      reason: "exact attribution recorded",
      teacherEvaluation: evaluation,
    });
    store.setTrainingStateJson("last_teacher_update_cycle_json", {
      version: 1,
      generatedAt: "2026-04-07T18:02:00Z",
      eligibleEpisodeCount: 1,
      appliedEpisodeCount: 1,
      skippedEpisodeCount: 0,
      skippedReasons: {
        missing_supervision: 0,
        zero_policy_delta: 0,
      },
      decisions: [
        {
          episodeId: `episode-${traceId}`,
          status: "applied",
          reason: "route linkage preserved",
          routeUpdateCount: 1,
          traceIds: [traceId],
          observationIds: [observation.id],
          supervisionIds: ["ts_exact"],
          baselineBefore: 0.44,
          baselineAfter: 0.51,
          summary: "applied update preserved the route linkage",
        },
      ],
      detail: "1 update kept the route linkage intact",
    });
    store.setTrainingStateJson("last_teacher_evaluation_cycle_json", {
      version: 1,
      generatedAt: "2026-04-07T18:01:30Z",
      eligibleObservationCount: 1,
      evaluatedObservationCount: 1,
      detail: "evaluation cycle kept the exact binding visible",
    });
    store.setTrainingState("last_promotion_reason", "route row promotion retained the proof path");
    store.setTrainingStateJson("last_promotion_verdict_json", {
      verdict: "promoted",
      summary: "promotion stayed bound to the captured route rows",
    });
    store.setTrainingStateJson("last_replay_gate_verdict_json", {
      verdict: "pass",
      summary: "replay gate stayed bounded",
    });

    const buildSnapshot = () => {
      const recentTrace = store.getRecentTraces(1)[0] ?? null;
      return {
        recentDecisionSummary: store.getRecentDecisionSummary(5),
        lastTraceSelectionMetadata: recentTrace?.routeTrace?.selectionMetadata ?? null,
        attributionTruth: summarizeAttributionTruth({
          observationAttribution: store.getObservationAttributionSummary(),
          teacherTruth: {
            queue: store.getTeacherQueueSummary(Number.MAX_SAFE_INTEGER, 10),
            lastEvaluationCycle: store.getTrainingStateJson("last_teacher_evaluation_cycle_json"),
            lastUpdateCycle: store.getTrainingStateJson("last_teacher_update_cycle_json"),
          },
        }),
        teacherTruth: {
          queue: store.getTeacherQueueSummary(Number.MAX_SAFE_INTEGER, 10),
          lastEvaluationCycle: store.getTrainingStateJson("last_teacher_evaluation_cycle_json"),
          lastUpdateCycle: store.getTrainingStateJson("last_teacher_update_cycle_json"),
        },
        learningHealth: { status: "healthy", detail: "restart-safe audit stitching" },
        promotionStory: {
          summary: {
            currentPackVersion: store.getCurrentPackVersion(),
            mutationBacklog: store.countMutationsByStatus(),
            lastPromotionReason: store.getTrainingState("last_promotion_reason"),
            lastReplayFailureReason: store.getTrainingState("last_replay_failure_reason"),
          },
          latestActivity: {
            type: "candidate_promoted",
            at: createdAt + 700,
            summary: "route row promotion retained the proof path",
            packVersion: store.getCurrentPackVersion(),
            candidateId: observation.id,
          },
          currentPack: {
            version: 42,
            createdAt,
            promotedAt: createdAt + 800,
            rolledBack: false,
            nodeCount: 3,
            edgeCount: 2,
            reason: "restart-safe audit chain",
            metadata: { reason: "restart-safe audit chain" },
            health: null,
          },
          recentPromotions: [],
          candidates: { pending: [], promoted: [], rejected: [] },
          integrations: { structuredVerdict: null, learningJournal: null },
        },
        lastPromotionReason: store.getTrainingState("last_promotion_reason"),
        lastPromotionVerdict: store.getTrainingStateJson("last_promotion_verdict_json"),
        lastReplayGateVerdict: store.getTrainingStateJson("last_replay_gate_verdict_json"),
        lastCompileReportSummary: recentTrace?.routeTrace?.selectionMetadata?.compileReportSummary ?? null,
      };
    };

    const beforeRestart = buildProvenanceAuditChainV1({
      bundleId: "bundle-restart-01",
      generatedAt: "2026-04-07T18:02:30Z",
      runtimeStatus: buildSnapshot(),
      proofTruth: buildProofTruth(traceId),
    });

    const beforeRouteRowId = beforeRestart.serveDecision.routeRows[0]?.rowId;
    const beforeObservationId = beforeRestart.linkages.observationIds[0];
    const beforeUpdateEpisodeId = beforeRestart.linkages.updateEpisodeIds[0];
    const beforeProofVerdict = beforeRestart.linkages.proofVerdict;

    db.close();
    storeBundle = makeStore();
    ({ db, store } = storeBundle);

    const afterRestart = buildProvenanceAuditChainV1({
      bundleId: "bundle-restart-01",
      generatedAt: "2026-04-07T18:02:30Z",
      runtimeStatus: buildSnapshot(),
      proofTruth: buildProofTruth(traceId),
    });

    expect(afterRestart.serveDecision.routeRows[0]?.rowId).toBe(beforeRouteRowId);
    expect(afterRestart.linkages.observationIds[0]).toBe(beforeObservationId);
    expect(afterRestart.linkages.updateEpisodeIds[0]).toBe(beforeUpdateEpisodeId);
    expect(afterRestart.linkages.proofVerdict).toBe(beforeProofVerdict);
    expect(afterRestart.linkages.restartSafe).toBe(true);

    db.close();
  });
});
