import { describe, expect, it } from "vitest";
import { evaluateContextUsefulness } from "../../src/brain-core/usefulness.js";
import type {
  BrainObservation,
  BrainObservationTeacherEvaluation,
  BrainObservationToolResult,
} from "../../src/brain-core/types.js";

function makeObservation(params: {
  id: string;
  followUpText?: string | null;
  toolResults?: BrainObservationToolResult[];
  bindingMode?: BrainObservation["routeMetadata"]["bindingMode"];
  selectionMetadata?: Partial<NonNullable<BrainObservation["routeMetadata"]["selectionMetadata"]>>;
  teacherEvaluation?: BrainObservationTeacherEvaluation | null;
}): BrainObservation {
  const selectionMetadata = {
    traceSliceVersion: 4,
    queryChars: 48,
    budgetChars: 4000,
    maxHops: 8,
    maxFanoutPerNode: 4,
    maxFrontierSize: 32,
    seedCount: 1,
    seedSelectionCount: 1,
    candidateCount: 1,
    hopCount: 1,
    expansionCount: 1,
    selectionSubstepCount: 2,
    firedCount: 1,
    vetoedCount: 0,
    chosenSeedNodeId: "node_1",
    selectedSeedNodeIds: ["node_1"],
    routeSelectionMs: 4,
    embeddingMs: 2,
    totalQueryMs: 9,
    queryEmbeddingSource: "provided" as const,
    servedPartial: false,
    compileDeadlineHit: false,
    contextClipped: false,
    queryInterrupted: false,
    brainDropReason: null,
    brainDropStage: null,
    interruption: null,
    interruptionStage: null,
    interruptionReason: null,
    ...params.selectionMetadata,
  };

  return {
    id: params.id,
    episodeId: `ep_${params.id}`,
    conversationId: 42,
    traceId: `bt_${params.id}`,
    queryText: "how should I do this?",
    retrievedContext: [],
    routeMetadata: {
      requestDigest: "digest",
      activePackId: "brain-pack-v1",
      routerIdentity: "brain-router",
      bindingMode: params.bindingMode ?? "exact_decision_id",
      serveDecisionRecordId: `decision-${params.id}`,
      selectionDigest: `selection-${params.id}`,
      turnCompileEventId: `compile-${params.id}`,
      decisionRecordedAt: "2026-03-31T23:50:00.000Z",
      activePackEventExportDigest: "export-digest",
      activePackGraphChecksum: "graph-checksum",
      activePackRouterChecksum: "router-checksum",
      activePackBuiltAt: "2026-03-31T23:49:00.000Z",
      servedArtifact: { kind: "runtime_compile_v1", traceId: `bt_${params.id}` },
      candidateNodeIds: ["node_1"],
      selectedNodeIds: ["node_1"],
      selectedTraversalNodeIds: ["node_1"],
      selectedPathNodeIds: ["node_1"],
      selectedSeedNodeIds: ["node_1"],
      sourceSummary: {
        injectedCount: 1,
        kinds: { workflow: 1 },
        trusts: { human: 1 },
        sourceUris: ["PLAYBOOK.md"],
        sourceRefs: ["PLAYBOOK.md#L1"],
      },
      selectionMetadata,
    },
    assistantResponse: "answer",
    toolResults: params.toolResults ?? [],
    followUpText: params.followUpText ?? null,
    phase1Score: null,
    phase2Score: null,
    finalScore: null,
    confidence: null,
    reason: null,
    status: params.followUpText ? "pending_teacher" : "pending_followup",
    teacherEvaluation: params.teacherEvaluation ?? null,
    createdAt: Date.now() - 30_000,
    updatedAt: Date.now() - 30_000,
    evaluatedAt: null,
  } as BrainObservation;
}

describe("evaluateContextUsefulness", () => {
  it("scores a strongly positive follow-up as helpful", () => {
    const observation = makeObservation({
      id: "helpful",
      followUpText: "Perfect, that worked.",
      toolResults: [
        {
          sourceRole: "tool",
          toolCallId: "call_1",
          toolName: "bash",
          input: "pnpm test",
          output: '{"ok":true,"exitCode":0}',
          isError: false,
          excerpt: '{"ok":true,"exitCode":0}',
        },
      ],
    });

    const evaluation = evaluateContextUsefulness(observation);
    expect(evaluation.verdict).toBe("helpful");
    expect(evaluation.finalScore).toBeGreaterThanOrEqual(0.35);
    expect(evaluation.signals.followUp.class).toBe("confirmation");
    expect(evaluation.signals.toolOutcome.class).toBe("success");
    expect(evaluation.signals.routeIntegrity.class).toBe("exact_full_serve");
    expect(evaluation.signals.authorityGate.blocked).toBe(false);
    expect(evaluation.reason).toContain("follow-up=confirmation");
  });

  it("scales an explicit correction with tool failure into a harmful verdict", () => {
    const observation = makeObservation({
      id: "harmful",
      followUpText: "No, that's wrong. Please try again.",
      bindingMode: "unbound",
      selectionMetadata: {
        servedPartial: true,
        contextClipped: true,
        compileDeadlineHit: true,
        brainDropReason: "assembly_fail_open",
      },
      toolResults: [
        {
          sourceRole: "tool",
          toolCallId: "call_2",
          toolName: "bash",
          input: "pnpm test",
          output: '{"ok":false,"exitCode":2,"error":"ENOENT"}',
          isError: true,
          excerpt: '{"ok":false,"exitCode":2,"error":"ENOENT"}',
        },
      ],
    });

    const evaluation = evaluateContextUsefulness(observation);
    expect(evaluation.verdict).toBe("harmful");
    expect(evaluation.finalScore).toBeLessThanOrEqual(-0.35);
    expect(evaluation.signals.followUp.class).toBe("correction");
    expect(["failure", "error"]).toContain(evaluation.signals.toolOutcome.class);
    expect(evaluation.signals.routeIntegrity.class).toBe("unbound");
    expect(evaluation.signals.authorityGate.blocked).toBe(true);
    expect(evaluation.reason).toContain("human follow-up blocks promotion");
  });

  it("keeps ambiguous acknowledgements near neutral", () => {
    const observation = makeObservation({
      id: "neutral",
      followUpText: "Thanks.",
      toolResults: [],
      bindingMode: "trace_id",
    });

    const evaluation = evaluateContextUsefulness(observation);
    expect(evaluation.verdict).toBe("irrelevant");
    expect(evaluation.finalScore).toBeGreaterThan(-0.35);
    expect(evaluation.finalScore).toBeLessThan(0.35);
    expect(evaluation.signals.followUp.class).toBe("confirmation");
    expect(evaluation.signals.toolOutcome.class).toBe("missing");
  });
});
