import { describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import type { BrainObservation, DecisionPointSnapshotV1 } from "../../src/brain-core/types.js";
import {
  BrainTeacher,
  isTeacherEligibleObservation,
  materializeDecisionPointSnapshots,
  materializeTeacherLabelInput,
} from "../../src/brain-core/teacher.js";

function makeObservation(overrides: Partial<BrainObservation> = {}): BrainObservation {
  return {
    id: "bo_1",
    episodeId: "ep_1",
    conversationId: 42,
    traceId: "bt_1",
    queryText: "how do I open a pull request?",
    retrievedContext: [
      {
        nodeId: "node_pr",
        kind: "workflow",
        trust: "human",
        provenanceRef: "prov_playbook",
        sourceUri: "PLAYBOOK.md",
        tags: ["git", "pr"],
        tokenCount: 24,
        contentPreview: "Use gh pr create for pull request workflows.",
      },
    ],
    routeMetadata: {
      requestDigest: "deadbeefcafebabe",
      activePackId: "brain-pack-v3",
      routerIdentity: "brain-graph-traverse.v2",
      bindingMode: "exact_decision_id",
      serveDecisionRecordId: "decision-1",
      selectionDigest: "selection-digest-1",
      turnCompileEventId: "evt-compile-1",
      decisionRecordedAt: "2026-03-25T01:02:03.000Z",
      activePackEventExportDigest: "export-digest-1",
      activePackGraphChecksum: "graph-checksum-1",
      activePackRouterChecksum: "router-checksum-1",
      activePackBuiltAt: "2026-03-25T01:00:00.000Z",
      servedArtifact: {
        kind: "runtime_compile_v1",
        packId: "brain-pack-v3",
      },
      candidateNodeIds: ["node_pr", "node_review"],
      selectedNodeIds: ["node_pr"],
      selectedTraversalNodeIds: ["node_pr"],
      selectedPathNodeIds: ["node_pr"],
      selectedSeedNodeIds: ["node_pr"],
      sourceSummary: {
        injectedCount: 1,
        kinds: { workflow: 1 },
        trusts: { human: 1 },
        sourceUris: ["PLAYBOOK.md"],
        sourceRefs: ["prov_playbook"],
      },
      selectionMetadata: {
        traceSliceVersion: 4,
        queryChars: 29,
        budgetChars: 4000,
        maxHops: 8,
        maxFanoutPerNode: 4,
        maxFrontierSize: 32,
        seedCount: 2,
        seedSelectionCount: 1,
        candidateCount: 2,
        hopCount: 1,
        expansionCount: 1,
        selectionSubstepCount: 2,
        firedCount: 1,
        vetoedCount: 0,
        chosenSeedNodeId: "node_pr",
        selectedSeedNodeIds: ["node_pr"],
        routeSelectionMs: 9,
        embeddingMs: 3,
        totalQueryMs: 14,
        queryEmbeddingSource: "provided",
        chosenStopCount: 0,
        forcedStopCount: 1,
        droppedProposalCount: 1,
        droppedProposalReasons: {
          missing_target_node: 1,
        },
      },
    },
    assistantResponse: "Use `gh pr create` to open the pull request.",
    toolResults: [
      {
        sourceRole: "tool",
        toolCallId: "call_1",
        toolName: "bash",
        input: "{\"cmd\":\"gh pr create\"}",
        output: "{\"ok\":true}",
        isError: false,
        excerpt: "{\"ok\":true}",
      },
    ],
    followUpText: "That worked, thanks.",
    phase1Score: null,
    phase2Score: null,
    finalScore: null,
    confidence: null,
    reason: null,
    status: "pending_teacher",
    teacherEvaluation: null,
    createdAt: 123,
    updatedAt: 123,
    evaluatedAt: null,
    ...overrides,
  };
}

function makeDecisionPointSnapshot(): DecisionPointSnapshotV1 {
  return {
    schemaVersion: 1,
    decisionPointId: "dp_1",
    traceId: "bt_1",
    episodeId: "ep_1",
    conversationId: 42,
    sourceNodeId: "node_pr",
    expansionIndex: 0,
    selectionIndex: 0,
    decisionPointKind: "local",
    localActionSet: [
      {
        actionId: "traverse:node_pr",
        actionKind: "traverse",
        nodeId: "node_pr",
        toolName: null,
        toolCapabilityId: null,
        toolInstanceId: null,
        toolArgsShape: null,
        priorScore: 1,
        probability: 0.8,
        retrievalFeatures: null,
      },
    ],
    chosenActionId: "traverse:node_pr",
    chosenActionKind: "traverse",
    chosenNodeId: "node_pr",
    chosenToolName: null,
    chosenToolCapabilityId: null,
    chosenToolInstanceId: null,
    chosenActionProbability: 0.8,
    stopProbability: 0.2,
    stopTruth: null,
    stopReason: null,
    budgetContext: {
      budgetRemaining: 100,
      initialBudget: 100,
      reservedTokenCost: 0,
      budgetUsed: 0,
      budgetUsedFraction: 0,
      maxHops: 8,
      maxFrontierSize: 32,
      frontierSize: 1,
      visitedCount: 1,
      firedCount: 1,
      pendingSelectionCount: 0,
      pressureLevel: null,
      frontierPressure: null,
      budgetPressure: null,
      budgetFraction: null,
      queryBudgetChars: 4000,
      maxContextChars: 240,
      injectedChars: 180,
      droppedChars: 72,
      contextClipped: true,
      routeSelectionMs: 9,
      totalQueryMs: 14,
      compileDeadlineMs: null,
      compileDeadlineHit: null,
    },
    routeContext: {
      requestDigest: "deadbeefcafebabe",
      activePackId: "brain-pack-v3",
      routerIdentity: "brain-graph-traverse.v2",
      candidateNodeIds: ["node_pr", "node_review"],
      selectedNodeIds: ["node_pr"],
      selectedTraversalNodeIds: ["node_pr"],
      selectedSeedNodeIds: ["node_pr"],
    },
  };
}

describe("teacher observation plumbing", () => {
  it("materializes the persisted observation surface for teacher-v2", () => {
    const input = materializeTeacherLabelInput(makeObservation());

    expect(input).toMatchObject({
      version: 2,
      observationId: "bo_1",
      traceId: "bt_1",
      queryText: expect.stringContaining("[redacted query chars="),
      routeMetadata: {
        bindingMode: "exact_decision_id",
        serveDecisionRecordId: "decision-1",
        selectionDigest: "selection-digest-1",
        selectedNodeIds: ["node_pr"],
        selectionMetadata: {
          traceSliceVersion: 4,
          chosenStopCount: 0,
          forcedStopCount: 1,
          droppedProposalCount: 1,
          droppedProposalReasons: {
            missing_target_node: 1,
          },
        },
      },
      selectedContext: [
        expect.objectContaining({
          nodeId: "node_pr",
          provenanceRef: "prov_playbook",
          sourceUri: null,
          contentPreview: expect.stringContaining("[redacted source_content chars="),
        }),
      ],
      assistantResponse: expect.stringContaining("[redacted assistant_response chars="),
      nextUserTurn: expect.stringContaining("[redacted follow_up chars="),
    });
    expect(input?.routeMetadata.selectionMetadata).not.toHaveProperty("maxContextChars");
    expect(input?.routeMetadata.selectionMetadata).not.toHaveProperty("contextClipped");
    expect(isTeacherEligibleObservation(makeObservation())).toBe(true);
  });

  it("materializes persisted decision-point snapshots for downstream teacher input", () => {
    const observation = makeObservation({
      routeMetadata: {
        ...makeObservation().routeMetadata,
        selectionMetadata: {
          ...makeObservation().routeMetadata.selectionMetadata!,
          decisionPointSnapshots: [makeDecisionPointSnapshot()],
        },
      },
    });

    const snapshots = materializeDecisionPointSnapshots(observation);

    expect(snapshots).toEqual([makeDecisionPointSnapshot()]);
    expect(snapshots).not.toBe(observation.routeMetadata.selectionMetadata!.decisionPointSnapshots);
    snapshots![0].localActionSet[0].actionId = "mutated";
    expect(observation.routeMetadata.selectionMetadata!.decisionPointSnapshots?.[0]?.localActionSet[0]?.actionId).toBe(
      "traverse:node_pr",
    );
  });

  it("preserves persisted post-injection clip attribution in teacher input", () => {
    const observation = makeObservation({
      routeMetadata: {
        ...makeObservation().routeMetadata,
        selectionMetadata: {
          ...makeObservation().routeMetadata.selectionMetadata!,
          compileElapsedMs: 12,
          brainDropReason: "injection_cap_clipped",
          brainDropStage: "injection",
          budgetFraction: 0.3,
          servedPartial: true,
          maxContextChars: 240,
          injectedChars: 180,
          droppedChars: 72,
          contextClipped: true,
          fitStrategy: "structured_node_budget",
          retrievedNodeCount: 3,
          fittedNodeCount: 2,
          droppedNodeCount: 1,
          fittingDropReasons: {
            omitted_for_max_context_chars: 1,
          },
        },
      },
    });

    const input = materializeTeacherLabelInput(observation);

    expect(input?.routeMetadata.selectionMetadata).toMatchObject({
      budgetChars: 4000,
      compileElapsedMs: 12,
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      servedPartial: true,
      maxContextChars: 240,
      injectedChars: 180,
      droppedChars: 72,
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 2,
      droppedNodeCount: 1,
      fittingDropReasons: {
        omitted_for_max_context_chars: 1,
      },
    });
  });

  it("preserves interruption truth in teacher input when routing is cut short", () => {
    const observation = makeObservation({
      routeMetadata: {
        ...makeObservation().routeMetadata,
        selectionMetadata: {
          ...makeObservation().routeMetadata.selectionMetadata!,
          compileElapsedMs: 12,
          compileDeadlineMs: 10,
          compileDeadlineHit: true,
          brainDropReason: "deadline_after_query",
          brainDropStage: "query",
          queryInterrupted: true,
          interruptionStage: "query",
          interruptionReason: "deadline_after_query",
          servedPartial: false,
        },
      },
    });

    const input = materializeTeacherLabelInput(observation);

    expect(input?.routeMetadata.selectionMetadata).toMatchObject({
      compileElapsedMs: 12,
      compileDeadlineMs: 10,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: true,
      interruptionStage: "query",
      interruptionReason: "deadline_after_query",
      servedPartial: false,
    });
  });

  it("rejects observations without a teacher-eligible query", () => {
    const observation = makeObservation({ queryText: "   " });

    expect(materializeTeacherLabelInput(observation)).toBeNull();
    expect(isTeacherEligibleObservation(observation)).toBe(false);
  });

  it("evaluates full-turn observations with retrieval, agent, and outcome scores", async () => {
    const complete = vi.fn(async () => ({
      content: [{
        text: "{\"retrieval_relevance\":0.9,\"agent_usage\":0.7,\"outcome_support\":0.8,\"final_score\":0.82,\"confidence\":0.64,\"reason\":\"retrieved context was relevant and the assistant used it well\"}",
      }],
    }));
    const teacher = new BrainTeacher(
      complete,
      () => ({ provider: "openai", model: "gpt-4.1-mini" }),
      async () => "api-key",
      new BrainGraph(),
      { info: vi.fn(), error: vi.fn() },
    );

    const result = await teacher.evaluateObservation(makeObservation());

    expect(result).toMatchObject({
      version: 2,
      observationId: "bo_1",
      traceId: "bt_1",
      serveDecisionRecordId: "decision-1",
      selectionDigest: "selection-digest-1",
      turnCompileEventId: "evt-compile-1",
      activePackGraphChecksum: "graph-checksum-1",
      bindingMode: "exact_decision_id",
      retrievalRelevance: 0.9,
      agentUsage: 0.7,
      outcomeSupport: 0.8,
      confidence: 0.64,
      reason: "retrieved context was relevant and the assistant used it well",
      input: {
        routeMetadata: {
          persistenceMode: "redacted",
          selectedNodeIds: ["node_pr"],
        },
        toolResults: [
          expect.objectContaining({
            toolName: "bash",
          }),
        ],
      },
    });
    expect(result?.finalScore).toBeCloseTo(0.82346, 5);
    expect(complete).toHaveBeenCalledTimes(1);
    const request = ((complete.mock.calls as unknown) as Array<[{
      system?: string;
      messages?: Array<{ content?: unknown }>;
    }]>) [0]?.[0];
    expect(request?.system).toContain("persisted OpenClawBrain turn observation");
    const prompt = String(request?.messages?.[0]?.content ?? "");
    expect(prompt).toContain("\"assistantResponse\"");
    expect(prompt).toContain("\"toolResults\"");
    expect(prompt).toContain("\"nextUserTurn\"");
  });

  it("adds reward-shaping bonus for selective branching with a clean chosen stop", async () => {
    const complete = vi.fn(async () => ({
      content: [{
        text: "{\"retrieval_relevance\":0.9,\"agent_usage\":0.8,\"outcome_support\":0.7,\"final_score\":0.45,\"confidence\":0.7,\"reason\":\"assistant used the selected route cleanly\"}",
      }],
    }));
    const teacher = new BrainTeacher(
      complete,
      () => ({ provider: "openai", model: "gpt-4.1-mini" }),
      async () => "api-key",
      new BrainGraph(),
      { info: vi.fn(), error: vi.fn() },
    );

    const observation = makeObservation({
      routeMetadata: {
        ...makeObservation().routeMetadata,
        candidateNodeIds: ["node_pr", "node_review", "node_a", "node_b", "node_c"],
        selectionMetadata: {
          ...makeObservation().routeMetadata.selectionMetadata!,
          candidateCount: 5,
          firedCount: 1,
          chosenStopCount: 1,
          forcedStopCount: 0,
          droppedProposalCount: 0,
          droppedProposalReasons: null,
          totalQueryMs: 12,
        },
      },
    });

    const result = await teacher.evaluateObservation(observation);

    expect(result?.finalScore).toBeCloseTo(0.56712, 5);
    expect(result?.finalScore ?? 0).toBeGreaterThan(0.45);
  });

  it("penalizes precision-sensitive routes that clip context to save time", async () => {
    const complete = vi.fn(async () => ({
      content: [{
        text: "{\"retrieval_relevance\":0.8,\"agent_usage\":0.8,\"outcome_support\":0.8,\"final_score\":0.8,\"confidence\":0.7,\"reason\":\"assistant mostly used the route\"}",
      }],
    }));
    const teacher = new BrainTeacher(
      complete,
      () => ({ provider: "openai", model: "gpt-4.1-mini" }),
      async () => "api-key",
      new BrainGraph(),
      { info: vi.fn(), error: vi.fn() },
    );

    const observation = makeObservation({
      queryText: "what exact command and file path should I use?",
      routeMetadata: {
        ...makeObservation().routeMetadata,
        candidateNodeIds: ["node_pr", "node_review", "node_a", "node_b"],
        selectedNodeIds: ["node_pr", "node_review", "node_a"],
        selectedTraversalNodeIds: ["node_pr", "node_review", "node_a"],
        selectedPathNodeIds: ["node_pr", "node_review", "node_a"],
        selectionMetadata: {
          ...makeObservation().routeMetadata.selectionMetadata!,
          candidateCount: 4,
          firedCount: 3,
          chosenStopCount: 0,
          forcedStopCount: 1,
          droppedProposalCount: 0,
          totalQueryMs: 18,
          compileDeadlineHit: true,
          servedPartial: true,
          contextClipped: true,
          droppedNodeCount: 1,
        },
      },
    });

    const result = await teacher.evaluateObservation(observation);

    expect(result?.finalScore).toBeCloseTo(0.696, 5);
    expect(result?.finalScore ?? 1).toBeLessThan(0.8);
  });
});
