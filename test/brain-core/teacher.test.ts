import { describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import type { BrainObservation } from "../../src/brain-core/types.js";
import { BrainTeacher, isTeacherEligibleObservation, materializeTeacherLabelInput } from "../../src/brain-core/teacher.js";

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
        sourceUri: "PLAYBOOK.md",
        tags: ["git", "pr"],
        tokenCount: 24,
        contentPreview: "Use gh pr create for pull request workflows.",
      },
    ],
    routeMetadata: {
      requestDigest: "deadbeefcafebabe",
      activePackId: "brain-pack-v3",
      routerIdentity: "brain-graph-traverse.v1",
      candidateNodeIds: ["node_pr", "node_review"],
      selectedNodeIds: ["node_pr"],
      selectedPathNodeIds: ["node_pr"],
      sourceSummary: {
        injectedCount: 1,
        kinds: { workflow: 1 },
        trusts: { human: 1 },
        sourceUris: ["PLAYBOOK.md"],
      },
      selectionMetadata: {
        traceSliceVersion: 1,
        queryChars: 29,
        budgetChars: 4000,
        maxHops: 8,
        seedCount: 2,
        candidateCount: 2,
        hopCount: 1,
        firedCount: 1,
        vetoedCount: 0,
        chosenSeedNodeId: "node_pr",
        routeSelectionMs: 9,
        embeddingMs: 3,
        totalQueryMs: 14,
        queryEmbeddingSource: "provided",
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

describe("teacher observation plumbing", () => {
  it("materializes the persisted observation surface for teacher-v2", () => {
    const input = materializeTeacherLabelInput(makeObservation());

    expect(input).toMatchObject({
      version: 2,
      observationId: "bo_1",
      traceId: "bt_1",
      queryText: "how do I open a pull request?",
      routeMetadata: {
        selectedNodeIds: ["node_pr"],
      },
      selectedContext: [
        expect.objectContaining({
          nodeId: "node_pr",
          contentPreview: expect.stringContaining("gh pr create"),
        }),
      ],
      assistantResponse: expect.stringContaining("gh pr create"),
      nextUserTurn: "That worked, thanks.",
    });
    expect(isTeacherEligibleObservation(makeObservation())).toBe(true);
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
      retrievalRelevance: 0.9,
      agentUsage: 0.7,
      outcomeSupport: 0.8,
      finalScore: 0.82,
      confidence: 0.64,
      reason: "retrieved context was relevant and the assistant used it well",
      input: {
        routeMetadata: {
          selectedNodeIds: ["node_pr"],
        },
        toolResults: [
          expect.objectContaining({
            toolName: "bash",
          }),
        ],
      },
    });
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
});
