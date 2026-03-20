import { describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import type { DecisionTrace } from "../../src/brain-core/types.js";
import { BrainTeacher, isTeacherEligibleTrace, materializeTeacherLabelInput } from "../../src/brain-core/teacher.js";

function makeTrace(overrides: Partial<DecisionTrace> = {}): DecisionTrace {
  return {
    id: "bt_trace_1",
    episodeId: "ep_1",
    packVersion: 3,
    queryText: "how do I open a pull request?",
    seedScores: [],
    trajectory: [],
    firedNodes: ["node_pr"],
    vetoedNodes: [],
    contextChars: 128,
    footer: "trace footer",
    routeTrace: {
      requestDigest: "deadbeefcafebabe",
      conversationId: 42,
      activePackId: "brain-pack-v3",
      routerIdentity: "brain-graph-traverse.v1",
      candidateNodeIds: ["node_pr", "node_review"],
      selectedNodeIds: ["node_pr"],
      selectedPathNodeIds: ["node_pr"],
      injectedNodeSummaries: [
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
    createdAt: 123,
    ...overrides,
  };
}

describe("teacher traced label plumbing", () => {
  it("materializes a structured teacher-label input from persisted route traces", () => {
    const trace = makeTrace();

    const input = materializeTeacherLabelInput(trace);

    expect(input).toMatchObject({
      version: 1,
      traceId: "bt_trace_1",
      episodeId: "ep_1",
      queryText: "how do I open a pull request?",
      routeDecision: {
        requestDigest: "deadbeefcafebabe",
        routerIdentity: "brain-graph-traverse.v1",
        selectedNodeIds: ["node_pr"],
      },
      selectedContext: [
        expect.objectContaining({
          nodeId: "node_pr",
          sourceUri: "PLAYBOOK.md",
          contentPreview: expect.stringContaining("gh pr create"),
        }),
      ],
    });
    expect(isTeacherEligibleTrace(trace)).toBe(true);
  });

  it("rejects traces that do not carry a teacher-eligible route slice", () => {
    const trace = makeTrace({ routeTrace: null });

    expect(materializeTeacherLabelInput(trace)).toBeNull();
    expect(isTeacherEligibleTrace(trace)).toBe(false);
  });

  it("evaluates traced route decisions using only the structured trace surface", async () => {
    const complete = vi.fn(async () => ({
      content: [{ text: '{"score":0.75,"reason":"selected context is directly relevant"}' }],
    }));
    const teacher = new BrainTeacher(
      complete,
      () => ({ provider: "openai", model: "gpt-4.1-mini" }),
      async () => "api-key",
      new BrainGraph(),
      { info: vi.fn(), error: vi.fn() },
    );

    const result = await teacher.evaluateTrace(makeTrace());

    expect(result).toMatchObject({
      version: 1,
      traceId: "bt_trace_1",
      requestDigest: "deadbeefcafebabe",
      score: 0.75,
      reason: "selected context is directly relevant",
      input: {
        version: 1,
        routeDecision: {
          selectedNodeIds: ["node_pr"],
        },
      },
    });
    expect(complete).toHaveBeenCalledTimes(1);
    const request = ((complete.mock.calls as unknown) as Array<[{
      system?: string;
      messages?: Array<{ content?: unknown }>;
    }]>) [0]?.[0];
    expect(request?.system).toContain("traced context routing decision");
    const prompt = String(request?.messages?.[0]?.content ?? "");
    expect(prompt).toContain('"traceId": "bt_trace_1"');
    expect(prompt).toContain('"selectedContext"');
    expect(prompt).not.toContain("Candidate nodes the router could have chosen:");
  });
});
