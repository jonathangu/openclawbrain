import { describe, expect, it, vi } from "vitest";
import { BrainAssemblerExtension } from "../../src/brain-runtime/assembler-extension.js";
import type { TraversalResult } from "../../src/brain-core/types.js";

function makeTraversalResult(): TraversalResult {
  return {
    fired: [
      {
        nodeId: "bn_1",
        kind: "correction",
        content: "Use gh pr create for pull requests.",
        tokenCount: 8,
      },
    ],
    vetoed: [],
    episode: {
      id: "ep_1",
      conversationId: 42,
      queryText: "how do i open a pull request",
      queryEmbedding: null,
      trajectory: [],
      firedNodes: ["bn_1"],
      vetoedNodes: [],
      contextChars: 64,
      reward: null,
      rewardSource: null,
      packVersion: 1,
      createdAt: Date.now(),
    },
    trace: {
      id: "tr_1",
      episodeId: "ep_1",
      packVersion: 1,
      queryText: "how do i open a pull request",
      seedScores: [],
      trajectory: [],
      firedNodes: ["bn_1"],
      vetoedNodes: [],
      contextChars: 64,
      footer: "[brain] used graph retrieval for this turn.",
      createdAt: Date.now(),
    },
  };
}

function makeStructuredTraversalResult(): TraversalResult {
  return {
    fired: [
      {
        nodeId: "bn_1",
        kind: "correction",
        content: "Use gh pr create for pull requests and include the tail-marker-never-rendered suffix so fallback would be obvious.",
        tokenCount: 20,
      },
      {
        nodeId: "bn_2",
        kind: "chunk",
        content: "Deployment evidence tail-marker-never-rendered with raw transcript detail that should stay out of the compact summary.",
        tokenCount: 18,
      },
      {
        nodeId: "bn_3",
        kind: "workflow",
        content: "Check CI, inspect logs, then retry tail-marker-never-rendered after you understand the failure.",
        tokenCount: 16,
      },
    ],
    vetoed: [],
    episode: {
      id: "ep_structured_1",
      conversationId: 42,
      queryText: "how do i open a pull request",
      queryEmbedding: null,
      trajectory: [],
      firedNodes: ["bn_1", "bn_2", "bn_3"],
      vetoedNodes: [],
      contextChars: 196,
      reward: null,
      rewardSource: null,
      packVersion: 7,
      createdAt: Date.now(),
    },
    trace: {
      id: "tr_structured_1",
      episodeId: "ep_structured_1",
      packVersion: 7,
      queryText: "how do i open a pull request",
      seedScores: [],
      trajectory: [],
      firedNodes: ["bn_1", "bn_2", "bn_3"],
      vetoedNodes: [],
      contextChars: 196,
      footer: "Brain v7 · 2 seed candidates · 1 seed picks · 2 expansions · 3 fired · 0 veto · 196 chars · trace tr_structured_1",
      routeTrace: {
        requestDigest: "feedfacec0ffee12",
        conversationId: 42,
        activePackId: "brain-pack-v7",
        routerIdentity: "brain-graph-traverse.v2",
        candidateNodeIds: ["bn_1", "bn_2", "bn_3", "bn_4"],
        selectedNodeIds: ["bn_1", "bn_2", "bn_3"],
        selectedTraversalNodeIds: ["bn_1", "bn_2", "bn_3"],
        selectedPathNodeIds: ["bn_1", "bn_2", "bn_3"],
        selectedSeedNodeIds: ["bn_1"],
        injectedNodeSummaries: [
          {
            nodeId: "bn_1",
            kind: "correction",
            trust: "human",
            sourceUri: "PLAYBOOK.md",
            tags: ["pull-request"],
            tokenCount: 20,
            contentPreview: "Use gh pr create for pull requests, keep the flow operator-auditable, and avoid the tail-marker-never-rendered suffix in compact rendering.",
          },
          {
            nodeId: "bn_2",
            kind: "chunk",
            trust: "scanner",
            sourceUri: "docs/deploy.md",
            tags: ["deploy"],
            tokenCount: 18,
            contentPreview: "Deployment evidence says to inspect CI before retrying, but compact rendering should not replay every raw detail tail-marker-never-rendered.",
          },
          {
            nodeId: "bn_3",
            kind: "workflow",
            trust: "scanner",
            sourceUri: "docs/deploy.md",
            tags: ["workflow"],
            tokenCount: 16,
            contentPreview: "Check CI, inspect logs, then retry with context once the failure mode is understood.",
          },
        ],
        sourceSummary: {
          injectedCount: 3,
          kinds: { correction: 1, chunk: 1, workflow: 1 },
          trusts: { human: 1, scanner: 2 },
          sourceUris: ["PLAYBOOK.md", "docs/deploy.md"],
        },
        selectionMetadata: {
          traceSliceVersion: 2,
          queryChars: 29,
          budgetChars: 1024,
          maxHops: 8,
          maxFanoutPerNode: 4,
          maxFrontierSize: 32,
          seedCount: 2,
          seedSelectionCount: 1,
          candidateCount: 4,
          hopCount: 2,
          expansionCount: 2,
          selectionSubstepCount: 4,
          firedCount: 3,
          vetoedCount: 0,
          chosenSeedNodeId: "bn_1",
          selectedSeedNodeIds: ["bn_1"],
          routeSelectionMs: 12,
          embeddingMs: 4,
          totalQueryMs: 18,
          queryEmbeddingSource: "runtime",
        },
      },
      createdAt: Date.now(),
    },
  };
}

function createBrainStub(overrides?: {
  enabled?: boolean;
  initialized?: boolean;
  embeddingConfigured?: boolean;
  shadowMode?: boolean;
  query?: (params: { conversationId: number; queryText: string; budgetChars: number }) => Promise<TraversalResult | null>;
}) {
  return {
    isEnabled: () => overrides?.enabled ?? true,
    isInitialized: () => overrides?.initialized ?? true,
    isEmbeddingConfigured: () => overrides?.embeddingConfigured ?? true,
    isShadowMode: () => overrides?.shadowMode ?? false,
    noteAssemblyDecision: vi.fn(),
    query: overrides?.query ?? vi.fn(async () => null),
  };
}

describe("BrainAssemblerExtension", () => {
  it("skips missing query text with an explicit reason", async () => {
    const brain = createBrainStub();
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 7,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "assistant", content: "no fresh user text" }],
        estimatedTokens: 4,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "assistant", content: "no fresh user text" }],
    });

    expect(result.brainDecision?.mode).toBe("skip_no_query");
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith({
      mode: "skip_no_query",
      conversationId: 7,
      footer: "[brain] bypassed: no user query text.",
    });
  });

  it("skips short static lookups with an explicit reason", async () => {
    const brain = createBrainStub();
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 7,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "open src/engine.ts" }],
        estimatedTokens: 4,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "user", content: "open src/engine.ts" }],
    });

    expect(result.brainDecision?.mode).toBe("skip_short_static_lookup");
    expect(result.brainDecision?.footer).toContain("bypassed");
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith({
      mode: "skip_short_static_lookup",
      conversationId: 7,
      footer: "[brain] bypassed: short static lookup.",
    });
  });

  it("prefers explicit file-open commands over procedural keyword matches", async () => {
    const brain = createBrainStub();
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 8,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "open PLAYBOOK.md" }],
        estimatedTokens: 4,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "user", content: "open PLAYBOOK.md" }],
    });

    expect(result.brainDecision?.mode).toBe("skip_short_static_lookup");
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith({
      mode: "skip_short_static_lookup",
      conversationId: 8,
      footer: "[brain] bypassed: short static lookup.",
    });
  });

  it("returns unchanged assembly when disabled or uninitialized", async () => {
    const brain = createBrainStub({ enabled: false });
    const extension = new BrainAssemblerExtension(brain as never);

    const assembled = {
      messages: [{ role: "user", content: "How do I open a pull request?" }],
      estimatedTokens: 6,
      stats: {
        rawMessageCount: 1,
        summaryCount: 0,
        totalContextItems: 1,
      },
    };

    const result = await extension.augmentAssembly({
      conversationId: 9,
      tokenBudget: 4096,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.messages).toEqual(assembled.messages);
    expect(result.estimatedTokens).toBe(assembled.estimatedTokens);
    expect(result.brainDecision?.mode).toBe("skip_uninitialized");
  });

  it("renders compact route-trace summaries with provenance when trace summaries exist", async () => {
    const brain = createBrainStub({
      query: async () => makeStructuredTraversalResult(),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "live tail" }],
        estimatedTokens: 2,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    const injected = String(result.messages[0]?.content ?? "");
    expect(injected).toContain("## Provenance And Audit");
    expect(injected).toContain("Pack brain-pack-v7 · 3 injected nodes · 2 sources");
    expect(injected).toContain("Kinds: correction 1, chunk 1, workflow 1");
    expect(injected).toContain("Trusts: human 1, scanner 2");
    expect(injected).toContain("`bn_1` [correction/human] from PLAYBOOK.md");
    expect(injected).toContain("`bn_2` [chunk/scanner] from docs/deploy.md");
    expect(injected).toContain("Use gh pr create for pull requests");
    expect(injected).not.toContain("tail-marker-never-rendered suffix so fallback would be obvious");
    expect(injected).not.toContain("tail-marker-never-rendered with raw transcript detail");
    expect(injected).not.toContain("tail-marker-never-rendered after you understand the failure");
  });

  it("falls back to legacy verbatim rendering when route-trace summaries are unavailable", async () => {
    const brain = createBrainStub({
      query: async () => makeTraversalResult(),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "live tail" }],
        estimatedTokens: 2,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    const injected = String(result.messages[0]?.content ?? "");
    expect(injected).toContain("## Correction Cards");
    expect(injected).toContain("Use gh pr create for pull requests.");
    expect(injected).toContain("Trace: [brain] used graph retrieval for this turn.");
    expect(injected).not.toContain("## Provenance And Audit");
  });

  it("records episode and trace ids when the brain fires", async () => {
    const brain = createBrainStub({
      query: async () => makeTraversalResult(),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "live tail" }],
        estimatedTokens: 2,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.brainDecision).toEqual({
      mode: "use_brain",
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain] used graph retrieval for this turn.",
    });
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]?.content).toContain("Correction Cards");
  });

  it("records shadow routing without injecting learned context", async () => {
    const brain = createBrainStub({
      shadowMode: true,
      query: async () => makeTraversalResult(),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const assembled = {
      messages: [{ role: "user", content: "live tail" }],
      estimatedTokens: 2,
      stats: {
        rawMessageCount: 1,
        summaryCount: 0,
        totalContextItems: 1,
      },
    };

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request again?" }],
    });

    expect(result.messages).toEqual(assembled.messages);
    expect(result.estimatedTokens).toBe(assembled.estimatedTokens);
    expect(result.brainDecision).toEqual({
      mode: "shadow",
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain shadow] recorded routing without injecting learned context.",
    });
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith({
      mode: "shadow",
      conversationId: 42,
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain shadow] recorded routing without injecting learned context.",
    });
  });

  it("adds summary-routing guidance for precision-sensitive questions over summaries", async () => {
    const brain = createBrainStub({
      query: async () => makeTraversalResult(),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "summary tail" }],
        estimatedTokens: 2,
        summaryMetadata: {
          totalCount: 2,
          maxDepth: 3,
          condensedCount: 2,
          items: [
            {
              summaryId: "sum_1",
              kind: "condensed",
              depth: 3,
              descendantCount: 12,
              earliestAt: null,
              latestAt: null,
            },
          ],
        },
        stats: {
          rawMessageCount: 1,
          summaryCount: 1,
          totalContextItems: 2,
        },
      },
      liveMessages: [{ role: "user", content: "what exact file path was mentioned?" }],
    });

    expect(result.systemPromptAddition).toContain("expand toward source material before asserting exact details");
  });
});
