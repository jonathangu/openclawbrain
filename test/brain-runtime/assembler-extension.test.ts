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
