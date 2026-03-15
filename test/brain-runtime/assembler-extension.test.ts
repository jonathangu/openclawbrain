import { describe, expect, it, vi } from "vitest";
import { BrainAssemblerExtension } from "../../src/brain-runtime/assembler-extension.js";
import type { TraversalResult } from "../../src/brain-core/types.js";

function createBrainStub(overrides?: {
  query?: (params: { conversationId: number; queryText: string; budgetChars: number }) => Promise<TraversalResult | null>;
}) {
  return {
    isEnabled: () => true,
    isInitialized: () => true,
    isEmbeddingConfigured: () => true,
    noteAssemblyDecision: vi.fn(),
    query: overrides?.query ?? vi.fn(async () => null),
  };
}

describe("BrainAssemblerExtension", () => {
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

  it("records episode and trace ids when the brain fires", async () => {
    const brain = createBrainStub({
      query: async () => ({
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
      }),
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
});
