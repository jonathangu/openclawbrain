import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BrainAssemblerExtension,
  resolveBrainQueryBudgetChars,
} from "../../src/brain-runtime/assembler-extension.js";
import type { BrainInterruptionMetadata, TraversalResult } from "../../src/brain-core/types.js";

const QUERY_BUDGET_CHARS_FOR_4096_TOKENS = resolveBrainQueryBudgetChars(4096, 0.3);

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
        branchOutcomes: [],
        injectedNodeSummaries: [
          {
            nodeId: "bn_1",
            kind: "correction",
            trust: "human",
            provenanceRef: "prov_bn_1",
            sourceUri: "PLAYBOOK.md",
            tags: ["pull-request"],
            tokenCount: 20,
            contentPreview: "Use gh pr create for pull requests, keep the flow operator-auditable, and avoid the tail-marker-never-rendered suffix in compact rendering.",
          },
          {
            nodeId: "bn_2",
            kind: "chunk",
            trust: "scanner",
            provenanceRef: "prov_bn_2",
            sourceUri: "docs/deploy.md",
            tags: ["deploy"],
            tokenCount: 18,
            contentPreview: "Deployment evidence says to inspect CI before retrying, but compact rendering should not replay every raw detail tail-marker-never-rendered.",
          },
          {
            nodeId: "bn_3",
            kind: "workflow",
            trust: "scanner",
            provenanceRef: "prov_bn_3",
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
          sourceRefs: ["prov_bn_1", "prov_bn_2", "prov_bn_3"],
        },
        selectionMetadata: {
          traceSliceVersion: 4,
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
          chosenStopCount: 0,
          forcedStopCount: 1,
          branchOutcomeSummary: {
            branchCount: 1,
            continuingBranchCount: 1,
            stoppedWithoutProgressCount: 0,
            chosenStopBranchCount: 0,
            forcedStopBranchCount: 1,
            terminationReasons: {
              fanout_cap: 1,
            },
            detail: "1/1 branches continued; chosen=0; forced=1; reasons fanout_cap=1",
          },
          droppedProposalCount: 0,
          droppedProposalReasons: null,
          interruptionAccounting: null,
        },
      },
      createdAt: Date.now(),
    },
  };
}

function makeConflictAwareStructuredTraversalResult(): TraversalResult {
  const result = makeStructuredTraversalResult();
  const routeTrace = result.trace.routeTrace!;
  routeTrace.injectedNodeSummaries = [
    {
      ...routeTrace.injectedNodeSummaries[0],
      contentPreview: "Use gh pr create for pull requests.",
      correctionState: "current",
      correctionSubjectKey: "pull_request_command",
      correctionSubjectText: "pull request command",
      correctionNeedsSourceExpansion: false,
    },
    {
      nodeId: "bn_conflict_a",
      kind: "correction",
      trust: "human",
      provenanceRef: "prov_bn_conflict_a",
      sourceUri: "MEMORY.md",
      tags: ["pull-request"],
      tokenCount: 12,
      contentPreview: "Use hub pull-request for pull requests in this repo.",
      correctionState: "conflicting",
      correctionSubjectKey: "pull_request_command",
      correctionSubjectText: "pull request command",
      correctionConflictSetId: "cm_conflict_pull_request_command",
      correctionNeedsSourceExpansion: true,
    },
    {
      nodeId: "bn_conflict_b",
      kind: "correction",
      trust: "human",
      provenanceRef: "prov_bn_conflict_b",
      sourceUri: "MEMORY.md",
      tags: ["pull-request"],
      tokenCount: 11,
      contentPreview: "Use gh pr create for pull requests, but prior memory disagrees.",
      correctionState: "conflicting",
      correctionSubjectKey: "pull_request_command",
      correctionSubjectText: "pull request command",
      correctionConflictSetId: "cm_conflict_pull_request_command",
      correctionNeedsSourceExpansion: true,
    },
    {
      ...routeTrace.injectedNodeSummaries[1],
    },
  ];
  routeTrace.sourceSummary = {
    injectedCount: 4,
    kinds: { correction: 3, chunk: 1 },
    trusts: { human: 3, scanner: 1 },
    sourceUris: ["PLAYBOOK.md", "MEMORY.md", "docs/deploy.md"],
    sourceRefs: ["prov_bn_1", "prov_bn_conflict_a", "prov_bn_conflict_b", "prov_bn_2"],
  };
  return result;
}

function makeSupersededStructuredTraversalResult(): TraversalResult {
  const result = makeStructuredTraversalResult();
  const routeTrace = result.trace.routeTrace!;
  routeTrace.injectedNodeSummaries = [
    {
      ...routeTrace.injectedNodeSummaries[0],
      contentPreview: "Use gh pr create for pull requests.",
      correctionState: "current",
      correctionSubjectKey: "pull_request_command",
      correctionSubjectText: "pull request command",
      correctionNeedsSourceExpansion: false,
    },
    {
      nodeId: "bn_superseded",
      kind: "correction",
      trust: "human",
      provenanceRef: "prov_bn_superseded",
      sourceUri: "MEMORY.md",
      tags: ["pull-request"],
      tokenCount: 10,
      contentPreview: "Use hub pull-request for pull requests.",
      correctionState: "superseded",
      correctionSubjectKey: "pull_request_command",
      correctionSubjectText: "pull request command",
      correctionNeedsSourceExpansion: false,
    },
    {
      ...routeTrace.injectedNodeSummaries[1],
    },
  ];
  routeTrace.sourceSummary = {
    injectedCount: 3,
    kinds: { correction: 2, chunk: 1 },
    trusts: { human: 2, scanner: 1 },
    sourceUris: ["PLAYBOOK.md", "MEMORY.md", "docs/deploy.md"],
    sourceRefs: ["prov_bn_1", "prov_bn_superseded", "prov_bn_2"],
  };
  return result;
}

function createBrainStub(overrides?: {
  enabled?: boolean;
  initialized?: boolean;
  embeddingConfigured?: boolean;
  shadowMode?: boolean;
  compileDeadlineMs?: number | null;
  budgetFraction?: number;
  lastQueryInterruption?: BrainInterruptionMetadata | null;
  lastPrefetchDecision?: Record<string, unknown> | null;
  query?: (params: {
    conversationId: number;
    queryText: string;
    budgetChars: number;
    deadlineAtMs?: number | null;
  }) => Promise<TraversalResult | null>;
}) {
  return {
    isEnabled: () => overrides?.enabled ?? true,
    isInitialized: () => overrides?.initialized ?? true,
    isEmbeddingConfigured: () => overrides?.embeddingConfigured ?? true,
    isShadowMode: () => overrides?.shadowMode ?? false,
    getCompileDeadlineMs: () => overrides?.compileDeadlineMs ?? null,
    getBudgetFraction: () => overrides?.budgetFraction ?? 0.3,
    getLastQueryInterruption: () => overrides?.lastQueryInterruption ?? null,
    getLastPrefetchDecision: () => overrides?.lastPrefetchDecision ?? null,
    schedulePrefetch: vi.fn(async () => null),
    noteAssemblyDecision: vi.fn(),
    recordTraceSelectionMetadata: vi.fn(),
    query: overrides?.query ?? vi.fn(async () => null),
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

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
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "skip_no_query",
      conversationId: 7,
      footer: "[brain] bypassed: no user query text.",
      queryBudgetChars: 0,
      brainDropReason: "skip_no_query",
      brainDropStage: "decision",
      compileElapsedMs: expect.any(Number),
    }));
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
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "skip_short_static_lookup",
      conversationId: 7,
      footer: "[brain] bypassed: short static lookup.",
      queryBudgetChars: 0,
      brainDropReason: "skip_short_static_lookup",
      brainDropStage: "decision",
      compileElapsedMs: expect.any(Number),
    }));
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
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "skip_short_static_lookup",
      conversationId: 8,
      footer: "[brain] bypassed: short static lookup.",
      queryBudgetChars: 0,
      brainDropReason: "skip_short_static_lookup",
      brainDropStage: "decision",
      compileElapsedMs: expect.any(Number),
    }));
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

  it("renders conflicting correction clusters with an explicit source-expansion warning", async () => {
    const brain = createBrainStub({
      query: async () => makeConflictAwareStructuredTraversalResult(),
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
      liveMessages: [{ role: "user", content: "What are the conflicting pull request corrections?" }],
    });

    const injected = String(result.messages[0]?.content ?? "");
    expect(injected).toContain("Apply current correction cards directly. Treat superseded alternatives as disallowed answer content unless the user explicitly asks for history, migration, compatibility, or tradeoffs.");
    expect(injected).toContain("Warning: conflicting correction cluster retrieved; expand toward source before asserting exact current truth.");
    expect(injected).toContain("[conflict · human] Use hub pull-request for pull requests in this repo.");
    expect(injected).toContain("[conflict · human] Use gh pr create for pull requests, but prior memory disagrees.");
    expect(injected).toContain("`bn_conflict_a` [correction/conflicting/human] from MEMORY.md");
  });

  it("omits superseded correction cards from the default structured display", async () => {
    const brain = createBrainStub({
      query: async () => makeSupersededStructuredTraversalResult(),
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
    expect(injected).toContain("[human] Use gh pr create for pull requests.");
    expect(injected).not.toContain("Use hub pull-request for pull requests.");
    expect(injected).toContain("`bn_superseded` [correction/superseded/human] from MEMORY.md");
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
    expect(injected).toContain("Apply current correction cards directly. Treat superseded alternatives as disallowed answer content unless the user explicitly asks for history, migration, compatibility, or tradeoffs.");
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

    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "use_brain",
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain] used graph retrieval for this turn.",
      queryBudgetChars: expect.any(Number),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "none",
    }));
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
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "shadow",
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain shadow] recorded routing without injecting learned context.",
      queryBudgetChars: expect.any(Number),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "shadow_mode",
      brainDropStage: "injection",
    }));
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "shadow",
      conversationId: 42,
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain shadow] recorded routing without injecting learned context.",
      queryBudgetChars: expect.any(Number),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "shadow_mode",
      brainDropStage: "injection",
    }));
  });

  it("surfaces would-be clip metrics in shadow mode when maxContextChars trims the block", async () => {
    const brain = createBrainStub({
      shadowMode: true,
      query: async () => makeStructuredTraversalResult(),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 240,
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

    expect(result.messages).toEqual([{ role: "user", content: "live tail" }]);
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "shadow",
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      fittingDropReasons: expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "shadow_mode",
      brainDropStage: "injection",
    }));
    expect((result.brainDecision?.fittedNodeCount ?? 0)).toBeGreaterThan(0);
    expect((result.brainDecision?.fittedNodeCount ?? 0)).toBeLessThan(3);
    expect(result.brainDecision?.droppedNodeCount).toBe(3 - (result.brainDecision?.fittedNodeCount ?? 0));
    expect(brain.recordTraceSelectionMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ id: "tr_structured_1" }),
      expect.objectContaining({
        budgetFraction: 0.3,
        maxContextChars: 240,
        queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
        injectedChars: expect.any(Number),
        droppedChars: expect.any(Number),
        contextClipped: true,
        fitStrategy: "structured_node_budget",
        retrievedNodeCount: 3,
        fittedNodeCount: expect.any(Number),
        droppedNodeCount: expect.any(Number),
        fittingDropReasons: expect.objectContaining({
          omitted_for_max_context_chars: expect.any(Number),
        }),
        compileElapsedMs: expect.any(Number),
        brainDropReason: "shadow_mode",
        brainDropStage: "injection",
      }),
    );
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "shadow",
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      fittingDropReasons: expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "shadow_mode",
      brainDropStage: "injection",
    }));
  });

  it("caps injected context to explicit maxContextChars and records clip metrics", async () => {
    const brain = createBrainStub({
      lastPrefetchDecision: {
        enabled: true,
        state: "materialized",
        kind: "traversal",
        budgetClass: "standard",
        key: "prefetch-key",
        queryDigest: "deadbeefdeadbeef",
        activePackId: "brain-pack-v7",
        activePackVersion: 7,
        summaryRoutingMode: "ignore",
        prefetchMs: 9,
        cacheAgeMs: 3,
        invalidatedReason: null,
        reusedNodeCount: 3,
        reusedChars: 196,
        savingsChars: 196,
      },
      query: vi.fn(async () => makeStructuredTraversalResult()),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 240,
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

    expect(brain.query).toHaveBeenCalledWith({
      conversationId: 42,
      queryText: "How do I open a pull request?",
      budgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      summaryRoutingMode: "ignore",
    });
    expect(brain.schedulePrefetch).toHaveBeenCalledWith({
      conversationId: 42,
      queryText: "How do I open a pull request?",
      budgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      summaryRoutingMode: "ignore",
      deadlineAtMs: undefined,
    });
    expect(String(result.messages[0]?.content ?? "").length).toBeLessThanOrEqual(240);
    expect(String(result.messages[0]?.content ?? "")).toContain("[brain]");
    expect(String(result.messages[0]?.content ?? "")).not.toContain("## Provenance And Audit");
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "use_brain",
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      fittingDropReasons: expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      servedPartial: true,
      chosenStopCount: 0,
      forcedStopCount: 1,
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      prefetch: expect.objectContaining({
        state: "materialized",
        budgetClass: "standard",
      }),
    }));
    expect((result.brainDecision?.fittedNodeCount ?? 0)).toBeGreaterThan(0);
    expect((result.brainDecision?.fittedNodeCount ?? 0)).toBeLessThan(3);
    expect(result.brainDecision?.droppedNodeCount).toBe(3 - (result.brainDecision?.fittedNodeCount ?? 0));
    expect(brain.recordTraceSelectionMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ id: "tr_structured_1" }),
      expect.objectContaining({
        budgetFraction: 0.3,
        maxContextChars: 240,
        queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
        injectedChars: expect.any(Number),
        droppedChars: expect.any(Number),
        contextClipped: true,
        fitStrategy: "structured_node_budget",
        retrievedNodeCount: 3,
        fittedNodeCount: expect.any(Number),
        droppedNodeCount: expect.any(Number),
        fittingDropReasons: expect.objectContaining({
          omitted_for_max_context_chars: expect.any(Number),
        }),
        compileElapsedMs: expect.any(Number),
        brainDropReason: "injection_cap_clipped",
        brainDropStage: "injection",
        servedPartial: true,
        chosenStopCount: 0,
        forcedStopCount: 1,
        droppedProposalCount: 0,
        droppedProposalReasons: null,
        prefetch: expect.objectContaining({
          state: "materialized",
          budgetClass: "standard",
        }),
      }),
    );
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "use_brain",
      conversationId: 42,
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      fittingDropReasons: expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      servedPartial: true,
      chosenStopCount: 0,
      forcedStopCount: 1,
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      prefetch: expect.objectContaining({
        state: "materialized",
        budgetClass: "standard",
      }),
    }));
  });

  it("keeps correction cards first when a tight structured budget clips later evidence", async () => {
    const brain = createBrainStub({
      query: vi.fn(async () => makeStructuredTraversalResult()),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 130,
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
    expect(injected).toContain("[brain]");
    expect(injected).toContain("Use gh pr create for pull requests, keep the flow operator-auditable");
    expect(injected).not.toContain("Deployment evidence tail-marker-never-rendered");
    expect(injected).not.toContain("Check CI, inspect logs, then retry tail-marker-never-rendered");
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "use_brain",
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 1,
      droppedNodeCount: 2,
      fittingDropReasons: {
        omitted_for_max_context_chars: 2,
      },
      contextClipped: true,
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
    }));
  });

  it("reports structured fit metrics even when the structured block is not clipped", async () => {
    const brain = createBrainStub({
      query: vi.fn(async () => makeStructuredTraversalResult()),
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

    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "use_brain",
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 3,
      droppedNodeCount: 0,
      fittingDropReasons: null,
      contextClipped: false,
    }));
  });

  it("derives retrieval budget from the live brain budget fraction", async () => {
    const query = vi.fn(async () => makeTraversalResult());
    const brain = createBrainStub({
      budgetFraction: 0.5,
      query,
    });
    const extension = new BrainAssemblerExtension(brain as never);
    const expectedQueryBudgetChars = resolveBrainQueryBudgetChars(4096, 0.5);

    await extension.augmentAssembly({
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

    expect(query).toHaveBeenCalledWith({
      conversationId: 42,
      queryText: "How do I open a pull request?",
      budgetChars: expectedQueryBudgetChars,
      summaryRoutingMode: "ignore",
    });
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      budgetFraction: 0.5,
      queryBudgetChars: expectedQueryBudgetChars,
    }));
  });

  it("warns when summary context includes stale or superseded lineages", async () => {
    const brain = createBrainStub({
      query: vi.fn(async () => makeTraversalResult()),
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      assembled: {
        messages: [{ role: "user", content: "What exact command should I use?" }],
        estimatedTokens: 4,
        systemPromptAddition: undefined,
        summaryMetadata: {
          totalCount: 1,
          maxDepth: 0,
          condensedCount: 0,
          episodeCount: 0,
          snapshotCount: 0,
          branchCount: 1,
          typedMemoryRefCount: 0,
          freshnessStateCounts: { fresh: 0, superseded: 1 },
          hasNonFreshSummaries: true,
          hasTruthConflict: false,
          latestRole: "support",
          items: [{
            summaryId: "sum_stale",
            kind: "leaf",
            depth: 0,
            descendantCount: 0,
            earliestAt: null,
            latestAt: null,
            freshnessState: "superseded",
          }],
        },
        stats: {
          rawMessageCount: 1,
          summaryCount: 1,
          totalContextItems: 2,
        },
      },
      liveMessages: [{ role: "user", content: "What exact command should I use?" }],
    });

    expect(result.brainDecision?.mode).toBe("use_brain");
    expect(result.systemPromptAddition).toContain("stale or superseded");
    expect(result.systemPromptAddition).toContain("Compaction pressure: 1 summary item(s), 1 stale or superseded");
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      mode: "use_brain",
    }));
  });

  it("records zero-cap clipping as durable trace metadata without injecting a message", async () => {
    const brain = createBrainStub({
      query: vi.fn(async () => makeStructuredTraversalResult()),
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
      maxContextChars: 0,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.messages).toEqual(assembled.messages);
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "use_brain",
      budgetFraction: 0.3,
      maxContextChars: 0,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: 0,
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 0,
      droppedNodeCount: 3,
      fittingDropReasons: {
        omitted_for_max_context_chars: 3,
      },
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
    }));
    expect(brain.recordTraceSelectionMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ id: "tr_structured_1" }),
      expect.objectContaining({
        budgetFraction: 0.3,
        maxContextChars: 0,
        queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
        injectedChars: 0,
        droppedChars: expect.any(Number),
        contextClipped: true,
        fitStrategy: "structured_node_budget",
        retrievedNodeCount: 3,
        fittedNodeCount: 0,
        droppedNodeCount: 3,
        fittingDropReasons: {
          omitted_for_max_context_chars: 3,
        },
        compileElapsedMs: expect.any(Number),
        brainDropReason: "injection_cap_clipped",
        brainDropStage: "injection",
      }),
    );
  });

  it("skips brain query before retrieval when the soft compile deadline is already exhausted", async () => {
    const query = vi.fn(async () => makeTraversalResult());
    const brain = createBrainStub({
      compileDeadlineMs: 0,
      query,
    });
    const extension = new BrainAssemblerExtension(brain as never);
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(100)
      .mockReturnValueOnce(100);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 240,
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

    expect(query).not.toHaveBeenCalled();
    expect(result.messages).toEqual([{ role: "user", content: "live tail" }]);
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "skip_deadline_before_query",
      footer: "[brain] bypassed: soft compile deadline hit before query.",
      interruption: {
        interrupted: true,
        stage: "query",
        reason: "deadline_before_query",
        servedPartial: false,
      },
      queryInterrupted: false,
      interruptionStage: "query",
      interruptionReason: "deadline_before_query",
      servedPartial: false,
      compileElapsedMs: 0,
      compileDeadlineMs: 0,
      compileDeadlineHit: true,
      brainDropReason: "deadline_before_query",
      brainDropStage: "decision",
      budgetFraction: 0.3,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
    }));
  });

  it("partially serves a committed prefix when the deadline hits after query", async () => {
    const traversalResult = makeStructuredTraversalResult();
    const brain = createBrainStub({
      compileDeadlineMs: 5,
      query: vi.fn(async () => traversalResult),
    });
    const extension = new BrainAssemblerExtension(brain as never);
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(100)
      .mockReturnValueOnce(104)
      .mockReturnValueOnce(110);

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
      maxContextChars: 240,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toMatchObject({
      role: "user",
      content: expect.stringContaining("[brain partial]"),
    });
    expect(String(result.messages[0]?.content ?? "")).toContain("Use gh pr create");
    expect(String(result.messages[0]?.content ?? "")).not.toContain("Transcript Support");
    expect(result.messages.slice(1)).toEqual(assembled.messages);
    expect(result.estimatedTokens).toBeGreaterThan(assembled.estimatedTokens);
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "partial_deadline_after_query",
      episodeId: "ep_structured_1",
      traceId: "tr_structured_1",
      footer: "[brain] partial serve: soft compile deadline hit after query; injected committed prefix.",
      interruption: {
        interrupted: true,
        stage: "query",
        reason: "soft_compile_deadline",
        servedPartial: true,
      },
      queryInterrupted: false,
      interruptionStage: "query",
      interruptionReason: "soft_compile_deadline",
      servedPartial: true,
      compileElapsedMs: 10,
      compileDeadlineMs: 5,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      chosenStopCount: 0,
      forcedStopCount: 1,
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    }));
    if ((result.brainDecision?.droppedNodeCount ?? 0) > 0) {
      expect(result.brainDecision?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_partial_serve: expect.any(Number),
      }));
    } else {
      expect(result.brainDecision?.fittingDropReasons ?? null).toBeNull();
    }
    expect(brain.recordTraceSelectionMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ id: "tr_structured_1" }),
      expect.objectContaining({
        interruption: {
          interrupted: true,
          stage: "query",
          reason: "soft_compile_deadline",
          servedPartial: true,
        },
        queryInterrupted: false,
        interruptionStage: "query",
        interruptionReason: "soft_compile_deadline",
        servedPartial: true,
        compileElapsedMs: 10,
        compileDeadlineMs: 5,
        compileDeadlineHit: true,
        brainDropReason: "deadline_after_query",
        brainDropStage: "query",
        chosenStopCount: 0,
        forcedStopCount: 1,
        droppedProposalCount: 0,
        droppedProposalReasons: null,
        budgetFraction: 0.3,
        maxContextChars: 240,
        queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
        injectedChars: expect.any(Number),
        droppedChars: expect.any(Number),
        contextClipped: true,
        fitStrategy: "structured_node_budget",
        retrievedNodeCount: 3,
        fittedNodeCount: expect.any(Number),
        droppedNodeCount: expect.any(Number),
      }),
    );
  });

  it("partially serves a committed prefix when the deadline hits before injection", async () => {
    const traversalResult = makeStructuredTraversalResult();
    const brain = createBrainStub({
      compileDeadlineMs: 5,
      query: vi.fn(async () => traversalResult),
    });
    const extension = new BrainAssemblerExtension(brain as never);
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(100)
      .mockReturnValueOnce(101)
      .mockReturnValueOnce(102)
      .mockReturnValueOnce(110);

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
      maxContextChars: 240,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toMatchObject({
      role: "user",
      content: expect.stringContaining("[brain partial]"),
    });
    expect(String(result.messages[0]?.content ?? "")).toContain("Use gh pr create");
    expect(String(result.messages[0]?.content ?? "")).not.toContain("Transcript Support");
    expect(result.messages.slice(1)).toEqual(assembled.messages);
    expect(result.estimatedTokens).toBeGreaterThan(assembled.estimatedTokens);
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "partial_deadline_before_injection",
      episodeId: "ep_structured_1",
      traceId: "tr_structured_1",
      footer: "[brain] partial serve: soft compile deadline hit before injection; injected committed prefix.",
      interruption: {
        interrupted: true,
        stage: "injection",
        reason: "soft_compile_deadline",
        servedPartial: true,
      },
      queryInterrupted: false,
      interruptionStage: "injection",
      interruptionReason: "soft_compile_deadline",
      servedPartial: true,
      compileElapsedMs: 10,
      compileDeadlineMs: 5,
      compileDeadlineHit: true,
      brainDropReason: "deadline_before_injection",
      brainDropStage: "injection",
      chosenStopCount: 0,
      forcedStopCount: 1,
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    }));
    if ((result.brainDecision?.droppedNodeCount ?? 0) > 0) {
      expect(result.brainDecision?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_partial_serve: expect.any(Number),
      }));
    } else {
      expect(result.brainDecision?.fittingDropReasons ?? null).toBeNull();
    }
    expect(brain.recordTraceSelectionMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ id: "tr_structured_1" }),
      expect.objectContaining({
        interruption: {
          interrupted: true,
          stage: "injection",
          reason: "soft_compile_deadline",
          servedPartial: true,
        },
        queryInterrupted: false,
        interruptionStage: "injection",
        interruptionReason: "soft_compile_deadline",
        servedPartial: true,
        compileElapsedMs: 10,
        compileDeadlineMs: 5,
        compileDeadlineHit: true,
        brainDropReason: "deadline_before_injection",
        brainDropStage: "injection",
        chosenStopCount: 0,
        forcedStopCount: 1,
        droppedProposalCount: 0,
        droppedProposalReasons: null,
        budgetFraction: 0.3,
        maxContextChars: 240,
        queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
        injectedChars: expect.any(Number),
        droppedChars: expect.any(Number),
        contextClipped: true,
        fitStrategy: "structured_node_budget",
        retrievedNodeCount: 3,
        fittedNodeCount: expect.any(Number),
        droppedNodeCount: expect.any(Number),
      }),
    );
  });

  it("partially serves committed context when query traversal itself interrupts under deadline pressure", async () => {
    const traversalResult = makeStructuredTraversalResult();
    const brain = createBrainStub({
      query: vi.fn(async () => traversalResult),
      lastQueryInterruption: {
        interrupted: true,
        stage: "traversal",
        reason: "deadline_during_traversal",
        servedPartial: true,
      },
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
      maxContextChars: 240,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toMatchObject({
      role: "user",
      content: expect.stringContaining("[brain partial]"),
    });
    expect(String(result.messages[0]?.content ?? "")).toContain("Use gh pr create");
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "partial_query_interruption",
      episodeId: "ep_structured_1",
      traceId: "tr_structured_1",
      footer: "[brain] partial serve: query interrupted under budget pressure; injected committed prefix.",
      interruption: {
        interrupted: true,
        stage: "traversal",
        reason: "deadline_during_traversal",
        servedPartial: true,
      },
      queryInterrupted: true,
      interruptionStage: "traversal",
      interruptionReason: "deadline_during_traversal",
      servedPartial: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      chosenStopCount: 0,
      forcedStopCount: 1,
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: QUERY_BUDGET_CHARS_FOR_4096_TOKENS,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    }));
    if ((result.brainDecision?.droppedNodeCount ?? 0) > 0) {
      expect(result.brainDecision?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_partial_serve: expect.any(Number),
      }));
    } else {
      expect(result.brainDecision?.fittingDropReasons ?? null).toBeNull();
    }
    expect(brain.recordTraceSelectionMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ id: "tr_structured_1" }),
      expect.objectContaining({
        interruptionStage: "traversal",
        interruptionReason: "deadline_during_traversal",
        chosenStopCount: 0,
        forcedStopCount: 1,
        fitStrategy: "structured_node_budget",
        retrievedNodeCount: 3,
      }),
    );
  });

  it("surfaces structured service-side interruption truth when query work aborts before any trace exists", async () => {
    const brain = createBrainStub({
      query: vi.fn(async () => null),
      lastQueryInterruption: {
        interrupted: true,
        stage: "embedding",
        reason: "deadline_during_embedding",
        servedPartial: false,
      },
    });
    const extension = new BrainAssemblerExtension(brain as never);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 240,
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

    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "skip_deadline_after_query",
      footer: "[brain] bypassed: soft compile deadline hit after query.",
      interruption: {
        interrupted: true,
        stage: "embedding",
        reason: "deadline_during_embedding",
        servedPartial: false,
      },
      queryInterrupted: true,
      interruptionStage: "embedding",
      interruptionReason: "deadline_during_embedding",
      servedPartial: false,
      traceId: null,
      episodeId: null,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
    }));
    expect(brain.recordTraceSelectionMetadata).not.toHaveBeenCalled();
    expect(brain.noteAssemblyDecision).toHaveBeenCalledWith(expect.objectContaining({
      interruption: {
        interrupted: true,
        stage: "embedding",
        reason: "deadline_during_embedding",
        servedPartial: false,
      },
      traceId: null,
      episodeId: null,
    }));
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
          episodeCount: 0,
          snapshotCount: 0,
          branchCount: 0,
          typedMemoryRefCount: 0,
          freshnessStateCounts: {},
          hasNonFreshSummaries: false,
          hasTruthConflict: false,
          latestRole: null,
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
