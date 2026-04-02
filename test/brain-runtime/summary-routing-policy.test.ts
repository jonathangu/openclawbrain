import { describe, expect, it } from "vitest";
import { decideSummaryRouting } from "../../src/brain-runtime/summary-routing-policy.js";
import type { AssembledSummaryMetadata } from "../../src/assembler.js";

function makeSummaryMetadata(overrides?: Partial<AssembledSummaryMetadata>): AssembledSummaryMetadata {
  return {
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
    items: [],
    ...overrides,
  };
}

describe("summary routing policy", () => {
  it("expands to source when summary context is stale or superseded", () => {
    const decision = decideSummaryRouting({
      queryText: "how should I proceed?",
      summaryMetadata: makeSummaryMetadata(),
    });

    expect(decision.mode).toBe("expand_to_source");
    expect(decision.reason).toContain("stale summary context");
  });

  it("still prefers typed memory when current-truth queries have correction refs", () => {
    const decision = decideSummaryRouting({
      queryText: "what should I use now?",
      summaryMetadata: makeSummaryMetadata({
        typedMemoryRefCount: 2,
      }),
    });

    expect(decision.mode).toBe("prefer_typed_memory");
    expect(decision.reason).toContain("typed correction memory");
  });

  it("does not let recap wording hide a current-truth query", () => {
    const decision = decideSummaryRouting({
      queryText: "give me the latest summary of what I should use now",
      summaryMetadata: makeSummaryMetadata({
        typedMemoryRefCount: 1,
      }),
    });

    expect(decision.mode).toBe("prefer_typed_memory");
    expect(decision.reason).toContain("current-truth");
  });
});
