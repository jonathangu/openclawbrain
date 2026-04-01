import { describe, expect, it } from "vitest";
import { decideSummaryRouting } from "../../src/brain-runtime/summary-routing-policy.js";

const baseMetadata = {
  totalCount: 3,
  maxDepth: 2,
  condensedCount: 2,
  episodeCount: 1,
  snapshotCount: 1,
  branchCount: 2,
  typedMemoryRefCount: 1,
  hasTruthConflict: true,
  latestRole: "snapshot" as const,
  items: [],
};

describe("summary routing policy", () => {
  it("prefers summaries for broad recap requests even when the tree is branch-heavy", () => {
    const decision = decideSummaryRouting({
      queryText: "give me a quick recap of what happened",
      summaryMetadata: baseMetadata,
    });

    expect(decision.mode).toBe("summary_suffices");
  });

  it("prefers typed memory for current-truth questions when correction refs exist", () => {
    const decision = decideSummaryRouting({
      queryText: "what should I use now",
      summaryMetadata: {
        ...baseMetadata,
        branchCount: 1,
        snapshotCount: 0,
        hasTruthConflict: false,
        typedMemoryRefCount: 2,
      },
    });

    expect(decision.mode).toBe("prefer_typed_memory");
  });

  it("expands to source for precision-sensitive branch-heavy history", () => {
    const decision = decideSummaryRouting({
      queryText: "what exact command and file path should I use",
      summaryMetadata: {
        ...baseMetadata,
        typedMemoryRefCount: 0,
        branchCount: 3,
        snapshotCount: 2,
        hasTruthConflict: true,
      },
    });

    expect(decision.mode).toBe("expand_to_source");
    expect(decision.reason).toContain("branch-heavy");
  });
});
