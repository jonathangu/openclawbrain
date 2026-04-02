import type { AssembledSummaryMetadata } from "../assembler.js";

export type SummaryRoutingDecision = {
  mode: "ignore" | "summary_suffices" | "expand_to_source" | "prefer_typed_memory";
  reason: string;
};

function normalize(text: string): string {
  return text.trim().toLowerCase();
}

export function decideSummaryRouting(params: {
  queryText: string;
  summaryMetadata?: AssembledSummaryMetadata;
}): SummaryRoutingDecision {
  const query = normalize(params.queryText);
  const summaryMetadata = params.summaryMetadata;

  if (!query || !summaryMetadata || summaryMetadata.totalCount === 0) {
    return { mode: "ignore", reason: "no summaries present" };
  }

  const branchHeavy =
    (summaryMetadata.branchCount ?? 0) > 1 ||
    (summaryMetadata.snapshotCount ?? 0) > 0 ||
    (summaryMetadata.hasNonFreshSummaries ?? false) ||
    (summaryMetadata.hasTruthConflict ?? false);

  const staleSummaryCount = Object.entries(summaryMetadata.freshnessStateCounts ?? {})
    .filter(([freshnessState]) => freshnessState !== "fresh")
    .reduce((count, [, value]) => count + (value ?? 0), 0);

  const preferTypedMemory = [
    /\bcurrent\b/i,
    /\blatest\b/i,
    /\bnow\b/i,
    /\bchanged\b/i,
    /\bcodeword\b/i,
    /\bpreference\b/i,
    /\brule\b/i,
    /\bshould\s+i\s+use\b/i,
    /\bwhat should\b/i,
  ].some((pattern) => pattern.test(query));
  if (preferTypedMemory) {
    if ((summaryMetadata.typedMemoryRefCount ?? 0) > 0) {
      return {
        mode: "prefer_typed_memory",
        reason: "current-truth or conflict-sensitive query should prefer typed correction memory over summary recap",
      };
    }

    if (branchHeavy) {
      return {
        mode: "expand_to_source",
        reason: "current-truth query over branch-heavy compacted history should expand to source before asserting specifics",
      };
    }

    return {
      mode: "prefer_typed_memory",
      reason: "current-truth or conflict-sensitive query should prefer typed correction memory over summary recap",
    };
  }

  const broadRecap = [
    /\brecap\b/i,
    /\bsummary\b/i,
    /\boverview\b/i,
    /\bcatch me up\b/i,
    /\bwhat happened\b/i,
    /\btl;dr\b/i,
  ].some((pattern) => pattern.test(query));

  const heavilyCompacted =
    summaryMetadata.maxDepth >= 2 ||
    summaryMetadata.condensedCount >= 2 ||
    branchHeavy ||
    staleSummaryCount > 0;
  const precisionSensitive = [
    /\bexact\b/i,
    /\bquote\b/i,
    /\bpath\b/i,
    /\bfile\b/i,
    /\bcommand\b/i,
    /\bsha\b/i,
    /\bcommit\b/i,
    /\btimestamp\b/i,
    /\bwhen\b/i,
    /\bwhy\b/i,
    /\brationale\b/i,
    /\bconfig\b/i,
    /\bvalue\b/i,
    /\bproof\b/i,
  ].some((pattern) => pattern.test(query));
  if (precisionSensitive) {
    return {
      mode: "expand_to_source",
      reason: "precision-sensitive query should expand beyond summaries before asserting specifics",
    };
  }

  if (broadRecap) {
    return { mode: "summary_suffices", reason: "broad recap question can start from summaries" };
  }

  if (heavilyCompacted) {
    return {
      mode: "expand_to_source",
      reason: branchHeavy || staleSummaryCount > 0
        ? "branch-heavy, snapshot-heavy, or stale summary context should expand before making exact claims"
        : "deeply compacted summary context should expand before making exact claims",
    };
  }

  return { mode: "ignore", reason: "no special summary-routing override needed" };
}
