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

  const broadRecap = [
    /\brecap\b/i,
    /\bsummary\b/i,
    /\boverview\b/i,
    /\bcatch me up\b/i,
    /\bwhat happened\b/i,
    /\btl;dr\b/i,
  ].some((pattern) => pattern.test(query));
  if (broadRecap) {
    return { mode: "summary_suffices", reason: "broad recap question can start from summaries" };
  }

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
    return {
      mode: "prefer_typed_memory",
      reason: "current-truth or conflict-sensitive query should prefer typed correction memory over summary recap",
    };
  }

  const heavilyCompacted = summaryMetadata.maxDepth >= 2 || summaryMetadata.condensedCount >= 2;
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
  if (precisionSensitive || heavilyCompacted) {
    return {
      mode: "expand_to_source",
      reason: precisionSensitive
        ? "precision-sensitive query should expand beyond summaries before asserting specifics"
        : "deeply compacted summary context should expand before making exact claims",
    };
  }

  return { mode: "ignore", reason: "no special summary-routing override needed" };
}
