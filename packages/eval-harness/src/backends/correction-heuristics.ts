import type { BackendResult, EvalBackend } from "../backend-types.ts";

export const correctionHeuristicsBackend: EvalBackend = {
  id: "correction+heuristics",
  displayName: "Correction plus heuristics",
  async run(trace, context): Promise<BackendResult> {
    const toolResults = context.tools.invokeAll(trace);
    const rationale = [
      trace.correction
        ? `Correction memory matched: ${trace.correction.summary}`
        : "No correction memory matched.",
      `Slices considered: ${trace.slices.join(", ") || "none"}`,
    ];
    if (toolResults.length > 0) {
      rationale.push("Read-only tool fixture results were available to the heuristic adapter.");
    }

    return {
      schema_version: "ocb.backend-output.v1",
      trace_id: trace.trace_id,
      backend_id: "correction+heuristics",
      adapter_kind: "eval_fixture_adapter",
      generated_at: new Date().toISOString(),
      intervention: trace.correction ? "heuristic" : toolResults.length > 0 ? "heuristic" : "none",
      answer: buildAnswer(trace.correction?.recommended_action, toolResults.length),
      rationale,
      tool_results: toolResults,
      external_mutation_allowed: false,
      model_id: null,
      cost_usd: null,
      warnings: trace.counts_as_product_evidence
        ? []
        : ["NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY"],
    };
  },
};

function buildAnswer(correctionAction: string | undefined, fixtureCount: number): string {
  const parts = [];
  if (correctionAction) {
    parts.push(`Apply correction: ${correctionAction}`);
  }
  if (fixtureCount > 0) {
    parts.push(`Use ${fixtureCount} read-only fixture result(s) as bounded context.`);
  }
  return parts.length > 0
    ? parts.join(" ")
    : "No intervention is recommended by correction+heuristics.";
}
