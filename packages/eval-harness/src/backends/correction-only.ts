import type { BackendResult, EvalBackend } from "../backend-types.ts";

export const correctionOnlyBackend: EvalBackend = {
  id: "correction-only",
  displayName: "Correction only",
  async run(trace, context): Promise<BackendResult> {
    const toolResults = context.tools.invokeAll(trace);
    const hasCorrection = Boolean(trace.correction);
    return {
      schema_version: "ocb.backend-output.v1",
      trace_id: trace.trace_id,
      backend_id: "correction-only",
      adapter_kind: "eval_fixture_adapter",
      generated_at: new Date().toISOString(),
      intervention: hasCorrection ? "correction" : "none",
      answer: hasCorrection
        ? `Apply correction: ${trace.correction?.recommended_action}`
        : "No correction memory was available for this trace.",
      rationale: hasCorrection
        ? [`Correction memory matched: ${trace.correction?.summary}`]
        : ["Correction-only adapter stays silent when no correction is present."],
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
