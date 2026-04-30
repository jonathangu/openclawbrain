import type { BackendResult, EvalBackend } from "../backend-types.ts";

export const noneBackend: EvalBackend = {
  id: "none",
  displayName: "No intervention",
  async run(trace, context): Promise<BackendResult> {
    const toolResults = context.tools.invokeAll(trace);
    return {
      schema_version: "ocb.backend-output.v1",
      trace_id: trace.trace_id,
      backend_id: "none",
      adapter_kind: "eval_fixture_adapter",
      generated_at: new Date().toISOString(),
      intervention: "none",
      answer: "No OpenClawBrain intervention was applied for this trace.",
      rationale: ["Baseline adapter intentionally emits no memory/correction help."],
      tool_results: toolResults,
      external_mutation_allowed: false,
      model_id: null,
      cost_usd: null,
      warnings: syntheticWarnings(trace.counts_as_product_evidence),
    };
  },
};

function syntheticWarnings(countsAsProductEvidence: boolean): string[] {
  return countsAsProductEvidence
    ? []
    : ["NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY"];
}
