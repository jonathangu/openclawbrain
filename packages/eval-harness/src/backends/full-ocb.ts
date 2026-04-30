import type { BackendResult, EvalBackend } from "../backend-types.ts";

export const fullOcbBackend: EvalBackend = {
  id: "full-ocb",
  displayName: "Full OCB eval adapter",
  async run(trace, context): Promise<BackendResult> {
    const toolResults = context.tools.invokeAll(trace);
    const contextNotes = [
      `Goal: ${trace.user_goal}`,
      trace.expected_behavior ? `Expected behavior: ${trace.expected_behavior}` : undefined,
      trace.correction ? `Correction: ${trace.correction.recommended_action}` : undefined,
      toolResults.length > 0 ? `Fixture-backed tool results: ${toolResults.length}` : undefined,
    ].filter((note): note is string => Boolean(note));

    return {
      schema_version: "ocb.backend-output.v1",
      trace_id: trace.trace_id,
      backend_id: "full-ocb",
      adapter_kind: "eval_fixture_adapter",
      generated_at: new Date().toISOString(),
      intervention: "full-context",
      answer:
        "Use the available correction, slice, and read-only fixture context while preserving the evidence limits for this trace.",
      rationale: [
        "This is a deterministic eval adapter, not a new OpenClawBrain runtime.",
        ...contextNotes,
      ],
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
