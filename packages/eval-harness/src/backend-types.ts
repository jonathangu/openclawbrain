import type { ReproducibilityMetadata } from "./reproducibility.ts";
import type { EvalToolRuntime, ToolInvocationResult } from "./tool-fixtures.ts";
import type { EvalTrace } from "./trace.ts";

export type BackendId = "none" | "correction-only" | "correction+heuristics" | "full-ocb";

export interface EvalBackendContext {
  readonly run_id: string;
  readonly reproducibility: ReproducibilityMetadata;
  readonly tools: EvalToolRuntime;
}

export interface BackendResult {
  schema_version: "ocb.backend-output.v1";
  trace_id: string;
  backend_id: BackendId;
  adapter_kind: "eval_fixture_adapter";
  generated_at: string;
  intervention: "none" | "correction" | "heuristic" | "full-context";
  answer: string;
  rationale: string[];
  tool_results: ToolInvocationResult[];
  external_mutation_allowed: false;
  model_id: null;
  cost_usd: null;
  warnings: string[];
}

export interface EvalBackend {
  readonly id: BackendId;
  readonly displayName: string;
  run(trace: Readonly<EvalTrace>, context: EvalBackendContext): Promise<BackendResult>;
}
