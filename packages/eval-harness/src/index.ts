import { correctionHeuristicsBackend } from "./backends/correction-heuristics.ts";
import { correctionOnlyBackend } from "./backends/correction-only.ts";
import { fullOcbBackend } from "./backends/full-ocb.ts";
import { noneBackend } from "./backends/none.ts";

import type { EvalBackend } from "./backend-types.ts";

export const EVAL_BACKENDS: ReadonlyArray<EvalBackend> = Object.freeze([
  noneBackend,
  correctionOnlyBackend,
  correctionHeuristicsBackend,
  fullOcbBackend,
]);

export { makeBlindPackets } from "./blind-packets.ts";
export { runEvalHarness } from "./run.ts";
export {
  SYNTHETIC_EVIDENCE_LABEL,
  loadTraces,
  validateTrace,
  type EvalTrace,
  type TraceToolCall,
} from "./trace.ts";
export {
  captureReproducibilityMetadata,
  sha256File,
  stableHash,
  type ReproducibilityMetadata,
} from "./reproducibility.ts";
export {
  createFixtureRuntime,
  loadToolFixtures,
  type EvalToolRuntime,
  type ToolFixture,
} from "./tool-fixtures.ts";
export type {
  BackendId,
  BackendResult,
  EvalBackend,
  EvalBackendContext,
} from "./backend-types.ts";
