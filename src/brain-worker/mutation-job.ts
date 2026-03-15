import type { BrainWorkerJobResult } from "./jobs.js";

export function mutationJobResult(changed: boolean, details?: Record<string, unknown>): BrainWorkerJobResult {
  return { job: "mutation", changed, details };
}
