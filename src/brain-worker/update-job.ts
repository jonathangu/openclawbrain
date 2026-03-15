import type { BrainWorkerJobResult } from "./jobs.js";

export function updateJobResult(changed: boolean, details?: Record<string, unknown>): BrainWorkerJobResult {
  return { job: "update", changed, details };
}
