import type { BrainWorkerJobResult } from "./jobs.js";

export function promotionJobResult(changed: boolean, details?: Record<string, unknown>): BrainWorkerJobResult {
  return { job: "promotion", changed, details };
}
