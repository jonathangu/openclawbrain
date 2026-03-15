import type { BrainWorkerJobResult } from "./jobs.js";

export function teacherJobResult(changed: boolean, details?: Record<string, unknown>): BrainWorkerJobResult {
  return { job: "teacher", changed, details };
}
