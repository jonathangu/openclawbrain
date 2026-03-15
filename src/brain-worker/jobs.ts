export type BrainWorkerJobName =
  | "process-labels"
  | "teacher"
  | "update"
  | "mutation"
  | "promotion";

export interface BrainWorkerJobResult {
  job: BrainWorkerJobName;
  changed: boolean;
  details?: Record<string, unknown>;
}
