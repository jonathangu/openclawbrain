export const OPERATOR_HEALTH_CONTRACT: "openclawbrain_operator_health.v1";

export interface TeacherLoopTruthInput {
  failureMode: string | null;
  lastNoOpReason: string | null;
  latestFreshness: string | null;
  queueDepth: number | null;
  running: boolean | null;
  watch?: {
    state: string | null;
  } | null;
  watchState?: string | null;
}

export interface TeacherLoopTruthSummary {
  healthy: boolean | null;
  idle: boolean | null;
  stale: boolean | null;
}

export interface OperatorHealthInput {
  workerHealthy: boolean | null;
  workerMode: string | null;
  workerStatus: string | null;
  watchState: string | null;
  proofState: string | null;
  teacherArtifactCount: number | null;
  teacherLoopTruth?: TeacherLoopTruthSummary | null;
}

export type OperatorHealthStatus =
  | "healthy"
  | "partial"
  | "unknown"
  | "stale"
  | "unhealthy";

export interface OperatorHealthSummary {
  contract: typeof OPERATOR_HEALTH_CONTRACT;
  status: OperatorHealthStatus;
  healthy: boolean | null;
  partial: boolean;
  unknown: boolean;
  stale: boolean;
  detail: string;
  workerHealthy: boolean | null;
  workerMode: string | null;
  workerStatus: string | null;
  watchState: string | null;
  proofState: string | null;
  teacherArtifactCount: number | null;
  backgroundLearning: TeacherLoopTruthSummary;
  reasons: string[];
}

export function summarizeTeacherLoopTruth(input: TeacherLoopTruthInput): TeacherLoopTruthSummary;
export function summarizeOperatorHealth(input: OperatorHealthInput): OperatorHealthSummary;
export function isOperatorHealthSummary(value: unknown): value is OperatorHealthSummary;
