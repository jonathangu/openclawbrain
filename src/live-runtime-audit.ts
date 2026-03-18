export interface TeacherLoopTruthInput {
  failureMode: string | null;
  lastNoOpReason: string | null;
  latestFreshness: string | null;
  queueDepth: number | null;
  running: boolean | null;
  watchState: string | null;
}

export interface TeacherLoopTruthSummary {
  healthy: boolean | null;
  idle: boolean | null;
  stale: boolean | null;
}

export function summarizeTeacherLoopTruth(input: TeacherLoopTruthInput): TeacherLoopTruthSummary {
  const watchState = input.watchState ?? "not_visible";
  const failureMode = input.failureMode ?? "unavailable";

  const idle =
    input.running === false
    && (input.queueDepth ?? 0) === 0
    && failureMode === "none";

  if (failureMode !== "none" && failureMode !== "unavailable") {
    return {
      healthy: false,
      idle,
      stale: watchState === "stale_snapshot",
    };
  }

  if (watchState === "not_visible") {
    return {
      healthy: null,
      idle: null,
      stale: null,
    };
  }

  const staleByHeartbeat = watchState === "stale_snapshot";
  const staleByArtifacts =
    input.latestFreshness === "stale"
    && input.lastNoOpReason !== "no_teacher_artifacts";
  const stale = staleByHeartbeat || staleByArtifacts;

  return {
    healthy: failureMode === "none" && stale === false,
    idle,
    stale,
  };
}
