export const OPERATOR_HEALTH_CONTRACT = "openclawbrain_operator_health.v1";

function normalizeNullableBoolean(value) {
  return typeof value === "boolean" ? value : null;
}

function normalizeNullableString(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeNullableNumber(value) {
  return Number.isFinite(value) ? value : null;
}

function pushUnique(values, next) {
  if (!next || values.includes(next)) {
    return;
  }
  values.push(next);
}

export function summarizeTeacherLoopTruth(input) {
  const watchState = normalizeNullableString(input?.watch?.state)
    ?? normalizeNullableString(input?.watchState)
    ?? "not_visible";
  const failureMode = normalizeNullableString(input?.failureMode) ?? "unavailable";
  const queueDepth = normalizeNullableNumber(input?.queueDepth) ?? 0;

  const idle =
    input?.running === false
    && queueDepth === 0
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
    input?.latestFreshness === "stale"
    && input?.lastNoOpReason !== "no_teacher_artifacts";
  const stale = staleByHeartbeat || staleByArtifacts;

  return {
    healthy: failureMode === "none" && stale === false && watchState === "watching",
    idle,
    stale,
  };
}

function normalizeTeacherLoopTruth(input) {
  return {
    healthy: normalizeNullableBoolean(input?.healthy),
    idle: normalizeNullableBoolean(input?.idle),
    stale: normalizeNullableBoolean(input?.stale),
  };
}

function defaultDetail(status) {
  switch (status) {
    case "healthy":
      return "worker and watcher signals support a healthy background-learning surface";
    case "stale":
      return "background-learning truth is stale in the current status surface";
    case "unhealthy":
      return "background-learning truth is explicitly unhealthy in the current status surface";
    case "partial":
      return "background-learning truth is only partially visible in the current status surface";
    default:
      return "background-learning truth is not visible in the current status surface";
  }
}

export function summarizeOperatorHealth(input) {
  const workerHealthy = normalizeNullableBoolean(input?.workerHealthy);
  const workerMode = normalizeNullableString(input?.workerMode);
  const workerStatus = normalizeNullableString(input?.workerStatus);
  const watchState = normalizeNullableString(input?.watchState);
  const proofState = normalizeNullableString(input?.proofState);
  const teacherArtifactCount = normalizeNullableNumber(input?.teacherArtifactCount);
  const teacherLoopTruth = normalizeTeacherLoopTruth(input?.teacherLoopTruth);
  const backgroundLearningStale =
    teacherLoopTruth.stale === true
    || watchState === "stale_snapshot";
  const backgroundLearningHealthy =
    backgroundLearningStale === true
      ? false
      : teacherLoopTruth.healthy;
  const backgroundLearning = {
    healthy: backgroundLearningHealthy,
    idle: teacherLoopTruth.idle,
    stale: backgroundLearningStale === true ? true : teacherLoopTruth.stale,
  };

  const hasWorkerSignal = workerHealthy !== null || workerMode !== null || workerStatus !== null;
  const hasWatchSignal = watchState !== null || proofState !== null || teacherArtifactCount !== null;
  const hasBackgroundSignal =
    backgroundLearning.healthy !== null
    || backgroundLearning.idle !== null
    || backgroundLearning.stale !== null;

  const reasons = [];

  if (workerHealthy === false) {
    pushUnique(reasons, "worker health is failing in the live status surface");
  }
  if (backgroundLearning.stale === true) {
    pushUnique(
      reasons,
      `watcher state is ${watchState ?? "stale_snapshot"}; background-learning truth is stale`,
    );
  } else if (watchState !== null && watchState !== "watching") {
    pushUnique(
      reasons,
      `watcher state is ${watchState}; do not treat this as a fully healthy background-learning surface`,
    );
  }
  if (backgroundLearning.healthy === false && backgroundLearning.stale !== true) {
    pushUnique(reasons, "background-learning loop reported an explicit failure");
  }
  if (workerHealthy === null && (hasWorkerSignal || hasWatchSignal || hasBackgroundSignal)) {
    pushUnique(reasons, "worker health is unknown in the live status surface");
  }
  if (backgroundLearning.healthy === null && (hasWorkerSignal || hasWatchSignal)) {
    pushUnique(reasons, "background-learning truth is partial in the current status surface");
  }

  let status;
  if (workerHealthy === false || (backgroundLearning.healthy === false && backgroundLearning.stale !== true)) {
    status = "unhealthy";
  } else if (backgroundLearning.stale === true) {
    status = "stale";
  } else if (!hasWorkerSignal && !hasWatchSignal && !hasBackgroundSignal) {
    status = "unknown";
    pushUnique(reasons, "background-learning truth is not visible in the current status surface");
  } else if (workerHealthy === true && backgroundLearning.healthy === true) {
    status = "healthy";
  } else {
    status = "partial";
  }

  return {
    contract: OPERATOR_HEALTH_CONTRACT,
    status,
    healthy:
      status === "healthy"
        ? true
        : status === "partial" || status === "unknown"
          ? null
          : false,
    partial: status === "partial",
    unknown: status === "unknown",
    stale: backgroundLearning.stale === true,
    detail: reasons[0] ?? defaultDetail(status),
    workerHealthy,
    workerMode,
    workerStatus,
    watchState,
    proofState,
    teacherArtifactCount,
    backgroundLearning,
    reasons,
  };
}

export function isOperatorHealthSummary(value) {
  return Boolean(value)
    && typeof value === "object"
    && value.contract === OPERATOR_HEALTH_CONTRACT
    && typeof value.status === "string"
    && Array.isArray(value.reasons);
}
