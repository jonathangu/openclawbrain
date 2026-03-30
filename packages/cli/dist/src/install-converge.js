const CHANGE_REASON_LABELS = {
  install_identity: "plugin install identity changed",
  install_layout: "authoritative install layout changed",
  hook_path: "hook path changed",
  hook_state: "hook install state changed",
  loadability: "hook loadability changed",
  activation_root: "pinned activation root changed",
  loader_source: "loader source changed",
  runtime_guard_source: "runtime-guard source changed",
  plugins_config: "OpenClaw plugins config changed",
};

function sameValue(left, right) {
  return left === right;
}

function installIdentityOf(fingerprint) {
  if (fingerprint?.selectedInstall === null || fingerprint?.selectedInstall === undefined) {
    return null;
  }
  return JSON.stringify({
    extensionDir: fingerprint.selectedInstall.extensionDir ?? null,
    manifestId: fingerprint.selectedInstall.manifestId ?? null,
    installId: fingerprint.selectedInstall.installId ?? null,
    packageName: fingerprint.selectedInstall.packageName ?? null,
  });
}

export function planOpenClawBrainConvergePluginAction(fingerprint) {
  const selectedInstall = fingerprint?.selectedInstall ?? null;
  if (selectedInstall === null) {
    return {
      action: "install",
      packageSpec: "@openclawbrain/openclaw",
      pluginId: "openclawbrain",
      reason: "no authoritative OpenClawBrain plugin install was discovered for this OpenClaw home",
    };
  }
  if (selectedInstall.installLayout !== "native_package_plugin") {
    return {
      action: "install",
      packageSpec: "@openclawbrain/openclaw",
      pluginId: "openclawbrain",
      reason: "the authoritative install is still a generated shadow extension, so converge must install the split-package plugin lane",
    };
  }
  return {
    action: "update",
    packageSpec: "@openclawbrain/openclaw",
    pluginId: "openclawbrain",
    reason: "refresh the existing split-package plugin install so install/upgrade/repair stays on one converge path",
  };
}

export function diffOpenClawBrainConvergeRuntimeFingerprint(before, after) {
  const reasons = [];
  if (!sameValue(installIdentityOf(before), installIdentityOf(after))) {
    reasons.push("install_identity");
  }
  if (!sameValue(before?.installLayout ?? null, after?.installLayout ?? null)) {
    reasons.push("install_layout");
  }
  if (!sameValue(before?.hookPath ?? null, after?.hookPath ?? null)) {
    reasons.push("hook_path");
  }
  if (!sameValue(before?.hookState ?? null, after?.hookState ?? null)) {
    reasons.push("hook_state");
  }
  if (!sameValue(before?.loadability ?? null, after?.loadability ?? null)) {
    reasons.push("loadability");
  }
  if (!sameValue(before?.activationRoot ?? null, after?.activationRoot ?? null)) {
    reasons.push("activation_root");
  }
  if (!sameValue(before?.loaderSource ?? null, after?.loaderSource ?? null)) {
    reasons.push("loader_source");
  }
  if (!sameValue(before?.runtimeGuardSource ?? null, after?.runtimeGuardSource ?? null)) {
    reasons.push("runtime_guard_source");
  }
  if (!sameValue(before?.pluginsConfig ?? null, after?.pluginsConfig ?? null)) {
    reasons.push("plugins_config");
  }
  return {
    changed: reasons.length > 0,
    reasons,
  };
}

export function describeOpenClawBrainConvergeChangeReasons(reasons) {
  const normalized = reasons
    .map((reason) => CHANGE_REASON_LABELS[reason] ?? reason)
    .filter((reason, index, values) => values.indexOf(reason) === index);
  return normalized.length === 0 ? "no runtime-affecting install changes detected" : normalized.join("; ");
}

export function buildOpenClawBrainConvergeRestartPlan(input) {
  const changeReasons = input.changeReasons ?? [];
  if (changeReasons.length === 0) {
    return {
      required: false,
      automatic: false,
      reason: "no_runtime_affecting_changes",
      detail: "Skipped gateway restart because converge did not change plugin files, hook wiring, or the pinned activation root.",
    };
  }
  if (input.profileName === null) {
    return {
      required: true,
      automatic: false,
      reason: "exact_profile_unresolved",
      detail: "Restart is still required because converge changed runtime-affecting install state, but install could not infer an exact OpenClaw profile token for automatic restart.",
    };
  }
  return {
    required: true,
    automatic: true,
    reason: "runtime_affecting_changes_detected",
    detail: `Restart is required because converge changed runtime-affecting install state for profile ${input.profileName}.`,
  };
}

export function classifyOpenClawBrainConvergeVerification(input) {
  const warnings = [];
  const blockingReasons = [];
  const installLayout = input.installLayout ?? null;
  const installState = input.installState ?? "unknown";
  const loadability = input.loadability ?? "unverified";
  const displayedStatus = input.displayedStatus ?? "unknown";
  const runtimeLoad = input.runtimeLoad ?? "unverified";
  const loadProof = input.loadProof ?? "unverified";
  const serveState = input.serveState ?? "unknown";
  if (installLayout !== "native_package_plugin") {
    blockingReasons.push("split-package native package plugin is not authoritative after converge");
  }
  if (installState !== "installed") {
    blockingReasons.push(`hook install state is ${installState}`);
  }
  if (loadability !== "loadable") {
    blockingReasons.push(`hook loadability is ${loadability}`);
  }
  if (displayedStatus === "fail") {
    blockingReasons.push("status still reports fail");
  }
  const runtimeTruthAlreadyProven = runtimeLoad === "proven"
    && loadProof === "status_probe_ready"
    && installState === "installed"
    && loadability === "loadable";
  if (input.restartRequired === true && input.restartPerformed !== true && !runtimeTruthAlreadyProven) {
    blockingReasons.push("restart is still required before runtime load can be trusted");
  }
  if (blockingReasons.length > 0) {
    return {
      state: "failed",
      manualActionRequired: true,
      blockingReasons,
      warnings,
    };
  }
  if (input.restartRequired === true && input.restartPerformed !== true && runtimeTruthAlreadyProven) {
    warnings.push("automatic restart was not performed because install could not infer an exact OpenClaw profile token, but current status already proves runtime load");
  }
  if (displayedStatus !== "ok") {
    warnings.push(`status is ${displayedStatus}`);
  }
  if (runtimeLoad !== "proven") {
    warnings.push(`runtime load is ${runtimeLoad}`);
  }
  if (loadProof !== "status_probe_ready") {
    warnings.push(`loadProof is ${loadProof}`);
  }
  if (serveState !== "serving_active_pack") {
    warnings.push(`serve state is ${serveState}`);
  }
  if (input.routeFnAvailable !== true) {
    warnings.push("route_fn availability is not yet proven");
  }
  if (input.awaitingFirstExport === true) {
    warnings.push("the attached profile has not emitted its first export yet");
  }
  return {
    state: warnings.length > 0 ? "warning" : "healthy",
    manualActionRequired: false,
    blockingReasons,
    warnings,
  };
}

export function finalizeOpenClawBrainConvergeResult(input) {
  const warnings = [...(input.warnings ?? []), ...(input.verification?.warnings ?? [])];
  const uniqueWarnings = warnings.filter((warning, index, values) => values.indexOf(warning) === index);
  if (input.stepFailure !== null && input.stepFailure !== undefined) {
    return {
      verdict: "failed",
      why: input.stepFailure,
      warnings: uniqueWarnings,
    };
  }
  if (input.verification?.state === "failed") {
    return {
      verdict: input.verification.manualActionRequired ? "manual_action_required" : "failed",
      why: input.verification.blockingReasons.join("; "),
      warnings: uniqueWarnings,
    };
  }
  if (input.verification?.state === "warning") {
    return {
      verdict: "converged_with_warnings",
      why: input.verification.warnings.join("; "),
      warnings: uniqueWarnings,
    };
  }
  return {
    verdict: "converged",
    why: "plugin install, hook wiring, restart behavior, and status verification converged on the expected split-package state",
    warnings: uniqueWarnings,
  };
}
