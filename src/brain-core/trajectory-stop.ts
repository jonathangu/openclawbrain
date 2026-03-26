import type { TrajectoryStopTruth, TrajectorySubstep } from "./types.js";

function hasTraverseCandidate(substep: TrajectorySubstep): boolean {
  return substep.candidates.some((candidate) => candidate.action.type === "traverse");
}

export function resolveStopTruth(substep: TrajectorySubstep): TrajectoryStopTruth | null {
  if (substep.chosenAction.type !== "stop_local") {
    return null;
  }
  if (substep.stopTruth === "forced" || substep.stopTruth === "chosen") {
    return substep.stopTruth;
  }
  if (substep.stopReason && substep.stopReason !== "policy_stop") {
    return "forced";
  }
  return hasTraverseCandidate(substep) ? "chosen" : "forced";
}

export function isForcedStopSubstep(substep: TrajectorySubstep): boolean {
  return resolveStopTruth(substep) === "forced";
}

export function isChosenPolicyStopSubstep(substep: TrajectorySubstep): boolean {
  return resolveStopTruth(substep) === "chosen";
}
