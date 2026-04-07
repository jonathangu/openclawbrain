/**
 * REINFORCE update rule implementing Lemma 6.1 from the paper.
 *
 * Paper's correct update direction:
 *   ∂/∂ρ v_ρ(s_t) = E[z · Σ_{l=t}^{T} ∂logP_ρ(a_l|s_l)/∂ρ]
 *
 * Key insight: The sum goes over the ENTIRE trajectory from t to T.
 * This assigns credit to every routing decision that led to the outcome,
 * not just the last step. Williams (1992) equation (1) is one-step only.
 *
 * For the softmax policy over edge weights:
 *   ∂logP(a_j|s)/∂w_j = 1 - P(a_j|s)     (for the chosen action's weight)
 *   ∂logP(a_j|s)/∂w_k = -P(a_k|s)         (for other actions' weights)
 *
 * We update only the chosen action's parameter at each step, using:
 *   Δw_j = learningRate × (z - baseline) × (1 - P(a_j|s))
 *
 * The full-trajectory sum is achieved by accumulating updates across all steps.
 */

import type {
  BrainNode,
  Episode,
  PolicyGradientRouteUpdateContribution,
  PolicyGradientSupervisionArtifact,
  PolicyWeightUpdate,
  RewardSource,
} from "./types.js";
import { START_NODE_ID } from "./types.js";
import type { BrainGraph } from "./graph.js";
import { isChosenPolicyStopSubstep, isForcedStopSubstep } from "./trajectory-stop.js";

const DIRECT_SUPERVISION_STEP_CAP = 0.18;

const SUPERVISION_AUTHORITY_WEIGHT: Record<RewardSource, number> = {
  human: 1,
  self: 0.82,
  scanner: 0.66,
  teacher: 0.48,
};

const TOOL_ROLE_WEIGHT: Record<"tool_capability" | "tool_instance", number> = {
  tool_capability: 1,
  tool_instance: 0.84,
};

function isToolActionTarget(graph: BrainGraph | undefined, targetNodeId: string): boolean {
  return graph?.getNode(targetNodeId)?.kind === "toolcard";
}

function readStringMetadata(metadata: Record<string, unknown> | null | undefined, keys: string[]): string | null {
  for (const key of keys) {
    const value = metadata?.[key];
    if (typeof value !== "string") {
      continue;
    }
    const trimmed = value.trim();
    if (trimmed.length > 0) {
      return trimmed;
    }
  }
  return null;
}

function readToolActionRole(node: BrainNode | undefined): "tool_capability" | "tool_instance" | null {
  if (!node) {
    return null;
  }
  const role = readStringMetadata(node.metadata, ["action_kind", "actionKind", "toolRole", "tool_role", "toolActionRole", "tool_action_role"]);
  switch (role?.toLowerCase()) {
    case "tool_capability":
    case "capability":
      return "tool_capability";
    case "tool_instance":
    case "instance":
      return "tool_instance";
    default:
      return null;
  }
}

function readToolCapabilityLink(node: BrainNode | undefined): string | null {
  if (!node) {
    return null;
  }
  return readStringMetadata(node.metadata, [
    "toolCapabilityId",
    "tool_capability_id",
    "capabilityId",
    "capability_id",
    "toolCapability",
    "tool_capability",
  ]);
}

function supervisionAuthorityWeight(source: RewardSource): number {
  return SUPERVISION_AUTHORITY_WEIGHT[source] ?? 0.5;
}

function clampMagnitude(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value));
}

function boundDirectSupervisionDelta(
  rawDelta: number,
  currentWeight: number,
  stepCap = DIRECT_SUPERVISION_STEP_CAP,
): number {
  const anchoredDelta = rawDelta / (1 + Math.abs(currentWeight));
  return clampMagnitude(anchoredDelta, -stepCap, stepCap);
}

function combineTeacherSignal(supervision: PolicyGradientSupervisionArtifact[]): number {
  let signal = 0;
  for (const entry of supervision) {
    if (!Number.isFinite(entry.value) || !Number.isFinite(entry.confidence)) {
      continue;
    }
    const confidence = clampMagnitude(entry.confidence, 0, 1);
    const authority = supervisionAuthorityWeight(entry.source);
    signal += clampMagnitude(entry.value, -1, 1) * confidence * authority;
  }
  return clampMagnitude(signal, -1, 1);
}

function supervisionStrength(entry: PolicyGradientSupervisionArtifact): number {
  if (!Number.isFinite(entry.value) || !Number.isFinite(entry.confidence)) {
    return 0;
  }
  return Math.abs(clampMagnitude(entry.value, -1, 1)) * clampMagnitude(entry.confidence, 0, 1) * supervisionAuthorityWeight(entry.source);
}

export function policyWeightUpdateKey(update: PolicyWeightUpdate): string {
  switch (update.kind) {
    case "seed":
      return `seed→${update.nodeId}`;
    case "stop_local":
      return `stop→${update.sourceNodeId}`;
    case "tool_action":
      return `tool→${update.sourceNodeId}→${update.toolNodeId}`;
    case "edge":
      return `${update.source}→${update.target}`;
  }
}

export function collectReinforceUpdateContributions(
  episode: Episode,
  learningRate: number,
  baseline: number,
  graph?: BrainGraph,
): PolicyGradientRouteUpdateContribution[] {
  if (episode.reward === null) {
    return [];
  }

  const advantage = episode.reward - baseline;
  if (Math.abs(advantage) < 1e-8) {
    return [];
  }

  const contributions: PolicyGradientRouteUpdateContribution[] = [];

  for (const expansion of episode.trajectory) {
    for (const substep of expansion.substeps) {
      const sourceNodeId = substep.stateSnapshot.sourceNodeId ?? START_NODE_ID;
      const gradLogP = 1 - substep.chosenActionProbability;
      const delta = learningRate * advantage * gradLogP;
      if (Math.abs(delta) < 1e-12) {
        continue;
      }

      if (substep.chosenAction.type === "stop_local") {
        if (!isChosenPolicyStopSubstep(substep)) {
          continue;
        }
        contributions.push({
          updateKey: `stop→${sourceNodeId}`,
          kind: "stop_local",
          sourceNodeId,
          targetNodeId: null,
          expansionIndex: substep.stateSnapshot.expansionIndex,
          selectionIndex: substep.stateSnapshot.selectionIndex,
          chosenActionProbability: substep.chosenActionProbability,
          delta,
          stopTruth: substep.stopTruth ?? null,
          stopReason: substep.stopReason ?? null,
        });
        continue;
      }

      const targetNodeId = substep.chosenAction.targetNodeId;
      if (isToolActionTarget(graph, targetNodeId)) {
        contributions.push({
          updateKey: `tool→${sourceNodeId}→${targetNodeId}`,
          kind: "tool_action",
          sourceNodeId,
          targetNodeId,
          expansionIndex: substep.stateSnapshot.expansionIndex,
          selectionIndex: substep.stateSnapshot.selectionIndex,
          chosenActionProbability: substep.chosenActionProbability,
          delta,
          stopTruth: null,
          stopReason: null,
        });
        continue;
      }

      const updateKey = sourceNodeId === START_NODE_ID
        ? `seed→${targetNodeId}`
        : `${sourceNodeId}→${targetNodeId}`;
      contributions.push({
        updateKey,
        kind: sourceNodeId === START_NODE_ID ? "seed" : "edge",
        sourceNodeId,
        targetNodeId,
        expansionIndex: substep.stateSnapshot.expansionIndex,
        selectionIndex: substep.stateSnapshot.selectionIndex,
        chosenActionProbability: substep.chosenActionProbability,
        delta,
        stopTruth: null,
        stopReason: null,
      });
    }
  }

  return contributions;
}

function selectPrimaryTeacherSupervision(
  episode: Episode,
  supervision: PolicyGradientSupervisionArtifact[],
): PolicyGradientSupervisionArtifact | null {
  const promoted = supervision.filter((entry) => entry.source === "human" || entry.source === "self" || entry.source === "scanner" || entry.source === "teacher");
  if (promoted.length === 0) {
    return null;
  }

  const byRewardSource = episode.rewardSource
    ? promoted.find((entry) => entry.source === episode.rewardSource)
    : null;
  if (byRewardSource) {
    return byRewardSource;
  }

  return promoted.reduce<PolicyGradientSupervisionArtifact | null>((best, entry) => {
    if (!best) {
      return entry;
    }
    return supervisionStrength(entry) > supervisionStrength(best) ? entry : best;
  }, null);
}

function normalizeTeacherTargetIds(targetIds: string[]): Set<string> {
  return new Set(
    targetIds
      .map((targetId) => targetId.trim())
      .filter((targetId) => targetId.length > 0),
  );
}

function normalizeTeacherTargetSequence(targetIds: string[]): string[] {
  const sequence: string[] = [];

  for (const targetId of targetIds) {
    const trimmed = targetId.trim();
    if (trimmed.length === 0) {
      continue;
    }
    if (sequence.at(-1) === trimmed) {
      continue;
    }
    sequence.push(trimmed);
  }

  return sequence;
}

function resolveTeacherPathTarget(
  sourceNodeId: string,
  teacherPathNodeIds: string[],
): { targetNodeId: string | null; terminalStop: boolean } | null {
  if (teacherPathNodeIds.length === 0) {
    return null;
  }

  if (sourceNodeId === START_NODE_ID) {
    return {
      targetNodeId: teacherPathNodeIds[0] ?? null,
      terminalStop: false,
    };
  }

  const sourceIndex = teacherPathNodeIds.indexOf(sourceNodeId);
  if (sourceIndex < 0) {
    return null;
  }

  const targetNodeId = teacherPathNodeIds[sourceIndex + 1] ?? null;
  return {
    targetNodeId,
    terminalStop: targetNodeId === null,
  };
}

function teacherActionAlignmentMultiplier(
  substep: { chosenAction: { type: string; targetNodeId?: string } },
  teacherTargetNodeId: string | null,
  terminalStop: boolean,
): number {
  if (teacherTargetNodeId === null) {
    return substep.chosenAction.type === "stop_local" ? 1 : 0.85;
  }

  if (substep.chosenAction.type === "stop_local") {
    return terminalStop ? 0.85 : 0.75;
  }

  if (substep.chosenAction.targetNodeId === teacherTargetNodeId) {
    return 1;
  }

  return terminalStop ? 0.8 : 0.75;
}

export function collectTeacherActionDistillContributions(
  episode: Episode,
  learningRate: number,
  supervision: PolicyGradientSupervisionArtifact[],
  graph?: BrainGraph,
): PolicyGradientRouteUpdateContribution[] {
  const primary = selectPrimaryTeacherSupervision(episode, supervision);
  if (!primary) {
    return [];
  }

  const teacherSignal = combineTeacherSignal(supervision);
  if (Math.abs(teacherSignal) < 1e-8) {
    return [];
  }

  const directNodeIds = normalizeTeacherTargetIds(primary.traceSelectedNodeIds);
  const directPathNodeIds = normalizeTeacherTargetSequence(primary.traceSelectedPathNodeIds);
  const hasOrderedTeacherPath = directPathNodeIds.length > 0;
  const hasDirectTargets = directNodeIds.size > 0 || hasOrderedTeacherPath;

  const contributions: PolicyGradientRouteUpdateContribution[] = [];

  function addToolContribution(params: {
    sourceNodeId: string;
    targetNodeId: string;
    expansionIndex: number;
    selectionIndex: number;
    chosenActionProbability: number;
    delta: number;
    stopTruth: null;
    stopReason: null;
  }): void {
    if (Math.abs(params.delta) < 1e-12) {
      return;
    }
    contributions.push({
      updateKey: `tool→${params.sourceNodeId}→${params.targetNodeId}`,
      kind: "tool_action",
      sourceNodeId: params.sourceNodeId,
      targetNodeId: params.targetNodeId,
      expansionIndex: params.expansionIndex,
      selectionIndex: params.selectionIndex,
      chosenActionProbability: params.chosenActionProbability,
      delta: params.delta,
      stopTruth: null,
      stopReason: null,
    });
  }

  function addTraversedContribution(params: {
    kind: "seed" | "edge";
    sourceNodeId: string;
    targetNodeId: string;
    expansionIndex: number;
    selectionIndex: number;
    chosenActionProbability: number;
    delta: number;
  }): void {
    if (Math.abs(params.delta) < 1e-12) {
      return;
    }
    contributions.push({
      updateKey: params.kind === "seed"
        ? `seed→${params.targetNodeId}`
        : `${params.sourceNodeId}→${params.targetNodeId}`,
      kind: params.kind,
      sourceNodeId: params.sourceNodeId,
      targetNodeId: params.targetNodeId,
      expansionIndex: params.expansionIndex,
      selectionIndex: params.selectionIndex,
      chosenActionProbability: params.chosenActionProbability,
      delta: params.delta,
      stopTruth: null,
      stopReason: null,
    });
  }

  function addAnchoredTargetContribution(params: {
    sourceNodeId: string;
    targetNodeId: string;
    expansionIndex: number;
    selectionIndex: number;
    chosenActionProbability: number;
    teacherAlignment: number;
    roleMultiplier?: number;
  }): void {
    const targetNode = graph?.getNode(params.targetNodeId);
    const targetKind = targetNode?.kind ?? null;
    if (targetKind === "toolcard") {
      const role = readToolActionRole(targetNode) ?? "tool_capability";
      const roleMultiplier = params.roleMultiplier ?? TOOL_ROLE_WEIGHT[role];
      const rawDelta = learningRate * teacherSignal * params.teacherAlignment * roleMultiplier * (1 - params.chosenActionProbability);
      const currentWeight = graph?.getToolActionPrior(params.sourceNodeId, params.targetNodeId) ?? 0;
      const delta = boundDirectSupervisionDelta(rawDelta, currentWeight);
      addToolContribution({
        sourceNodeId: params.sourceNodeId,
        targetNodeId: params.targetNodeId,
        expansionIndex: params.expansionIndex,
        selectionIndex: params.selectionIndex,
        chosenActionProbability: params.chosenActionProbability,
        delta,
        stopTruth: null,
        stopReason: null,
      });

      if (role === "tool_instance") {
        const capabilityNodeId = readToolCapabilityLink(targetNode);
        const capabilityNode = capabilityNodeId ? graph?.getNode(capabilityNodeId) : null;
        if (capabilityNodeId !== null && capabilityNode?.kind === "toolcard" && capabilityNodeId !== params.targetNodeId) {
          const capabilityWeight = graph?.getToolActionPrior(params.sourceNodeId, capabilityNodeId) ?? 0;
          const capabilityDelta = boundDirectSupervisionDelta(rawDelta * 0.5, capabilityWeight, DIRECT_SUPERVISION_STEP_CAP * 0.75);
          addToolContribution({
            sourceNodeId: params.sourceNodeId,
            targetNodeId: capabilityNodeId,
            expansionIndex: params.expansionIndex,
            selectionIndex: params.selectionIndex,
            chosenActionProbability: params.chosenActionProbability,
            delta: capabilityDelta,
            stopTruth: null,
            stopReason: null,
          });
        }
      }
      return;
    }

    const rawDelta = learningRate * teacherSignal * params.teacherAlignment * (1 - params.chosenActionProbability);
    if (params.sourceNodeId === START_NODE_ID) {
      const currentWeight = graph?.getSeedWeight(params.targetNodeId) ?? 0;
      const delta = boundDirectSupervisionDelta(rawDelta, currentWeight);
      addTraversedContribution({
        kind: "seed",
        sourceNodeId: params.sourceNodeId,
        targetNodeId: params.targetNodeId,
        expansionIndex: params.expansionIndex,
        selectionIndex: params.selectionIndex,
        chosenActionProbability: params.chosenActionProbability,
        delta,
      });
      return;
    }

    const currentWeight = graph?.getEdge(params.sourceNodeId, params.targetNodeId)?.weight ?? 0;
    const delta = boundDirectSupervisionDelta(rawDelta, currentWeight);
    addTraversedContribution({
      kind: "edge",
      sourceNodeId: params.sourceNodeId,
      targetNodeId: params.targetNodeId,
      expansionIndex: params.expansionIndex,
      selectionIndex: params.selectionIndex,
      chosenActionProbability: params.chosenActionProbability,
      delta,
    });
  }

  for (const expansion of episode.trajectory) {
    for (const substep of expansion.substeps) {
      const sourceNodeId = substep.stateSnapshot.sourceNodeId ?? START_NODE_ID;
      const gradLogP = 1 - substep.chosenActionProbability;

      if (substep.chosenAction.type === "stop_local" && isForcedStopSubstep(substep)) {
        continue;
      }

      if (hasOrderedTeacherPath) {
        const teacherTransition = resolveTeacherPathTarget(sourceNodeId, directPathNodeIds);
        if (teacherTransition) {
          const alignmentMultiplier = teacherActionAlignmentMultiplier(
            substep,
            teacherTransition.targetNodeId,
            teacherTransition.terminalStop,
          );

          if (teacherTransition.targetNodeId === null) {
            const rawDelta = learningRate * teacherSignal * alignmentMultiplier * gradLogP;
            const currentWeight = graph?.getStopLocalWeight(sourceNodeId) ?? 0;
            const delta = boundDirectSupervisionDelta(rawDelta, currentWeight, DIRECT_SUPERVISION_STEP_CAP * 0.9);
            if (Math.abs(delta) > 1e-12) {
              contributions.push({
                updateKey: `stop→${sourceNodeId}`,
                kind: "stop_local",
                sourceNodeId,
                targetNodeId: null,
                expansionIndex: substep.stateSnapshot.expansionIndex,
                selectionIndex: substep.stateSnapshot.selectionIndex,
                chosenActionProbability: substep.chosenActionProbability,
                delta,
                stopTruth: substep.stopTruth ?? null,
                stopReason: substep.stopReason ?? null,
              });
            }
            continue;
          }

          addAnchoredTargetContribution({
            sourceNodeId,
            targetNodeId: teacherTransition.targetNodeId,
            expansionIndex: substep.stateSnapshot.expansionIndex,
            selectionIndex: substep.stateSnapshot.selectionIndex,
            chosenActionProbability: substep.chosenActionProbability,
            teacherAlignment: alignmentMultiplier,
          });
          continue;
        }

        if (directPathNodeIds.length > 1) {
          continue;
        }
      }

      if (substep.chosenAction.type === "stop_local") {
        if (!isChosenPolicyStopSubstep(substep)) {
          continue;
        }
        const rawDelta = learningRate * teacherSignal * gradLogP;
        const currentWeight = graph?.getStopLocalWeight(sourceNodeId) ?? 0;
        const delta = boundDirectSupervisionDelta(rawDelta, currentWeight, DIRECT_SUPERVISION_STEP_CAP * 0.9);
        if (Math.abs(delta) > 1e-12) {
          contributions.push({
            updateKey: `stop→${sourceNodeId}`,
            kind: "stop_local",
            sourceNodeId,
            targetNodeId: null,
            expansionIndex: substep.stateSnapshot.expansionIndex,
            selectionIndex: substep.stateSnapshot.selectionIndex,
            chosenActionProbability: substep.chosenActionProbability,
            delta,
            stopTruth: substep.stopTruth ?? null,
            stopReason: substep.stopReason ?? null,
          });
        }
        continue;
      }

      const targetNodeId = substep.chosenAction.targetNodeId;
      const directTargetMatch = hasDirectTargets
        ? directNodeIds.has(targetNodeId)
        : true;
      if (!directTargetMatch) {
        continue;
      }

      addAnchoredTargetContribution({
        sourceNodeId,
        targetNodeId,
        expansionIndex: substep.stateSnapshot.expansionIndex,
        selectionIndex: substep.stateSnapshot.selectionIndex,
        chosenActionProbability: substep.chosenActionProbability,
        teacherAlignment: 1,
      });
    }
  }

  return contributions;
}

/**
 * Compute REINFORCE weight updates from a completed episode.
 *
 * Implements Lemma 6.1: full-trajectory credit assignment.
 * Every step l from 0 to T contributes to the gradient.
 */
export function computeReinforceUpdates(
  episode: Episode,
  learningRate: number,
  baseline: number,
  graph?: BrainGraph,
): PolicyWeightUpdate[] {
  return mergePolicyWeightUpdates(
    collectReinforceUpdateContributions(episode, learningRate, baseline, graph).map((contribution) => contributionToPolicyUpdate(contribution)),
  );
}

export function computeTeacherActionUpdates(
  episode: Episode,
  learningRate: number,
  supervision: PolicyGradientSupervisionArtifact[],
  graph?: BrainGraph,
): PolicyWeightUpdate[] {
  return mergePolicyWeightUpdates(
    collectTeacherActionDistillContributions(episode, learningRate, supervision, graph).map((contribution) => contributionToPolicyUpdate(contribution)),
  );
}

export function mergePolicyWeightUpdates(updates: PolicyWeightUpdate[]): PolicyWeightUpdate[] {
  const merged: Map<string, PolicyWeightUpdate> = new Map();

  for (const update of updates) {
    const key = policyWeightUpdateKey(update);
    const existing = merged.get(key);
    if (existing && existing.kind === update.kind) {
      existing.delta += update.delta;
      continue;
    }
    merged.set(key, { ...update });
  }

  return [...merged.values()];
}

function contributionToPolicyUpdate(contribution: PolicyGradientRouteUpdateContribution): PolicyWeightUpdate {
  if (contribution.kind === "stop_local") {
    return {
      kind: "stop_local",
      sourceNodeId: contribution.sourceNodeId,
      delta: contribution.delta,
    };
  }

  if (contribution.kind === "tool_action") {
    return {
      kind: "tool_action",
      sourceNodeId: contribution.sourceNodeId,
      toolNodeId: contribution.targetNodeId ?? "",
      delta: contribution.delta,
    };
  }

  if (contribution.kind === "seed") {
    return {
      kind: "seed",
      nodeId: contribution.targetNodeId ?? "",
      delta: contribution.delta,
    };
  }

  return {
    kind: "edge",
    source: contribution.sourceNodeId,
    target: contribution.targetNodeId ?? "",
    delta: contribution.delta,
  };
}

/**
 * Update running baseline via exponential moving average.
 *
 * baseline_{n+1} = α × z_n + (1 - α) × baseline_n
 *
 * The baseline reduces variance in the REINFORCE estimate
 * without introducing bias (standard variance reduction technique).
 */
export function updateBaseline(
  currentBaseline: number,
  newReward: number,
  alpha: number,
): number {
  return alpha * newReward + (1 - alpha) * currentBaseline;
}

/**
 * Apply computed weight updates to graph edges.
 *
 * After applying, edge weights may become negative (inhibitory).
 * This is intentional — the paper allows signed outcomes.
 */
export function applyWeightUpdates(
  graph: BrainGraph,
  updates: PolicyWeightUpdate[],
): void {
  for (const update of updates) {
    if (update.kind === "seed") {
      const nextWeight = Math.max(-10, Math.min(10, graph.getSeedWeight(update.nodeId) + update.delta));
      graph.setSeedWeight(update.nodeId, nextWeight);
      continue;
    }

    if (update.kind === "stop_local") {
      const nextWeight = Math.max(
        -10,
        Math.min(10, graph.getStopLocalWeight(update.sourceNodeId) + update.delta),
      );
      graph.setStopLocalWeight(update.sourceNodeId, nextWeight);
      continue;
    }

    if (update.kind === "tool_action") {
      const nextWeight = Math.max(
        -10,
        Math.min(10, graph.getToolActionPrior(update.sourceNodeId, update.toolNodeId) + update.delta),
      );
      graph.setToolActionPrior(update.sourceNodeId, update.toolNodeId, nextWeight);
      continue;
    }

    const edge = graph.getEdge(update.source, update.target);
    if (!edge) continue;

    // Update weight: w_new = w_old + Δw
    edge.weight += update.delta;

    // Clamp to prevent numerical explosion
    edge.weight = Math.max(-10, Math.min(10, edge.weight));
  }
}
