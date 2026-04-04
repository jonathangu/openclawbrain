import type { RetentionState, RetentionTargetRefV1, RetentionTransitionDecisionV1, RetentionTransitionKind } from "./teacher-v3-contracts.js";
import { evaluateRetentionTransitionV1 } from "./teacher-v3-contracts.js";
import { resetShadowCandidateState, type ShadowCandidateState, type ShadowMutationApplication } from "./shadow-application.js";

export interface TeacherShadowReplayCommonV1 {
  proposalId: string;
  proposalClass: "mutation" | "forgetting";
  reviewMode: "shadow_only";
  shadowOnly: true;
  promotionBypass: false;
  rollbackKey: string;
  applied: boolean;
  reversible: boolean;
  replayOutcome: "idle" | "applied" | "blocked" | "mixed";
  summary: string;
}

export interface TeacherMutationReplayApplicationSummaryV1 {
  index: number;
  proposalId: string;
  proposalKind: ShadowMutationApplication["proposalKind"];
  applied: boolean;
  reversible: boolean;
  reason: string | null;
  before: ShadowMutationApplication["before"];
  after: ShadowMutationApplication["after"];
  operationKinds: string[];
}

export interface TeacherMutationShadowReplaySummaryV1 extends TeacherShadowReplayCommonV1 {
  proposalClass: "mutation";
  candidateStateKind: "graph";
  before: {
    nodeCount: number;
    edgeCount: number;
  };
  after: {
    nodeCount: number;
    edgeCount: number;
  };
  applications: TeacherMutationReplayApplicationSummaryV1[];
  rollback: {
    strategy: "reset_shadow_candidate_state";
    restored: boolean;
    before: {
      nodeCount: number;
      edgeCount: number;
    };
    after: {
      nodeCount: number;
      edgeCount: number;
    };
    summary: string;
  };
}

export interface TeacherForgettingShadowReplaySummaryV1 extends TeacherShadowReplayCommonV1 {
  proposalClass: "forgetting";
  candidateStateKind: "retention";
  target: RetentionTargetRefV1;
  before: {
    retentionState: RetentionState;
  };
  after: {
    retentionState: RetentionState;
  };
  decision: RetentionTransitionDecisionV1;
  guardrail?: RetentionTransitionDecisionV1["guardrail"];
  reason: string;
  requestedTransition: RetentionTransitionKind;
  rollback: {
    strategy: "restore_retention_state";
    restored: boolean;
    before: {
      retentionState: RetentionState;
    };
    after: {
      retentionState: RetentionState;
    };
    summary: string;
  };
}

export type TeacherShadowReplaySummaryV1 =
  | TeacherMutationShadowReplaySummaryV1
  | TeacherForgettingShadowReplaySummaryV1;

function snapshotShadowGraph(graph: ShadowCandidateState["candidateGraph"]): {
  nodeCount: number;
  edgeCount: number;
} {
  return {
    nodeCount: graph.getAllNodes().length,
    edgeCount: graph.getAllEdges().length,
  };
}

function summarizeMutationApplications(
  applications: ShadowMutationApplication[],
): TeacherMutationReplayApplicationSummaryV1[] {
  return applications.map((application, index) => ({
    index,
    proposalId: application.proposalId,
    proposalKind: application.proposalKind,
    applied: application.applied,
    reversible: application.reversible,
    reason: application.reason,
    before: application.before,
    after: application.after,
    operationKinds: application.operations.map((operation) => operation.kind),
  }));
}

export function summarizeTeacherMutationShadowReplayV1(params: {
  proposalId: string;
  rollbackKey: string;
  state: ShadowCandidateState;
}): TeacherMutationShadowReplaySummaryV1 {
  const before = snapshotShadowGraph(params.state.baseGraph);
  const after = snapshotShadowGraph(params.state.candidateGraph);
  const applications = summarizeMutationApplications(params.state.applications);
  const appliedCount = applications.filter((application) => application.applied).length;
  const blockedCount = applications.length - appliedCount;
  const replayOutcome: TeacherShadowReplayCommonV1["replayOutcome"] = applications.length === 0
    ? "idle"
    : blockedCount === 0
      ? "applied"
      : appliedCount === 0
        ? "blocked"
        : "mixed";

  const rollbackState: ShadowCandidateState = {
    baseGraph: params.state.baseGraph,
    candidateGraph: params.state.candidateGraph.clone(),
    applications: [...params.state.applications],
  };
  resetShadowCandidateState(rollbackState);
  const rollbackAfter = snapshotShadowGraph(rollbackState.candidateGraph);
  const rollbackRestored = rollbackAfter.nodeCount === before.nodeCount && rollbackAfter.edgeCount === before.edgeCount;

  return {
    proposalId: params.proposalId,
    proposalClass: "mutation",
    reviewMode: "shadow_only",
    shadowOnly: true,
    promotionBypass: false,
    rollbackKey: params.rollbackKey,
    applied: appliedCount > 0,
    reversible: params.state.applications.every((application) => application.reversible),
    replayOutcome,
    summary: applications.length > 0
      ? `Mutation replay stayed shadow-only on the candidate graph (${appliedCount}/${applications.length} application(s) applied) and rollback restored the base graph without any promotion bypass.`
      : "Mutation replay stayed shadow-only on the candidate graph with no applied applications and no promotion bypass.",
    candidateStateKind: "graph",
    before,
    after,
    applications,
    rollback: {
      strategy: "reset_shadow_candidate_state",
      restored: rollbackRestored,
      before: after,
      after: rollbackAfter,
      summary: rollbackRestored
        ? "Rollback restored the candidate graph to the base graph."
        : "Rollback did not restore the candidate graph to the base graph.",
    },
  };
}

export function summarizeTeacherForgettingShadowReplayV1(params: {
  proposalId: string;
  rollbackKey: string;
  current: RetentionState;
  target: RetentionTargetRefV1;
  requestedTransition: RetentionTransitionKind;
  decision?: RetentionTransitionDecisionV1;
}): TeacherForgettingShadowReplaySummaryV1 {
  const decision = params.decision ?? evaluateRetentionTransitionV1({
    current: params.current,
    requested: params.requestedTransition,
    target: params.target,
  });
  const nextState = decision.allowed ? decision.to : params.current;
  const replayOutcome: TeacherShadowReplayCommonV1["replayOutcome"] = decision.allowed ? "applied" : "blocked";

  return {
    proposalId: params.proposalId,
    proposalClass: "forgetting",
    reviewMode: "shadow_only",
    shadowOnly: true,
    promotionBypass: false,
    rollbackKey: params.rollbackKey,
    applied: decision.allowed,
    reversible: true,
    replayOutcome,
    summary: decision.allowed
      ? `Forgetting replay moved ${params.target.sourceId} from ${params.current} to ${decision.to} in shadow-only mode and can roll back to ${params.current} with no promotion bypass.`
      : `Forgetting replay stayed shadow-only and was blocked by ${decision.guardrail ?? "policy"} while preserving rollback identity and no promotion bypass.`,
    candidateStateKind: "retention",
    target: params.target,
    before: {
      retentionState: params.current,
    },
    after: {
      retentionState: nextState,
    },
    decision,
    guardrail: decision.guardrail,
    reason: decision.reason,
    requestedTransition: params.requestedTransition,
    rollback: {
      strategy: "restore_retention_state",
      restored: true,
      before: {
        retentionState: nextState,
      },
      after: {
        retentionState: params.current,
      },
      summary: decision.allowed
        ? `Rollback restores ${params.target.sourceId} to ${params.current}.`
        : `Rollback preserves the original ${params.current} retention state after the blocked replay.`,
    },
  };
}
