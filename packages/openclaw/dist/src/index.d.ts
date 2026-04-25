import type { CompileSelectionMode } from "@openclawbrain/compiler";
import type {
  ActivationPointerRecordV1,
  BrainServeHotPathTimingV1,
  ContextCompactionMode,
  RouteMode,
  RuntimeCompileResponseV1,
} from "@openclawbrain/contracts";
import type { ActivationSlotInspection } from "@openclawbrain/pack-format";

export {
  clearOpenClawProfileRuntimeLoadProof,
  listOpenClawProfileRuntimeLoadProofs,
  recordOpenClawProfileRuntimeLoadProof,
  resolveAttachmentRuntimeLoadProofsPath,
  type OpenClawHomeProfileSource,
  type OpenClawProfileRuntimeLoadProofRecordV1,
  type OpenClawProfileRuntimeLoadProofSetV1,
  type OpenClawProfileRuntimeLoadProofsV1,
} from "./attachment-truth.js";

export interface CompileServeRouteBreadcrumbInput {
  invocationSurface: "direct_compile_call" | "installed_extension_before_prompt_build" | "runtime_turn_helper";
  hostEvent: "before_prompt_build" | null;
  installedEntryPath?: string | null;
}

export type RuntimeComparativeReplayMode = "vector_only" | "graph_prior_only" | "learned_route";
export interface FrozenReplayEvalIdentityV1 {
  packId: string;
  routerIdentity?: string | null;
}

export interface CompileRuntimeContextInput {
  activationRoot: string;
  message: string;
  agentId?: string;
  maxContextBlocks?: number;
  budgetStrategy?: "fixed_v1" | "empirical_v1";
  maxContextChars?: number;
  mode?: RouteMode | RuntimeComparativeReplayMode;
  selectionMode?: CompileSelectionMode;
  compactionMode?: ContextCompactionMode;
  runtimeHints?: readonly string[];
  sessionId?: string;
  channel?: string;
  _suppressServeLog?: boolean;
  _serveRouteBreadcrumbs?: CompileServeRouteBreadcrumbInput;
  _frozenReplayEvalIdentity?: FrozenReplayEvalIdentityV1;
}

export interface ActiveCompileTarget {
  activationRoot: string;
  activePointer: ActivationPointerRecordV1;
  inspection: ActivationSlotInspection;
}

export interface RuntimeCompileSuccess {
  ok: true;
  fallbackToStaticContext: false;
  hardRequirementViolated: false;
  activationRoot: string;
  activePackId: string;
  packRootDir: string;
  compileResponse: RuntimeCompileResponseV1;
  brainContext: string;
  timing: BrainServeHotPathTimingV1;
}

export interface RuntimeCompileFailOpenFailure {
  ok: false;
  fallbackToStaticContext: true;
  hardRequirementViolated: false;
  activationRoot: string;
  error: string;
  brainContext: string;
  timing: BrainServeHotPathTimingV1;
}

export interface RuntimeCompileHardFailure {
  ok: false;
  fallbackToStaticContext: false;
  hardRequirementViolated: true;
  activationRoot: string;
  error: string;
  brainContext: string;
  timing: BrainServeHotPathTimingV1;
}

export type RuntimeCompileFailure = RuntimeCompileFailOpenFailure | RuntimeCompileHardFailure;
export type RuntimeCompileResult = RuntimeCompileSuccess | RuntimeCompileFailure;

/**
 * Accounting summary for bounded-anytime serving interruptions.
 */
export interface InterruptionAccounting {
    droppedFrontierNodeIds: string[];
    droppedProposalNodeIds: string[];
    interruptedExpansionSourceNodeId: string | null;
    completedExpansionCount: number;
    maxExpansions: number;
    budgetUsed: number;
    remainingBudgetChars: number;
    budgetTotal: number;
    budgetUtilization: number;
    droppedProposalCount: number;
    droppedProposalReasons: Record<string, number>;
}
export type ContextFeedbackVerdict = "helpful" | "irrelevant" | "harmful";
export type ContextFeedbackFocusAction = "review_harmful_context" | "capture_follow_up" | "wait_for_teacher" | "increase_feedback_coverage" | "monitor";
export interface ContextFeedbackCoverageSummary {
    routeTraceCount: number;
    identifiedRouteTraceCount: number;
    unidentifiedRouteTraceCount: number;
    agentIdentityCoverage: number;
    observationCount: number;
    completedObservationCount: number;
    supervisedTraceCount: number;
    unsupervisedTraceCount: number;
    observationCoverage: number;
    supervisionCoverage: number;
    pendingFollowupCount: number;
    pendingTeacherCount: number;
}
export interface ContextFeedbackLatestVerdict {
    traceId: string;
    episodeId: string;
    observationId: string | null;
    agentIdentity: unknown | null;
    source: string;
    verdict: ContextFeedbackVerdict;
    score: number;
    confidence: number;
    reason: string | null;
    bindingMode: string | null;
    createdAt: number;
}
export interface ContextFeedbackAgentCoverageSummary {
    agentIdentity: unknown;
    routeTraceCount: number;
    supervisedTraceCount: number;
    unsupervisedTraceCount: number;
    supervisionCoverage: number;
    verdictCounts: Record<ContextFeedbackVerdict, number>;
    latestTraceAt: number | null;
    latestVerdictAt: number | null;
    detail: string;
}
export interface ContextFeedbackSummary {
    scoreBands: {
        helpfulMin: number;
        harmfulMax: number;
    };
    verdictCounts: Record<ContextFeedbackVerdict, number>;
    coverage: ContextFeedbackCoverageSummary;
    agents: ContextFeedbackAgentCoverageSummary[];
    latest: ContextFeedbackLatestVerdict | null;
    focus: {
        action: ContextFeedbackFocusAction;
        detail: string;
    };
    detail: string;
}
export declare function compileRuntimeContext(input: CompileRuntimeContextInput): RuntimeCompileResult;
