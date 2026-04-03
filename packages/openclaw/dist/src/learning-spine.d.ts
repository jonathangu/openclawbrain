import { type AlwaysOnLearningMaterializationJobV1 } from "@openclawbrain/learner";
import { type BrainServeHotPathTimingV1, type NormalizedEventExportV1, type RouteMode, type RuntimeCompileResponseV1 } from "@openclawbrain/contracts";
import { type LearningSpinePgRouteUpdateLogEntryV1, type LearningSpineServeRouteBreadcrumbsV1, type LearningSpineServeRouteDecisionLogEntryV1 as PackFormatLearningSpineServeRouteDecisionLogEntryV1, type PackDescriptor } from "@openclawbrain/pack-format";
export interface LearningSpineServeRouteCandidateScoreV1 {
    blockId: string;
    selected: boolean;
    actionScore: number;
    actionProbability: number;
    compactedFrom?: string[];
    matchedTokens?: string[];
    routingChannels?: string[];
}
export type LearningSpineServeRouteDecisionLogEntryV1 = Omit<PackFormatLearningSpineServeRouteDecisionLogEntryV1, "candidateSetIds" | "chosenContextIds" | "candidateScores" | "selectedKernelContextIds" | "selectedBrainContextIds"> & {
    candidateSetIds: string[];
    chosenContextIds: string[];
    candidateScores: LearningSpineServeRouteCandidateScoreV1[];
    selectedKernelContextIds: string[];
    selectedBrainContextIds: string[];
};
type CompileFailureLike = {
    ok: false;
    fallbackToStaticContext: boolean;
    hardRequirementViolated: boolean;
    error: string;
    timing: BrainServeHotPathTimingV1;
};
type CompileSuccessLike = {
    ok: true;
    activePackId: string;
    compileResponse: RuntimeCompileResponseV1;
    timing: BrainServeHotPathTimingV1;
};
type RuntimeCompileResultLike = CompileFailureLike | CompileSuccessLike;
interface RuntimeTurnLike {
    sessionId: string;
    channel: string;
    userMessage: string;
    createdAt?: string | null;
    sequenceStart?: number | null;
    maxContextBlocks?: number;
    maxContextChars?: number;
    budgetStrategy?: "fixed_v1" | "empirical_v1";
    mode?: RouteMode;
    runtimeHints?: readonly string[];
    compile?: {
        createdAt?: string | null;
    } | null;
}
export declare function appendServeTimeRouteDecisionLog(input: {
    activationRoot: string;
    turn: RuntimeTurnLike;
    compileResult: RuntimeCompileResultLike;
    recordedAt: string;
    normalizedEventExport?: NormalizedEventExportV1;
    breadcrumbs?: LearningSpineServeRouteBreadcrumbsV1;
}): LearningSpineServeRouteDecisionLogEntryV1;
export declare function appendLearningUpdateLogs(input: {
    activationRoot: string;
    materialization: AlwaysOnLearningMaterializationJobV1;
    activeBeforePack: PackDescriptor | null;
    candidateDescriptor: PackDescriptor;
}): {
    updateId: string;
    bindingCount: number;
    pgRouteUpdate: LearningSpinePgRouteUpdateLogEntryV1;
};
export {};
