import { type FeedbackEventKind, type FeedbackEventV1, type InteractionEventV1, type NormalizedEventSourceV1, type PrincipalMetadataV1, type PrincipalRoleV1 } from "@openclawbrain/contracts";
export type FeedbackExtractionRecordFormatV1 = "raw" | "normalized";
export type FeedbackExtractionActorRoleV1 = PrincipalRoleV1 | "unknown";
export type FeedbackExtractionPriorityV1 = "highest" | "high" | "normal" | "low";
export type FeedbackExtractionCueV1 = "explicit_principal_feedback" | "explicit_principal_correction" | "explicit_principal_teaching" | "implicit_positive_continuation" | "assistant_adjacent_directive" | "suppression_cue" | "explicit_correction_cue" | "negative_corrective_signal" | "explicit_teaching_cue" | "positive_signal";
export interface FeedbackExtractionInteractionRecordV1 {
    recordId: string;
    format: FeedbackExtractionRecordFormatV1;
    createdAt: string;
    content: string;
    sequence?: number | null;
    messageId?: string | null;
    interactionId?: string | null;
    relatedInteractionId?: string | null;
    actorRole?: FeedbackExtractionActorRoleV1 | null;
    principal?: PrincipalMetadataV1;
}
export interface FeedbackSignalClassificationV1 {
    kind: FeedbackEventKind;
    confidence: number;
    priority: FeedbackExtractionPriorityV1;
    cues: FeedbackExtractionCueV1[];
    notes: string[];
}
export interface ExtractedFeedbackEventV1 {
    feedbackEvent: FeedbackEventV1;
    extraction: FeedbackSignalClassificationV1 & {
        sourceRecordId: string;
        sourceFormat: FeedbackExtractionRecordFormatV1;
    };
}
export interface FeedbackEventExtractionResultV1 {
    runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
    inputRecordCount: number;
    extractedCount: number;
    skippedRecordIds: string[];
    events: ExtractedFeedbackEventV1[];
}
export interface ExtractFeedbackEventsFromInteractionRecordsInputV1 {
    agentId: string;
    sessionId: string;
    channel: string;
    source: NormalizedEventSourceV1;
    records: readonly FeedbackExtractionInteractionRecordV1[];
    defaultRelatedInteractionId?: string | null;
}
export interface ExtractFeedbackEventsFromNormalizedInteractionsInputV1 {
    interactionEvents: readonly InteractionEventV1[];
    contentsByInteractionId: Readonly<Record<string, string>>;
    actorRolesByInteractionId?: Readonly<Record<string, FeedbackExtractionActorRoleV1 | null | undefined>>;
    principalsByInteractionId?: Readonly<Record<string, PrincipalMetadataV1 | undefined>>;
    relatedInteractionIdsByInteractionId?: Readonly<Record<string, string | null | undefined>>;
}
/**
 * Returns true if the content looks like a system/runtime message
 * rather than genuine human feedback. Used as an early gate to
 * prevent misclassification of subagent completions, runtime context
 * blocks, and internal events as corrections or teaching.
 */
export declare function isSystemMessage(content: string): boolean;
export declare function classifyFeedbackSignalContent(content: string, context?: Pick<FeedbackExtractionInteractionRecordV1, "actorRole" | "principal">): FeedbackSignalClassificationV1 | null;
export declare function extractFeedbackEventsFromInteractionRecords(input: ExtractFeedbackEventsFromInteractionRecordsInputV1): FeedbackEventExtractionResultV1;
export declare function buildFeedbackExtractionRecordsFromNormalizedInteractions(input: ExtractFeedbackEventsFromNormalizedInteractionsInputV1): FeedbackExtractionInteractionRecordV1[];
export declare function extractFeedbackEventsFromNormalizedInteractions(input: ExtractFeedbackEventsFromNormalizedInteractionsInputV1): FeedbackEventExtractionResultV1;
