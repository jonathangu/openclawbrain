import { type FeedbackEventV1, type InteractionEventV1, type NormalizedEventExportV1 } from "@openclawbrain/contracts";
import { type FeedbackEventExtractionResultV1 } from "@openclawbrain/event-export";
import type { OpenClawSessionIndex, OpenClawSessionIndexEntry, OpenClawSessionRecord } from "./session-store.js";
export interface OpenClawPassiveLearningPrivacySummaryV1 {
    sanitized: true;
    rules: string[];
    strippedMetadataBlockCount: number;
    strippedThinkingBlockCount: number;
    droppedToolResultCount: number;
    droppedRuntimeNoiseCount: number;
}
export interface OpenClawPassiveLearningSessionEvidenceV1 {
    sessionKey: string;
    sessionId: string;
    updatedAt: string;
    channel: string;
    sourceStream: string;
    sessionFileBasename: string | null;
    model: string | null;
    modelProvider: string | null;
    chatType: string | null;
    systemPromptFingerprint: {
        available: boolean;
        workspaceDirBasename: string | null;
        injectedWorkspaceFileCount: number;
        missingInjectedWorkspaceFileCount: number;
        toolNames: string[];
    };
}
export interface OpenClawPassiveLearningSessionExportV1 {
    session: OpenClawPassiveLearningSessionEvidenceV1;
    privacy: OpenClawPassiveLearningPrivacySummaryV1;
    interactionEvents: InteractionEventV1[];
    feedbackEvents: FeedbackEventV1[];
    interactionContentsById: Record<string, string>;
    feedbackExtraction: FeedbackEventExtractionResultV1;
    normalizedEventExport: NormalizedEventExportV1;
    warnings: string[];
}
export interface OpenClawPassiveLearningStoreExportV1 {
    sessions: OpenClawPassiveLearningSessionExportV1[];
    interactionEvents: InteractionEventV1[];
    feedbackEvents: FeedbackEventV1[];
    normalizedEventExport: NormalizedEventExportV1;
    warnings: string[];
}
export declare function buildPassiveLearningSessionExportFromOpenClawSessionStore(input: {
    sessionKey: string;
    indexEntry: OpenClawSessionIndexEntry;
    records: readonly OpenClawSessionRecord[];
    agentId?: string;
    sequenceStart?: number;
}): OpenClawPassiveLearningSessionExportV1 & {
    nextSequence: number;
};
export declare function buildPassiveLearningStoreExportFromOpenClawSessionIndex(input: {
    sessionIndex: OpenClawSessionIndex;
    readSessionRecords: (sessionKey: string, entry: OpenClawSessionIndexEntry) => readonly OpenClawSessionRecord[];
    sessionKeys?: readonly string[];
    agentId?: string;
}): OpenClawPassiveLearningStoreExportV1;
