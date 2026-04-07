import { type LearningScanPolicy, type FeedbackEventKind, type InteractionEventKind, type NormalizedEventExportV1, type NormalizedEventRangeV1 } from "@openclawbrain/contracts";
export interface SupervisionFreshnessBySource {
    sourceStream: string;
    eventCount: number;
    interactionCount: number;
    feedbackCount: number;
    humanLabelCount: number;
    selfLabelCount: number;
    freshestEventId: string;
    freshestSequence: number;
    freshestCreatedAt: string;
    freshestKind: FeedbackEventKind | InteractionEventKind;
}
export interface TeacherFreshness {
    freshestEventId: string | null;
    freshestSequence: number | null;
    freshestCreatedAt: string | null;
    freshestKind: FeedbackEventKind | InteractionEventKind | null;
    sourceStream: string | null;
    humanLabelCount: number;
    sources: string[];
}
export interface LearningSurfaceObservability {
    scanPolicy: LearningScanPolicy;
    scanSurfaces: string[];
    humanLabelCount: number;
    selfLabelCount: number;
}
export interface AttributionCoverage {
    totalEventCount: number;
    attributedEventCount: number;
    attributedInteractionCount: number;
    attributedFeedbackCount: number;
    selectionDigestCount: number;
    profileSelectors: string[];
    profileIds: string[];
    brainStatuses: string[];
    activePackIds: string[];
    routerIdentities: string[];
}
export interface NormalizedEventExportObservabilityReport {
    exportDigest: string;
    range: NormalizedEventRangeV1;
    sourceStreams: string[];
    learningSurface: LearningSurfaceObservability;
    supervisionFreshnessBySource: SupervisionFreshnessBySource[];
    teacherFreshness: TeacherFreshness;
    attributionCoverage: AttributionCoverage;
}
export declare function describeNormalizedEventExportObservability(normalizedEventExport: NormalizedEventExportV1): NormalizedEventExportObservabilityReport;
