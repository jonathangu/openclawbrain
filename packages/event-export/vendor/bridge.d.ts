import { type EventContractId, type FeedbackEventV1, type InteractionEventV1, type NormalizedEventExportV1, type NormalizedEventSourceV1, type NormalizedEventV1 } from "@openclawbrain/contracts";
export declare const DEFAULT_EVENT_EXPORT_LIVE_SLICE_SIZE = 64;
export declare const DEFAULT_EVENT_EXPORT_BACKFILL_SLICE_SIZE = 64;
export type EventExportLaneV1 = "live" | "backfill";
export interface EventExportWatermarkV1 {
    runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
    contract: EventContractId;
    eventId: string;
    sequence: number;
    createdAt: string;
    dedupId: string;
}
export interface EventExportCursorV1 {
    runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
    live: {
        after: EventExportWatermarkV1 | null;
        exhausted: boolean;
    };
    backfill: {
        before: EventExportWatermarkV1 | null;
        exhausted: boolean;
    };
}
export interface EventExportSliceProvenanceV1 {
    runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
    lane: EventExportLaneV1;
    sliceDigest: string;
    bridgeDigest: string;
    sourceStreams: string[];
    contracts: EventContractId[];
    dedupedEventCount: number;
    duplicateIdentityCount: number;
}
export interface NormalizedEventExportSliceV1 {
    lane: EventExportLaneV1;
    sliceId: string;
    export: NormalizedEventExportV1;
    eventIdentities: string[];
    dedupedEventCount: number;
    duplicateIdentityCount: number;
    watermark: {
        first: EventExportWatermarkV1 | null;
        last: EventExportWatermarkV1 | null;
    };
    nextCursor: EventExportCursorV1;
    provenance: EventExportSliceProvenanceV1;
}
export interface NormalizedEventExportBridgeV1 {
    runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
    slices: NormalizedEventExportSliceV1[];
    cursor: EventExportCursorV1;
    dedupedInputCount: number;
    duplicateIdentityCount: number;
    bridgeDigest: string;
}
export interface NormalizedEventExportBundleEntryV1 {
    lane: EventExportLaneV1;
    sliceId: string;
    export: NormalizedEventExportV1;
    eventIdentities: string[];
    watermark: {
        first: EventExportWatermarkV1 | null;
        last: EventExportWatermarkV1 | null;
    };
    nextCursor: EventExportCursorV1;
}
export interface NormalizedEventExportBundleV1 {
    runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
    bridgeDigest: string;
    bundleDigest: string;
    cursor: EventExportCursorV1;
    dedupedInputCount: number;
    duplicateIdentityCount: number;
    entries: NormalizedEventExportBundleEntryV1[];
}
export interface BuildNormalizedEventExportBridgeInput {
    interactionEvents: readonly InteractionEventV1[];
    feedbackEvents: readonly FeedbackEventV1[];
    cursor?: EventExportCursorV1;
    liveSliceSize?: number;
    backfillSliceSize?: number;
}
export declare function buildNormalizedEventDedupId(event: NormalizedEventV1): string;
export declare function buildEventExportWatermark(event: NormalizedEventV1): EventExportWatermarkV1;
export declare function createEventExportCursor(): EventExportCursorV1;
export declare function buildNormalizedEventExportBridge(input: BuildNormalizedEventExportBridgeInput): NormalizedEventExportBridgeV1;
export declare function buildNormalizedEventExportBundleFromEvents(input: BuildNormalizedEventExportBridgeInput): NormalizedEventExportBundleV1;
export declare function buildNormalizedEventExportBundle(bridge: NormalizedEventExportBridgeV1): NormalizedEventExportBundleV1;
export declare function validateEventExportWatermark(value: EventExportWatermarkV1): string[];
export declare function validateEventExportCursor(value: EventExportCursorV1): string[];
export declare function validateNormalizedEventExportSlice(value: NormalizedEventExportSliceV1): string[];
export declare function validateNormalizedEventExportBridge(value: NormalizedEventExportBridgeV1): string[];
