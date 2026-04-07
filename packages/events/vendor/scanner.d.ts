import { type NormalizedEventSourceV1 } from "@openclawbrain/contracts";
type PathSegment = string | number;
type ScannedMetadataSource = "raw" | "hook" | "defaulted" | "inferred" | "missing";
export type ScannedInteractionEventKind = "user_turn" | "assistant_turn" | "tool_call" | "tool_result";
export type ScannedInteractionActor = "user" | "assistant" | "tool" | "system" | "unknown";
export type ScannedTimestampSource = "event" | "turn" | "session" | "observed" | "missing";
export type ScannedContentFormat = "text" | "json" | "empty";
export interface ScannedProfileResolutionV1 {
    profileId?: string | null;
    profileSelector?: string | null;
}
export interface ScanSessionProfileResolverInput {
    rawSession: unknown;
    rawNode: unknown;
    path: readonly PathSegment[];
    session: ScannedSessionMetadataV1;
    inferredKind: ScannedInteractionEventKind;
    inferredActor: ScannedInteractionActor;
}
export interface ScanSessionProvenanceResolverInput {
    rawSession: unknown;
    rawNode: unknown;
    path: readonly PathSegment[];
    session: ScannedSessionMetadataV1;
}
export interface ScanSessionOptions {
    observedAt?: string;
    defaultAgentId?: string;
    defaultChannel?: string;
    defaultSourceStream?: string;
    profileResolver?: (input: ScanSessionProfileResolverInput) => ScannedProfileResolutionV1 | null | undefined;
    provenanceResolver?: (input: ScanSessionProvenanceResolverInput) => readonly string[] | null | undefined;
}
export interface ScannedSessionMetadataV1 {
    sessionId: string;
    sessionIdSource: ScannedMetadataSource;
    channel: string;
    channelSource: ScannedMetadataSource;
    agentId: string | null;
    agentIdSource: ScannedMetadataSource;
    profileId: string | null;
    profileIdSource: ScannedMetadataSource;
    profileSelector: string | null;
    profileSelectorSource: ScannedMetadataSource;
    source: NormalizedEventSourceV1;
    sourceStreamSource: ScannedMetadataSource;
    rawSessionHash: string;
    provenance: string[];
}
export interface ScannedInteractionProvenanceV1 {
    path: string;
    turnId: string | null;
    rawEventId: string | null;
    rawMessageId: string | null;
    rawRole: string | null;
    rawKind: string | null;
    notes: string[];
}
export interface ScannedInteractionEventV1 {
    eventId: string;
    sessionId: string;
    channel: string;
    agentId: string | null;
    profileId: string | null;
    profileSelector: string | null;
    sequence: number;
    kind: ScannedInteractionEventKind;
    actor: ScannedInteractionActor;
    createdAt: string | null;
    createdAtSource: ScannedTimestampSource;
    source: NormalizedEventSourceV1;
    messageId: string | null;
    toolCallId: string | null;
    toolName: string | null;
    parentEventId: string | null;
    content: string | null;
    contentFormat: ScannedContentFormat;
    rawHash: string;
    provenance: ScannedInteractionProvenanceV1;
}
export interface ScannedSessionV1 {
    session: ScannedSessionMetadataV1;
    events: ScannedInteractionEventV1[];
    warnings: string[];
}
export declare function scanSession(rawSession: unknown, options?: ScanSessionOptions): ScannedSessionV1;
export {};
