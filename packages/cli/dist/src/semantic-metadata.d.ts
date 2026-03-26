import type { EventSemanticMetadataV1, EventSourceKindV1, FeedbackEventKind, InteractionEventKind } from "@openclawbrain/contracts";
export declare function buildInteractionSemanticMetadata(sourceKind: EventSourceKindV1, kind: InteractionEventKind): EventSemanticMetadataV1;
export declare function buildAssistantMessageSemanticMetadata(): EventSemanticMetadataV1;
export declare function isInstructionalScaffoldingContent(kind: FeedbackEventKind, content: string | undefined): boolean;
export declare function buildFeedbackSemanticMetadata(sourceKind: EventSourceKindV1, kind: FeedbackEventKind, content?: string): EventSemanticMetadataV1;
