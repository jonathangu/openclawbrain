import { type ArtifactProvenanceV1, type EventExportProvenanceV1, type NormalizedEventRangeV1, type WorkspaceMetadataV1 } from "@openclawbrain/contracts";
import { type WorkspaceMetadataInput } from "@openclawbrain/workspace-metadata";
export interface BuildArtifactProvenanceInput {
    workspace: WorkspaceMetadataInput;
    eventRange: NormalizedEventRangeV1;
    eventExports?: EventExportProvenanceV1 | null;
    builtAt: string;
    offlineArtifacts?: readonly string[];
}
export declare function validateArtifactProvenance(value: ArtifactProvenanceV1): string[];
export declare function buildArtifactProvenance(input: BuildArtifactProvenanceInput): ArtifactProvenanceV1;
export { type ArtifactProvenanceV1, type EventExportProvenanceV1, type NormalizedEventRangeV1, type WorkspaceMetadataV1 };
