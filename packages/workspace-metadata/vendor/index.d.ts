import { FIXTURE_WORKSPACE_METADATA, type WorkspaceMetadataV1, validateWorkspaceMetadata } from "@openclawbrain/contracts";
export interface WorkspaceMetadataObjectInput {
    workspaceId: string;
    snapshotId?: string;
    capturedAt?: string;
    rootDir?: string;
    branch?: string | null;
    revision?: string | null;
    dirty?: boolean;
    manifestDigest?: string | null;
    labels?: readonly string[];
    files?: readonly string[];
}
export type WorkspaceMetadataInput = string | WorkspaceMetadataObjectInput | WorkspaceMetadataV1;
export declare function createWorkspaceMetadata(input: WorkspaceMetadataInput): WorkspaceMetadataV1;
export declare function workspaceSnapshotId(input: WorkspaceMetadataInput): string;
export declare function workspaceManifestDigest(input: WorkspaceMetadataInput): string | null;
export { FIXTURE_WORKSPACE_METADATA, validateWorkspaceMetadata, type WorkspaceMetadataV1 };
