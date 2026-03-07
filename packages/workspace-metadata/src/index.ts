import {
  FIXTURE_WORKSPACE_METADATA,
  checksumJsonPayload,
  type WorkspaceMetadataV1,
  validateWorkspaceMetadata
} from "@openclawbrain/contracts";

const DEFAULT_CAPTURED_AT = "2026-03-06T00:00:00.000Z";

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

function uniqueNonEmpty(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

export function createWorkspaceMetadata(input: WorkspaceMetadataInput): WorkspaceMetadataV1 {
  const base = typeof input === "string" ? ({ workspaceId: input } satisfies WorkspaceMetadataObjectInput) : input;
  const workspaceId = base.workspaceId;
  const snapshotId = base.snapshotId ?? workspaceId;
  const rootDir = base.rootDir ?? workspaceId;
  const labels = uniqueNonEmpty(base.labels ?? []);
  const files = uniqueNonEmpty(base.files ?? []);

  const metadata: WorkspaceMetadataV1 = {
    workspaceId,
    snapshotId,
    capturedAt: base.capturedAt ?? DEFAULT_CAPTURED_AT,
    rootDir,
    branch: base.branch ?? null,
    revision: base.revision ?? null,
    dirty: base.dirty ?? false,
    manifestDigest:
      base.manifestDigest ??
      checksumJsonPayload({
        workspaceId,
        snapshotId,
        rootDir,
        branch: base.branch ?? null,
        revision: base.revision ?? null,
        dirty: base.dirty ?? false,
        labels,
        files
      }),
    labels,
    files
  };

  const errors = validateWorkspaceMetadata(metadata);
  if (errors.length > 0) {
    throw new Error(`Invalid workspace metadata: ${errors.join("; ")}`);
  }

  return metadata;
}

export function workspaceSnapshotId(input: WorkspaceMetadataInput): string {
  return createWorkspaceMetadata(input).snapshotId;
}

export function workspaceManifestDigest(input: WorkspaceMetadataInput): string | null {
  return createWorkspaceMetadata(input).manifestDigest;
}

export { FIXTURE_WORKSPACE_METADATA, validateWorkspaceMetadata, type WorkspaceMetadataV1 };
