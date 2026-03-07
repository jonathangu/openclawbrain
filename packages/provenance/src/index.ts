import {
  type ArtifactProvenanceV1,
  type EventExportProvenanceV1,
  type NormalizedEventRangeV1,
  type WorkspaceMetadataV1,
  validateEventExportProvenance,
  validateNormalizedEventRange,
  validateWorkspaceMetadata
} from "@openclawbrain/contracts";
import { createWorkspaceMetadata, type WorkspaceMetadataInput } from "@openclawbrain/workspace-metadata";

export interface BuildArtifactProvenanceInput {
  workspace: WorkspaceMetadataInput;
  eventRange: NormalizedEventRangeV1;
  eventExports?: EventExportProvenanceV1 | null;
  builtAt: string;
  offlineArtifacts?: readonly string[];
}

function uniqueNonEmpty(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

function isIsoDate(value: string): boolean {
  return !Number.isNaN(Date.parse(value));
}

export function validateArtifactProvenance(value: ArtifactProvenanceV1): string[] {
  const errors = [
    ...validateWorkspaceMetadata(value.workspace),
    ...validateNormalizedEventRange(value.eventRange)
  ];

  if (!isIsoDate(value.builtAt)) {
    errors.push("artifact provenance builtAt must be an ISO timestamp");
  }
  if (value.workspace.snapshotId !== value.workspaceSnapshot) {
    errors.push("artifact provenance workspaceSnapshot must match workspace.snapshotId");
  }
  if (value.eventExports !== null) {
    errors.push(...validateEventExportProvenance(value.eventExports, value.eventRange));
  }
  if (value.offlineArtifacts.some((artifact) => artifact.length === 0)) {
    errors.push("artifact provenance offlineArtifacts must be non-empty when set");
  }

  return errors;
}

export function buildArtifactProvenance(input: BuildArtifactProvenanceInput): ArtifactProvenanceV1 {
  const workspace = createWorkspaceMetadata(input.workspace);
  const provenance: ArtifactProvenanceV1 = {
    workspace,
    workspaceSnapshot: workspace.snapshotId,
    eventRange: input.eventRange,
    eventExports: input.eventExports ?? null,
    builtAt: input.builtAt,
    offlineArtifacts: uniqueNonEmpty(input.offlineArtifacts ?? [])
  };

  const errors = validateArtifactProvenance(provenance);
  if (errors.length > 0) {
    throw new Error(`Invalid artifact provenance: ${errors.join("; ")}`);
  }

  return provenance;
}

export { type ArtifactProvenanceV1, type EventExportProvenanceV1, type NormalizedEventRangeV1, type WorkspaceMetadataV1 };
