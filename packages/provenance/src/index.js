import { validateEventExportProvenance, validateNormalizedEventRange, validateWorkspaceMetadata } from "@openclawbrain/contracts";
import { createWorkspaceMetadata } from "@openclawbrain/workspace-metadata";
function uniqueNonEmpty(values) {
    return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}
function isIsoDate(value) {
    return !Number.isNaN(Date.parse(value));
}
export function validateArtifactProvenance(value) {
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
export function buildArtifactProvenance(input) {
    const workspace = createWorkspaceMetadata(input.workspace);
    const provenance = {
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
export {};
//# sourceMappingURL=index.js.map