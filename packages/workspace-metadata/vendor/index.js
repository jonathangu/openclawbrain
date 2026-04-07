import { FIXTURE_WORKSPACE_METADATA, checksumJsonPayload, validateWorkspaceMetadata } from "@openclawbrain/contracts";
const DEFAULT_CAPTURED_AT = "2026-03-06T00:00:00.000Z";
function uniqueNonEmpty(values) {
    return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}
export function createWorkspaceMetadata(input) {
    const base = typeof input === "string" ? { workspaceId: input } : input;
    const workspaceId = base.workspaceId;
    const snapshotId = base.snapshotId ?? workspaceId;
    const rootDir = base.rootDir ?? workspaceId;
    const labels = uniqueNonEmpty(base.labels ?? []);
    const files = uniqueNonEmpty(base.files ?? []);
    const metadata = {
        workspaceId,
        snapshotId,
        capturedAt: base.capturedAt ?? DEFAULT_CAPTURED_AT,
        rootDir,
        branch: base.branch ?? null,
        revision: base.revision ?? null,
        dirty: base.dirty ?? false,
        manifestDigest: base.manifestDigest ??
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
export function workspaceSnapshotId(input) {
    return createWorkspaceMetadata(input).snapshotId;
}
export function workspaceManifestDigest(input) {
    return createWorkspaceMetadata(input).manifestDigest;
}
export { FIXTURE_WORKSPACE_METADATA, validateWorkspaceMetadata };
//# sourceMappingURL=index.js.map