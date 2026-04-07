/**
 * Brain import/export: backup and restore the activation root directory.
 *
 * export: tar + gzip the entire activation root → output.tar.gz
 * import: extract tar.gz → activation root, with safety checks
 */
export interface ExportOptions {
    activationRoot: string;
    outputPath: string;
}
export interface ExportResult {
    ok: boolean;
    outputPath: string;
    activationRoot: string;
    error?: string;
}
export interface ImportOptions {
    archivePath: string;
    activationRoot: string;
    force: boolean;
}
export interface ImportResult {
    ok: boolean;
    activationRoot: string;
    archivePath: string;
    warning?: string;
    error?: string;
}
export interface GraphifySourceBundleExportOptionsV1 {
    openclawHome?: string;
    activationRoot?: string;
    outputDir: string;
    homeDir?: string;
    profileRoots?: readonly string[];
    cursor?: unknown;
    observedAt?: string;
    createdAt?: string;
}
export interface GraphifySourceBundleExportResultV1 {
    ok: boolean;
    bundleDir: string | null;
    bundleId?: string;
    corpusId?: string;
    corpusDigest?: string;
    outputPaths?: {
        corpusManifest: string;
        normalizedEventExport: string;
        runtimeStatus: string;
        workspaceMetadata: string;
        proofDir: string;
        proofFiles: Record<string, string>;
    };
    corpusManifest?: unknown;
    normalizedEventExport?: unknown;
    runtimeStatus?: unknown;
    workspaceMetadata?: unknown;
    sourceSummaries?: unknown[];
    error?: string;
}
export interface GraphifyExportOptionsV1 {
    activationRoot: string;
    outputRoot?: string | null;
    runId?: string | null;
    repoRoot?: string | null;
    workspaceRoot?: string | null;
    sessionKey?: string | null;
    sessionTimestamp?: string | null;
    sessionSourcePath?: string | null;
    proofSummarySourcePath?: string | null;
    docsRoot?: string | null;
    codeRoot?: string | null;
    generatedAt?: string | null;
}
export interface GraphifyExportResultV1 {
    ok: boolean;
    runId: string;
    bundleRoot: string;
    sourceBundleHash: string | null;
    canonicalArchivePath: string;
    canonicalArchiveSha256: string | null;
    manifestPath: string | null;
    sessionProjectionPath: string | null;
    workspaceMemoryPath: string | null;
    workspaceTasksPath: string | null;
    proofSummaryPath: string | null;
    docsMirrorRoot: string | null;
    codeMirrorRoot: string | null;
    warnings?: string[];
    error?: string;
}
export interface GraphifyCompiledArtifactsExportOptionsV1 {
    bundleStartedAt?: string | Date;
    bundleId?: string | null;
    outputDir?: string | null;
    proposalId?: string | null;
    packId?: string | null;
    graphifyRunId?: string | null;
    graphifyVersion?: string | null;
    graphifyCommand?: string | null;
    sourceBundleId?: string | null;
    sourceBundleHash?: string | null;
    graphHash?: string | null;
    configHash?: string | null;
    labelsHash?: string | null;
    sourceDocs?: string[];
    sourceFixtures?: string[];
}
export interface GraphifyCompiledArtifactsExportResultV1 {
    ok: boolean;
    bundleId?: string;
    packId?: string;
    proposalId?: string;
    outputDir: string;
    manifestPath?: string;
    compilerProposalPath?: string;
    surfaceMapPath?: string;
    proposalReportPath?: string;
    verdictPath?: string;
    artifactCount?: number;
    validation?: {
        ok: boolean;
        errors: string[];
        bundleHash: string;
        fileCount: number;
        artifactCount: number;
    };
    digest?: {
        bundleId: string;
        packId: string;
        proposalId: string;
        files: Record<string, string>;
        fileCount: number;
        bundleHash: string;
    };
    writtenFiles?: string[];
    fileCount?: number;
    error?: string;
}
/**
 * Export (backup) the activation root to a tar.gz archive.
 */
export declare function exportBrain(options: ExportOptions): ExportResult;
/**
 * Export a canonical Graphify source bundle from the current OpenClaw corpus.
 */
export declare function exportGraphifySourceBundle(options: GraphifySourceBundleExportOptionsV1): GraphifySourceBundleExportResultV1;
/**
 * Project the canonical machine export into Graphify-friendly markdown and filesystem surfaces.
 */
export declare function exportGraphifyProjection(options: GraphifyExportOptionsV1): GraphifyExportResultV1;
/**
 * Import (restore) a tar.gz archive into the activation root.
 */
export declare function importBrain(options: ImportOptions): ImportResult;
/**
 * Build and write a Graphify-derived compiled artifact pack.
 */
export declare function exportGraphifyCompiledArtifactsPack(options: GraphifyCompiledArtifactsExportOptionsV1): GraphifyCompiledArtifactsExportResultV1;
