export interface GraphifyMaintenanceDiffOptionsV1 {
    graphifyRoot?: string;
    ocbRoot?: string;
    repoRoot?: string;
    workspaceRoot?: string;
    outputRoot?: string | null;
    runId?: string | null;
    diffId?: string | null;
    proposalId?: string | null;
    rollbackKey?: string | null;
}
export interface GraphifyMaintenanceDiffFindingRefV1 {
    sourceKind: string;
    sourceId: string;
    authority: string;
    derivation: string;
    excerpt?: string;
}
export interface GraphifyMaintenanceDiffClassBucketV1<T = unknown> {
    items: T[];
    truncated: boolean;
    total: number;
}
export interface GraphifyMaintenanceDiffReportV1 {
    contract: string;
    diffId: string;
    proposalId: string;
    rollbackKey: string;
    graphifyRoot: string;
    ocbRoot: string;
    repoRoot: string;
    workspaceRoot: string;
    currentBundleRoots: {
        role: string;
        bundleRoot: string;
        relativePath: string;
    }[];
    ocbBundleRoots: {
        role: string;
        bundleRoot: string;
        relativePath: string;
    }[];
    counts: Record<string, number>;
    findings: {
        missing_from_ocb: GraphifyMaintenanceDiffClassBucketV1<Record<string, unknown>>;
        stale_in_ocb: GraphifyMaintenanceDiffClassBucketV1<Record<string, unknown>>;
        candidate_only_edges_without_source_support: GraphifyMaintenanceDiffClassBucketV1<Record<string, unknown>>;
        new_current_source_hubs: GraphifyMaintenanceDiffClassBucketV1<Record<string, unknown>>;
        provenance_gap_candidates: GraphifyMaintenanceDiffClassBucketV1<Record<string, unknown>>;
        possible_merge_split_review_hints: GraphifyMaintenanceDiffClassBucketV1<Record<string, unknown>>;
    };
    evidenceRefs: GraphifyMaintenanceDiffFindingRefV1[];
    createdAt: string;
    updatedAt: string;
    sourceUniverse: {
        currentSurfaceIds: string[];
        ocbSurfaceIds: string[];
    };
    proposalSuggestion?: GraphifyMaintenanceDiffProposalSuggestionV1;
    verdict?: GraphifyMaintenanceDiffVerdictV1;
    summary?: string;
    bundleHash?: string;
}
export interface GraphifyMaintenanceDiffProposalSuggestionV1 {
    contract: string;
    diffId: string;
    proposalId: string;
    rollbackKey: string;
    reviewMode: string;
    status: string;
    summary: string;
    suggestionCount: number;
    suggestions: Array<Record<string, unknown>>;
    counts: Record<string, number>;
    currentBundleRoots: GraphifyMaintenanceDiffReportV1["currentBundleRoots"];
    ocbBundleRoots: GraphifyMaintenanceDiffReportV1["ocbBundleRoots"];
    createdAt: string;
    updatedAt: string;
    bundleHash?: string;
}
export interface GraphifyMaintenanceDiffVerdictV1 {
    contract: string;
    diffId: string;
    proposalId: string;
    verdict: string;
    severity: string;
    findingCount: number;
    proposalSuggestionCount: number;
    currentSurfaceCount: number;
    ocbSurfaceCount: number;
    why: string;
    reviewMode: string;
    targetStateOnly: boolean;
    rollbackKey: string;
    createdAt: string;
    updatedAt: string;
    bundleHash?: string;
}
export interface GraphifyMaintenanceDiffBundleV1 {
    ok: boolean;
    runId: string;
    diffId: string;
    proposalId: string;
    rollbackKey: string;
    repoRoot: string;
    workspaceRoot: string;
    graphifyRoot: string;
    ocbRoot: string;
    outputRoot: string;
    outputDir: string;
    report: GraphifyMaintenanceDiffReportV1;
    proposalSuggestion: GraphifyMaintenanceDiffProposalSuggestionV1;
    verdict: GraphifyMaintenanceDiffVerdictV1;
    files: Record<string, string>;
    paths: {
        maintenanceDiff: string;
        summary: string;
        proposalSuggestion: string;
        verdict: string;
    };
    digest: {
        bundleHash: string;
        fileCount: number;
        files: Record<string, string>;
    };
    currentRecords: Record<string, unknown>[];
    ocbRecords: Record<string, unknown>[];
}
export interface GraphifyMaintenanceDiffExportResultV1 extends GraphifyMaintenanceDiffBundleV1 {
    writtenFiles?: string[];
    fileCount?: number;
    error?: string;
}
export declare function buildGraphifyMaintenanceDiffBundle(options?: GraphifyMaintenanceDiffOptionsV1): GraphifyMaintenanceDiffBundleV1;
export declare function writeGraphifyMaintenanceDiffBundle(outputDir: string, bundle: GraphifyMaintenanceDiffBundleV1): {
    writtenFiles: string[];
    fileCount: number;
};
export declare function parseGraphifyMaintenanceDiffCliArgs(argv: readonly string[]): {
    command: "graphify-maintenance-diff";
    graphifyRoot: string;
    ocbRoot: string;
    repoRoot: string;
    workspaceRoot: string;
    outputRoot: string | null;
    runId: string | null;
    json: boolean;
    help: boolean;
};
export declare function formatGraphifyMaintenanceDiffSummary(result: GraphifyMaintenanceDiffBundleV1): string;
export declare function runGraphifyMaintenanceDiff(argvOrOptions?: readonly string[] | GraphifyMaintenanceDiffOptionsV1 | ({
    command?: string;
    json?: boolean;
    help?: boolean;
    graphifyRoot?: string;
    ocbRoot?: string;
    repoRoot?: string;
    workspaceRoot?: string;
    outputRoot?: string | null;
    runId?: string | null;
} & Record<string, unknown>)): GraphifyMaintenanceDiffExportResultV1 & {
    json: boolean;
    summary: string;
    help?: boolean;
    outputDir?: string | null;
    outputRoot?: string | null;
};
