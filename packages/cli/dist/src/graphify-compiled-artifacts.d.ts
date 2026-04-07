export declare const GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1: readonly ["map_of_territory", "concept_page", "neighborhood_summary", "provenance_gap_report"];
export declare const GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1: {
    readonly manifest: "pack.manifest.json";
    readonly artifactsDir: "artifacts";
    readonly proposalsDir: "proposals";
    readonly compilerProposal: "proposals/compiler-proposal.json";
    readonly surfaceMap: "surface-map.json";
    readonly proposalReport: "proposal-report.json";
    readonly verdict: "verdict.json";
};
export interface GraphifyCompiledArtifactSourceRefV1 {
    sourceKind: string;
    sourceId: string;
    excerpt: string;
    authority: string;
    derivation: string;
    sourceHash?: string;
}
export interface GraphifyCompiledArtifactEvidenceV1 extends GraphifyCompiledArtifactSourceRefV1 {
    evidenceId: string;
}
export interface GraphifyCompiledArtifactClaimV1 {
    claimId: string;
    text: string;
    confidence: number;
    status: string;
    evidenceIds: string[];
}
export interface GraphifyCompiledArtifactSpecV1 {
    artifactId: string;
    kind: (typeof GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1)[number];
    title: string;
    summary: string;
    subjectIds: string[];
    confidence: number;
    evidence: GraphifyCompiledArtifactEvidenceV1[];
    counterevidence: GraphifyCompiledArtifactEvidenceV1[];
    openQuestions: string[];
    promotionNotes: string[];
    claims: GraphifyCompiledArtifactClaimV1[];
    replaySuites: string[];
    rollbackKey: string;
    sourceRoots: string[];
    createdAt?: string;
    updatedAt?: string;
    proposalLane?: string;
    status?: string;
    packId?: string | null;
    proposalId?: string | null;
}
export interface GraphifyCompiledArtifactPackInputV1 {
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
    artifactSpecs?: GraphifyCompiledArtifactSpecV1[];
}
export interface GraphifyCompiledArtifactPackBuildResultV1 {
    bundleId: string;
    bundleSlug: string;
    bundleStartedAt: string;
    outputDir: string;
    packId: string;
    proposalId: string;
    graphifyRunId: string;
    graphifyRun: Record<string, unknown>;
    packManifest: Record<string, unknown>;
    compilerProposal: Record<string, unknown>;
    surfaceMap: Record<string, unknown>;
    proposalReport: Record<string, unknown>;
    verdict: Record<string, unknown>;
    artifactEntries: Array<{
        artifactId: string;
        kind: string;
        title: string;
        summary: string;
        markdown: string;
        meta: Record<string, unknown>;
        contentHash: string;
        markdownPath: string;
        metaPath: string;
    }>;
    artifactSummaries: Array<Record<string, unknown>>;
    bundlePaths: Record<string, unknown>;
    paths: Record<string, unknown>;
    validation: {
        ok: boolean;
        errors: string[];
        bundleHash: string;
        fileCount: number;
        artifactCount: number;
    };
    digest: {
        bundleId: string;
        packId: string;
        proposalId: string;
        files: Record<string, string>;
        fileCount: number;
        bundleHash: string;
    };
    files: Record<string, string>;
}
export interface GraphifyCompiledArtifactPackWriteResultV1 {
    outputDir: string;
    writtenFiles: string[];
    fileCount: number;
}
export declare function resolveGraphifyCompiledArtifactPackOutputDir(options?: {
    outputDir?: string | null;
    bundleStartedAt?: Date | string;
    bundleId?: string | null;
}): string;
export declare function buildGraphifyCompiledArtifactPack(input?: GraphifyCompiledArtifactPackInputV1): GraphifyCompiledArtifactPackBuildResultV1;
export declare function writeGraphifyCompiledArtifactPack(outputDir: string, bundle: GraphifyCompiledArtifactPackBuildResultV1): GraphifyCompiledArtifactPackWriteResultV1;
export declare function buildGraphifyCompiledArtifactPackDigest(bundle: Pick<GraphifyCompiledArtifactPackBuildResultV1, "bundleId" | "packId" | "proposalId" | "files">): GraphifyCompiledArtifactPackBuildResultV1["digest"];
export declare function validateGraphifyCompiledArtifactPackBundle(bundle: Pick<GraphifyCompiledArtifactPackBuildResultV1, "bundleId" | "packId" | "proposalId" | "packManifest" | "artifactEntries" | "bundlePaths" | "paths" | "files">): GraphifyCompiledArtifactPackBuildResultV1["validation"];
