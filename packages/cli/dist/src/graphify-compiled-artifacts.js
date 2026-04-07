#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../../../..");

export const GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1 = [
    "map_of_territory",
    "concept_page",
    "neighborhood_summary",
    "provenance_gap_report",
];
export const GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1 = {
    manifest: "pack.manifest.json",
    artifactsDir: "artifacts",
    proposalsDir: "proposals",
    compilerProposal: "proposals/compiler-proposal.json",
    surfaceMap: "surface-map.json",
    proposalReport: "proposal-report.json",
    verdict: "verdict.json",
};
export const DEFAULT_GRAPHIFY_COMPILED_ARTIFACT_PACK_PARENT = path.join(repoRoot, "artifacts", "teacher-v3-proof");
function sha256Text(text) {
    return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}
function renderJson(value) {
    return `${JSON.stringify(value, null, 2)}\n`;
}
function ensureDir(dirPath) {
    mkdirSync(dirPath, { recursive: true });
}
function writeText(filePath, text) {
    ensureDir(path.dirname(filePath));
    writeFileSync(filePath, text, "utf8");
}
function writeJson(filePath, value) {
    writeText(filePath, renderJson(value));
}
function normalizeText(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}
function normalizeBoolean(value) {
    return value === true ? true : value === false ? false : null;
}
function normalizeNumber(value) {
    return Number.isFinite(value) ? Number(value) : null;
}
function slugify(value) {
    return value
        .toLowerCase()
        .replace(/[^a-z0-9]+/gu, "-")
        .replace(/^-+|-+$/gu, "")
        .replace(/-{2,}/gu, "-") || "bundle";
}
function timestampToken(date = new Date()) {
    const resolvedDate = date instanceof Date ? date : new Date(date);
    return resolvedDate.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").replace("T", "-");
}
function relativeRepoPath(filePath) {
    const resolvedPath = path.resolve(filePath);
    const relativePath = path.relative(repoRoot, resolvedPath);
    return relativePath.startsWith("..") ? resolvedPath : relativePath;
}
function relativeBundlePath(filePath, baseDir) {
    return path.relative(path.resolve(baseDir), path.resolve(filePath));
}
function makeSourceRef(sourceKind, sourceId, excerpt, authority = "raw_source", derivation = "teacher_compilation") {
    return {
        sourceKind,
        sourceId,
        excerpt,
        authority,
        derivation,
        sourceHash: sha256Text(`${sourceId}\n${excerpt}`),
    };
}
function makeEvidence(evidenceId, sourceRef, derivation = sourceRef.derivation ?? "teacher_compilation") {
    return {
        evidenceId,
        sourceKind: sourceRef.sourceKind,
        sourceId: sourceRef.sourceId,
        authority: sourceRef.authority,
        derivation,
        excerpt: sourceRef.excerpt,
        sourceHash: sourceRef.sourceHash ?? sha256Text(`${sourceRef.sourceId}\n${sourceRef.excerpt}`),
    };
}
function makeClaim(claimId, text, confidence, evidenceIds, status = "supported") {
    return {
        claimId,
        text,
        confidence,
        status,
        evidenceIds,
    };
}
function makeFrontmatter(lines) {
    return `---\n${lines.join("\n")}\n---\n`;
}
function quoteList(values) {
    return values.map((value) => `- ${value}`).join("\n");
}
function quoteSourceList(entries) {
    return entries.map((entry) => `- \`${entry.evidenceId}\` — ${entry.sourceId}`).join("\n");
}
function quoteAnchorList(entries) {
    return entries.map((entry) => `- \`${entry.id}\` (${entry.state}/${entry.kind}) — ${entry.source}`).join("\n");
}
function hashJoinedLines(lines) {
    return sha256Text(lines.join("\n"));
}
function normalizeIsoTimestamp(value) {
    if (value instanceof Date) {
        return value.toISOString();
    }
    const normalized = normalizeText(value);
    return normalized ?? new Date().toISOString();
}
function defaultGraphifyRunMetadata(input) {
    const bundleStartedAt = normalizeIsoTimestamp(input.bundleStartedAt);
    const bundleId = normalizeText(input.bundleId) ?? `graphify-compiled-artifacts-${timestampToken(bundleStartedAt)}`;
    const bundleSlug = slugify(bundleId);
    const graphifyRunId = normalizeText(input.graphifyRunId) ?? bundleId;
    const graphifyVersion = normalizeText(input.graphifyVersion) ?? "graphify-bridge@wave1";
    const graphifyCommand = normalizeText(input.graphifyCommand) ?? "graphify compile compiled-artifacts";
    const sourceBundleId = normalizeText(input.sourceBundleId) ?? "compiled-artifacts-target-state-scaffold";
    const sourceDocs = input.sourceDocs ?? [
        "docs/architecture/compiled-artifacts.md",
        "docs/architecture/teacher-v3.md",
        "docs/architecture/teacher-v3-proposals.md",
        "docs/architecture/teacher-v3-lints.md",
        "docs/architecture/teacher-v3-proof.md",
    ];
    const sourceFixtures = input.sourceFixtures ?? [
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/README.md",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/pack.manifest.json",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_concept_substrate_01/artifact.md",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_concept_substrate_01/artifact.meta.json",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_map_of_territory_01/artifact.md",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_map_of_territory_01/artifact.meta.json",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_provenance_gap_report_01/artifact.md",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_provenance_gap_report_01/artifact.meta.json",
    ];
    const sourceBundleHash = normalizeText(input.sourceBundleHash) ?? sha256Text(JSON.stringify({
        bundleId,
        graphifyRunId,
        graphifyVersion,
        graphifyCommand,
        sourceBundleId,
        sourceDocs,
        sourceFixtures,
    }));
    const graphHash = normalizeText(input.graphHash) ?? sha256Text(JSON.stringify({
        sourceBundleId,
        sourceBundleHash,
        graphifyRunId,
        graphifyVersion,
    }));
    const configHash = normalizeText(input.configHash) ?? sha256Text(JSON.stringify({
        graphifyVersion,
        graphifyCommand,
        sourceBundleId,
        bundleId,
    }));
    const labelsHash = normalizeText(input.labelsHash) ?? sha256Text(JSON.stringify(GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1));
    return {
        bundleStartedAt,
        bundleId,
        bundleSlug,
        graphifyRunId,
        graphifyVersion,
        graphifyCommand,
        sourceBundleId,
        sourceBundleHash,
        graphHash,
        configHash,
        labelsHash,
        sourceDocs,
        sourceFixtures,
    };
}
function buildDefaultArtifactSpecs(metadata) {
    const createdAt = metadata.bundleStartedAt;
    const sharedSourceRoots = [
        "docs/architecture",
        "artifacts/fixtures/compiled-artifacts/target-state-scaffold",
    ];
    const mapOfTerritoryEvidence = [
        makeEvidence(
            "ev-graphify-surface-hierarchy",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3-proof.md#surface-hierarchy",
                "Teacher v3 should read truth in runtime, proof, docs, then proposal order.",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-runtime-layout",
            makeSourceRef(
                "file",
                "docs/architecture/compiled-artifacts.md#runtime-storage-layout",
                "The authoritative runtime store should live under the activation root, not in the source repo.",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-scaffold-readme",
            makeSourceRef(
                "file",
                "artifacts/fixtures/compiled-artifacts/target-state-scaffold/README.md",
                "This directory is intentionally synthetic target-state scaffolding.",
            ),
        ),
    ];
    const conceptPageEvidence = [
        makeEvidence(
            "ev-compiled-artifacts-core-rules",
            makeSourceRef(
                "file",
                "docs/architecture/compiled-artifacts.md#core-rules",
                "compiled artifacts are derived, off-path knowledge products",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-sidecar-shape",
            makeSourceRef(
                "file",
                "docs/architecture/compiled-artifacts.md#markdown--sidecar-shape",
                "The sidecar JSON should be the authoritative metadata source.",
            ),
        ),
        makeEvidence(
            "ev-teacher-v3-off-path",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3.md#what-problem-teacher-v3-solves",
                "Teacher v3 is an off-path compiler of graph structure and compiled artifacts, not an arbiter of current truth.",
            ),
        ),
    ];
    const neighborhoodSummaryEvidence = [
        makeEvidence(
            "ev-teacher-v3-layers",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3.md#layers",
                "Compiled artifact layer, candidate graph layer, promoted pack layer.",
            ),
        ),
        makeEvidence(
            "ev-teacher-v3-proof-surface-hierarchy",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3-proof.md#surface-hierarchy",
                "Teacher v3 should read truth in runtime, proof, docs, then proposal order.",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-scaffold-manifest",
            makeSourceRef(
                "file",
                "artifacts/fixtures/compiled-artifacts/target-state-scaffold/pack.manifest.json",
                "Sidecar hashes must be regenerated whenever the markdown bodies change.",
            ),
        ),
    ];
    const provenanceGapEvidence = [
        makeEvidence(
            "ev-compiled-artifacts-provenance",
            makeSourceRef(
                "file",
                "docs/architecture/compiled-artifacts.md#provenance-fields",
                "The provenance block should explain where the artifact came from and how it was produced.",
            ),
        ),
        makeEvidence(
            "ev-teacher-v3-lints-ci-first",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3-lints.md#1-ci-first-deterministic-lint-family",
                "Deterministic checks run first; teacher-assisted audits run second.",
            ),
        ),
        makeEvidence(
            "ev-teacher-v3-lints-release-drift",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3-lints.md#3-release-drift-motivating-case",
                "Objective mismatches should be caught before semantic audits run.",
            ),
        ),
    ];
    return [
        {
            artifactId: "ca_graphify_map_of_territory_01",
            kind: "map_of_territory",
            title: "Graphify territory map",
            summary: "Shows the split between runtime/proof/docs truth and the Graphify-derived compiled-artifact bridge, with the target-state pack kept off the serve path.",
            subjectIds: ["topic:graphify", "topic:compiled-artifacts", "topic:teacher-v3"],
            confidence: 0.96,
            evidence: mapOfTerritoryEvidence,
            counterevidence: [
                makeEvidence(
                    "cevi-graphify-target-only",
                    makeSourceRef(
                        "file",
                        "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
                        "Teacher v3 reporting may summarize and cross-reference truth, but it must not become a new source of truth for the live runtime.",
                        "raw_source",
                        "teacher_lint",
                    ),
                    "teacher_lint",
                ),
            ],
            openQuestions: [
                "Should the next bridge lane generate a source bundle hash or only the compiled artifact pack?",
                "Which later surface should consume this pack first: import slice, lint diff, or replay packet?",
            ],
            promotionNotes: [
                "Keep the bridge off the serve path.",
                "Treat the scaffold and docs as source evidence, not as live authority.",
            ],
            claims: [
                makeClaim(
                    "claim-graphify-territory-subordinate",
                    "Graphify-derived compiled artifacts stay subordinate to runtime, proof, and docs truth layers.",
                    0.98,
                    ["ev-graphify-surface-hierarchy", "ev-compiled-artifacts-runtime-layout"],
                ),
                makeClaim(
                    "claim-graphify-target-state-scaffold",
                    "The bridge output is target-state scaffolding and should not be treated as live truth.",
                    0.96,
                    ["ev-compiled-artifacts-scaffold-readme", "ev-graphify-surface-hierarchy"],
                ),
            ],
            replaySuites: ["graphify-territory-map-smoke", "proof-surface-order-smoke"],
            rollbackKey: "rollback:graphify-compiled-artifacts:map-of-territory",
            sourceRoots: sharedSourceRoots,
        },
        {
            artifactId: "ca_graphify_concept_page_01",
            kind: "concept_page",
            title: "Compiled artifact substrate concept page",
            summary: "Explains the markdown body plus authoritative sidecar pair, the provenance block contract, and why the bridge remains off-path.",
            subjectIds: ["topic:compiled-artifacts", "topic:graphify", "topic:provenance"],
            confidence: 0.95,
            evidence: conceptPageEvidence,
            counterevidence: [
                makeEvidence(
                    "cevi-concept-live-boundary",
                    makeSourceRef(
                        "file",
                        "docs/architecture/teacher-v3.md#what-problem-teacher-v3-solves",
                        "Teacher v3 is an off-path compiler of graph structure and compiled artifacts, not an arbiter of current truth.",
                        "raw_source",
                        "teacher_lint",
                    ),
                    "teacher_lint",
                ),
            ],
            openQuestions: [
                "Should future content-hash verification include normalized frontmatter or just the markdown body?",
                "Which later lane should own stale-hash detection after the pack is generated?",
            ],
            promotionNotes: [
                "The sidecar JSON should be the authoritative metadata source.",
                "Use the sidecar as the machine-readable source of truth.",
                "Recompute content hashes whenever the markdown body changes.",
            ],
            claims: [
                makeClaim(
                    "claim-concept-body-plus-sidecar",
                    "A compiled artifact should be represented as a markdown body plus a canonical sidecar metadata record.",
                    0.99,
                    ["ev-compiled-artifacts-core-rules", "ev-compiled-artifacts-sidecar-shape"],
                ),
                makeClaim(
                    "claim-concept-provenance-explains-production",
                    "The provenance block must explain where the artifact came from and how it was produced.",
                    0.97,
                    ["ev-compiled-artifacts-provenance", "ev-compiled-artifacts-sidecar-shape"],
                ),
            ],
            replaySuites: ["compiled-artifact-pair-shape-smoke", "sidecar-authority-smoke"],
            rollbackKey: "rollback:graphify-compiled-artifacts:concept-page",
            sourceRoots: sharedSourceRoots,
        },
        {
            artifactId: "ca_graphify_neighborhood_summary_01",
            kind: "neighborhood_summary",
            title: "Graphify neighborhood summary",
            summary: "Summarizes the source neighborhoods feeding the bridge: architecture docs, scaffold fixtures, and the target-state compiled-artifact pack surface.",
            subjectIds: ["topic:graphify", "topic:source-neighborhoods", "topic:compiled-artifacts"],
            confidence: 0.92,
            evidence: neighborhoodSummaryEvidence,
            counterevidence: [
                makeEvidence(
                    "cevi-neighborhood-not-authority",
                    makeSourceRef(
                        "file",
                        "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
                        "Teacher v3 reporting may summarize and cross-reference truth, but it must not become a new source of truth for the live runtime.",
                        "raw_source",
                        "teacher_lint",
                    ),
                    "teacher_lint",
                ),
            ],
            openQuestions: [
                "Should future neighborhood summaries group by truth layer or by source topic?",
                "Which repository surface should own source-neighborhood promotion review?",
            ],
            promotionNotes: [
                "Keep the source neighborhoods explicit and bounded.",
                "Make shipped-versus-target labels visible in downstream review surfaces.",
            ],
            claims: [
                makeClaim(
                    "claim-neighborhood-bounded",
                    "Neighborhood summaries should stay bounded and inspectable while preserving evidence refs.",
                    0.94,
                    ["ev-teacher-v3-layers", "ev-compiled-artifacts-scaffold-manifest"],
                ),
                makeClaim(
                    "claim-neighborhood-ordered",
                    "Source neighborhoods must remain ordered under runtime, proof, docs, then proposal truth precedence.",
                    0.95,
                    ["ev-teacher-v3-proof-surface-hierarchy", "ev-teacher-v3-layers"],
                ),
            ],
            replaySuites: ["graphify-neighborhood-summary-smoke", "bounded-review-surface-smoke"],
            rollbackKey: "rollback:graphify-compiled-artifacts:neighborhood-summary",
            sourceRoots: sharedSourceRoots,
        },
        {
            artifactId: "ca_graphify_provenance_gap_report_01",
            kind: "provenance_gap_report",
            title: "Graphify provenance gap report",
            summary: "Lists the follow-on implementation gaps that still need hash verification, replay wiring, and import-slice work before any promotion or import claim.",
            subjectIds: ["topic:graphify", "topic:provenance", "topic:teacher-v3-lints"],
            confidence: 0.90,
            evidence: provenanceGapEvidence,
            counterevidence: [
                makeEvidence(
                    "cevi-provenance-shadow-only",
                    makeSourceRef(
                        "file",
                        "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
                        "The target-state proof bundle is an overlay on top of the first three surfaces, not a replacement for them.",
                        "raw_source",
                        "teacher_lint",
                    ),
                    "teacher_lint",
                ),
            ],
            openQuestions: [
                "Should hash verification live in the bridge writer or a separate validator?",
                "Which later lane should consume the replay suites carried on this proposal envelope?",
            ],
            promotionNotes: [
                "Use this report only for shadow review.",
                "Do not treat missing launch lanes as successful live adoption.",
            ],
            claims: [
                makeClaim(
                    "claim-gap-needs-hash-verification",
                    "The bridge still needs deterministic hash verification and replayable manifest wiring before it should be treated as promotable.",
                    0.98,
                    ["ev-compiled-artifacts-provenance", "ev-teacher-v3-lints-ci-first"],
                ),
                makeClaim(
                    "claim-gap-shadow-only",
                    "This report is a shadow-review surface and not live truth.",
                    0.97,
                    ["ev-teacher-v3-lints-ci-first", "cevi-provenance-shadow-only"],
                ),
            ],
            replaySuites: ["provenance-gap-smoke", "lint-boundary-smoke"],
            rollbackKey: "rollback:graphify-compiled-artifacts:provenance-gap-report",
            sourceRoots: sharedSourceRoots,
        },
    ].map((spec) => ({
        ...spec,
        createdAt,
        updatedAt: createdAt,
        proposalLane: "compiler",
        status: "proposed",
        packId: null,
        proposalId: null,
    }));
}
function buildArtifactMarkdown(spec, context) {
    const frontmatter = makeFrontmatter([
        `artifact_id: ${spec.artifactId}`,
        `kind: ${spec.kind}`,
        `status: proposed`,
        `title: ${spec.title}`,
        `proposal_id: ${context.proposalId}`,
        `proposal_lane: compiler`,
        `pack_id: ${context.packId}`,
        `graphify_run_id: ${context.graphifyRunId}`,
        `graphify_run_hash: ${context.graphifyRun.graphHash}`,
        `source_bundle_id: ${context.graphifyRun.sourceBundleId}`,
        `source_bundle_hash: ${context.graphifyRun.sourceBundleHash}`,
        `subject_ids:`,
        ...spec.subjectIds.map((subjectId) => `  - ${subjectId}`),
        `confidence: ${spec.confidence}`,
        `created_at: ${spec.createdAt}`,
        `updated_at: ${spec.updatedAt}`,
    ]);
    const evidenceLines = quoteSourceList(spec.evidence);
    const counterevidenceLines = spec.counterevidence.length > 0 ? quoteSourceList(spec.counterevidence) : "- none";
    const claimLines = spec.claims.map((claim) => `- \`${claim.claimId}\` (${claim.status}, ${claim.confidence.toFixed(2)}) — ${claim.text}`).join("\n");
    const summaryText = spec.summary;
    return [
        frontmatter,
        "## Summary",
        "",
        summaryText,
        "",
        "## Stronger-truth anchors",
        "",
        "- runtime truth: not used",
        "- proof truth: not used",
        "- docs truth: see evidence refs below",
        "- fixture truth: see evidence refs below",
        "",
        "## Evidence",
        evidenceLines,
        "",
        "## Counterevidence / boundary",
        counterevidenceLines,
        "",
        "## Claims",
        claimLines,
        "",
        "## Open questions",
        quoteList(spec.openQuestions),
        "",
        "## Promotion notes",
        quoteList(spec.promotionNotes),
        "",
    ].join("\n");
}
function buildArtifactMeta(spec, context, contentHash) {
    return {
        schemaVersion: 1,
        artifactId: spec.artifactId,
        kind: spec.kind,
        title: spec.title,
        status: "proposed",
        packId: context.packId,
        proposalId: context.proposalId,
        proposalLane: "compiler",
        subjectIds: [...spec.subjectIds],
        evidence: spec.evidence,
        counterevidence: spec.counterevidence,
        provenance: {
            producer: "graphify",
            producerVersion: context.graphifyRun.graphifyVersion,
            promptHash: sha256Text(JSON.stringify({
                artifactId: spec.artifactId,
                bundleId: context.bundleId,
                graphifyRunId: context.graphifyRunId,
                graphHash: context.graphifyRun.graphHash,
                sourceBundleHash: context.graphifyRun.sourceBundleHash,
            })),
            scope: "graphify/compiled-artifacts",
            idempotencyKey: sha256Text(JSON.stringify({
                packId: context.packId,
                proposalId: context.proposalId,
                artifactId: spec.artifactId,
                graphHash: context.graphifyRun.graphHash,
            })),
            sourceRoots: [...spec.sourceRoots],
            transformChain: ["extract", "cluster", "synthesize", "validate"],
            sourceBundleId: context.graphifyRun.sourceBundleId,
            sourceBundleHash: context.graphifyRun.sourceBundleHash,
            graphHash: context.graphifyRun.graphHash,
            graphifyRunId: context.graphifyRunId,
        },
        contentHash,
        markdownPath: context.bundlePaths.artifacts[spec.artifactId].markdown,
        metaPath: context.bundlePaths.artifacts[spec.artifactId].meta,
        createdAt: spec.createdAt,
        updatedAt: spec.updatedAt,
        confidence: spec.confidence,
        claims: spec.claims,
        promotion: {
            replaySuites: [...spec.replaySuites],
            rollbackKey: spec.rollbackKey,
        },
    };
}
function buildDefaultGraphifyCompiledArtifactPack(input = {}) {
    const graphifyRun = defaultGraphifyRunMetadata(input);
    const bundleId = graphifyRun.bundleId;
    const bundleSlug = graphifyRun.bundleSlug;
    const proposalId = normalizeText(input.proposalId) ?? `prop_graphify_compiled_artifacts_${bundleSlug}`;
    const packId = normalizeText(input.packId) ?? `pack_graphify_compiled_artifacts_${bundleSlug}`;
    const outputDir = path.resolve(input.outputDir ?? resolveGraphifyCompiledArtifactPackOutputDir({ bundleId, bundleStartedAt: graphifyRun.bundleStartedAt }));
    const graphifyRunId = graphifyRun.graphifyRunId;
    const createdAt = graphifyRun.bundleStartedAt;
    const bundleRoot = outputDir;
    const paths = {
        manifest: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.manifest),
        compilerProposal: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.compilerProposal),
        surfaceMap: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.surfaceMap),
        proposalReport: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.proposalReport),
        verdict: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.verdict),
        artifactKinds: {},
        artifacts: {},
    };
    const artifactSpecs = (input.artifactSpecs ?? buildDefaultArtifactSpecs(graphifyRun)).map((spec) => ({
        ...spec,
        proposalId,
        packId,
    }));
    for (const spec of artifactSpecs) {
        const artifactPaths = {
            markdown: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.artifactsDir, spec.artifactId, "artifact.md"),
            meta: path.join(bundleRoot, GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.artifactsDir, spec.artifactId, "artifact.meta.json"),
        };
        paths.artifactKinds[spec.artifactId] = artifactPaths;
        paths.artifacts[spec.artifactId] = artifactPaths;
    }
    const bundlePaths = {
        manifest: relativeBundlePath(paths.manifest, bundleRoot),
        compilerProposal: relativeBundlePath(paths.compilerProposal, bundleRoot),
        surfaceMap: relativeBundlePath(paths.surfaceMap, bundleRoot),
        proposalReport: relativeBundlePath(paths.proposalReport, bundleRoot),
        verdict: relativeBundlePath(paths.verdict, bundleRoot),
        artifacts: Object.fromEntries(Object.entries(paths.artifacts).map(([artifactId, artifactPaths]) => [artifactId, {
                markdown: relativeBundlePath(artifactPaths.markdown, bundleRoot),
                meta: relativeBundlePath(artifactPaths.meta, bundleRoot),
            }])),
    };
    const artifactEntries = artifactSpecs.map((spec) => {
        const markdown = buildArtifactMarkdown(spec, {
            bundleId,
            packId,
            proposalId,
            graphifyRunId,
            graphifyRun,
            paths,
            bundlePaths,
        });
        const contentHash = sha256Text(markdown);
        const meta = buildArtifactMeta(spec, {
            bundleId,
            packId,
            proposalId,
            graphifyRunId,
            graphifyRun,
            paths,
            bundlePaths,
        }, contentHash);
        return {
            artifactId: spec.artifactId,
            kind: spec.kind,
            title: spec.title,
            summary: spec.summary,
            markdown,
            meta,
            contentHash,
            markdownPath: meta.markdownPath,
            metaPath: meta.metaPath,
        };
    });
    const artifactSummaries = artifactEntries.map((entry) => ({
        artifactId: entry.artifactId,
        kind: entry.kind,
        title: entry.title,
        markdownPath: entry.markdownPath,
        metaPath: entry.metaPath,
        contentHash: entry.contentHash,
        summary: entry.summary,
    }));
    const packManifest = {
        contract: "graphify_compiled_artifact_pack.v1",
        packId,
        title: "Graphify compiled artifact pack (bridge scaffold)",
        status: "proposed",
        proposalId,
        lane: "compiler",
        scope: "graphify/compiled-artifacts",
        createdAt: graphifyRun.bundleStartedAt,
        updatedAt: graphifyRun.bundleStartedAt,
        graphifyRun: {
            runId: graphifyRunId,
            graphifyVersion: graphifyRun.graphifyVersion,
            graphifyCommand: graphifyRun.graphifyCommand,
            sourceBundleId: graphifyRun.sourceBundleId,
            sourceBundleHash: graphifyRun.sourceBundleHash,
            graphHash: graphifyRun.graphHash,
            configHash: graphifyRun.configHash,
            labelsHash: graphifyRun.labelsHash,
        },
        sourceDocs: [...graphifyRun.sourceDocs],
        sourceFixtures: [...graphifyRun.sourceFixtures],
        sourceRoots: ["docs/architecture", "artifacts/fixtures/compiled-artifacts/target-state-scaffold"],
        artifacts: artifactSummaries,
        notes: [
            "Graphify outputs are derived and do not supersede runtime, proof, or docs truth.",
            "The sidecar remains authoritative for each compiled artifact record.",
            "Sidecar hashes should be regenerated whenever the markdown body changes.",
        ],
    };
    const sourceTruthAnchors = [
        {
            id: "compiled-artifacts-core-rules",
            state: "shipped",
            kind: "docs_truth",
            source: "docs/architecture/compiled-artifacts.md#core-rules",
            note: "compiled artifacts are derived, off-path knowledge products",
        },
        {
            id: "compiled-artifacts-provenance-fields",
            state: "shipped",
            kind: "docs_truth",
            source: "docs/architecture/compiled-artifacts.md#provenance-fields",
            note: "The provenance block should explain where the artifact came from and how it was produced.",
        },
        {
            id: "teacher-v3-layers",
            state: "shipped",
            kind: "docs_truth",
            source: "docs/architecture/teacher-v3.md#layers",
            note: "Compiled artifact layer, candidate graph layer, promoted pack layer.",
        },
        {
            id: "teacher-v3-proof-surface-hierarchy",
            state: "shipped",
            kind: "docs_truth",
            source: "docs/architecture/teacher-v3-proof.md#surface-hierarchy",
            note: "Teacher v3 should read truth in runtime, proof, docs, then proposal order.",
        },
        {
            id: "compiled-artifacts-scaffold",
            state: "shipped",
            kind: "fixture_truth",
            source: "artifacts/fixtures/compiled-artifacts/target-state-scaffold/pack.manifest.json",
            note: "This directory is intentionally synthetic target-state scaffolding.",
        },
    ];
    const compilerProposalEvidence = [
        makeEvidence(
            "ev-graphify-surface-hierarchy",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3-proof.md#surface-hierarchy",
                "Teacher v3 should read truth in runtime, proof, docs, then proposal order.",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-core-rules",
            makeSourceRef(
                "file",
                "docs/architecture/compiled-artifacts.md#core-rules",
                "compiled artifacts are derived, off-path knowledge products",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-provenance",
            makeSourceRef(
                "file",
                "docs/architecture/compiled-artifacts.md#provenance-fields",
                "The provenance block should explain where the artifact came from and how it was produced.",
            ),
        ),
        makeEvidence(
            "ev-compiled-artifacts-scaffold-readme",
            makeSourceRef(
                "file",
                "artifacts/fixtures/compiled-artifacts/target-state-scaffold/README.md",
                "This directory is intentionally synthetic target-state scaffolding.",
            ),
        ),
    ];
    const compilerProposalCounterevidence = [
        makeEvidence(
            "cevi-graphify-target-state-only",
            makeSourceRef(
                "file",
                "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
                "The target-state proof bundle is an overlay on top of the first three surfaces, not a replacement for them.",
                "raw_source",
                "teacher_lint",
            ),
            "teacher_lint",
        ),
    ];
    const compilerProposal = {
        contract: "graphify_compiled_artifact_pack_compiler_proposal.v1",
        proposalId,
        proposalClass: "compiler",
        lane: "compiler",
        status: "validated",
        reviewMode: "promotable",
        lineage: {
            proposalClass: "compiler",
            producerVersion: graphifyRun.graphifyVersion,
            producerBuildId: graphifyRunId,
            promptHash: sha256Text(JSON.stringify({
                bundleId,
                proposalId,
                packId,
                graphifyRunId,
                graphHash: graphifyRun.graphHash,
                sourceBundleHash: graphifyRun.sourceBundleHash,
            })),
            templateId: "graphify-compiled-artifacts/compiler-v1",
            scope: "graphify/compiled-artifacts",
            profile: "bridge-scaffold",
            idempotencyKey: sha256Text(JSON.stringify({
                bundleId,
                proposalId,
                packId,
                graphifyRunId,
                graphHash: graphifyRun.graphHash,
            })),
            sourceBundleId: graphifyRun.sourceBundleId,
            parentProposalIds: [],
        },
        subjectIds: ["topic:graphify", "topic:compiled-artifacts", "topic:teacher-v3", "topic:provenance"],
        evidence: compilerProposalEvidence,
        counterevidence: compilerProposalCounterevidence,
        payload: {
            kind: "graphify-compiled-artifact-pack",
            summary: "Graphify-derived compiled-artifact pack with explicit evidence refs, provenance, and stronger-truth anchoring.",
            packId,
            outputRoot: "compiled-artifacts",
            artifactKinds: [...GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1],
            sourceBundleId: graphifyRun.sourceBundleId,
            sourceBundleHash: graphifyRun.sourceBundleHash,
            graphHash: graphifyRun.graphHash,
            graphifyRunId,
            sourceDocs: [...graphifyRun.sourceDocs],
            sourceFixtures: [...graphifyRun.sourceFixtures],
        },
        expectedEffect: {
            retrieval: "better",
            truthRisk: "low",
            reviewBurden: "bounded",
        },
        confidence: 0.95,
        replaySuites: ["graphify-pack-shape-smoke", "compiled-artifacts-provenance-smoke"],
        rollbackKey: `rollback:graphify-compiled-artifacts:${packId}`,
        replayGate: {
            proposalClass: "compiler",
            reviewMode: "promotable",
            dimensions: {
                truthInvariants: {
                    name: "truth_invariants",
                    summary: "Graphify bridge: keep derived output subordinate to explicit authority.",
                    requirements: [
                        "Explicit correction memory still outranks Graphify synthesis.",
                        "The live path stays read-only to the proposal.",
                        "Evidence refs stay attached to any non-trivial claim.",
                    ],
                },
                attributionFloor: {
                    name: "attribution_floor",
                    summary: "Graphify bridge: every proposed change needs clear evidence coverage.",
                    requirements: [
                        "Every proposal carries durable evidence refs.",
                        "Source ids must be stable record ids, not display labels.",
                        "Unattributed payload stays out of promotion.",
                    ],
                },
                boundedness: {
                    name: "boundedness",
                    summary: "Graphify bridge: keep the reviewable surface compact and inspectable.",
                    requirements: [
                        "Proposal subject sets stay finite and small.",
                        "Payloads avoid raw corpus dumps and unbounded excerpts.",
                        "Replay fits inside a single review pass.",
                    ],
                },
                reversibility: {
                    name: "reversibility",
                    summary: "Graphify bridge: preserve rollback and replay identity.",
                    requirements: [
                        "RollbackKey identifies the reversible path.",
                        "Prior state remains recoverable for replay.",
                        "Rejected or superseded proposals keep lineage.",
                    ],
                },
            },
        },
        strongerTruthAnchors: sourceTruthAnchors,
        createdAt,
        updatedAt: createdAt,
        targetStateOnly: true,
    };
    const controlSurfaces = [
        {
            id: "pack-manifest",
            state: "target",
            kind: "proposal_truth",
            source: bundlePaths.manifest,
            note: "compiled-artifact pack manifest",
        },
        {
            id: "compiler-proposal",
            state: "target",
            kind: "proposal_truth",
            source: bundlePaths.compilerProposal,
            note: "proposal envelope with evidence, rollback key, and replay gate",
        },
        {
            id: "surface-map",
            state: "target",
            kind: "proposal_truth",
            source: bundlePaths.surfaceMap,
            note: "shipped-vs-target inventory",
        },
        {
            id: "proposal-report",
            state: "target",
            kind: "proposal_truth",
            source: bundlePaths.proposalReport,
            note: "bounded machine-readable report",
        },
        {
            id: "verdict",
            state: "target",
            kind: "proposal_truth",
            source: bundlePaths.verdict,
            note: "review verdict",
        },
        ...artifactEntries.map((entry) => ({
            id: entry.artifactId,
            state: "target",
            kind: entry.kind,
            source: entry.markdownPath,
            note: "markdown + sidecar pair",
        })),
    ];
    const surfaceMap = {
        contract: "graphify_compiled_artifact_pack_surface_map.v1",
        bundleId,
        packId,
        proposalId,
        sourceTruthAnchors,
        controlSurfaces,
        counts: {
            shippedSurfaceCount: sourceTruthAnchors.length,
            targetSurfaceCount: controlSurfaces.length,
            totalSurfaceCount: sourceTruthAnchors.length + controlSurfaces.length,
        },
    };
    const validation = validateGraphifyCompiledArtifactPackBundle({
        packManifest,
        compilerProposal,
        artifactEntries,
        bundlePaths,
        paths,
    });
    const proposalReport = {
        contract: "graphify_compiled_artifact_pack_proposal_report.v1",
        bundleId,
        packId,
        proposalId,
        proposalClass: compilerProposal.proposalClass,
        lane: compilerProposal.lane,
        status: compilerProposal.status,
        reviewMode: compilerProposal.reviewMode,
        surfaceCounts: surfaceMap.counts,
        sourceTruthAnchors,
        compilerProposalSummary: {
            proposalId: compilerProposal.proposalId,
            status: compilerProposal.status,
            reviewMode: compilerProposal.reviewMode,
            rollbackKey: compilerProposal.rollbackKey,
            replaySuites: [...compilerProposal.replaySuites],
            evidenceCount: compilerProposal.evidence.length,
            counterevidenceCount: compilerProposal.counterevidence.length,
            subjectCount: compilerProposal.subjectIds.length,
            confidence: compilerProposal.confidence,
        },
        packManifest: {
            path: bundlePaths.manifest,
            artifactCount: packManifest.artifacts.length,
            sourceBundleId: graphifyRun.sourceBundleId,
            sourceBundleHash: graphifyRun.sourceBundleHash,
            graphHash: graphifyRun.graphHash,
        },
        graphifyRun,
        validation: {
            ok: validation.ok,
            errors: [...validation.errors],
            bundleHash: validation.bundleHash,
            fileCount: validation.fileCount,
            artifactCount: validation.artifactCount,
        },
        publicationSafeArtifacts: controlSurfaces.map((surface) => ({
            artifactId: surface.id,
            kind: surface.kind,
            path: surface.source,
            redactions: surface.state === "target" ? ["raw source payloads", "secret-bearing values"] : ["raw source payloads"],
            containsRawLogs: false,
        })),
        recommendations: validation.ok
            ? [
                "Keep Graphify-derived surfaces off the serve path.",
                "Use the compiled-artifact pack as an inspection and diff surface first.",
                "Treat the scaffold and architecture docs as stronger truth than the proposal envelope.",
            ]
            : [
                "Repair the validation errors before considering any promotion or import claim.",
                "Keep the bridge target-state only until hash and provenance checks pass.",
            ],
        createdAt: graphifyRun.bundleStartedAt,
        updatedAt: graphifyRun.bundleStartedAt,
    };
    const verdict = {
        contract: "graphify_compiled_artifact_pack_verdict.v1",
        bundleId,
        packId,
        proposalId,
        verdict: validation.ok ? "reviewable" : "rejected",
        severity: validation.ok ? "info" : "blocking",
        why: validation.ok
            ? "Graphify compiled-artifact pack is bounded, hash-consistent, and anchored below runtime/proof/docs truth; it remains target-state only and does not supersede live authority."
            : `Graphify compiled-artifact pack failed validation: ${validation.errors.join("; ")}`,
        reviewMode: compilerProposal.reviewMode,
        targetStateOnly: true,
        surfaceCounts: surfaceMap.counts,
        strongerTruthAnchors: sourceTruthAnchors,
        validation: {
            ok: validation.ok,
            errors: [...validation.errors],
            bundleHash: validation.bundleHash,
        },
        recommendations: [...proposalReport.recommendations],
        createdAt: graphifyRun.bundleStartedAt,
        updatedAt: graphifyRun.bundleStartedAt,
    };
    const files = {
        [GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.manifest]: renderJson(packManifest),
        [GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.compilerProposal]: renderJson(compilerProposal),
        [GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.surfaceMap]: renderJson(surfaceMap),
        [GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.proposalReport]: renderJson(proposalReport),
        [GRAPHIFY_COMPILED_ARTIFACT_PACK_LAYOUT_V1.verdict]: renderJson(verdict),
    };
    for (const entry of artifactEntries) {
        files[entry.meta.markdownPath] = entry.markdown;
        files[entry.meta.metaPath] = renderJson(entry.meta);
    }
    const digest = buildGraphifyCompiledArtifactPackDigest({
        bundleId,
        packId,
        proposalId,
        files,
    });
    return {
        bundleId,
        bundleSlug,
        bundleStartedAt: graphifyRun.bundleStartedAt,
        outputDir,
        packId,
        proposalId,
        graphifyRunId,
        graphifyRun,
        packManifest,
        compilerProposal,
        surfaceMap,
        proposalReport,
        verdict,
        artifactEntries,
        artifactSummaries,
        bundlePaths,
        paths: {
            manifest: relativeRepoPath(paths.manifest),
            compilerProposal: relativeRepoPath(paths.compilerProposal),
            surfaceMap: relativeRepoPath(paths.surfaceMap),
            proposalReport: relativeRepoPath(paths.proposalReport),
            verdict: relativeRepoPath(paths.verdict),
            artifacts: Object.fromEntries(Object.entries(paths.artifactKinds).map(([artifactId, pathsForArtifact]) => [artifactId, {
                    markdown: relativeRepoPath(pathsForArtifact.markdown),
                    meta: relativeRepoPath(pathsForArtifact.meta),
                }])),
        },
        validation,
        digest,
        files,
    };
}
function validateGraphifyCompiledArtifactPackBundle(bundle) {
    const errors = [];
    const artifacts = bundle.artifactEntries ?? [];
    const requiredKinds = [...GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1];
    const seenKinds = artifacts.map((entry) => entry.kind);
    for (const requiredKind of requiredKinds) {
        if (!seenKinds.includes(requiredKind)) {
            errors.push(`missing required artifact kind: ${requiredKind}`);
        }
    }
    for (const entry of artifacts) {
        const markdownPath = bundle.bundlePaths?.artifacts?.[entry.artifactId]?.markdown ?? bundle.paths.artifacts[entry.artifactId]?.markdown;
        const metaPath = bundle.bundlePaths?.artifacts?.[entry.artifactId]?.meta ?? bundle.paths.artifacts[entry.artifactId]?.meta;
        if (!markdownPath || !metaPath) {
            errors.push(`missing artifact paths for ${entry.artifactId}`);
            continue;
        }
        if (entry.meta.markdownPath !== markdownPath) {
            errors.push(`markdown path mismatch for ${entry.artifactId}`);
        }
        if (entry.meta.metaPath !== metaPath) {
            errors.push(`meta path mismatch for ${entry.artifactId}`);
        }
        if (entry.meta.contentHash !== entry.contentHash) {
            errors.push(`content hash mismatch for ${entry.artifactId}`);
        }
        const manifestEntry = bundle.packManifest.artifacts.find((artifact) => artifact.artifactId === entry.artifactId);
        if (!manifestEntry) {
            errors.push(`manifest entry missing for ${entry.artifactId}`);
            continue;
        }
        if (manifestEntry.contentHash !== entry.contentHash) {
            errors.push(`manifest content hash mismatch for ${entry.artifactId}`);
        }
    }
    const digest = buildGraphifyCompiledArtifactPackDigest({
        bundleId: bundle.bundleId,
        packId: bundle.packId,
        proposalId: bundle.proposalId,
        files: bundle.files ?? {},
    });
    return {
        ok: errors.length === 0,
        errors,
        bundleHash: digest.bundleHash,
        fileCount: digest.fileCount,
        artifactCount: artifacts.length,
    };
}
export function resolveGraphifyCompiledArtifactPackOutputDir({ outputDir = null, bundleStartedAt = new Date(), bundleId = null, } = {}) {
    if (typeof outputDir === "string" && outputDir.trim().length > 0) {
        return path.resolve(outputDir);
    }
    const resolvedStartedAt = normalizeIsoTimestamp(bundleStartedAt);
    const resolvedBundleId = normalizeText(bundleId) ?? `graphify-compiled-artifacts-${timestampToken(resolvedStartedAt)}`;
    return path.join(DEFAULT_GRAPHIFY_COMPILED_ARTIFACT_PACK_PARENT, slugify(resolvedBundleId), "compiled-artifacts");
}
export function buildGraphifyCompiledArtifactPack(input = {}) {
    return buildDefaultGraphifyCompiledArtifactPack(input);
}
export function writeGraphifyCompiledArtifactPack(outputDir, bundle) {
    ensureDir(outputDir);
    const writtenFiles = [];
    for (const [relativePath, content] of Object.entries(bundle.files)) {
        const absolutePath = path.join(outputDir, relativePath);
        writeText(absolutePath, content);
        writtenFiles.push(absolutePath);
    }
    return {
        outputDir: path.resolve(outputDir),
        writtenFiles,
        fileCount: writtenFiles.length,
    };
}
export function buildGraphifyCompiledArtifactPackDigest(bundle) {
    const fileEntries = Object.entries(bundle.files ?? {}).sort(([left], [right]) => left.localeCompare(right));
    const files = Object.fromEntries(fileEntries.map(([relativePath, content]) => [relativePath, sha256Text(content)]));
    const bundleHash = sha256Text(fileEntries.map(([relativePath, content]) => `${relativePath}\n${sha256Text(content)}`).join("\n"));
    return {
        bundleId: bundle.bundleId,
        packId: bundle.packId,
        proposalId: bundle.proposalId,
        files,
        fileCount: fileEntries.length,
        bundleHash,
    };
}
//# sourceMappingURL=graphify-compiled-artifacts.js.map
