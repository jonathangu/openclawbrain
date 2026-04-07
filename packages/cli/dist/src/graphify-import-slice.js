#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalJson } from "@openclawbrain/contracts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultRepoRoot = path.resolve(__dirname, "../../../..");
const defaultWorkspaceRoot = path.resolve(defaultRepoRoot, "..");
const defaultOutputRoot = path.join(defaultWorkspaceRoot, "artifacts", "graphify-imports");

export const GRAPHIFY_IMPORT_SLICE_LAYOUT_V1 = {
    importSlice: "import-slice.json",
    candidatePackInput: "candidate-pack-input.json",
    importReport: "import-report.md",
    proposalEnvelope: "proposal-envelope.json",
    replayGate: "replay-gate.json",
};
export const GRAPHIFY_IMPORT_SLICE_CANDIDATE_PACK_INPUT_CONTRACT_V1 = "graphify_import_slice_candidate_pack_input.v1";
const TRUST_CLASS_EXTRACTED = "EXTRACTED";
const BLOCKED_TRUST_CLASSES = ["INFERRED", "AMBIGUOUS"];
const BLOCKED_EFFECTS = [
    "current_truth_write",
    "correction_like_memory",
    "live_eligible_edge",
    "hot_path_serve_integration",
];
const SOURCE_TRUTH_ANCHORS = [
    {
        id: "graphify-bridge-artifact-first",
        state: "shipped",
        kind: "docs_truth",
        source: "docs/architecture/graphify-bridge.md#4-artifact-first-rule",
        note: "Graphify stays artifact-first and import comes after derived artifacts.",
    },
    {
        id: "graphify-bridge-extracted-only",
        state: "shipped",
        kind: "docs_truth",
        source: "docs/architecture/graphify-bridge.md#6-extracted-inferred-ambiguous-handling",
        note: "EXTRACTED is the only trust class that can feed a later live-eligible import slice.",
    },
    {
        id: "graphify-bridge-rollback-discipline",
        state: "shipped",
        kind: "docs_truth",
        source: "docs/architecture/graphify-bridge.md#7-promotion-and-rollback-discipline",
        note: "Graphify outputs never promote themselves and must keep rollback binding explicit.",
    },
    {
        id: "compiled-artifacts-core-rules",
        state: "shipped",
        kind: "docs_truth",
        source: "docs/architecture/compiled-artifacts.md#core-rules",
        note: "Derived artifacts remain off-path and subordinate to stronger truth layers.",
    },
];

function normalizeText(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}
function slugify(value) {
    return String(value ?? "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/gu, "-")
        .replace(/^-+|-+$/gu, "")
        .replace(/-{2,}/gu, "-") || "bundle";
}
function timestampToken(value = new Date().toISOString()) {
    return String(value).replace(/[:]/g, "-");
}
function stableJson(value) {
    return canonicalJson(value);
}
function sha256Text(text) {
    return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}
function ensureDir(dirPath) {
    mkdirSync(dirPath, { recursive: true });
}
function writeText(filePath, text) {
    ensureDir(path.dirname(filePath));
    writeFileSync(filePath, text, "utf8");
    return filePath;
}
function writeJson(filePath, value) {
    return writeText(filePath, `${stableJson(value)}\n`);
}
function readJson(filePath) {
    return JSON.parse(readFileSync(filePath, "utf8"));
}
function readJsonIfExists(filePath) {
    return existsSync(filePath) ? readJson(filePath) : null;
}
function readTextIfExists(filePath) {
    return existsSync(filePath) ? readFileSync(filePath, "utf8") : null;
}
function relativeWorkspacePath(absPath, workspaceRoot) {
    const resolvedPath = path.resolve(absPath);
    const relative = path.relative(workspaceRoot, resolvedPath);
    return relative.startsWith("..") ? resolvedPath : relative.replace(/\\/g, "/");
}
function uniqueBy(items, keyFn) {
    const seen = new Set();
    const unique = [];
    for (const item of items) {
        const key = keyFn(item);
        if (seen.has(key)) {
            continue;
        }
        seen.add(key);
        unique.push(item);
    }
    return unique;
}
function summarizeArtifactRecord(record) {
    return {
        artifactId: record.artifactId,
        kind: record.kind,
        title: record.title,
        subjectIds: [...(record.subjectIds ?? [])],
        sourceRoots: [...(record.provenance?.sourceRoots ?? record.sourceRoots ?? [])],
        markdownPath: record.markdownPath,
        metaPath: record.metaPath,
        contentHash: record.contentHash ?? null,
    };
}
function loadGraphifyCompiledArtifactPack(bundleRoot) {
    const resolvedBundleRoot = path.resolve(bundleRoot);
    const manifestPath = path.join(resolvedBundleRoot, "pack.manifest.json");
    if (!existsSync(manifestPath)) {
        throw new Error(`graphify import slice expects a compiled-artifact pack root with pack.manifest.json: ${resolvedBundleRoot}`);
    }
    const manifest = readJson(manifestPath);
    const artifactSummaries = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
    if (artifactSummaries.length === 0) {
        throw new Error(`graphify import slice found no artifacts in pack.manifest.json at ${resolvedBundleRoot}`);
    }
    const artifactRecords = artifactSummaries.map((summary) => {
        const summaryMetaPath = typeof summary.metaPath === "string" && summary.metaPath.trim().length > 0
            ? path.resolve(resolvedBundleRoot, summary.metaPath)
            : path.join(resolvedBundleRoot, "artifacts", summary.artifactId, "artifact.meta.json");
        if (!existsSync(summaryMetaPath)) {
            throw new Error(`graphify import slice could not read artifact meta for ${summary.artifactId}: ${summaryMetaPath}`);
        }
        const meta = readJson(summaryMetaPath);
        const markdownPath = typeof meta.markdownPath === "string" && meta.markdownPath.trim().length > 0
            ? path.resolve(resolvedBundleRoot, meta.markdownPath)
            : path.join(resolvedBundleRoot, "artifacts", meta.artifactId, "artifact.md");
        const markdownText = readTextIfExists(markdownPath);
        return {
            ...meta,
            markdownText,
            markdownPath,
            metaPath: summaryMetaPath,
        };
    });
    const surfaceMap = readJsonIfExists(path.join(resolvedBundleRoot, "surface-map.json"));
    const proposalReport = readJsonIfExists(path.join(resolvedBundleRoot, "proposal-report.json"));
    const verdict = readJsonIfExists(path.join(resolvedBundleRoot, "verdict.json"));
    return {
        bundleRoot: resolvedBundleRoot,
        manifestPath,
        manifest,
        surfaceMap,
        proposalReport,
        verdict,
        artifacts: artifactRecords,
    };
}
function buildSourceBundleSummary(pack, workspaceRoot) {
    const artifacts = pack.artifacts.map((record) => summarizeArtifactRecord(record));
    return {
        contract: "graphify_import_slice_source_bundle.v1",
        bundleRoot: relativeWorkspacePath(pack.bundleRoot, workspaceRoot),
        packId: pack.manifest.packId ?? path.basename(pack.bundleRoot),
        title: pack.manifest.title ?? null,
        contractId: pack.manifest.contract ?? null,
        status: pack.manifest.status ?? null,
        proposalId: pack.manifest.proposalId ?? null,
        lane: pack.manifest.lane ?? null,
        scope: pack.manifest.scope ?? null,
        createdAt: pack.manifest.createdAt ?? null,
        updatedAt: pack.manifest.updatedAt ?? null,
        graphifyRun: pack.manifest.graphifyRun ?? null,
        sourceDocs: Array.isArray(pack.manifest.sourceDocs) ? [...pack.manifest.sourceDocs] : [],
        sourceFixtures: Array.isArray(pack.manifest.sourceFixtures) ? [...pack.manifest.sourceFixtures] : [],
        artifacts,
    };
}
function hashBundleSummary(summary) {
    return sha256Text(stableJson(summary));
}
function collectEvidencePointers(artifacts) {
    const pointers = [];
    const seen = new Set();
    for (const artifact of artifacts) {
        for (const evidence of artifact.evidence ?? []) {
            const pointerId = evidence.evidenceId ?? sha256Text(`${artifact.artifactId}:${evidence.sourceId ?? evidence.excerpt ?? "evidence"}`).slice(7, 19);
            const key = `${pointerId}:${artifact.artifactId}`;
            if (seen.has(key)) {
                continue;
            }
            seen.add(key);
            pointers.push({
                pointerId,
                trustClass: TRUST_CLASS_EXTRACTED,
                sourceArtifactId: artifact.artifactId,
                sourceArtifactKind: artifact.kind,
                sourceArtifactTitle: artifact.title,
                sourceKind: evidence.sourceKind,
                sourceId: evidence.sourceId,
                authority: evidence.authority ?? "raw_source",
                derivation: evidence.derivation ?? "graphify_import_slice",
                excerpt: evidence.excerpt ?? null,
                sourceHash: evidence.sourceHash ?? null,
            });
        }
    }
    return pointers;
}
function collectRationalePointers(artifacts) {
    const pointers = [];
    const seen = new Set();
    for (const artifact of artifacts) {
        for (const claim of artifact.claims ?? []) {
            const pointerId = claim.claimId ?? sha256Text(`${artifact.artifactId}:${claim.text ?? "claim"}`).slice(7, 19);
            const key = `${pointerId}:${artifact.artifactId}`;
            if (seen.has(key)) {
                continue;
            }
            seen.add(key);
            pointers.push({
                pointerId,
                trustClass: TRUST_CLASS_EXTRACTED,
                sourceArtifactId: artifact.artifactId,
                sourceArtifactKind: artifact.kind,
                sourceArtifactTitle: artifact.title,
                text: claim.text,
                confidence: claim.confidence ?? null,
                status: claim.status ?? null,
                evidenceIds: Array.isArray(claim.evidenceIds) ? [...claim.evidenceIds] : [],
            });
        }
    }
    return pointers;
}
function buildHubPriors(artifacts, sourceBundle, sliceId) {
    const hubArtifacts = artifacts.filter((artifact) => artifact.kind === "map_of_territory" || artifact.kind === "concept_page");
    return hubArtifacts.map((artifact, index) => {
        const evidenceIds = uniqueBy(artifact.evidence ?? [], (evidence) => evidence.evidenceId ?? JSON.stringify(evidence)).map((evidence) => evidence.evidenceId ?? null).filter((value) => value !== null);
        const rationaleIds = uniqueBy(artifact.claims ?? [], (claim) => claim.claimId ?? JSON.stringify(claim)).map((claim) => claim.claimId ?? null).filter((value) => value !== null);
        return {
            priorId: `hub-prior-${index + 1}-${slugify(artifact.artifactId)}`,
            trustClass: TRUST_CLASS_EXTRACTED,
            kind: "hub_prior",
            hubId: artifact.artifactId,
            label: artifact.title,
            title: artifact.title,
            artifactKind: artifact.kind,
            subjectIds: Array.isArray(artifact.subjectIds) ? [...artifact.subjectIds] : [],
            sourceRoots: Array.isArray(artifact.sourceRoots ?? artifact.provenance?.sourceRoots) ? [...(artifact.sourceRoots ?? artifact.provenance?.sourceRoots)] : [],
            sourceArtifactId: artifact.artifactId,
            sourceArtifactPath: artifact.markdownPath,
            sourceMetaPath: artifact.metaPath,
            sourceBundleId: sourceBundle.packId,
            sourceBundleHash: sourceBundle.sourceBundleHash,
            evidencePointerIds: evidenceIds,
            rationalePointerIds: rationaleIds,
            authority: "raw_source",
            derivation: "graphify_import_slice",
            rollbackKey: `rollback:graphify-import-slice:${sliceId}:${slugify(artifact.artifactId)}`,
        };
    });
}
function buildNeighborhoodPriors(artifacts, sourceBundle, sliceId) {
    const neighborhoodArtifacts = artifacts.filter((artifact) => artifact.kind === "neighborhood_summary");
    return neighborhoodArtifacts.map((artifact, index) => {
        const evidenceIds = uniqueBy(artifact.evidence ?? [], (evidence) => evidence.evidenceId ?? JSON.stringify(evidence)).map((evidence) => evidence.evidenceId ?? null).filter((value) => value !== null);
        const rationaleIds = uniqueBy(artifact.claims ?? [], (claim) => claim.claimId ?? JSON.stringify(claim)).map((claim) => claim.claimId ?? null).filter((value) => value !== null);
        return {
            priorId: `neighborhood-prior-${index + 1}-${slugify(artifact.artifactId)}`,
            trustClass: TRUST_CLASS_EXTRACTED,
            kind: "neighborhood_prior",
            neighborhoodId: artifact.artifactId,
            label: artifact.title,
            title: artifact.title,
            artifactKind: artifact.kind,
            subjectIds: Array.isArray(artifact.subjectIds) ? [...artifact.subjectIds] : [],
            sourceRoots: Array.isArray(artifact.sourceRoots ?? artifact.provenance?.sourceRoots) ? [...(artifact.sourceRoots ?? artifact.provenance?.sourceRoots)] : [],
            sourceArtifactId: artifact.artifactId,
            sourceArtifactPath: artifact.markdownPath,
            sourceMetaPath: artifact.metaPath,
            sourceBundleId: sourceBundle.packId,
            sourceBundleHash: sourceBundle.sourceBundleHash,
            evidencePointerIds: evidenceIds,
            rationalePointerIds: rationaleIds,
            authority: "raw_source",
            derivation: "graphify_import_slice",
            rollbackKey: `rollback:graphify-import-slice:${sliceId}:${slugify(artifact.artifactId)}`,
        };
    });
}
function buildTruthBoundary(sourceBundle) {
    return {
        authority: "graphify_proposal_truth",
        strongerTruthLayers: ["runtime truth", "proof truth", "docs truth"],
        correctionPrecedence: [
            "explicit correction memory",
            "recent raw user/source turns",
            "raw proof/runtime evidence",
            "frozen docs truth",
        ],
        allowedTrustClasses: [TRUST_CLASS_EXTRACTED],
        blockedTrustClasses: [...BLOCKED_TRUST_CLASSES],
        blockedEffects: [...BLOCKED_EFFECTS],
        artifactFirst: true,
        rollbackSafe: true,
        removable: true,
        liveEligible: false,
        sourceBundleKind: sourceBundle.contractId,
    };
}
function buildReplayGate(input) {
    return {
        contract: "graphify_import_slice_replay_gate.v1",
        gateId: `replay-gate:${input.sliceId}`,
        bundleId: input.sliceId,
        proposalId: input.proposalId,
        proposalClass: "import",
        reviewMode: "candidate_only",
        status: "validated",
        targetStateOnly: true,
        allowedTrustClasses: [TRUST_CLASS_EXTRACTED],
        blockedTrustClasses: [...BLOCKED_TRUST_CLASSES],
        blockedEffects: [...BLOCKED_EFFECTS],
        rollbackKey: input.rollbackKey,
        strongerTruthAnchors: [...SOURCE_TRUTH_ANCHORS],
        requirements: [
            {
                id: "artifact-first",
                summary: "Import stays artifact-first and never outranks runtime, proof, or docs truth.",
                requirements: [
                    "Import slice is derived from Graphify output, not live state.",
                    "No hot-path serve integration is added here.",
                    "Graphify remains off-path and review-only.",
                ],
            },
            {
                id: "extracted-only",
                summary: "Only EXTRACTED priors are allowed into the Wave 1 slice.",
                requirements: [
                    "Hub priors remain EXTRACTED only.",
                    "Neighborhood priors remain EXTRACTED only.",
                    "INFERRED and AMBIGUOUS stay blocked from the import slice.",
                ],
            },
            {
                id: "correction-precedence",
                summary: "Explicit corrections keep precedence over any Graphify-derived prior.",
                requirements: [
                    "No correction-like durable memory is written.",
                    "No current-truth-like overwrite path is introduced.",
                    "Any conflict routes to review rather than mutation.",
                ],
            },
            {
                id: "rollback-safe",
                summary: "The slice must be removable and rollback-bound.",
                requirements: [
                    "Rollback key is explicit and stable.",
                    "Import surfaces are bounded and inspectable.",
                    "Rejected slices keep no live eligibility.",
                ],
            },
            {
                id: "boundedness",
                summary: "Keep the reviewable surface compact enough for one operator sitting.",
                requirements: [
                    "Evidence and rationale pointers stay source-grounded.",
                    "The slice avoids raw corpus dumps.",
                    "The output remains small and explicit.",
                ],
            },
        ],
    };
}
function buildCandidatePackInput(input) {
    const importedPriors = {
        hubPriors: input.hubPriors.map((prior) => ({ ...prior })),
        neighborhoodPriors: input.neighborhoodPriors.map((prior) => ({ ...prior })),
        evidencePointers: input.evidencePointers.map((pointer) => ({ ...pointer })),
    };
    return {
        contract: GRAPHIFY_IMPORT_SLICE_CANDIDATE_PACK_INPUT_CONTRACT_V1,
        inputId: `candidate-pack-input:${input.sliceId}`,
        bundleId: `candidate-pack-input:${input.sliceId}`,
        sliceId: input.sliceId,
        proposalId: input.proposalId,
        candidatePackId: `candidate-pack:${input.sliceId}`,
        sourceBundleId: input.sourceBundleId,
        sourceBundleHash: input.sourceBundleHash,
        sourceBundleKind: input.sourceBundleKind,
        generatedAt: input.generatedAt,
        updatedAt: input.generatedAt,
        reviewMode: "candidate_only",
        targetStateOnly: true,
        truthBoundary: input.truthBoundary,
        seedingBoundary: {
            liveEligible: false,
            currentTruthWrites: false,
            correctionMemoryWrites: false,
            hotPathDependency: false,
            removable: true,
            rollbackSafe: true,
        },
        provenance: {
            producer: "graphify-import-slice",
            producerVersion: input.graphifyVersion,
            producerRunId: input.graphifyRunId,
            graphifyCommand: input.graphifyCommand,
            scope: "graphify/import-slice/candidate-pack-input",
            idempotencyKey: sha256Text(stableJson({
                sliceId: input.sliceId,
                proposalId: input.proposalId,
                sourceBundleId: input.sourceBundleId,
                sourceBundleHash: input.sourceBundleHash,
                sourceBundleKind: input.sourceBundleKind,
                graphifyRunId: input.graphifyRunId,
                graphifyVersion: input.graphifyVersion,
            })),
            sourceRoots: Array.isArray(input.sourceBundle.sourceDocs)
                ? input.sourceBundle.sourceDocs.filter((value) => typeof value === "string")
                : [],
            transformChain: [
                "extract",
                "candidate_pack_seed",
                "rollback_bound",
            ],
        },
        importedPriors,
        counts: {
            hubPriors: input.hubPriorCount,
            neighborhoodPriors: input.neighborhoodPriorCount,
            evidencePointers: input.evidencePointerCount,
        },
        notes: [
            "This candidate-pack input is derived from EXTRACTED-only Graphify priors.",
            "It is rollback-bound, removable, and never hot-path eligible.",
            "It seeds later candidate compilation inputs without writing current truth or correction memory.",
        ],
    };
}
function buildProposalEnvelope(input, sourceBundle) {
    const promptHash = sha256Text(stableJson({
        sliceId: input.sliceId,
        proposalId: input.proposalId,
        sourceBundleHash: input.sourceBundleHash,
        sourceBundleId: input.sourceBundleId,
        graphifyRunId: input.graphifyRunId,
        graphifyVersion: input.graphifyVersion,
    }));
    return {
        contract: "graphify_import_slice_proposal.v1",
        proposalId: input.proposalId,
        proposalClass: "import",
        lane: "import",
        status: "validated",
        reviewMode: "candidate_only",
        lineage: {
            proposalClass: "import",
            producerVersion: input.graphifyVersion,
            producerBuildId: input.graphifyRunId,
            promptHash,
            templateId: "graphify-import-slice/import-v1",
            scope: "graphify/import-slice",
            profile: "bridge-scaffold",
            idempotencyKey: sha256Text(stableJson({
                proposalId: input.proposalId,
                sliceId: input.sliceId,
                sourceBundleHash: input.sourceBundleHash,
                sourceBundleId: input.sourceBundleId,
                graphifyRunId: input.graphifyRunId,
            })),
            sourceBundleId: input.sourceBundleId,
            parentProposalIds: [],
        },
        subjectIds: ["topic:graphify", "topic:import-slice", "topic:compiled-artifacts", "topic:truth-boundary"],
        evidence: [
            {
                evidenceId: "ev-graphify-bridge-artifact-first",
                sourceKind: "file",
                sourceId: "docs/architecture/graphify-bridge.md#4-artifact-first-rule",
                authority: "raw_source",
                derivation: "teacher_compilation",
                excerpt: "Graphify may help produce compiled artifacts, lints, or candidate imports, but it does not invent a separate lifecycle beside that chain.",
                sourceHash: sha256Text("docs/architecture/graphify-bridge.md#4-artifact-first-rule\nGraphify may help produce compiled artifacts, lints, or candidate imports, but it does not invent a separate lifecycle beside that chain."),
            },
            {
                evidenceId: "ev-graphify-bridge-extracted",
                sourceKind: "file",
                sourceId: "docs/architecture/graphify-bridge.md#6-extracted-inferred-ambiguous-handling",
                authority: "raw_source",
                derivation: "teacher_compilation",
                excerpt: "EXTRACTED is the only class that can be considered for any later live-eligible import slice.",
                sourceHash: sha256Text("docs/architecture/graphify-bridge.md#6-extracted-inferred-ambiguous-handling\nEXTRACTED is the only class that can be considered for any later live-eligible import slice."),
            },
            {
                evidenceId: "ev-graphify-bridge-rollback",
                sourceKind: "file",
                sourceId: "docs/architecture/graphify-bridge.md#7-promotion-and-rollback-discipline",
                authority: "raw_source",
                derivation: "teacher_compilation",
                excerpt: "Graphify outputs never promote themselves.",
                sourceHash: sha256Text("docs/architecture/graphify-bridge.md#7-promotion-and-rollback-discipline\nGraphify outputs never promote themselves."),
            },
            {
                evidenceId: "ev-compiled-artifacts-core-rules",
                sourceKind: "file",
                sourceId: "docs/architecture/compiled-artifacts.md#core-rules",
                authority: "raw_source",
                derivation: "teacher_compilation",
                excerpt: "compiled artifacts are derived, off-path knowledge products",
                sourceHash: sha256Text("docs/architecture/compiled-artifacts.md#core-rules\ncompiled artifacts are derived, off-path knowledge products"),
            },
            {
                evidenceId: "ev-source-bundle-pack",
                sourceKind: "file",
                sourceId: `${sourceBundle.bundleRoot}/pack.manifest.json`,
                authority: "raw_source",
                derivation: "graphify_import_slice",
                excerpt: "Graphify compiled-artifact pack selected for conservative EXTRACTED-only slicing.",
                sourceHash: sourceBundle.sourceBundleHash,
            },
        ],
        counterevidence: [
            {
                evidenceId: "cevi-graphify-not-truth",
                sourceKind: "file",
                sourceId: "docs/architecture/graphify-bridge.md#8-what-this-bridge-is-not",
                authority: "raw_source",
                derivation: "teacher_compilation",
                excerpt: "This bridge does not make Graphify a live truth layer.",
                sourceHash: sha256Text("docs/architecture/graphify-bridge.md#8-what-this-bridge-is-not\nThis bridge does not make Graphify a live truth layer."),
            },
        ],
        payload: {
            kind: "graphify-import-slice",
            summary: "Conservative EXTRACTED-only Graphify prior slice with explicit evidence/rationale pointers and rollback binding.",
            sliceId: input.sliceId,
            sourceBundleId: input.sourceBundleId,
            sourceBundleHash: input.sourceBundleHash,
            sourceBundleKind: sourceBundle.contractId,
            trustClass: TRUST_CLASS_EXTRACTED,
            hubPriorCount: input.hubPriorCount,
            neighborhoodPriorCount: input.neighborhoodPriorCount,
            evidencePointerCount: input.evidencePointerCount,
            rationalePointerCount: input.rationalePointerCount,
            removedEffects: [...BLOCKED_EFFECTS],
        },
        expectedEffect: {
            retrieval: "better",
            truthRisk: "low",
            reviewBurden: "bounded",
        },
        confidence: 0.92,
        replaySuites: ["graphify-import-slice-smoke", "truth-boundary-smoke"],
        rollbackKey: input.rollbackKey,
        replayGate: buildReplayGate(input),
        strongerTruthAnchors: [...SOURCE_TRUTH_ANCHORS],
        createdAt: input.generatedAt,
        updatedAt: input.generatedAt,
        targetStateOnly: true,
    };
}
function buildImportReportMarkdown(input) {
    const hubLines = input.hubPriors.length === 0
        ? ["- none"]
        : input.hubPriors.map((prior) => `- ${prior.label} (${prior.trustClass}) — ${prior.sourceArtifactId}`);
    const neighborhoodLines = input.neighborhoodPriors.length === 0
        ? ["- none"]
        : input.neighborhoodPriors.map((prior) => `- ${prior.label} (${prior.trustClass}) — ${prior.sourceArtifactId}`);
    const evidenceLines = input.evidencePointers.length === 0
        ? ["- none"]
        : input.evidencePointers.slice(0, 12).map((pointer) => `- \`${pointer.pointerId}\` — ${pointer.sourceId}`);
    const rationaleLines = input.rationalePointers.length === 0
        ? ["- none"]
        : input.rationalePointers.slice(0, 12).map((pointer) => `- \`${pointer.pointerId}\` — ${pointer.sourceArtifactId}: ${pointer.text}`);
    const boundaryLines = [
        "- artifact-first then import-second",
        "- EXTRACTED only for Wave 1",
        "- correction precedence preserved",
        "- rollback-safe and removable",
        "- no current-truth-like write",
        "- no correction-like durable memory",
        "- no INFERRED live-eligible edges",
        "- no hot-path serve integration",
    ];
    const sourceBundle = input.sourceBundle;
    const lines = [
        `# Graphify import slice report`,
        "",
        "Conservative EXTRACTED-only slice. Derived structure only; no live truth write.",
        "",
        `- slice id: \`${input.sliceId}\``,
        `- proposal id: \`${input.proposalId}\``,
        `- source pack: \`${sourceBundle.packId}\``,
        `- source bundle hash: \`${input.sourceBundleHash}\``,
        `- graphify run: \`${input.graphifyRunId}\``,
        `- graphify version: \`${input.graphifyVersion}\``,
        `- graphify command: \`${input.graphifyCommand}\``,
        `- output root: \`${input.outputRoot}\``,
        `- rollback key: \`${input.rollbackKey}\``,
        "",
        "## Imported priors",
        "",
        `Hub priors: ${input.hubPriorCount}`,
        `Neighborhood priors: ${input.neighborhoodPriorCount}`,
        `Evidence pointers: ${input.evidencePointerCount}`,
        `Rationale pointers: ${input.rationalePointerCount}`,
        "",
        "## Candidate pack input",
        "",
        `- candidate pack id: \`${input.candidatePackInput.candidatePackId}\``,
        `- candidate input id: \`${input.candidatePackInput.inputId}\``,
        `- target-state only: \`${input.candidatePackInput.targetStateOnly}\``,
        `- removable: \`${input.candidatePackInput.seedingBoundary.removable}\``,
        `- rollback safe: \`${input.candidatePackInput.seedingBoundary.rollbackSafe}\``,
        `- live eligible: \`${input.candidatePackInput.seedingBoundary.liveEligible}\``,
        `- current truth writes: \`${input.candidatePackInput.seedingBoundary.currentTruthWrites}\``,
        `- hot-path dependency: \`${input.candidatePackInput.seedingBoundary.hotPathDependency}\``,
        `- correction memory writes: \`${input.candidatePackInput.seedingBoundary.correctionMemoryWrites}\``,
        `- source bundle: \`${sourceBundle.packId}\` / \`${input.sourceBundleHash}\``,
        "",
        "### Imported prior material",
        "",
        `- hub priors: ${input.candidatePackInput.counts.hubPriors}`,
        `- neighborhood priors: ${input.candidatePackInput.counts.neighborhoodPriors}`,
        `- evidence pointers: ${input.candidatePackInput.counts.evidencePointers}`,
        `- candidate input path: \`${input.candidatePackInputPath}\``,
        "",
        "### Hub priors",
        ...hubLines,
        "",
        "### Neighborhood priors",
        ...neighborhoodLines,
        "",
        "### Evidence pointers",
        ...evidenceLines,
        "",
        "### Rationale pointers",
        ...rationaleLines,
        "",
        "## Truth boundary",
        "",
        ...boundaryLines,
        "",
        "## Replay gate",
        "",
        `- review mode: \`${input.replayGate.reviewMode}\``,
        `- allowed trust classes: ${input.replayGate.allowedTrustClasses.join(", ")}`,
        `- blocked trust classes: ${input.replayGate.blockedTrustClasses.join(", ")}`,
        `- blocked effects: ${input.replayGate.blockedEffects.join(", ")}`,
        "",
        "This slice is bounded, explicit, rollback-bound, and removable. If a later lane wants broader import, it must do so after replay and proof checks, not by widening this file.",
        "",
    ];
    return lines.join("\n");
}
function buildImportSliceDigest(files) {
    const digest = createHash("sha256");
    const entries = Object.entries(files).sort(([left], [right]) => left.localeCompare(right));
    for (const [name, text] of entries) {
        digest.update(`${name}\u0000${text}\n`);
    }
    return `sha256:${digest.digest("hex")}`;
}
export function buildGraphifyImportSlice(options = {}) {
    const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
    const workspaceRoot = path.resolve(options.workspaceRoot ?? defaultWorkspaceRoot);
    const bundleRoot = path.resolve(options.bundleRoot ?? options.bundleDir ?? options.bundlePath ?? "");
    if (!existsSync(bundleRoot)) {
        throw new Error(`graphify import slice bundle root does not exist: ${bundleRoot}`);
    }
    const generatedAt = normalizeText(options.generatedAt) ?? new Date().toISOString();
    const runId = normalizeText(options.runId) ?? `graphify-import-slice-${timestampToken(generatedAt)}`;
    const outputRoot = path.resolve(options.outputRoot ?? defaultOutputRoot);
    const outputDir = path.join(outputRoot, runId);
    const sliceId = normalizeText(options.sliceId) ?? `graphify-import-slice-${slugify(runId)}`;
    const proposalId = normalizeText(options.proposalId) ?? `prop_${slugify(sliceId)}`;
    const rollbackKey = normalizeText(options.rollbackKey) ?? `rollback:graphify-import-slice:${slugify(sliceId)}`;
    const pack = loadGraphifyCompiledArtifactPack(bundleRoot);
    const sourceBundle = buildSourceBundleSummary(pack, workspaceRoot);
    const sourceBundleHash = hashBundleSummary(sourceBundle);
    sourceBundle.sourceBundleHash = sourceBundleHash;
    const sourceBundleId = sourceBundle.packId;
    const sourceBundleKind = sourceBundle.contractId;
    const sourceBundleArtifacts = pack.artifacts;
    const evidencePointers = collectEvidencePointers(sourceBundleArtifacts);
    const rationalePointers = collectRationalePointers(sourceBundleArtifacts);
    const hubPriors = buildHubPriors(sourceBundleArtifacts, { packId: sourceBundleId, sourceBundleHash }, sliceId);
    const neighborhoodPriors = buildNeighborhoodPriors(sourceBundleArtifacts, { packId: sourceBundleId, sourceBundleHash }, sliceId);
    const counts = {
        hubPriors: hubPriors.length,
        neighborhoodPriors: neighborhoodPriors.length,
        evidencePointers: evidencePointers.length,
        rationalePointers: rationalePointers.length,
        sourceArtifacts: sourceBundleArtifacts.length,
    };
    const truthBoundary = buildTruthBoundary({ contractId: sourceBundleKind });
    const candidatePackInput = buildCandidatePackInput({
        sliceId,
        proposalId,
        sourceBundleId,
        sourceBundleHash,
        sourceBundleKind,
        generatedAt,
        graphifyRunId: sourceBundle.graphifyRun?.runId ?? null,
        graphifyVersion: sourceBundle.graphifyRun?.graphifyVersion ?? null,
        graphifyCommand: sourceBundle.graphifyRun?.graphifyCommand ?? null,
        truthBoundary,
        hubPriors,
        neighborhoodPriors,
        evidencePointers,
        hubPriorCount: counts.hubPriors,
        neighborhoodPriorCount: counts.neighborhoodPriors,
        evidencePointerCount: counts.evidencePointers,
        sourceBundle,
    });
    const importSlice = {
        contract: "graphify_import_slice.v1",
        sliceId,
        proposalId,
        bundleId: sliceId,
        sourceBundleId,
        sourceBundleHash,
        sourceBundleKind,
        sourceBundle,
        generatedAt,
        updatedAt: generatedAt,
        truthBoundary,
        counts,
        hubPriors,
        neighborhoodPriors,
        evidencePointers,
        rationalePointers,
        notes: [
            "Artifact-first then import-second.",
            "Wave 1 is EXTRACTED only.",
            "Explicit correction precedence remains stronger than any Graphify-derived prior.",
            "The slice is candidate-only and never live-eligible in this cut.",
        ],
    };
    const replayGate = buildReplayGate({
        sliceId,
        proposalId,
        rollbackKey,
        generatedAt,
    });
    const proposalEnvelope = buildProposalEnvelope({
        sliceId,
        proposalId,
        sourceBundleId,
        sourceBundleHash,
        graphifyRunId: sourceBundle.graphifyRun?.runId ?? null,
        graphifyVersion: sourceBundle.graphifyRun?.graphifyVersion ?? null,
        hubPriorCount: counts.hubPriors,
        neighborhoodPriorCount: counts.neighborhoodPriors,
        evidencePointerCount: counts.evidencePointers,
        rationalePointerCount: counts.rationalePointers,
        rollbackKey,
        generatedAt,
    }, sourceBundle);
    const reportMarkdown = buildImportReportMarkdown({
        sliceId,
        proposalId,
        sourceBundle,
        sourceBundleHash,
        graphifyRunId: sourceBundle.graphifyRun?.runId ?? null,
        graphifyVersion: sourceBundle.graphifyRun?.graphifyVersion ?? null,
        graphifyCommand: sourceBundle.graphifyRun?.graphifyCommand ?? null,
        outputRoot: relativeWorkspacePath(outputDir, workspaceRoot),
        rollbackKey,
        hubPriors,
        neighborhoodPriors,
        evidencePointers,
        rationalePointers,
        hubPriorCount: counts.hubPriors,
        neighborhoodPriorCount: counts.neighborhoodPriors,
        evidencePointerCount: counts.evidencePointers,
        rationalePointerCount: counts.rationalePointers,
        candidatePackInput,
        candidatePackInputPath: relativeWorkspacePath(path.join(outputDir, GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.candidatePackInput), workspaceRoot),
        replayGate,
    });
    const files = {
        [GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.importSlice]: stableJson(importSlice),
        [GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.candidatePackInput]: stableJson(candidatePackInput),
        [GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.importReport]: reportMarkdown,
        [GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.proposalEnvelope]: stableJson(proposalEnvelope),
        [GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.replayGate]: stableJson(replayGate),
    };
    const digest = {
        bundleHash: buildImportSliceDigest(files),
        fileCount: Object.keys(files).length,
        files: Object.fromEntries(Object.entries(files).map(([name, text]) => [name, sha256Text(text)])),
    };
    const paths = {
        importSlice: path.join(outputDir, GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.importSlice),
        candidatePackInput: path.join(outputDir, GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.candidatePackInput),
        importReport: path.join(outputDir, GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.importReport),
        proposalEnvelope: path.join(outputDir, GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.proposalEnvelope),
        replayGate: path.join(outputDir, GRAPHIFY_IMPORT_SLICE_LAYOUT_V1.replayGate),
    };
    return {
        ok: true,
        runId,
        sliceId,
        proposalId,
        rollbackKey,
        outputRoot,
        outputDir,
        bundleRoot,
        repoRoot,
        workspaceRoot,
        sourceBundleId,
        sourceBundleHash,
        sourceBundleKind,
        graphifyRunId: sourceBundle.graphifyRun?.runId ?? null,
        graphifyVersion: sourceBundle.graphifyRun?.graphifyVersion ?? null,
        graphifyCommand: sourceBundle.graphifyRun?.graphifyCommand ?? null,
        counts,
        truthBoundary,
        candidatePackInput,
        importSlice,
        proposalEnvelope,
        replayGate,
        reportMarkdown,
        files,
        paths,
        digest,
    };
}
export function writeGraphifyImportSliceBundle(outputDir, bundle) {
    rmSync(outputDir, { recursive: true, force: true });
    ensureDir(outputDir);
    const writtenFiles = [];
    for (const [name, text] of Object.entries(bundle.files)) {
        const filePath = path.join(outputDir, name);
        writeText(filePath, text);
        writtenFiles.push(filePath);
    }
    return {
        writtenFiles,
        fileCount: writtenFiles.length,
    };
}
export function resolveGraphifyImportSliceOutputDir(options = {}) {
    const runId = normalizeText(options.runId) ?? `graphify-import-slice-${timestampToken(new Date().toISOString())}`;
    return path.join(path.resolve(options.outputRoot ?? defaultOutputRoot), runId);
}
export function exportGraphifyImportSlice(options = {}) {
    try {
        const bundle = buildGraphifyImportSlice(options);
        const writeResult = writeGraphifyImportSliceBundle(bundle.outputDir, bundle);
        return {
            ok: true,
            runId: bundle.runId,
            sliceId: bundle.sliceId,
            proposalId: bundle.proposalId,
            rollbackKey: bundle.rollbackKey,
            outputRoot: bundle.outputRoot,
            outputDir: bundle.outputDir,
            bundleRoot: bundle.bundleRoot,
            repoRoot: bundle.repoRoot,
            workspaceRoot: bundle.workspaceRoot,
            sourceBundleId: bundle.sourceBundleId,
            sourceBundleHash: bundle.sourceBundleHash,
            sourceBundleKind: bundle.sourceBundleKind,
            graphifyRunId: bundle.graphifyRunId,
            graphifyVersion: bundle.graphifyVersion,
            graphifyCommand: bundle.graphifyCommand,
            counts: bundle.counts,
            truthBoundary: bundle.truthBoundary,
            candidatePackInput: bundle.candidatePackInput,
            importSlice: bundle.importSlice,
            proposalEnvelope: bundle.proposalEnvelope,
            replayGate: bundle.replayGate,
            report: bundle.reportMarkdown,
            paths: bundle.paths,
            digest: bundle.digest,
            writtenFiles: writeResult.writtenFiles,
            fileCount: writeResult.fileCount,
        };
    }
    catch (error) {
        const outputRoot = path.resolve(options.outputRoot ?? defaultOutputRoot);
        const runId = normalizeText(options.runId) ?? null;
        return {
            ok: false,
            outputRoot,
            outputDir: runId === null ? outputRoot : path.join(outputRoot, runId),
            bundleRoot: path.resolve(options.bundleRoot ?? options.bundleDir ?? options.bundlePath ?? "."),
            candidatePackInput: null,
            error: error instanceof Error ? error.message : String(error),
        };
    }
}
function formatGraphifyImportSliceSummary(result) {
    if (result.help) {
        return "";
    }
    const lines = [
        "GRAPHIFY IMPORT SLICE ok",
        `  Slice:       ${result.sliceId}`,
        `  Proposal:    ${result.proposalId}`,
        `  Source pack: ${result.sourceBundleId}`,
        `  Source hash: ${result.sourceBundleHash}`,
        `  Hub priors:  ${result.counts.hubPriors}`,
        `  Neighborhoods: ${result.counts.neighborhoodPriors}`,
        `  Evidence:    ${result.counts.evidencePointers}`,
        `  Rationale:   ${result.counts.rationalePointers}`,
        `  Candidate input: ${result.paths.candidatePackInput}`,
        `  Output root: ${result.outputRoot}`,
        `  Output dir:  ${result.outputDir}`,
        `  Import:      ${result.paths.importSlice}`,
        `  Report:      ${result.paths.importReport}`,
        `  Envelope:    ${result.paths.proposalEnvelope}`,
        `  Replay gate: ${result.paths.replayGate}`,
    ];
    return `${lines.join("\n")}\n`;
}
export function parseGraphifyImportSliceCliArgs(argv) {
    let bundleRoot = null;
    let repoRoot = defaultRepoRoot;
    let workspaceRoot = defaultWorkspaceRoot;
    let outputRoot = null;
    let runId = null;
    let json = false;
    let help = false;
    for (let index = 0; index < argv.length; index += 1) {
        const arg = argv[index];
        if (arg === "--help" || arg === "-h") {
            help = true;
            continue;
        }
        if (arg === "--json") {
            json = true;
            continue;
        }
        if (arg === "--bundle-root") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--bundle-root requires a value");
            }
            bundleRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--repo-root") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--repo-root requires a value");
            }
            repoRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--workspace-root") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--workspace-root requires a value");
            }
            workspaceRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--output-root") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--output-root requires a value");
            }
            outputRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--run-id") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--run-id requires a value");
            }
            runId = next;
            index += 1;
            continue;
        }
        throw new Error(`unknown argument for graphify-import-slice: ${arg}`);
    }
    return {
        command: "graphify-import-slice",
        bundleRoot: bundleRoot === null ? null : path.resolve(bundleRoot),
        repoRoot: path.resolve(repoRoot),
        workspaceRoot: path.resolve(workspaceRoot),
        outputRoot: outputRoot === null ? null : path.resolve(outputRoot),
        runId,
        json,
        help,
    };
}
export function runGraphifyImportSlice(argvOrOptions = {}) {
    const parsed = Array.isArray(argvOrOptions)
        ? parseGraphifyImportSliceCliArgs(argvOrOptions)
        : { command: "graphify-import-slice", json: false, help: false, ...argvOrOptions };
    if (parsed.help) {
        return {
            ok: true,
            help: true,
            summary: "",
            candidatePackInput: null,
            importSlice: null,
            proposalEnvelope: null,
            replayGate: null,
            report: null,
            paths: null,
            outputRoot: null,
            outputDir: null,
            bundleRoot: null,
            repoRoot: null,
            workspaceRoot: null,
            runId: null,
            sliceId: null,
            proposalId: null,
            counts: null,
        };
    }
    const result = exportGraphifyImportSlice({
        bundleRoot: parsed.bundleRoot,
        repoRoot: parsed.repoRoot,
        workspaceRoot: parsed.workspaceRoot,
        outputRoot: parsed.outputRoot ?? undefined,
        runId: parsed.runId ?? undefined,
    });
    if (!result.ok) {
        return {
            ok: false,
            help: false,
            summary: "",
            candidatePackInput: null,
            importSlice: null,
            proposalEnvelope: null,
            replayGate: null,
            report: null,
            paths: null,
            outputRoot: result.outputRoot ?? null,
            outputDir: result.outputDir ?? null,
            bundleRoot: result.bundleRoot ?? null,
            repoRoot: null,
            workspaceRoot: null,
            runId: null,
            sliceId: null,
            proposalId: null,
            counts: null,
            error: result.error ?? "graphify import slice failed",
        };
    }
    return {
        ok: true,
        help: false,
        summary: formatGraphifyImportSliceSummary(result),
        candidatePackInput: result.candidatePackInput,
        importSlice: result.importSlice,
        proposalEnvelope: result.proposalEnvelope,
        replayGate: result.replayGate,
        report: result.report,
        paths: result.paths,
        outputRoot: result.outputRoot,
        outputDir: result.outputDir,
        bundleRoot: result.bundleRoot,
        repoRoot: result.repoRoot,
        workspaceRoot: result.workspaceRoot,
        runId: result.runId,
        sliceId: result.sliceId,
        proposalId: result.proposalId,
        counts: result.counts,
    };
}
