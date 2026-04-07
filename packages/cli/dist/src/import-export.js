/**
 * Brain import/export: backup and restore the activation root directory.
 *
 * export: tar + gzip the entire activation root → output.tar.gz
 * import: extract tar.gz → activation root, with safety checks
 */
import { createHash } from "node:crypto";
import { execSync } from "node:child_process";
import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, readlinkSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import path from "node:path";
import { canonicalJson } from "@openclawbrain/contracts";
import { buildOpenClawSessionCorpusSnapshot } from "./session-tail.js";
import { discoverOpenClawHomes, inspectOpenClawHome } from "./openclaw-home-layout.js";
import { resolveActivationRoot } from "./resolve-activation-root.js";
import { inspectOpenClawBrainHookStatus } from "./openclaw-hook-truth.js";
import { listOpenClawProfileRuntimeLoadProofs, resolveAttachmentRuntimeLoadProofsPath } from "./attachment-truth.js";
import { buildGraphifyCompiledArtifactPack, writeGraphifyCompiledArtifactPack } from "./graphify-compiled-artifacts.js";
import { buildGraphifyImportSlice, resolveGraphifyImportSliceOutputDir, writeGraphifyImportSliceBundle } from "./graphify-import-slice.js";
import { buildGraphifyMaintenanceDiffBundle, writeGraphifyMaintenanceDiffBundle } from "./graphify-maintenance-diff.js";

function hashCanonicalJson(value) {
    return `sha256:${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}
function writeCanonicalJsonFile(filePath, value) {
    mkdirSync(path.dirname(filePath), { recursive: true });
    const text = canonicalJson(value);
    writeFileSync(filePath, `${text}\n`, "utf8");
    return {
        path: filePath,
        digest: `sha256:${createHash("sha256").update(text).digest("hex")}`
    };
}
function normalizeGraphifyOutputDir(outputDir) {
    return path.resolve(outputDir);
}
function validateActivationRoot(activationRoot) {
    if (!existsSync(activationRoot)) {
        throw new Error(`Activation root does not exist: ${activationRoot}`);
    }
    const pointersPath = path.join(activationRoot, "activation-pointers.json");
    if (!existsSync(pointersPath)) {
        throw new Error(`activation-pointers.json not found in ${activationRoot}. ` +
            `This doesn't look like a valid activation root.`);
    }
}
function archiveContainsPointers(archivePath) {
    try {
        const listing = execSync(`tar tzf ${JSON.stringify(archivePath)}`, {
            encoding: "utf8",
            maxBuffer: 10 * 1024 * 1024,
        });
        const entries = listing.split("\n").map((entry) => entry.replace(/^\.\//, ""));
        return entries.some((entry) => entry === "activation-pointers.json" || entry.endsWith("/activation-pointers.json"));
    }
    catch {
        return false;
    }
}
function activationRootHasData(activationRoot) {
    if (!existsSync(activationRoot)) {
        return false;
    }
    try {
        return readdirSync(activationRoot).length > 0;
    }
    catch {
        return false;
    }
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
function timestampToken(value = new Date().toISOString()) {
    return String(value).replace(/[:]/g, "-");
}
function sha256Text(text) {
    return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}
function stableJson(value) {
    return canonicalJson(value);
}
function writeTextFile(filePath, text) {
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, text, "utf8");
    return filePath;
}
function readTextIfExists(filePath) {
    return existsSync(filePath) ? readFileSync(filePath, "utf8") : null;
}
function normalizeStableRelativePath(rootPath, candidatePath) {
    const relativePath = path.relative(rootPath, candidatePath);
    if (relativePath.length === 0) {
        return "";
    }
    return relativePath.split(path.sep).join(path.posix.sep);
}
function hashStablePathTreeEntry(digest, entry) {
    digest.update(`${entry.kind}\u0000${entry.path}\u0000`);
    if (entry.kind === "file") {
        digest.update(`${entry.size}\u0000${entry.hash}\n`);
        return;
    }
    if (entry.kind === "symlink") {
        digest.update(`${entry.target}\n`);
        return;
    }
    digest.update("\n");
}
function walkStablePathTree(inputPath, rootPath, entries, digest, totals) {
    const dirents = readdirSync(inputPath, { withFileTypes: true })
        .slice()
        .sort((left, right) => left.name.localeCompare(right.name));
    for (const dirent of dirents) {
        const absolutePath = path.join(inputPath, dirent.name);
        const relativePath = normalizeStableRelativePath(rootPath, absolutePath);
        const fileStat = lstatSync(absolutePath);
        if (fileStat.isSymbolicLink()) {
            const target = readlinkSync(absolutePath);
            const entry = { kind: "symlink", path: relativePath, target };
            entries.push(entry);
            totals.symlinkCount += 1;
            hashStablePathTreeEntry(digest, entry);
            continue;
        }
        if (fileStat.isDirectory()) {
            const entry = { kind: "directory", path: relativePath };
            entries.push(entry);
            totals.directoryCount += 1;
            hashStablePathTreeEntry(digest, entry);
            walkStablePathTree(absolutePath, rootPath, entries, digest, totals);
            continue;
        }
        if (fileStat.isFile()) {
            const fileBuffer = readFileSync(absolutePath);
            const fileHash = createHash("sha256").update(fileBuffer).digest("hex");
            const entry = { kind: "file", path: relativePath, size: fileStat.size, hash: fileHash };
            entries.push(entry);
            totals.fileCount += 1;
            totals.totalBytes += fileStat.size;
            hashStablePathTreeEntry(digest, entry);
        }
    }
}
/**
 * Describe a file tree using stable relative paths and a deterministic SHA-256 hash.
 *
 * The hash only depends on relative paths, file contents, and symlink targets.
 */
export function describeStablePathTree(inputPath) {
    const resolvedPath = path.resolve(inputPath);
    if (!existsSync(resolvedPath)) {
        throw new Error(`Path does not exist: ${resolvedPath}`);
    }
    const digest = createHash("sha256");
    const entries = [];
    const totals = {
        fileCount: 0,
        directoryCount: 0,
        symlinkCount: 0,
        totalBytes: 0,
    };
    const stats = lstatSync(resolvedPath);
    if (stats.isSymbolicLink()) {
        const target = readlinkSync(resolvedPath);
        const entry = { kind: "symlink", path: path.basename(resolvedPath), target };
        entries.push(entry);
        totals.symlinkCount += 1;
        hashStablePathTreeEntry(digest, entry);
    }
    else if (stats.isFile()) {
        const fileBuffer = readFileSync(resolvedPath);
        const fileHash = createHash("sha256").update(fileBuffer).digest("hex");
        const entry = { kind: "file", path: path.basename(resolvedPath), size: stats.size, hash: fileHash };
        entries.push(entry);
        totals.fileCount += 1;
        totals.totalBytes += stats.size;
        hashStablePathTreeEntry(digest, entry);
    }
    else {
        const entry = { kind: "directory", path: "." };
        entries.push(entry);
        totals.directoryCount += 1;
        hashStablePathTreeEntry(digest, entry);
        walkStablePathTree(resolvedPath, resolvedPath, entries, digest, totals);
    }
    return {
        path: resolvedPath,
        kind: stats.isDirectory() ? "directory" : stats.isFile() ? "file" : "symlink",
        hash: digest.digest("hex"),
        entries,
        ...totals,
    };
}
function ensureDir(dirPath) {
    mkdirSync(dirPath, { recursive: true });
}
function tryMirrorTree(sourceRoot, destinationRoot) {
    rmSync(destinationRoot, { recursive: true, force: true });
    try {
        symlinkSync(sourceRoot, destinationRoot, "dir");
        return { path: destinationRoot, mode: "symlink" };
    }
    catch {
        cpSync(sourceRoot, destinationRoot, { recursive: true });
        return { path: destinationRoot, mode: "copy" };
    }
}
function buildProjectionMarkdown(options) {
    const extraDetails = Array.isArray(options.extraDetails) ? options.extraDetails.filter((line) => typeof line === "string" && line.trim().length > 0) : [];
    const body = [
        `# ${options.title}`,
        "",
        "Projection-only surface; non-authoritative by design.",
        "",
        `- kind: ${options.kind}`,
        `- bundle root: \`${path.resolve(options.bundleRoot)}\``,
        `- source bundle hash: \`${options.sourceBundleHash}\``,
        `- canonical archive: \`${path.relative(options.bundleRoot, options.canonicalArchivePath)}\``,
        `- generated at: \`${options.generatedAt}\``,
        `- source path: \`${options.sourcePath}\``,
        ...extraDetails.map((line) => `- ${line}`),
        "",
        "## Source projection",
        "",
        options.sourceText === null ? "_Source unavailable._" : options.sourceText.replace(/\n?$/u, ""),
        ""
    ];
    return body.join("\n");
}
function writeProjectionSurface(filePath, text) {
    return writeTextFile(filePath, text);
}
function resolveGraphifySourceBundleOpenClawHome(options = {}) {
    const explicitOpenClawHome = typeof options.openclawHome === "string" && options.openclawHome.trim().length > 0
        ? path.resolve(options.openclawHome)
        : null;
    if (explicitOpenClawHome !== null) {
        return explicitOpenClawHome;
    }
    const explicitProfileRoots = Array.isArray(options.profileRoots)
        ? options.profileRoots.filter((root) => typeof root === "string" && root.trim().length > 0).map((root) => path.resolve(root))
        : [];
    if (explicitProfileRoots.length === 1) {
        return explicitProfileRoots[0];
    }
    if (explicitProfileRoots.length > 1) {
        throw new Error("graphify source bundle export expects one OpenClaw home; pass --openclaw-home to disambiguate multiple profile roots");
    }
    const homeDir = path.resolve(options.homeDir ?? process.env.HOME ?? process.env.USERPROFILE ?? path.resolve("."));
    const discoveredHomes = discoverOpenClawHomes(homeDir);
    if (discoveredHomes.length === 1) {
        return discoveredHomes[0].openclawHome;
    }
    if (discoveredHomes.length === 0) {
        throw new Error(`No OpenClaw home found beneath ${homeDir}. Pass --openclaw-home <path> to export a canonical source bundle.`);
    }
    throw new Error(`Multiple OpenClaw homes were discovered beneath ${homeDir}. Pass --openclaw-home <path> to select one.`);
}
function resolveGraphifySourceBundleActivationRoot(openclawHome, explicitActivationRoot) {
    const normalizedExplicitActivationRoot = typeof explicitActivationRoot === "string" && explicitActivationRoot.trim().length > 0
        ? path.resolve(explicitActivationRoot)
        : null;
    if (normalizedExplicitActivationRoot !== null) {
        return normalizedExplicitActivationRoot;
    }
    const resolved = resolveActivationRoot({ openclawHome, quiet: true });
    if (typeof resolved === "string" && resolved.trim().length > 0) {
        return path.resolve(resolved);
    }
    return path.resolve(path.dirname(openclawHome), ".openclawbrain", "activation");
}
function summarizeSourceBundleFiles(fileDigests) {
    return Object.fromEntries(Object.entries(fileDigests).sort((left, right) => left[0].localeCompare(right[0])));
}
function readRuntimeLoadProofSnapshot(activationRoot) {
    const snapshot = listOpenClawProfileRuntimeLoadProofs(activationRoot);
    return {
        contract: "graphify_source_runtime_load_proofs.v1",
        runtimeOwner: "openclaw",
        activationRoot: path.resolve(activationRoot),
        path: snapshot.path,
        proofs: snapshot.proofs,
        error: snapshot.error
    };
}
function buildGraphifySourceBundleStatus(input) {
    const sourceBundleRoot = path.resolve(input.outputDir);
    const runtimeLoadProofPath = resolveAttachmentRuntimeLoadProofsPath(input.activationRoot);
    return {
        contract: "graphify_source_runtime_status.v1",
        runtimeOwner: "openclaw",
        bundleId: input.bundleId,
        corpusId: input.corpusId,
        corpusDigest: input.corpusDigest,
        createdAt: input.createdAt,
        observedAt: input.observedAt,
        openclawHome: input.openclawHome,
        activationRoot: input.activationRoot,
        sourceBundleRoot,
        openclawHomeInspection: input.openclawHomeInspection,
        hookStatus: input.hookStatus,
        runtimeLoadProof: input.runtimeLoadProof,
        runtimeLoadProofPath,
        sessionTail: {
            lane: input.sessionTail.lane,
            observedAt: input.sessionTail.observedAt,
            noopReason: input.sessionTail.poll.noopReason,
            warnings: [...input.sessionTail.poll.warnings],
            sourceCount: input.sessionTail.poll.sources.length,
            changeCount: input.sessionTail.poll.changes.length,
            emittedEventCount: input.sessionTail.interactionEvents.length + input.sessionTail.feedbackEvents.length,
            cursorCount: input.sessionTail.poll.cursor.length
        },
        normalizedEventExport: input.normalizedEventExport === null ? null : {
            exportDigest: input.normalizedEventExport.provenance.exportDigest,
            range: input.normalizedEventExport.range,
            interactionCount: input.normalizedEventExport.provenance.interactionCount,
            feedbackCount: input.normalizedEventExport.provenance.feedbackCount,
            sourceStreams: [...input.normalizedEventExport.provenance.sourceStreams],
            contracts: [...input.normalizedEventExport.provenance.contracts],
            semanticSurface: input.normalizedEventExport.provenance.semanticSurface ?? null
        },
        proofFiles: input.proofFiles,
        sourceSummaries: input.sourceSummaries,
        provenance: {
            authority: "canonical_machine_export",
            sourceAuthority: "session_tail",
            lane: "local_session_tail"
        }
    };
}
function buildGraphifyWorkspaceMetadata(input) {
    return {
        contract: "graphify_source_workspace_metadata.v1",
        runtimeOwner: "openclaw",
        bundleId: input.bundleId,
        corpusId: input.corpusId,
        corpusDigest: input.corpusDigest,
        createdAt: input.createdAt,
        openclawHome: input.openclawHome,
        activationRoot: input.activationRoot,
        sourceBundleRoot: path.resolve(input.outputDir),
        sourceBundleRunId: path.basename(path.resolve(input.outputDir)),
        profileInspection: input.openclawHomeInspection,
        hookStatus: input.hookStatus,
        sourceRoots: [...new Set(input.sourceSummaries.map((source) => source.profileRoot))].sort((left, right) => left.localeCompare(right)),
        sourceCount: input.sourceSummaries.length,
        sessionCount: input.sessionTail.poll.sources.length,
        proofPaths: {
            runtimeLoadProof: input.runtimeLoadProof.path,
            openclawHomeInspection: input.proofFiles["openclaw-home-inspection.json"].path,
            hookStatus: input.proofFiles["hook-status.json"].path,
            sessionTail: input.proofFiles["session-tail.json"].path,
            corpusSources: input.proofFiles["corpus-sources.json"].path
        },
        sourceSummaries: input.sourceSummaries.map((source) => ({
            sourceId: source.sourceId,
            profileRoot: source.profileRoot,
            agentId: source.agentId,
            sourceIndexPath: source.sourceIndexPath,
            sessionKey: source.sessionKey,
            sessionId: source.sessionId,
            sessionFile: source.sessionFile,
            sessionIndexDigest: source.sessionIndexDigest,
            sessionFileDigest: source.sessionFileDigest,
            sourceManifestDigest: source.sourceManifestDigest,
            changeKind: source.changeKind,
            eventCounts: source.eventCounts
        })),
        provenance: {
            authority: "canonical_machine_export",
            sourceAuthority: "session_tail",
            lane: "local_session_tail"
        }
    };
}
function buildCorpusManifest(input) {
    return {
        contract: "graphify_source_bundle_manifest.v1",
        runtimeOwner: "openclaw",
        bundleId: input.bundleId,
        corpusId: input.corpusId,
        corpusDigest: input.corpusDigest,
        createdAt: input.createdAt,
        observedAt: input.observedAt,
        openclawHome: input.openclawHome,
        activationRoot: input.activationRoot,
        sourceBundleRoot: path.resolve(input.outputDir),
        openclawHomeInspection: input.openclawHomeInspection,
        hookStatus: input.hookStatus,
        sourceAuthority: "session_tail",
        sessionTail: {
            lane: input.sessionTail.lane,
            observedAt: input.sessionTail.observedAt,
            noopReason: input.sessionTail.poll.noopReason,
            warnings: [...input.sessionTail.poll.warnings],
            sourceCount: input.sessionTail.poll.sources.length,
            changeCount: input.sessionTail.poll.changes.length,
            emittedEventCount: input.sessionTail.interactionEvents.length + input.sessionTail.feedbackEvents.length,
            cursorCount: input.sessionTail.poll.cursor.length
        },
        normalizedEventExport: input.normalizedEventExport === null ? null : {
            exportDigest: input.normalizedEventExport.provenance.exportDigest,
            range: input.normalizedEventExport.range,
            interactionCount: input.normalizedEventExport.provenance.interactionCount,
            feedbackCount: input.normalizedEventExport.provenance.feedbackCount,
            sourceStreams: [...input.normalizedEventExport.provenance.sourceStreams],
            contracts: [...input.normalizedEventExport.provenance.contracts],
            semanticSurface: input.normalizedEventExport.provenance.semanticSurface ?? null
        },
        sourceSummaries: input.sourceSummaries,
        fileDigests: input.fileDigests,
        proofFiles: summarizeSourceBundleFiles(input.fileDigests.proofFiles),
        provenance: {
            authority: "canonical_machine_export",
            sourceAuthority: "session_tail",
            lane: "local_session_tail",
            exportMode: "graphify_source_bundle"
        }
    };
}

/**
 * Export (backup) the activation root to a tar.gz archive.
 */
export function exportBrain(options) {
    const { activationRoot, outputPath } = options;
    const resolvedRoot = path.resolve(activationRoot);
    const resolvedOutput = path.resolve(outputPath);
    try {
        validateActivationRoot(resolvedRoot);
        const outputDir = path.dirname(resolvedOutput);
        if (!existsSync(outputDir)) {
            mkdirSync(outputDir, { recursive: true });
        }
        execSync(`tar czf ${JSON.stringify(resolvedOutput)} -C ${JSON.stringify(resolvedRoot)} .`, { stdio: "pipe" });
        if (!existsSync(resolvedOutput)) {
            return {
                ok: false,
                outputPath: resolvedOutput,
                activationRoot: resolvedRoot,
                error: "Archive was not created (tar returned success but file missing)",
            };
        }
        statSync(resolvedOutput);
        return {
            ok: true,
            outputPath: resolvedOutput,
            activationRoot: resolvedRoot,
        };
    }
    catch (err) {
        return {
            ok: false,
            outputPath: resolvedOutput,
            activationRoot: resolvedRoot,
            error: err instanceof Error ? err.message : String(err),
        };
    }
}

/**
 * Export a canonical Graphify source bundle from the current OpenClaw corpus.
 */
export function exportGraphifySourceBundle(options) {
    const outputDir = options.outputDir === undefined || options.outputDir === null || String(options.outputDir).trim().length === 0
        ? null
        : normalizeGraphifyOutputDir(options.outputDir);
    if (outputDir === null) {
        return {
            ok: false,
            error: "graphify source bundle export requires --output-dir <path>",
            bundleDir: null
        };
    }
    try {
        const openclawHome = resolveGraphifySourceBundleOpenClawHome(options);
        const activationRoot = resolveGraphifySourceBundleActivationRoot(openclawHome, options.activationRoot);
        const sessionTail = buildOpenClawSessionCorpusSnapshot({
            profileRoots: options.profileRoots ?? [openclawHome],
            ...(options.homeDir === undefined ? {} : { homeDir: options.homeDir }),
            ...(options.cursor === undefined ? {} : { cursor: options.cursor }),
            ...(options.observedAt === undefined ? {} : { observedAt: options.observedAt }),
            emitExistingOnFirstPoll: true
        });
        if (sessionTail.normalizedEventExport === null) {
            return {
                ok: false,
                bundleDir: outputDir,
                error: "graphify source bundle export found no bridgeable session events"
            };
        }
        const createdAt = typeof options.createdAt === "string" && options.createdAt.trim().length > 0
            ? options.createdAt
            : new Date().toISOString();
        const observedAt = sessionTail.observedAt;
        const bundleId = sessionTail.corpusId;
        const corpusId = sessionTail.corpusId;
        const openclawHomeInspection = inspectOpenClawHome(openclawHome);
        const hookStatus = inspectOpenClawBrainHookStatus(openclawHome);
        const runtimeLoadProof = readRuntimeLoadProofSnapshot(activationRoot);
        const proofDir = path.join(outputDir, "proof");
        mkdirSync(proofDir, { recursive: true });
        const proofFiles = {
            "openclaw-home-inspection.json": writeCanonicalJsonFile(path.join(proofDir, "openclaw-home-inspection.json"), {
                contract: "graphify_source_openclaw_home_inspection.v1",
                runtimeOwner: "openclaw",
                openclawHomeInspection,
                bundleId,
                corpusId,
                corpusDigest: sessionTail.corpusDigest,
                createdAt,
                observedAt
            }),
            "hook-status.json": writeCanonicalJsonFile(path.join(proofDir, "hook-status.json"), {
                contract: "graphify_source_hook_status.v1",
                runtimeOwner: "openclaw",
                hookStatus,
                bundleId,
                corpusId,
                corpusDigest: sessionTail.corpusDigest,
                createdAt,
                observedAt
            }),
            "runtime-load-proofs.json": writeCanonicalJsonFile(path.join(proofDir, "runtime-load-proofs.json"), runtimeLoadProof),
            "session-tail.json": writeCanonicalJsonFile(path.join(proofDir, "session-tail.json"), {
                contract: "graphify_source_session_tail_snapshot.v1",
                runtimeOwner: "openclaw",
                bundleId,
                corpusId,
                corpusDigest: sessionTail.corpusDigest,
                createdAt,
                observedAt,
                sessionTail: sessionTail.poll
            }),
            "corpus-sources.json": writeCanonicalJsonFile(path.join(proofDir, "corpus-sources.json"), {
                contract: "graphify_source_corpus_sources.v1",
                runtimeOwner: "openclaw",
                bundleId,
                corpusId,
                corpusDigest: sessionTail.corpusDigest,
                createdAt,
                observedAt,
                sourceSummaries: sessionTail.sourceSummaries
            })
        };
        const normalizedEventExportResult = writeCanonicalJsonFile(path.join(outputDir, "normalized-event-export.json"), sessionTail.normalizedEventExport);
        const runtimeStatus = buildGraphifySourceBundleStatus({
            bundleId,
            corpusId,
            corpusDigest: sessionTail.corpusDigest,
            createdAt,
            observedAt,
            openclawHome,
            activationRoot,
            outputDir,
            openclawHomeInspection,
            hookStatus,
            runtimeLoadProof,
            sessionTail,
            normalizedEventExport: sessionTail.normalizedEventExport,
            sourceSummaries: sessionTail.sourceSummaries,
            proofFiles
        });
        const runtimeStatusResult = writeCanonicalJsonFile(path.join(outputDir, "runtime-status.json"), runtimeStatus);
        const workspaceMetadata = buildGraphifyWorkspaceMetadata({
            bundleId,
            corpusId,
            corpusDigest: sessionTail.corpusDigest,
            createdAt,
            openclawHome,
            activationRoot,
            outputDir,
            openclawHomeInspection,
            hookStatus,
            sessionTail,
            sourceSummaries: sessionTail.sourceSummaries,
            runtimeLoadProof,
            proofFiles
        });
        const workspaceMetadataResult = writeCanonicalJsonFile(path.join(outputDir, "workspace-metadata.json"), workspaceMetadata);
        const fileDigests = {
            normalizedEventExport: normalizedEventExportResult.digest,
            runtimeStatus: runtimeStatusResult.digest,
            workspaceMetadata: workspaceMetadataResult.digest,
            proofFiles: summarizeSourceBundleFiles(Object.fromEntries(Object.entries(proofFiles).map(([name, result]) => [name, result.digest])))
        };
        const corpusManifest = buildCorpusManifest({
            bundleId,
            corpusId,
            corpusDigest: sessionTail.corpusDigest,
            createdAt,
            observedAt,
            openclawHome,
            activationRoot,
            outputDir,
            openclawHomeInspection,
            hookStatus,
            sessionTail,
            normalizedEventExport: sessionTail.normalizedEventExport,
            sourceSummaries: sessionTail.sourceSummaries,
            fileDigests
        });
        const corpusManifestResult = writeCanonicalJsonFile(path.join(outputDir, "corpus-manifest.json"), corpusManifest);
        return {
            ok: true,
            bundleDir: outputDir,
            bundleId,
            corpusId,
            corpusDigest: sessionTail.corpusDigest,
            outputPaths: {
                corpusManifest: corpusManifestResult.path,
                normalizedEventExport: normalizedEventExportResult.path,
                runtimeStatus: runtimeStatusResult.path,
                workspaceMetadata: workspaceMetadataResult.path,
                proofDir,
                proofFiles: Object.fromEntries(Object.entries(proofFiles).map(([name, result]) => [name, result.path]))
            },
            corpusManifest,
            normalizedEventExport: sessionTail.normalizedEventExport,
            runtimeStatus,
            workspaceMetadata,
            sourceSummaries: sessionTail.sourceSummaries,
        };
    }
    catch (err) {
        return {
            ok: false,
            bundleDir: outputDir,
            error: err instanceof Error ? err.message : String(err)
        };
    }
}

/**
 * Project the canonical machine export into Graphify-friendly markdown and
 * filesystem surfaces while keeping the projection explicitly non-authoritative.
 */
export function exportGraphifyProjection(options) {
    const resolvedActivationRoot = path.resolve(options.activationRoot);
    const resolvedRepoRoot = path.resolve(options.repoRoot ?? process.cwd());
    const resolvedWorkspaceRoot = path.resolve(options.workspaceRoot ?? resolvedRepoRoot);
    const resolvedOutputRoot = path.resolve(options.outputRoot ?? path.join(process.cwd(), "artifacts", "graphify-source-bundles"));
    const runId = normalizeOptionalString(options.runId) ?? timestampToken(new Date().toISOString());
    const bundleRoot = path.join(resolvedOutputRoot, runId);
    const generatedAt = normalizeOptionalString(options.generatedAt) ?? new Date().toISOString();
    const sessionTimestamp = normalizeOptionalString(options.sessionTimestamp) ?? generatedAt;
    const sessionKey = normalizeOptionalString(options.sessionKey) ?? "current-session";
    const resolvedSessionSourcePath = normalizeOptionalString(options.sessionSourcePath);
    const resolvedProofSummarySourcePath = normalizeOptionalString(options.proofSummarySourcePath);
    const resolvedDocsRoot = path.resolve(options.docsRoot ?? path.join(resolvedRepoRoot, "docs"));
    const resolvedCodeRoot = path.resolve(options.codeRoot ?? path.join(resolvedRepoRoot, "packages", "cli", "dist", "src"));
    rmSync(bundleRoot, { recursive: true, force: true });
    ensureDir(path.join(bundleRoot, "canonical"));
    ensureDir(path.join(bundleRoot, "workspace"));
    ensureDir(path.join(bundleRoot, "proof"));
    ensureDir(path.join(bundleRoot, "sessions", sessionKey));
    const canonicalArchivePath = path.join(bundleRoot, "canonical", "machine-export.tar.gz");
    const exportResult = exportBrain({
        activationRoot: resolvedActivationRoot,
        outputPath: canonicalArchivePath,
    });
    if (!exportResult.ok) {
        return {
            ok: false,
            runId,
            bundleRoot,
            sourceBundleHash: null,
            canonicalArchivePath,
            canonicalArchiveSha256: null,
            manifestPath: null,
            sessionProjectionPath: null,
            workspaceMemoryPath: null,
            workspaceTasksPath: null,
            proofSummaryPath: null,
            docsMirrorRoot: null,
            codeMirrorRoot: null,
            error: exportResult.error ?? "canonical machine export failed",
            warnings: [],
        };
    }
    const canonicalArchiveText = readFileSync(canonicalArchivePath);
    const canonicalArchiveSha256 = createHash("sha256").update(canonicalArchiveText).digest("hex");
    const canonicalArchiveResultPath = path.join(bundleRoot, "canonical", "machine-export.json");
    writeTextFile(canonicalArchiveResultPath, JSON.stringify({
        ok: exportResult.ok,
        activationRoot: exportResult.activationRoot,
        outputPath: exportResult.outputPath,
        archiveSha256: canonicalArchiveSha256,
        archiveBytes: statSync(canonicalArchivePath).size,
    }, null, 2));
    const memorySourcePath = path.join(resolvedWorkspaceRoot, "MEMORY.md");
    const tasksSourcePath = path.join(resolvedWorkspaceRoot, "TASKS.md");
    const memoryText = readTextIfExists(memorySourcePath);
    const tasksText = readTextIfExists(tasksSourcePath);
    const sessionSourceText = resolvedSessionSourcePath === null ? null : readTextIfExists(resolvedSessionSourcePath);
    const proofSummarySourceText = resolvedProofSummarySourcePath === null ? null : readTextIfExists(resolvedProofSummarySourcePath);
    const docsMirror = existsSync(resolvedDocsRoot) ? tryMirrorTree(resolvedDocsRoot, path.join(bundleRoot, "docs")) : { path: path.join(bundleRoot, "docs"), mode: "copy" };
    const codeMirror = existsSync(resolvedCodeRoot) ? tryMirrorTree(resolvedCodeRoot, path.join(bundleRoot, "code")) : { path: path.join(bundleRoot, "code"), mode: "copy" };
    if (!existsSync(docsMirror.path)) {
        ensureDir(docsMirror.path);
    }
    if (!existsSync(codeMirror.path)) {
        ensureDir(codeMirror.path);
    }
    const sourceBundleHash = createHash("sha256").update(stableJson({
        contract: "graphify_source_bundle.v1",
        runId,
        activationRoot: resolvedActivationRoot,
        repoRoot: resolvedRepoRoot,
        workspaceRoot: resolvedWorkspaceRoot,
        generatedAt,
        sessionKey,
        sessionTimestamp,
        sessionSourcePath: resolvedSessionSourcePath,
        proofSummarySourcePath: resolvedProofSummarySourcePath,
        canonicalArchiveSha256,
        memorySha256: memoryText === null ? null : sha256Text(memoryText),
        tasksSha256: tasksText === null ? null : sha256Text(tasksText),
        sessionSourceSha256: sessionSourceText === null ? null : sha256Text(sessionSourceText),
        proofSummarySourceSha256: proofSummarySourceText === null ? null : sha256Text(proofSummarySourceText),
        docsMirrorMode: docsMirror.mode,
        codeMirrorMode: codeMirror.mode,
    })).digest("hex");
    const sessionProjectionPath = path.join(bundleRoot, "sessions", sessionKey, `${timestampToken(sessionTimestamp)}.md`);
    const workspaceMemoryPath = path.join(bundleRoot, "workspace", "MEMORY.md");
    const workspaceTasksPath = path.join(bundleRoot, "workspace", "TASKS.md");
    const proofSummaryPath = path.join(bundleRoot, "proof", "summary.md");
    const manifestPath = path.join(bundleRoot, "corpus-manifest.json");
    const readmePath = path.join(bundleRoot, "README.md");
    const sessionMarkdown = buildProjectionMarkdown({
        title: "Graphify session projection",
        kind: "session_projection",
        bundleRoot,
        sourceBundleHash,
        canonicalArchivePath,
        sourcePath: resolvedSessionSourcePath ?? path.join(resolvedWorkspaceRoot, "agents", "main", "sessions"),
        generatedAt,
        sourceText: sessionSourceText,
        extraDetails: [
            `session key: \`${sessionKey}\``,
            `session timestamp: \`${sessionTimestamp}\``,
            `source bundle linkage: \`${path.relative(bundleRoot, canonicalArchivePath)}\``,
        ],
    });
    const memoryMarkdown = buildProjectionMarkdown({
        title: "Graphify workspace MEMORY projection",
        kind: "workspace_memory_projection",
        bundleRoot,
        sourceBundleHash,
        canonicalArchivePath,
        sourcePath: memorySourcePath,
        generatedAt,
        sourceText: memoryText,
        extraDetails: [
            `workspace root: \`${resolvedWorkspaceRoot}\``,
            `source bundle linkage: \`${path.relative(bundleRoot, canonicalArchivePath)}\``,
        ],
    });
    const tasksMarkdown = buildProjectionMarkdown({
        title: "Graphify workspace TASKS projection",
        kind: "workspace_tasks_projection",
        bundleRoot,
        sourceBundleHash,
        canonicalArchivePath,
        sourcePath: tasksSourcePath,
        generatedAt,
        sourceText: tasksText,
        extraDetails: [
            `workspace root: \`${resolvedWorkspaceRoot}\``,
            `source bundle linkage: \`${path.relative(bundleRoot, canonicalArchivePath)}\``,
        ],
    });
    const proofMarkdown = buildProjectionMarkdown({
        title: "Graphify proof summary projection",
        kind: "proof_summary_projection",
        bundleRoot,
        sourceBundleHash,
        canonicalArchivePath,
        sourcePath: resolvedProofSummarySourcePath ?? path.join(bundleRoot, "proof", "summary.source.md"),
        generatedAt,
        sourceText: proofSummarySourceText,
        extraDetails: [
            `source bundle linkage: \`${path.relative(bundleRoot, canonicalArchivePath)}\``,
            `workspace root: \`${resolvedWorkspaceRoot}\``,
        ],
    });
    writeTextFile(readmePath, [
        "# Graphify projection bundle",
        "",
        "Projection-only surface; non-authoritative by design.",
        "",
        `- bundle root: \`${bundleRoot}\``,
        `- run id: \`${runId}\``,
        `- source bundle hash: \`${sourceBundleHash}\``,
        `- canonical archive: \`${path.relative(bundleRoot, canonicalArchivePath)}\``,
        "",
        "This bundle mirrors canonical machine export data into Graphify-friendly surfaces for review only.",
        ""
    ].join("\n"));
    writeProjectionSurface(sessionProjectionPath, sessionMarkdown);
    writeProjectionSurface(workspaceMemoryPath, memoryMarkdown);
    writeProjectionSurface(workspaceTasksPath, tasksMarkdown);
    writeProjectionSurface(proofSummaryPath, proofMarkdown);
    const manifest = {
        contract: "graphify_source_bundle.v1",
        sourceKind: "canonical_ocb_source_bundle",
        generatedAt,
        runId,
        authoritative: false,
        projectionTruth: "projection_only",
        sourceBundleHash,
        canonicalMachineExport: {
            path: path.relative(bundleRoot, canonicalArchivePath),
            sha256: canonicalArchiveSha256,
            bytes: statSync(canonicalArchivePath).size,
            activationRoot: resolvedActivationRoot,
        },
        provenance: {
            repoRoot: resolvedRepoRoot,
            workspaceRoot: resolvedWorkspaceRoot,
            sessionKey,
            sessionTimestamp,
            sessionSourcePath: resolvedSessionSourcePath,
            proofSummarySourcePath: resolvedProofSummarySourcePath,
            docsMirrorRoot: resolvedDocsRoot,
            codeMirrorRoot: resolvedCodeRoot,
        },
        outputs: {
            bundleRoot: ".",
            canonicalArchive: path.relative(bundleRoot, canonicalArchivePath),
            canonicalArchiveResult: path.relative(bundleRoot, canonicalArchiveResultPath),
            corpusManifest: path.relative(bundleRoot, manifestPath),
            readme: path.relative(bundleRoot, readmePath),
            sessionProjection: path.relative(bundleRoot, sessionProjectionPath),
            workspaceMemory: path.relative(bundleRoot, workspaceMemoryPath),
            workspaceTasks: path.relative(bundleRoot, workspaceTasksPath),
            proofSummary: path.relative(bundleRoot, proofSummaryPath),
            docsMirror: path.relative(bundleRoot, path.join(bundleRoot, "docs")),
            codeMirror: path.relative(bundleRoot, path.join(bundleRoot, "code")),
        },
        mirrorModes: {
            docs: docsMirror.mode,
            code: codeMirror.mode,
        },
        sourceHashes: {
            canonicalArchiveSha256,
            memorySha256: memoryText === null ? null : sha256Text(memoryText),
            tasksSha256: tasksText === null ? null : sha256Text(tasksText),
            sessionSourceSha256: sessionSourceText === null ? null : sha256Text(sessionSourceText),
            proofSummarySourceSha256: proofSummarySourceText === null ? null : sha256Text(proofSummarySourceText),
        },
        notes: [
            "This bundle is projection-only and non-authoritative.",
            "Canonical machine export linkage remains explicit.",
            "Docs/code mirrors are curated Graphify-friendly surfaces only.",
        ],
    };
    writeTextFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    return {
        ok: true,
        runId,
        bundleRoot,
        sourceBundleHash,
        canonicalArchivePath,
        canonicalArchiveSha256,
        manifestPath,
        sessionProjectionPath,
        workspaceMemoryPath,
        workspaceTasksPath,
        proofSummaryPath,
        docsMirrorRoot: docsMirror.path,
        codeMirrorRoot: codeMirror.path,
        warnings: docsMirror.mode === "copy" || codeMirror.mode === "copy"
            ? [docsMirror.mode === "copy" ? "docs mirror was copied because symlink creation failed" : null, codeMirror.mode === "copy" ? "code mirror was copied because symlink creation failed" : null].filter((warning) => warning !== null)
            : [],
        outputPaths: {
            canonicalArchive: canonicalArchivePath,
            canonicalArchiveResult: canonicalArchiveResultPath,
            corpusManifest: manifestPath,
            readme: readmePath,
            sessionProjection: sessionProjectionPath,
            workspaceMemory: workspaceMemoryPath,
            workspaceTasks: workspaceTasksPath,
            proofSummary: proofSummaryPath,
            docsMirror: docsMirror.path,
            codeMirror: codeMirror.path,
        },
    };
}

/**
 * Import (restore) a tar.gz archive into the activation root.
 */
export function importBrain(options) {
    const { archivePath, activationRoot, force } = options;
    const resolvedArchive = path.resolve(archivePath);
    const resolvedRoot = path.resolve(activationRoot);
    try {
        if (!existsSync(resolvedArchive)) {
            return {
                ok: false,
                activationRoot: resolvedRoot,
                archivePath: resolvedArchive,
                error: `Archive not found: ${resolvedArchive}`,
            };
        }
        if (!archiveContainsPointers(resolvedArchive)) {
            return {
                ok: false,
                activationRoot: resolvedRoot,
                archivePath: resolvedArchive,
                error: "Archive does not contain activation-pointers.json. " +
                    "This doesn't look like a valid brain backup.",
            };
        }
        let warning;
        if (activationRootHasData(resolvedRoot)) {
            if (!force) {
                return {
                    ok: false,
                    activationRoot: resolvedRoot,
                    archivePath: resolvedArchive,
                    error: `Activation root ${resolvedRoot} already contains data. ` +
                        `Use --force to overwrite.`,
                };
            }
            warning = `Overwrote existing data in ${resolvedRoot}`;
        }
        if (!existsSync(resolvedRoot)) {
            mkdirSync(resolvedRoot, { recursive: true });
        }
        execSync(`tar xzf ${JSON.stringify(resolvedArchive)} -C ${JSON.stringify(resolvedRoot)}`, { stdio: "pipe" });
        const pointersPath = path.join(resolvedRoot, "activation-pointers.json");
        if (!existsSync(pointersPath)) {
            return {
                ok: false,
                activationRoot: resolvedRoot,
                archivePath: resolvedArchive,
                error: "Extraction completed but activation-pointers.json not found. " +
                    "The archive may have a nested directory structure.",
            };
        }
        const result = {
            ok: true,
            activationRoot: resolvedRoot,
            archivePath: resolvedArchive,
        };
        if (warning !== undefined) {
            result.warning = warning;
        }
        return result;
    }
    catch (err) {
        return {
            ok: false,
            activationRoot: resolvedRoot,
            archivePath: resolvedArchive,
            error: err instanceof Error ? err.message : String(err),
        };
    }
}

/**
 * Build and write a Graphify-derived compiled artifact pack.
 */
export function exportGraphifyCompiledArtifactsPack(options = {}) {
    try {
        const bundle = buildGraphifyCompiledArtifactPack(options);
        const writeResult = writeGraphifyCompiledArtifactPack(bundle.outputDir, bundle);
        return {
            ok: true,
            bundleId: bundle.bundleId,
            packId: bundle.packId,
            proposalId: bundle.proposalId,
            outputDir: bundle.outputDir,
            manifestPath: bundle.bundlePaths.manifest,
            compilerProposalPath: bundle.bundlePaths.compilerProposal,
            surfaceMapPath: bundle.bundlePaths.surfaceMap,
            proposalReportPath: bundle.bundlePaths.proposalReport,
            verdictPath: bundle.bundlePaths.verdict,
            artifactCount: bundle.artifactEntries.length,
            validation: bundle.validation,
            digest: bundle.digest,
            writtenFiles: writeResult.writtenFiles,
            fileCount: writeResult.fileCount,
        };
    }
    catch (error) {
        return {
            ok: false,
            outputDir: path.resolve(options.outputDir ?? "."),
            candidatePackInput: null,
            error: error instanceof Error ? error.message : String(error),
        };
    }
}

/**
 * Build and write a conservative EXTRACTED-only Graphify import slice.
 */
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
        return {
            ok: false,
            outputRoot: path.resolve(options.outputRoot ?? path.join(process.cwd(), "artifacts", "graphify-imports")),
            outputDir: resolveGraphifyImportSliceOutputDir(options),
            bundleRoot: path.resolve(options.bundleRoot ?? options.bundleDir ?? options.bundlePath ?? "."),
            error: error instanceof Error ? error.message : String(error),
        };
    }
}

/**
 * Build and write a Graphify maintenance diff bundle.
 */
export function exportGraphifyMaintenanceDiff(options = {}) {
    try {
        const bundle = buildGraphifyMaintenanceDiffBundle(options);
        const writeResult = writeGraphifyMaintenanceDiffBundle(bundle.outputDir, bundle);
        return {
            ok: true,
            runId: bundle.runId,
            diffId: bundle.diffId,
            proposalId: bundle.proposalId,
            rollbackKey: bundle.rollbackKey,
            repoRoot: bundle.repoRoot,
            workspaceRoot: bundle.workspaceRoot,
            graphifyRoot: bundle.graphifyRoot,
            ocbRoot: bundle.ocbRoot,
            outputRoot: bundle.outputRoot,
            outputDir: bundle.outputDir,
            report: bundle.report,
            proposalSuggestion: bundle.proposalSuggestion,
            verdict: bundle.verdict,
            paths: bundle.paths,
            digest: bundle.digest,
            writtenFiles: writeResult.writtenFiles,
            fileCount: writeResult.fileCount,
        };
    }
    catch (error) {
        return {
            ok: false,
            outputRoot: path.resolve(options.outputRoot ?? path.join(process.cwd(), "artifacts", "graphify-maintenance-diff")),
            outputDir: path.join(path.resolve(options.outputRoot ?? path.join(process.cwd(), "artifacts", "graphify-maintenance-diff")), options.runId ?? `graphify-maintenance-diff-${Date.now()}`),
            error: error instanceof Error ? error.message : String(error),
        };
    }
}
