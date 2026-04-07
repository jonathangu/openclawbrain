/**
 * Brain import/export: backup and restore the activation root directory.
 *
 * export: tar + gzip the entire activation root → output.tar.gz
 * import: extract tar.gz → activation root, with safety checks
 */
import { createHash } from "node:crypto";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { canonicalJson } from "@openclawbrain/contracts";
import { buildOpenClawSessionCorpusSnapshot } from "./session-tail.js";
import { discoverOpenClawHomes, inspectOpenClawHome } from "./openclaw-home-layout.js";
import { resolveActivationRoot } from "./resolve-activation-root.js";
import { inspectOpenClawBrainHookStatus } from "./openclaw-hook-truth.js";
import { listOpenClawProfileRuntimeLoadProofs, resolveAttachmentRuntimeLoadProofsPath } from "./attachment-truth.js";

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
