#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  describeStablePathTree,
  exportGraphifyCompiledArtifactsPack,
  exportGraphifyProjection,
  exportGraphifySourceBundle,
} from "./import-export.js";
import { runManagedGraphifyRunner } from "./graphify-runner.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../../../..");
const workspaceRoot = path.resolve(repoRoot, "..");

export const GRAPHIFY_BRIDGE_CUT_CONTRACT_V1 = "graphify_bridge_cut.v1";
export const GRAPHIFY_BRIDGE_DUAL_SOURCE_BUNDLE_CONTRACT_V1 = "graphify_dual_source_bundle_manifest.v1";
export const GRAPHIFY_BRIDGE_VERSION_V1 = "graphify-bridge-cut@1";

export const GRAPHIFY_BRIDGE_CUT_LAYOUT_V1 = {
  sourceBundleRoot: "source-bundle",
  canonicalSourceBundle: "source-bundle/canonical",
  projectionSourceBundle: "source-bundle/projection",
  sourceBundleManifest: "source-bundle/corpus-manifest.json",
  graphifyRun: "graphify-run",
  compiledArtifactPack: "compiled-artifact-pack",
  status: "bridge-status.json",
  summary: "summary.md",
};

function canonicalizeJsonValue(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeJsonValue(entry));
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  const result = {};
  for (const key of Object.keys(value).sort((left, right) => left.localeCompare(right))) {
    const nextValue = canonicalizeJsonValue(value[key]);
    if (nextValue !== undefined) {
      result[key] = nextValue;
    }
  }
  return result;
}

function stableJsonStringify(value) {
  return `${JSON.stringify(canonicalizeJsonValue(value), null, 2)}\n`;
}

function sha256Text(value) {
  return `sha256:${createHash("sha256").update(String(value ?? ""), "utf8").digest("hex")}`;
}

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, stableJsonStringify(value), "utf8");
  return filePath;
}

function writeText(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${String(value).replace(/\n?$/u, "")}\n`, "utf8");
  return filePath;
}

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function timestampToken(value = new Date().toISOString()) {
  return String(value).replace(/[:]/g, "-");
}

function relativePathFromRoot(absPath, rootPath) {
  const resolved = path.resolve(absPath);
  const relative = path.relative(path.resolve(rootPath), resolved);
  return relative.startsWith("..") ? resolved : relative.split(path.sep).join(path.posix.sep);
}

function withSha256Prefix(hash) {
  const normalized = normalizeText(hash);
  if (normalized === null) {
    return null;
  }
  return normalized.startsWith("sha256:") ? normalized : `sha256:${normalized}`;
}

function readJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function buildDualSourceBundleManifest(input) {
  const canonicalHash = input.canonicalSourceBundle.corpusDigest ?? null;
  const projectionHash = withSha256Prefix(input.projectionSourceBundle.sourceBundleHash ?? null);
  const dualSourceDigest = sha256Text(stableJsonStringify({
    canonicalHash,
    projectionHash,
    canonicalBundleId: input.canonicalSourceBundle.bundleId ?? null,
    projectionRunId: input.projectionSourceBundle.runId ?? null,
    bridgeVersion: GRAPHIFY_BRIDGE_VERSION_V1,
    runId: input.runId,
  }));
  return {
    contract: GRAPHIFY_BRIDGE_DUAL_SOURCE_BUNDLE_CONTRACT_V1,
    bridgeVersion: GRAPHIFY_BRIDGE_VERSION_V1,
    runId: input.runId,
    bundleId: `graphify-dual-source:${input.runId}`,
    createdAt: input.generatedAt,
    sourceKind: "dual_ocb_graphify_source_bundle",
    sourceAuthority: "canonical_machine_export",
    authoritativeLane: "canonical",
    projectionTruth: "projection_only",
    dualSourceDigest,
    labels: [
      "graphify",
      "graphify-bridge",
      "dual-source-bundle",
      "canonical-machine-export",
      "projection-export",
      "off-path",
      "provenance-first",
    ],
    exports: {
      canonical: {
        role: "authoritative_machine_export",
        path: relativePathFromRoot(input.canonicalSourceBundle.bundleDir, input.sourceBundleRoot),
        bundleId: input.canonicalSourceBundle.bundleId ?? null,
        corpusId: input.canonicalSourceBundle.corpusId ?? null,
        corpusDigest: canonicalHash,
        manifestPath: relativePathFromRoot(input.canonicalSourceBundle.outputPaths?.corpusManifest ?? path.join(input.canonicalSourceBundle.bundleDir, "corpus-manifest.json"), input.sourceBundleRoot),
        normalizedEventExportPath: relativePathFromRoot(input.canonicalSourceBundle.outputPaths?.normalizedEventExport ?? path.join(input.canonicalSourceBundle.bundleDir, "normalized-event-export.json"), input.sourceBundleRoot),
      },
      projection: {
        role: "graphify_facing_projection",
        path: relativePathFromRoot(input.projectionSourceBundle.bundleRoot, input.sourceBundleRoot),
        runId: input.projectionSourceBundle.runId ?? null,
        sourceBundleHash: projectionHash,
        manifestPath: relativePathFromRoot(input.projectionSourceBundle.manifestPath ?? path.join(input.projectionSourceBundle.bundleRoot, "corpus-manifest.json"), input.sourceBundleRoot),
        canonicalArchivePath: relativePathFromRoot(input.projectionSourceBundle.canonicalArchivePath, input.sourceBundleRoot),
      },
    },
    truthBoundary: {
      offPath: true,
      beforePromptBuildEligible: false,
      liveRuntimeEligible: false,
      correctionPrecedence: "explicit_corrections_and_raw_sources_win",
      derivedOnly: true,
      promotion: "requires_existing_ocb_replay_review_and_promotion_gates",
    },
    provenance: {
      repoRoot: input.repoRoot,
      workspaceRoot: input.workspaceRoot,
      openclawHome: input.openclawHome,
      activationRoot: input.activationRoot,
      outputRoot: input.outputRoot,
    },
    notes: [
      "The canonical export is authoritative inside this source bundle.",
      "The projection export is disposable and rebuildable for Graphify inspection.",
      "This bundle is an artifact input only; it is never read by before_prompt_build or the live prompt path.",
    ],
  };
}

function buildBridgeStatus(input) {
  return {
    contract: GRAPHIFY_BRIDGE_CUT_CONTRACT_V1,
    bridgeVersion: GRAPHIFY_BRIDGE_VERSION_V1,
    runId: input.runId,
    generatedAt: input.generatedAt,
    status: "completed",
    offPath: true,
    liveRuntimeTouched: false,
    beforePromptBuildTouched: false,
    truthBoundary: {
      sourceBundleAuthority: "canonical_export_wins_over_projection",
      graphifyAuthority: "derived_artifact_only",
      compiledPackAuthority: "proposal_backed_review_surface_only",
      correctionPrecedence: "explicit_corrections_win",
      importOrPromotionPerformed: false,
    },
    sourceBundle: {
      root: input.sourceBundleRoot,
      manifestPath: input.sourceBundleManifestPath,
      dualSourceDigest: input.sourceBundleManifest.dualSourceDigest,
      treeHash: withSha256Prefix(input.sourceBundleTree.hash),
      tree: {
        fileCount: input.sourceBundleTree.fileCount,
        directoryCount: input.sourceBundleTree.directoryCount,
        symlinkCount: input.sourceBundleTree.symlinkCount,
        totalBytes: input.sourceBundleTree.totalBytes,
      },
      canonical: input.sourceBundleManifest.exports.canonical,
      projection: input.sourceBundleManifest.exports.projection,
    },
    graphifyRun: {
      runId: input.graphifyRun.runId,
      runDir: input.graphifyRun.runDir,
      sourceBundleHash: withSha256Prefix(input.graphifyRun.sourceBundleHash),
      graphHash: withSha256Prefix(input.graphifyRun.graph.hash),
      graphifyVersion: input.graphifyRun.graphifyVersion,
      graphifyMode: input.graphifyRun.graphifyMode,
      execution: input.graphifyRun.execution,
      outputs: input.graphifyRun.outputs,
    },
    compiledArtifactPack: {
      bundleId: input.compiledPack.bundleId,
      packId: input.compiledPack.packId,
      proposalId: input.compiledPack.proposalId,
      outputDir: input.compiledPack.outputDir,
      manifestPath: input.compiledPack.manifestPath,
      surfaceMapPath: input.compiledPack.surfaceMapPath,
      proposalReportPath: input.compiledPack.proposalReportPath,
      verdictPath: input.compiledPack.verdictPath,
      artifactCount: input.compiledPack.artifactCount,
      validation: input.compiledPack.validation,
      digest: input.compiledPack.digest,
    },
  };
}

function renderBridgeSummary(status) {
  const artifactKinds = readJsonIfExists(status.compiledArtifactPack.manifestPath)?.artifacts?.map((artifact) => artifact.kind) ?? [];
  return [
    `# Graphify bridge cut ${status.runId}`,
    "",
    "First safe bridge cut: dual source-bundle export, managed Graphify run, and OCB-shaped compiled artifact pack.",
    "",
    "## Boundary",
    "",
    `- off live runtime path: ${status.offPath ? "yes" : "no"}`,
    `- before_prompt_build touched: ${status.beforePromptBuildTouched ? "yes" : "no"}`,
    `- live runtime touched: ${status.liveRuntimeTouched ? "yes" : "no"}`,
    `- import / promotion performed: ${status.truthBoundary.importOrPromotionPerformed ? "yes" : "no"}`,
    "- authority: canonical source export > projection export > Graphify derived artifacts",
    "",
    "## Outputs",
    "",
    `- dual source bundle: \`${status.sourceBundle.root}\``,
    `- dual source manifest: \`${status.sourceBundle.manifestPath}\``,
    `- managed Graphify run: \`${status.graphifyRun.runDir}\``,
    `- compiled artifact pack: \`${status.compiledArtifactPack.outputDir}\``,
    `- surface map: \`${status.compiledArtifactPack.surfaceMapPath}\``,
    `- proposal report: \`${status.compiledArtifactPack.proposalReportPath}\``,
    `- verdict: \`${status.compiledArtifactPack.verdictPath}\``,
    "",
    "## Compiled artifact kinds",
    "",
    ...(artifactKinds.length === 0 ? ["- unavailable"] : artifactKinds.map((kind) => `- ${kind}`)),
    "",
    "## Provenance",
    "",
    `- dual source digest: \`${status.sourceBundle.dualSourceDigest}\``,
    `- source bundle tree hash: \`${status.sourceBundle.treeHash}\``,
    `- Graphify graph hash: \`${status.graphifyRun.graphHash}\``,
    `- compiled pack hash: \`${status.compiledArtifactPack.digest?.bundleHash ?? "unavailable"}\``,
    "",
    "This bridge cut writes artifacts only. It does not add Graphify to prompt assembly, traversal, correction memory, or promotion.",
  ].join("\n");
}

export function runGraphifyBridgeCut(options = {}) {
  const generatedAt = normalizeText(options.generatedAt ?? options.createdAt) ?? new Date().toISOString();
  const runId = normalizeText(options.runId) ?? `graphify-bridge-${timestampToken(generatedAt)}`;
  const resolvedRepoRoot = path.resolve(options.repoRoot ?? repoRoot);
  const resolvedWorkspaceRoot = path.resolve(options.workspaceRoot ?? workspaceRoot);
  const outputRoot = path.resolve(options.outputRoot ?? path.join(resolvedRepoRoot, "artifacts", "graphify-bridge"));
  const runRoot = path.join(outputRoot, runId);
  const sourceBundleRoot = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.sourceBundleRoot);
  const canonicalSourceBundleDir = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.canonicalSourceBundle);
  const projectionOutputRoot = sourceBundleRoot;
  const projectionRunId = "projection";
  const graphifyRunRoot = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.graphifyRun);
  const compiledPackDir = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.compiledArtifactPack);
  const statusPath = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.status);
  const summaryPath = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.summary);
  const sourceBundleManifestPath = path.join(runRoot, GRAPHIFY_BRIDGE_CUT_LAYOUT_V1.sourceBundleManifest);
  const openclawHome = path.resolve(options.openclawHome ?? path.join(os.homedir(), ".openclaw"));
  const activationRoot = path.resolve(options.activationRoot ?? path.join(path.dirname(openclawHome), ".openclawbrain", "activation"));

  if (options.clean !== false) {
    rmSync(runRoot, { recursive: true, force: true });
  }
  ensureDir(sourceBundleRoot);

  const canonicalSourceBundle = exportGraphifySourceBundle({
    openclawHome,
    activationRoot,
    outputDir: canonicalSourceBundleDir,
    createdAt: generatedAt,
    observedAt: generatedAt,
    ...(options.profileRoots === undefined ? {} : { profileRoots: options.profileRoots }),
    ...(options.homeDir === undefined ? {} : { homeDir: options.homeDir }),
    ...(options.cursor === undefined ? {} : { cursor: options.cursor }),
  });
  if (!canonicalSourceBundle.ok) {
    throw new Error(`Graphify bridge canonical source export failed: ${canonicalSourceBundle.error ?? "unknown error"}`);
  }

  const projectionSourceBundle = exportGraphifyProjection({
    activationRoot,
    outputRoot: projectionOutputRoot,
    runId: projectionRunId,
    repoRoot: resolvedRepoRoot,
    workspaceRoot: resolvedWorkspaceRoot,
    sessionKey: normalizeText(options.sessionKey) ?? "graphify-bridge",
    sessionTimestamp: generatedAt,
    ...(options.sessionSourcePath === undefined ? {} : { sessionSourcePath: options.sessionSourcePath }),
    ...(options.proofSummarySourcePath === undefined ? {} : { proofSummarySourcePath: options.proofSummarySourcePath }),
    docsRoot: options.docsRoot ?? path.join(resolvedRepoRoot, "docs"),
    codeRoot: options.codeRoot ?? path.join(resolvedRepoRoot, "packages", "cli", "dist", "src"),
    generatedAt,
  });
  if (!projectionSourceBundle.ok) {
    throw new Error(`Graphify bridge projection source export failed: ${projectionSourceBundle.error ?? "unknown error"}`);
  }

  const sourceBundleManifest = buildDualSourceBundleManifest({
    runId,
    generatedAt,
    repoRoot: resolvedRepoRoot,
    workspaceRoot: resolvedWorkspaceRoot,
    outputRoot,
    openclawHome,
    activationRoot,
    sourceBundleRoot,
    canonicalSourceBundle,
    projectionSourceBundle,
  });
  writeJson(sourceBundleManifestPath, sourceBundleManifest);
  const sourceBundleTree = describeStablePathTree(sourceBundleRoot);

  const graphifyRun = runManagedGraphifyRunner({
    sourceBundlePath: sourceBundleRoot,
    outputRoot: graphifyRunRoot,
    runId: "managed-run",
    graphifyVersion: normalizeText(options.graphifyVersion) ?? GRAPHIFY_BRIDGE_VERSION_V1,
    graphifyMode: normalizeText(options.graphifyMode) ?? "dual-source-compiled-artifacts",
    graphifyConfig: {
      bridgeVersion: GRAPHIFY_BRIDGE_VERSION_V1,
      sourceBundleManifest: relativePathFromRoot(sourceBundleManifestPath, runRoot),
      canonicalBundleId: canonicalSourceBundle.bundleId ?? null,
      projectionRunId: projectionSourceBundle.runId ?? null,
      offPath: true,
      liveRuntimeEligible: false,
      ...(options.graphifyConfig && typeof options.graphifyConfig === "object" ? options.graphifyConfig : {}),
    },
    graphifyFlags: ["off-path", "artifact-first", "dual-source", "provenance-first", ...(Array.isArray(options.graphifyFlags) ? options.graphifyFlags : [])],
    labels: ["graphify-bridge", "dual-source-bundle", "compiled-artifact-pack", "off-path", ...(Array.isArray(options.labels) ? options.labels : [])],
    ...(options.graphifyCommand === undefined ? {} : { graphifyCommand: options.graphifyCommand }),
    ...(options.graphifyArgs === undefined ? {} : { graphifyArgs: options.graphifyArgs }),
  });
  if (!graphifyRun.ok) {
    throw new Error(`Graphify bridge managed Graphify run failed: ${graphifyRun.execution?.failure ?? "unknown error"}`);
  }

  const compiledPack = exportGraphifyCompiledArtifactsPack({
    bundleStartedAt: generatedAt,
    bundleId: `graphify-bridge-${runId}`,
    outputDir: compiledPackDir,
    proposalId: `prop_graphify_bridge_${runId.replace(/[^A-Za-z0-9_]+/gu, "_")}`,
    packId: `pack_graphify_bridge_${runId.replace(/[^A-Za-z0-9_]+/gu, "_")}`,
    graphifyRunId: graphifyRun.runId,
    graphifyVersion: graphifyRun.graphifyVersion,
    graphifyCommand: "managed graphify bridge cut",
    sourceBundleId: sourceBundleManifest.bundleId,
    sourceBundleHash: withSha256Prefix(graphifyRun.sourceBundleHash),
    graphHash: withSha256Prefix(graphifyRun.graph.hash),
    configHash: withSha256Prefix(graphifyRun.graphifyConfigHash),
    labelsHash: sha256Text(stableJsonStringify(graphifyRun.labels)),
    sourceDocs: [
      "docs/architecture/graphify-bridge.md",
      "docs/architecture/compiled-artifacts.md",
      "docs/architecture/teacher-v3-proof.md",
    ],
    sourceFixtures: [
      relativePathFromRoot(sourceBundleManifestPath, resolvedRepoRoot),
      relativePathFromRoot(canonicalSourceBundle.outputPaths?.corpusManifest ?? canonicalSourceBundle.bundleDir, resolvedRepoRoot),
      relativePathFromRoot(projectionSourceBundle.manifestPath ?? projectionSourceBundle.bundleRoot, resolvedRepoRoot),
      relativePathFromRoot(graphifyRun.outputs.summary, resolvedRepoRoot),
    ],
  });
  if (!compiledPack.ok) {
    throw new Error(`Graphify bridge compiled artifact pack failed: ${compiledPack.error ?? "unknown error"}`);
  }

  const status = buildBridgeStatus({
    runId,
    generatedAt,
    sourceBundleRoot,
    sourceBundleManifestPath,
    sourceBundleManifest,
    sourceBundleTree,
    graphifyRun,
    compiledPack,
  });
  writeJson(statusPath, status);
  writeText(summaryPath, renderBridgeSummary(status));

  return {
    ok: true,
    runId,
    generatedAt,
    runRoot,
    sourceBundleRoot,
    sourceBundleManifestPath,
    canonicalSourceBundle,
    projectionSourceBundle,
    sourceBundleManifest,
    sourceBundleTree,
    graphifyRun,
    compiledPack,
    status,
    statusPath,
    summaryPath,
    outputs: {
      sourceBundleRoot,
      sourceBundleManifest: sourceBundleManifestPath,
      canonicalSourceBundle: canonicalSourceBundle.bundleDir,
      projectionSourceBundle: projectionSourceBundle.bundleRoot,
      graphifyRun: graphifyRun.runDir,
      compiledArtifactPack: compiledPack.outputDir,
      status: statusPath,
      summary: summaryPath,
    },
  };
}
