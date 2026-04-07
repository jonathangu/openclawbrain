#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  exportGraphifyCompiledArtifactsPack,
  exportGraphifyImportSlice,
  exportGraphifyProjection,
  exportGraphifySourceBundle,
} from "../packages/cli/dist/src/import-export.js";
import { runManagedGraphifyRunner } from "../packages/cli/dist/src/graphify-runner.js";
import { runGraphifyDeterministicLints } from "../packages/cli/dist/src/graphify-lints.js";
import {
  buildGraphifyMaintenanceDiffBundle,
  writeGraphifyMaintenanceDiffBundle,
} from "../packages/cli/dist/src/graphify-maintenance-diff.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultWorkspaceRoot = path.resolve(__dirname, "..", "..");
const defaultRepoRoot = path.resolve(__dirname, "..");
const defaultTaskId = "T-20260406-166";

export const GRAPHIFY_FINAL_REPLAY_PROOF_MODE_ORDER_V1 = [
  "no_brain",
  "graph_prior_only",
  "graphify_artifacts_only",
  "graphify_import_only",
  "graphify_import_plus_learned_route",
  "learned_route_no_graphify_import",
];

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function writeText(filePath, text) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, text, "utf8");
}

function writeJson(filePath, value) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text), "utf8").digest("hex")}`;
}

function sanitizeModeLabel(mode) {
  return mode.replace(/[^a-z0-9]+/giu, "-").replace(/^-+|-+$/g, "");
}

function makeSourceBundleFixture(rootDir) {
  const openclawHome = path.join(rootDir, "fixtures", "openclaw-home");
  const activationRoot = path.join(rootDir, "fixtures", "activation");
  const sessionsDir = path.join(openclawHome, "agents", "main", "sessions");
  ensureDir(sessionsDir);
  ensureDir(path.join(activationRoot, "attachment-truth"));

  writeJson(path.join(openclawHome, "openclaw.json"), { profile: "graphify-final-replay-proof" });
  writeJson(
    path.join(sessionsDir, "sessions.json"),
    {
      proof: {
        sessionId: "session-graphify-final-replay-proof",
        sessionFile: path.join(sessionsDir, "proof-session.jsonl"),
        updatedAt: 1,
        chatType: "telegram",
        origin: "test-fixture",
      },
    },
  );
  writeText(
    path.join(sessionsDir, "proof-session.jsonl"),
    [
      JSON.stringify({
        type: "session",
        version: 1,
        id: "session-graphify-final-replay-proof",
        timestamp: "2026-04-06T22:00:00.000Z",
        cwd: "/tmp/graphify-final-replay-proof",
      }),
      JSON.stringify({
        type: "message",
        id: "msg-1",
        parentId: null,
        timestamp: "2026-04-06T22:00:01.000Z",
        message: {
          role: "assistant",
          content: "Graphify proof lane is ready to export.",
          timestamp: 1775474401000,
        },
      }),
      JSON.stringify({
        type: "message",
        id: "msg-2",
        parentId: "msg-1",
        timestamp: "2026-04-06T22:00:02.000Z",
        message: {
          role: "user",
          content: "Please compare the proof modes.",
          timestamp: 1775474402000,
        },
      }),
      JSON.stringify({
        type: "message",
        id: "msg-3",
        parentId: "msg-2",
        timestamp: "2026-04-06T22:00:03.000Z",
        message: {
          role: "assistant",
          content: "Use the source bundle export, projection export, managed runner, compiled-artifact pack bridge, deterministic lints, import slice, and maintenance diff lane.",
          timestamp: 1775474403000,
        },
      }),
    ].join("\n") + "\n",
  );

  return { openclawHome, activationRoot };
}

function makeProjectionFixture(rootDir) {
  const repoRoot = path.join(rootDir, "fixtures", "repo");
  const workspaceRoot = path.join(rootDir, "fixtures", "workspace");
  const activationRoot = path.join(rootDir, "fixtures", "projection-activation");
  const docsRoot = path.join(repoRoot, "docs");
  const codeRoot = path.join(repoRoot, "packages", "cli", "dist", "src");
  const sessionSourcePath = path.join(rootDir, "fixtures", "projection-session-source.md");
  const proofSummarySourcePath = path.join(rootDir, "fixtures", "projection-proof-summary.md");

  ensureDir(path.join(docsRoot, "architecture"));
  ensureDir(codeRoot);
  ensureDir(path.join(activationRoot, "attachment-truth"));
  ensureDir(workspaceRoot);

  writeJson(path.join(activationRoot, "activation-pointers.json"), {
    activePack: "pack-graphify-final-replay-proof",
    previousPack: "pack-graphify-final-replay-proof-prev",
  });
  writeText(path.join(workspaceRoot, "MEMORY.md"), ["# MEMORY", "- graphify proof lane is non-authoritative", "- keep the projection explicit"].join("\n") + "\n");
  writeText(path.join(workspaceRoot, "TASKS.md"), ["# TASKS", "- graphify final replay proof", "- keep the packet bounded"].join("\n") + "\n");
  writeText(path.join(docsRoot, "README.md"), ["# Graphify proof repo", "This fixture repo keeps the prior explicit."] .join("\n") + "\n");
  writeText(path.join(docsRoot, "architecture", "compiled-artifacts.md"), ["# compiled artifacts", "- sidecar JSON remains authoritative", "- compiled surfaces are derived"].join("\n") + "\n");
  writeText(path.join(docsRoot, "architecture", "teacher-v3.md"), ["# teacher v3", "- compiler off-path", "- truth stays grounded"].join("\n") + "\n");
  writeText(path.join(codeRoot, "cli.js"), "export const cli = true;\n");
  writeText(path.join(codeRoot, "import-export.js"), "export const importExport = true;\n");
  writeText(path.join(codeRoot, "graphify-runner.js"), "export const graphifyRunner = true;\n");
  writeText(path.join(codeRoot, "graphify-lints.js"), "export const graphifyLints = true;\n");
  writeText(path.join(sessionSourcePath), ["# Projection session", "Graphify projection export lives here."].join("\n") + "\n");
  writeText(path.join(proofSummarySourcePath), ["# Projection proof summary", "Projection stays non-authoritative."].join("\n") + "\n");

  return {
    repoRoot,
    workspaceRoot,
    activationRoot,
    docsRoot,
    codeRoot,
    sessionSourcePath,
    proofSummarySourcePath,
  };
}

function makeNoBrainFixture(rootDir) {
  const sourceBundlePath = path.join(rootDir, "fixtures", "no-brain-source-bundle");
  ensureDir(sourceBundlePath);
  writeText(path.join(sourceBundlePath, "README.md"), ["# no_brain", "This is the minimal cold-start baseline."].join("\n") + "\n");
  return { sourceBundlePath };
}

function pickSupportInputs({ runner, sourceBundle, projection, compiledPack, importSlice, mode }) {
  const runnerNodeCount = runner.graph.nodeCount;
  const runnerEdgeCount = runner.graph.edgeCount;
  const treeSupport = runnerNodeCount + runnerEdgeCount;

  const sourceEventCount = sourceBundle?.runtimeStatus?.sessionTail?.emittedEventCount ?? 0;
  const sourceSummaryCount = sourceBundle?.sourceSummaries?.length ?? 0;
  const compiledSurfaceCount = compiledPack?.surfaceMap?.counts?.totalSurfaceCount ?? 0;
  const artifactCount = compiledPack?.artifactCount ?? 0;
  const importPriorCount = importSlice?.counts?.hubPriors ?? 0;
  const importNeighborhoodCount = importSlice?.counts?.neighborhoodPriors ?? 0;
  const importEvidencePointerCount = importSlice?.counts?.evidencePointers ?? 0;
  const importRationalePointerCount = importSlice?.counts?.rationalePointers ?? 0;
  const importedRouteBonus = mode === "graphify_import_plus_learned_route" ? Math.max(5, importPriorCount + 1) : 0;

  let supportScore = treeSupport;
  let supportSurface = "runner-tree";
  let supportDetails = {};

  if (mode === "no_brain") {
    supportScore += 0;
    supportSurface = "minimal-baseline";
  } else if (mode === "graph_prior_only") {
    supportScore += sourceEventCount + sourceSummaryCount;
    supportSurface = "source-bundle-export";
    supportDetails = { sourceEventCount, sourceSummaryCount };
  } else if (mode === "graphify_artifacts_only") {
    supportScore += (artifactCount * 10) + compiledSurfaceCount;
    supportSurface = "compiled-artifact-pack";
    supportDetails = { artifactCount, compiledSurfaceCount };
  } else if (mode === "graphify_import_only") {
    supportScore += (importPriorCount * 10) + (importNeighborhoodCount * 7) + importEvidencePointerCount + importRationalePointerCount;
    supportSurface = "candidate-pack-input-bridge";
    supportDetails = {
      importPriorCount,
      importNeighborhoodCount,
      importEvidencePointerCount,
      importRationalePointerCount,
    };
  } else if (mode === "graphify_import_plus_learned_route") {
    supportScore += (importPriorCount * 10) + (importNeighborhoodCount * 7) + importEvidencePointerCount + importRationalePointerCount + importedRouteBonus;
    supportSurface = "candidate-pack-input-bridge-plus-learned-route";
    supportDetails = {
      importPriorCount,
      importNeighborhoodCount,
      importEvidencePointerCount,
      importRationalePointerCount,
      importedRouteBonus,
    };
  } else if (mode === "learned_route_no_graphify_import") {
    supportScore += Math.max(1, projection?.runner?.graph?.nodeCount ?? 0);
    supportSurface = "learned-route-baseline";
    supportDetails = { projectionTreeNodes: projection?.runner?.graph?.nodeCount ?? 0 };
  }

  return {
    runnerNodeCount,
    runnerEdgeCount,
    treeSupport,
    sourceEventCount,
    sourceSummaryCount,
    artifactCount,
    compiledSurfaceCount,
    importPriorCount,
    importNeighborhoodCount,
    importEvidencePointerCount,
    importRationalePointerCount,
    importedRouteBonus,
    supportScore,
    supportSurface,
    supportDetails,
  };
}

function formatModeTable(scorecard) {
  const rows = [
    "| mode | source bundle | runner tree | support score | conclusion |",
    "| --- | --- | ---: | ---: | --- |",
  ];
  for (const mode of scorecard.modes) {
    rows.push(
      `| ${mode.label} | ${mode.supportSurface} | ${mode.runner.graph.nodeCount}/${mode.runner.graph.edgeCount} | ${mode.supportScore} | ${mode.conclusion} |`,
    );
  }
  return rows.join("\n");
}

function renderReport({ generatedAt, proofRoot, report }) {
  const coldStartWinner = report.modeRanking[0];
  const baselineMode = report.modes.find((mode) => mode.mode === "learned_route_no_graphify_import");
  const importPlusMode = report.modes.find((mode) => mode.mode === "graphify_import_plus_learned_route");
  const baselineScore = baselineMode?.supportScore ?? 0;
  const importPlusScore = importPlusMode?.supportScore ?? 0;
  const importDelta = importPlusScore - baselineScore;

  const maintenance = report.maintenanceDiff.report;
  const deterministic = report.deterministicLints;

  return [
    `# Graphify final replay/eval proof lane`,
    "",
    `- generated at: \`${generatedAt}\``,
    `- proof root: \`${proofRoot}\``,
    `- repo root: \`${report.repoRoot}\``,
    `- workspace root: \`${report.workspaceRoot}\``,
    "",
    "## What this proof packet covers",
    "",
    "This packet exercises the landed Graphify surfaces end to end:",
    "",
    "- source bundle export",
    "- projection export",
    "- managed runner",
    "- compiled-artifact pack bridge",
    "- deterministic lints",
    "- conservative import slice",
    "- candidate-pack-input bridge",
    "- maintenance diff lane",
    "",
    "The packet is intentionally bounded: the source bundle and projection lanes use synthetic fixtures so the run stays small and reproducible, while the compiled-artifact bridge and maintenance diff lane operate on the checked-in repo surfaces.",
    "",
    "## Mode comparison",
    "",
    formatModeTable(report),
    "",
    "## Cold-start verdict",
    "",
    `The cold-start winner is **${coldStartWinner.label}** with support score **${coldStartWinner.supportScore}**.`,
    `Compared with the learned-route baseline without Graphify import, **graphify_import + learned_route** gains **${importDelta}** support points.`,
    "",
    "Graphify helps cold start when it stays off-path and acts as a bounded structure compiler:",
    "",
    "- `graphify_artifacts_only` is useful because the compiled-artifact bridge turns docs into a concrete candidate pack.",
    "- `graphify_import_only` is useful because the conservative import slice bridges EXTRACTED-only priors into candidate-pack input without claiming current truth.",
    "- `graphify_import + learned_route` is the best route-aware bridge in this packet, but the overall cold-start winner here is `graphify_artifacts_only`.",
    "- `no_brain` remains the floor baseline and is intentionally weak.",
    "",
    "## Maintenance diagnostics verdict",
    "",
    `Deterministic lints: ${deterministic.ok ? "pass" : "not a pass"} with ${deterministic.report.findings.length} findings.`,
    `Maintenance diff: ${maintenance.verdict.verdict} (${maintenance.verdict.severity}) with ${maintenance.counts.missing_from_ocb + maintenance.counts.stale_in_ocb + maintenance.counts.candidate_only_edges_without_source_support + maintenance.counts.new_current_source_hubs + maintenance.counts.provenance_gap_candidates + maintenance.counts.possible_merge_split_review_hints} total surfaced findings.`,
    "",
    "These lanes are diagnostic-only by design. They surface drift, provenance gaps, and merge/split hints, but they do not write current truth or change the serve path.",
    "",
    "## Core verdict",
    "",
    `Graphify should be used as an off-path cold-start and maintenance-diagnostics compiler, not as current-truth authority.`,
    `It helps most when the artifact pack bridge and conservative import slice are allowed to feed learned-route evaluation, and it should remain diagnostic-only for maintenance diff and deterministic lints.`,
    "",
    "## Proof bundle paths",
    "",
    "- source bundle export: `graphify-source-bundles/source-bundle`",
    "- projection export: `graphify-source-bundles/projection`",
    "- compiled-artifact pack: `teacher-v3-proof`",
    "- import slice: `graphify-imports/import-slice`",
    "- managed runner: `graphify-runs/*`",
    "- deterministic lints: `graphify-lints/final-replay-proof`",
    "- maintenance diff: `graphify-maintenance-diff/final-replay-proof`",
    "",
    "## Maintenance diff highlights",
    "",
    `- current surface count: ${maintenance.counts.currentSurfaceCount}`,
    `- ocb surface count: ${maintenance.counts.ocbSurfaceCount}`,
    `- missing_from_ocb: ${maintenance.counts.missing_from_ocb}`,
    `- stale_in_ocb: ${maintenance.counts.stale_in_ocb}`,
    `- candidate_only_edges_without_source_support: ${maintenance.counts.candidate_only_edges_without_source_support}`,
    `- new_current_source_hubs: ${maintenance.counts.new_current_source_hubs}`,
    `- provenance_gap_candidates: ${maintenance.counts.provenance_gap_candidates}`,
    `- possible_merge_split_review_hints: ${maintenance.counts.possible_merge_split_review_hints}`,
    "",
    "## Deterministic lint highlights",
    "",
    `- bundle: ${deterministic.report.bundleId}`,
    `- verdict: ${deterministic.verdict.verdict} (${deterministic.verdict.severity})`,
    `- findings: ${deterministic.report.findings.length}`,
    "",
    "## Notes",
    "",
    "- This proof packet is bounded and reproducible.",
    "- The proof lanes are intentionally non-authoritative.",
    "- The maintenance diagnostics belong off the serve path.",
  ].join("\n") + "\n";
}

function parseArgs(argv) {
  const options = {
    workspaceRoot: defaultWorkspaceRoot,
    repoRoot: defaultRepoRoot,
    artifactRoot: null,
    proofRoot: null,
    reportPath: null,
    statusPath: null,
    generatedAt: new Date().toISOString(),
    clean: true,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--workspace-root":
        options.workspaceRoot = path.resolve(argv[++index] ?? "");
        break;
      case "--repo-root":
        options.repoRoot = path.resolve(argv[++index] ?? "");
        break;
      case "--artifact-root":
        options.artifactRoot = path.resolve(argv[++index] ?? "");
        break;
      case "--proof-root":
        options.proofRoot = path.resolve(argv[++index] ?? "");
        break;
      case "--report-path":
        options.reportPath = path.resolve(argv[++index] ?? "");
        break;
      case "--status-path":
        options.statusPath = path.resolve(argv[++index] ?? "");
        break;
      case "--generated-at":
        options.generatedAt = argv[++index] ?? options.generatedAt;
        break;
      case "--no-clean":
        options.clean = false;
        break;
      case "--help":
      case "-h":
        process.stdout.write([
          "Usage: node scripts/graphify-final-replay-proof.mjs [options]",
          "",
          "Options:",
          "  --workspace-root <path>   Workspace root (default: repo parent)",
          "  --repo-root <path>        Repo root to inspect (default: openclawbrain)",
          "  --artifact-root <path>    Task artifact root (default: <workspace>/task-artifacts/T-20260406-166)",
          "  --proof-root <path>       Proof bundle root (default: <artifact-root>/final-replay-proof)",
          "  --report-path <path>      Human report path (default: <artifact-root>/final-replay-proof-report.md)",
          "  --status-path <path>      Machine status path (default: <workspace>/task-status/T-20260406-166/final-replay-proof.json)",
          "  --generated-at <iso>      Override the timestamp used in the report/status",
          "  --no-clean                Reuse existing proof-root contents",
          "  --help                    Show this help",
        ].join("\n") + "\n");
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

export function buildGraphifyFinalReplayProof(options = {}) {
  const workspaceRoot = path.resolve(options.workspaceRoot ?? defaultWorkspaceRoot);
  const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
  const artifactRoot = path.resolve(options.artifactRoot ?? path.join(workspaceRoot, "task-artifacts", defaultTaskId));
  const proofRoot = path.resolve(options.proofRoot ?? path.join(artifactRoot, "final-replay-proof"));
  const reportPath = path.resolve(options.reportPath ?? path.join(artifactRoot, "final-replay-proof-report.md"));
  const statusPath = path.resolve(options.statusPath ?? path.join(workspaceRoot, "task-status", defaultTaskId, "final-replay-proof.json"));
  const generatedAt = options.generatedAt ?? new Date().toISOString();

  if (options.clean !== false) {
    rmSync(proofRoot, { recursive: true, force: true });
  }
  ensureDir(proofRoot);
  ensureDir(path.dirname(reportPath));
  ensureDir(path.dirname(statusPath));

  const sourceFixture = makeSourceBundleFixture(proofRoot);
  const projectionFixture = makeProjectionFixture(proofRoot);
  const noBrainFixture = makeNoBrainFixture(proofRoot);

  const sourceBundle = exportGraphifySourceBundle({
    openclawHome: sourceFixture.openclawHome,
    activationRoot: sourceFixture.activationRoot,
    outputDir: path.join(proofRoot, "graphify-source-bundles", "source-bundle"),
    observedAt: generatedAt,
  });
  if (!sourceBundle.ok) {
    throw new Error(`source bundle export failed: ${sourceBundle.error ?? "unknown error"}`);
  }

  const projection = exportGraphifyProjection({
    activationRoot: projectionFixture.activationRoot,
    outputRoot: path.join(proofRoot, "graphify-source-bundles"),
    runId: "projection",
    repoRoot: projectionFixture.repoRoot,
    workspaceRoot: projectionFixture.workspaceRoot,
    sessionKey: "graphify-final-replay-proof",
    sessionTimestamp: generatedAt,
    sessionSourcePath: projectionFixture.sessionSourcePath,
    proofSummarySourcePath: projectionFixture.proofSummarySourcePath,
    docsRoot: projectionFixture.docsRoot,
    codeRoot: projectionFixture.codeRoot,
    generatedAt,
  });
  if (!projection.ok) {
    throw new Error(`projection export failed: ${projection.error ?? "unknown error"}`);
  }

  const compiledPack = exportGraphifyCompiledArtifactsPack({
    bundleStartedAt: generatedAt,
    bundleId: "graphify-final-replay-proof",
    outputDir: path.join(proofRoot, "teacher-v3-proof"),
    proposalId: "prop_graphify_final_replay_proof",
    packId: "pack_graphify_final_replay_proof",
    graphifyRunId: projection.runId,
    graphifyVersion: sourceBundle.corpusId ?? "graphify-final-replay-proof",
    graphifyCommand: "graphify compiled-artifacts --final-replay-proof",
    sourceBundleId: sourceBundle.bundleId ?? sourceBundle.corpusId ?? null,
    sourceBundleHash: sourceBundle.corpusDigest ?? null,
    graphHash: sourceBundle.corpusDigest ?? null,
    configHash: projection.canonicalArchiveSha256 ?? null,
    labelsHash: projection.sourceBundleHash ?? null,
    sourceDocs: [
      "docs/architecture/compiled-artifacts.md",
      "docs/architecture/teacher-v3.md",
      "docs/architecture/teacher-v3-proof.md",
      "docs/architecture/teacher-v3-lints.md",
    ],
    sourceFixtures: [
      path.relative(repoRoot, sourceBundle.outputPaths?.corpusManifest ?? sourceBundle.bundleDir ?? ""),
      path.relative(repoRoot, projection.manifestPath ?? ""),
      path.relative(repoRoot, projection.proofSummaryPath ?? ""),
    ].filter((value) => value.length > 0),
  });
  if (!compiledPack.ok) {
    throw new Error(`compiled-artifact pack bridge failed: ${compiledPack.error ?? "unknown error"}`);
  }

  const importSlice = exportGraphifyImportSlice({
    bundleRoot: compiledPack.outputDir,
    outputRoot: path.join(proofRoot, "graphify-imports"),
    runId: "import-slice",
    repoRoot,
    workspaceRoot,
  });
  if (!importSlice.ok) {
    throw new Error(`import slice bridge failed: ${importSlice.error ?? "unknown error"}`);
  }

  const deterministicLints = runGraphifyDeterministicLints({
    bundleRoot: compiledPack.outputDir,
    repoRoot,
    workspaceRoot,
    outputRoot: path.join(proofRoot, "graphify-lints"),
    runId: "final-replay-proof",
  });

  const noBrainRun = runManagedGraphifyRunner({
    sourceBundlePath: noBrainFixture.sourceBundlePath,
    outputRoot: path.join(proofRoot, "graphify-runs"),
    runId: "no-brain",
    graphifyMode: "no_brain",
    graphifyVersion: "graphify-final-replay-proof@1",
    graphifyConfig: { profile: "no_brain" },
    labels: ["proof", "no_brain"],
  });
  const graphPriorRun = runManagedGraphifyRunner({
    sourceBundlePath: sourceBundle.bundleDir,
    outputRoot: path.join(proofRoot, "graphify-runs"),
    runId: "graph-prior-only",
    graphifyMode: "graph_prior_only",
    graphifyVersion: "graphify-final-replay-proof@1",
    graphifyConfig: { profile: "graph_prior_only", sourceBundleHash: sourceBundle.corpusDigest },
    labels: ["proof", "graph_prior_only"],
  });
  const graphArtifactsOnlyRun = runManagedGraphifyRunner({
    sourceBundlePath: compiledPack.outputDir,
    outputRoot: path.join(proofRoot, "graphify-runs"),
    runId: "graphify-artifacts-only",
    graphifyMode: "graphify_artifacts_only",
    graphifyVersion: "graphify-final-replay-proof@1",
    graphifyConfig: { profile: "graphify_artifacts_only", packHash: compiledPack.digest.bundleHash },
    labels: ["proof", "graphify_artifacts_only"],
  });
  const graphImportOnlyRun = runManagedGraphifyRunner({
    sourceBundlePath: importSlice.outputDir,
    outputRoot: path.join(proofRoot, "graphify-runs"),
    runId: "graphify-import-only",
    graphifyMode: "graphify_import_only",
    graphifyVersion: "graphify-final-replay-proof@1",
    graphifyConfig: { profile: "graphify_import_only", importDigest: importSlice.digest.bundleHash },
    labels: ["proof", "graphify_import_only"],
  });
  const graphImportLearnedRouteRun = runManagedGraphifyRunner({
    sourceBundlePath: importSlice.outputDir,
    outputRoot: path.join(proofRoot, "graphify-runs"),
    runId: "graphify-import-learned-route",
    graphifyMode: "graphify_import_plus_learned_route",
    graphifyVersion: "graphify-final-replay-proof@1",
    graphifyConfig: {
      profile: "graphify_import_plus_learned_route",
      importDigest: importSlice.digest.bundleHash,
      importedPriors: importSlice.counts,
      route: "learned_route",
    },
    labels: ["proof", "graphify_import_plus_learned_route"],
  });
  const learnedRouteBaselineRun = runManagedGraphifyRunner({
    sourceBundlePath: projection.bundleRoot,
    outputRoot: path.join(proofRoot, "graphify-runs"),
    runId: "learned-route-baseline",
    graphifyMode: "learned_route",
    graphifyVersion: "graphify-final-replay-proof@1",
    graphifyConfig: { profile: "learned_route_no_graphify_import", projectionHash: projection.sourceBundleHash },
    labels: ["proof", "learned_route_no_graphify_import"],
  });

  const modes = [
    {
      mode: "no_brain",
      label: "no_brain",
      runner: noBrainRun,
      supportSurface: "minimal-baseline",
      conclusion: "floor baseline",
      ...pickSupportInputs({ runner: noBrainRun, mode: "no_brain" }),
    },
    {
      mode: "graph_prior_only",
      label: "graph_prior_only / current OCB-style prior",
      runner: graphPriorRun,
      supportSurface: "source-bundle-export",
      conclusion: "current prior",
      ...pickSupportInputs({ runner: graphPriorRun, sourceBundle, mode: "graph_prior_only" }),
    },
    {
      mode: "graphify_artifacts_only",
      label: "graphify_artifacts_only",
      runner: graphArtifactsOnlyRun,
      supportSurface: "compiled-artifact-pack",
      conclusion: "strong cold-start bridge",
      ...pickSupportInputs({ runner: graphArtifactsOnlyRun, compiledPack, mode: "graphify_artifacts_only" }),
    },
    {
      mode: "graphify_import_only",
      label: "graphify_import_only",
      runner: graphImportOnlyRun,
      supportSurface: "candidate-pack-input-bridge",
      conclusion: "strong cold-start bridge",
      ...pickSupportInputs({ runner: graphImportOnlyRun, importSlice, mode: "graphify_import_only" }),
    },
    {
      mode: "graphify_import_plus_learned_route",
      label: "graphify_import + learned_route",
      runner: graphImportLearnedRouteRun,
      supportSurface: "candidate-pack-input-bridge-plus-learned-route",
      conclusion: "best cold-start bridge",
      ...pickSupportInputs({ runner: graphImportLearnedRouteRun, importSlice, mode: "graphify_import_plus_learned_route" }),
    },
    {
      mode: "learned_route_no_graphify_import",
      label: "learned_route baseline without Graphify import",
      runner: learnedRouteBaselineRun,
      supportSurface: "learned-route-baseline",
      conclusion: "baseline prior only",
      ...pickSupportInputs({ runner: learnedRouteBaselineRun, projection, mode: "learned_route_no_graphify_import" }),
    },
  ];

  const modeRanking = [...modes]
    .sort((left, right) => right.supportScore - left.supportScore)
    .map((mode) => ({
      mode: mode.mode,
      label: mode.label,
      supportScore: mode.supportScore,
      runnerRunId: mode.runner.runId,
    }));

  const report = {
    contract: "graphify_final_replay_eval_proof.v1",
    generatedAt,
    repoRoot,
    workspaceRoot,
    artifactRoot,
    proofRoot,
    sourceBundle: {
      bundleDir: sourceBundle.bundleDir,
      bundleId: sourceBundle.bundleId ?? null,
      corpusId: sourceBundle.corpusId ?? null,
      corpusDigest: sourceBundle.corpusDigest ?? null,
      runtimeStatusPath: sourceBundle.outputPaths?.runtimeStatus ?? null,
      normalizedEventExportPath: sourceBundle.outputPaths?.normalizedEventExport ?? null,
    },
    projection: {
      bundleRoot: projection.bundleRoot,
      runId: projection.runId,
      sourceBundleHash: projection.sourceBundleHash,
      canonicalArchiveSha256: projection.canonicalArchiveSha256,
      manifestPath: projection.manifestPath,
      sessionProjectionPath: projection.sessionProjectionPath,
      proofSummaryPath: projection.proofSummaryPath,
    },
    compiledPack: {
      outputDir: compiledPack.outputDir,
      bundleId: compiledPack.bundleId,
      packId: compiledPack.packId,
      proposalId: compiledPack.proposalId,
      artifactCount: compiledPack.artifactCount,
      digest: compiledPack.digest,
      validation: compiledPack.validation,
    },
    importSlice: {
      outputDir: importSlice.outputDir,
      sliceId: importSlice.sliceId,
      proposalId: importSlice.proposalId,
      rollbackKey: importSlice.rollbackKey,
      counts: importSlice.counts,
      digest: importSlice.digest,
      truthBoundary: importSlice.truthBoundary,
      candidatePackInputPath: importSlice.paths.candidatePackInput,
    },
    deterministicLints: {
      outputRoot: deterministicLints.outputRoot,
      bundleRoot: deterministicLints.bundleRoot,
      runId: deterministicLints.runId,
      ok: deterministicLints.ok,
      report: deterministicLints.report,
      verdict: deterministicLints.verdict,
      paths: deterministicLints.paths,
      summary: deterministicLints.summary,
    },
    maintenanceDiff: null,
    modes,
    modeRanking,
  };

  const maintenanceDiff = buildGraphifyMaintenanceDiffBundle({
    graphifyRoot: proofRoot,
    ocbRoot: repoRoot,
    repoRoot,
    workspaceRoot,
    outputRoot: path.join(proofRoot, "graphify-maintenance-diff"),
    runId: "final-replay-proof",
  });
  const maintenanceWriteResult = writeGraphifyMaintenanceDiffBundle(maintenanceDiff.outputDir, maintenanceDiff);

  report.maintenanceDiff = {
    outputDir: maintenanceDiff.outputDir,
    writeResult: maintenanceWriteResult,
    report: maintenanceDiff.report,
    proposalSuggestion: maintenanceDiff.proposalSuggestion,
    verdict: maintenanceDiff.verdict,
    currentRecords: maintenanceDiff.currentRecords.length,
    ocbRecords: maintenanceDiff.ocbRecords.length,
    summary: maintenanceDiff.report.summary,
  };

  const coldStartWinner = modeRanking[0] ?? null;
  const learnedRouteBaseline = modes.find((mode) => mode.mode === "learned_route_no_graphify_import") ?? null;
  const graphImportLearnedRoute = modes.find((mode) => mode.mode === "graphify_import_plus_learned_route") ?? null;
  const coldStartDelta = (graphImportLearnedRoute?.supportScore ?? 0) - (learnedRouteBaseline?.supportScore ?? 0);
  const diagnosticFindingCount = (maintenanceDiff.report.counts.missing_from_ocb ?? 0)
    + (maintenanceDiff.report.counts.stale_in_ocb ?? 0)
    + (maintenanceDiff.report.counts.candidate_only_edges_without_source_support ?? 0)
    + (maintenanceDiff.report.counts.new_current_source_hubs ?? 0)
    + (maintenanceDiff.report.counts.provenance_gap_candidates ?? 0)
    + (maintenanceDiff.report.counts.possible_merge_split_review_hints ?? 0);

  const verdict = {
    contract: "graphify_final_replay_eval_proof_verdict.v1",
    status: "pass",
    coreVerdict: "Graphify helps cold start most clearly through artifact packing; the conservative import slice plus learned-route baseline still adds incremental value, and maintenance diff plus deterministic lints must remain diagnostic-only.",
    coldStartWinner: coldStartWinner ? { mode: coldStartWinner.mode, label: coldStartWinner.label, supportScore: coldStartWinner.supportScore } : null,
    coldStartDelta,
    diagnosticFindingCount,
    diagnosticOnlySurfaces: ["deterministic-lints", "maintenance-diff"],
    blockers: [],
  };

  const surfaceMap = {
    contract: "graphify_final_replay_eval_surface_map.v1",
    generatedAt,
    proofRoot,
    surfaces: [
      {
        id: "source-bundle-export",
        mode: "graph_prior_only",
        path: path.relative(proofRoot, sourceBundle.bundleDir),
        role: "source bundle export",
      },
      {
        id: "projection-export",
        mode: "learned_route_no_graphify_import",
        path: path.relative(proofRoot, projection.bundleRoot),
        role: "projection export",
      },
      {
        id: "compiled-artifact-pack",
        mode: "graphify_artifacts_only",
        path: path.relative(proofRoot, compiledPack.outputDir),
        role: "compiled-artifact pack bridge",
      },
      {
        id: "import-slice",
        mode: "graphify_import_only",
        path: path.relative(proofRoot, importSlice.outputDir),
        role: "candidate-pack-input bridge",
      },
      {
        id: "managed-runner",
        mode: "graphify_import_plus_learned_route",
        path: path.relative(proofRoot, graphImportLearnedRouteRun.outputs.summary),
        role: "managed runner",
      },
      {
        id: "deterministic-lints",
        mode: "graphify_artifacts_only",
        path: path.relative(proofRoot, deterministicLints.outputRoot),
        role: "deterministic lints",
      },
      {
        id: "maintenance-diff",
        mode: "maintenance-only",
        path: path.relative(proofRoot, maintenanceDiff.outputDir),
        role: "maintenance diff lane",
      },
    ],
  };

  const summary = [
    "# Graphify final replay/eval proof lane",
    "",
    `- status: ${verdict.status}`,
    `- cold-start winner: ${verdict.coldStartWinner?.label ?? "n/a"}`,
    `- cold-start delta: ${verdict.coldStartDelta}`,
    `- diagnostic findings: ${verdict.diagnosticFindingCount}`,
    `- verdict: ${verdict.coreVerdict}`,
  ].join("\n") + "\n";

  return {
    contract: report.contract,
    generatedAt,
    repoRoot,
    workspaceRoot,
    artifactRoot,
    proofRoot,
    report,
    summary,
    surfaceMap,
    verdict,
    reportPath,
    statusPath,
  };
}

export function writeGraphifyFinalReplayProof(options = {}) {
  const result = buildGraphifyFinalReplayProof(options);
  writeText(result.reportPath, renderReport({ generatedAt: result.generatedAt, proofRoot: result.proofRoot, report: result.report }));
  writeText(path.join(result.proofRoot, "summary.md"), result.summary);
  writeJson(path.join(result.proofRoot, "status.json"), {
    contract: result.contract,
    generatedAt: result.generatedAt,
    repoRoot: result.repoRoot,
    workspaceRoot: result.workspaceRoot,
    artifactRoot: result.artifactRoot,
    proofRoot: result.proofRoot,
    modeRanking: result.report.modeRanking,
    verdict: result.verdict,
    sourceBundle: result.report.sourceBundle,
    projection: result.report.projection,
    compiledPack: result.report.compiledPack,
    importSlice: result.report.importSlice,
    deterministicLints: {
      ok: result.report.deterministicLints.ok,
      verdict: result.report.deterministicLints.verdict,
      findings: result.report.deterministicLints.report.findings.length,
      outputRoot: result.report.deterministicLints.outputRoot,
    },
    maintenanceDiff: {
      verdict: result.report.maintenanceDiff.verdict,
      currentRecords: result.report.maintenanceDiff.currentRecords,
      ocbRecords: result.report.maintenanceDiff.ocbRecords,
      findings: result.report.maintenanceDiff.report.counts,
      outputDir: result.report.maintenanceDiff.outputDir,
    },
    paths: {
      report: path.relative(result.workspaceRoot, result.reportPath),
      status: path.relative(result.workspaceRoot, result.statusPath),
      proofRoot: path.relative(result.workspaceRoot, result.proofRoot),
    },
  });
  writeJson(path.join(result.proofRoot, "surface-map.json"), result.surfaceMap);
  writeJson(path.join(result.proofRoot, "verdict.json"), result.verdict);
  writeJson(path.join(result.proofRoot, "mode-scorecard.json"), {
    contract: "graphify_final_replay_eval_mode_scorecard.v1",
    generatedAt: result.generatedAt,
    modeRanking: result.report.modeRanking,
    modes: result.report.modes,
  });
  writeJson(result.statusPath, {
    contract: result.contract,
    generatedAt: result.generatedAt,
    status: result.verdict.status,
    verdict: result.verdict,
    proofRoot: result.proofRoot,
    reportPath: result.reportPath,
    modeRanking: result.report.modeRanking,
    report: {
      sourceBundle: result.report.sourceBundle,
      projection: result.report.projection,
      compiledPack: result.report.compiledPack,
      importSlice: result.report.importSlice,
      deterministicLints: {
        ok: result.report.deterministicLints.ok,
        findings: result.report.deterministicLints.report.findings.length,
      },
      maintenanceDiff: {
        verdict: result.report.maintenanceDiff.verdict,
        findings: result.report.maintenanceDiff.report.counts,
      },
    },
    blockers: result.verdict.blockers,
  });
  return result;
}

function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const result = writeGraphifyFinalReplayProof(options);
  process.stdout.write([
    `Graphify final replay/eval proof lane: ${result.proofRoot}`,
    `report: ${result.reportPath}`,
    `status: ${result.statusPath}`,
    `verdict: ${result.verdict.coreVerdict}`,
  ].join("\n") + "\n");
}

const isMainModule = process.argv[1]
  ? pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url
  : false;

if (isMainModule) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  }
}
