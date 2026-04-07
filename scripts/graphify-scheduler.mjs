#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { exportGraphifyCompiledArtifactsPack, exportGraphifyImportSlice, exportGraphifySourceBundle } from "../packages/cli/dist/src/import-export.js";
import { runManagedGraphifyRunner } from "../packages/cli/dist/src/graphify-runner.js";
import { runGraphifyDeterministicLints } from "../packages/cli/dist/src/graphify-lints.js";
import { buildGraphifyMaintenanceDiffBundle, writeGraphifyMaintenanceDiffBundle } from "../packages/cli/dist/src/graphify-maintenance-diff.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultRepoRoot = path.resolve(__dirname, "..");
const defaultWorkspaceRoot = path.resolve(defaultRepoRoot, "..");
const defaultOpenClawHome = path.join(os.homedir(), ".openclaw");
const SCHEDULER_VERSION = "graphify-scheduler@1";
const CONTRACT = "graphify_scheduler_registry.v1";

export const GRAPHIFY_SCHEDULER_CADENCE_ORDER_V1 = ["delta", "reorg"];
export const GRAPHIFY_SCHEDULER_LAYOUT_V1 = {
  registry: "registry.json",
  retentionPolicy: "retention-policy.json",
  retentionPolicyMarkdown: "retention-policy.md",
  summary: "summary.md",
  status: "status.json",
  registryEntry: "registry-entry.json",
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

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, stableJsonStringify(value), "utf8");
  return filePath;
}

function writeText(filePath, text) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${text}\n`, "utf8");
  return filePath;
}

function loadJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeCadence(value) {
  const cadence = normalizeText(value)?.toLowerCase() ?? "delta";
  if (!GRAPHIFY_SCHEDULER_CADENCE_ORDER_V1.includes(cadence)) {
    throw new Error(`Unknown Graphify cadence: ${value}`);
  }
  return cadence;
}

function timestampToken(value = new Date().toISOString()) {
  return String(value).replace(/[:]/g, "-");
}

function relativeWorkspacePath(absPath, workspaceRoot) {
  const resolvedPath = path.resolve(absPath);
  const relative = path.relative(workspaceRoot, resolvedPath);
  return relative.startsWith("..") ? resolvedPath : relative.replace(/\\/g, "/");
}

function safeStringify(value) {
  return stableJsonStringify(value).trimEnd();
}

function makeRetentionPolicy(cadence, generatedAt) {
  const cadenceFocus = cadence === "delta"
    ? "recent material and quick feedback"
    : "older material and broader reorganization";
  return {
    contract: "graphify_scheduler_retention_policy.v1",
    schedulerVersion: SCHEDULER_VERSION,
    cadence,
    cadenceFocus,
    generatedAt,
    truthBoundary: {
      offPath: true,
      belowCorrectionRawAuthority: true,
      replayable: true,
      inspectable: true,
    },
    classes: {
      sourceBundles: {
        role: "replay input",
        retention: "pinned-while-referenced",
        prunePolicy: "vacuum-eligible-only-after-registry-unlinks",
      },
      importSlices: {
        role: "candidate-pack input",
        retention: "pinned-while-referenced",
        prunePolicy: "vacuum-eligible-only-after-registry-unlinks",
      },
      candidatePackInputs: {
        role: "replay seed",
        retention: "pinned-while-referenced",
        prunePolicy: "vacuum-eligible-only-after-registry-unlinks",
      },
      lintDiffOutputs: {
        role: "diagnostic record",
        retention: "pinned-while-referenced",
        prunePolicy: "vacuum-eligible-only-after-registry-unlinks",
      },
    },
    rules: [
      "keep every class while a registry entry still points at it",
      "treat source bundles as replay inputs, not truth layers",
      "treat import slices and candidate-pack inputs as reviewable bridge material",
      "treat lint and maintenance diff outputs as diagnostic records only",
      "vacuum only after registry linkage is removed or replaced",
    ],
  };
}

function renderRetentionPolicyMarkdown(policy) {
  const lines = [
    "# Graphify scheduler retention policy",
    "",
    `- cadence: \`${policy.cadence}\``,
    `- cadence focus: ${policy.cadenceFocus}`,
    `- scheduler version: ${policy.schedulerVersion}`,
    `- off-path: ${policy.truthBoundary.offPath ? "yes" : "no"}`,
    `- below correction/raw authority: ${policy.truthBoundary.belowCorrectionRawAuthority ? "yes" : "no"}`,
    "",
    "## Retained classes",
    "",
    "### source bundles",
    `- role: ${policy.classes.sourceBundles.role}`,
    `- retention: ${policy.classes.sourceBundles.retention}`,
    `- prune policy: ${policy.classes.sourceBundles.prunePolicy}`,
    "",
    "### import slices",
    `- role: ${policy.classes.importSlices.role}`,
    `- retention: ${policy.classes.importSlices.retention}`,
    `- prune policy: ${policy.classes.importSlices.prunePolicy}`,
    "",
    "### candidate-pack inputs",
    `- role: ${policy.classes.candidatePackInputs.role}`,
    `- retention: ${policy.classes.candidatePackInputs.retention}`,
    `- prune policy: ${policy.classes.candidatePackInputs.prunePolicy}`,
    "",
    "### lint / diff outputs",
    `- role: ${policy.classes.lintDiffOutputs.role}`,
    `- retention: ${policy.classes.lintDiffOutputs.retention}`,
    `- prune policy: ${policy.classes.lintDiffOutputs.prunePolicy}`,
    "",
    "## Rules",
    ...policy.rules.map((rule) => `- ${rule}`),
    "",
    "The registry is the authoritative pointer table. Artifacts are removable only after their registry link is gone.",
  ];
  return lines.join("\n");
}

function buildRegistrySkeleton({ generatedAt, repoRoot, workspaceRoot, outputRoot }) {
  return {
    contract: CONTRACT,
    schedulerVersion: SCHEDULER_VERSION,
    generatedAt,
    repoRoot: relativeWorkspacePath(repoRoot, workspaceRoot),
    workspaceRoot: relativeWorkspacePath(workspaceRoot, workspaceRoot),
    outputRoot: relativeWorkspacePath(outputRoot, workspaceRoot),
    runCount: 0,
    latestByCadence: {},
    runs: [],
  };
}

function loadRegistry(registryPath, context) {
  const existing = loadJsonIfExists(registryPath);
  if (existing !== null && existing.contract === CONTRACT) {
    return existing;
  }
  return buildRegistrySkeleton(context);
}

function summarizeArtifactLink(kind, pathValue, hash, extra = {}) {
  return {
    kind,
    path: pathValue,
    hash,
    ...extra,
  };
}

function buildRetentionLinkSet(runRoot, sourceBundle, graphifyRun, compiledPack, importSlice, deterministicLints, maintenanceDiff, retentionPolicy, retentionPolicyPath, retentionPolicyMarkdownPath) {
  return [
    summarizeArtifactLink("source-bundle", sourceBundle.bundleDir, sourceBundle.corpusDigest, {
      bundleId: sourceBundle.bundleId ?? null,
      corpusId: sourceBundle.corpusId ?? null,
      replayable: true,
    }),
    summarizeArtifactLink("graphify-run", graphifyRun.runDir, graphifyRun.graph.hash, {
      graphifyRunId: graphifyRun.runId,
      replayable: true,
    }),
    summarizeArtifactLink("compiled-artifact-pack", compiledPack.outputDir, compiledPack.digest?.bundleHash ?? null, {
      packId: compiledPack.packId ?? null,
      proposalId: compiledPack.proposalId ?? null,
      replayable: true,
    }),
    summarizeArtifactLink("candidate-pack-input", importSlice.paths.candidatePackInput, importSlice.digest?.bundleHash ?? null, {
      sliceId: importSlice.sliceId ?? null,
      proposalId: importSlice.proposalId ?? null,
      replayable: true,
    }),
    summarizeArtifactLink("import-slice", importSlice.outputDir, importSlice.digest?.bundleHash ?? null, {
      sliceId: importSlice.sliceId ?? null,
      replayable: true,
    }),
    summarizeArtifactLink("deterministic-lints", deterministicLints.outputRoot, deterministicLints.report?.bundleManifestHash ?? null, {
      verdict: deterministicLints.verdict?.verdict ?? null,
      replayable: true,
    }),
    summarizeArtifactLink("maintenance-diff", maintenanceDiff.outputDir, maintenanceDiff.digest?.bundleHash ?? null, {
      diffId: maintenanceDiff.report?.diffId ?? null,
      verdict: maintenanceDiff.verdict?.verdict ?? null,
      replayable: true,
    }),
    summarizeArtifactLink("retention-policy-json", retentionPolicyPath, sha256Text(safeStringify(retentionPolicy)), {
      replayable: false,
    }),
    summarizeArtifactLink("retention-policy-markdown", retentionPolicyMarkdownPath, sha256Text(renderRetentionPolicyMarkdown(retentionPolicy)), {
      replayable: false,
    }),
  ];
}

function upsertRegistryRun(registry, entry) {
  const runKey = `${entry.cadence}:${entry.runId}`;
  const nextRuns = (Array.isArray(registry.runs) ? registry.runs : []).filter((candidate) => `${candidate.cadence}:${candidate.runId}` !== runKey);
  nextRuns.push(entry);
  nextRuns.sort((left, right) => {
    const leftKey = `${left.generatedAt ?? ""}:${left.cadence}:${left.runId}`;
    const rightKey = `${right.generatedAt ?? ""}:${right.cadence}:${right.runId}`;
    return leftKey.localeCompare(rightKey);
  });
  registry.runs = nextRuns;
  registry.runCount = nextRuns.length;
  registry.latestByCadence = Object.fromEntries(
    GRAPHIFY_SCHEDULER_CADENCE_ORDER_V1.map((cadence) => {
      const latest = [...nextRuns].reverse().find((candidate) => candidate.cadence === cadence) ?? null;
      return [cadence, latest === null ? null : {
        runId: latest.runId,
        runRoot: latest.runRoot,
        generatedAt: latest.generatedAt,
        status: latest.status,
        maintenanceVerdict: latest.maintenanceDiff?.verdict?.verdict ?? null,
      }];
    }),
  );
  return registry;
}

function renderSummary(result) {
  return [
    "# Graphify scheduler cadence run",
    "",
    `- cadence: ${result.cadence}`,
    `- run id: ${result.runId}`,
    `- generated at: ${result.generatedAt}`,
    `- status: ${result.status}`,
    `- off-path: ${result.offPath ? "yes" : "no"}`,
    `- inspectable: ${result.inspectable ? "yes" : "no"}`,
    `- replayable: ${result.replayable ? "yes" : "no"}`,
    `- truth boundary: ${result.truthBoundary}`,
    `- source bundle: ${result.sourceBundle.bundleDir}`,
    `- graphify run: ${result.graphifyRun.runDir}`,
    `- compiled pack: ${result.compiledPack.outputDir}`,
    `- import slice: ${result.importSlice.outputDir}`,
    `- deterministic lints verdict: ${result.deterministicLints.verdict?.verdict ?? "unknown"}`,
    `- maintenance diff verdict: ${result.maintenanceDiff.verdict?.verdict ?? "unknown"}`,
    `- registry: ${result.registryPath}`,
    `- retention policy: ${result.retentionPolicyPath}`,
    "",
    "This cadence run is intentionally off the serve path. The registry links every downstream artifact so the run can be inspected and replayed without guessing.",
  ].join("\n") + "\n";
}

function buildStatus(result) {
  return {
    contract: CONTRACT,
    schedulerVersion: SCHEDULER_VERSION,
    cadence: result.cadence,
    runId: result.runId,
    generatedAt: result.generatedAt,
    status: result.status,
    offPath: result.offPath,
    inspectable: result.inspectable,
    replayable: result.replayable,
    truthBoundary: {
      offPath: true,
      belowCorrectionRawAuthority: true,
      inspectable: true,
      replayable: true,
    },
    sourceBundle: result.sourceBundle,
    graphifyRun: result.graphifyRun,
    compiledPack: result.compiledPack,
    importSlice: result.importSlice,
    deterministicLints: {
      ok: result.deterministicLints.ok,
      verdict: result.deterministicLints.verdict,
      findings: Array.isArray(result.deterministicLints.report?.findings) ? result.deterministicLints.report.findings.length : 0,
      outputRoot: result.deterministicLints.outputRoot,
    },
    maintenanceDiff: {
      verdict: result.maintenanceDiff.verdict,
      counts: result.maintenanceDiff.report?.counts ?? null,
      outputDir: result.maintenanceDiff.outputDir,
    },
    registryPath: result.registryPath,
    retentionPolicyPath: result.retentionPolicyPath,
    downstreamArtifacts: result.downstreamArtifacts,
    blockers: [],
  };
}

function parseGraphifySchedulerCliArgs(argv, defaultCadence = null) {
  let cadence = defaultCadence;
  let repoRoot = defaultRepoRoot;
  let workspaceRoot = defaultWorkspaceRoot;
  let openclawHome = defaultOpenClawHome;
  let activationRoot = null;
  let outputRoot = null;
  let runId = null;
  let generatedAt = new Date().toISOString();
  let clean = true;
  let json = false;
  let help = false;

  const args = [...argv];
  if (cadence === null && args.length > 0 && !args[0].startsWith("--")) {
    cadence = args.shift();
  }
  if (cadence !== null) {
    cadence = normalizeCadence(cadence);
  }

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }
    if (arg === "--json") {
      json = true;
      continue;
    }
    if (arg === "--repo-root") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--repo-root requires a value");
      repoRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--workspace-root") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--workspace-root requires a value");
      workspaceRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--openclaw-home") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--openclaw-home requires a value");
      openclawHome = next;
      index += 1;
      continue;
    }
    if (arg === "--activation-root") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--activation-root requires a value");
      activationRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--output-root") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--output-root requires a value");
      outputRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--run-id") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--run-id requires a value");
      runId = next;
      index += 1;
      continue;
    }
    if (arg === "--generated-at") {
      const next = args[index + 1];
      if (next === undefined) throw new Error("--generated-at requires a value");
      generatedAt = next;
      index += 1;
      continue;
    }
    if (arg === "--no-clean") {
      clean = false;
      continue;
    }
    if (arg.startsWith("--")) {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return {
    cadence,
    repoRoot: path.resolve(repoRoot),
    workspaceRoot: path.resolve(workspaceRoot),
    openclawHome: path.resolve(openclawHome),
    activationRoot: activationRoot === null ? null : path.resolve(activationRoot),
    outputRoot: outputRoot === null ? null : path.resolve(outputRoot),
    runId: normalizeText(runId),
    generatedAt,
    clean,
    json,
    help,
  };
}

export function writeGraphifySchedulerRun(options = {}) {
  const cadence = normalizeCadence(options.cadence ?? "delta");
  const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
  const workspaceRoot = path.resolve(options.workspaceRoot ?? defaultWorkspaceRoot);
  const openclawHome = path.resolve(options.openclawHome ?? defaultOpenClawHome);
  const activationRoot = options.activationRoot === undefined || options.activationRoot === null
    ? null
    : path.resolve(options.activationRoot);
  const outputRoot = path.resolve(options.outputRoot ?? path.join(workspaceRoot, "artifacts", "graphify-scheduler"));
  const generatedAt = normalizeText(options.generatedAt) ?? new Date().toISOString();
  const runId = normalizeText(options.runId) ?? `${cadence}-${timestampToken(generatedAt)}`;
  const cadenceRoot = path.join(outputRoot, cadence);
  const runRoot = path.join(cadenceRoot, runId);
  const registryPath = path.join(outputRoot, GRAPHIFY_SCHEDULER_LAYOUT_V1.registry);
  const retentionPolicyPath = path.join(runRoot, GRAPHIFY_SCHEDULER_LAYOUT_V1.retentionPolicy);
  const retentionPolicyMarkdownPath = path.join(runRoot, GRAPHIFY_SCHEDULER_LAYOUT_V1.retentionPolicyMarkdown);
  const summaryPath = path.join(runRoot, GRAPHIFY_SCHEDULER_LAYOUT_V1.summary);
  const statusPath = path.join(runRoot, GRAPHIFY_SCHEDULER_LAYOUT_V1.status);
  const registryEntryPath = path.join(runRoot, GRAPHIFY_SCHEDULER_LAYOUT_V1.registryEntry);

  if (options.clean !== false) {
    rmSync(runRoot, { recursive: true, force: true });
  }
  ensureDir(runRoot);
  ensureDir(outputRoot);

  const sourceBundle = exportGraphifySourceBundle({
    openclawHome,
    ...(activationRoot === null ? {} : { activationRoot }),
    outputDir: path.join(runRoot, "source-bundle"),
    generatedAt,
    observedAt: generatedAt,
    ...(options.profileRoots === undefined ? {} : { profileRoots: options.profileRoots }),
    ...(options.homeDir === undefined ? {} : { homeDir: options.homeDir }),
  });
  if (!sourceBundle.ok) {
    throw new Error(`Graphify scheduler source bundle export failed: ${sourceBundle.error ?? "unknown error"}`);
  }

  const graphifyRun = runManagedGraphifyRunner({
    sourceBundlePath: sourceBundle.bundleDir,
    outputRoot: path.join(runRoot, "run"),
    runId: `${cadence}-run`,
    graphifyVersion: SCHEDULER_VERSION,
    graphifyMode: `${cadence}-cadence`,
    graphifyConfig: {
      cadence,
      schedulerVersion: SCHEDULER_VERSION,
      sourceBundleDigest: sourceBundle.corpusDigest,
      sourceBundleId: sourceBundle.bundleId ?? null,
    },
    graphifyFlags: ["off-path", "inspectable", "replayable", `cadence:${cadence}`],
    labels: ["graphify-scheduler", `cadence:${cadence}`, "off-path", "inspectable", "replayable"],
  });
  if (!graphifyRun.ok) {
    throw new Error(`Graphify scheduler managed run failed: ${graphifyRun.execution?.failure ?? "unknown error"}`);
  }

  const compiledPack = exportGraphifyCompiledArtifactsPack({
    bundleStartedAt: generatedAt,
    bundleId: `${cadence}-${runId}`,
    outputDir: path.join(runRoot, "compiled"),
    proposalId: `prop_graphify_scheduler_${cadence}_${runId}`,
    packId: `pack_graphify_scheduler_${cadence}_${runId}`,
    graphifyRunId: graphifyRun.runId,
    graphifyVersion: SCHEDULER_VERSION,
    graphifyCommand: `graphify scheduler ${cadence}`,
    sourceBundleId: sourceBundle.bundleId ?? sourceBundle.corpusId ?? null,
    sourceBundleHash: sourceBundle.corpusDigest ?? null,
    graphHash: graphifyRun.graph.hash,
    configHash: graphifyRun.graphifyConfigHash,
    labelsHash: sha256Text(safeStringify(graphifyRun.labels)),
    sourceDocs: [
      "docs/architecture/graphify-bridge.md",
      "docs/architecture/graphify-scheduler.md",
      "docs/architecture/compiled-artifacts.md",
    ],
    sourceFixtures: [
      path.relative(repoRoot, sourceBundle.outputPaths?.corpusManifest ?? sourceBundle.bundleDir),
      path.relative(repoRoot, sourceBundle.outputPaths?.normalizedEventExport ?? sourceBundle.bundleDir),
    ],
  });
  if (!compiledPack.ok) {
    throw new Error(`Graphify scheduler compiled-artifact pack failed: ${compiledPack.error ?? "unknown error"}`);
  }

  const importSlice = exportGraphifyImportSlice({
    bundleRoot: compiledPack.outputDir,
    outputRoot: path.join(runRoot, "import"),
    runId: `${cadence}-import`,
    repoRoot,
    workspaceRoot,
  });
  if (!importSlice.ok) {
    throw new Error(`Graphify scheduler import slice failed: ${importSlice.error ?? "unknown error"}`);
  }

  const deterministicLints = runGraphifyDeterministicLints({
    bundleRoot: compiledPack.outputDir,
    repoRoot,
    workspaceRoot,
    outputRoot: path.join(runRoot, "lints"),
    runId: `${cadence}-lints`,
  });

  const maintenanceDiff = buildGraphifyMaintenanceDiffBundle({
    graphifyRoot: runRoot,
    ocbRoot: repoRoot,
    repoRoot,
    workspaceRoot,
    outputRoot: path.join(runRoot, "maintenance"),
    runId: `${cadence}-maintenance`,
  });
  writeGraphifyMaintenanceDiffBundle(maintenanceDiff.outputDir, maintenanceDiff);

  const retentionPolicy = makeRetentionPolicy(cadence, generatedAt);
  const downstreamArtifacts = buildRetentionLinkSet(
    runRoot,
    sourceBundle,
    graphifyRun,
    compiledPack,
    importSlice,
    deterministicLints,
    maintenanceDiff,
    retentionPolicy,
    retentionPolicyPath,
    retentionPolicyMarkdownPath,
  );
  const registryEntry = {
    contract: CONTRACT,
    schedulerVersion: SCHEDULER_VERSION,
    cadence,
    runId,
    generatedAt,
    runRoot: relativeWorkspacePath(runRoot, workspaceRoot),
    offPath: true,
    inspectable: true,
    replayable: true,
    truthBoundary: {
      offPath: true,
      belowCorrectionRawAuthority: true,
      inspectable: true,
      replayable: true,
    },
    sourceBundle: {
      bundleDir: relativeWorkspacePath(sourceBundle.bundleDir ?? "", workspaceRoot),
      bundleId: sourceBundle.bundleId ?? null,
      corpusId: sourceBundle.corpusId ?? null,
      corpusDigest: sourceBundle.corpusDigest ?? null,
      outputPaths: sourceBundle.outputPaths ?? null,
    },
    graphifyRun: {
      runId: graphifyRun.runId,
      runDir: relativeWorkspacePath(graphifyRun.runDir, workspaceRoot),
      sourceBundleHash: graphifyRun.sourceBundleHash,
      graph: graphifyRun.graph,
      graphifyVersion: graphifyRun.graphifyVersion,
      graphifyMode: graphifyRun.graphifyMode,
      graphifyConfigHash: graphifyRun.graphifyConfigHash,
      graphifyFlags: graphifyRun.graphifyFlags,
      execution: {
        state: graphifyRun.execution.state,
        exitCode: graphifyRun.execution.exitCode,
        failure: graphifyRun.execution.failure,
      },
      outputs: graphifyRun.outputs,
    },
    compiledPack: {
      outputDir: relativeWorkspacePath(compiledPack.outputDir, workspaceRoot),
      bundleId: compiledPack.bundleId ?? null,
      packId: compiledPack.packId ?? null,
      proposalId: compiledPack.proposalId ?? null,
      digest: compiledPack.digest ?? null,
      validation: compiledPack.validation ?? null,
    },
    importSlice: {
      outputDir: relativeWorkspacePath(importSlice.outputDir, workspaceRoot),
      sliceId: importSlice.sliceId ?? null,
      proposalId: importSlice.proposalId ?? null,
      rollbackKey: importSlice.rollbackKey ?? null,
      digest: importSlice.digest ?? null,
      truthBoundary: importSlice.truthBoundary ?? null,
      paths: importSlice.paths ?? null,
    },
    deterministicLints: {
      ok: deterministicLints.ok,
      runId: deterministicLints.runId,
      outputRoot: relativeWorkspacePath(deterministicLints.outputRoot, workspaceRoot),
      bundleRoot: relativeWorkspacePath(deterministicLints.bundleRoot, workspaceRoot),
      report: deterministicLints.report ?? null,
      verdict: deterministicLints.verdict ?? null,
      paths: deterministicLints.paths ?? null,
      summary: deterministicLints.summary ?? null,
    },
    maintenanceDiff: {
      outputDir: relativeWorkspacePath(maintenanceDiff.outputDir, workspaceRoot),
      report: maintenanceDiff.report ?? null,
      proposalSuggestion: maintenanceDiff.proposalSuggestion ?? null,
      verdict: maintenanceDiff.verdict ?? null,
      digest: maintenanceDiff.digest ?? null,
      paths: maintenanceDiff.paths ?? null,
    },
    downstreamArtifacts,
    retentionPolicy: {
      path: relativeWorkspacePath(retentionPolicyPath, workspaceRoot),
      markdownPath: relativeWorkspacePath(retentionPolicyMarkdownPath, workspaceRoot),
      contract: retentionPolicy.contract,
      cadence: retentionPolicy.cadence,
    },
  };

  const registry = loadRegistry(registryPath, {
    generatedAt,
    repoRoot,
    workspaceRoot,
    outputRoot,
  });
  upsertRegistryRun(registry, registryEntry);
  writeJson(registryPath, registry);
  writeJson(registryEntryPath, registryEntry);
  writeJson(retentionPolicyPath, retentionPolicy);
  writeText(retentionPolicyMarkdownPath, renderRetentionPolicyMarkdown(retentionPolicy));

  const result = {
    contract: CONTRACT,
    schedulerVersion: SCHEDULER_VERSION,
    cadence,
    runId,
    generatedAt,
    status: "completed",
    offPath: true,
    inspectable: true,
    replayable: true,
    truthBoundary: "below correction/raw-authority truth",
    sourceBundle,
    graphifyRun,
    compiledPack,
    importSlice,
    deterministicLints,
    maintenanceDiff,
    registry,
    registryPath,
    registryEntryPath,
    retentionPolicy,
    retentionPolicyPath,
    retentionPolicyMarkdownPath,
    downstreamArtifacts,
    runRoot,
    outputRoot,
  };

  writeText(summaryPath, renderSummary(result));
  writeJson(statusPath, buildStatus(result));

  return {
    ...result,
    summaryPath,
    statusPath,
  };
}

export function runGraphifySchedulerCli(defaultCadence = null, argv = process.argv.slice(2)) {
  const parsed = parseGraphifySchedulerCliArgs(argv, defaultCadence);
  if (parsed.help || parsed.cadence === null) {
    process.stdout.write([
      "Usage:",
      "  node scripts/graphify-scheduler.mjs <delta|reorg> [options]",
      "  node scripts/graphify-delta-cadence.mjs [options]",
      "  node scripts/graphify-reorg-cadence.mjs [options]",
      "",
      "Options:",
      `  --repo-root <path>        Repo root (default: ${defaultRepoRoot})`,
      `  --workspace-root <path>   Workspace root (default: ${defaultWorkspaceRoot})`,
      `  --openclaw-home <path>    OpenClaw home used to build the source bundle (default: ${defaultOpenClawHome})`,
      "  --activation-root <path>  Activation root used to build the source bundle",
      `  --output-root <path>      Scheduler output root (default: <workspace>/artifacts/graphify-scheduler)`,
      "  --run-id <id>            Run identifier (default: cadence + timestamp token)",
      "  --generated-at <iso>     Timestamp used across the cadence bundle",
      "  --no-clean               Keep an existing run-root if present",
      "  --json                   Emit machine-readable JSON",
      "  --help                   Show this help",
      "",
      "The scheduler remains off the serve path and keeps the registry linked to downstream artifacts.",
    ].join("\n") + "\n");
    return 0;
  }

  const result = writeGraphifySchedulerRun(parsed);
  if (parsed.json) {
    process.stdout.write(`${JSON.stringify({
      contract: result.contract,
      schedulerVersion: result.schedulerVersion,
      cadence: result.cadence,
      runId: result.runId,
      generatedAt: result.generatedAt,
      status: result.status,
      offPath: result.offPath,
      inspectable: result.inspectable,
      replayable: result.replayable,
      registryPath: result.registryPath,
      registryEntryPath: result.registryEntryPath,
      retentionPolicyPath: result.retentionPolicyPath,
      summaryPath: result.summaryPath,
      statusPath: result.statusPath,
      downstreamArtifacts: result.downstreamArtifacts,
    }, null, 2)}\n`);
  }
  else {
    process.stdout.write([
      `Graphify scheduler ${result.cadence} cadence completed`,
      `summary: ${result.summaryPath}`,
      `status: ${result.statusPath}`,
      `registry: ${result.registryPath}`,
      `retention: ${result.retentionPolicyPath}`,
    ].join("\n") + "\n");
  }
  return 0;
}

function main(argv = process.argv.slice(2)) {
  runGraphifySchedulerCli(null, argv);
}

const isMainModule = process.argv[1]
  ? pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url
  : false;

if (isMainModule) {
  try {
    main();
  }
  catch (error) {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  }
}
