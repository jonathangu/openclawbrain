import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { describeStablePathTree } from "./import-export.js";

export const GRAPHIFY_RUN_BUNDLE_LAYOUT = {
  command: "graphify-command.json",
  graph: "graph.json",
  html: "graph.html",
  report: "GRAPH_REPORT.md",
  summary: "graphify-summary.json",
  benchmark: "benchmark.json",
  labels: "labels.json",
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
  return JSON.stringify(canonicalizeJsonValue(value), null, 2) + "\n";
}

function writeJsonFile(filePath, value) {
  writeFileSync(filePath, stableJsonStringify(value), "utf8");
}

function shortHash(value, length = 12) {
  return value.slice(0, length);
}

function normalizeOptionalString(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function loadJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  }
  catch {
    return null;
  }
}

function loadSourceBundleManifest(sourceBundlePath) {
  const resolvedPath = path.resolve(sourceBundlePath);
  const stats = statSync(resolvedPath);
  if (stats.isFile()) {
    const data = loadJsonIfExists(resolvedPath);
    return data === null ? null : { path: resolvedPath, data };
  }
  const candidateFiles = [
    "corpus-manifest.json",
    "manifest.json",
    "normalized-event-export.json",
    "workspace-metadata.json",
  ];
  for (const candidate of candidateFiles) {
    const candidatePath = path.join(resolvedPath, candidate);
    const data = loadJsonIfExists(candidatePath);
    if (data !== null) {
      return { path: candidatePath, data };
    }
  }
  return null;
}

function extractLabelsFromManifest(manifest) {
  if (manifest === null || manifest === undefined) {
    return [];
  }
  const labels = new Set();
  const pushLabel = (value) => {
    if (typeof value !== "string") {
      return;
    }
    const normalized = value.trim();
    if (normalized.length > 0) {
      labels.add(normalized);
    }
  };
  const pushLabelList = (value) => {
    if (!Array.isArray(value)) {
      return;
    }
    for (const item of value) {
      pushLabel(item);
    }
  };
  pushLabel(manifest.label);
  pushLabel(manifest.name);
  pushLabel(manifest.contract);
  pushLabelList(manifest.labels);
  pushLabelList(manifest.tags);
  pushLabelList(manifest.topics);
  pushLabelList(manifest.sourceLabels);
  pushLabelList(manifest.bundleLabels);
  pushLabelList(manifest?.provenance?.labels);
  pushLabelList(manifest?.metadata?.labels);
  return [...labels].sort((left, right) => left.localeCompare(right));
}

function buildGraphifyLabels(input) {
  const labels = new Set([
    "graphify",
    "managed-run",
    `mode:${input.mode}`,
    `version:${input.version}`,
    `source-hash:${shortHash(input.sourceBundleHash)}`,
  ]);
  for (const label of input.sourceLabels ?? []) {
    labels.add(label);
  }
  for (const label of input.requestedLabels ?? []) {
    labels.add(label);
  }
  for (const flag of input.flags ?? []) {
    labels.add(`flag:${flag}`);
  }
  return [...labels].sort((left, right) => left.localeCompare(right));
}

function deriveGraphifyRunId(input) {
  const digest = createHash("sha256");
  digest.update(input.sourceBundleHash);
  digest.update("\u0000");
  digest.update(input.version);
  digest.update("\u0000");
  digest.update(input.mode);
  digest.update("\u0000");
  digest.update(stableJsonStringify(input.config));
  digest.update("\u0000");
  for (const label of input.labels ?? []) {
    digest.update(label);
    digest.update("\u0000");
  }
  for (const flag of input.flags ?? []) {
    digest.update(flag);
    digest.update("\u0000");
  }
  return `graphify-${digest.digest("hex").slice(0, 16)}`;
}

function getNodePath(entryPath) {
  return entryPath === "." ? "" : entryPath;
}

function buildGraphPayload(input) {
  const treeEntries = input.sourceTree.entries;
  const nodes = [
    {
      id: "source-bundle",
      kind: input.sourceTree.kind,
      path: ".",
      label: path.basename(input.sourceBundlePath),
      hash: input.sourceBundleHash,
      fileCount: input.sourceTree.fileCount,
      directoryCount: input.sourceTree.directoryCount,
      symlinkCount: input.sourceTree.symlinkCount,
      totalBytes: input.sourceTree.totalBytes,
    },
  ];
  const edges = [];
  const nodeIdsByPath = new Map([["", "source-bundle"]]);
  for (const entry of treeEntries) {
    const normalizedPath = getNodePath(entry.path);
    if (entry.kind === "directory" && normalizedPath === "") {
      continue;
    }
    const nodeId = `${entry.kind}:${normalizedPath}`;
    const node = {
      id: nodeId,
      kind: entry.kind,
      path: normalizedPath,
      label: normalizedPath === "" ? path.basename(input.sourceBundlePath) : path.posix.basename(normalizedPath),
    };
    if (entry.kind === "file") {
      node.hash = entry.hash;
      node.size = entry.size;
      const ext = path.posix.extname(normalizedPath);
      node.extension = ext.length > 0 ? ext : null;
    }
    if (entry.kind === "symlink") {
      node.target = entry.target;
    }
    nodes.push(node);
    nodeIdsByPath.set(normalizedPath, nodeId);
    const parentPath = normalizedPath.length === 0 ? "" : path.posix.dirname(normalizedPath);
    const parentId = nodeIdsByPath.get(parentPath === "." ? "" : parentPath) ?? "source-bundle";
    edges.push({
      from: parentId,
      to: nodeId,
      kind: "contains",
    });
  }
  const graph = {
    contract: "graphify_graph.v1",
    runId: input.runId,
    sourceBundleHash: input.sourceBundleHash,
    graphify: {
      version: input.version,
      mode: input.mode,
      config: canonicalizeJsonValue(input.config),
      flags: input.flags,
    },
    sourceBundle: {
      path: input.sourceBundlePath,
      kind: input.sourceTree.kind,
      fileCount: input.sourceTree.fileCount,
      directoryCount: input.sourceTree.directoryCount,
      symlinkCount: input.sourceTree.symlinkCount,
      totalBytes: input.sourceTree.totalBytes,
    },
    nodes,
    edges,
  };
  const graphHash = createHash("sha256").update(stableJsonStringify(graph)).digest("hex");
  graph.graphHash = graphHash;
  return { graph, graphHash };
}

function summarizeGraphPayload(graphPayload) {
  const nodes = Array.isArray(graphPayload?.nodes) ? graphPayload.nodes : [];
  const edges = Array.isArray(graphPayload?.edges) ? graphPayload.edges : [];
  const nodeKinds = new Map();
  for (const node of nodes) {
    const kind = typeof node?.kind === "string" ? node.kind : "unknown";
    nodeKinds.set(kind, (nodeKinds.get(kind) ?? 0) + 1);
  }
  return {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    nodeKinds: Object.fromEntries([...nodeKinds.entries()].sort((left, right) => left[0].localeCompare(right[0]))),
    graphHash: typeof graphPayload?.graphHash === "string" ? graphPayload.graphHash : null,
  };
}

function buildGraphHtml(input) {
  const json = stableJsonStringify(input.graph);
  return [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="utf-8" />',
    `  <title>Graphify run ${input.runId}</title>`,
    '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
    '  <style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45;margin:24px;max-width:1100px} pre{background:#f6f8fa;padding:16px;overflow:auto;border-radius:8px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px 10px;text-align:left}</style>',
    "</head>",
    "<body>",
    `  <h1>Graphify run ${input.runId}</h1>`,
    '  <p><strong>Managed off-path bundle.</strong> This HTML is reproducible from the source bundle hash and recorded Graphify metadata.</p>',
    '  <table>',
    `    <tr><th>Source bundle hash</th><td>${input.sourceBundleHash}</td></tr>`,
    `    <tr><th>Graphify version</th><td>${input.version}</td></tr>`,
    `    <tr><th>Mode</th><td>${input.mode}</td></tr>`,
    `    <tr><th>Config hash</th><td>${input.configHash}</td></tr>`,
    `    <tr><th>Graph hash</th><td>${input.graphHash}</td></tr>`,
    `    <tr><th>Nodes</th><td>${input.summary.nodeCount}</td></tr>`,
    `    <tr><th>Edges</th><td>${input.summary.edgeCount}</td></tr>`,
    "  </table>",
    '  <h2>Graph JSON</h2>',
    `  <script type="application/json" id="graph-json">${json.replace(/</g, "\\u003c")}</script>`,
    "  <pre id=\"graph-pre\"></pre>",
    "  <script>",
    "    document.getElementById('graph-pre').textContent = document.getElementById('graph-json').textContent;",
    "  </script>",
    "</body>",
    "</html>",
  ].join("\n") + "\n";
}

function buildGraphReport(input) {
  const lines = [
    `# Graphify run ${input.runId}`,
    "",
    "## Managed execution summary",
    `- Source bundle: ${input.sourceBundlePath}`,
    `- Source bundle hash: ${input.sourceBundleHash}`,
    `- Graphify version: ${input.version}`,
    `- Mode: ${input.mode}`,
    `- Config hash: ${input.configHash}`,
    `- Flags: ${input.flags.length === 0 ? "none" : input.flags.join(", ")}`,
    `- Execution state: ${input.execution.state}`,
    `- Execution exit code: ${input.execution.exitCode === null ? "none" : input.execution.exitCode}`,
    `- Execution failure: ${input.execution.failure === null ? "none" : input.execution.failure}`,
    "",
    "## Structural summary",
    `- Files: ${input.sourceTree.fileCount}`,
    `- Directories: ${input.sourceTree.directoryCount}`,
    `- Symlinks: ${input.sourceTree.symlinkCount}`,
    `- Bytes: ${input.sourceTree.totalBytes}`,
    `- Nodes: ${input.summary.nodeCount}`,
    `- Edges: ${input.summary.edgeCount}`,
    `- Graph hash: ${input.graphHash}`,
    "",
    "## Labels",
    input.labels.length === 0 ? "- none" : input.labels.map((label) => `- ${label}`).join("\n"),
  ];
  return lines.join("\n") + "\n";
}

function buildBenchmarkRecord(input) {
  return {
    contract: "graphify_benchmark.v1",
    runId: input.runId,
    sourceBundleHash: input.sourceBundleHash,
    graphifyVersion: input.version,
    graphifyMode: input.mode,
    configHash: input.configHash,
    execution: {
      state: input.execution.state,
      exitCode: input.execution.exitCode,
      durationMs: input.execution.durationMs,
    },
    hashMs: input.hashMs,
    synthesisMs: input.synthesisMs,
    writeMs: input.writeMs,
    totalMs: input.totalMs,
  };
}

function buildCommandRecord(input) {
  return {
    contract: "graphify_command_record.v1",
    runId: input.runId,
    sourceBundle: {
      path: input.sourceBundlePath,
      kind: input.sourceTree.kind,
      hash: input.sourceBundleHash,
      fileCount: input.sourceTree.fileCount,
      directoryCount: input.sourceTree.directoryCount,
      symlinkCount: input.sourceTree.symlinkCount,
      totalBytes: input.sourceTree.totalBytes,
      manifest: input.manifest,
    },
    graphify: {
      version: input.version,
      versionSource: input.versionSource,
      mode: input.mode,
      config: canonicalizeJsonValue(input.config),
      configHash: input.configHash,
      flags: input.flags,
      command: input.command === null ? null : {
        executable: input.command.executable,
        args: input.command.args,
      },
    },
    execution: {
      state: input.execution.state,
      command: input.execution.command,
      args: input.execution.args,
      cwd: input.execution.cwd,
      startedAt: input.execution.startedAt,
      finishedAt: input.execution.finishedAt,
      durationMs: input.execution.durationMs,
      exitCode: input.execution.exitCode,
      signal: input.execution.signal,
      stdout: input.execution.stdout,
      stderr: input.execution.stderr,
      failure: input.execution.failure,
    },
    outputs: input.outputs,
    reproducibility: {
      runId: input.runId,
      sourceBundleHash: input.sourceBundleHash,
      graphifyVersion: input.version,
      graphifyMode: input.mode,
      configHash: input.configHash,
      outputHash: input.graphHash,
    },
  };
}

function probeGraphifyVersion(command, flags) {
  if (command === null) {
    return { version: "unknown", versionSource: "unavailable" };
  }
  const probe = spawnSync(command.executable, ["--version", ...flags], {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: "pipe",
  });
  const stdout = typeof probe.stdout === "string" ? probe.stdout.trim() : "";
  const stderr = typeof probe.stderr === "string" ? probe.stderr.trim() : "";
  const combined = [stdout, stderr].filter((value) => value.length > 0).join("\n").trim();
  if (probe.error || probe.status !== 0) {
    return {
      version: normalizeOptionalString(combined) ?? "unknown",
      versionSource: probe.error ? `probe-error:${probe.error.message}` : `probe-exit:${probe.status}`,
    };
  }
  return {
    version: normalizeOptionalString(combined) ?? "unknown",
    versionSource: "probed",
  };
}

function runOptionalManagedCommand(input) {
  if (input.command === null) {
    return {
      state: "synthesized",
      command: null,
      args: [],
      cwd: input.cwd,
      startedAt: null,
      finishedAt: null,
      durationMs: 0,
      exitCode: 0,
      signal: null,
      stdout: "",
      stderr: "",
      failure: null,
    };
  }
  const startedAt = new Date().toISOString();
  const startedMs = Date.now();
  const result = spawnSync(input.command.executable, input.command.args, {
    cwd: input.cwd,
    encoding: "utf8",
    stdio: "pipe",
    env: {
      ...process.env,
      GRAPHIFY_RUN_DIR: input.cwd,
      GRAPHIFY_SOURCE_BUNDLE: input.sourceBundlePath,
      GRAPHIFY_SOURCE_BUNDLE_HASH: input.sourceBundleHash,
      GRAPHIFY_GRAPH_JSON: input.outputs.graph,
      GRAPHIFY_GRAPH_HTML: input.outputs.html,
      GRAPHIFY_GRAPH_REPORT: input.outputs.report,
      GRAPHIFY_GRAPHIFY_SUMMARY: input.outputs.summary,
      GRAPHIFY_BENCHMARK: input.outputs.benchmark,
      GRAPHIFY_LABELS: input.outputs.labels,
      GRAPHIFY_VERSION: input.version,
      GRAPHIFY_MODE: input.mode,
      GRAPHIFY_CONFIG_JSON: stableJsonStringify(input.config),
      GRAPHIFY_FLAGS_JSON: stableJsonStringify(input.flags),
      GRAPHIFY_RUN_ID: input.runId,
    },
  });
  const finishedAt = new Date().toISOString();
  const stdout = typeof result.stdout === "string" ? result.stdout : "";
  const stderr = typeof result.stderr === "string" ? result.stderr : "";
  const failure = result.error || result.status !== 0
    ? [
        result.error instanceof Error ? result.error.message : null,
        typeof result.status === "number" && result.status !== 0 ? `exitCode=${result.status}` : null,
        typeof result.signal === "string" && result.signal.length > 0 ? `signal=${result.signal}` : null,
        stderr.trim().length > 0 ? stderr.trim() : null,
      ].filter((value) => value !== null).join(" | ")
    : null;
  return {
    state: result.error || result.status !== 0 ? "failed" : "executed",
    command: input.command.executable,
    args: input.command.args,
    cwd: input.cwd,
    startedAt,
    finishedAt,
    durationMs: Date.now() - startedMs,
    exitCode: typeof result.status === "number" ? result.status : null,
    signal: typeof result.signal === "string" ? result.signal : null,
    stdout,
    stderr,
    failure,
  };
}

function ensureRunDirectory(runDir) {
  rmSync(runDir, { recursive: true, force: true });
  mkdirSync(runDir, { recursive: true });
}

export function runManagedGraphifyRunner(options) {
  const startedAt = new Date().toISOString();
  const startedMs = Date.now();
  const sourceBundlePath = path.resolve(options.sourceBundlePath);
  const outputRoot = path.resolve(options.outputRoot ?? path.join("artifacts", "graphify-runs"));
  if (!existsSync(sourceBundlePath)) {
    throw new Error(`Source bundle does not exist: ${sourceBundlePath}`);
  }
  const sourceTreeStartMs = Date.now();
  const sourceTree = describeStablePathTree(sourceBundlePath);
  const hashMs = Date.now() - sourceTreeStartMs;
  const manifest = loadSourceBundleManifest(sourceBundlePath);
  const manifestLabels = manifest === null ? [] : extractLabelsFromManifest(manifest.data);
  const config = canonicalizeJsonValue(options.graphifyConfig ?? {});
  const flags = Array.isArray(options.graphifyFlags) ? options.graphifyFlags.filter((flag) => typeof flag === "string" && flag.trim().length > 0) : [];
  const mode = normalizeOptionalString(options.graphifyMode) ?? "managed-off-path";
  const command = normalizeOptionalString(options.graphifyCommand);
  const commandArgs = Array.isArray(options.graphifyArgs) ? options.graphifyArgs.filter((arg) => typeof arg === "string") : [];
  const explicitVersion = normalizeOptionalString(options.graphifyVersion);
  const commandObject = command === null ? null : {
    executable: command,
    args: commandArgs,
  };
  const versionProbe = explicitVersion === null ? probeGraphifyVersion(commandObject, []) : { version: explicitVersion, versionSource: "provided" };
  const version = versionProbe.version;
  const versionSource = versionProbe.versionSource;
  const sourceBundleHash = sourceTree.hash;
  const configHash = createHash("sha256").update(stableJsonStringify(config)).digest("hex");
  const runId = normalizeOptionalString(options.runId) ?? deriveGraphifyRunId({
    sourceBundleHash,
    version,
    mode,
    config,
    labels: Array.isArray(options.labels) ? options.labels.filter((label) => typeof label === "string" && label.trim().length > 0) : [],
    flags,
  });
  const runDir = path.join(outputRoot, runId);
  ensureRunDirectory(runDir);
  const outputPaths = {
    command: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.command),
    graph: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.graph),
    html: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.html),
    report: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.report),
    summary: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.summary),
    benchmark: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.benchmark),
    labels: path.join(runDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.labels),
  };
  const executionStartMs = Date.now();
  const execution = runOptionalManagedCommand({
    command: commandObject,
    cwd: runDir,
    sourceBundlePath,
    sourceBundleHash,
    version,
    mode,
    config,
    runId,
    outputs: outputPaths,
  });
  const executionDurationMs = Date.now() - executionStartMs;
  const graphPayloadStartMs = Date.now();
  const graphPayload = buildGraphPayload({
    runId,
    sourceBundlePath,
    sourceBundleHash,
    sourceTree,
    version,
    mode,
    config,
    flags,
  });
  const labels = buildGraphifyLabels({
    sourceBundleHash,
    version,
    mode,
    flags,
    sourceLabels: manifestLabels,
    requestedLabels: Array.isArray(options.labels) ? options.labels.filter((label) => typeof label === "string" && label.trim().length > 0) : [],
  });
  const graphHash = graphPayload.graphHash;
  const graphSummary = summarizeGraphPayload(graphPayload.graph);
  const runSummary = {
    contract: "graphify_run_summary.v1",
    runId,
    sourceBundleHash,
    graphifyVersion: version,
    graphifyVersionSource: versionSource,
    graphifyMode: mode,
    graphifyConfigHash: configHash,
    graphifyFlags: flags,
    sourceBundle: {
      path: sourceBundlePath,
      kind: sourceTree.kind,
      fileCount: sourceTree.fileCount,
      directoryCount: sourceTree.directoryCount,
      symlinkCount: sourceTree.symlinkCount,
      totalBytes: sourceTree.totalBytes,
      labels: manifestLabels,
    },
    manifest: manifest === null ? null : manifest.path,
    graph: graphSummary,
    labels,
    execution: {
      state: execution.state,
      exitCode: execution.exitCode,
      failure: execution.failure,
    },
    outputs: {
      command: GRAPHIFY_RUN_BUNDLE_LAYOUT.command,
      graph: GRAPHIFY_RUN_BUNDLE_LAYOUT.graph,
      html: GRAPHIFY_RUN_BUNDLE_LAYOUT.html,
      report: GRAPHIFY_RUN_BUNDLE_LAYOUT.report,
      summary: GRAPHIFY_RUN_BUNDLE_LAYOUT.summary,
      benchmark: GRAPHIFY_RUN_BUNDLE_LAYOUT.benchmark,
      labels: GRAPHIFY_RUN_BUNDLE_LAYOUT.labels,
    },
    graphHash,
  };
  const graphWriteStartMs = Date.now();
  writeJsonFile(outputPaths.graph, graphPayload.graph);
  writeFileSync(outputPaths.html, buildGraphHtml({
    runId,
    sourceBundleHash,
    version,
    mode,
    configHash,
    graphHash,
    graph: graphPayload.graph,
    summary: graphSummary,
  }), "utf8");
  writeJsonFile(outputPaths.labels, {
    contract: "graphify_labels.v1",
    runId,
    sourceBundleHash,
    graphifyVersion: version,
    graphifyMode: mode,
    configHash,
    labels,
    manifestLabels,
  });
  const report = buildGraphReport({
    runId,
    sourceBundlePath,
    sourceBundleHash,
    version,
    mode,
    configHash,
    flags,
    execution,
    sourceTree,
    summary: graphSummary,
    graphHash,
    labels,
  });
  writeFileSync(outputPaths.report, report, "utf8");
  writeJsonFile(outputPaths.summary, runSummary);
  const benchmark = buildBenchmarkRecord({
    runId,
    sourceBundleHash,
    version,
    mode,
    configHash,
    execution: {
      state: execution.state,
      exitCode: execution.exitCode,
      durationMs: executionDurationMs,
    },
    hashMs,
    synthesisMs: Date.now() - graphPayloadStartMs,
    writeMs: Date.now() - graphWriteStartMs,
    totalMs: Date.now() - startedMs,
  });
  writeJsonFile(outputPaths.benchmark, benchmark);
  const commandRecord = buildCommandRecord({
    runId,
    sourceBundlePath,
    sourceTree,
    sourceBundleHash,
    manifest,
    version,
    versionSource,
    mode,
    config,
    configHash,
    flags,
    command: commandObject,
    execution: {
      ...execution,
      startedAt,
      finishedAt: new Date().toISOString(),
    },
    outputs: outputPaths,
    graphHash,
  });
  writeJsonFile(outputPaths.command, commandRecord);
  const finishedAt = new Date().toISOString();
  const result = {
    ok: true,
    runId,
    outputRoot,
    runDir,
    sourceBundlePath,
    sourceBundleHash,
    sourceTree,
    graphifyVersion: version,
    graphifyVersionSource: versionSource,
    graphifyMode: mode,
    graphifyConfig: config,
    graphifyConfigHash: configHash,
    graphifyFlags: flags,
    labels,
    manifest,
    command: commandObject,
    execution: {
      ...execution,
      startedAt,
      finishedAt,
    },
    graph: {
      path: outputPaths.graph,
      hash: graphHash,
      nodeCount: graphSummary.nodeCount,
      edgeCount: graphSummary.edgeCount,
      nodeKinds: graphSummary.nodeKinds,
    },
    outputs: outputPaths,
    benchmark,
    startedAt,
    finishedAt,
    durationMs: Date.now() - startedMs,
  };
  return result;
}

export function describeGraphifyRunBundle(runDir) {
  const resolvedRunDir = path.resolve(runDir);
  const commandPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.command);
  const graphPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.graph);
  const htmlPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.html);
  const reportPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.report);
  const summaryPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.summary);
  const benchmarkPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.benchmark);
  const labelsPath = path.join(resolvedRunDir, GRAPHIFY_RUN_BUNDLE_LAYOUT.labels);
  return {
    runDir: resolvedRunDir,
    command: loadJsonIfExists(commandPath),
    graph: loadJsonIfExists(graphPath),
    html: existsSync(htmlPath),
    report: existsSync(reportPath),
    summary: loadJsonIfExists(summaryPath),
    benchmark: loadJsonIfExists(benchmarkPath),
    labels: loadJsonIfExists(labelsPath),
  };
}
