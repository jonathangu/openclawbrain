#!/usr/bin/env tsx

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import type { RecordedSessionTraceV1 } from "../packages/cli/dist/src/index.js";
import { writeRecordedSessionReplayProofLane } from "../src/replay-proof-lane.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT = "canonical_recorded_session_trace_set_manifest.v1";
const DEFAULT_RECORDED_SESSION_REPLAY_TRACE_MANIFEST_PATH = path.resolve(
  repoRoot,
  "evals",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);

interface ParsedArgs {
  tracePaths: string[];
  traceManifestPath: string | null;
  artifactRoot: string | null;
  workedTraceLimit: number | null;
  assumptions: string[];
}

interface TraceInputSpec {
  tracePath: string;
  bundleDir: string | null;
}

interface LoadedManifest {
  traceInputs: TraceInputSpec[];
  artifactRoot: string | null;
  workedTraceLimit: number | null;
  assumptions: string[];
}

interface CanonicalRecordedSessionTraceSetManifestV1 {
  contract: typeof CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT;
  setId?: string;
  entries: Array<{
    path?: string;
  }>;
  realTraceCoverage?: {
    summary?: string;
  };
}

function usage(): void {
  process.stderr.write(
    [
      "Usage: tsx scripts/build-recorded-session-replay-lane.ts [options]",
      "",
      "Options:",
      "  --trace <path>           Sanitized recorded trace JSON. Repeatable.",
      `  --trace-manifest <path>  Trace manifest. Defaults to ${DEFAULT_RECORDED_SESSION_REPLAY_TRACE_MANIFEST_PATH}`,
      "                           Canonical contract: lane-c frozen set manifest with entries[].path.",
      "                           Compatibility fallback also accepts traceEntries, traces, entries, or a bare array.",
      "  --artifact-root <path>   Output root. Defaults to docs/evidence/YYYY-MM-DD/<git-sha>/recorded-session-replay",
      "  --worked-trace-limit <n> Maximum number of traces to include in _lane/worked-traces.md (default: 8)",
      "  --assumption <text>      Repeatable note recorded in the lane index/report.",
      "  --help                   Show this help",
      "",
      "Output layout:",
      "  <artifact-root>/<trace-id>/... per-trace replay proof bundle",
      "  <artifact-root>/_lane/README.md",
      "  <artifact-root>/_lane/index.json",
      "  <artifact-root>/_lane/summary-tables.json",
      "  <artifact-root>/_lane/pairwise-deltas.json",
      "  <artifact-root>/_lane/win-rate-matrix.json",
      "  <artifact-root>/_lane/worked-traces.md",
      "  <artifact-root>/_lane/generation-report.json",
    ].join("\n") + "\n",
  );
}

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length === 0 ? null : trimmed;
}

function parseArgs(argv: string[]): ParsedArgs {
  const parsed: ParsedArgs = {
    tracePaths: [],
    traceManifestPath: null,
    artifactRoot: null,
    workedTraceLimit: null,
    assumptions: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--trace":
        parsed.tracePaths.push(path.resolve(argv[index + 1] ?? ""));
        index += 1;
        break;
      case "--trace-manifest":
        parsed.traceManifestPath = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--artifact-root":
        parsed.artifactRoot = path.resolve(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--worked-trace-limit":
        parsed.workedTraceLimit = Number(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--assumption":
        parsed.assumptions.push(argv[index + 1] ?? "");
        index += 1;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (parsed.workedTraceLimit !== null && (!Number.isFinite(parsed.workedTraceLimit) || parsed.workedTraceLimit < 1)) {
    throw new Error("--worked-trace-limit must be a positive integer");
  }
  return parsed;
}

function gitShaOrUnknown(): string {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8",
    }).trim();
  } catch {
    return "unknown-git-sha";
  }
}

function defaultArtifactRoot(): string {
  return path.resolve(
    repoRoot,
    "docs",
    "evidence",
    new Date().toISOString().slice(0, 10),
    gitShaOrUnknown(),
    "recorded-session-replay",
  );
}

function readTrace(tracePath: string): RecordedSessionTraceV1 {
  return JSON.parse(readFileSync(tracePath, "utf8")) as RecordedSessionTraceV1;
}

function isCanonicalRecordedSessionTraceSetManifest(
  parsed: unknown,
): parsed is CanonicalRecordedSessionTraceSetManifestV1 {
  return !!parsed
    && typeof parsed === "object"
    && !Array.isArray(parsed)
    && (parsed as { contract?: unknown }).contract === CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT
    && Array.isArray((parsed as { entries?: unknown }).entries);
}

function coerceManifestTraceEntries(parsed: unknown): unknown[] {
  if (Array.isArray(parsed)) {
    return parsed;
  }
  if (!parsed || typeof parsed !== "object") {
    return [];
  }
  const record = parsed as Record<string, unknown>;
  for (const key of ["traceEntries", "traces", "entries"]) {
    if (Array.isArray(record[key])) {
      return record[key] as unknown[];
    }
  }
  return [];
}

function resolveEntryTracePath(entry: unknown, manifestDir: string): string {
  if (typeof entry === "string") {
    return path.resolve(manifestDir, entry);
  }
  if (!entry || typeof entry !== "object") {
    throw new Error("Trace manifest entry must be a string or object");
  }
  const record = entry as Record<string, unknown>;
  for (const key of ["tracePath", "path", "trace", "source"]) {
    const value = normalizeCliString(typeof record[key] === "string" ? record[key] : undefined);
    if (value !== null) {
      return path.resolve(manifestDir, value);
    }
  }
  throw new Error("Trace manifest entry is missing tracePath/path/trace/source");
}

function resolveEntryBundleDir(entry: unknown, artifactRoot: string): string | null {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const record = entry as Record<string, unknown>;
  for (const key of ["bundleDir", "artifactDir", "outputDir"]) {
    const value = normalizeCliString(typeof record[key] === "string" ? record[key] : undefined);
    if (value !== null) {
      return path.isAbsolute(value) ? value : path.resolve(artifactRoot, value);
    }
  }
  return null;
}

function loadCanonicalManifest(manifestPath: string, manifest: CanonicalRecordedSessionTraceSetManifestV1): LoadedManifest {
  const manifestDir = path.dirname(manifestPath);
  const realTraceCoverageSummary = normalizeCliString(manifest.realTraceCoverage?.summary);
  const traceInputs = manifest.entries.map((entry, index) => {
    const relativeTracePath = normalizeCliString(entry.path);
    if (relativeTracePath === null) {
      throw new Error(`Canonical manifest entries[${index}].path is required`);
    }
    return {
      tracePath: path.resolve(manifestDir, relativeTracePath),
      bundleDir: null,
    };
  });
  if (traceInputs.length === 0) {
    throw new Error(`Canonical manifest did not provide any trace entries: ${manifestPath}`);
  }
  const assumptions = [
    `trace manifest contract=${manifest.contract}`,
    ...(manifest.setId ? [`trace manifest setId=${manifest.setId}`] : []),
    ...(realTraceCoverageSummary ? [realTraceCoverageSummary] : []),
  ];
  return {
    traceInputs,
    artifactRoot: null,
    workedTraceLimit: null,
    assumptions,
  };
}

function loadManifest(manifestPath: string, artifactRoot: string): LoadedManifest {
  const resolvedManifestPath = path.resolve(manifestPath);
  if (!existsSync(resolvedManifestPath)) {
    throw new Error(`Trace manifest not found: ${resolvedManifestPath}`);
  }
  const manifestDir = path.dirname(resolvedManifestPath);
  const parsed = JSON.parse(readFileSync(resolvedManifestPath, "utf8")) as unknown;
  if (isCanonicalRecordedSessionTraceSetManifest(parsed)) {
    return loadCanonicalManifest(resolvedManifestPath, parsed);
  }
  const traceInputs = coerceManifestTraceEntries(parsed).map((entry) => ({
    tracePath: resolveEntryTracePath(entry, manifestDir),
    bundleDir: resolveEntryBundleDir(entry, artifactRoot),
  }));
  if (traceInputs.length === 0) {
    throw new Error(`Trace manifest did not provide any trace entries: ${resolvedManifestPath}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {
      traceInputs,
      artifactRoot: null,
      workedTraceLimit: null,
      assumptions: [],
    };
  }
  const record = parsed as Record<string, unknown>;
  const rawArtifactRoot = normalizeCliString(typeof record.artifactRoot === "string" ? record.artifactRoot : undefined);
  const assumptions = Array.isArray(record.assumptions)
    ? record.assumptions.map((value) => String(value))
    : Array.isArray(record.notes)
      ? record.notes.map((value) => String(value))
      : [];
  return {
    traceInputs,
    artifactRoot: rawArtifactRoot === null ? null : path.resolve(manifestDir, rawArtifactRoot),
    workedTraceLimit: typeof record.workedTraceLimit === "number" ? record.workedTraceLimit : null,
    assumptions,
  };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const provisionalArtifactRoot = args.artifactRoot ?? defaultArtifactRoot();
  const traceManifestPath = args.traceManifestPath ?? (
    args.tracePaths.length === 0
      ? DEFAULT_RECORDED_SESSION_REPLAY_TRACE_MANIFEST_PATH
      : null
  );
  const manifest = traceManifestPath === null ? null : loadManifest(traceManifestPath, provisionalArtifactRoot);
  const artifactRoot = args.artifactRoot ?? manifest?.artifactRoot ?? provisionalArtifactRoot;
  const manifestTraceInputs = manifest?.traceInputs ?? [];
  const cliTraceInputs = args.tracePaths.map((tracePath) => ({
    tracePath,
    bundleDir: null,
  }));
  const traceInputs = [...manifestTraceInputs, ...cliTraceInputs];
  if (traceInputs.length === 0) {
    throw new Error("No traces resolved after loading the manifest");
  }
  const descriptor = writeRecordedSessionReplayProofLane({
    artifactRoot,
    traces: traceInputs.map((traceInput) => ({
      trace: readTrace(traceInput.tracePath),
      tracePath: traceInput.tracePath,
      bundleDir: traceInput.bundleDir,
    })),
    workedTraceLimit: args.workedTraceLimit ?? manifest?.workedTraceLimit ?? undefined,
    sourceManifestPath: traceManifestPath,
    assumptions: [...(manifest?.assumptions ?? []), ...args.assumptions],
  });
  const failedEntries = descriptor.generationReport.entries.filter((entry) => entry.result === "failed");
  const invalidEntries = descriptor.generationReport.entries.filter(
    (entry) => entry.result === "passed" && entry.validation?.ok === false,
  );
  const lines = [
    `Recorded session replay proof lane: ${descriptor.artifactRoot}`,
    `laneDir: ${descriptor.laneDir}`,
    `requestedTraceCount: ${descriptor.index.requestedTraceCount}`,
    `successfulTraceCount: ${descriptor.index.successfulTraceCount}`,
    `failedTraceCount: ${descriptor.index.failedTraceCount}`,
    `index: ${descriptor.indexPath}`,
    ...(traceManifestPath === null ? [] : [`traceManifestPath: ${traceManifestPath}`]),
    `summaryTables: ${descriptor.summaryTablesPath}`,
    `pairwiseDeltas: ${descriptor.pairwiseDeltasPath}`,
    `winRateMatrix: ${descriptor.winRateMatrixPath}`,
    `workedTraces: ${descriptor.workedTracesPath}`,
    `generationReport: ${descriptor.generationReportPath}`,
  ];
  process.stdout.write(`${lines.join("\n")}\n`);
  if (failedEntries.length > 0 || invalidEntries.length > 0) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
}
