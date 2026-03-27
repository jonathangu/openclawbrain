#!/usr/bin/env tsx

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { validateRecordedSessionReplayProofBundle, writeRecordedSessionReplayProofBundle, type RecordedSessionTraceV1 } from "../packages/cli/dist/src/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

function usage() {
  process.stderr.write(
    [
      "Usage: tsx scripts/validate-recorded-session-replay.ts --trace <path> [options]",
      "",
      "Options:",
      "  --artifact-dir <path>   Directory for the replay proof bundle.",
      "                          Defaults to docs/evidence/YYYY-MM-DD/<git-sha>/recorded-session-replay/<trace-id>",
      "  --help                  Show this help",
      "",
      "Artifacts written:",
      "  manifest.json, trace.json, fixture.json, bundle.json, environment.json,",
      "  summary.md, summary-tables.json, coverage-snapshot.json,",
      "  hardening-snapshot.json, hashes.json, modes/*.json, validation-report.json",
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

function parseArgs(argv: string[]) {
  let tracePath: string | null = null;
  let artifactDir: string | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--trace":
        tracePath = argv[index + 1] ?? null;
        index += 1;
        break;
      case "--artifact-dir":
        artifactDir = argv[index + 1] ?? null;
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

  if (tracePath === null) {
    throw new Error("--trace is required");
  }

  return {
    tracePath: path.resolve(tracePath),
    artifactDir: artifactDir === null ? null : path.resolve(artifactDir),
  };
}

function readTrace(tracePath: string): RecordedSessionTraceV1 {
  return JSON.parse(readFileSync(tracePath, "utf8")) as RecordedSessionTraceV1;
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

function defaultArtifactDir(trace: RecordedSessionTraceV1): string {
  const artifactDate = new Date().toISOString().slice(0, 10);
  return path.resolve(
    repoRoot,
    "docs",
    "evidence",
    artifactDate,
    gitShaOrUnknown(),
    "recorded-session-replay",
    trace.traceId,
  );
}

function main() {
  const { tracePath, artifactDir } = parseArgs(process.argv.slice(2));
  const trace = readTrace(tracePath);
  const outputDir = artifactDir ?? defaultArtifactDir(trace);
  mkdirSync(outputDir, { recursive: true });
  const descriptor = writeRecordedSessionReplayProofBundle({
    rootDir: outputDir,
    trace,
  });
  const validation = validateRecordedSessionReplayProofBundle(outputDir);
  writeFileSync(path.join(outputDir, "validation-report.json"), `${JSON.stringify(validation, null, 2)}\n`, "utf8");
  const lines = [
    `Recorded session replay proof bundle: ${descriptor.rootDir}`,
    `traceId: ${descriptor.bundle.traceId}`,
    `winnerMode: ${descriptor.bundle.summary.winnerMode ?? "none"}`,
    `bundleHash: ${descriptor.bundle.bundleHash}`,
    `validation: ${validation.ok ? "ok" : "failed"}`,
    `validationReport: ${path.join(outputDir, "validation-report.json")}`,
  ];
  process.stdout.write(`${lines.join("\n")}\n`);
  if (!validation.ok) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
}
