#!/usr/bin/env node

import path from "node:path";
import { pathToFileURL } from "node:url";

import { runGraphifyBridgeCut } from "../packages/cli/dist/src/graphify-bridge.js";

function usage() {
  return [
    "Usage: node scripts/graphify-bridge.mjs [options]",
    "",
    "First safe Graphify bridge cut: dual source-bundle export + managed Graphify run + OCB compiled artifact pack.",
    "",
    "Options:",
    "  --openclaw-home <path>          OpenClaw home to export from",
    "  --activation-root <path>        OpenClawBrain activation root for projection canonical archive",
    "  --output-root <path>            Parent directory for bridge runs",
    "  --run-id <id>                   Stable run id",
    "  --repo-root <path>              Repo root for docs/code mirrors",
    "  --workspace-root <path>         Workspace root for MEMORY/TASKS projections",
    "  --home-dir <path>               Home-dir override for fixture/session discovery",
    "  --session-key <key>             Projection session key",
    "  --session-source <path>         Optional markdown/text source for session projection",
    "  --proof-summary-source <path>   Optional markdown/text source for proof projection",
    "  --docs-root <path>              Docs root to mirror into projection",
    "  --code-root <path>              Code root to mirror into projection",
    "  --generated-at <iso>            Deterministic timestamp",
    "  --graphify-version <text>       Managed run version label",
    "  --graphify-command <path>       Optional external Graphify executable; omitted means synthesized only",
    "  --graphify-arg <arg>            Optional external Graphify arg; repeatable",
    "  --label <label>                 Extra run label; repeatable",
    "  --no-clean                      Keep existing run root contents",
    "  --json                          Print machine-readable result",
    "  -h, --help                      Show help",
  ].join("\n");
}

function takeValue(argv, index, flag, options = {}) {
  const value = argv[index + 1];
  if (typeof value !== "string" || (options.allowFlagLike !== true && value.startsWith("--"))) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

export function parseGraphifyBridgeCliArgs(argv = process.argv.slice(2)) {
  const options = { graphifyArgs: [], labels: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "-h":
      case "--help":
        options.help = true;
        break;
      case "--json":
        options.json = true;
        break;
      case "--no-clean":
        options.clean = false;
        break;
      case "--openclaw-home":
        options.openclawHome = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--activation-root":
        options.activationRoot = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--output-root":
        options.outputRoot = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--run-id":
        options.runId = takeValue(argv, index, arg);
        index += 1;
        break;
      case "--repo-root":
        options.repoRoot = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--workspace-root":
        options.workspaceRoot = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--home-dir":
        options.homeDir = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--session-key":
        options.sessionKey = takeValue(argv, index, arg);
        index += 1;
        break;
      case "--session-source":
        options.sessionSourcePath = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--proof-summary-source":
        options.proofSummarySourcePath = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--docs-root":
        options.docsRoot = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--code-root":
        options.codeRoot = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--generated-at":
        options.generatedAt = takeValue(argv, index, arg);
        index += 1;
        break;
      case "--graphify-version":
        options.graphifyVersion = takeValue(argv, index, arg);
        index += 1;
        break;
      case "--graphify-command":
        options.graphifyCommand = path.resolve(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--graphify-arg":
        options.graphifyArgs.push(takeValue(argv, index, arg, { allowFlagLike: true }));
        index += 1;
        break;
      case "--label":
        options.labels.push(takeValue(argv, index, arg));
        index += 1;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }
  return options;
}

export function runGraphifyBridgeCli(argv = process.argv.slice(2)) {
  const options = parseGraphifyBridgeCliArgs(argv);
  if (options.help) {
    console.log(usage());
    return 0;
  }
  const result = runGraphifyBridgeCut(options);
  if (options.json) {
    console.log(JSON.stringify({
      ok: result.ok,
      runId: result.runId,
      runRoot: result.runRoot,
      outputs: result.outputs,
      sourceBundleDigest: result.sourceBundleManifest.dualSourceDigest,
      graphifyRunId: result.graphifyRun.runId,
      compiledPackId: result.compiledPack.packId,
    }, null, 2));
    return 0;
  }
  console.log(`GRAPHIFY BRIDGE ok ${result.runId}`);
  console.log(`source bundle: ${result.outputs.sourceBundleRoot}`);
  console.log(`graphify run: ${result.outputs.graphifyRun}`);
  console.log(`compiled artifact pack: ${result.outputs.compiledArtifactPack}`);
  console.log(`summary: ${result.outputs.summary}`);
  return 0;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exitCode = runGraphifyBridgeCli(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
