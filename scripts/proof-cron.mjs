#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

const CONTRACT = "openclawbrain_proof_cron.v1";
const DEFAULT_OUTPUT_ROOT = path.join(workspaceRoot, "artifacts", "openclawbrain-proof-cron");
const DEFAULT_CONFIG_PATH = path.join(DEFAULT_OUTPUT_ROOT, "cron-config.json");
const DEFAULT_OPENCLAW_HOME = path.join(process.env.HOME ?? "", ".openclaw");
const RECORDED_SESSION_REPLAY_MODE_ORDER = ["no_brain", "vector_only", "graph_prior_only", "learned_route"];
const LEGACY_STATUS_COMMAND = [
  process.execPath,
  path.join(repoRoot, "bin", "openclawbrain.js"),
  "status",
  "--openclaw-home",
  "{{openclawHome}}",
  "--json",
];
const DEFAULT_STATUS_COMMAND = [
  process.execPath,
  path.join(repoRoot, "packages", "cli", "dist", "src", "cli.js"),
  "status",
  "--openclaw-home",
  "{{openclawHome}}",
  "--json",
];
const DEFAULT_PRICING_TABLE_PATH = path.join(repoRoot, "scripts", "pricing-table.v1.json");

function usage() {
  process.stderr.write(
    [
      "Usage:",
      "  node scripts/proof-cron.mjs health [options]",
      "  node scripts/proof-cron.mjs nightly [options]",
      "",
      "Options:",
      `  --config <path>       Config file path (default: ${path.relative(repoRoot, DEFAULT_CONFIG_PATH)})`,
      `  --output-dir <path>   Output directory (default: ${path.relative(repoRoot, DEFAULT_OUTPUT_ROOT)}/<run>)`,
      "  --openclaw-home <p>   Override the OpenClaw home used by the status probe",
      "  --help                Show this help",
      "",
      "The script writes proof-cron surfaces for lightweight health snapshots and nightly proof aggregation.",
    ].join("\n") + "\n",
  );
}

function parseArgs(argv) {
  const options = {
    command: null,
    configPath: DEFAULT_CONFIG_PATH,
    outputDir: null,
    openclawHome: DEFAULT_OPENCLAW_HOME,
  };

  const args = [...argv];
  if (args.length > 0 && !args[0].startsWith("--")) {
    options.command = args.shift();
  }

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    switch (arg) {
      case "--config":
        options.configPath = path.resolve(args[++index] ?? "");
        break;
      case "--output-dir":
        options.outputDir = path.resolve(args[++index] ?? "");
        break;
      case "--openclaw-home":
        options.openclawHome = path.resolve(args[++index] ?? "");
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

  if (options.command !== "health" && options.command !== "nightly") {
    throw new Error("First argument must be either 'health' or 'nightly'");
  }

  return options;
}

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function resolveToken(value, context) {
  if (typeof value !== "string") {
    return value;
  }
  return value
    .replaceAll("{{repoRoot}}", context.repoRoot)
    .replaceAll("{{workspaceRoot}}", context.workspaceRoot)
    .replaceAll("{{openclawHome}}", context.openclawHome)
    .replaceAll("{{outputRoot}}", context.outputRoot);
}

function normalizePathSpec(spec) {
  if (typeof spec === "string") {
    return { base: "workspace", path: spec };
  }
  if (!spec || typeof spec !== "object") {
    return null;
  }
  const base = spec.base === "repo" ? "repo" : "workspace";
  const normalizedPath = typeof spec.path === "string" ? spec.path : "";
  return normalizedPath.length > 0 ? { base, path: normalizedPath } : null;
}

function resolvePathSpec(spec, context) {
  const normalized = normalizePathSpec(spec);
  if (!normalized) {
    return null;
  }
  const baseRoot = normalized.base === "repo" ? context.repoRoot : context.workspaceRoot;
  return path.resolve(baseRoot, resolveToken(normalized.path, context));
}

function defaultConfig(context) {
  return {
    contract: CONTRACT,
    openclawHome: "{{openclawHome}}",
    healthFreshnessDays: 7,
    freshnessThresholdDays: 21,
    scanRoots: [
      { base: "workspace", path: "artifacts" },
      { base: "repo", path: "docs/evidence" },
    ],
    excludeRoots: [
      { base: "workspace", path: "artifacts/openclawbrain-proof-cron" },
    ],
    statusCommand: DEFAULT_STATUS_COMMAND,
  };
}

function isLegacyStatusCommand(command) {
  if (!Array.isArray(command) || command.length < 6) {
    return false;
  }
  const [, scriptPath, subcommand, openclawHomeFlag, openclawHomeValue, jsonFlag] = command.map((part) => String(part));
  return scriptPath.endsWith(path.join("bin", "openclawbrain.js"))
    && subcommand === "status"
    && openclawHomeFlag === "--openclaw-home"
    && openclawHomeValue === "{{openclawHome}}"
    && jsonFlag === "--json";
}

function loadConfig(configPath, context) {
  const defaults = defaultConfig(context);
  if (!existsSync(configPath)) {
    return defaults;
  }
  try {
    const parsed = JSON.parse(readFileSync(configPath, "utf8"));
    const parsedStatusCommand = Array.isArray(parsed.statusCommand) && parsed.statusCommand.length > 0
      ? parsed.statusCommand
      : defaults.statusCommand;
    const statusCommand = isLegacyStatusCommand(parsedStatusCommand)
      ? defaults.statusCommand
      : parsedStatusCommand;
    return {
      ...defaults,
      ...parsed,
      openclawHome: typeof parsed.openclawHome === "string" ? parsed.openclawHome : defaults.openclawHome,
      scanRoots: Array.isArray(parsed.scanRoots) && parsed.scanRoots.length > 0 ? parsed.scanRoots : defaults.scanRoots,
      excludeRoots: Array.isArray(parsed.excludeRoots) ? parsed.excludeRoots : defaults.excludeRoots,
      statusCommand,
    };
  } catch {
    return defaults;
  }
}

function saveJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function saveText(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, value, "utf8");
}

function parseJsonText(text, label = "json") {
  const trimmed = String(text ?? "").trim();
  if (trimmed.length === 0) {
    throw new Error(`Expected ${label} text but received an empty response`);
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start >= 0 && end > start) {
      return JSON.parse(trimmed.slice(start, end + 1));
    }
    throw new Error(`Unable to parse ${label} output as JSON`);
  }
}

function runStatusProbe(config, context) {
  const command = config.statusCommand.map((part) => resolveToken(String(part), context));
  const [commandName, ...commandArgs] = command;
  const startedAt = Date.now();
  const result = spawnSync(commandName, commandArgs, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const endedAt = Date.now();
  if (result.error) {
    throw result.error;
  }
  if (typeof result.status === "number" && result.status !== 0) {
    throw new Error(
      `status probe failed with exit code ${result.status}: ${(result.stderr ?? result.stdout ?? "").toString().slice(0, 2000)}`,
    );
  }
  const stdout = result.stdout ?? "";
  const stderr = result.stderr ?? "";
  const status = parseJsonText(stdout, "status");
  return {
    command: command.join(" "),
    commandName,
    commandArgs,
    startedAt: new Date(startedAt).toISOString(),
    endedAt: new Date(endedAt).toISOString(),
    durationMs: endedAt - startedAt,
    exitCode: typeof result.status === "number" ? result.status : null,
    signal: result.signal ?? null,
    stdout,
    stderr,
    parsed: status,
  };
}

function readJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function readTextIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  try {
    return readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
}

function collectFiles(rootDir, excludeDirs = new Set()) {
  const out = [];
  const stack = [path.resolve(rootDir)];
  while (stack.length > 0) {
    const currentDir = stack.pop();
    if (!currentDir || excludeDirs.has(currentDir)) {
      continue;
    }
    let entries;
    try {
      entries = readdirSync(currentDir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const childPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        if (!excludeDirs.has(childPath)) {
          stack.push(childPath);
        }
        continue;
      }
      if (entry.isFile()) {
        out.push(childPath);
      }
    }
  }
  return out;
}

function collectBundleCandidates(roots, excludeRoots = []) {
  const excludeSet = new Set(excludeRoots.map((p) => path.resolve(p)));
  const bundles = [];
  const visited = new Set();

  const shouldExclude = (dirPath) => {
    const resolved = path.resolve(dirPath);
    for (const excluded of excludeSet) {
      if (resolved === excluded || resolved.startsWith(`${excluded}${path.sep}`)) {
        return true;
      }
    }
    return false;
  };

  const stack = roots.map((root) => path.resolve(root)).filter(Boolean);
  while (stack.length > 0) {
    const currentDir = stack.pop();
    if (!currentDir || visited.has(currentDir) || shouldExclude(currentDir)) {
      continue;
    }
    visited.add(currentDir);

    let entries;
    try {
      entries = readdirSync(currentDir, { withFileTypes: true });
    } catch {
      continue;
    }

    const fileNames = new Set(entries.filter((entry) => entry.isFile()).map((entry) => entry.name));
    const classification = classifyBundleRoot(fileNames);
    if (classification) {
      bundles.push({
        path: currentDir,
        kind: classification,
      });
      continue;
    }

    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const childDir = path.join(currentDir, entry.name);
      if (!shouldExclude(childDir)) {
        stack.push(childDir);
      }
    }
  }

  return bundles;
}

function classifyBundleRoot(fileNames) {
  const hasSummary = fileNames.has("summary.md");
  const hasValidation = fileNames.has("validation-report.json");
  const hasSteps = fileNames.has("steps.json");
  const hasVerdict = fileNames.has("verdict.json");
  const hasManifest = fileNames.has("manifest.json");
  const hasBundle = fileNames.has("bundle.json");
  const hasStatus = fileNames.has("status.json");
  const hasDoctor = fileNames.has("doctor.json");
  const hasConfigSnapshot = fileNames.has("config-snapshot.json");

  if (hasSummary && (hasSteps || hasVerdict)) {
    return "operator-proof";
  }
  if (hasSummary && hasManifest && hasBundle) {
    return "recorded-session-replay";
  }
  if (hasSummary && hasStatus && hasDoctor && hasConfigSnapshot) {
    return "host-evidence";
  }
  if (hasSummary && hasValidation) {
    return "generic-proof";
  }
  return null;
}

function parseTimestampFromPath(bundlePath) {
  const segments = bundlePath.split(path.sep).reverse();
  for (const segment of segments) {
    const operatorMatch = segment.match(/^operator-proof-(\d{8})-(\d{6})Z$/);
    if (operatorMatch) {
      const [, datePart, timePart] = operatorMatch;
      const iso = `${datePart.slice(0, 4)}-${datePart.slice(4, 6)}-${datePart.slice(6, 8)}T${timePart.slice(0, 2)}:${timePart.slice(2, 4)}:${timePart.slice(4, 6)}.000Z`;
      const parsed = Date.parse(iso);
      if (!Number.isNaN(parsed)) {
        return new Date(parsed);
      }
    }
    if (/^\d{4}-\d{2}-\d{2}$/.test(segment)) {
      const parsed = Date.parse(`${segment}T00:00:00.000Z`);
      if (!Number.isNaN(parsed)) {
        return new Date(parsed);
      }
    }
  }
  return null;
}

function safeParseDate(value) {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return null;
  }
  return new Date(parsed);
}

function bundleTimestamp(bundlePath, payloads) {
  const candidates = [
    payloads?.operator?.verdict?.bundleStartedAt,
    payloads?.operator?.summary?.bundleStartedAt,
    payloads?.replay?.bundle?.generatedAt,
    payloads?.replay?.bundle?.recordedAt,
    payloads?.host?.summary?.bundleStartedAt,
    payloads?.host?.summary?.generatedAt,
  ].map(safeParseDate);
  const fromPayload = candidates.find((item) => item !== null);
  if (fromPayload) {
    return fromPayload;
  }
  return parseTimestampFromPath(bundlePath);
}

function round(value, places = 2) {
  if (!Number.isFinite(value)) {
    return null;
  }
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function mean(values) {
  const filtered = values.filter((value) => Number.isFinite(value));
  if (filtered.length === 0) {
    return null;
  }
  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

function median(values) {
  const filtered = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (filtered.length === 0) {
    return null;
  }
  const middle = Math.floor(filtered.length / 2);
  return filtered.length % 2 === 0 ? (filtered[middle - 1] + filtered[middle]) / 2 : filtered[middle];
}

function percentile(values, p) {
  const filtered = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (filtered.length === 0) {
    return null;
  }
  const index = Math.min(filtered.length - 1, Math.max(0, Math.ceil((p / 100) * filtered.length) - 1));
  return filtered[index];
}

function sum(values) {
  return values.filter((value) => Number.isFinite(value)).reduce((total, value) => total + value, 0);
}

function countStringChars(values) {
  if (!Array.isArray(values)) {
    return 0;
  }
  return values.reduce((total, value) => total + (typeof value === "string" ? value.length : 0), 0);
}

function countTextChars(value) {
  if (typeof value === "string") {
    return value.length;
  }
  if (Array.isArray(value)) {
    return value.reduce((total, entry) => total + (typeof entry === "string" ? entry.length : 0), 0);
  }
  return 0;
}

function sumOrNull(values) {
  const filtered = values.filter(Number.isFinite);
  if (filtered.length === 0) {
    return null;
  }
  return filtered.reduce((total, value) => total + value, 0);
}

function loadPricingTable(pricingTablePath = DEFAULT_PRICING_TABLE_PATH) {
  if (!existsSync(pricingTablePath)) {
    return null;
  }
  const pricingTable = readJsonIfExists(pricingTablePath);
  if (!pricingTable || pricingTable.contract !== "openclawbrain_pricing_table.v1") {
    return null;
  }
  const charsPerToken = Number(pricingTable.charsPerToken ?? pricingTable.tokenCharsPerToken ?? 4);
  const promptPriceUsdPer1mTokens = Number(pricingTable.promptPriceUsdPer1mTokens);
  const completionPriceUsdPer1mTokens = Number(pricingTable.completionPriceUsdPer1mTokens);
  if (!Number.isFinite(charsPerToken) || charsPerToken <= 0 || !Number.isFinite(promptPriceUsdPer1mTokens) || !Number.isFinite(completionPriceUsdPer1mTokens)) {
    return null;
  }
  return {
    contract: pricingTable.contract,
    version: typeof pricingTable.version === "string" ? pricingTable.version : "v1",
    path: path.relative(repoRoot, pricingTablePath).split(path.sep).join("/"),
    charsPerToken,
    promptPriceUsdPer1mTokens,
    completionPriceUsdPer1mTokens,
  };
}

function estimateTokensFromChars(chars, charsPerToken) {
  if (!Number.isFinite(chars) || !Number.isFinite(charsPerToken) || charsPerToken <= 0) {
    return null;
  }
  return Math.ceil(chars / charsPerToken);
}

function estimateUsdFromTokens(tokens, pricePer1mTokens) {
  if (!Number.isFinite(tokens) || !Number.isFinite(pricePer1mTokens)) {
    return null;
  }
  return round((tokens / 1_000_000) * pricePer1mTokens, 6);
}

function summarizeTurnCompletionChars(turn) {
  const fields = ["assistantTexts", "completionTexts", "outputTexts", "assistantText", "completionText", "responseText"];
  for (const field of fields) {
    if (Object.prototype.hasOwnProperty.call(turn ?? {}, field)) {
      const value = turn?.[field];
      if (typeof value === "string" || Array.isArray(value)) {
        return countTextChars(value);
      }
      if (value === null || value === undefined) {
        return 0;
      }
    }
  }
  return null;
}

function normalizeLabelKind(kind) {
  return typeof kind === "string" ? kind.trim().toLowerCase() : "";
}

function summarizeTraceFeedback(turns) {
  const traceTurns = Array.isArray(turns) ? turns : [];
  let feedbackEventCount = 0;
  let nonApprovalFeedbackEventCount = 0;
  let turnsWithFeedbackCount = 0;
  let turnsWithNonApprovalFeedbackCount = 0;

  for (const turn of traceTurns) {
    const feedback = Array.isArray(turn?.feedback) ? turn.feedback : [];
    if (feedback.length > 0) {
      turnsWithFeedbackCount += 1;
    }

    let turnHasNonApprovalFeedback = false;
    for (const item of feedback) {
      feedbackEventCount += 1;
      if (normalizeLabelKind(item?.kind) !== "approval") {
        nonApprovalFeedbackEventCount += 1;
        turnHasNonApprovalFeedback = true;
      }
    }

    if (turnHasNonApprovalFeedback) {
      turnsWithNonApprovalFeedbackCount += 1;
    }
  }

  return {
    feedbackEventCount,
    nonApprovalFeedbackEventCount,
    turnsWithFeedbackCount,
    turnsWithNonApprovalFeedbackCount,
    turnsWithNonApprovalFeedbackRate: traceTurns.length > 0 ? round(turnsWithNonApprovalFeedbackCount / traceTurns.length, 4) : null,
  };
}

function summarizeReplayModeSavings(mode, traceTurnById = new Map(), pricingTable = null) {
  const turns = Array.isArray(mode?.turns) ? mode.turns : [];
  const turnCount = turns.length;
  const selectedContextBlockCount = turns.reduce(
    (total, turn) => total + (Array.isArray(turn?.selectedContextIds) ? turn.selectedContextIds.length : Array.isArray(turn?.selectedContextTexts) ? turn.selectedContextTexts.length : 0),
    0,
  );
  const selectedContextChars = turns.reduce((total, turn) => total + countStringChars(turn?.selectedContextTexts), 0);
  const turnsWithSelectedContextCount = turns.filter((turn) => countStringChars(turn?.selectedContextTexts) > 0).length;
  const charsPerToken = Number.isFinite(pricingTable?.charsPerToken) && pricingTable.charsPerToken > 0 ? pricingTable.charsPerToken : 4;
  const estimatedPromptTokens = estimateTokensFromChars(selectedContextChars, charsPerToken) ?? 0;
  let completionCharsObservedTurnCount = 0;
  let completionCharsTotal = 0;
  let completionCharsMissing = turnCount === 0;

  for (const turn of turns) {
    const completionChars = summarizeTurnCompletionChars(turn);
    if (completionChars === null) {
      completionCharsMissing = true;
      continue;
    }
    completionCharsObservedTurnCount += 1;
    completionCharsTotal += completionChars;
  }

  const completionChars = completionCharsMissing ? null : completionCharsTotal;
  const estimatedCompletionTokens = completionChars === null ? null : estimateTokensFromChars(completionChars, charsPerToken);
  const estimatedPromptCostUsd = estimateUsdFromTokens(estimatedPromptTokens, pricingTable?.promptPriceUsdPer1mTokens ?? null);
  const estimatedCompletionCostUsd = estimatedCompletionTokens === null ? null : estimateUsdFromTokens(estimatedCompletionTokens, pricingTable?.completionPriceUsdPer1mTokens ?? null);
  const estimatedTotalCostUsd = estimatedPromptCostUsd !== null && estimatedCompletionCostUsd !== null
    ? round(estimatedPromptCostUsd + estimatedCompletionCostUsd, 6)
    : null;
  let retrievalToolHopCount = 0;
  let retrievalToolHopTurnCount = 0;

  for (const turn of turns) {
    const digestCount = Number(turn?.observability?.selectionDigestCount);
    if (Number.isFinite(digestCount)) {
      if (digestCount > 0) {
        retrievalToolHopCount += digestCount;
        retrievalToolHopTurnCount += 1;
      }
      continue;
    }

    const traceTurn = traceTurnById.get(turn?.turnId);
    const hasSelectedContext = countStringChars(traceTurn?.selectedContextTexts) > 0 || (Array.isArray(turn?.selectedContextIds) && turn.selectedContextIds.length > 0);
    if (hasSelectedContext) {
      retrievalToolHopCount += 1;
      retrievalToolHopTurnCount += 1;
    }
  }

  return {
    mode: mode?.mode ?? null,
    turnCount,
    pricingTableVersion: pricingTable?.version ?? null,
    pricingTablePath: pricingTable?.path ?? null,
    selectedContextBlockCount,
    selectedContextChars,
    completionChars,
    estimatedPromptTokens,
    estimatedCompletionTokens,
    estimatedPromptCostUsd,
    estimatedCompletionCostUsd,
    estimatedTotalCostUsd,
    retrievalToolHopCount,
    retrievalToolHopTurnCount,
    selectedContextCharsPerTurnMean: turnCount > 0 ? round(selectedContextChars / turnCount, 2) : null,
    selectedContextBlocksPerTurnMean: turnCount > 0 ? round(selectedContextBlockCount / turnCount, 2) : null,
    estimatedPromptTokensPerTurnMean: turnCount > 0 ? round(estimatedPromptTokens / turnCount, 2) : null,
    completionCharsObservedTurnCount,
    completionCharsObservedRate: turnCount > 0 ? round(completionCharsObservedTurnCount / turnCount, 4) : null,
    completionCharsPerTurnMean: completionChars === null || turnCount === 0 ? null : round(completionChars / turnCount, 2),
    estimatedCompletionTokensPerTurnMean: estimatedCompletionTokens === null || turnCount === 0 ? null : round(estimatedCompletionTokens / turnCount, 2),
    estimatedPromptCostUsdPerTurnMean: estimatedPromptCostUsd === null || turnCount === 0 ? null : round(estimatedPromptCostUsd / turnCount, 6),
    estimatedCompletionCostUsdPerTurnMean: estimatedCompletionCostUsd === null || turnCount === 0 ? null : round(estimatedCompletionCostUsd / turnCount, 6),
    estimatedTotalCostUsdPerTurnMean: estimatedTotalCostUsd === null || turnCount === 0 ? null : round(estimatedTotalCostUsd / turnCount, 6),
    turnsWithSelectedContextCount,
    turnsWithSelectedContextRate: turnCount > 0 ? round(turnsWithSelectedContextCount / turnCount, 4) : null,
    retrievalToolHopPerTurnMean: turnCount > 0 ? round(retrievalToolHopCount / turnCount, 2) : null,
    retrievalToolHopTurnRate: turnCount > 0 ? round(retrievalToolHopTurnCount / turnCount, 4) : null,
  };
}

function aggregateReplaySavings(modeSavings) {
  const byMode = new Map();
  for (const entry of modeSavings) {
    if (!entry || typeof entry.mode !== "string") {
      continue;
    }
    const current = byMode.get(entry.mode) ?? {
      mode: entry.mode,
      turnCount: 0,
      pricingTableVersion: entry.pricingTableVersion ?? null,
      pricingTablePath: entry.pricingTablePath ?? null,
      selectedContextBlockCount: 0,
      selectedContextChars: 0,
      estimatedPromptTokens: 0,
      completionChars: 0,
      completionCharsObservedTurnCount: 0,
      estimatedCompletionTokens: 0,
      estimatedPromptCostUsd: 0,
      estimatedCompletionCostUsd: 0,
      estimatedTotalCostUsd: 0,
      retrievalToolHopCount: 0,
      retrievalToolHopTurnCount: 0,
      turnsWithSelectedContextCount: 0,
      completionCharsMissing: false,
      estimatedPromptCostUsdMissing: false,
      estimatedCompletionTokensMissing: false,
      estimatedCompletionCostUsdMissing: false,
      estimatedTotalCostUsdMissing: false,
    };
    current.turnCount += Number(entry.turnCount ?? 0);
    current.selectedContextBlockCount += Number(entry.selectedContextBlockCount ?? 0);
    current.selectedContextChars += Number(entry.selectedContextChars ?? 0);
    current.estimatedPromptTokens += Number(entry.estimatedPromptTokens ?? 0);
    current.completionCharsObservedTurnCount += Number(entry.completionCharsObservedTurnCount ?? 0);
    if (entry.completionChars === null || entry.completionChars === undefined) {
      current.completionCharsMissing = true;
    } else {
      current.completionChars += Number(entry.completionChars ?? 0);
    }
    if (entry.estimatedPromptCostUsd === null || entry.estimatedPromptCostUsd === undefined) {
      current.estimatedPromptCostUsdMissing = true;
    } else {
      current.estimatedPromptCostUsd += Number(entry.estimatedPromptCostUsd ?? 0);
    }
    if (entry.estimatedCompletionTokens === null || entry.estimatedCompletionTokens === undefined) {
      current.estimatedCompletionTokensMissing = true;
    } else {
      current.estimatedCompletionTokens += Number(entry.estimatedCompletionTokens ?? 0);
    }
    if (entry.estimatedCompletionCostUsd === null || entry.estimatedCompletionCostUsd === undefined) {
      current.estimatedCompletionCostUsdMissing = true;
    } else {
      current.estimatedCompletionCostUsd += Number(entry.estimatedCompletionCostUsd ?? 0);
    }
    if (entry.estimatedTotalCostUsd === null || entry.estimatedTotalCostUsd === undefined) {
      current.estimatedTotalCostUsdMissing = true;
    } else {
      current.estimatedTotalCostUsd += Number(entry.estimatedTotalCostUsd ?? 0);
    }
    current.retrievalToolHopCount += Number(entry.retrievalToolHopCount ?? 0);
    current.retrievalToolHopTurnCount += Number(entry.retrievalToolHopTurnCount ?? 0);
    current.turnsWithSelectedContextCount += Number(entry.turnsWithSelectedContextCount ?? 0);
    if (current.pricingTableVersion === null && entry.pricingTableVersion !== null && entry.pricingTableVersion !== undefined) {
      current.pricingTableVersion = entry.pricingTableVersion;
    }
    if (current.pricingTablePath === null && entry.pricingTablePath !== null && entry.pricingTablePath !== undefined) {
      current.pricingTablePath = entry.pricingTablePath;
    }
    byMode.set(entry.mode, current);
  }

  return RECORDED_SESSION_REPLAY_MODE_ORDER.map((mode) => {
    const entry = byMode.get(mode) ?? {
      mode,
      turnCount: 0,
      pricingTableVersion: null,
      pricingTablePath: null,
      selectedContextBlockCount: 0,
      selectedContextChars: 0,
      estimatedPromptTokens: 0,
      completionChars: 0,
      completionCharsObservedTurnCount: 0,
      estimatedCompletionTokens: 0,
      estimatedPromptCostUsd: 0,
      estimatedCompletionCostUsd: 0,
      estimatedTotalCostUsd: 0,
      retrievalToolHopCount: 0,
      retrievalToolHopTurnCount: 0,
      turnsWithSelectedContextCount: 0,
      completionCharsMissing: false,
      estimatedPromptCostUsdMissing: false,
      estimatedCompletionTokensMissing: false,
      estimatedCompletionCostUsdMissing: false,
      estimatedTotalCostUsdMissing: false,
    };
    const {
      completionCharsMissing,
      estimatedPromptCostUsdMissing,
      estimatedCompletionTokensMissing,
      estimatedCompletionCostUsdMissing,
      estimatedTotalCostUsdMissing,
      ...cleanEntry
    } = entry;
    const completionChars = entry.completionCharsMissing ? null : entry.completionChars;
    const estimatedCompletionTokens = entry.estimatedCompletionTokensMissing ? null : entry.estimatedCompletionTokens;
    const estimatedPromptCostUsd = entry.estimatedPromptCostUsdMissing ? null : round(entry.estimatedPromptCostUsd, 6);
    const estimatedCompletionCostUsd = entry.estimatedCompletionCostUsdMissing ? null : round(entry.estimatedCompletionCostUsd, 6);
    const estimatedTotalCostUsd = entry.estimatedTotalCostUsdMissing ? null : round(entry.estimatedTotalCostUsd, 6);
    return {
      ...cleanEntry,
      completionChars,
      estimatedCompletionTokens,
      estimatedPromptCostUsd,
      estimatedCompletionCostUsd,
      estimatedTotalCostUsd,
      selectedContextCharsPerTurnMean: entry.turnCount > 0 ? round(entry.selectedContextChars / entry.turnCount, 2) : null,
      selectedContextBlocksPerTurnMean: entry.turnCount > 0 ? round(entry.selectedContextBlockCount / entry.turnCount, 2) : null,
      estimatedPromptTokensPerTurnMean: entry.turnCount > 0 ? round(entry.estimatedPromptTokens / entry.turnCount, 2) : null,
      completionCharsObservedRate: entry.turnCount > 0 ? round(entry.completionCharsObservedTurnCount / entry.turnCount, 4) : null,
      completionCharsPerTurnMean: completionChars === null || entry.turnCount === 0 ? null : round(completionChars / entry.turnCount, 2),
      estimatedCompletionTokensPerTurnMean: estimatedCompletionTokens === null || entry.turnCount === 0 ? null : round(estimatedCompletionTokens / entry.turnCount, 2),
      estimatedPromptCostUsdPerTurnMean: estimatedPromptCostUsd === null || entry.turnCount === 0 ? null : round(estimatedPromptCostUsd / entry.turnCount, 6),
      estimatedCompletionCostUsdPerTurnMean: estimatedCompletionCostUsd === null || entry.turnCount === 0 ? null : round(estimatedCompletionCostUsd / entry.turnCount, 6),
      estimatedTotalCostUsdPerTurnMean: estimatedTotalCostUsd === null || entry.turnCount === 0 ? null : round(estimatedTotalCostUsd / entry.turnCount, 6),
      turnsWithSelectedContextRate: entry.turnCount > 0 ? round(entry.turnsWithSelectedContextCount / entry.turnCount, 4) : null,
      retrievalToolHopPerTurnMean: entry.turnCount > 0 ? round(entry.retrievalToolHopCount / entry.turnCount, 2) : null,
      retrievalToolHopTurnRate: entry.turnCount > 0 ? round(entry.retrievalToolHopTurnCount / entry.turnCount, 4) : null,
    };
  });
}

function summarizeOperatorBundle(bundlePath, workspaceRoot) {
  const files = collectFiles(bundlePath);
  const fileStats = files.map((filePath) => {
    const stat = statSync(filePath);
    return {
      path: filePath,
      relativePath: path.relative(workspaceRoot, filePath).split(path.sep).join("/"),
      size: stat.size,
    };
  });

  const fileNames = new Set(fileStats.map((entry) => path.basename(entry.path)));
  const summary = readTextIfExists(path.join(bundlePath, "summary.md"));
  const steps = readJsonIfExists(path.join(bundlePath, "steps.json"));
  const verdict = readJsonIfExists(path.join(bundlePath, "verdict.json"));
  const validation = readJsonIfExists(path.join(bundlePath, "validation-report.json"));
  const status = readJsonIfExists(path.join(bundlePath, "status.json"));

  const stepRows = Array.isArray(steps?.steps) ? steps.steps : Array.isArray(steps) ? steps : [];
  const stepDurations = stepRows.map((step) => Number(step.durationMs ?? 0)).filter(Number.isFinite);
  const executedSteps = stepRows.filter((step) => step.skipped !== true);

  return {
    kind: "operator-proof",
    bundleId: path.basename(bundlePath),
    path: bundlePath,
    relativePath: path.relative(workspaceRoot, bundlePath).split(path.sep).join("/"),
    canonicalAt: bundleTimestamp(bundlePath, { operator: { verdict, summary }, host: { summary }, replay: {} })?.toISOString() ?? null,
    fileCount: fileStats.length,
    artifactBytes: sum(fileStats.map((entry) => entry.size)),
    fileNames: [...fileNames].sort(),
    validationOk: validation?.ok ?? null,
    validationReport: validation,
    verdict: verdict ?? null,
    status: status ?? null,
    summary: summary ?? null,
    metrics: {
      stepCount: stepRows.length,
      executedStepCount: executedSteps.length,
      skippedStepCount: stepRows.filter((step) => step.skipped === true).length,
      successStepCount: stepRows.filter((step) => step.resultClass === "success").length,
      failedStepCount: stepRows.filter((step) => step.resultClass && step.resultClass !== "success").length,
      totalStepDurationMs: sum(stepDurations),
      meanStepDurationMs: mean(stepDurations),
      medianStepDurationMs: median(stepDurations),
      p95StepDurationMs: percentile(stepDurations, 95),
      durationMs: verdict?.bundleStartedAt && stepRows.length > 0
        ? Math.max(...stepRows.map((step) => Number(step.durationMs ?? 0)))
        : null,
      runtimeLoadProofPath: verdict?.runtimeLoadProofPath ?? null,
      verdict: verdict?.verdict ?? null,
      severity: verdict?.severity ?? null,
      warnings: Array.isArray(verdict?.warnings) ? verdict.warnings.length : 0,
    },
  };
}

function summarizeReplayBundle(bundlePath, workspaceRoot) {
  const files = collectFiles(bundlePath);
  const fileStats = files.map((filePath) => {
    const stat = statSync(filePath);
    return {
      path: filePath,
      relativePath: path.relative(workspaceRoot, filePath).split(path.sep).join("/"),
      size: stat.size,
    };
  });

  const fileNames = new Set(fileStats.map((entry) => path.basename(entry.path)));
  const trace = readJsonIfExists(path.join(bundlePath, "trace.json"));
  const bundle = readJsonIfExists(path.join(bundlePath, "bundle.json"));
  const summary = readTextIfExists(path.join(bundlePath, "summary.md"));
  const summaryTables = readJsonIfExists(path.join(bundlePath, "summary-tables.json"));
  const coverage = readJsonIfExists(path.join(bundlePath, "coverage-snapshot.json"));
  const hardening = readJsonIfExists(path.join(bundlePath, "hardening-snapshot.json"));
  const hashes = readJsonIfExists(path.join(bundlePath, "hashes.json"));
  const validation = readJsonIfExists(path.join(bundlePath, "validation-report.json"));
  const traceTurns = Array.isArray(trace?.turns) ? trace.turns : [];
  const traceTurnById = new Map(traceTurns.map((turn) => [turn.turnId, turn]));
  const traceFeedback = summarizeTraceFeedback(traceTurns);
  const pricingTable = loadPricingTable();

  const modes = Array.isArray(bundle?.modes) ? bundle.modes : [];
  const winnerMode = summaryTables?.winnerMode ?? bundle?.summary?.winnerMode ?? null;
  const ranking = Array.isArray(summaryTables?.ranking) ? summaryTables.ranking : [];
  const winnerScore = ranking.length > 0 ? Number(ranking[0]?.qualityScore ?? 0) : null;
  const qualityScores = ranking.map((row) => Number(row.qualityScore ?? 0)).filter(Number.isFinite);
  const modeNames = modes.map((mode) => mode.mode).filter(Boolean);
  const savingsByMode = modes.map((mode) => summarizeReplayModeSavings(mode, traceTurnById, pricingTable));
  const selectedContextChars = sum(savingsByMode.map((mode) => mode.selectedContextChars));
  const selectedContextBlockCount = sum(savingsByMode.map((mode) => mode.selectedContextBlockCount));
  const estimatedPromptTokens = savingsByMode.length > 0 ? sum(savingsByMode.map((mode) => mode.estimatedPromptTokens)) : null;
  const completionChars = sumOrNull(savingsByMode.map((mode) => mode.completionChars));
  const estimatedCompletionTokens = sumOrNull(savingsByMode.map((mode) => mode.estimatedCompletionTokens));
  const estimatedPromptCostUsd = sumOrNull(savingsByMode.map((mode) => mode.estimatedPromptCostUsd));
  const estimatedCompletionCostUsd = sumOrNull(savingsByMode.map((mode) => mode.estimatedCompletionCostUsd));
  const estimatedTotalCostUsd = sumOrNull(savingsByMode.map((mode) => mode.estimatedTotalCostUsd));
  const turnsWithSelectedContextCount = sum(savingsByMode.map((mode) => mode.turnsWithSelectedContextCount));
  const totalTurnCount = sum(savingsByMode.map((mode) => mode.turnCount));
  const totalRetrievalToolHopCount = sum(savingsByMode.map((mode) => mode.retrievalToolHopCount));
  const totalRetrievalToolHopTurnCount = sum(savingsByMode.map((mode) => mode.retrievalToolHopTurnCount));

  return {
    kind: "recorded-session-replay",
    bundleId: path.basename(bundlePath),
    path: bundlePath,
    relativePath: path.relative(workspaceRoot, bundlePath).split(path.sep).join("/"),
    canonicalAt: bundleTimestamp(bundlePath, { replay: { bundle }, operator: {}, host: {} })?.toISOString() ?? null,
    fileCount: fileStats.length,
    artifactBytes: sum(fileStats.map((entry) => entry.size)),
    fileNames: [...fileNames].sort(),
    validationOk: validation?.ok ?? null,
    validationReport: validation,
    summary,
    summaryTables: summaryTables
      ? {
          winnerMode: summaryTables.winnerMode ?? null,
          ranking: Array.isArray(summaryTables.ranking) ? summaryTables.ranking : [],
        }
      : null,
    coverage: coverage
      ? {
          totalTurns: coverage.totalTurns ?? null,
          compileOkRate: coverage.compileOkRate ?? null,
          phraseHitRate: coverage.phraseHitRate ?? null,
          modes: Array.isArray(coverage.modes)
            ? coverage.modes.map((mode) => ({
                mode: mode.mode ?? null,
                compileOkRate: mode.compileOkRate ?? null,
                phraseHitRate: mode.phraseHitRate ?? null,
                learnedRouteTurnRate: mode.learnedRouteTurnRate ?? null,
              }))
            : [],
        }
      : null,
    hardening: hardening
      ? {
          warnings: Array.isArray(hardening.warnings) ? hardening.warnings : [],
          clipRate: hardening.clipRate ?? null,
          failOpenRate: hardening.failOpenRate ?? null,
        }
      : null,
    hashes: hashes
      ? {
          traceHash: hashes.traceHash ?? bundle?.traceHash ?? null,
          fixtureHash: hashes.fixtureHash ?? bundle?.fixtureHash ?? null,
          scoreHash: hashes.scoreHash ?? bundle?.scoreHash ?? null,
          bundleHash: hashes.bundleHash ?? bundle?.bundleHash ?? null,
        }
      : {
          traceHash: bundle?.traceHash ?? null,
          fixtureHash: bundle?.fixtureHash ?? null,
          scoreHash: bundle?.scoreHash ?? null,
          bundleHash: bundle?.bundleHash ?? null,
        },
    metrics: {
      traceId: bundle?.traceId ?? trace?.traceId ?? null,
      modeCount: modes.length,
      winnerMode,
      winnerScore,
      qualityScoreMean: mean(qualityScores),
      qualityScoreMedian: median(qualityScores),
      qualityScoreSpread: qualityScores.length > 0 ? Math.max(...qualityScores) - Math.min(...qualityScores) : null,
      totalTurns: coverage?.totalTurns ?? null,
      compileOkRate: coverage?.compileOkRate ?? null,
      phraseHitRate: coverage?.phraseHitRate ?? null,
      learnedRouteTurnRate: coverage?.modes ? mean(coverage.modes.map((mode) => Number(mode.learnedRouteTurnRate ?? 0))) : null,
      modeNames,
      bundleHash: bundle?.bundleHash ?? null,
      traceHash: bundle?.traceHash ?? null,
      fixtureHash: bundle?.fixtureHash ?? null,
      scoreHash: bundle?.scoreHash ?? null,
      validatedFileCount: validation?.verifiedFileCount ?? null,
      savingsByMode,
      selectedContextChars: savingsByMode.length > 0 ? selectedContextChars : null,
      selectedContextBlockCount: savingsByMode.length > 0 ? selectedContextBlockCount : null,
      estimatedPromptTokens,
      completionChars,
      estimatedCompletionTokens,
      estimatedPromptCostUsd,
      estimatedCompletionCostUsd,
      estimatedTotalCostUsd,
      pricingTableVersion: pricingTable?.version ?? null,
      pricingTablePath: pricingTable?.path ?? null,
      retrievalToolHopCount: savingsByMode.length > 0 ? totalRetrievalToolHopCount : null,
      retrievalToolHopTurnCount: savingsByMode.length > 0 ? totalRetrievalToolHopTurnCount : null,
      retrievalToolHopTurnRate: totalTurnCount > 0 ? round(totalRetrievalToolHopTurnCount / totalTurnCount, 4) : null,
      turnsWithSelectedContextCount: savingsByMode.length > 0 ? turnsWithSelectedContextCount : null,
      turnsWithSelectedContextRate: totalTurnCount > 0 ? round(turnsWithSelectedContextCount / totalTurnCount, 4) : null,
      feedbackEventCount: traceFeedback.feedbackEventCount,
      nonApprovalFeedbackEventCount: traceFeedback.nonApprovalFeedbackEventCount,
      turnsWithFeedbackCount: traceFeedback.turnsWithFeedbackCount,
      turnsWithNonApprovalFeedbackCount: traceFeedback.turnsWithNonApprovalFeedbackCount,
      turnsWithNonApprovalFeedbackRate: traceFeedback.turnsWithNonApprovalFeedbackRate,
    },
  };
}

function summarizeHostBundle(bundlePath, workspaceRoot) {
  const files = collectFiles(bundlePath);
  const fileStats = files.map((filePath) => {
    const stat = statSync(filePath);
    return {
      path: filePath,
      relativePath: path.relative(workspaceRoot, filePath).split(path.sep).join("/"),
      size: stat.size,
    };
  });

  const fileNames = new Set(fileStats.map((entry) => path.basename(entry.path)));
  const summary = readTextIfExists(path.join(bundlePath, "summary.md"));
  const status = readJsonIfExists(path.join(bundlePath, "status.json"));
  const validation = readJsonIfExists(path.join(bundlePath, "validation-report.json"));

  const securitySummary = status?.securityAudit?.summary ?? null;
  const recentSession = status?.sessions?.recent?.[0] ?? null;
  const decisionSummary = status?.recentDecisionSummary ?? null;
  const promotionSummary = status?.promotionStory?.summary ?? null;

  return {
    kind: "host-evidence",
    bundleId: path.basename(bundlePath),
    path: bundlePath,
    relativePath: path.relative(workspaceRoot, bundlePath).split(path.sep).join("/"),
    canonicalAt: bundleTimestamp(bundlePath, { host: { summary }, replay: {}, operator: {} })?.toISOString() ?? null,
    fileCount: fileStats.length,
    artifactBytes: sum(fileStats.map((entry) => entry.size)),
    fileNames: [...fileNames].sort(),
    validationOk: validation?.ok ?? null,
    validationReport: validation,
    summary,
    statusSummary: status
      ? {
          runtimeVersion: status.runtimeVersion ?? null,
          workerHealthy: status.workerHealthy ?? null,
          workerMode: status.workerMode ?? null,
          currentPackVersion: status.currentPackVersion ?? null,
          gatewayReachable: status.gateway?.reachable ?? null,
          memoryFiles: status.memory?.files ?? null,
          memoryChunks: status.memory?.chunks ?? null,
          sessionCount: status.sessions?.count ?? null,
          securityCriticalCount: status.securityAudit?.summary?.critical ?? null,
          securityWarnCount: status.securityAudit?.summary?.warn ?? null,
          decisionSampleSize: status.recentDecisionSummary?.sampleSize ?? null,
          clipRate: status.recentDecisionSummary?.clipRate?.rate ?? null,
          failOpenRate: status.recentDecisionSummary?.failOpenRate?.rate ?? null,
        }
      : null,
    metrics: {
      runtimeVersion: status?.runtimeVersion ?? null,
      gatewayReachable: status?.gateway?.reachable ?? null,
      workerHealthy: status?.workerHealthy ?? null,
      memoryFiles: status?.memory?.files ?? null,
      memoryChunks: status?.memory?.chunks ?? null,
      sessionCount: status?.sessions?.count ?? null,
      securityCriticalCount: securitySummary?.critical ?? null,
      securityWarnCount: securitySummary?.warn ?? null,
      recentSessionPercentUsed: recentSession?.percentUsed ?? null,
      clipRate: decisionSummary?.clipRate?.rate ?? null,
      failOpenRate: decisionSummary?.failOpenRate?.rate ?? null,
      decisionSampleSize: decisionSummary?.sampleSize ?? null,
      currentPackVersion: promotionSummary?.currentPackVersion ?? null,
    },
  };
}

function summarizeGenericBundle(bundlePath, workspaceRoot) {
  const files = collectFiles(bundlePath);
  const fileStats = files.map((filePath) => {
    const stat = statSync(filePath);
    return {
      path: filePath,
      relativePath: path.relative(workspaceRoot, filePath).split(path.sep).join("/"),
      size: stat.size,
    };
  });
  const fileNames = new Set(fileStats.map((entry) => path.basename(entry.path)));
  const validation = readJsonIfExists(path.join(bundlePath, "validation-report.json"));
  const summary = readTextIfExists(path.join(bundlePath, "summary.md"));

  return {
    kind: "generic-proof",
    bundleId: path.basename(bundlePath),
    path: bundlePath,
    relativePath: path.relative(workspaceRoot, bundlePath).split(path.sep).join("/"),
    canonicalAt: parseTimestampFromPath(bundlePath)?.toISOString() ?? null,
    fileCount: fileStats.length,
    artifactBytes: sum(fileStats.map((entry) => entry.size)),
    fileNames: [...fileNames].sort(),
    validationOk: validation?.ok ?? null,
    validationReport: validation,
    summary,
    metrics: {},
  };
}

function summarizeBundle(bundlePath, kind, workspaceRoot) {
  if (kind === "operator-proof") {
    return summarizeOperatorBundle(bundlePath, workspaceRoot);
  }
  if (kind === "recorded-session-replay") {
    return summarizeReplayBundle(bundlePath, workspaceRoot);
  }
  if (kind === "host-evidence") {
    return summarizeHostBundle(bundlePath, workspaceRoot);
  }
  return summarizeGenericBundle(bundlePath, workspaceRoot);
}

function ageDaysFrom(now, canonicalAt) {
  if (!canonicalAt) {
    return null;
  }
  return round((now.getTime() - Date.parse(canonicalAt)) / (24 * 60 * 60 * 1000));
}

function summarizeScan(bundles, now, workspaceRoot) {
  return bundles
    .map((bundle) => ({
      ...bundle,
      ...summarizeBundle(bundle.path, bundle.kind, workspaceRoot),
    }))
    .map((bundle) => ({
      ...bundle,
      ageDays: ageDaysFrom(now, bundle.canonicalAt),
    }))
    .sort((left, right) => {
      const leftMs = left.canonicalAt ? Date.parse(left.canonicalAt) : 0;
      const rightMs = right.canonicalAt ? Date.parse(right.canonicalAt) : 0;
      if (rightMs !== leftMs) {
        return rightMs - leftMs;
      }
      return left.relativePath.localeCompare(right.relativePath);
    });
}

function summarizeStatus(statusProbe) {
  const status = statusProbe.parsed ?? {};
  const brainStatus = status.brainStatus ?? {};
  const brain = status.brain ?? {};
  const passiveLearning = status.passiveLearning ?? {};
  const currentTurnAttribution = status.currentTurnAttribution ?? {};
  const legacyWorkerHealthy = status.workerHealthy ?? null;
  const serveState = brainStatus.serveState ?? null;
  const activePackId = brain.activePackId ?? null;
  const runtimeHealthy =
    legacyWorkerHealthy ??
    (brainStatus.status === "ok"
      ? true
      : serveState === "serving_active_pack"
        ? true
        : brainStatus.status === "fail"
          ? false
          : null);
  return {
    runtimeVersion: status.runtimeVersion ?? null,
    updateSha: status.update?.git?.sha ?? null,
    runtimeHealthy,
    serveState,
    brainStatus: brainStatus.status ?? null,
    activePackId,
    routeFreshness: brain.routeFreshness ?? null,
    usedLearnedRouteFn: brainStatus.usedLearnedRouteFn ?? currentTurnAttribution.usedLearnedRouteFn ?? null,
    loadProof: status.hook?.loadProof ?? null,
    workerHealthy: legacyWorkerHealthy,
    workerStatus: status.workerStatus ?? passiveLearning.watchState ?? null,
    workerMode: status.workerMode ?? null,
    workerPid: status.workerPid ?? null,
    currentPackVersion: status.currentPackVersion ?? activePackId,
    packVersion: status.packVersion ?? null,
    recentTraceCount: status.recentTraceCount ?? null,
    totalEpisodes: status.totalEpisodes ?? null,
    pendingObservations: status.pendingObservations ?? null,
    pendingLabels: status.pendingLabels ?? null,
    mutationBacklog: status.mutationBacklog ?? null,
    recentMutationBundles: Array.isArray(status.recentMutationBundles) ? status.recentMutationBundles.length : null,
    decisionSummary: status.recentDecisionSummary ?? status.decisionSummary ?? null,
    promotionStory: status.promotionStory ?? null,
    securityAudit: status.securityAudit?.summary ?? null,
    gatewayReachable: status.gateway?.reachable ?? null,
    memoryFiles: status.memory?.files ?? null,
    memoryChunks: status.memory?.chunks ?? null,
    nodeCount: status.nodeCount ?? null,
    edgeCount: status.edgeCount ?? null,
    avgReward: status.avgReward ?? null,
    churn: status.churn ?? null,
  };
}

function latestBundlesByKind(bundles) {
  const byKind = new Map();
  for (const bundle of bundles) {
    const existing = byKind.get(bundle.kind);
    if (!existing) {
      byKind.set(bundle.kind, bundle);
      continue;
    }
    const existingMs = existing.canonicalAt ? Date.parse(existing.canonicalAt) : 0;
    const currentMs = bundle.canonicalAt ? Date.parse(bundle.canonicalAt) : 0;
    if (currentMs >= existingMs) {
      byKind.set(bundle.kind, bundle);
    }
  }
  return Object.fromEntries(byKind.entries());
}

function summarizePerformance(bundles, statusProbe, scanDurationMs) {
  const operatorBundles = bundles.filter((bundle) => bundle.kind === "operator-proof");
  const replayBundles = bundles.filter((bundle) => bundle.kind === "recorded-session-replay");

  const operatorDurations = operatorBundles.flatMap((bundle) => [bundle.metrics?.totalStepDurationMs ?? null].filter(Number.isFinite));
  const replayFileCounts = replayBundles.map((bundle) => bundle.fileCount ?? 0);
  const replayBytes = replayBundles.map((bundle) => bundle.artifactBytes ?? 0);
  const replayQuality = replayBundles.map((bundle) => bundle.metrics?.winnerScore ?? null).filter(Number.isFinite);
  const replayCompileRate = replayBundles.map((bundle) => bundle.metrics?.compileOkRate ?? null).filter(Number.isFinite);
  const replayPhraseRate = replayBundles.map((bundle) => bundle.metrics?.phraseHitRate ?? null).filter(Number.isFinite);
  const replayContextChars = replayBundles.map((bundle) => bundle.metrics?.selectedContextChars ?? null).filter(Number.isFinite);
  const replayContextBlocks = replayBundles.map((bundle) => bundle.metrics?.selectedContextBlockCount ?? null).filter(Number.isFinite);
  const replayEstimatedPromptTokens = replayBundles.map((bundle) => bundle.metrics?.estimatedPromptTokens ?? null).filter(Number.isFinite);
  const replayCompletionChars = replayBundles.map((bundle) => bundle.metrics?.completionChars ?? null).filter(Number.isFinite);
  const replayEstimatedCompletionTokens = replayBundles.map((bundle) => bundle.metrics?.estimatedCompletionTokens ?? null).filter(Number.isFinite);
  const replayRetrievalToolHopCount = replayBundles.map((bundle) => bundle.metrics?.retrievalToolHopCount ?? null).filter(Number.isFinite);
  const replayRetrievalToolHopTurnCount = replayBundles.map((bundle) => bundle.metrics?.retrievalToolHopTurnCount ?? null).filter(Number.isFinite);
  const replayFeedbackEvents = replayBundles.map((bundle) => bundle.metrics?.feedbackEventCount ?? null).filter(Number.isFinite);
  const replayNonApprovalFeedbackEvents = replayBundles.map((bundle) => bundle.metrics?.nonApprovalFeedbackEventCount ?? null).filter(Number.isFinite);
  const replayTurnsWithNonApprovalFeedback = replayBundles.map((bundle) => bundle.metrics?.turnsWithNonApprovalFeedbackCount ?? null).filter(Number.isFinite);
  const replayEstimatedPromptCostUsd = replayBundles.map((bundle) => bundle.metrics?.estimatedPromptCostUsd ?? null).filter(Number.isFinite);
  const replayEstimatedCompletionCostUsd = replayBundles.map((bundle) => bundle.metrics?.estimatedCompletionCostUsd ?? null).filter(Number.isFinite);
  const replayEstimatedTotalCostUsd = replayBundles.map((bundle) => bundle.metrics?.estimatedTotalCostUsd ?? null).filter(Number.isFinite);

  return {
    statusProbeMs: statusProbe.durationMs,
    scanMs: scanDurationMs,
    operatorProofCount: operatorBundles.length,
    replayProofCount: replayBundles.length,
    operatorStepMsTotal: sum(operatorDurations),
    operatorStepMsMean: mean(operatorDurations),
    operatorStepMsMedian: median(operatorDurations),
    operatorStepMsP95: percentile(operatorDurations, 95),
    replayFileCountMean: mean(replayFileCounts),
    replayArtifactBytesTotal: sum(replayBytes),
    replayArtifactBytesMean: mean(replayBytes),
    replayWinnerScoreMean: mean(replayQuality),
    replayCompileRateMean: mean(replayCompileRate),
    replayPhraseRateMean: mean(replayPhraseRate),
    replayContextCharsTotal: sum(replayContextChars),
    replayContextCharsMean: mean(replayContextChars),
    replayContextBlockTotal: sum(replayContextBlocks),
    replayContextBlockMean: mean(replayContextBlocks),
    replayEstimatedPromptTokensTotal: sum(replayEstimatedPromptTokens),
    replayEstimatedPromptTokensMean: mean(replayEstimatedPromptTokens),
    replayCompletionCharsTotal: sumOrNull(replayCompletionChars),
    replayCompletionCharsMean: mean(replayCompletionChars),
    replayEstimatedCompletionTokensTotal: sumOrNull(replayEstimatedCompletionTokens),
    replayEstimatedCompletionTokensMean: mean(replayEstimatedCompletionTokens),
    replayRetrievalToolHopCountTotal: sum(replayRetrievalToolHopCount),
    replayRetrievalToolHopTurnCountTotal: sum(replayRetrievalToolHopTurnCount),
    replayFeedbackEventCountTotal: sum(replayFeedbackEvents),
    replayNonApprovalFeedbackEventCountTotal: sum(replayNonApprovalFeedbackEvents),
    replayTurnsWithNonApprovalFeedbackCountTotal: sum(replayTurnsWithNonApprovalFeedback),
    replayEstimatedPromptCostUsdTotal: sumOrNull(replayEstimatedPromptCostUsd),
    replayEstimatedCompletionCostUsdTotal: sumOrNull(replayEstimatedCompletionCostUsd),
    replayEstimatedTotalCostUsdTotal: sumOrNull(replayEstimatedTotalCostUsd),
  };
}

function buildCostProxy(performance, bundles) {
  const proofMinutes = round(((performance.statusProbeMs ?? 0) + (performance.scanMs ?? 0) + (performance.operatorStepMsTotal ?? 0)) / 60000, 4);
  const artifactBytes = sum(bundles.map((bundle) => bundle.artifactBytes ?? 0));
  return {
    proofMinutes,
    bundleCount: bundles.length,
    operatorProofCount: performance.operatorProofCount,
    replayProofCount: performance.replayProofCount,
    artifactBytes,
    artifactMB: round(artifactBytes / (1024 * 1024), 3),
    statusProbeMs: performance.statusProbeMs,
    scanMs: performance.scanMs,
  };
}

function freshnessBand(ageDays, healthFreshnessDays, freshnessThresholdDays) {
  if (ageDays === null) {
    return "unknown";
  }
  if (ageDays <= healthFreshnessDays) {
    return "fresh";
  }
  if (ageDays <= freshnessThresholdDays) {
    return "warm";
  }
  return "stale";
}

function buildHealthSnapshot({ config, statusProbe, bundles, now, scanDurationMs }) {
  const status = summarizeStatus(statusProbe);
  const latest = latestBundlesByKind(bundles);
  const latestOperator = latest["operator-proof"] ?? null;
  const latestReplay = latest["recorded-session-replay"] ?? null;
  const latestHost = latest["host-evidence"] ?? null;
  const performance = summarizePerformance(bundles, statusProbe, scanDurationMs);
  const costProxy = buildCostProxy(performance, bundles);
  const healthFreshnessDays = Number(config.healthFreshnessDays ?? 7);
  const freshnessThresholdDays = Number(config.freshnessThresholdDays ?? 21);

  const bundleFreshness = [latestOperator, latestReplay, latestHost]
    .filter(Boolean)
    .map((bundle) => ({
      kind: bundle.kind,
      bundleId: bundle.bundleId,
      relativePath: bundle.relativePath,
      canonicalAt: bundle.canonicalAt,
      ageDays: bundle.ageDays,
      freshness: freshnessBand(bundle.ageDays, healthFreshnessDays, freshnessThresholdDays),
      validationOk: bundle.validationOk,
      metrics: bundle.metrics,
    }));

  return {
    contract: CONTRACT,
    generatedAt: now.toISOString(),
    probe: {
      command: statusProbe.command,
      startedAt: statusProbe.startedAt,
      endedAt: statusProbe.endedAt,
      durationMs: statusProbe.durationMs,
      exitCode: statusProbe.exitCode,
      signal: statusProbe.signal,
    },
    status,
    latestBundles: bundleFreshness,
    performance,
    costProxy,
    replaySavings: latestReplay?.metrics?.savingsByMode ?? [],
    replayCostProxy: latestReplay
      ? {
          pricingTableVersion: latestReplay.metrics?.pricingTableVersion ?? null,
          pricingTablePath: latestReplay.metrics?.pricingTablePath ?? null,
          estimatedPromptCostUsd: latestReplay.metrics?.estimatedPromptCostUsd ?? null,
          estimatedCompletionCostUsd: latestReplay.metrics?.estimatedCompletionCostUsd ?? null,
          estimatedTotalCostUsd: latestReplay.metrics?.estimatedTotalCostUsd ?? null,
        }
      : null,
    proofInventory: {
      bundleCount: bundles.length,
      operatorProofCount: bundles.filter((bundle) => bundle.kind === "operator-proof").length,
      replayProofCount: bundles.filter((bundle) => bundle.kind === "recorded-session-replay").length,
      hostEvidenceCount: bundles.filter((bundle) => bundle.kind === "host-evidence").length,
      genericProofCount: bundles.filter((bundle) => bundle.kind === "generic-proof").length,
      validationOkCount: bundles.filter((bundle) => bundle.validationOk === true).length,
      validationFailCount: bundles.filter((bundle) => bundle.validationOk === false).length,
    },
  };
}

function buildNightlyAggregate({ config, bundles, now, scanDurationMs }) {
  const operatorBundles = bundles.filter((bundle) => bundle.kind === "operator-proof");
  const replayBundles = bundles.filter((bundle) => bundle.kind === "recorded-session-replay");
  const hostBundles = bundles.filter((bundle) => bundle.kind === "host-evidence");
  const genericBundles = bundles.filter((bundle) => bundle.kind === "generic-proof");
  const healthFreshnessDays = Number(config.healthFreshnessDays ?? 7);
  const freshnessThresholdDays = Number(config.freshnessThresholdDays ?? 21);

  const freshnessCounts = {
    fresh: 0,
    warm: 0,
    stale: 0,
    unknown: 0,
  };

  for (const bundle of bundles) {
    const band = freshnessBand(bundle.ageDays, healthFreshnessDays, freshnessThresholdDays);
    freshnessCounts[band] = (freshnessCounts[band] ?? 0) + 1;
  }

  const replayScores = replayBundles.map((bundle) => bundle.metrics?.winnerScore ?? null).filter(Number.isFinite);
  const replayCompileRates = replayBundles.map((bundle) => bundle.metrics?.compileOkRate ?? null).filter(Number.isFinite);
  const replayPhraseRates = replayBundles.map((bundle) => bundle.metrics?.phraseHitRate ?? null).filter(Number.isFinite);
  const replayLearnedRouteRates = replayBundles.map((bundle) => bundle.metrics?.learnedRouteTurnRate ?? null).filter(Number.isFinite);
  const replayPricingTableVersions = [...new Set(replayBundles.map((bundle) => bundle.metrics?.pricingTableVersion ?? null).filter((value) => typeof value === "string" && value.length > 0))];
  const replayPricingTablePaths = [...new Set(replayBundles.map((bundle) => bundle.metrics?.pricingTablePath ?? null).filter((value) => typeof value === "string" && value.length > 0))];
  const replaySavingsByMode = aggregateReplaySavings(
    replayBundles.flatMap((bundle) => Array.isArray(bundle.metrics?.savingsByMode) ? bundle.metrics.savingsByMode : []),
  );
  const operatorStepDurations = operatorBundles.map((bundle) => bundle.metrics?.totalStepDurationMs ?? null).filter(Number.isFinite);
  const operatorFileSizes = operatorBundles.map((bundle) => bundle.artifactBytes ?? 0);
  const replayFileSizes = replayBundles.map((bundle) => bundle.artifactBytes ?? 0);

  const bundleTypeCounts = {
    operatorProof: operatorBundles.length,
    recordedSessionReplay: replayBundles.length,
    hostEvidence: hostBundles.length,
    genericProof: genericBundles.length,
  };

  const validationCounts = {
    ok: bundles.filter((bundle) => bundle.validationOk === true).length,
    fail: bundles.filter((bundle) => bundle.validationOk === false).length,
    unknown: bundles.filter((bundle) => bundle.validationOk === null || bundle.validationOk === undefined).length,
  };

  const winnerModeCounts = {};
  for (const bundle of replayBundles) {
    const winnerMode = bundle.metrics?.winnerMode ?? "unknown";
    winnerModeCounts[winnerMode] = (winnerModeCounts[winnerMode] ?? 0) + 1;
  }

  const latestReplay = latestBundlesByKind(replayBundles)["recorded-session-replay"] ?? null;
  const latestOperator = latestBundlesByKind(operatorBundles)["operator-proof"] ?? null;
  const latestHost = latestBundlesByKind(hostBundles)["host-evidence"] ?? null;

  const aggregate = {
    contract: CONTRACT,
    generatedAt: now.toISOString(),
    scanMs: scanDurationMs,
    bundles: bundles.map((bundle) => ({
      kind: bundle.kind,
      bundleId: bundle.bundleId,
      relativePath: bundle.relativePath,
      canonicalAt: bundle.canonicalAt,
      ageDays: bundle.ageDays,
      fileCount: bundle.fileCount,
      artifactBytes: bundle.artifactBytes,
      validationOk: bundle.validationOk,
      metrics: bundle.metrics,
    })),
    bundleTypeCounts,
    freshnessCounts,
    validationCounts,
    replayMetrics: {
      winnerModeCounts,
      winnerScoreMean: mean(replayScores),
      winnerScoreMedian: median(replayScores),
      compileRateMean: mean(replayCompileRates),
      phraseRateMean: mean(replayPhraseRates),
      learnedRouteRateMean: mean(replayLearnedRouteRates),
      pricingTableVersion: replayPricingTableVersions.length === 1 ? replayPricingTableVersions[0] : replayPricingTableVersions.length > 1 ? "mixed" : null,
      pricingTablePath: replayPricingTablePaths.length === 1 ? replayPricingTablePaths[0] : replayPricingTablePaths.length > 1 ? "mixed" : null,
      replayFileBytesTotal: sum(replayFileSizes),
      replayFileBytesMean: mean(replayFileSizes),
      totalTurns: sum(replayBundles.map((bundle) => bundle.metrics?.totalTurns ?? 0)),
      savingsByMode: replaySavingsByMode,
      selectedContextCharsTotal: sum(replaySavingsByMode.map((mode) => mode.selectedContextChars ?? 0)),
      selectedContextBlocksTotal: sum(replaySavingsByMode.map((mode) => mode.selectedContextBlockCount ?? 0)),
      estimatedPromptTokensTotal: sum(replaySavingsByMode.map((mode) => mode.estimatedPromptTokens ?? 0)),
      completionCharsTotal: sumOrNull(replaySavingsByMode.map((mode) => mode.completionChars)),
      estimatedCompletionTokensTotal: sumOrNull(replaySavingsByMode.map((mode) => mode.estimatedCompletionTokens)),
      estimatedPromptCostUsdTotal: sumOrNull(replaySavingsByMode.map((mode) => mode.estimatedPromptCostUsd)),
      estimatedCompletionCostUsdTotal: sumOrNull(replaySavingsByMode.map((mode) => mode.estimatedCompletionCostUsd)),
      estimatedTotalCostUsdTotal: sumOrNull(replaySavingsByMode.map((mode) => mode.estimatedTotalCostUsd)),
      turnsWithSelectedContextTotal: sum(replaySavingsByMode.map((mode) => mode.turnsWithSelectedContextCount ?? 0)),
      retrievalToolHopCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.retrievalToolHopCount ?? 0)),
      retrievalToolHopTurnCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.retrievalToolHopTurnCount ?? 0)),
      feedbackEventCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.feedbackEventCount ?? 0)),
      nonApprovalFeedbackEventCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.nonApprovalFeedbackEventCount ?? 0)),
      turnsWithNonApprovalFeedbackCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.turnsWithNonApprovalFeedbackCount ?? 0)),
    },
    operatorMetrics: {
      stepMsTotal: sum(operatorStepDurations),
      stepMsMean: mean(operatorStepDurations),
      stepMsMedian: median(operatorStepDurations),
      stepMsP95: percentile(operatorStepDurations, 95),
      operatorFileBytesTotal: sum(operatorFileSizes),
      operatorFileBytesMean: mean(operatorFileSizes),
      totalSteps: sum(operatorBundles.map((bundle) => bundle.metrics?.stepCount ?? 0)),
    },
    hostMetrics: {
      securityCriticalTotal: sum(hostBundles.map((bundle) => bundle.metrics?.securityCriticalCount ?? 0)),
      securityWarnTotal: sum(hostBundles.map((bundle) => bundle.metrics?.securityWarnCount ?? 0)),
      gatewayReachableCount: hostBundles.filter((bundle) => bundle.metrics?.gatewayReachable === true).length,
      workerHealthyCount: hostBundles.filter((bundle) => bundle.metrics?.workerHealthy === true).length,
      memoryFilesTotal: sum(hostBundles.map((bundle) => bundle.metrics?.memoryFiles ?? 0)),
      sessionCountTotal: sum(hostBundles.map((bundle) => bundle.metrics?.sessionCount ?? 0)),
    },
    latestBundles: {
      operatorProof: latestOperator,
      recordedSessionReplay: latestReplay,
      hostEvidence: latestHost,
    },
    performance: {
      scanMs: scanDurationMs,
      replayBundleCount: replayBundles.length,
      operatorBundleCount: operatorBundles.length,
      hostBundleCount: hostBundles.length,
      replayContextCharsTotal: sum(replaySavingsByMode.map((mode) => mode.selectedContextChars ?? 0)),
      replaySelectedContextBlocksTotal: sum(replaySavingsByMode.map((mode) => mode.selectedContextBlockCount ?? 0)),
      replayEstimatedPromptTokensTotal: sum(replaySavingsByMode.map((mode) => mode.estimatedPromptTokens ?? 0)),
      replayRetrievalToolHopCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.retrievalToolHopCount ?? 0)),
      replayRetrievalToolHopTurnCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.retrievalToolHopTurnCount ?? 0)),
      replayFeedbackEventCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.feedbackEventCount ?? 0)),
      replayNonApprovalFeedbackEventCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.nonApprovalFeedbackEventCount ?? 0)),
      replayTurnsWithNonApprovalFeedbackCountTotal: sum(replayBundles.map((bundle) => bundle.metrics?.turnsWithNonApprovalFeedbackCount ?? 0)),
    },
    costProxy: {
      proofMinutes: round((scanDurationMs + sum(operatorStepDurations)) / 60000, 4),
      artifactBytes: sum(bundles.map((bundle) => bundle.artifactBytes ?? 0)),
      artifactMB: round(sum(bundles.map((bundle) => bundle.artifactBytes ?? 0)) / (1024 * 1024), 3),
      bundleCount: bundles.length,
      validationCount: validationCounts.ok + validationCounts.fail,
    },
  };

  return aggregate;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return "n/a";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kib = bytes / 1024;
  if (kib < 1024) {
    return `${round(kib, 1)} KiB`;
  }
  return `${round(kib / 1024, 2)} MiB`;
}

function formatAge(ageDays) {
  if (!Number.isFinite(ageDays)) {
    return "n/a";
  }
  return `${round(ageDays, 2)}d`;
}

function formatUsd(amount) {
  if (!Number.isFinite(amount)) {
    return "n/a";
  }
  const rounded = round(amount, 6);
  if (rounded === null) {
    return "n/a";
  }
  return `$${rounded.toFixed(6)}`;
}

function formatHealthMarkdown(snapshot) {
  const lines = [];
  lines.push("# OpenClawBrain proof health snapshot");
  lines.push("");
  lines.push(`- generated at: ${snapshot.generatedAt}`);
  lines.push(`- status probe: ${snapshot.probe.command} (${snapshot.probe.durationMs} ms)`);
  lines.push(`- bundle count: ${snapshot.proofInventory.bundleCount}`);
  lines.push(`- operator proofs: ${snapshot.proofInventory.operatorProofCount}`);
  lines.push(`- replay proofs: ${snapshot.proofInventory.replayProofCount}`);
  lines.push(`- host evidence bundles: ${snapshot.proofInventory.hostEvidenceCount}`);
  lines.push("");
  lines.push("## Live status");
  lines.push(`- runtime healthy: ${snapshot.status.runtimeHealthy}`);
  lines.push(`- serve state: ${snapshot.status.serveState ?? "n/a"}`);
  lines.push(`- active pack: ${snapshot.status.activePackId ?? snapshot.status.currentPackVersion ?? "n/a"}`);
  lines.push(`- learned route active: ${snapshot.status.usedLearnedRouteFn ?? "n/a"}`);
  lines.push(`- load proof: ${snapshot.status.loadProof ?? "n/a"}`);
  lines.push(`- worker healthy: ${snapshot.status.workerHealthy}`);
  lines.push(`- worker mode: ${snapshot.status.workerMode}`);
  lines.push(`- recent traced decisions: ${snapshot.status.decisionSummary?.sampleSize ?? 0}`);
  lines.push(`- clip rate: ${snapshot.status.decisionSummary?.clipRate?.rate ?? "n/a"}`);
  lines.push(`- fail-open rate: ${snapshot.status.decisionSummary?.failOpenRate?.rate ?? "n/a"}`);
  lines.push(`- security critical findings: ${snapshot.status.securityAudit?.critical ?? 0}`);
  lines.push("");
  lines.push("## Latest bundle surface");
  for (const bundle of snapshot.latestBundles) {
    lines.push(`- ${bundle.kind}: ${bundle.relativePath} (${formatAge(bundle.ageDays)}, ${bundle.validationOk === true ? "ok" : bundle.validationOk === false ? "fail" : "unknown"})`);
  }
  lines.push("");
  lines.push("## Performance and cost proxies");
  lines.push(`- status probe ms: ${snapshot.performance.statusProbeMs}`);
  lines.push(`- scan ms: ${snapshot.performance.scanMs}`);
  lines.push(`- operator step ms total: ${round(snapshot.performance.operatorStepMsTotal ?? 0, 2)}`);
  lines.push(`- replay winner score mean: ${snapshot.performance.replayWinnerScoreMean ?? "n/a"}`);
  lines.push(`- replay context chars total: ${snapshot.performance.replayContextCharsTotal ?? "n/a"}`);
  lines.push(`- replay selected context blocks total: ${snapshot.performance.replaySelectedContextBlocksTotal ?? "n/a"}`);
  lines.push(`- replay estimated prompt tokens total: ${snapshot.performance.replayEstimatedPromptTokensTotal ?? "n/a"}`);
  lines.push(`- replay completion chars total: ${snapshot.performance.replayCompletionCharsTotal ?? "n/a"}`);
  lines.push(`- replay estimated completion tokens total: ${snapshot.performance.replayEstimatedCompletionTokensTotal ?? "n/a"}`);
  lines.push(`- replay retrieval/tool-hop count total: ${snapshot.performance.replayRetrievalToolHopCountTotal ?? "n/a"}`);
  lines.push(`- replay retrieval/tool-hop turns total: ${snapshot.performance.replayRetrievalToolHopTurnCountTotal ?? "n/a"}`);
  lines.push(`- replay feedback events total: ${snapshot.performance.replayFeedbackEventCountTotal ?? "n/a"}`);
  lines.push(`- replay non-approval feedback events total: ${snapshot.performance.replayNonApprovalFeedbackEventCountTotal ?? "n/a"}`);
  lines.push(`- replay turns with non-approval feedback total: ${snapshot.performance.replayTurnsWithNonApprovalFeedbackCountTotal ?? "n/a"}`);
  lines.push(`- replay pricing table version: ${snapshot.replayCostProxy?.pricingTableVersion ?? "n/a"}`);
  lines.push(`- replay pricing table path: ${snapshot.replayCostProxy?.pricingTablePath ?? "n/a"}`);
  lines.push(`- replay estimated prompt cost: ${formatUsd(snapshot.replayCostProxy?.estimatedPromptCostUsd)}`);
  lines.push(`- replay estimated completion cost: ${formatUsd(snapshot.replayCostProxy?.estimatedCompletionCostUsd)}`);
  lines.push(`- replay estimated total cost: ${formatUsd(snapshot.replayCostProxy?.estimatedTotalCostUsd)}`);
  lines.push(`- proof minutes proxy: ${snapshot.costProxy.proofMinutes}`);
  lines.push(`- artifact bytes scanned: ${snapshot.costProxy.artifactBytes} (${snapshot.costProxy.artifactMB} MiB)`);
  if (Array.isArray(snapshot.replaySavings) && snapshot.replaySavings.length > 0) {
    lines.push("");
    lines.push("## Replay savings proxy");
    for (const mode of snapshot.replaySavings) {
      lines.push(`- ${mode.mode}: ${mode.selectedContextChars} prompt chars, ${mode.completionChars ?? "n/a"} completion chars, ${mode.selectedContextBlockCount} blocks, ${mode.estimatedPromptTokens} estimated prompt tokens, ${mode.estimatedCompletionTokens ?? "n/a"} estimated completion tokens, ${formatUsd(mode.estimatedPromptCostUsd)} prompt cost, ${formatUsd(mode.estimatedCompletionCostUsd)} completion cost, ${formatUsd(mode.estimatedTotalCostUsd)} total cost, ${mode.retrievalToolHopCount} retrieval/tool-hop proxy count, ${mode.retrievalToolHopTurnCount} retrieval/tool-hop turns`);
    }
  }
  lines.push("");
  lines.push("## What to watch");
  if (snapshot.status.decisionSummary?.sampleSize === 0) {
    lines.push("- no recent traced decisions were found; this is a quiet surface, not a proof failure");
  }
  if (snapshot.latestBundles.length === 0) {
    lines.push("- no proof bundles matched the scan roots");
  }
  return `${lines.join("\n")}\n`;
}

function formatNightlyMarkdown(aggregate) {
  const lines = [];
  lines.push("# OpenClawBrain nightly proof aggregate");
  lines.push("");
  lines.push(`- generated at: ${aggregate.generatedAt}`);
  lines.push(`- scanned bundles: ${aggregate.bundles.length}`);
  lines.push(`- scan ms: ${aggregate.scanMs}`);
  lines.push("");
  lines.push("## Bundle counts");
  lines.push(`- operator proofs: ${aggregate.bundleTypeCounts.operatorProof}`);
  lines.push(`- replay proofs: ${aggregate.bundleTypeCounts.recordedSessionReplay}`);
  lines.push(`- host evidence bundles: ${aggregate.bundleTypeCounts.hostEvidence}`);
  lines.push(`- generic proof bundles: ${aggregate.bundleTypeCounts.genericProof}`);
  lines.push("");
  lines.push("## Freshness bands");
  lines.push(`- fresh: ${aggregate.freshnessCounts.fresh}`);
  lines.push(`- warm: ${aggregate.freshnessCounts.warm}`);
  lines.push(`- stale: ${aggregate.freshnessCounts.stale}`);
  lines.push(`- unknown: ${aggregate.freshnessCounts.unknown}`);
  lines.push("");
  lines.push("## Replay proof metrics");
  lines.push(`- winner modes: ${JSON.stringify(aggregate.replayMetrics.winnerModeCounts)}`);
  lines.push(`- mean winner score: ${aggregate.replayMetrics.winnerScoreMean ?? "n/a"}`);
  lines.push(`- mean compile rate: ${aggregate.replayMetrics.compileRateMean ?? "n/a"}`);
  lines.push(`- mean phrase-hit rate: ${aggregate.replayMetrics.phraseRateMean ?? "n/a"}`);
  lines.push(`- mean learned-route rate: ${aggregate.replayMetrics.learnedRouteRateMean ?? "n/a"}`);
  lines.push(`- replay bytes total: ${aggregate.replayMetrics.replayFileBytesTotal} (${formatBytes(aggregate.replayMetrics.replayFileBytesTotal)})`);
  lines.push(`- replay context chars total: ${aggregate.replayMetrics.selectedContextCharsTotal}`);
  lines.push(`- replay selected context blocks total: ${aggregate.replayMetrics.selectedContextBlocksTotal}`);
  lines.push(`- replay estimated prompt tokens total: ${aggregate.replayMetrics.estimatedPromptTokensTotal}`);
  lines.push(`- replay completion chars total: ${aggregate.replayMetrics.completionCharsTotal ?? "n/a"}`);
  lines.push(`- replay estimated completion tokens total: ${aggregate.replayMetrics.estimatedCompletionTokensTotal ?? "n/a"}`);
  lines.push(`- replay retrieval/tool-hop count total: ${aggregate.replayMetrics.retrievalToolHopCountTotal}`);
  lines.push(`- replay retrieval/tool-hop turns total: ${aggregate.replayMetrics.retrievalToolHopTurnCountTotal}`);
  lines.push(`- replay feedback events total: ${aggregate.replayMetrics.feedbackEventCountTotal}`);
  lines.push(`- replay non-approval feedback events total: ${aggregate.replayMetrics.nonApprovalFeedbackEventCountTotal}`);
  lines.push(`- replay turns with non-approval feedback total: ${aggregate.replayMetrics.turnsWithNonApprovalFeedbackCountTotal}`);
  lines.push(`- pricing table version: ${aggregate.replayMetrics.pricingTableVersion ?? "n/a"}`);
  lines.push(`- pricing table path: ${aggregate.replayMetrics.pricingTablePath ?? "n/a"}`);
  lines.push(`- replay estimated prompt cost total: ${formatUsd(aggregate.replayMetrics.estimatedPromptCostUsdTotal)}`);
  lines.push(`- replay estimated completion cost total: ${formatUsd(aggregate.replayMetrics.estimatedCompletionCostUsdTotal)}`);
  lines.push(`- replay estimated total cost total: ${formatUsd(aggregate.replayMetrics.estimatedTotalCostUsdTotal)}`);
  lines.push("");
  lines.push("## Replay savings proxy");
  lines.push("| mode | prompt chars | completion chars | selected blocks | estimated prompt tokens | estimated completion tokens | prompt cost USD | completion cost USD | total cost USD | retrieval/tool-hop proxy count | retrieval/tool-hop turns | turns with context | turn coverage | retrieval/tool-hop turn rate |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const mode of aggregate.replayMetrics.savingsByMode) {
    lines.push(`| ${mode.mode} | ${mode.selectedContextChars} | ${mode.completionChars ?? "n/a"} | ${mode.selectedContextBlockCount} | ${mode.estimatedPromptTokens} | ${mode.estimatedCompletionTokens ?? "n/a"} | ${formatUsd(mode.estimatedPromptCostUsd)} | ${formatUsd(mode.estimatedCompletionCostUsd)} | ${formatUsd(mode.estimatedTotalCostUsd)} | ${mode.retrievalToolHopCount} | ${mode.retrievalToolHopTurnCount} | ${mode.turnsWithSelectedContextCount} | ${mode.turnsWithSelectedContextRate ?? "n/a"} | ${mode.retrievalToolHopTurnRate ?? "n/a"} |`);
  }
  lines.push("");
  lines.push("## Operator proof performance");
  lines.push(`- step ms total: ${round(aggregate.operatorMetrics.stepMsTotal ?? 0, 2)}`);
  lines.push(`- step ms mean: ${aggregate.operatorMetrics.stepMsMean ?? "n/a"}`);
  lines.push(`- step ms median: ${aggregate.operatorMetrics.stepMsMedian ?? "n/a"}`);
  lines.push(`- step ms p95: ${aggregate.operatorMetrics.stepMsP95 ?? "n/a"}`);
  lines.push(`- operator proof bytes total: ${aggregate.operatorMetrics.operatorFileBytesTotal} (${formatBytes(aggregate.operatorMetrics.operatorFileBytesTotal)})`);
  lines.push("");
  lines.push("## Host evidence health");
  lines.push(`- security critical total: ${aggregate.hostMetrics.securityCriticalTotal}`);
  lines.push(`- security warn total: ${aggregate.hostMetrics.securityWarnTotal}`);
  lines.push(`- gateway reachable bundles: ${aggregate.hostMetrics.gatewayReachableCount}`);
  lines.push(`- worker healthy bundles: ${aggregate.hostMetrics.workerHealthyCount}`);
  lines.push(`- memory files total: ${aggregate.hostMetrics.memoryFilesTotal}`);
  lines.push(`- session count total: ${aggregate.hostMetrics.sessionCountTotal}`);
  lines.push("");
  lines.push("## Cost proxy");
  lines.push(`- proof minutes: ${aggregate.costProxy.proofMinutes}`);
  lines.push(`- artifact bytes: ${aggregate.costProxy.artifactBytes} (${aggregate.costProxy.artifactMB} MiB)`);
  lines.push(`- bundle count: ${aggregate.costProxy.bundleCount}`);
  lines.push(`- validation count: ${aggregate.costProxy.validationCount}`);
  lines.push("");
  lines.push("## Latest bundle surface");
  for (const [key, bundle] of Object.entries(aggregate.latestBundles)) {
    if (!bundle) {
      continue;
    }
    lines.push(`- ${key}: ${bundle.relativePath} (${formatAge(bundle.ageDays)}, ${bundle.validationOk === true ? "ok" : bundle.validationOk === false ? "fail" : "unknown"})`);
  }
  return `${lines.join("\n")}\n`;
}

function writeHealthOutputs(outputDir, snapshot, statusProbe) {
  ensureDir(outputDir);
  saveJson(path.join(outputDir, "status.json"), statusProbe);
  saveJson(path.join(outputDir, "snapshot.json"), snapshot);
  saveText(path.join(outputDir, "summary.md"), formatHealthMarkdown(snapshot));
}

function writeNightlyOutputs(outputDir, aggregate, bundles) {
  ensureDir(outputDir);
  saveJson(path.join(outputDir, "aggregate.json"), aggregate);
  saveJson(path.join(outputDir, "bundle-index.json"), bundles);
  saveText(path.join(outputDir, "summary.md"), formatNightlyMarkdown(aggregate));
}

function buildWorkspaceOutputDir(subdir) {
  return path.join(DEFAULT_OUTPUT_ROOT, subdir);
}

function readAndResolveConfig(configPath, context) {
  const config = loadConfig(configPath, context);
  return {
    ...config,
    openclawHome: resolveToken(config.openclawHome, context),
    scanRoots: config.scanRoots.map((spec) => resolvePathSpec(spec, context)).filter(Boolean),
    excludeRoots: config.excludeRoots.map((spec) => resolvePathSpec(spec, context)).filter(Boolean),
    statusCommand: config.statusCommand.map((part) => resolveToken(String(part), context)),
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const context = {
    repoRoot,
    workspaceRoot,
    openclawHome: options.openclawHome,
    outputRoot: DEFAULT_OUTPUT_ROOT,
  };
  const config = readAndResolveConfig(options.configPath, context);
  const scanStart = Date.now();
  const bundles = summarizeScan(
    collectBundleCandidates(config.scanRoots, config.excludeRoots),
    new Date(),
    workspaceRoot,
  );
  const scanDurationMs = Date.now() - scanStart;

  if (options.command === "health") {
    const outputDir = options.outputDir ?? buildWorkspaceOutputDir("health-snapshot");
    const statusProbe = runStatusProbe(config, context);
    const snapshot = buildHealthSnapshot({
      config,
      statusProbe,
      bundles,
      now: new Date(),
      scanDurationMs,
    });
    writeHealthOutputs(outputDir, snapshot, statusProbe);
    process.stdout.write(
      [
        `health snapshot written to ${outputDir}`,
        `status probe ms: ${snapshot.performance.statusProbeMs}`,
        `scan ms: ${snapshot.performance.scanMs}`,
        `bundle count: ${snapshot.proofInventory.bundleCount}`,
        `latest operator proof: ${snapshot.latestBundles.find((bundle) => bundle.kind === "operator-proof")?.relativePath ?? "none"}`,
      ].join("\n") + "\n",
    );
    return;
  }

  const outputDir = options.outputDir ?? buildWorkspaceOutputDir("nightly-aggregate");
  const aggregate = buildNightlyAggregate({
    config,
    bundles,
    now: new Date(),
    scanDurationMs,
  });
  writeNightlyOutputs(outputDir, aggregate, bundles);
  process.stdout.write(
    [
      `nightly aggregate written to ${outputDir}`,
      `scan ms: ${aggregate.scanMs}`,
      `bundle count: ${aggregate.bundles.length}`,
      `replay proofs: ${aggregate.bundleTypeCounts.recordedSessionReplay}`,
      `operator proofs: ${aggregate.bundleTypeCounts.operatorProof}`,
    ].join("\n") + "\n",
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  }
}

export {
  buildHealthSnapshot,
  buildNightlyAggregate,
  bundleTimestamp,
  classifyBundleRoot,
  collectBundleCandidates,
  formatHealthMarkdown,
  formatNightlyMarkdown,
  loadConfig,
  parseArgs,
  parseTimestampFromPath,
  readAndResolveConfig,
  summarizeBundle,
  summarizeHostBundle,
  summarizeOperatorBundle,
  summarizeReplayBundle,
  summarizeScan,
};
