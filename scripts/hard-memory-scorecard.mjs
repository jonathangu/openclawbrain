import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

export const HARD_MEMORY_SCORECARD_CONTRACT = "openclawbrain_hard_memory_scorecard.v1";
export const HARD_MEMORY_SCORECARD_JSON_FILE = "hard-memory-scorecard.json";
export const HARD_MEMORY_SCORECARD_MARKDOWN_FILE = "hard-memory-scorecard.md";
export const HARD_MEMORY_LABEL_SCHEMA = "learned-route-labels.v1";

const HARD_MEMORY_LANE = "hard_memory";
const GATE_EPSILON = 1e-9;
const ORACLE_RELATION_BY_MODE = {
  learned_route: "better",
  tie: "tied",
  graph_prior_only: "worse",
};

const LEAD_BLOCK_FIELDS = [
  { key: "focus_lane", label: "Focus lane" },
  { key: "focus_cohort_id", label: "Focus cohort" },
  { key: "trace_count", label: "Traces" },
  { key: "lr_vs_gpo_better", label: "Better" },
  { key: "lr_vs_gpo_tied", label: "Tied" },
  { key: "lr_vs_gpo_worse", label: "Worse" },
  { key: "tie_or_better_rate", label: "Tie-or-better rate" },
  { key: "regression_rate", label: "Regression rate" },
  { key: "required_context_recall_delta", label: "Required-context recall delta" },
  { key: "net_utility_delta", label: "Net utility delta" },
  { key: "cost_per_incremental_win", label: "Cost per incremental win" },
  { key: "latency_per_incremental_win", label: "Latency per incremental win" },
  { key: "gate_status", label: "Gate" },
];

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeNumber(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function round(value, digits = 6) {
  if (!Number.isFinite(value)) {
    return null;
  }
  const factor = 10 ** digits;
  return Math.round(Number(value) * factor) / factor;
}

function mean(values) {
  const numericValues = (Array.isArray(values) ? values : []).map((value) => Number(value)).filter(Number.isFinite);
  if (numericValues.length === 0) {
    return null;
  }
  const total = numericValues.reduce((sum, value) => sum + value, 0);
  return total / numericValues.length;
}

function formatValue(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  if (typeof value === "number") {
    const rounded = round(value, 6);
    return rounded === null ? "n/a" : String(rounded);
  }
  if (typeof value === "string") {
    return value.length > 0 ? `\`${value}\`` : "n/a";
  }
  return `\`${String(value)}\``;
}

function readStructuredFile(filePath) {
  const text = readFileSync(filePath, "utf8");
  if (filePath.endsWith(".jsonl")) {
    return text
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line, index) => {
        try {
          return JSON.parse(line);
        }
        catch (error) {
          throw new Error(`Failed to parse JSONL line ${index + 1} in ${filePath}: ${error.message}`);
        }
      });
  }
  return JSON.parse(text);
}

function collectStructuredFiles(rootDir) {
  const collected = [];
  const visit = (currentDir) => {
    const entries = readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        visit(entryPath);
        continue;
      }
      if (entry.isFile() && (entry.name.endsWith(".json") || entry.name.endsWith(".jsonl"))) {
        collected.push(entryPath);
      }
    }
  };
  visit(rootDir);
  return collected.sort();
}

function normalizeStructuredRows(payload) {
  if (Array.isArray(payload)) {
    return payload.filter((row) => row && typeof row === "object");
  }
  if (!payload || typeof payload !== "object") {
    return [];
  }
  if (Array.isArray(payload.rows)) {
    return normalizeStructuredRows(payload.rows);
  }
  if (Array.isArray(payload.records)) {
    return normalizeStructuredRows(payload.records);
  }
  if (Array.isArray(payload.traces)) {
    return normalizeStructuredRows(payload.traces);
  }
  if (Array.isArray(payload.metrics)) {
    return normalizeStructuredRows(payload.metrics);
  }
  if (Array.isArray(payload.entries)) {
    return normalizeStructuredRows(payload.entries);
  }
  return [payload];
}

function extractTraceIdFromSeedEntry(entry) {
  if (typeof entry === "string") {
    return normalizeText(entry);
  }
  if (!entry || typeof entry !== "object") {
    return null;
  }
  return normalizeText(entry.traceId)
    ?? normalizeText(entry.trace_id)
    ?? normalizeText(entry.id);
}

function collectSeedTraceIds(seedManifest, labelRecords) {
  const fromManifest = [];
  const manifestTraceEntries = Array.isArray(seedManifest?.traces)
    ? seedManifest.traces
    : Array.isArray(seedManifest?.entries)
      ? seedManifest.entries
      : Array.isArray(seedManifest?.traceIds)
        ? seedManifest.traceIds
        : [];
  for (const entry of manifestTraceEntries) {
    const traceId = extractTraceIdFromSeedEntry(entry);
    if (traceId) {
      fromManifest.push(traceId);
    }
  }
  if (fromManifest.length > 0) {
    return [...new Set(fromManifest)];
  }
  return [...new Set(normalizeStructuredRows(labelRecords).map((record) => normalizeText(record?.trace_id ?? record?.traceId)).filter(Boolean))];
}

function buildLabelIndex(labelRecords) {
  const index = new Map();
  for (const record of normalizeStructuredRows(labelRecords)) {
    const traceId = normalizeText(record?.trace_id ?? record?.traceId);
    if (!traceId || index.has(traceId)) {
      continue;
    }
    index.set(traceId, record);
  }
  return index;
}

function buildReplayTraceIndex(replayLaneSummaryTables) {
  const index = new Map();
  for (const row of Array.isArray(replayLaneSummaryTables?.traces) ? replayLaneSummaryTables.traces : []) {
    const traceId = normalizeText(row?.traceId ?? row?.trace_id);
    if (!traceId || index.has(traceId)) {
      continue;
    }
    index.set(traceId, row);
  }
  return index;
}

function firstNumber(...values) {
  for (const value of values) {
    const normalized = normalizeNumber(value);
    if (normalized !== null) {
      return normalized;
    }
  }
  return null;
}

function extractUtilityMetricRow(row) {
  const metrics = row?.metrics ?? row?.supplementalMetrics ?? row?.route_objective_hint ?? row?.routeObjectiveHint ?? null;
  return {
    netUtilityDelta: firstNumber(
      row?.net_utility_delta,
      row?.netUtilityDelta,
      row?.utility_delta_vs_graph_prior_only,
      row?.utilityDeltaVsBaseline,
      row?.utilityDelta,
      metrics?.net_utility_delta,
      metrics?.netUtilityDelta,
      metrics?.utility_delta_vs_graph_prior_only,
      metrics?.utilityDeltaVsBaseline,
      metrics?.utilityDelta,
    ),
    costDelta: firstNumber(
      row?.cost_delta,
      row?.costDelta,
      row?.costDeltaVsBaseline,
      row?.costUsdDelta,
      metrics?.cost_delta,
      metrics?.costDelta,
      metrics?.costDeltaVsBaseline,
      metrics?.costUsdDelta,
    ),
    latencyDelta: firstNumber(
      row?.latency_delta,
      row?.latencyDelta,
      row?.latencyDeltaVsBaseline,
      row?.latencyMsDelta,
      metrics?.latency_delta,
      metrics?.latencyDelta,
      metrics?.latencyDeltaVsBaseline,
      metrics?.latencyMsDelta,
    ),
  };
}

function buildSupplementalMetricsIndex(supplementalMetrics) {
  const index = new Map();
  for (const row of normalizeStructuredRows(supplementalMetrics)) {
    const traceId = normalizeText(row?.traceId ?? row?.trace_id);
    if (!traceId || index.has(traceId)) {
      continue;
    }
    index.set(traceId, extractUtilityMetricRow(row));
  }
  return index;
}

export function isStrictHardMemoryLabelRecord(labelRecord) {
  const labels = labelRecord?.labels ?? {};
  return normalizeText(labelRecord?.schema_version) === HARD_MEMORY_LABEL_SCHEMA
    && labels.human_semantic_task === "yes"
    && labels.wrapper_noise === "no"
    && labels.continuation_only === "no"
    && labels.memory_needed === "yes"
    && typeof labels.oracle_best_mode === "string"
    && labels.oracle_best_mode !== "unclear";
}

function relationFromOracleBestMode(oracleBestMode) {
  if (typeof oracleBestMode !== "string") {
    return null;
  }
  return ORACLE_RELATION_BY_MODE[oracleBestMode] ?? null;
}

function buildGateStatus({ netUtilityDelta, betterCount, worseCount, blockers }) {
  if (blockers.length > 0 || netUtilityDelta === null) {
    return null;
  }
  if (netUtilityDelta < -GATE_EPSILON || worseCount > betterCount) {
    return "fail";
  }
  if (Math.abs(netUtilityDelta) <= GATE_EPSILON || betterCount === worseCount) {
    return "watch";
  }
  return "pass";
}

function buildTraceRows({ seedTraceIds, labelIndex, replayTraceIndex, supplementalMetricsIndex }) {
  return seedTraceIds.map((traceId) => {
    const labelRecord = labelIndex.get(traceId) ?? null;
    const labels = labelRecord?.labels ?? null;
    const strictEligible = isStrictHardMemoryLabelRecord(labelRecord);
    const oracleBestMode = strictEligible ? labels?.oracle_best_mode ?? null : normalizeText(labels?.oracle_best_mode);
    const outcomeVsBaseline = strictEligible ? relationFromOracleBestMode(oracleBestMode) : null;
    const replayRow = replayTraceIndex.get(traceId) ?? null;
    const supplementalMetrics = supplementalMetricsIndex.get(traceId) ?? extractUtilityMetricRow(labelRecord);
    const notes = [];
    if (!labelRecord) {
      notes.push("missing_label_record");
    } else if (!strictEligible) {
      notes.push("not_strict_hard_memory_eligible");
    }
    if (strictEligible && supplementalMetrics?.netUtilityDelta === null) {
      notes.push("missing_net_utility_delta");
    }
    return {
      trace_id: traceId,
      label_schema_version: normalizeText(labelRecord?.schema_version),
      strict_hard_memory_eligible: strictEligible,
      oracle_best_mode: oracleBestMode,
      outcome_vs_graph_prior_only: outcomeVsBaseline,
      replay_relation_vs_graph_prior_only: normalizeText(replayRow?.candidateRelationVsBaseline),
      net_utility_delta: supplementalMetrics?.netUtilityDelta ?? null,
      notes,
    };
  });
}

export function buildHardMemoryScorecard({
  focusCohortId,
  seedManifest = null,
  labelRecords = [],
  replayLaneSummaryTables = null,
  supplementalMetrics = [],
} = {}) {
  const seedTraceIds = collectSeedTraceIds(seedManifest, labelRecords);
  const labelIndex = buildLabelIndex(labelRecords);
  const replayTraceIndex = buildReplayTraceIndex(replayLaneSummaryTables);
  const supplementalMetricsIndex = buildSupplementalMetricsIndex(supplementalMetrics);
  const traceRows = buildTraceRows({ seedTraceIds, labelIndex, replayTraceIndex, supplementalMetricsIndex });
  const evaluatedTraceRows = traceRows.filter((row) => row.strict_hard_memory_eligible && typeof row.outcome_vs_graph_prior_only === "string");
  const utilityCoveredRows = evaluatedTraceRows.filter((row) => Number.isFinite(row.net_utility_delta));

  const betterCount = evaluatedTraceRows.filter((row) => row.outcome_vs_graph_prior_only === "better").length;
  const tiedCount = evaluatedTraceRows.filter((row) => row.outcome_vs_graph_prior_only === "tied").length;
  const worseCount = evaluatedTraceRows.filter((row) => row.outcome_vs_graph_prior_only === "worse").length;
  const traceCount = evaluatedTraceRows.length;
  const tieOrBetterRate = traceCount > 0 ? round((betterCount + tiedCount) / traceCount, 6) : null;
  const regressionRate = traceCount > 0 ? round(worseCount / traceCount, 6) : null;
  const requiredContextRecallDelta = normalizeNumber(replayLaneSummaryTables?.scorecard?.requiredContextRecall?.delta);
  const netUtilityDelta = traceCount > 0 && utilityCoveredRows.length === traceCount
    ? round(mean(utilityCoveredRows.map((row) => row.net_utility_delta)), 6)
    : null;

  const blockers = [];
  if (seedTraceIds.length === 0) {
    blockers.push("seed_set_empty");
  }
  if (labelIndex.size === 0) {
    blockers.push("no_label_records_loaded");
  }
  if (traceCount === 0) {
    blockers.push("no_strict_hard_memory_labels_scored");
  }
  if (traceCount > 0 && utilityCoveredRows.length !== traceCount) {
    blockers.push(`net_utility_delta_unavailable:${utilityCoveredRows.length}/${traceCount}_traces_have_numeric_utility`);
  }

  const gateStatus = buildGateStatus({
    netUtilityDelta,
    betterCount,
    worseCount,
    blockers,
  });

  const notes = [
    "better/tied/worse is derived from oracle_best_mode on strict-eligible learned-route-labels.v1 records",
    "strict eligibility requires human_semantic_task=yes, wrapper_noise=no, continuation_only=no, memory_needed=yes, and oracle_best_mode != unclear",
  ];
  if (netUtilityDelta === null) {
    notes.push("net utility delta stays unavailable until every scored hard-memory trace has an explicit numeric utility delta vs graph_prior_only");
  }
  if (requiredContextRecallDelta === null) {
    notes.push("required-context recall delta is unavailable unless a matching replay-lane summary table is supplied for the same focus cohort");
  }

  const resolvedFocusCohortId = normalizeText(focusCohortId)
    ?? normalizeText(seedManifest?.manifestId)
    ?? normalizeText(seedManifest?.setId)
    ?? normalizeText(seedManifest?.focusCohortId)
    ?? "hard-memory-unspecified";

  return {
    contract: HARD_MEMORY_SCORECARD_CONTRACT,
    focus_lane: HARD_MEMORY_LANE,
    focus_cohort_id: resolvedFocusCohortId,
    seed_trace_count: seedTraceIds.length,
    label_record_count: labelIndex.size,
    trace_count: traceCount,
    lr_vs_gpo_better: betterCount,
    lr_vs_gpo_tied: tiedCount,
    lr_vs_gpo_worse: worseCount,
    tie_or_better_rate: tieOrBetterRate,
    regression_rate: regressionRate,
    required_context_recall_delta: requiredContextRecallDelta,
    net_utility_delta: netUtilityDelta,
    cost_per_incremental_win: null,
    latency_per_incremental_win: null,
    gate_status: gateStatus,
    blockers,
    notes,
    coverage: {
      seed_trace_count: seedTraceIds.length,
      labels_loaded: labelIndex.size,
      strict_eligible_trace_count: traceRows.filter((row) => row.strict_hard_memory_eligible).length,
      scored_trace_count: traceCount,
      numeric_utility_trace_count: utilityCoveredRows.length,
      replay_trace_match_count: traceRows.filter((row) => typeof row.replay_relation_vs_graph_prior_only === "string").length,
    },
    traces: traceRows,
  };
}

export function buildHardMemoryScorecardMarkdown(scorecard) {
  const lines = [
    "# Hard-memory scorecard",
    "",
    "## Lead block",
    "| Order | Field | Label | Value |",
    "| ---: | --- | --- | --- |",
  ];

  LEAD_BLOCK_FIELDS.forEach((field, index) => {
    lines.push(`| ${index + 1} | \`${field.key}\` | ${field.label} | ${formatValue(scorecard?.[field.key])} |`);
  });

  lines.push("", "## Coverage", "");
  lines.push(`- seed traces: ${scorecard?.coverage?.seed_trace_count ?? 0}`);
  lines.push(`- labels loaded: ${scorecard?.coverage?.labels_loaded ?? 0}`);
  lines.push(`- strict-eligible traces: ${scorecard?.coverage?.strict_eligible_trace_count ?? 0}`);
  lines.push(`- scored traces: ${scorecard?.coverage?.scored_trace_count ?? 0}`);
  lines.push(`- numeric utility traces: ${scorecard?.coverage?.numeric_utility_trace_count ?? 0}`);
  lines.push(`- replay trace matches: ${scorecard?.coverage?.replay_trace_match_count ?? 0}`);

  lines.push("", "## Notes", "");
  for (const note of Array.isArray(scorecard?.notes) ? scorecard.notes : []) {
    lines.push(`- ${note}`);
  }
  if (Array.isArray(scorecard?.blockers) && scorecard.blockers.length > 0) {
    lines.push("", "## Blockers", "");
    for (const blocker of scorecard.blockers) {
      lines.push(`- ${blocker}`);
    }
  }

  lines.push("", "## Trace rows", "", "| trace_id | strict eligible | oracle_best_mode | outcome vs graph_prior_only | net utility delta | notes |", "| --- | --- | --- | --- | ---: | --- |");
  for (const trace of Array.isArray(scorecard?.traces) ? scorecard.traces : []) {
    lines.push(`| \`${trace.trace_id}\` | ${trace.strict_hard_memory_eligible ? "yes" : "no"} | ${trace.oracle_best_mode ?? "n/a"} | ${trace.outcome_vs_graph_prior_only ?? "n/a"} | ${trace.net_utility_delta ?? "n/a"} | ${(trace.notes ?? []).join(", ") || "-"} |`);
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

export function writeHardMemoryScorecardOutputs(outputDir, scorecard) {
  mkdirSync(outputDir, { recursive: true });
  const jsonPath = path.join(outputDir, HARD_MEMORY_SCORECARD_JSON_FILE);
  const markdownPath = path.join(outputDir, HARD_MEMORY_SCORECARD_MARKDOWN_FILE);
  writeFileSync(jsonPath, `${JSON.stringify(scorecard, null, 2)}\n`, "utf8");
  writeFileSync(markdownPath, buildHardMemoryScorecardMarkdown(scorecard), "utf8");
  return { jsonPath, markdownPath };
}

function loadLabelRecords({ labelFiles, labelsDir }) {
  const files = [];
  if (labelsDir) {
    const resolvedDir = path.resolve(labelsDir);
    if (statSync(resolvedDir).isDirectory()) {
      files.push(...collectStructuredFiles(resolvedDir));
    }
  }
  files.push(...labelFiles.map((filePath) => path.resolve(filePath)));
  return [...new Set(files)].flatMap((filePath) => normalizeStructuredRows(readStructuredFile(filePath)));
}

function parseArgs(argv) {
  const args = {
    seedManifestPath: null,
    replaySummaryTablesPath: null,
    supplementalMetricsPath: null,
    labelsDir: null,
    labelFiles: [],
    focusCohortId: null,
    outDir: null,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    const nextValue = argv[index + 1];
    switch (token) {
      case "--seed-manifest":
        args.seedManifestPath = nextValue;
        index += 1;
        break;
      case "--replay-summary-tables":
        args.replaySummaryTablesPath = nextValue;
        index += 1;
        break;
      case "--supplemental-metrics":
        args.supplementalMetricsPath = nextValue;
        index += 1;
        break;
      case "--labels-dir":
        args.labelsDir = nextValue;
        index += 1;
        break;
      case "--labels-file":
        args.labelFiles.push(nextValue);
        index += 1;
        break;
      case "--focus-cohort-id":
        args.focusCohortId = nextValue;
        index += 1;
        break;
      case "--out-dir":
        args.outDir = nextValue;
        index += 1;
        break;
      case "--help":
      case "-h":
        args.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${token}`);
    }
  }
  return args;
}

function usageText() {
  return [
    "Usage: node scripts/hard-memory-scorecard.mjs [options]",
    "",
    "Options:",
    "  --seed-manifest <path>          JSON manifest with trace ids or trace entries",
    "  --labels-dir <dir>              Directory of learned-route-labels.v1 JSON/JSONL records",
    "  --labels-file <path>            Single learned-route-labels.v1 JSON or JSONL record file (repeatable)",
    "  --replay-summary-tables <path>  Matching replay-lane summary-tables.json for recall delta",
    "  --supplemental-metrics <path>   JSON or JSONL rows with traceId + netUtilityDelta (optional)",
    "  --focus-cohort-id <id>          Override focus cohort id",
    "  --out-dir <dir>                 Write hard-memory-scorecard.json and .md here",
    "  --help                          Show this help",
  ].join("\n");
}

export function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(usageText());
    return;
  }
  if (!args.outDir) {
    throw new Error("--out-dir is required");
  }

  const seedManifest = args.seedManifestPath ? readStructuredFile(path.resolve(args.seedManifestPath)) : null;
  const replayLaneSummaryTables = args.replaySummaryTablesPath ? readStructuredFile(path.resolve(args.replaySummaryTablesPath)) : null;
  const supplementalMetrics = args.supplementalMetricsPath ? readStructuredFile(path.resolve(args.supplementalMetricsPath)) : [];
  const labelRecords = loadLabelRecords({ labelFiles: args.labelFiles, labelsDir: args.labelsDir });
  const scorecard = buildHardMemoryScorecard({
    focusCohortId: args.focusCohortId,
    seedManifest,
    labelRecords,
    replayLaneSummaryTables,
    supplementalMetrics,
  });
  const outputs = writeHardMemoryScorecardOutputs(path.resolve(args.outDir), scorecard);
  console.log(JSON.stringify({ contract: scorecard.contract, focus_cohort_id: scorecard.focus_cohort_id, ...outputs }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
