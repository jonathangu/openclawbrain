#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { captureRuntimeEvent } from "./capture-event.mjs";
import { exportCandidate } from "./export-candidate.mjs";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_EVENT_OUT_DIR = "eval/runtime-events";
const DEFAULT_FIRE_THRESHOLD = 0.75;
const DECISION_SCHEMA_VERSION = "ocb.runtime.decision_input.v1";
const DECISION_ALGORITHM = "ocb.runtime.decide.minimal.v1";
const VALID_ACTIONS = new Set(["fire", "stay_silent"]);
const SECRET_KEY_RE = /(^|[_-])(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|secret|private[_-]?key)$/iu;
const RAW_KEY_RE = /(^|[_-])(raw[_-]?(messages?|transcript|text|content)|unredacted|user[_-]?task[_-]?raw|assistant[_-]?answer[_-]?raw)$/iu;
const SECRET_VALUE_RE = /(-----BEGIN [A-Z ]*PRIVATE KEY-----|(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,}|AKIA[0-9A-Z]{12,}|API_KEY=|PASSWORD=|ACCESS_TOKEN=)/u;

export async function decideRuntimeTurn(options) {
  const inputPath = required(options.input, "input");
  const input = JSON.parse(await readFile(resolvePath(inputPath), "utf8"));
  const event = buildRuntimeDecisionEvent(input, options);
  const eventInputPath = options.eventInputOut || defaultEventInputPath(event.event_id);
  await writeJson(eventInputPath, event);
  const capture = await captureRuntimeEvent({
    event: eventInputPath,
    outDir: options.outDir ?? DEFAULT_EVENT_OUT_DIR,
    capturedAt: event.captured_at,
  });
  const result = {
    ok: true,
    action: event.memory_fired ? "fire" : "stay_silent",
    event_id: event.event_id,
    trace_id: event.trace_id,
    selected_memory_ids_redacted: event.retrieved_memory_ids_redacted,
    decision_reason: event.decision_reason,
    event_input: eventInputPath,
    event_file: capture.event_file,
    manifest: capture.manifest,
    counts_as_product_evidence: false,
    reproducibility: event.reproducibility,
    candidate_export: null,
  };
  if (options.candidateOut) {
    result.candidate_export = await exportCandidate({ event: capture.event_file, out: options.candidateOut });
  }
  return result;
}

export function buildRuntimeDecisionEvent(input, options = {}) {
  const issues = validateDecisionInput(input);
  if (issues.length > 0) {
    const error = new Error(`runtime decision rejected: ${issues.join("; ")}`);
    error.issues = issues;
    throw error;
  }
  const eventId = stringValue(input.event_id || input.turn_id || input.request_id || stableId("event", input));
  const threshold = numberOrDefault(input.fire_threshold ?? options.fireThreshold, DEFAULT_FIRE_THRESHOLD);
  const candidates = normalizeMemoryCandidates(input.memory_candidates_redacted);
  const selected = selectMemories(candidates, threshold);
  const forcedAction = input.force_action ? String(input.force_action).trim() : null;
  if (forcedAction && !VALID_ACTIONS.has(forcedAction)) throw new Error("runtime decision rejected: force_action must be fire or stay_silent");
  const action = forcedAction ?? (selected.length > 0 ? "fire" : "stay_silent");
  const decisionReason = action === "fire"
    ? `Selected ${selected.length} redacted memory candidate(s) at or above threshold ${threshold}.`
    : `Stayed silent because no non-stale redacted memory candidate met threshold ${threshold}.`;
  const inputHash = hashJson(input);
  return {
    event_id: eventId,
    trace_id: stringValue(input.trace_id || `runtime-${safeId(eventId)}`),
    source: input.source,
    provenance_type: input.provenance_type ?? "real",
    title: input.title,
    task_type: input.task_type,
    user_task_redacted: input.user_task_redacted,
    current_context_redacted: input.current_context_redacted ?? "",
    expected_memory_opportunity: Boolean(input.expected_memory_opportunity ?? selected.length > 0),
    memory_fired: action === "fire",
    backend_observed: "minimal-runtime-decision",
    retrieved_memory_ids_redacted: action === "fire" ? selected.map((candidate) => candidate.id) : [],
    decision_reason: decisionReason,
    decision_algorithm: DECISION_ALGORITHM,
    memory_candidates_considered: candidates.length,
    slice: input.slice,
    privacy_scrubbed: input.privacy_scrubbed,
    contains_real_user_data: input.contains_real_user_data,
    collected_at: input.collected_at,
    captured_at: options.capturedAt || input.captured_at || input.collected_at,
    redaction_notes: input.redaction_notes,
    memory_snapshot_id: input.memory_snapshot_id,
    memory_snapshot_created_at: input.memory_snapshot_created_at,
    ocb_config_hash: input.ocb_config_hash,
    model_id: input.model_id,
    prompt_hash: input.prompt_hash,
    code_commit: input.code_commit,
    allowed_evidence: input.allowed_evidence ?? ["redacted user task", "redacted current context", "redacted memory candidate IDs", "deterministic decision metadata"],
    prohibited_evidence: input.prohibited_evidence ?? ["raw private messages", "secrets", "unredacted user identifiers", "mutating external service calls"],
    reproducibility: {
      deterministic: true,
      replay_safe: true,
      decision_schema_version: DECISION_SCHEMA_VERSION,
      decision_algorithm: DECISION_ALGORITHM,
      fire_threshold: threshold,
      input_hash: `sha256:${inputHash}`,
      mutating_external_services: false,
      ...input.reproducibility,
      deterministic: true,
    },
    tool_fixture_mode: input.tool_fixture_mode,
  };
}

export function validateDecisionInput(input) {
  const issues = [];
  if (!isRecord(input)) return ["decision input must be a JSON object"];
  scanForUnsafeContent(input, [], issues);
  for (const field of ["source", "title", "task_type", "user_task_redacted", "slice", "collected_at", "redaction_notes", "memory_snapshot_id", "memory_snapshot_created_at", "ocb_config_hash", "model_id", "prompt_hash", "code_commit"])
    requireNonEmptyString(input, field, issues);
  for (const field of ["privacy_scrubbed", "contains_real_user_data"])
    requireBoolean(input, field, issues);
  if (input.privacy_scrubbed !== true) issues.push("runtime decision requires privacy_scrubbed=true");
  if (input.contains_real_user_data !== false) issues.push("runtime decision requires contains_real_user_data=false after redaction");
  if (String(input.user_task_redacted ?? "").trim().length < 8) issues.push("user_task_redacted is too short to identify the task");
  if (input.memory_candidates_redacted !== undefined && !Array.isArray(input.memory_candidates_redacted)) issues.push("memory_candidates_redacted must be an array when present");
  return issues;
}

function normalizeMemoryCandidates(candidates = []) {
  return candidates.filter(isRecord).map((candidate, index) => ({
    id: stringValue(candidate.id || candidate.memory_id_redacted || `memory-${index + 1}`),
    relevance_score: numberOrDefault(candidate.relevance_score, 0),
    stale: candidate.stale === true,
    conflict: candidate.conflict === true,
  }));
}

function selectMemories(candidates, threshold) {
  return candidates
    .filter((candidate) => candidate.id && candidate.relevance_score >= threshold && candidate.stale !== true && candidate.conflict !== true)
    .sort((a, b) => b.relevance_score - a.relevance_score || a.id.localeCompare(b.id));
}

function scanForUnsafeContent(value, path, issues) {
  if (Array.isArray(value)) {
    value.forEach((item, index) => scanForUnsafeContent(item, [...path, String(index)], issues));
    return;
  }
  if (isRecord(value)) {
    for (const [key, child] of Object.entries(value)) {
      const currentPath = [...path, key];
      if (SECRET_KEY_RE.test(key)) issues.push(`${currentPath.join(".")}: secret-like field is not allowed in runtime decision input`);
      if (RAW_KEY_RE.test(key)) issues.push(`${currentPath.join(".")}: raw/unredacted field is not allowed in runtime decision input`);
      scanForUnsafeContent(child, currentPath, issues);
    }
    return;
  }
  if (typeof value === "string" && SECRET_VALUE_RE.test(value)) issues.push(`${path.join(".") || "<root>"}: secret-like value is not allowed in runtime decision input`);
}

function parseArgs(argv) {
  const args = { outDir: DEFAULT_EVENT_OUT_DIR };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--input") args.input = argv[++i];
    else if (arg === "--out-dir") args.outDir = argv[++i];
    else if (arg === "--event-input-out") args.eventInputOut = argv[++i];
    else if (arg === "--candidate-out") args.candidateOut = argv[++i];
    else if (arg === "--captured-at") args.capturedAt = argv[++i];
    else if (arg === "--fire-threshold") args.fireThreshold = Number(argv[++i]);
    else if (arg === "--help" || arg === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function help() {
  return `Usage: node scripts/runtime/decide.mjs --input redacted-decision-input.json [--out-dir eval/runtime-events] [--candidate-out eval/trace-candidates/candidate.json]\n\nDeterministically chooses fire or stay_silent for one redacted agent turn, captures the runtime event through ocb:runtime:capture-event, and optionally exports an admission candidate. This command is local, read-only except for evidence artifacts, and never calls mutating external services.`;
}

async function runCli() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(help()); return; }
  console.log(JSON.stringify(await decideRuntimeTurn(args), null, 2));
}

function defaultEventInputPath(eventId) { return `eval/runtime-events/${safeId(eventId)}.input.json`; }
function hashJson(value) { return createHash("sha256").update(JSON.stringify(value)).digest("hex"); }
function stableId(prefix, value) { return `${prefix}-${hashJson(value).slice(0, 16)}`; }
function numberOrDefault(value, fallback) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function stringValue(value) { return typeof value === "string" && value.trim() ? value.trim() : String(value ?? "").trim(); }
function safeId(value) { return String(value).trim().replace(/[^a-zA-Z0-9._-]/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "event"; }
function isRecord(value) { return typeof value === "object" && value !== null && !Array.isArray(value); }
function requireNonEmptyString(value, field, issues) { if (typeof value[field] !== "string" || value[field].trim() === "") issues.push(`${field} must be a non-empty string`); }
function requireBoolean(value, field, issues) { if (typeof value[field] !== "boolean") issues.push(`${field} must be a boolean`); }
function required(value, name) { if (!value) throw new Error(`--${name} is required`); return value; }
function resolvePath(path) { return resolve(PROJECT_ROOT, path); }
async function writeJson(path, value) { await mkdir(dirname(resolvePath(path)), { recursive: true }); await writeFile(resolvePath(path), `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli().catch((error) => {
    console.error(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error), issues: error?.issues ?? undefined }, null, 2));
    process.exitCode = 1;
  });
}
