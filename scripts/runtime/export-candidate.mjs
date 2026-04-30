#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { admitTraceCandidate } from "../traces/admit.mjs";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const VALID_SOURCES = new Set(["telegram", "github", "session", "synthetic", "repo-derived", "adversarial"]);
const VALID_SLICES = new Set(["direct-answer", "continuation", "correction-follow-up", "retrieval-heavy", "tool-heavy", "stale-memory-conflict"]);
const VALID_PROVENANCE = new Set(["real", "synthetic", "repo-derived", "adversarial"]);
const SECRET_KEY_RE = /(^|[_-])(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|secret|private[_-]?key)$/iu;
const RAW_KEY_RE = /(^|[_-])(raw[_-]?(messages?|transcript|text|content)|unredacted|user[_-]?task[_-]?raw|assistant[_-]?answer[_-]?raw)$/iu;
const SECRET_VALUE_RE = /(-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,}|AKIA[0-9A-Z]{12,}|API_KEY=|PASSWORD=|ACCESS_TOKEN=)/u;

export async function exportCandidate(options) {
  const eventPath = required(options.event, "event");
  const outPath = required(options.out, "out");
  const event = JSON.parse(await readFile(resolvePath(eventPath), "utf8"));
  const issues = validateRuntimeEvent(event);
  if (issues.length > 0) {
    const error = new Error(`runtime event rejected: ${issues.join("; ")}`);
    error.issues = issues;
    throw error;
  }
  const candidate = toCandidate(event);
  await writeJson(outPath, candidate);
  const result = {
    ok: true,
    candidate: outPath,
    trace_id: candidate.trace_id,
    counts_as_product_evidence: false,
    admission: null,
  };
  if (options.admit === true) {
    result.admission = await admitTraceCandidate({
      candidate: outPath,
      admit: true,
      manifest: options.manifest,
      outRoot: options.outRoot,
    });
  }
  return result;
}

export function validateRuntimeEvent(event) {
  const issues = [];
  if (!isRecord(event)) return ["event must be a JSON object"];
  scanForUnsafeContent(event, [], issues);
  for (const field of ["event_id", "source", "title", "task_type", "user_task_redacted", "slice", "collected_at", "redaction_notes", "memory_snapshot_id", "memory_snapshot_created_at", "ocb_config_hash", "model_id", "prompt_hash", "code_commit"])
    requireNonEmptyString(event, field, issues);
  for (const field of ["expected_memory_opportunity", "privacy_scrubbed", "contains_real_user_data"])
    requireBoolean(event, field, issues);
  if (!VALID_SOURCES.has(event.source)) issues.push(`source must be one of ${[...VALID_SOURCES].join(", ")}`);
  if (!VALID_SLICES.has(event.slice)) issues.push(`slice must be one of ${[...VALID_SLICES].join(", ")}`);
  const provenanceType = event.provenance_type ?? "real";
  if (!VALID_PROVENANCE.has(provenanceType)) issues.push(`provenance_type must be one of ${[...VALID_PROVENANCE].join(", ")}`);
  if (!isRecord(event.reproducibility)) issues.push("reproducibility must be an object");
  else if (event.reproducibility.deterministic !== true) issues.push("reproducibility.deterministic must be true");
  if (event.slice === "tool-heavy" && event.tool_fixture_mode !== "read_only_fixture_safe") issues.push("tool-heavy runtime events require tool_fixture_mode=read_only_fixture_safe");
  if (event.privacy_scrubbed !== true) issues.push("runtime candidate export requires privacy_scrubbed=true");
  if (event.contains_real_user_data !== false) issues.push("runtime candidate export requires contains_real_user_data=false after redaction");
  if (String(event.user_task_redacted ?? "").trim().length < 8) issues.push("user_task_redacted is too short to identify the task");
  return issues;
}

export function toCandidate(event) {
  const traceId = event.trace_id || `runtime-${safeId(event.event_id)}`;
  return {
    trace_id: traceId,
    title: event.title,
    source: event.source,
    provenance_type: event.provenance_type ?? "real",
    slice: event.slice,
    task_type: event.task_type,
    user_task_redacted: event.user_task_redacted,
    current_context_redacted: event.current_context_redacted ?? "",
    expected_memory_opportunity: event.expected_memory_opportunity,
    privacy_scrubbed: event.privacy_scrubbed,
    contains_real_user_data: event.contains_real_user_data,
    collected_at: event.collected_at,
    redaction_notes: event.redaction_notes,
    memory_snapshot_id: event.memory_snapshot_id,
    memory_snapshot_created_at: event.memory_snapshot_created_at,
    ocb_config_hash: event.ocb_config_hash,
    model_id: event.model_id,
    prompt_hash: event.prompt_hash,
    code_commit: event.code_commit,
    allowed_evidence: event.allowed_evidence ?? ["redacted runtime task", "redacted runtime context", "memory decision metadata"],
    prohibited_evidence: event.prohibited_evidence ?? ["raw private messages", "secrets", "unredacted user identifiers"],
    reproducibility: event.reproducibility,
    tool_fixture_mode: event.tool_fixture_mode,
    runtime_observation: {
      event_id: event.event_id,
      memory_fired: event.memory_fired ?? null,
      backend_observed: event.backend_observed ?? null,
      retrieved_memory_ids_redacted: event.retrieved_memory_ids_redacted ?? [],
      export_schema: "ocb.runtime.trace_candidate.v1",
    },
  };
}

function scanForUnsafeContent(value, path, issues) {
  if (Array.isArray(value)) {
    value.forEach((item, index) => scanForUnsafeContent(item, [...path, String(index)], issues));
    return;
  }
  if (isRecord(value)) {
    for (const [key, child] of Object.entries(value)) {
      const currentPath = [...path, key];
      if (SECRET_KEY_RE.test(key)) issues.push(`${currentPath.join(".")}: secret-like field is not allowed in trace export input`);
      if (RAW_KEY_RE.test(key)) issues.push(`${currentPath.join(".")}: raw/unredacted field is not allowed in trace export input`);
      scanForUnsafeContent(child, currentPath, issues);
    }
    return;
  }
  if (typeof value === "string" && SECRET_VALUE_RE.test(value)) issues.push(`${path.join(".") || "<root>"}: secret-like value is not allowed in trace export input`);
}

function parseArgs(argv) {
  const args = { admit: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--event") args.event = argv[++i];
    else if (arg === "--out") args.out = argv[++i];
    else if (arg === "--admit") args.admit = true;
    else if (arg === "--manifest") args.manifest = argv[++i];
    else if (arg === "--out-root") args.outRoot = argv[++i];
    else if (arg === "--help" || arg === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function help() {
  return `Usage: node scripts/runtime/export-candidate.mjs --event redacted-runtime-event.json --out trace-candidate.json [--admit]\n\nExports a privacy-scrubbed runtime observation into the ocb:traces:admit candidate format. Raw messages, secret-like fields, unredacted text, and non-deterministic events fail closed.`;
}

async function runCli() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(help()); return; }
  const result = await exportCandidate(args);
  console.log(JSON.stringify(result, null, 2));
}

function isRecord(value) { return typeof value === "object" && value !== null && !Array.isArray(value); }
function requireNonEmptyString(value, field, issues) { if (typeof value[field] !== "string" || value[field].trim() === "") issues.push(`${field} must be a non-empty string`); }
function requireBoolean(value, field, issues) { if (typeof value[field] !== "boolean") issues.push(`${field} must be a boolean`); }
function required(value, name) { if (!value) throw new Error(`--${name} is required`); return value; }
function safeId(value) { return String(value).trim().replace(/[^a-zA-Z0-9._-]/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "event"; }
function resolvePath(path) { return resolve(PROJECT_ROOT, path); }
async function writeJson(path, value) { await mkdir(dirname(resolvePath(path)), { recursive: true }); await writeFile(resolvePath(path), `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli().catch((error) => {
    console.error(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error), issues: error?.issues ?? undefined }, null, 2));
    process.exitCode = 1;
  });
}
