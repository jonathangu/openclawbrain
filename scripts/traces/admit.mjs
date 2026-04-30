#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_OUT_ROOT = "eval/traces/production";
const DEFAULT_MANIFEST = "eval/traces/production.manifest.json";
const MANIFEST_VERSION = "ocb.traces.manifest.v1";
const VALID_SLICES = new Set(["direct-answer", "continuation", "correction-follow-up", "retrieval-heavy", "tool-heavy", "stale-memory-conflict"]);
const PRIMARY_SLICES = new Set(["correction-follow-up", "continuation", "stale-memory-conflict"]);
const VALID_PROVENANCE = new Set(["real", "synthetic", "repo-derived", "adversarial"]);
const REQUIRED_REAL_TRACE_COUNT = 40;
const SLICE_MINIMUMS = {
  "direct-answer": 6,
  "continuation": 6,
  "correction-follow-up": 8,
  "retrieval-heavy": 6,
  "tool-heavy": 6,
  "stale-memory-conflict": 8,
};

export async function admitTraceCandidate(options) {
  const candidatePath = required(options.candidate, "candidate");
  const outRoot = options.outRoot ?? DEFAULT_OUT_ROOT;
  const manifestPath = options.manifest ?? DEFAULT_MANIFEST;
  const candidate = JSON.parse(await readFile(resolvePath(candidatePath), "utf8"));
  const issues = validateCandidate(candidate, { admit: options.admit === true });
  if (issues.length > 0) {
    const error = new Error(`trace candidate rejected: ${issues.join("; ")}`);
    error.issues = issues;
    throw error;
  }

  const traceId = candidate.trace_id.trim();
  const traceDir = join(outRoot, safePathSegment(traceId));
  const admitted = options.admit === true;
  const countsAsProductEvidence = admitted;
  const traceRecord = {
    id: traceId,
    title: candidate.title.trim(),
    path: traceDir,
    input_file: "input.json",
    provenance_file: "provenance.json",
    mode: "production",
    admitted,
    provenance_type: candidate.provenance_type,
    counts_as_product_evidence: countsAsProductEvidence,
    privacy_scrubbed: candidate.privacy_scrubbed,
    slice: candidate.slice,
    priority_class: priorityClass(candidate.slice),
  };

  const input = {
    trace_id: traceId,
    slice: candidate.slice,
    source: candidate.source,
    task_type: candidate.task_type,
    user_task_redacted: candidate.user_task_redacted,
    expected_memory_opportunity: candidate.expected_memory_opportunity,
    current_context_redacted: candidate.current_context_redacted ?? "",
    allowed_evidence: candidate.allowed_evidence ?? [],
    prohibited_evidence: candidate.prohibited_evidence ?? ["raw private content", "unredacted user identifiers", "secrets"],
    admission_note: admitted ? "Admitted real privacy-scrubbed production trace." : "Recorded but not admitted as product evidence.",
  };

  const provenance = {
    trace_id: traceId,
    provenance_type: candidate.provenance_type,
    mode: "production",
    admitted,
    counts_as_product_evidence: countsAsProductEvidence,
    privacy_scrubbed: candidate.privacy_scrubbed,
    contains_real_user_data: candidate.contains_real_user_data,
    slice: candidate.slice,
    source: candidate.source,
    collected_at: candidate.collected_at,
    redaction_notes: candidate.redaction_notes,
    memory_snapshot_id: candidate.memory_snapshot_id,
    memory_snapshot_created_at: candidate.memory_snapshot_created_at,
    ocb_config_hash: candidate.ocb_config_hash,
    model_id: candidate.model_id,
    prompt_hash: candidate.prompt_hash,
    code_commit: candidate.code_commit,
    reproducibility: candidate.reproducibility,
    tool_fixture_mode: candidate.tool_fixture_mode ?? null,
    non_fabrication_rule: "Admission CLI only records caller-supplied redacted trace metadata; it does not manufacture product evidence.",
  };

  await mkdir(resolvePath(traceDir), { recursive: true });
  await writeJson(join(traceDir, "input.json"), input);
  await writeJson(join(traceDir, "provenance.json"), provenance);
  const manifest = await upsertManifest(manifestPath, traceRecord);
  return {
    ok: true,
    trace_id: traceId,
    admitted,
    counts_as_product_evidence: countsAsProductEvidence,
    trace_dir: traceDir,
    manifest: manifestPath,
    admitted_real_product_trace_count: manifest.traces.filter((trace) => trace.admitted && trace.provenance_type === "real" && trace.counts_as_product_evidence).length,
    evidence_e2e_complete: false,
  };
}

function validateCandidate(candidate, options) {
  const issues = [];
  if (!isRecord(candidate)) return ["candidate must be a JSON object"];
  for (const field of ["trace_id", "title", "source", "provenance_type", "slice", "task_type", "user_task_redacted", "collected_at", "redaction_notes", "memory_snapshot_id", "memory_snapshot_created_at", "ocb_config_hash", "model_id", "prompt_hash", "code_commit"])
    requireNonEmptyString(candidate, field, issues);
  for (const field of ["privacy_scrubbed", "contains_real_user_data", "expected_memory_opportunity"])
    requireBoolean(candidate, field, issues);
  if (!VALID_SLICES.has(candidate.slice)) issues.push(`slice must be one of ${[...VALID_SLICES].join(", ")}`);
  if (!VALID_PROVENANCE.has(candidate.provenance_type)) issues.push(`provenance_type must be one of ${[...VALID_PROVENANCE].join(", ")}`);
  if (!isRecord(candidate.reproducibility)) issues.push("reproducibility must be an object");
  else if (candidate.reproducibility.deterministic !== true) issues.push("reproducibility.deterministic must be true");
  if (candidate.slice === "tool-heavy" && candidate.tool_fixture_mode !== "read_only_fixture_safe") issues.push("tool-heavy candidates require tool_fixture_mode=read_only_fixture_safe");

  if (options.admit) {
    if (candidate.provenance_type !== "real") issues.push("--admit requires provenance_type=real");
    if (candidate.privacy_scrubbed !== true) issues.push("--admit requires privacy_scrubbed=true");
    if (candidate.contains_real_user_data !== false) issues.push("--admit requires contains_real_user_data=false after redaction");
    if (String(candidate.user_task_redacted ?? "").includes("SECRET") || String(candidate.user_task_redacted ?? "").includes("API_KEY")) issues.push("redacted task text appears to contain secret placeholders that must be removed before admission");
  }
  return issues;
}

async function upsertManifest(manifestPath, traceRecord) {
  let manifest;
  try {
    manifest = JSON.parse(await readFile(resolvePath(manifestPath), "utf8"));
  } catch {
    manifest = {
      manifest_version: MANIFEST_VERSION,
      program: "OpenClawBrain V5 evidence scoreboard",
      mode: "production",
      evidence_label: "PRODUCTION EVIDENCE REQUIRES 40 ADMITTED REAL PRIVACY-SCRUBBED TRACES",
      production_requirements: {
        min_admitted_real_traces: REQUIRED_REAL_TRACE_COUNT,
        required_slice_minimums: SLICE_MINIMUMS,
      },
      traces: [],
    };
  }
  if (manifest.manifest_version !== MANIFEST_VERSION) throw new Error(`manifest_version must be ${MANIFEST_VERSION}`);
  if (manifest.mode !== "production") throw new Error("trace admission writes only production manifests");
  if (!Array.isArray(manifest.traces)) throw new Error("manifest.traces must be an array");
  manifest.traces = manifest.traces.filter((trace) => trace.id !== traceRecord.id);
  manifest.traces.push(traceRecord);
  manifest.traces.sort((a, b) => a.id.localeCompare(b.id));
  await writeJson(manifestPath, manifest);
  return manifest;
}

function parseArgs(argv) {
  const args = { admit: false, outRoot: DEFAULT_OUT_ROOT, manifest: DEFAULT_MANIFEST };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--candidate") args.candidate = argv[++i];
    else if (arg === "--out-root") args.outRoot = argv[++i];
    else if (arg === "--manifest") args.manifest = argv[++i];
    else if (arg === "--admit") args.admit = true;
    else if (arg === "--help" || arg === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function help() {
  return `Usage: node scripts/traces/admit.mjs --candidate trace.json [--admit] [--out-root eval/traces/production] [--manifest eval/traces/production.manifest.json]\n\nDefault records a candidate without product-evidence admission. --admit fails closed unless the candidate is real, privacy-scrubbed, redacted, deterministic, and slice-valid.`;
}

async function runCli() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(help()); return; }
  const result = await admitTraceCandidate(args);
  console.log(JSON.stringify(result, null, 2));
}

function priorityClass(slice) { return PRIMARY_SLICES.has(slice) ? "primary" : "secondary"; }
function isRecord(value) { return typeof value === "object" && value !== null && !Array.isArray(value); }
function requireNonEmptyString(value, field, issues) { if (typeof value[field] !== "string" || value[field].trim() === "") issues.push(`${field} must be a non-empty string`); }
function requireBoolean(value, field, issues) { if (typeof value[field] !== "boolean") issues.push(`${field} must be a boolean`); }
function required(value, name) { if (!value) throw new Error(`--${name} is required`); return value; }
function safePathSegment(value) { return value.replace(/[^a-zA-Z0-9._-]/g, "-"); }
function resolvePath(path) { return resolve(PROJECT_ROOT, path); }
async function writeJson(path, value) { await mkdir(dirname(resolvePath(path)), { recursive: true }); await writeFile(resolvePath(path), `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli().catch((error) => {
    console.error(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error), issues: error?.issues ?? undefined }, null, 2));
    process.exitCode = 1;
  });
}
