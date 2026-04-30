#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_MANIFEST = "eval/traces/manifest.json";
const SMOKE_NOTICE = "NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY";
const VALID_MODES = new Set(["smoke", "production"]);
const VALID_SLICES = new Set([
  "direct-answer",
  "continuation",
  "correction-follow-up",
  "retrieval-heavy",
  "tool-heavy",
  "stale-memory-conflict",
]);
const PRIMARY_SLICES = new Set(["correction-follow-up", "continuation", "stale-memory-conflict"]);
const VALID_PROVENANCE = new Set(["real", "synthetic", "repo-derived", "adversarial"]);
const PRODUCTION_SLICE_MINIMUMS = {
  "direct-answer": 6,
  "continuation": 6,
  "correction-follow-up": 8,
  "retrieval-heavy": 6,
  "tool-heavy": 6,
  "stale-memory-conflict": 8,
};
const REQUIRED_TRACE_FIELDS = [
  "id", "path", "input_file", "provenance_file", "mode", "admitted",
  "provenance_type", "counts_as_product_evidence", "privacy_scrubbed", "slice", "priority_class",
];

function parseArgs(argv) {
  const args = { mode: "smoke", manifest: DEFAULT_MANIFEST };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--mode") args.mode = argv[++i];
    else if (arg === "--manifest") args.manifest = argv[++i];
    else if (arg === "--help" || arg === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (args.mode === "prod") args.mode = "production";
  if (!VALID_MODES.has(args.mode)) throw new Error(`Invalid --mode ${JSON.stringify(args.mode)}; expected smoke or production`);
  if (!args.manifest) throw new Error("Missing --manifest value");
  return args;
}
function readJson(relativePath, errors) {
  try { return JSON.parse(readFileSync(resolve(PROJECT_ROOT, relativePath), "utf8")); }
  catch (error) { errors.push(`${relativePath}: cannot read valid JSON (${error.message})`); return null; }
}
function pushIf(condition, errors, message) { if (condition) errors.push(message); }
function countBy(items, getKey) {
  const counts = {};
  for (const item of items) {
    const key = getKey(item);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}
function expectedPriority(slice) { return PRIMARY_SLICES.has(slice) ? "primary" : "secondary"; }
function validateTraceRecord(trace, errors) {
  for (const field of REQUIRED_TRACE_FIELDS) pushIf(!(field in trace), errors, `${trace.id || "<unknown>"}: missing required field ${field}`);
  if (typeof trace.id !== "string" || trace.id.length === 0) return null;
  pushIf(!VALID_SLICES.has(trace.slice), errors, `${trace.id}: slice must be one of ${[...VALID_SLICES].join(", ")}`);
  pushIf(trace.priority_class !== expectedPriority(trace.slice), errors, `${trace.id}: priority_class must be ${expectedPriority(trace.slice)} for slice ${trace.slice}`);
  pushIf(!VALID_PROVENANCE.has(trace.provenance_type), errors, `${trace.id}: invalid provenance_type ${trace.provenance_type}`);
  pushIf(typeof trace.admitted !== "boolean", errors, `${trace.id}: admitted must be boolean`);
  pushIf(typeof trace.counts_as_product_evidence !== "boolean", errors, `${trace.id}: counts_as_product_evidence must be boolean`);
  pushIf(typeof trace.privacy_scrubbed !== "boolean", errors, `${trace.id}: privacy_scrubbed must be boolean`);

  const inputPath = join(trace.path || "", trace.input_file || "");
  const provenancePath = join(trace.path || "", trace.provenance_file || "");
  const input = readJson(inputPath, errors);
  const provenance = readJson(provenancePath, errors);
  if (input) {
    pushIf(input.trace_id !== trace.id, errors, `${inputPath}: trace_id does not match manifest id ${trace.id}`);
    pushIf(input.slice !== trace.slice, errors, `${inputPath}: slice does not match manifest slice ${trace.slice}`);
    pushIf(!input.non_evidence_notice && trace.mode === "smoke", errors, `${inputPath}: smoke input missing non_evidence_notice`);
    pushIf(typeof input.expected_memory_opportunity !== "boolean", errors, `${inputPath}: expected_memory_opportunity must be boolean`);
  }
  if (provenance) {
    for (const field of ["trace_id", "provenance_type", "mode", "counts_as_product_evidence", "privacy_scrubbed", "slice"]) {
      pushIf(provenance[field] !== trace[field], errors, `${trace.id}: ${field} mismatch between manifest and provenance.json`);
    }
    pushIf(provenance.contains_real_user_data !== false && trace.provenance_type === "synthetic", errors, `${trace.id}: synthetic provenance must declare contains_real_user_data=false`);
    pushIf(!provenance.reproducibility || provenance.reproducibility.deterministic !== true, errors, `${trace.id}: reproducibility.deterministic=true is required`);
    if (trace.slice === "tool-heavy") pushIf(provenance.tool_fixture_mode !== "read_only_fixture_safe", errors, `${trace.id}: tool-heavy trace requires tool_fixture_mode=read_only_fixture_safe`);
  }
  pushIf(trace.mode === "smoke" && trace.provenance_type !== "synthetic", errors, `${trace.id}: smoke traces must use provenance_type=synthetic`);
  pushIf(trace.mode === "smoke" && trace.counts_as_product_evidence !== false, errors, `${trace.id}: smoke traces must not count as product evidence`);
  pushIf(trace.mode === "smoke" && trace.admitted !== false, errors, `${trace.id}: smoke traces must not be admitted evidence`);
  return trace;
}
function validateManifest(manifestPath) {
  const errors = [];
  const manifest = readJson(manifestPath, errors);
  if (!manifest) return { manifest: null, errors, traces: [] };
  pushIf(manifest.manifest_version !== "ocb.traces.manifest.v1", errors, "manifest_version must be ocb.traces.manifest.v1");
  pushIf(!Array.isArray(manifest.traces), errors, "manifest.traces must be an array");
  pushIf(manifest.evidence_label !== SMOKE_NOTICE && manifest.mode === "smoke", errors, `manifest evidence_label must be ${JSON.stringify(SMOKE_NOTICE)}`);
  const seenIds = new Set();
  const traces = [];
  for (const trace of manifest.traces || []) {
    if (seenIds.has(trace.id)) errors.push(`${trace.id}: duplicate trace id`);
    seenIds.add(trace.id);
    const record = validateTraceRecord(trace, errors);
    if (record) traces.push(record);
  }
  return { manifest, errors, traces };
}
function buildCoverage(traces) {
  return {
    total_traces: traces.length,
    by_mode: countBy(traces, (t) => t.mode),
    by_provenance_type: countBy(traces, (t) => t.provenance_type),
    by_slice: countBy(traces, (t) => t.slice),
    product_evidence_trace_count: traces.filter((t) => t.counts_as_product_evidence).length,
    admitted_real_trace_count: traces.filter((t) => t.admitted && t.provenance_type === "real" && t.counts_as_product_evidence).length,
  };
}
function validateSmoke(manifest, traces, errors) {
  const smokeTraces = traces.filter((trace) => trace.mode === "smoke");
  pushIf(smokeTraces.length < 4 || smokeTraces.length > 8, errors, `smoke mode requires 4-8 smoke traces; found ${smokeTraces.length}`);
  pushIf(smokeTraces.length !== traces.length, errors, "smoke manifest must contain only smoke traces for this lane");
  for (const trace of smokeTraces) {
    pushIf(trace.provenance_type !== "synthetic", errors, `${trace.id}: smoke trace must set provenance_type=synthetic`);
    pushIf(trace.counts_as_product_evidence !== false, errors, `${trace.id}: smoke trace must set counts_as_product_evidence=false`);
    pushIf(trace.admitted !== false, errors, `${trace.id}: smoke trace must set admitted=false`);
  }
  pushIf(manifest.mode !== "smoke", errors, "smoke validation requires manifest.mode=smoke");
}
function validateProduction(manifest, traces, errors) {
  const requirements = manifest.production_requirements || { min_admitted_real_traces: 40, required_slice_minimums: PRODUCTION_SLICE_MINIMUMS };
  const admittedReal = traces.filter((t) => t.admitted && t.provenance_type === "real" && t.counts_as_product_evidence === true);
  const minRealTraces = requirements.min_admitted_real_traces || 40;
  pushIf(admittedReal.length < minRealTraces, errors, `production mode requires at least ${minRealTraces} admitted real product-evidence traces; found ${admittedReal.length}`);
  const sliceCounts = countBy(admittedReal, (t) => t.slice);
  for (const [slice, minimum] of Object.entries(requirements.required_slice_minimums || PRODUCTION_SLICE_MINIMUMS)) {
    pushIf((sliceCounts[slice] || 0) < minimum, errors, `production mode requires slice ${slice} >= ${minimum}; found ${sliceCounts[slice] || 0}`);
  }
  for (const trace of admittedReal) {
    pushIf(trace.privacy_scrubbed !== true, errors, `${trace.id}: admitted real trace must be privacy_scrubbed=true`);
    pushIf(trace.mode !== "production", errors, `${trace.id}: admitted real trace must use mode=production`);
  }
}
function main() {
  let args;
  try { args = parseArgs(process.argv.slice(2)); }
  catch (error) { console.error(`trace manifest validation failed: ${error.message}`); process.exit(2); return; }
  if (args.help) { console.log(`Usage: node scripts/traces/validate.mjs [--mode smoke|production] [--manifest ${DEFAULT_MANIFEST}]`); return; }
  const { manifest, errors, traces } = validateManifest(args.manifest);
  if (manifest) {
    if (args.mode === "smoke") validateSmoke(manifest, traces, errors);
    if (args.mode === "production") validateProduction(manifest, traces, errors);
  }
  const result = {
    validator: "ocb-v5-trace-manifest-validator",
    mode: args.mode,
    manifest: args.manifest,
    valid: errors.length === 0,
    engineering_e2e_signal: args.mode === "smoke" && errors.length === 0,
    evidence_e2e_complete: false,
    evidence_label: args.mode === "smoke" ? SMOKE_NOTICE : "PRODUCTION EVIDENCE REQUIRES 40 ADMITTED REAL PRIVACY-SCRUBBED TRACES",
    coverage: buildCoverage(traces),
    errors,
  };
  console.log(JSON.stringify(result, null, 2));
  if (errors.length > 0) process.exit(1);
}
main();
