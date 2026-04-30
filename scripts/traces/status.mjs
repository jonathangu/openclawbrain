#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_MANIFEST = "eval/traces/production.manifest.json";
const REQUIRED_REAL_TRACE_COUNT = 40;
const SLICE_MINIMUMS = {
  "direct-answer": 6,
  "continuation": 6,
  "correction-follow-up": 8,
  "retrieval-heavy": 6,
  "tool-heavy": 6,
  "stale-memory-conflict": 8,
};

export async function productionTraceStatus(manifestPath = DEFAULT_MANIFEST) {
  let manifest = null;
  const blockers = [];
  try {
    manifest = JSON.parse(await readFile(resolve(PROJECT_ROOT, manifestPath), "utf8"));
  } catch {
    blockers.push(`production manifest missing: ${manifestPath}`);
    manifest = { traces: [] };
  }
  const traces = Array.isArray(manifest.traces) ? manifest.traces : [];
  const admittedReal = traces.filter((trace) => trace.admitted === true && trace.provenance_type === "real" && trace.counts_as_product_evidence === true && trace.privacy_scrubbed === true);
  const bySlice = Object.fromEntries(Object.keys(SLICE_MINIMUMS).map((slice) => [slice, admittedReal.filter((trace) => trace.slice === slice).length]));
  if (admittedReal.length < REQUIRED_REAL_TRACE_COUNT) blockers.push(`admitted real privacy-scrubbed product traces ${admittedReal.length}/${REQUIRED_REAL_TRACE_COUNT}`);
  for (const [slice, minimum] of Object.entries(SLICE_MINIMUMS)) {
    if ((bySlice[slice] ?? 0) < minimum) blockers.push(`slice ${slice} ${bySlice[slice] ?? 0}/${minimum}`);
  }
  const evidenceE2EComplete = blockers.length === 0;
  return {
    ok: true,
    manifest: manifestPath,
    trace_count: traces.length,
    admitted_real_product_trace_count: admittedReal.length,
    required_real_trace_count: REQUIRED_REAL_TRACE_COUNT,
    by_slice: bySlice,
    blockers,
    evidence_e2e_complete: evidenceE2EComplete,
  };
}

function parseArgs(argv) {
  const args = { manifest: DEFAULT_MANIFEST };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--manifest") args.manifest = argv[++i];
    else if (arg === "--help" || arg === "-h") args.help = true;
    else if (arg === "--") continue;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

async function runCli() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(`Usage: node scripts/traces/status.mjs [--manifest ${DEFAULT_MANIFEST}]`); return; }
  console.log(JSON.stringify(await productionTraceStatus(args.manifest), null, 2));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli().catch((error) => {
    console.error(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }, null, 2));
    process.exitCode = 1;
  });
}
