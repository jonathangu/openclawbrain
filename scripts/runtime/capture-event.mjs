#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { validateRuntimeEvent } from "./export-candidate.mjs";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_OUT_DIR = "eval/runtime-events";
const EVENT_SCHEMA_VERSION = "ocb.runtime.event.v1";
const EVENT_STATUS = "CANDIDATE ONLY / NOT PRODUCT EVIDENCE";

export async function captureRuntimeEvent(options) {
  const inputPath = required(options.event, "event");
  const outDir = options.outDir ?? DEFAULT_OUT_DIR;
  const raw = JSON.parse(await readFile(resolvePath(inputPath), "utf8"));
  const event = normalizeRuntimeEvent(raw, { capturedAt: options.capturedAt });
  const issues = validateRuntimeEvent(event);
  if (issues.length > 0) {
    const error = new Error(`runtime capture rejected: ${issues.join("; ")}`);
    error.issues = issues;
    throw error;
  }

  const eventFile = join(outDir, `${safeId(event.event_id)}.json`);
  await writeJson(eventFile, event);
  const manifest = await upsertManifest(outDir, event);
  return {
    ok: true,
    event_id: event.event_id,
    trace_id: event.trace_id,
    event_file: eventFile,
    manifest: join(outDir, "manifest.json"),
    counts_as_product_evidence: false,
    captured_event_count: manifest.events.length,
  };
}

export function normalizeRuntimeEvent(raw, options = {}) {
  if (!isRecord(raw)) return raw;
  const eventId = stringValue(raw.event_id || raw.turn_id || raw.request_id || stableEventId(raw));
  const capturedAt = options.capturedAt || raw.captured_at || new Date().toISOString();
  const traceId = stringValue(raw.trace_id || `runtime-${safeId(eventId)}`);
  return {
    schema_version: EVENT_SCHEMA_VERSION,
    captured_at: capturedAt,
    evidence_status: EVENT_STATUS,
    counts_as_product_evidence: false,
    ...raw,
    event_id: eventId,
    trace_id: traceId,
  };
}

async function upsertManifest(outDir, event) {
  const manifestPath = join(outDir, "manifest.json");
  let manifest;
  try {
    manifest = JSON.parse(await readFile(resolvePath(manifestPath), "utf8"));
  } catch {
    manifest = {
      manifest_version: "ocb.runtime.events.manifest.v1",
      evidence_status: EVENT_STATUS,
      counts_as_product_evidence: false,
      events: [],
    };
  }
  if (!Array.isArray(manifest.events)) throw new Error("runtime events manifest must contain events array");
  const record = {
    event_id: event.event_id,
    trace_id: event.trace_id,
    source: event.source,
    slice: event.slice,
    expected_memory_opportunity: event.expected_memory_opportunity,
    memory_fired: event.memory_fired ?? null,
    captured_at: event.captured_at,
    event_file: `${safeId(event.event_id)}.json`,
    counts_as_product_evidence: false,
  };
  manifest.events = manifest.events.filter((item) => item.event_id !== event.event_id);
  manifest.events.push(record);
  manifest.events.sort((a, b) => a.event_id.localeCompare(b.event_id));
  await writeJson(manifestPath, manifest);
  return manifest;
}

function parseArgs(argv) {
  const args = { outDir: DEFAULT_OUT_DIR };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--event") args.event = argv[++i];
    else if (arg === "--out-dir") args.outDir = argv[++i];
    else if (arg === "--captured-at") args.capturedAt = argv[++i];
    else if (arg === "--help" || arg === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function help() {
  return `Usage: node scripts/runtime/capture-event.mjs --event redacted-runtime-event.json [--out-dir eval/runtime-events]\n\nValidates and stores a stable redacted runtime event. Captured events are candidate-only and never product evidence until exported and admitted through the production gate.`;
}

async function runCli() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(help()); return; }
  console.log(JSON.stringify(await captureRuntimeEvent(args), null, 2));
}

function stableEventId(raw) {
  const hash = createHash("sha256").update(JSON.stringify(raw)).digest("hex").slice(0, 16);
  return `event-${hash}`;
}
function stringValue(value) { return typeof value === "string" && value.trim() ? value.trim() : String(value ?? "").trim(); }
function safeId(value) { return String(value).trim().replace(/[^a-zA-Z0-9._-]/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "event"; }
function isRecord(value) { return typeof value === "object" && value !== null && !Array.isArray(value); }
function required(value, name) { if (!value) throw new Error(`--${name} is required`); return value; }
function resolvePath(path) { return resolve(PROJECT_ROOT, path); }
async function writeJson(path, value) { await mkdir(dirname(resolvePath(path)), { recursive: true }); await writeFile(resolvePath(path), `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli().catch((error) => {
    console.error(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error), issues: error?.issues ?? undefined }, null, 2));
    process.exitCode = 1;
  });
}
