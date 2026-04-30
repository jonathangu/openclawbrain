import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";
import { captureRuntimeEvent, normalizeRuntimeEvent } from "./capture-event.mjs";
import { exportCandidate } from "./export-candidate.mjs";

const projectRoot = resolve(import.meta.dirname, "..", "..");

test("captures redacted runtime event and manifest as candidate-only", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-capture-"));
  try {
    const eventPath = join(tmp, "event.json");
    await writeFile(eventPath, JSON.stringify(runtimeEvent(), null, 2));
    const result = await captureRuntimeEvent({ event: eventPath, outDir: relative(projectRoot, tmp), capturedAt: "2026-04-29T22:20:00-07:00" });
    assert.equal(result.counts_as_product_evidence, false);
    const event = JSON.parse(await readFile(join(tmp, "telegram-25311.json"), "utf8"));
    assert.equal(event.schema_version, "ocb.runtime.event.v1");
    assert.equal(event.evidence_status, "CANDIDATE ONLY / NOT PRODUCT EVIDENCE");
    const manifest = JSON.parse(await readFile(join(tmp, "manifest.json"), "utf8"));
    assert.equal(manifest.events.length, 1);
    assert.equal(manifest.events[0].counts_as_product_evidence, false);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("captured event can flow into candidate exporter", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-capture-"));
  try {
    const eventPath = join(tmp, "event.json");
    await writeFile(eventPath, JSON.stringify(runtimeEvent({ event_id: "session:runtime-capture", trace_id: "runtime-capture-001" }), null, 2));
    const captured = await captureRuntimeEvent({ event: eventPath, outDir: relative(projectRoot, tmp), capturedAt: "2026-04-29T22:20:00-07:00" });
    const candidatePath = join(tmp, "candidate.json");
    const exported = await exportCandidate({ event: resolve(projectRoot, captured.event_file), out: candidatePath });
    assert.equal(exported.trace_id, "runtime-capture-001");
    const candidate = JSON.parse(await readFile(candidatePath, "utf8"));
    assert.equal(candidate.privacy_scrubbed, true);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("capture rejects raw unsafe event payloads", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-capture-"));
  try {
    const eventPath = join(tmp, "event.json");
    await writeFile(eventPath, JSON.stringify(runtimeEvent({ raw_text: "unredacted", password: "PASSWORD=abc" }), null, 2));
    await assert.rejects(captureRuntimeEvent({ event: eventPath, outDir: relative(projectRoot, tmp) }), /raw\/unredacted field/);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("normalization creates stable trace id when missing", () => {
  const event = normalizeRuntimeEvent(runtimeEvent({ trace_id: undefined, event_id: "abc/123" }), { capturedAt: "now" });
  assert.equal(event.trace_id, "runtime-abc-123");
  assert.equal(event.captured_at, "now");
});

function runtimeEvent(overrides = {}) {
  return {
    event_id: "telegram:25311",
    trace_id: "runtime-telegram-25311",
    source: "telegram",
    provenance_type: "real",
    title: "Status request while continuing implementation",
    task_type: "status_and_continue",
    user_task_redacted: "User asks how the rebuild is going while implementation continues.",
    current_context_redacted: "V5 scoreboard exists; runtime capture hook is the current next step.",
    expected_memory_opportunity: true,
    memory_fired: true,
    backend_observed: "full-ocb",
    retrieved_memory_ids_redacted: ["task-ledger-current"],
    slice: "continuation",
    privacy_scrubbed: true,
    contains_real_user_data: false,
    collected_at: "2026-04-29T22:20:00-07:00",
    redaction_notes: "Identifiers and raw message content removed; retained only task intent and metadata.",
    memory_snapshot_id: "snapshot-runtime-redacted-002",
    memory_snapshot_created_at: "2026-04-29T22:19:00-07:00",
    ocb_config_hash: "sha256:runtime-config-redacted",
    model_id: "openai-codex/gpt-5.5",
    prompt_hash: "sha256:runtime-prompt-redacted",
    code_commit: "19f2765",
    allowed_evidence: ["redacted user intent", "redacted project state"],
    reproducibility: { deterministic: true, replay_safe: true },
    ...overrides,
  };
}
