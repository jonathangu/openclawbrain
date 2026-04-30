import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";
import { exportCandidate, validateRuntimeEvent } from "./export-candidate.mjs";

const projectRoot = resolve(import.meta.dirname, "..", "..");

test("exports redacted runtime event into admission candidate", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-runtime-export-"));
  try {
    const eventPath = join(tmp, "event.json");
    const outPath = join(tmp, "candidate.json");
    await writeFile(eventPath, JSON.stringify(runtimeEvent(), null, 2));
    const result = await exportCandidate({ event: eventPath, out: outPath });
    assert.equal(result.ok, true);
    const candidate = JSON.parse(await readFile(outPath, "utf8"));
    assert.equal(candidate.trace_id, "runtime-telegram-25309");
    assert.equal(candidate.privacy_scrubbed, true);
    assert.equal(candidate.runtime_observation.export_schema, "ocb.runtime.trace_candidate.v1");
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("export plus admit creates admitted production manifest", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-runtime-export-"));
  try {
    const eventPath = join(tmp, "event.json");
    const outPath = join(tmp, "candidate.json");
    await writeFile(eventPath, JSON.stringify(runtimeEvent({ event_id: "session:abc-123", trace_id: "prod-runtime-001" }), null, 2));
    const result = await exportCandidate({
      event: eventPath,
      out: outPath,
      admit: true,
      outRoot: relative(projectRoot, join(tmp, "traces")),
      manifest: relative(projectRoot, join(tmp, "manifest.json")),
    });
    assert.equal(result.admission.admitted, true);
    const manifest = JSON.parse(await readFile(join(tmp, "manifest.json"), "utf8"));
    assert.equal(manifest.traces[0].id, "prod-runtime-001");
    assert.equal(manifest.traces[0].counts_as_product_evidence, true);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("raw fields and secrets fail closed", () => {
  const issues = validateRuntimeEvent(runtimeEvent({ raw_messages: ["private raw text"], api_key: "sk-abc1234567890" }));
  assert.match(issues.join("\n"), /raw\/unredacted field/);
  assert.match(issues.join("\n"), /secret-like/);
});

test("unscrubbed runtime event cannot export", () => {
  const issues = validateRuntimeEvent(runtimeEvent({ privacy_scrubbed: false, contains_real_user_data: true }));
  assert.match(issues.join("\n"), /privacy_scrubbed=true/);
  assert.match(issues.join("\n"), /contains_real_user_data=false/);
});

function runtimeEvent(overrides = {}) {
  return {
    event_id: "telegram:25309",
    trace_id: "runtime-telegram-25309",
    source: "telegram",
    provenance_type: "real",
    title: "Keep-going continuation request",
    task_type: "agent_followup",
    user_task_redacted: "User asks for status and requests continued OpenClawBrain implementation.",
    current_context_redacted: "Repo contains V5 scoreboard and smoke E2E gate; next task is runtime trace export.",
    expected_memory_opportunity: true,
    memory_fired: true,
    backend_observed: "full-ocb",
    retrieved_memory_ids_redacted: ["task-ledger-current"],
    slice: "continuation",
    privacy_scrubbed: true,
    contains_real_user_data: false,
    collected_at: "2026-04-29T21:13:00-07:00",
    redaction_notes: "Direct identifiers and raw message body removed; retained only task intent and runtime metadata.",
    memory_snapshot_id: "snapshot-runtime-redacted-001",
    memory_snapshot_created_at: "2026-04-29T21:12:00-07:00",
    ocb_config_hash: "sha256:runtime-config-redacted",
    model_id: "openai-codex/gpt-5.5",
    prompt_hash: "sha256:runtime-prompt-redacted",
    code_commit: "9b2b9b5",
    allowed_evidence: ["redacted user intent", "redacted task ledger state"],
    reproducibility: { deterministic: true, replay_safe: true },
    ...overrides,
  };
}
