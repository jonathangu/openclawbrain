import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";
import { buildRuntimeDecisionEvent, decideRuntimeTurn, validateDecisionInput } from "./decide.mjs";
import { exportCandidate } from "./export-candidate.mjs";

const projectRoot = resolve(import.meta.dirname, "..", "..");

test("decides fire and captures a runtime event", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-decide-"));
  try {
    const inputPath = join(tmp, "input.json");
    await writeFile(inputPath, JSON.stringify(decisionInput(), null, 2));
    const result = await decideRuntimeTurn({ input: inputPath, outDir: relative(projectRoot, tmp), capturedAt: "2026-04-30T09:00:00-07:00" });
    assert.equal(result.action, "fire");
    assert.deepEqual(result.selected_memory_ids_redacted, ["mem-correction-redacted"]);
    assert.equal(result.counts_as_product_evidence, false);
    const event = JSON.parse(await readFile(join(tmp, "telegram-25317.json"), "utf8"));
    assert.equal(event.memory_fired, true);
    assert.equal(event.backend_observed, "minimal-runtime-decision");
    assert.equal(event.reproducibility.deterministic, true);
    assert.equal(event.reproducibility.mutating_external_services, false);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("decides stay_silent when restraint is correct", () => {
  const event = buildRuntimeDecisionEvent(decisionInput({
    event_id: "telegram:25318",
    expected_memory_opportunity: false,
    memory_candidates_redacted: [{ id: "mem-low-redacted", relevance_score: 0.31 }],
  }));
  assert.equal(event.memory_fired, false);
  assert.deepEqual(event.retrieved_memory_ids_redacted, []);
  assert.match(event.decision_reason, /Stayed silent/);
});

test("rejects raw fields, secrets, and unredacted privacy state", () => {
  const issues = validateDecisionInput(decisionInput({
    raw_messages: ["private raw content"],
    api_key: "sk-abc1234567890",
    privacy_scrubbed: false,
    contains_real_user_data: true,
  }));
  assert.match(issues.join("\n"), /raw\/unredacted field/);
  assert.match(issues.join("\n"), /secret-like/);
  assert.match(issues.join("\n"), /privacy_scrubbed=true/);
  assert.match(issues.join("\n"), /contains_real_user_data=false/);
});

test("rejects malformed input", () => {
  assert.throws(() => buildRuntimeDecisionEvent({ source: "telegram" }), /runtime decision rejected/);
});

test("captured decision event is compatible with candidate export", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-decide-"));
  try {
    const inputPath = join(tmp, "input.json");
    await writeFile(inputPath, JSON.stringify(decisionInput({ trace_id: "runtime-decision-export-001" }), null, 2));
    const result = await decideRuntimeTurn({ input: inputPath, outDir: relative(projectRoot, tmp) });
    const candidatePath = join(tmp, "candidate.json");
    const exported = await exportCandidate({ event: resolve(projectRoot, result.event_file), out: candidatePath });
    assert.equal(exported.trace_id, "runtime-decision-export-001");
    const candidate = JSON.parse(await readFile(candidatePath, "utf8"));
    assert.equal(candidate.runtime_observation.memory_fired, true);
    assert.deepEqual(candidate.runtime_observation.retrieved_memory_ids_redacted, ["mem-correction-redacted"]);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

function decisionInput(overrides = {}) {
  return {
    event_id: "telegram:25317",
    trace_id: "runtime-telegram-25317",
    source: "telegram",
    provenance_type: "real",
    title: "Continuation request during V5 rebuild",
    task_type: "agent_turn_decision",
    user_task_redacted: "User asks the agent to finish the V5 rebuild end to end.",
    current_context_redacted: "Scoreboard pipeline exists; minimal runtime decision event emission is the current task.",
    expected_memory_opportunity: true,
    memory_candidates_redacted: [{ id: "mem-correction-redacted", relevance_score: 0.92 }],
    slice: "continuation",
    privacy_scrubbed: true,
    contains_real_user_data: false,
    collected_at: "2026-04-30T09:00:00-07:00",
    redaction_notes: "Raw message text and identifiers removed; only task intent and redacted memory IDs remain.",
    memory_snapshot_id: "snapshot-runtime-redacted-003",
    memory_snapshot_created_at: "2026-04-30T08:59:00-07:00",
    ocb_config_hash: "sha256:runtime-config-redacted-v5",
    model_id: "openai-codex/gpt-5.5",
    prompt_hash: "sha256:runtime-prompt-redacted-v5",
    code_commit: "9d1e80c",
    reproducibility: { replay_safe: true },
    ...overrides,
  };
}
