import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";
import { admitTraceCandidate } from "./admit.mjs";

const projectRoot = resolve(import.meta.dirname, "..", "..");

test("records candidates without admitting product evidence by default", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-admit-"));
  try {
    const candidatePath = join(tmp, "candidate.json");
    await writeFile(candidatePath, JSON.stringify(candidate({ provenance_type: "repo-derived", privacy_scrubbed: true, contains_real_user_data: false }), null, 2));
    const result = await admitTraceCandidate({
      candidate: candidatePath,
      outRoot: relative(projectRoot, join(tmp, "traces")),
      manifest: relative(projectRoot, join(tmp, "manifest.json")),
    });
    assert.equal(result.admitted, false);
    assert.equal(result.counts_as_product_evidence, false);
    const manifest = JSON.parse(await readFile(join(tmp, "manifest.json"), "utf8"));
    assert.equal(manifest.traces[0].admitted, false);
    assert.equal(manifest.traces[0].counts_as_product_evidence, false);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("admits only redacted real deterministic production candidates", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-admit-"));
  try {
    const candidatePath = join(tmp, "candidate.json");
    await writeFile(candidatePath, JSON.stringify(candidate(), null, 2));
    const result = await admitTraceCandidate({
      candidate: candidatePath,
      admit: true,
      outRoot: relative(projectRoot, join(tmp, "traces")),
      manifest: relative(projectRoot, join(tmp, "manifest.json")),
    });
    assert.equal(result.admitted, true);
    assert.equal(result.counts_as_product_evidence, true);
    assert.equal(result.admitted_real_product_trace_count, 1);
    const input = JSON.parse(await readFile(join(tmp, "traces", "prod-redacted-001", "input.json"), "utf8"));
    assert.equal(input.expected_memory_opportunity, true);
    const provenance = JSON.parse(await readFile(join(tmp, "traces", "prod-redacted-001", "provenance.json"), "utf8"));
    assert.equal(provenance.contains_real_user_data, false);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("fails closed when --admit candidate is not scrubbed", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-admit-"));
  try {
    const candidatePath = join(tmp, "candidate.json");
    await writeFile(candidatePath, JSON.stringify(candidate({ privacy_scrubbed: false }), null, 2));
    await assert.rejects(
      admitTraceCandidate({
        candidate: candidatePath,
        admit: true,
        outRoot: relative(projectRoot, join(tmp, "traces")),
        manifest: relative(projectRoot, join(tmp, "manifest.json")),
      }),
      /privacy_scrubbed=true/,
    );
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

function candidate(overrides = {}) {
  return {
    trace_id: "prod-redacted-001",
    title: "Redacted continuation trace",
    source: "session",
    provenance_type: "real",
    slice: "continuation",
    task_type: "implementation_followup",
    user_task_redacted: "Continue the approved implementation from the current repo state.",
    current_context_redacted: "Repo has a failing test and a clear next patch.",
    expected_memory_opportunity: true,
    privacy_scrubbed: true,
    contains_real_user_data: false,
    collected_at: "2026-04-29T20:45:00-07:00",
    redaction_notes: "Names, account IDs, secrets, and raw private content removed before admission.",
    memory_snapshot_id: "snapshot-redacted-001",
    memory_snapshot_created_at: "2026-04-29T20:44:00-07:00",
    ocb_config_hash: "sha256:redacted-config",
    model_id: "redacted-model-id",
    prompt_hash: "sha256:redacted-prompt",
    code_commit: "bf22add",
    allowed_evidence: ["redacted task intent", "redacted repo state"],
    reproducibility: { deterministic: true, replay_safe: true },
    ...overrides,
  };
}
