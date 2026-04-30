import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";
import { productionTraceStatus } from "./status.mjs";

const projectRoot = resolve(import.meta.dirname, "..", "..");

test("missing production manifest reports honest blockers", async () => {
  const status = await productionTraceStatus("eval/traces/does-not-exist.json");
  assert.equal(status.evidence_e2e_complete, false);
  assert.match(status.blockers.join("\n"), /production manifest missing/);
  assert.equal(status.admitted_real_product_trace_count, 0);
});

test("production trace status counts admitted real slices", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "ocb-trace-status-"));
  try {
    const manifestPath = join(tmp, "manifest.json");
    await mkdir(tmp, { recursive: true });
    await writeFile(manifestPath, JSON.stringify({ traces: [trace("one", "continuation"), trace("two", "tool-heavy", { privacy_scrubbed: false }), trace("three", "direct-answer", { provenance_type: "synthetic" })] }, null, 2));
    const status = await productionTraceStatus(relative(projectRoot, manifestPath));
    assert.equal(status.admitted_real_product_trace_count, 1);
    assert.equal(status.by_slice.continuation, 1);
    assert.equal(status.by_slice["tool-heavy"], 0);
    assert.equal(status.evidence_e2e_complete, false);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

function trace(id, slice, overrides = {}) {
  return {
    id,
    slice,
    admitted: true,
    provenance_type: "real",
    counts_as_product_evidence: true,
    privacy_scrubbed: true,
    ...overrides,
  };
}
