import test from "node:test";
import assert from "node:assert/strict";
import { summarizeLedgerRows } from "../src/summary.ts";
import { productionRows, row } from "./helpers.ts";

test("summary marks synthetic smoke as non-evidence", () => {
  const summary = summarizeLedgerRows([row({ backend: "none", memory_fired: false, should_have_fired: false, false_fire: false }), row({ backend: "full-ocb" })]);
  assert.equal(summary.synthetic_pipeline_validation_only, true);
  assert.equal(summary.evidence_e2e_complete, false);
  assert.ok(summary.blockers.includes("synthetic/smoke data is not product evidence"));
});

test("summary requires 40 admitted rows and slice minimums", () => {
  const rows = [
    ...productionRows("none", 40),
    ...productionRows("correction-only", 40),
    ...productionRows("correction+heuristics", 40),
    ...productionRows("full-ocb", 40),
  ];
  const summary = summarizeLedgerRows(rows);
  assert.equal(summary.product_evidence_count, 160);
  assert.equal(summary.evidence_e2e_complete, true);
});
