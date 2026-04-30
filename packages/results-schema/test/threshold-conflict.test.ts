import test from "node:test";
import assert from "node:assert/strict";
import { applyThresholds } from "../src/thresholds.ts";
import { summarizeLedgerRows } from "../src/summary.ts";
import { productionRows, row } from "./helpers.ts";

test("thresholds block when evidence is incomplete", () => {
  const decision = applyThresholds(summarizeLedgerRows([row()]));
  assert.equal(decision.status, "blocked");
  assert.ok(decision.blockers.includes("evidence_e2e_complete=false"));
});

test("thresholds produce exactly one recommended product outcome when evidence is complete", () => {
  const rows = [
    ...productionRows("none", 40, { memory_fired: false, should_have_fired: false, false_fire: false, correctness_delta: 0, usefulness_delta: 0, raw_quality_delta: 0, normalized_quality_delta: 0, quality_delta: 0 }),
    ...productionRows("correction-only", 40, { harm_delta: 0, cost_penalty: 0 }),
    ...productionRows("correction+heuristics", 40, { harm_delta: 0, cost_penalty: 0 }),
    ...productionRows("full-ocb", 40, { correctness_delta: 2, usefulness_delta: 2, specificity_delta: 1, raw_quality_delta: 5, normalized_quality_delta: 2, quality_delta: 2, harm_delta: 0, cost_penalty: 0 }),
  ];
  const decision = applyThresholds(summarizeLedgerRows(rows));
  assert.equal(decision.status, "recommended");
  assert.equal(decision.recommended_product_outcome, "A");
});
