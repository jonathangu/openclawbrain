import test from "node:test";
import assert from "node:assert/strict";
import { parseLedgerRow } from "../src/ledger.ts";
import { row } from "./helpers.ts";

test("quality, harm, and cost are counted exactly once", () => {
  const parsed = parseLedgerRow(row({ correctness_delta: 2, usefulness_delta: 2, specificity_delta: 1, raw_quality_delta: 5, normalized_quality_delta: 2, quality_delta: 2, harm_delta: 1, cost_penalty: 0.25 }));
  assert.equal(parsed.activation_utility, 0.75);
  assert.equal(parsed.net_task_utility, 0.75);
});
