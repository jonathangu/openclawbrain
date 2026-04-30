import test from "node:test";
import assert from "node:assert/strict";
import { parseLedgerRow } from "../src/ledger.ts";
import { row } from "./helpers.ts";

test("valid ledger rows derive V5 utility fields", () => {
  const parsed = parseLedgerRow(row({ cost_penalty: 0.25 }));
  assert.equal(parsed.activation_utility, 0.75);
  assert.equal(parsed.net_task_utility, 0.75);
});

test("enum violations fail using V5 enums", () => {
  assert.throws(() => parseLedgerRow(row({ slice: "memory_helpful" as never })), /direct-answer/);
  assert.throws(() => parseLedgerRow(row({ backend: "full_ocb" as never })), /full-ocb/);
});

test("derived utility fields are rejected in ledger input", () => {
  assert.throws(() => parseLedgerRow({ ...row(), activation_utility: 99 }), /derived/);
  assert.throws(() => parseLedgerRow({ ...row(), net_task_utility: 99 }), /derived/);
});

test("quality fields must match raw and normalized components", () => {
  assert.throws(() => parseLedgerRow(row({ raw_quality_delta: 0 })), /raw_quality_delta/);
  assert.throws(() => parseLedgerRow(row({ normalized_quality_delta: 2 })), /normalized_quality_delta/);
});

test("smoke or non-real rows cannot count as product evidence", () => {
  assert.throws(() => parseLedgerRow(row({ counts_as_product_evidence: true })), /smoke rows/);
  assert.throws(() => parseLedgerRow(row({ mode: "production", counts_as_product_evidence: true })), /only real provenance/);
});
