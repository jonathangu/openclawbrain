import test from "node:test";
import assert from "node:assert/strict";
import { parseLedgerRow } from "../src/ledger.ts";
import { row } from "./helpers.ts";

test("memory fires: net task utility equals activation utility", () => {
  const parsed = parseLedgerRow(row({ memory_fired: true, should_have_fired: true, cost_penalty: 0.5 }));
  assert.equal(parsed.activation_utility, 0.5);
  assert.equal(parsed.net_task_utility, 0.5);
});

test("missed useful memory: net utility is abstention regret penalty", () => {
  const parsed = parseLedgerRow(row({ memory_fired: false, should_have_fired: true, abstention_regret: 2, false_fire: false }));
  assert.equal(parsed.abstention_regret_penalty, 1);
  assert.equal(parsed.net_task_utility, -1);
});

test("correct abstention has zero net task utility", () => {
  const parsed = parseLedgerRow(row({ memory_fired: false, should_have_fired: false, false_fire: false }));
  assert.equal(parsed.net_task_utility, 0);
});
