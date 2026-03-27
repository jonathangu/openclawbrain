import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadSource() {
  return readFileSync(path.join(__dirname, "..", "src", "local-session-passive-learning.js"), "utf8");
}

function loadSortSessionRecordsDeterministically() {
  const source = loadSource();
  const helperStart = source.indexOf("function sortSessionRecordsDeterministically");
  const helperEnd = source.indexOf("function deriveChannel");

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  assert.ok(helperEnd > helperStart);

  const helperSource = source.slice(helperStart, helperEnd);
  return new Function(`${helperSource}; return sortSessionRecordsDeterministically;`)();
}

test("passive-learning session export sorts records deterministically before extraction", () => {
  const sortSessionRecordsDeterministically = loadSortSessionRecordsDeterministically();
  const ordered = sortSessionRecordsDeterministically([
    {
      type: "message",
      id: "later",
      parentId: null,
      timestamp: "2026-03-25T12:01:00.000Z",
    },
    {
      type: "message",
      id: "reply",
      parentId: "root",
      timestamp: "2026-03-25T12:00:00.000Z",
    },
    {
      type: "custom",
      customType: "audit",
      data: {},
      id: "audit",
      parentId: null,
      timestamp: "2026-03-25T11:59:00.000Z",
    },
    {
      type: "message",
      id: "root",
      parentId: null,
      timestamp: "2026-03-25T12:00:00.000Z",
    },
  ]);

  assert.deepEqual(
    ordered.map((record) => record.id),
    ["audit", "root", "reply", "later"],
  );
});

test("passive-learning session export wires deterministic record ordering into the extraction loop", () => {
  const source = loadSource();

  assert.match(
    source,
    /for \(const record of sortSessionRecordsDeterministically\(input\.records\)\)/,
  );
});
