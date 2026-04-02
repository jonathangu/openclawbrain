import assert from "node:assert/strict";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { readOpenClawSessionFile } from "../src/session-store.js";

function makeTempDir() {
  const dir = path.join(tmpdir(), `ocb-session-store-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

test("readOpenClawSessionFile accepts custom_message session records", () => {
  const dir = makeTempDir();
  const sessionFile = path.join(dir, "session.jsonl");
  try {
    writeFileSync(
      sessionFile,
      [
        JSON.stringify({
          type: "session",
          version: 1,
          id: "session-1",
          timestamp: "2026-04-01T00:00:00.000Z",
          cwd: "/tmp",
        }),
        JSON.stringify({
          type: "custom_message",
          customType: "openclaw.sessions_yield",
          content: "Swarm launched",
          display: false,
          details: {
            source: "sessions_yield",
            message: "Swarm launched",
          },
          id: "custom-1",
          parentId: null,
          timestamp: "2026-04-01T00:00:01.000Z",
        }),
      ].join("\n") + "\n",
      "utf8",
    );

    const records = readOpenClawSessionFile(sessionFile);
    assert.equal(records.length, 2);
    assert.deepEqual(records[1], {
      type: "custom",
      customType: "openclaw.sessions_yield",
      data: {
        content: "Swarm launched",
        display: false,
        details: {
          source: "sessions_yield",
          message: "Swarm launched",
        },
      },
      id: "custom-1",
      parentId: null,
      timestamp: "2026-04-01T00:00:01.000Z",
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("readOpenClawSessionFile accepts string-valued message.content payloads", () => {
  const dir = makeTempDir();
  const sessionFile = path.join(dir, "session.jsonl");
  try {
    writeFileSync(
      sessionFile,
      [
        JSON.stringify({
          type: "session",
          version: 1,
          id: "session-1",
          timestamp: "2026-04-01T00:00:00.000Z",
          cwd: "/tmp",
        }),
        JSON.stringify({
          type: "message",
          id: "msg-1",
          parentId: null,
          timestamp: "2026-04-01T00:00:01.000Z",
          message: {
            role: "assistant",
            content: "Compacted answer body",
            timestamp: 1711929601000,
          },
        }),
      ].join("\n") + "\n",
      "utf8",
    );

    const records = readOpenClawSessionFile(sessionFile);
    assert.equal(records.length, 2);
    assert.deepEqual(records[1], {
      type: "message",
      id: "msg-1",
      parentId: null,
      timestamp: "2026-04-01T00:00:01.000Z",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Compacted answer body" }],
        timestamp: 1711929601000,
      },
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("readOpenClawSessionFile accepts compaction records", () => {
  const dir = makeTempDir();
  const sessionFile = path.join(dir, "session.jsonl");
  try {
    writeFileSync(
      sessionFile,
      [
        JSON.stringify({
          type: "session",
          version: 1,
          id: "session-1",
          timestamp: "2026-04-01T00:00:00.000Z",
          cwd: "/tmp",
        }),
        JSON.stringify({
          type: "compaction",
          id: "cmp-1",
          parentId: "msg-1",
          timestamp: "2026-04-01T00:00:02.000Z",
          summary: "Compacted prior context",
          firstKeptEntryId: "msg-2",
          tokensBefore: 2048,
          details: {
            readFiles: ["/tmp/file.md"],
          },
          fromHook: true,
        }),
      ].join("\n") + "\n",
      "utf8",
    );

    const records = readOpenClawSessionFile(sessionFile);
    assert.equal(records.length, 2);
    assert.deepEqual(records[1], {
      type: "custom",
      customType: "openclaw.compaction",
      data: {
        summary: "Compacted prior context",
        firstKeptEntryId: "msg-2",
        tokensBefore: 2048,
        details: {
          readFiles: ["/tmp/file.md"],
        },
        fromHook: true,
      },
      id: "cmp-1",
      parentId: "msg-1",
      timestamp: "2026-04-01T00:00:02.000Z",
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
