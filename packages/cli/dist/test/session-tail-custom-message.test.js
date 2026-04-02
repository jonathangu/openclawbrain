import assert from "node:assert/strict";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { OpenClawLocalSessionTail } from "../src/session-tail.js";

function makeTempDir() {
  const dir = path.join(tmpdir(), `ocb-session-tail-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

test("OpenClawLocalSessionTail tolerates custom_message session records", () => {
  const root = makeTempDir();
  const profileRoot = path.join(root, ".openclaw-smoke");
  const sessionsDir = path.join(profileRoot, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });

  const sessionFile = path.join(sessionsDir, "custom-message.jsonl");
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
          },
          id: "custom-1",
          parentId: null,
          timestamp: "2026-04-01T00:00:01.000Z",
        }),
      ].join("\n") + "\n",
      "utf8",
    );
    writeFileSync(
      path.join(sessionsDir, "sessions.json"),
      JSON.stringify(
        {
          session: {
            sessionId: "session-1",
            sessionFile,
            updatedAt: 1,
            chatType: "telegram",
            origin: "test",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const tail = new OpenClawLocalSessionTail({ homeDir: root, emitExistingOnFirstPoll: true });
    const result = tail.pollOnce({ observedAt: "2026-04-01T00:00:02.000Z" });

    assert.deepEqual(result.warnings, []);
    assert.equal(result.changes.length, 1);
    assert.equal(result.changes[0].changeKind, "new_session");
    assert.equal(result.changes[0].rawRecordCount, 2);
    assert.equal(result.changes[0].bridgedEventCount, 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("OpenClawLocalSessionTail stabilizes missing session-file states across polls", () => {
  const root = makeTempDir();
  const profileRoot = path.join(root, ".openclaw-smoke");
  const sessionsDir = path.join(profileRoot, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });

  const missingSessionFile = path.join(sessionsDir, "missing.jsonl");
  try {
    writeFileSync(
      path.join(sessionsDir, "sessions.json"),
      JSON.stringify(
        {
          session: {
            sessionId: "session-1",
            sessionFile: missingSessionFile,
            updatedAt: 1,
            chatType: "telegram",
            origin: "test",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const tail = new OpenClawLocalSessionTail({ homeDir: root, emitExistingOnFirstPoll: true });
    const first = tail.pollOnce({ observedAt: "2026-04-01T00:00:02.000Z" });
    const second = tail.pollOnce({ observedAt: "2026-04-01T00:00:03.000Z" });

    assert.equal(first.changes.length, 1);
    assert.equal(first.changes[0].changeKind, "missing_session_file");
    assert.equal(second.changes.length, 0);
    assert.equal(second.cursor.length, 1);
    assert.deepEqual(second.warnings, []);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("OpenClawLocalSessionTail stabilizes missing session-path states across polls", () => {
  const root = makeTempDir();
  const profileRoot = path.join(root, ".openclaw-smoke");
  const sessionsDir = path.join(profileRoot, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });

  try {
    writeFileSync(
      path.join(sessionsDir, "sessions.json"),
      JSON.stringify(
        {
          session: {
            sessionId: "session-1",
            updatedAt: 1,
            chatType: "telegram",
            origin: "test",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const tail = new OpenClawLocalSessionTail({ homeDir: root, emitExistingOnFirstPoll: true });
    const first = tail.pollOnce({ observedAt: "2026-04-01T00:00:02.000Z" });
    const second = tail.pollOnce({ observedAt: "2026-04-01T00:00:03.000Z" });

    assert.equal(first.changes.length, 1);
    assert.equal(first.changes[0].changeKind, "missing_session_path");
    assert.equal(second.changes.length, 0);
    assert.equal(second.cursor.length, 1);
    assert.deepEqual(second.warnings, []);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("OpenClawLocalSessionTail prunes stale cursor entries when sessions disappear from the index", () => {
  const root = makeTempDir();
  const profileRoot = path.join(root, ".openclaw-smoke");
  const sessionsDir = path.join(profileRoot, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });

  const sessionFile = path.join(sessionsDir, "session.jsonl");
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
      ].join("\n") + "\n",
      "utf8",
    );
    writeFileSync(
      path.join(sessionsDir, "sessions.json"),
      JSON.stringify(
        {
          session: {
            sessionId: "session-1",
            sessionFile,
            updatedAt: 1,
            chatType: "telegram",
            origin: "test",
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const tail = new OpenClawLocalSessionTail({ homeDir: root, emitExistingOnFirstPoll: true });
    const first = tail.pollOnce({ observedAt: "2026-04-01T00:00:02.000Z" });
    assert.equal(first.cursor.length, 1);

    writeFileSync(path.join(sessionsDir, "sessions.json"), JSON.stringify({}, null, 2), "utf8");
    const second = tail.pollOnce({ observedAt: "2026-04-01T00:00:03.000Z" });

    assert.equal(second.changes.length, 0);
    assert.equal(second.cursor.length, 0);
    assert.deepEqual(second.warnings, ["session tail pruned 1 stale cursor entry"]);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
