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
