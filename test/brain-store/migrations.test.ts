import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("runBrainMigrations usefulness schema", () => {
  it("creates the shadow usefulness table and indexes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "openclawbrain-migrations-"));
    tempDirs.push(tempDir);
    const db = new DatabaseSync(join(tempDir, "brain.db"));
    db.exec("PRAGMA foreign_keys = ON");

    runBrainMigrations(db);

    const tableInfo = db.prepare(`PRAGMA table_info(brain_context_usefulness)`).all() as Array<{ name?: string }>;
    expect(tableInfo.map((column) => column.name)).toEqual(expect.arrayContaining([
      "id",
      "observation_id",
      "episode_id",
      "trace_id",
      "conversation_id",
      "binding_mode",
      "follow_up_text",
      "tool_results_json",
      "signal_json",
      "final_score",
      "confidence",
      "verdict",
      "reason",
      "created_at",
      "updated_at",
      "evaluated_at",
    ]));

    const indexNames = db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'brain_context_usefulness_%'`)
      .all() as Array<{ name: string }>;
    expect(indexNames.map((row) => row.name)).toEqual(expect.arrayContaining([
      "brain_context_usefulness_observation_idx",
      "brain_context_usefulness_trace_idx",
      "brain_context_usefulness_episode_idx",
      "brain_context_usefulness_created_idx",
    ]));
  });
});
