import { describe, it, expect, vi, afterEach } from "vitest";
import { DatabaseSync } from "node:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LabelHarvester } from "../../src/brain-runtime/harvester-extension.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";

const tempDirs: string[] = [];

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "brain-harvester-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  const store = new BrainStore(db);
  const log = { info: vi.fn(), warn: vi.fn() };
  const harvester = new LabelHarvester(store, log);
  return { store, harvester };
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("LabelHarvester", () => {
  describe("detectLabel (human patterns)", () => {
    it("detects negative human correction", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "No, that's not right. Use the other approach.");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeLessThan(0);
    });

    it("detects 'don't use' pattern", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "Don't use that tool, try something else");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeLessThan(0);
    });

    it("detects 'instead use' pattern", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "Instead use gh pr create for this");
      expect(result).not.toBeNull();
      expect(result!.value).toBeLessThan(0);
    });

    it("detects positive human feedback", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "Perfect, that's exactly right!");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeGreaterThan(0);
    });

    it("returns null for neutral messages", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", "Can you read the file src/index.ts?")).toBeNull();
    });
  });

  describe("detectLabel (self patterns)", () => {
    it("detects tool failure", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("tool", "Error: ENOENT: no such file or directory");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("self");
      expect(result!.value).toBeLessThan(0);
    });

    it("detects tool success", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("tool", "Successfully deployed to production");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("self");
      expect(result!.value).toBeGreaterThan(0);
    });

    it("detects test pass", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("assistant", "All 247 tests passed");
      expect(result).not.toBeNull();
      expect(result!.value).toBeGreaterThan(0);
    });
  });
});
