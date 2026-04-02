import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  buildFrozenRecordedSessionFixtureManifest,
  buildFrozenRecordedSessionFixtureManifestSchema,
  canonicalJson,
  validateFrozenRecordedSessionFixtureManifest,
  type FrozenRecordedSessionFixtureManifestV1,
} from "../scripts/eval/frozen-fixture-manifest.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const manifestPath = path.join(
  repoRoot,
  "artifacts",
  "fixtures",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);
const schemaPath = path.join(
  repoRoot,
  "artifacts",
  "fixtures",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.schema.json",
);

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

describe("frozen recorded-session fixture manifest scaffold", () => {
  it("matches the generator output and preserves the canonical 20-slot family split", () => {
    const manifest = readJsonFile<FrozenRecordedSessionFixtureManifestV1>(manifestPath);
    const generated = buildFrozenRecordedSessionFixtureManifest();

    expect(canonicalJson(manifest)).toBe(canonicalJson(generated));
    expect(manifest.contract).toBe("frozen_recorded_session_fixture_manifest.v1");
    expect(manifest.manifestId).toBe("canonical-frozen-20");
    expect(manifest.status).toBe("scaffold_only");
    expect(manifest.fixtureContract).toBe("recorded_session_replay_fixture.v1");
    expect(manifest.traceManifest.setId).toBe("canonical-frozen-20");
    expect(manifest.entries).toHaveLength(20);
    expect(manifest.traceFamilyCounts).toEqual({
      direct_answer: 5,
      plan_execution: 5,
      retrieval_memory_heavy: 5,
      correction_follow_up_heavy: 5,
    });
    expect(manifest.materialization.fixtureFilesCheckedIn).toBe(0);
    expect(manifest.materialization.pendingFixtureCount).toBe(20);
    expect(manifest.entries.every((entry) => entry.fixtureHash === null)).toBe(true);
    expect(manifest.entries.every((entry) => entry.fixtureHashStatus === "pending_materialization")).toBe(true);
  });

  it("keeps the checked-in schema and validation report in sync with the scaffold", () => {
    const schema = readJsonFile<Record<string, unknown>>(schemaPath);
    const generatedSchema = buildFrozenRecordedSessionFixtureManifestSchema();
    const validation = validateFrozenRecordedSessionFixtureManifest(manifestPath);

    expect(canonicalJson(schema)).toBe(canonicalJson(generatedSchema));
    expect(validation.ok).toBe(true);
    expect(validation.errors).toEqual([]);
    expect(validation.entryCount).toBe(20);
  });
});
