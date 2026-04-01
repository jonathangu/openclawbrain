import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  validateRecordedSessionReplayProofBundle,
  writeRecordedSessionReplayProofBundle,
  type RecordedSessionTraceV1,
} from "../packages/cli/dist/src/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const frozenRoot = path.join(repoRoot, "evals", "recorded-session-replay", "canonical-frozen-20");
const manifestPath = path.join(frozenRoot, "manifest.json");

type FrozenTraceManifest = {
  contract: string;
  setId: string;
  traceContract: string;
  traceCount: number;
  categoryCounts: Record<string, number>;
  sourceSummary: Record<string, number>;
  realTraceCoverage: {
    availableCount: number;
    missingCount: number;
    summary: string;
  };
  entries: Array<{
    slotId: string;
    category: string;
    path: string;
    status: string;
    realTraceSourceAvailable: boolean;
  }>;
};

function loadJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

describe("canonical frozen recorded-session trace set", () => {
  it("keeps the canonical 20-slot manifest contract", () => {
    const manifest = loadJson<FrozenTraceManifest>(manifestPath);

    expect(manifest.contract).toBe("canonical_recorded_session_trace_set_manifest.v1");
    expect(manifest.setId).toBe("canonical-frozen-20");
    expect(manifest.traceContract).toBe("recorded_session_trace.v1");
    expect(manifest.traceCount).toBe(20);
    expect(manifest.entries).toHaveLength(20);
    expect(manifest.categoryCounts).toEqual({
      direct_answer: 5,
      plan_execution: 5,
      retrieval_memory_heavy: 5,
      correction_follow_up_heavy: 5,
    });
    expect(manifest.sourceSummary).toEqual({
      repo_published_fixture: 2,
      repo_test_fixture_static: 2,
      repo_test_fixture_normalized: 3,
      derived_replayable_equivalent: 13,
    });
    expect(manifest.realTraceCoverage.availableCount).toBe(0);
    expect(manifest.realTraceCoverage.missingCount).toBe(20);

    const slotIds = new Set<string>();
    const tracePaths = new Set<string>();
    for (const entry of manifest.entries) {
      expect(entry.status).toBe("frozen_replayable_equivalent");
      expect(entry.realTraceSourceAvailable).toBe(false);
      expect(slotIds.has(entry.slotId)).toBe(false);
      expect(tracePaths.has(entry.path)).toBe(false);
      slotIds.add(entry.slotId);
      tracePaths.add(entry.path);
    }
  });

  it(
    "round-trips every frozen trace through the proof-bundle writer",
    () => {
      const manifest = loadJson<FrozenTraceManifest>(manifestPath);
      const tempRoot = mkdtempSync(path.join(os.tmpdir(), "ocb-canonical-frozen-20-"));

      try {
        for (const entry of manifest.entries) {
          const tracePath = path.join(frozenRoot, entry.path);
          const trace = loadJson<RecordedSessionTraceV1>(tracePath);
          const bundleRoot = path.join(tempRoot, entry.slotId);
          const descriptor = writeRecordedSessionReplayProofBundle({
            rootDir: bundleRoot,
            trace,
            scratchRootDir: tempRoot,
          });
          const validation = validateRecordedSessionReplayProofBundle(bundleRoot);

          expect(trace.contract).toBe("recorded_session_trace.v1");
          expect(descriptor.bundle.traceId).toBe(trace.traceId);
          if (!validation.ok) {
            throw new Error(`${entry.slotId} failed proof validation: ${validation.errors.join(" | ")}`);
          }
        }
      } finally {
        rmSync(tempRoot, { recursive: true, force: true });
      }
    },
    120_000,
  );
});
