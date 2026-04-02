import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunction({ file, startMarker, endMarker, prelude = "" }) {
  const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  if (start === -1 || end === -1) {
    throw new Error(`failed to locate ${startMarker} in ${file}`);
  }
  const block = source.slice(start, end).replace(/^export\s+/gmu, "");
  const match = /function\s+([A-Za-z0-9_]+)/u.exec(startMarker);
  if (match === null) {
    throw new Error(`failed to extract function name from ${startMarker}`);
  }
  return new Function(`${prelude}\n${block}\nreturn ${match[1]};`)();
}

test("watch replay guard can skip stored replay without touching bundle roots", async () => {
  globalThis.__ocbReplayHarness = {
    listCalls: 0,
    loadCalls: [],
    roots: ["/tmp/export-a"],
    bundles: {
      "/tmp/export-a": {
        manifest: { exportedAt: "2026-04-01T00:00:00.000Z" },
        normalizedEventExport: {
          range: { start: 1, end: 1, count: 1 },
          provenance: { exportDigest: "sha256-a" },
        },
      },
    },
  };
  const replayWatchScanRootIntoTeacherLoop = loadFunction({
    file: "cli.js",
    startMarker: "async function replayWatchScanRootIntoTeacherLoop",
    endMarker: "function exportLocalSessionTailChangesToScanRoot",
    prelude: `
      const __harness = globalThis.__ocbReplayHarness;
      function listWatchRuntimeEventExportBundleRoots(scanRoot) {
        __harness.listCalls += 1;
        return __harness.roots;
      }
      function loadRuntimeEventExportBundle(rootDir) {
        __harness.loadCalls.push(rootDir);
        return __harness.bundles[rootDir];
      }
    `,
  });
  const teacherLoop = {
    enqueueNormalizedEventExport() {
      throw new Error("skip replay should not enqueue exports");
    },
    async flush() {
      throw new Error("skip replay should not flush");
    },
  };

  const result = await replayWatchScanRootIntoTeacherLoop(teacherLoop, "/tmp/scan-root", { skip: true });

  assert.deepEqual(result, {
    replayedBundleCount: 0,
    replayedEventCount: 0,
  });
  assert.equal(globalThis.__ocbReplayHarness.listCalls, 0);
  assert.deepEqual(globalThis.__ocbReplayHarness.loadCalls, []);
  delete globalThis.__ocbReplayHarness;
});

test("teacher autodetect prefers larger compatible qwen lanes when available", () => {
  const selectCompatibleLocalTeacherModel = loadFunction({
    file: "cli.js",
    startMarker: "function selectCompatibleLocalTeacherModel",
    endMarker: "function detectInstallTeacherDefaults",
    prelude: `
      const INSTALL_COMPATIBLE_LOCAL_TEACHER_MODEL_PREFIXES = [
        "unsloth-qwen3.5-27b:q4_k_m",
        "unsloth-qwen3.5-27b",
        "qwen3.5:32b",
        "qwen3.5:27b",
        "qwen3.5:14b",
        "qwen3.5:9b",
        "qwen3.5:8b",
        "qwen3:8b",
        "qwen2.5:7b",
      ];
    `,
  });

  const selected = selectCompatibleLocalTeacherModel([
    "qwen3.5:9b",
    "unsloth-qwen3.5-27b:q4_k_m",
    "qwen3.5:14b",
  ]);

  assert.equal(selected, "unsloth-qwen3.5-27b:q4_k_m");
});

test("advanceAlwaysOnLearningRuntime does not rebuild the runtime graph on empty selected slices", () => {
  globalThis.__ocbAdvanceHarness = {
    buildCalls: 0,
    learnedEventExport: null,
    initialState: {
      runtimeOwner: "openclaw",
      hotPathLearning: false,
      attachBlocksOnFullReplay: false,
      cursor: { watermark: 1 },
      pending: { live: [], backfill: [] },
      learnedEventExport: null,
      runtimeGraph: { packId: "existing-graph" },
      runtimePlasticity: { strongestBlockId: "blk-1" },
      learnedGraph: null,
      structuralController: { structuralOps: [] },
      sparseFeedback: { mode: "default" },
      lastMaterializedAt: "2026-04-01T00:00:00.000Z",
      materializationCount: 0,
    },
  };
  const advanceAlwaysOnLearningRuntime = loadFunction({
    file: "local-learner.js",
    startMarker: "export function advanceAlwaysOnLearningRuntime",
    endMarker: "export function drainAlwaysOnLearningRuntime",
    prelude: `
      const __harness = globalThis.__ocbAdvanceHarness;
      function normalizeAlwaysOnLearningCadence(value) { return value ?? { mode: "default" }; }
      function cloneAlwaysOnLearningRuntimeState(value) { return JSON.parse(JSON.stringify(value)); }
      function createAlwaysOnLearningRuntimeState() { return JSON.parse(JSON.stringify(__harness.initialState)); }
      function resolveAlwaysOnLearningStructuralController(current) { return current ?? { structuralOps: [] }; }
      function normalizeSparseFeedbackPolicy(value) { return value ?? { mode: "default" }; }
      function buildNormalizedEventExportBridge() { return { cursor: { watermark: 1 }, slices: [] }; }
      function mergePendingSlices(current) { return current; }
      function selectScheduledSlices() { return { selected: [], remaining: { live: [], backfill: [] }, selectedBucket: "none" }; }
      function mergeNormalizedEventExports(current) { return current; }
      function buildRuntimeGraphSnapshot() {
        __harness.buildCalls += 1;
        return { graph: { packId: "rebuilt-graph" }, plasticity: { strongestBlockId: "blk-rebuilt" } };
      }
      function createSparseFeedbackRuntimeDiagnostics(policy) { return policy; }
      function evaluateSparseFeedback(_feedbackEvents, _observedAt, policy) { return { diagnostics: policy }; }
      function buildAlwaysOnLearningMaterializationJob() { throw new Error("no-op cycle should not materialize"); }
      function summarizePrincipalBacklog() { return { principalCount: 0, pendingEventCount: 0, checkpoints: [], oldestUnlearnedEvent: null, newestPendingEvent: null }; }
      function cloneNormalizedEventExportSlice(value) { return value; }
    `,
  });

  const result = advanceAlwaysOnLearningRuntime({
    packLabel: "watch-cli",
    workspace: {
      workspaceId: "watch-cli",
      snapshotId: "watch-cli@test",
      capturedAt: "2026-04-01T00:00:00.000Z",
      rootDir: "/tmp",
      revision: "watch-cli-v2",
    },
    interactionEvents: [{ eventId: "evt-1" }],
    feedbackEvents: [],
    learnedRouting: false,
    state: globalThis.__ocbAdvanceHarness.initialState,
  });

  assert.equal(globalThis.__ocbAdvanceHarness.buildCalls, 0);
  assert.equal(result.materialization, null);
  assert.deepEqual(result.state.runtimeGraph, { packId: "existing-graph" });
  assert.deepEqual(result.state.runtimePlasticity, { strongestBlockId: "blk-1" });
  delete globalThis.__ocbAdvanceHarness;
});
