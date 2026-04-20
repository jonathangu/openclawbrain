import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunctionFromSourcePath({ sourcePath, startMarker, endMarker, prelude = "" }) {
    const source = readFileSync(sourcePath, "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${sourcePath}`);
    }
    const block = source.slice(start, end).replace(/^export\s+/gmu, "");
    const match = /function\s+([A-Za-z0-9_]+)/u.exec(startMarker);
    if (match === null) {
        throw new Error(`failed to extract function name from ${startMarker}`);
    }
    return new Function(`${prelude}\n${block}\nreturn ${match[1]};`)();
}

test("buildConvergePluginManagerEnv clears profile-pinned state and targets the requested home", () => {
    globalThis.__ocbPath = path;
    const buildConvergePluginManagerEnv = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "cli.js"),
        startMarker: "function buildConvergePluginManagerEnv",
        endMarker: "function runOpenClawBrainConvergePluginStep",
        prelude: `
            const path = globalThis.__ocbPath;
            const process = {
                env: {
                    OPENCLAW_PROFILE: "CormorantAI",
                    OPENCLAW_STATE_DIR: "/tmp/profile-state",
                    OPENCLAW_CONFIG_PATH: "/tmp/profile-state/openclaw.json",
                    KEEP_ME: "yes"
                }
            };
        `
    });
    const env = buildConvergePluginManagerEnv("/tmp/shared-home");
    assert.equal(env.OPENCLAW_HOME, path.resolve("/tmp/shared-home"));
    assert.equal(env.OPENCLAW_STATE_DIR, path.resolve("/tmp/shared-home"));
    assert.equal(env.OPENCLAW_CONFIG_PATH, path.join(path.resolve("/tmp/shared-home"), "openclaw.json"));
    assert.equal(env.OPENCLAW_PROFILE, undefined);
    assert.equal(env.KEEP_ME, "yes");
    delete globalThis.__ocbPath;
});

test("loadTeacherSurfaceFromInput returns a cached teacher surface without reopening the snapshot", () => {
    const sentinel = { sourceKind: "watch_snapshot" };
    globalThis.__teacherSurfaceSentinel = sentinel;
    const loadTeacherSurfaceFromInput = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "index.js"),
        startMarker: "function loadTeacherSurfaceFromInput",
        endMarker: "function summarizeTeacherLoopWatchState",
        prelude: `
            function resolveOperatorTeacherSnapshotPath() {
                throw new Error("should not resolve teacher snapshot path when a cached surface is present");
            }
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
            }
            function loadTeacherSurface() {
                throw new Error("should not load teacher surface when a cached surface is present");
            }
        `
    });
    assert.equal(loadTeacherSurfaceFromInput({ __loadedTeacherSurface: sentinel }), sentinel);
    delete globalThis.__teacherSurfaceSentinel;
});

test("loadOperatorEventExport returns a cached event export without rescanning roots", () => {
    const cachedExport = { sourceKind: "bundle_root", normalizedEventExport: { feedbackEvents: [] } };
    const loadOperatorEventExport = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "index.js"),
        startMarker: "function loadOperatorEventExport",
        endMarker: "function summarizePrincipalItem",
        prelude: `
            const path = globalThis.__ocbPath;
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
            }
            function loadOperatorEventExportFromPath() {
                throw new Error("should not load an explicit event export path when a cached export is present");
            }
            function loadLatestOperatorEventExportFromScanRoots() {
                throw new Error("should not scan event export roots when a cached export is present");
            }
            function resolveOperatorEventExportScanRoots() {
                throw new Error("should not resolve scan roots when a cached export is present");
            }
        `
    });
    assert.equal(loadOperatorEventExport({ __loadedOperatorEventExport: cachedExport }), cachedExport);
});

test("buildAttachStatusCompileInput suppresses serve-time route logging for status probes", () => {
    const buildAttachStatusCompileInput = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "index.js"),
        startMarker: "function buildAttachStatusCompileInput",
        endMarker: "export function describeAttachStatus",
        prelude: `
            const DEFAULT_AGENT_ID = "openclaw-runtime";
            const DEFAULT_ATTACH_STATUS_MESSAGE = "openclaw attach status probe";
            const DEFAULT_ATTACH_STATUS_RUNTIME_HINTS = ["attach", "status", "probe"];
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
            }
        `
    });
    const compileInput = buildAttachStatusCompileInput("/tmp/activation", undefined);
    assert.equal(compileInput._suppressServeLog, true);
    assert.deepEqual(compileInput.runtimeHints, ["attach", "status", "probe"]);
});

test("summarizeActivePackObservability forwards the shared pack cache", () => {
    globalThis.__sharedPackCache = new Map();
    const summarizeActivePackObservability = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "index.js"),
        startMarker: "function summarizeActivePackObservability",
        endMarker: "function summarizeLastPromotion",
        prelude: `
            function buildMissingLabelFlowSummary(detail) { return { detail }; }
            function buildMissingLearningPathSummary(detail) { return { detail }; }
            function summarizePackObservability() { return { labelFlow: { detail: "ok" }, learningPath: { detail: "ok" } }; }
            function toErrorMessage(error) { return error instanceof Error ? error.message : String(error); }
            function loadPackFromActivation(rootDir, slot, options = {}) {
                globalThis.__lastPackCache = options.packCache;
                return { manifest: {}, graph: {}, router: {} };
            }
        `
    });
    summarizeActivePackObservability("/tmp/activation", { packId: "pack-1", activationReady: true }, { packCache: globalThis.__sharedPackCache });
    assert.equal(globalThis.__lastPackCache, globalThis.__sharedPackCache);
    delete globalThis.__sharedPackCache;
    delete globalThis.__lastPackCache;
});

test("describeActivationObservability reuses caller inspection and does not reload the selected active pack twice", () => {
    const sourcePath = path.join(__dirname, "..", "..", "..", "pack-format", "vendor", "index.js");
    const describeActivationObservability = loadFunctionFromSourcePath({
        sourcePath,
        startMarker: "export function describeActivationObservability",
        endMarker: "export function activatePack",
        prelude: `
            let inspectCalls = 0;
            const loadCalls = [];
            globalThis.__inspectCalls = () => inspectCalls;
            globalThis.__loadCalls = () => loadCalls.slice();
            function inspectActivationState() {
                inspectCalls += 1;
                return globalThis.__inspection;
            }
            function buildCompileTargetFromInspection(inspection) {
                return inspection === null ? null : {
                    packId: inspection.packId,
                    builtAt: inspection.packId,
                    eventRange: { start: 0, end: 0, count: 0 }
                };
            }
            function loadPackFromActivation(rootDir, slot, options = {}) {
                loadCalls.push({ slot, packCache: options.packCache });
                return {
                    manifest: { routeArtifact: { slot } },
                    router: null,
                    graph: {}
                };
            }
            function buildServedArtifactProof() { return { served: true }; }
            function describeLearnedRouteFnFreshness() { return { available: true }; }
            function describeRouteArtifactDiff() { return { changed: false }; }
            function describeGraphDynamicsFreshness() { return { runtimePlasticitySource: "pack" }; }
            function describeGraphEvolutionLog() { return { structuralEvolutionSummary: { changed: false, operationsApplied: [], liveBlockCount: 0, prunedBlockCount: 0, prePruneBlockCount: 0, operatorSummary: "ok" }, connectDiagnostics: null, structuralOps: {}, blockCount: 0, strongestBlockId: null }; }
            function emptyInitHandoff() { return { handoffState: "missing", initMode: null, seedStateVisible: false, seedBlockCount: 0 }; }
            function describePackInitHandoff() { return { handoffState: "pg_promoted_pack_authoritative", initMode: "fast_boot_defaults", seedStateVisible: false, seedBlockCount: 0 }; }
            function promotionFreshnessDelta() { return null; }
            function isStrictlyFresherTarget() { return false; }
        `
    });
    globalThis.__inspection = {
        active: { packId: "pack-active" },
        candidate: { packId: "pack-candidate" },
        previous: null,
        pointers: {
            active: { updatedAt: "2026-04-19T00:00:00.000Z" },
            candidate: { updatedAt: "2026-04-19T00:00:01.000Z" },
            previous: null
        },
        promotion: { allowed: true, findings: [] },
        rollback: { allowed: false, findings: [] }
    };
    const packCache = new Map();
    describeActivationObservability("/tmp/activation", "active", {
        inspection: globalThis.__inspection,
        packCache
    });
    assert.equal(globalThis.__inspectCalls(), 0);
    assert.deepEqual(globalThis.__loadCalls().map((entry) => entry.slot), ["active", "candidate"]);
    assert.ok(globalThis.__loadCalls().every((entry) => entry.packCache === packCache));
    delete globalThis.__inspection;
    delete globalThis.__inspectCalls;
    delete globalThis.__loadCalls;
});

test("resolveActivationCompileTarget accepts a preloaded pack without reopening activation payloads", () => {
    const resolveActivationCompileTarget = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "..", "..", "compiler", "vendor", "index.js"),
        startMarker: "export function resolveActivationCompileTarget",
        endMarker: "export function loadPackForActivationCompile",
        prelude: `
            function loadPackFromActivation() {
                throw new Error("should not reload activation payloads when pack is already supplied");
            }
            function describePackCompileTarget(pack) {
                return { packId: pack.manifest.packId };
            }
            function resolveActivationCompileExpectation() { return undefined; }
            function validateRuntimeCompileExpectation() { return []; }
            function validateRuntimeCompileTargetExpectation() { return []; }
            function assertActivationCompileSafety() {}
        `
    });
    const pack = { manifest: { packId: "pack-active" } };
    const target = { packId: "pack-active" };
    const resolved = resolveActivationCompileTarget("/tmp/activation", { pack, target });
    assert.equal(resolved.pack, pack);
    assert.equal(resolved.target, target);
    assert.equal(resolved.slot, "active");
});

test("summary graph materialization falls back to active-pack truth without loading the teacher snapshot", () => {
    const summarizeLatestGraphMaterialization = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "index.js"),
        startMarker: "function summarizeLatestGraphMaterialization",
        endMarker: "function summarizeGraphObservability",
        prelude: `
            function resolveOperatorSurfaceReportDetailLevel() {
                return "summary";
            }
            function loadTeacherSurfaceFromInput() {
                throw new Error("summary graph truth should not load the teacher snapshot");
            }
        `
    });
    const result = summarizeLatestGraphMaterialization({}, { packId: "pack-active" }, null);
    assert.deepEqual(result, {
        known: true,
        packId: "pack-active",
        changed: null,
        connectDiagnostics: null,
        operatorSummary: null,
        detail: "summary status uses the active pack pack-active as the latest cheap local graph truth"
    });
});

test("summary status embeddings skip vector inspection and Ollama model probing", () => {
    const summarizeStatusEmbeddings = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "cli.js"),
        startMarker: "function summarizeStatusEmbeddings",
        endMarker: "function summarizeStatusLocalLlm",
        prelude: `
            let packLoads = 0;
            let probeCalls = 0;
            function loadPackFromActivation() {
                packLoads += 1;
                return null;
            }
            function summarizePackVectorEmbeddingState() {
                throw new Error("summary status should not inspect vectors");
            }
            function runOllamaProbe() {
                probeCalls += 1;
                return { detected: true, detail: "ollama responded" };
            }
            function toErrorMessage(error) {
                return error instanceof Error ? error.message : String(error);
            }
            globalThis.__summaryEmbeddingCounters = () => ({ packLoads, probeCalls });
        `
    });
    const result = summarizeStatusEmbeddings({
        active: { activationReady: true },
        activationRoot: "/tmp/activation"
    }, {
        embedder: {
            provider: "ollama",
            model: "nomic-embed-text"
        },
        embedderBaseUrl: "http://127.0.0.1:11434"
    });
    assert.equal(result.provisionedState, "not_checked");
    assert.equal(result.liveState, "unknown");
    assert.match(result.detail, /skipped active-pack vector inspection and Ollama model probing/);
    assert.deepEqual(globalThis.__summaryEmbeddingCounters(), {
        packLoads: 0,
        probeCalls: 0
    });
    delete globalThis.__summaryEmbeddingCounters;
});

test("summary status local LLM surface skips synchronous Ollama probing", () => {
    const summarizeStatusLocalLlm = loadFunctionFromSourcePath({
        sourcePath: path.join(__dirname, "..", "src", "cli.js"),
        startMarker: "function summarizeStatusLocalLlm",
        endMarker: "function summarizeStatusTeacher",
        prelude: `
            let probeCalls = 0;
            function runOllamaProbe() {
                probeCalls += 1;
                return { detected: true, detail: "ollama responded" };
            }
            globalThis.__summaryLocalLlmProbeCalls = () => probeCalls;
        `
    });
    const result = summarizeStatusLocalLlm({
        teacher: {
            provider: "ollama",
            model: "qwen3.5:14b"
        },
        teacherBaseUrl: "http://127.0.0.1:11434"
    });
    assert.equal(result.detected, null);
    assert.equal(result.enabled, true);
    assert.match(result.detail, /skipped synchronous Ollama probing/);
    assert.equal(globalThis.__summaryLocalLlmProbeCalls(), 0);
    delete globalThis.__summaryLocalLlmProbeCalls;
});
