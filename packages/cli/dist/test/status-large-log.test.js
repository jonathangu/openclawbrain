import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readBoundedJsonlTail } from "../src/bounded-jsonl-reader.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cliSource = readFileSync(path.join(__dirname, "..", "src", "cli.js"), "utf8");
const indexSource = readFileSync(path.join(__dirname, "..", "src", "index.js"), "utf8");
const LEARNING_SPINE_LOG_LAYOUT = { dir: path.join("logs", "learning-spine") };

function resolveLearningSpineLogPath(activationRoot, stream) {
    return path.join(activationRoot, LEARNING_SPINE_LOG_LAYOUT.dir, `${stream}.jsonl`);
}

function loadFunction(source, startMarker, endMarker, prelude = "") {
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker}`);
    }
    const block = source.slice(start, end).replace(/^export\s+/gmu, "");
    return new Function(`${prelude}\n${block}\nreturn ${startMarker.split(" ")[1]};`)();
}

globalThis.__testResolveLearningSpineLogPath = resolveLearningSpineLogPath;
globalThis.__testReadBoundedJsonlTail = readBoundedJsonlTail;
globalThis.__testLoadOrInitBaseline = () => ({ baseline: true });

const resolveServeTimeLearningRuntimeInput = loadFunction(
    cliSource,
    "function resolveServeTimeLearningRuntimeInput",
    "function resolveActivationInspectionPackId",
    `const resolveLearningSpineLogPath = globalThis.__testResolveLearningSpineLogPath;
const readBoundedJsonlTail = globalThis.__testReadBoundedJsonlTail;
const loadOrInitBaseline = globalThis.__testLoadOrInitBaseline;`
);
const readBoundedLearningSpineLogEntries = loadFunction(
    indexSource,
    "function readBoundedLearningSpineLogEntries",
    "function matchesActiveRouteFnLog",
    `const resolveLearningSpineLogPath = globalThis.__testResolveLearningSpineLogPath;
const readBoundedJsonlTail = globalThis.__testReadBoundedJsonlTail;`
);
globalThis.__testReadBoundedLearningSpineLogEntries = readBoundedLearningSpineLogEntries;
const summarizeCurrentProfileLastLearningUpdateAt = loadFunction(
    indexSource,
    "function summarizeCurrentProfileLastLearningUpdateAt",
    "function didCurrentProfileFirstExportOccur",
    `const readBoundedLearningSpineLogEntries = globalThis.__testReadBoundedLearningSpineLogEntries;`
);

function makeActivationRoot() {
    const root = mkdtempSync(path.join(os.tmpdir(), "ocb-status-large-log-"));
    mkdirSync(path.join(root, LEARNING_SPINE_LOG_LAYOUT.dir), { recursive: true });
    return root;
}

test("resolveServeTimeLearningRuntimeInput uses bounded reads for oversized decision logs", () => {
    const activationRoot = makeActivationRoot();
    const logPath = resolveLearningSpineLogPath(activationRoot, "serveTimeRouteDecisions");
    try {
        const lines = [];
        for (let i = 0; i < 1200; i++) {
            lines.push(JSON.stringify({
                recordId: `decision-${i}`,
                recordedAt: `2026-04-01T00:${String(i % 60).padStart(2, "0")}:00.000Z`,
                activePackId: "pack-live",
                usedLearnedRouteFn: true,
                actionScore: i,
                padding: "x".repeat(4096)
            }));
        }
        writeFileSync(logPath, lines.join("\n") + "\n", "utf8");
        const result = resolveServeTimeLearningRuntimeInput(activationRoot);
        assert.equal(result.pgVersion, "v2");
        assert.ok(result.decisionLogCount > 0);
        assert.ok(result.decisionLogCount <= 512);
        assert.equal(result.serveTimeDecisions.at(-1)?.recordId, "decision-1199");
        assert.match(result.fallbackReason ?? "", /serve_time_decision_log_/);
    } finally {
        rmSync(activationRoot, { recursive: true, force: true });
    }
});

test("summarizeCurrentProfileLastLearningUpdateAt returns the latest visible bounded update", () => {
    const activationRoot = makeActivationRoot();
    const logPath = resolveLearningSpineLogPath(activationRoot, "pgRouteUpdates");
    try {
        const lines = [];
        for (let i = 0; i < 1200; i++) {
            lines.push(JSON.stringify({
                recordId: `update-${i}`,
                recordedAt: `2026-04-01T01:${String(i % 60).padStart(2, "0")}:00.000Z`,
                nextPackId: "pack-live",
                nextRouterChecksum: `router-${i}`,
                padding: "y".repeat(4096)
            }));
        }
        writeFileSync(logPath, lines.join("\n") + "\n", "utf8");
        const recordedAt = summarizeCurrentProfileLastLearningUpdateAt(activationRoot, { lastMaterializedAt: null }, { lastRunAt: null });
        assert.equal(recordedAt, "2026-04-01T01:59:00.000Z");
    } finally {
        rmSync(activationRoot, { recursive: true, force: true });
    }
});
