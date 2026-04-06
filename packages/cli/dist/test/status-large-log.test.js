import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readBoundedJsonlTail } from "../src/bounded-jsonl-reader.js";
import { createServeTimeDecisionMatcher } from "../src/teacher-decision-match.js";

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
globalThis.__testLoadOrInitBaselineCalls = [];
globalThis.__testLoadOrInitBaseline = (activationRoot) => {
    globalThis.__testLoadOrInitBaselineCalls.push(activationRoot);
    return { baseline: true, activationRoot };
};
globalThis.__testReadFileSync = readFileSync;

const resolveServeTimeLearningRuntimeInput = loadFunction(
    cliSource,
    "function resolveServeTimeLearningRuntimeInput",
    "function resolveActivationInspectionPackId",
    `const resolveLearningSpineLogPath = globalThis.__testResolveLearningSpineLogPath;
const readBoundedJsonlTail = globalThis.__testReadBoundedJsonlTail;
const readFileSync = globalThis.__testReadFileSync;
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

function makeInteraction(overrides = {}) {
    return {
        eventId: "evt-interaction",
        sessionId: "sess-1",
        channel: "cli",
        createdAt: "2026-04-01T00:00:00.000Z",
        ...overrides,
    };
}

function makeNormalizedEventExport(interactionOverrides = []) {
    const interactionEvents = interactionOverrides.map((overrides) => makeInteraction(overrides));
    return {
        interactionEvents,
        feedbackEvents: interactionEvents.map((interaction, index) => ({
            eventId: `feedback-${index}`,
            relatedInteractionId: interaction.eventId,
        })),
    };
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

test("resolveServeTimeLearningRuntimeInput defaults cold-start installs to the learned v2 prior", () => {
    const activationRoot = makeActivationRoot();
    try {
        globalThis.__testLoadOrInitBaselineCalls.length = 0;
        const result = resolveServeTimeLearningRuntimeInput(activationRoot);
        assert.equal(result.pgVersion, "v2");
        assert.equal(result.decisionLogCount, 0);
        assert.deepEqual(result.baselineState, { baseline: true, activationRoot });
        assert.deepEqual(globalThis.__testLoadOrInitBaselineCalls, [activationRoot]);
    }
    finally {
        rmSync(activationRoot, { recursive: true, force: true });
    }
});

test("resolveServeTimeLearningRuntimeInput preserves upgrade baselines on top of the learned v2 prior", () => {
    const activationRoot = makeActivationRoot();
    const logPath = resolveLearningSpineLogPath(activationRoot, "serveTimeRouteDecisions");
    try {
        globalThis.__testLoadOrInitBaselineCalls.length = 0;
        writeFileSync(logPath, JSON.stringify({
            recordId: "decision-upgrade-1",
            recordedAt: "2026-04-01T00:00:00.000Z",
            turnCompileEventId: "evt-upgrade-1",
            sessionId: "sess-1",
            channel: "cli",
            userMessage: "upgrade decision"
        }) + "\n", "utf8");
        const result = resolveServeTimeLearningRuntimeInput(activationRoot);
        assert.equal(result.pgVersion, "v2");
        assert.equal(result.decisionLogCount, 1);
        assert.deepEqual(result.baselineState, { baseline: true, activationRoot });
        assert.deepEqual(globalThis.__testLoadOrInitBaselineCalls, [activationRoot]);
    }
    finally {
        rmSync(activationRoot, { recursive: true, force: true });
    }
});

test("resolveServeTimeLearningRuntimeInput recovers exact historical decisions outside the bounded tail", () => {
    const activationRoot = makeActivationRoot();
    const logPath = resolveLearningSpineLogPath(activationRoot, "serveTimeRouteDecisions");
    try {
        const lines = [
            JSON.stringify({
                recordId: "decision-historical-exact",
                turnCompileEventId: "evt-historical-exact",
                recordedAt: "2026-04-01T00:00:00.000Z",
                turnCreatedAt: "2026-04-01T00:00:00.000Z",
                sessionId: "sess-1",
                channel: "cli",
                userMessage: "historical exact decision",
            }),
        ];
        for (let i = 0; i < 1200; i++) {
            lines.push(JSON.stringify({
                recordId: `decision-${i}`,
                recordedAt: `2026-04-01T01:${String(i % 60).padStart(2, "0")}:00.000Z`,
                turnCreatedAt: `2026-04-01T01:${String(i % 60).padStart(2, "0")}:00.000Z`,
                turnCompileEventId: `evt-${i}`,
                sessionId: "sess-1",
                channel: "cli",
                userMessage: `decision ${i}`,
                padding: "x".repeat(4096)
            }));
        }
        writeFileSync(logPath, lines.join("\n") + "\n", "utf8");
        const result = resolveServeTimeLearningRuntimeInput(activationRoot, makeNormalizedEventExport([
            { serveDecisionRecordId: "decision-historical-exact" }
        ]));
        assert.equal(result.pgVersion, "v2");
        assert.equal(result.serveTimeDecisions.some((decision) => decision.recordId === "decision-historical-exact"), true);
        assert.equal(result.serveTimeDecisions.at(-1)?.recordId, "decision-1199");
        assert.ok(result.decisionLogCount > 512);
        assert.match(result.fallbackReason ?? "", /serve_time_decision_log_/);
    } finally {
        rmSync(activationRoot, { recursive: true, force: true });
    }
});

test("historical recovery keeps exact selection matches ahead of nearby tail fallback candidates", () => {
    const activationRoot = makeActivationRoot();
    const logPath = resolveLearningSpineLogPath(activationRoot, "serveTimeRouteDecisions");
    try {
        const lines = [
            JSON.stringify({
                recordId: "decision-historical-selection",
                selectionDigest: "selection-historical",
                activePackGraphChecksum: "graph-historical",
                turnCompileEventId: "evt-historical-selection",
                recordedAt: "2026-04-01T00:00:00.000Z",
                turnCreatedAt: "2026-04-01T00:00:00.000Z",
                sessionId: "sess-1",
                channel: "cli",
                userMessage: "historical selection decision",
            }),
        ];
        for (let i = 0; i < 1200; i++) {
            lines.push(JSON.stringify({
                recordId: `decision-tail-${i}`,
                selectionDigest: `selection-tail-${i}`,
                activePackGraphChecksum: "graph-historical",
                turnCompileEventId: `evt-tail-${i}`,
                recordedAt: `2026-04-01T02:${String(i % 60).padStart(2, "0")}:00.000Z`,
                turnCreatedAt: `2026-04-01T02:${String(i % 60).padStart(2, "0")}:00.000Z`,
                sessionId: "sess-1",
                channel: "cli",
                userMessage: `tail decision ${i}`,
                padding: "y".repeat(4096)
            }));
        }
        writeFileSync(logPath, lines.join("\n") + "\n", "utf8");
        const result = resolveServeTimeLearningRuntimeInput(activationRoot, makeNormalizedEventExport([
            {
                selectionDigest: "selection-historical",
                activePackGraphChecksum: "graph-historical",
                createdAt: "2026-04-01T02:59:00.100Z",
            }
        ]));
        const matcher = createServeTimeDecisionMatcher(result.serveTimeDecisions);
        assert.equal(matcher(makeInteraction({
            selectionDigest: "selection-historical",
            activePackGraphChecksum: "graph-historical",
            createdAt: "2026-04-01T02:59:00.100Z",
        }))?.recordId, "decision-historical-selection");
    } finally {
        rmSync(activationRoot, { recursive: true, force: true });
    }
});

test("historical recovery preserves ambiguous compile-event matches outside the bounded tail", () => {
    const activationRoot = makeActivationRoot();
    const logPath = resolveLearningSpineLogPath(activationRoot, "serveTimeRouteDecisions");
    try {
        const lines = [
            JSON.stringify({
                recordId: "decision-ambiguous-a",
                turnCompileEventId: "evt-historical-duplicate",
                recordedAt: "2026-04-01T00:00:00.000Z",
                turnCreatedAt: "2026-04-01T00:00:00.000Z",
                sessionId: "sess-1",
                channel: "cli",
                userMessage: "ambiguous decision a",
            }),
            JSON.stringify({
                recordId: "decision-ambiguous-b",
                turnCompileEventId: "evt-historical-duplicate",
                recordedAt: "2026-04-01T00:00:01.000Z",
                turnCreatedAt: "2026-04-01T00:00:01.000Z",
                sessionId: "sess-1",
                channel: "cli",
                userMessage: "ambiguous decision b",
            }),
        ];
        for (let i = 0; i < 1200; i++) {
            lines.push(JSON.stringify({
                recordId: `decision-tail-${i}`,
                turnCompileEventId: `evt-tail-${i}`,
                recordedAt: `2026-04-01T03:${String(i % 60).padStart(2, "0")}:00.000Z`,
                turnCreatedAt: `2026-04-01T03:${String(i % 60).padStart(2, "0")}:00.000Z`,
                sessionId: "sess-1",
                channel: "cli",
                userMessage: `tail decision ${i}`,
                padding: "z".repeat(4096)
            }));
        }
        writeFileSync(logPath, lines.join("\n") + "\n", "utf8");
        const result = resolveServeTimeLearningRuntimeInput(activationRoot, makeNormalizedEventExport([
            { turnCompileEventId: "evt-historical-duplicate" }
        ]));
        const matcher = createServeTimeDecisionMatcher(result.serveTimeDecisions);
        assert.equal(matcher(makeInteraction({
            turnCompileEventId: "evt-historical-duplicate",
            eventId: "evt-unrelated",
            createdAt: "2026-04-01T03:59:00.100Z",
        })), null);
        assert.equal(result.serveTimeDecisions.filter((decision) => decision.turnCompileEventId === "evt-historical-duplicate").length, 2);
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
