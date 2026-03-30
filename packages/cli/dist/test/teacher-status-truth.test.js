import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunction({ file, startMarker, endMarker, prelude = "" }) {
    const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${file}`);
    }
    return new Function(`${prelude}\n${source.slice(start, end)}\nreturn ${startMarker.split(" ")[1]};`)();
}

function loadBlock(file, startMarker, endMarker) {
    const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${file}`);
    }
    return source.slice(start, end);
}

const teacherMessagesBlock = loadBlock("cli.js", "const TEACHER_NO_OP_MESSAGES =", "function summarizeStatusInstallHook");
const summarizeStatusTeacher = loadFunction({
    file: "cli.js",
    startMarker: "function summarizeStatusTeacher",
    endMarker: "function summarizeStatusEmbedder",
    prelude: teacherMessagesBlock
});
const summarizeLearningWarningStates = loadFunction({
    file: "index.js",
    startMarker: "function summarizeLearningWarningStates",
    endMarker: "function summarizeAlwaysOnLearning"
});

const providerConfig = {
    teacher: {
        provider: "ollama",
        model: "unsloth-qwen3.5-27b:q4_k_m"
    },
    teacherBaseUrl: "http://127.0.0.1:11434"
};

const localLlm = { detected: true };

function makeTeacherReport(overrides = {}) {
    return {
        teacherLoop: {
            available: true,
            lastNoOpReason: "no_teacher_artifacts",
            latestFreshness: "stale",
            watchState: "watching",
            running: false,
            queueDepth: 0,
            failureMode: "none",
            failureDetail: null,
            ...overrides
        }
    };
}

function makeWarningInput(overrides = {}) {
    return {
        plan: {
            bootstrapped: true,
            pending: {
                total: 0,
                backfill: 0,
                byBucket: {
                    principal_immediate: 0,
                    principal_backfill: 0
                }
            }
        },
        principalLagStatus: "caught_up",
        teacherSnapshot: {
            queue: {
                capacity: 1,
                depth: 0
            },
            diagnostics: {
                latestFreshness: "stale",
                lastNoOpReason: "no_teacher_artifacts"
            }
        },
        ...overrides
    };
}

test("status teacher stays healthy for a fresh watch heartbeat with no teacher artifacts", () => {
    const summary = summarizeStatusTeacher(makeTeacherReport(), providerConfig, localLlm);
    assert.equal(summary.enabled, true);
    assert.equal(summary.latestCycle, "no_op");
    assert.equal(summary.healthy, true);
    assert.equal(summary.stale, false);
    assert.equal(summary.idle, true);
});

test("status teacher stays unhealthy when the watch snapshot itself is stale", () => {
    const summary = summarizeStatusTeacher(makeTeacherReport({
        latestFreshness: "fresh",
        watchState: "stale_snapshot"
    }), providerConfig, localLlm);
    assert.equal(summary.healthy, false);
    assert.equal(summary.stale, true);
    assert.equal(summary.idle, true);
});

test("learning warnings separate no-artifact no-ops from genuinely stale teacher labels", () => {
    const noArtifactWarnings = summarizeLearningWarningStates(makeWarningInput());
    assert.deepEqual(noArtifactWarnings, ["teacher_no_artifacts"]);
    const staleWarnings = summarizeLearningWarningStates(makeWarningInput({
        teacherSnapshot: {
            queue: {
                capacity: 1,
                depth: 0
            },
            diagnostics: {
                latestFreshness: "stale",
                lastNoOpReason: "none"
            }
        }
    }));
    assert.deepEqual(staleWarnings, ["teacher_labels_stale"]);
});
