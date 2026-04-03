import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { formatOperatorLearningHealthSummary } from "../src/status-learning-path.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunction({ file, startMarker, endMarker, prelude = "" }) {
    const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${file}`);
    }
    const block = source.slice(start, end).replace(/^export\s+/gmu, "");
    return new Function(`${prelude}\n${block}\nreturn ${startMarker.split(" ")[1]};`)();
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
const teacherNoArtifactBlock = loadBlock("index.js", "export function summarizeTeacherNoArtifactCycle", "function summarizeAlwaysOnLearning")
    .replace("export function summarizeTeacherNoArtifactCycle", "function summarizeTeacherNoArtifactCycle");
const summarizeStatusTeacher = loadFunction({
    file: "cli.js",
    startMarker: "function summarizeStatusTeacher",
    endMarker: "function summarizeStatusEmbedder",
    prelude: `${teacherMessagesBlock}\n${teacherNoArtifactBlock}`
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
            notes: [
                "teacher_last_cycle_deterministic_artifacts=0",
                "teacher_last_cycle_new_deterministic_artifacts=0",
                "teacher_last_cycle_labeler_candidates=0",
                "teacher_last_cycle_labeler_budgeted_candidates=0",
                "teacher_last_cycle_labeler_status=skipped",
                "teacher_last_cycle_labeler_detail=no_matching_interaction_text"
            ],
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
                lastNoOpReason: "no_teacher_artifacts",
                notes: [
                    "teacher_last_cycle_deterministic_artifacts=0",
                    "teacher_last_cycle_new_deterministic_artifacts=0",
                    "teacher_last_cycle_labeler_candidates=0",
                    "teacher_last_cycle_labeler_budgeted_candidates=0",
                    "teacher_last_cycle_labeler_status=skipped",
                    "teacher_last_cycle_labeler_detail=no_matching_interaction_text"
                ]
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
    assert.match(summary.detail, /no eligible feedback, operator overrides, or matched interaction text/);
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

test("status teacher treats lagging watch heartbeats as aging instead of stale", () => {
    const summary = summarizeStatusTeacher(makeTeacherReport({
        latestFreshness: "fresh",
        watchState: "lagging",
        watch: {
            state: "lagging",
            detail: "watch heartbeat missed the healthy window but has not crossed the stale snapshot threshold"
        }
    }), providerConfig, localLlm);
    assert.equal(summary.healthy, false);
    assert.equal(summary.stale, false);
    assert.equal(summary.idle, true);
    assert.match(summary.detail, /watch heartbeat is lagging/);
});

test("status truth shows stalled-learning separately from a healthy daemon", () => {
    const teacher = summarizeStatusTeacher(makeTeacherReport({
        latestFreshness: "fresh",
        watchState: "watching"
    }), providerConfig, localLlm);
    assert.equal(teacher.healthy, true);
    const health = formatOperatorLearningHealthSummary({
        teacher,
        tracedLearning: {
            present: true,
            teacherArtifactCount: 3,
            routeTraceCount: 0,
            supervisionCount: 0,
            routerUpdateCount: 0,
            attributionCoverage: {
                visible: true,
                readyCount: 2
            }
        }
    });
    assert.equal(health, "daemon=healthy-daemon learning=stalled-learning detail=harvested artifacts and eligible feedback are visible, but no matched routes, supervision, or router updates are visible");
});

test("learning warnings separate no-artifact no-ops from genuinely stale teacher labels", () => {
    const noArtifactWarnings = summarizeLearningWarningStates(makeWarningInput());
    assert.deepEqual(noArtifactWarnings, []);
    const staleWarnings = summarizeLearningWarningStates(makeWarningInput({
        teacherSnapshot: {
            queue: {
                capacity: 1,
                depth: 0
            },
            diagnostics: {
                latestFreshness: "stale",
                lastNoOpReason: "none",
                notes: []
            }
        }
    }));
    assert.deepEqual(staleWarnings, ["teacher_labels_stale"]);
});

test("learning warnings keep flagging no-artifact cycles when teachable material was missed", () => {
    const warnings = summarizeLearningWarningStates(makeWarningInput({
        teacherSnapshot: {
            queue: {
                capacity: 1,
                depth: 0
            },
            diagnostics: {
                latestFreshness: "stale",
                lastNoOpReason: "no_teacher_artifacts",
                notes: [
                    "teacher_last_cycle_deterministic_artifacts=0",
                    "teacher_last_cycle_new_deterministic_artifacts=0",
                    "teacher_last_cycle_labeler_candidates=2",
                    "teacher_last_cycle_labeler_budgeted_candidates=2",
                    "teacher_last_cycle_labeler_status=skipped",
                    "teacher_last_cycle_labeler_detail=no_labels_emitted"
                ]
            }
        }
    }));
    assert.deepEqual(warnings, ["teacher_no_artifacts"]);
});

test("status teacher explains when no-artifact no-ops should worry the operator", () => {
    const summary = summarizeStatusTeacher(makeTeacherReport({
        notes: [
            "teacher_last_cycle_deterministic_artifacts=0",
            "teacher_last_cycle_new_deterministic_artifacts=0",
            "teacher_last_cycle_labeler_candidates=2",
            "teacher_last_cycle_labeler_budgeted_candidates=2",
            "teacher_last_cycle_labeler_status=skipped",
            "teacher_last_cycle_labeler_detail=no_labels_emitted"
        ]
    }), providerConfig, localLlm);
    assert.equal(summary.healthy, true);
    assert.match(summary.detail, /candidate interactions were present/);
    assert.match(summary.detail, /no reusable labels/);
});
