import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cliSource = readFileSync(path.join(__dirname, "..", "src", "cli.js"), "utf8");
const messagesStart = cliSource.indexOf("const TEACHER_NO_OP_MESSAGES =");
const messagesEnd = cliSource.indexOf("function summarizeStatusInstallHook", messagesStart);
const teacherStart = cliSource.indexOf("function summarizeStatusTeacher");
const teacherEnd = cliSource.indexOf("function summarizeStatusEmbedder", teacherStart);

if (messagesStart === -1 || messagesEnd === -1 || teacherStart === -1 || teacherEnd === -1) {
    throw new Error("failed to locate teacher status logic in packages/openclaw/dist/src/cli.js");
}

const summarizeStatusTeacher = new Function(`
${cliSource.slice(messagesStart, messagesEnd)}
${cliSource.slice(teacherStart, teacherEnd)}
return summarizeStatusTeacher;
`)();

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

test("openclaw dist teacher summary keeps fresh no-artifact heartbeats healthy", () => {
    const summary = summarizeStatusTeacher(makeTeacherReport(), providerConfig, localLlm);
    assert.equal(summary.enabled, true);
    assert.equal(summary.latestCycle, "no_op");
    assert.equal(summary.healthy, true);
    assert.equal(summary.stale, false);
    assert.equal(summary.idle, true);
});

test("openclaw dist teacher summary still marks stale snapshots unhealthy", () => {
    const summary = summarizeStatusTeacher(makeTeacherReport({
        latestFreshness: "fresh",
        watchState: "stale_snapshot"
    }), providerConfig, localLlm);
    assert.equal(summary.healthy, false);
    assert.equal(summary.stale, true);
    assert.equal(summary.idle, true);
});
