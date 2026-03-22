import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { readBoundedJsonlTail } from "../src/bounded-jsonl-reader.js";

function makeTmpDir() {
    const dir = path.join(os.tmpdir(), `bounded-jsonl-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(dir, { recursive: true });
    return dir;
}

test("returns empty entries for missing file", () => {
    const result = readBoundedJsonlTail("/nonexistent/path.jsonl");
    assert.deepEqual(result, { entries: [], fallbackReason: null });
});

test("reads a small JSONL file fully", () => {
    const dir = makeTmpDir();
    const fp = path.join(dir, "small.jsonl");
    try {
        const lines = [];
        for (let i = 0; i < 5; i++) {
            lines.push(JSON.stringify({ blockId: `b${i}`, actionScore: i * 0.1 }));
        }
        writeFileSync(fp, lines.join("\n") + "\n", "utf8");
        const { entries, fallbackReason } = readBoundedJsonlTail(fp);
        assert.equal(entries.length, 5);
        assert.equal(fallbackReason, null);
        assert.equal(entries[0].blockId, "b0");
        assert.equal(entries[4].blockId, "b4");
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("huge historical log does not explode — tail-truncated with capped entries", () => {
    const dir = makeTmpDir();
    const fp = path.join(dir, "huge.jsonl");
    try {
        // Write enough data to exceed our explicit bounds
        const chunks = [];
        for (let i = 0; i < 400; i++) {
            chunks.push(JSON.stringify({
                blockId: `block-${i}`,
                actionScore: i * 0.01,
                padding: "x".repeat(2000)
            }));
        }
        writeFileSync(fp, chunks.join("\n") + "\n", "utf8");
        const { entries, fallbackReason } = readBoundedJsonlTail(fp, {
            tailBytes: 256 * 1024,
            maxEntries: 100
        });
        // Should cap at our explicit maxEntries
        assert.ok(entries.length <= 100, `expected <= 100 entries, got ${entries.length}`);
        // Should report truncation
        assert.ok(fallbackReason !== null, "expected a fallbackReason for large file");
        assert.ok(fallbackReason.includes("entry_count_capped") || fallbackReason.includes("tail_truncated"),
            `unexpected fallbackReason: ${fallbackReason}`);
        // Entries should still be valid objects with actionScore
        for (const entry of entries) {
            assert.ok(typeof entry.blockId === "string");
            assert.ok(typeof entry.actionScore === "number");
        }
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("oversized lines are skipped with fallbackReason", () => {
    const dir = makeTmpDir();
    const fp = path.join(dir, "oversized.jsonl");
    try {
        const normal = JSON.stringify({ blockId: "ok", actionScore: 1.0 });
        // Create a line larger than the 1 KB limit we'll set
        const oversized = JSON.stringify({ blockId: "big", payload: "x".repeat(2000) });
        writeFileSync(fp, [normal, oversized, normal].join("\n") + "\n", "utf8");
        const { entries, fallbackReason } = readBoundedJsonlTail(fp, { maxLineBytes: 1024 });
        assert.equal(entries.length, 2, "oversized line should be skipped");
        assert.ok(entries.every((e) => e.blockId === "ok"));
        assert.equal(fallbackReason, "oversized_lines_skipped");
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("maxEntries caps output and sets fallbackReason", () => {
    const dir = makeTmpDir();
    const fp = path.join(dir, "capped.jsonl");
    try {
        const lines = [];
        for (let i = 0; i < 10; i++) {
            lines.push(JSON.stringify({ blockId: `b${i}`, actionScore: i }));
        }
        writeFileSync(fp, lines.join("\n") + "\n", "utf8");
        const { entries, fallbackReason } = readBoundedJsonlTail(fp, { maxEntries: 3 });
        assert.equal(entries.length, 3);
        // Should keep the LAST 3
        assert.equal(entries[0].blockId, "b7");
        assert.equal(entries[2].blockId, "b9");
        assert.equal(fallbackReason, "entry_count_capped");
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("persisted candidateScores are compact but keep actionScore", () => {
    // Simulates the slim format now written by appendServeTimeRouteDecisionLog
    const compactEntry = {
        recordType: "serve_time_route_decision",
        candidateScores: [
            { blockId: "ctx-intro", selected: true, actionScore: 0.85, actionProbability: 0.6 },
            { blockId: "ctx-faq", selected: false, actionScore: 0.15, actionProbability: 0.4 }
        ],
        structuralSignals: null
    };
    const dir = makeTmpDir();
    const fp = path.join(dir, "compact.jsonl");
    try {
        writeFileSync(fp, JSON.stringify(compactEntry) + "\n", "utf8");
        const { entries } = readBoundedJsonlTail(fp);
        assert.equal(entries.length, 1);
        const scores = entries[0].candidateScores;
        assert.equal(scores.length, 2);
        // Must have blockId + actionScore
        assert.equal(scores[0].blockId, "ctx-intro");
        assert.equal(scores[0].actionScore, 0.85);
        assert.equal(scores[0].selected, true);
        assert.equal(scores[1].blockId, "ctx-faq");
        assert.equal(scores[1].actionScore, 0.15);
        // structuralSignals should be null (compact)
        assert.equal(entries[0].structuralSignals, null);
        // Verbose fields should NOT be present
        assert.equal(scores[0].matchedTokens, undefined);
        assert.equal(scores[0].channelScores, undefined);
        assert.equal(scores[0].routingChannels, undefined);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("tail read on large file still yields parseable entries from the end", () => {
    const dir = makeTmpDir();
    const fp = path.join(dir, "tail.jsonl");
    try {
        const lines = [];
        for (let i = 0; i < 100; i++) {
            lines.push(JSON.stringify({ idx: i, actionScore: i }));
        }
        writeFileSync(fp, lines.join("\n") + "\n", "utf8");
        // Use a tiny tailBytes so truncation kicks in
        const { entries, fallbackReason } = readBoundedJsonlTail(fp, { tailBytes: 200 });
        assert.ok(entries.length > 0 && entries.length < 100, "should read a subset");
        assert.equal(fallbackReason, "tail_truncated");
        // Last entry should be the final line
        assert.equal(entries[entries.length - 1].idx, 99);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
