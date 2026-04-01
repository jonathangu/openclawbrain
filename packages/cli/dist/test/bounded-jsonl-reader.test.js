import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { readBoundedJsonlTail } from "../src/bounded-jsonl-reader.js";

function makeTmpDir() {
    const dir = path.join(os.tmpdir(), `cli-bounded-jsonl-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(dir, { recursive: true });
    return dir;
}

test("large JSONL tail stays bounded and keeps recent parseable entries", () => {
    const dir = makeTmpDir();
    const fp = path.join(dir, "large.jsonl");
    try {
        const lines = [];
        for (let i = 0; i < 1200; i++) {
            lines.push(JSON.stringify({ idx: i, actionScore: i, padding: "x".repeat(4096) }));
        }
        writeFileSync(fp, lines.join("\n") + "\n", "utf8");
        const { entries, fallbackReason } = readBoundedJsonlTail(fp, { tailBytes: 256 * 1024, maxEntries: 32 });
        assert.equal(entries.length, 32);
        assert.equal(entries[entries.length - 1].idx, 1199);
        assert.ok(fallbackReason?.includes("tail_truncated"));
        assert.ok(fallbackReason?.includes("entry_count_capped") || fallbackReason?.includes("tail_truncated"));
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
