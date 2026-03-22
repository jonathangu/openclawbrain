import { existsSync, openSync, readSync, closeSync, statSync, readFileSync } from "node:fs";

const DEFAULT_TAIL_BYTES = 4 * 1024 * 1024;
const DEFAULT_MAX_ENTRIES = 512;
const DEFAULT_MAX_LINE_BYTES = 128 * 1024;

export function readBoundedJsonlTail(filePath, options = {}) {
    const tailBytes = options.tailBytes ?? DEFAULT_TAIL_BYTES;
    const maxEntries = options.maxEntries ?? DEFAULT_MAX_ENTRIES;
    const maxLineBytes = options.maxLineBytes ?? DEFAULT_MAX_LINE_BYTES;
    if (!existsSync(filePath)) {
        return { entries: [], fallbackReason: null };
    }
    let fallbackReason = null;
    let raw;
    try {
        const fileSize = statSync(filePath).size;
        if (fileSize <= tailBytes) {
            raw = readFileSync(filePath, "utf8");
        }
        else {
            fallbackReason = "tail_truncated";
            const offset = fileSize - tailBytes;
            const buf = Buffer.alloc(tailBytes);
            const fd = openSync(filePath, "r");
            try {
                readSync(fd, buf, 0, tailBytes, offset);
            }
            finally {
                closeSync(fd);
            }
            raw = buf.toString("utf8");
            const firstNewline = raw.indexOf("\n");
            if (firstNewline >= 0) {
                raw = raw.slice(firstNewline + 1);
            }
        }
    }
    catch {
        return { entries: [], fallbackReason: "read_error" };
    }
    const lines = raw.split(/\r?\n/u).map((l) => l.trim()).filter((l) => l.length > 0);
    const entries = [];
    let skippedOversized = 0;
    for (const line of lines) {
        if (Buffer.byteLength(line, "utf8") > maxLineBytes) {
            skippedOversized++;
            continue;
        }
        try {
            entries.push(JSON.parse(line));
        }
        catch {
            // skip malformed lines
        }
    }
    if (skippedOversized > 0) {
        fallbackReason = fallbackReason ? `${fallbackReason}+oversized_lines_skipped` : "oversized_lines_skipped";
    }
    const trimmed = entries.length > maxEntries ? entries.slice(-maxEntries) : entries;
    if (entries.length > maxEntries) {
        fallbackReason = fallbackReason ? `${fallbackReason}+entry_count_capped` : "entry_count_capped";
    }
    return { entries: trimmed, fallbackReason };
}
