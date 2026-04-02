import { existsSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { discoverOpenClawHomes } from "./openclaw-home-layout.js";
export function loadOpenClawSessionIndex(indexFilePath) {
    return parseJsonFile(indexFilePath);
}
export function readOpenClawSessionFile(sessionFilePath) {
    return readJsonlFile(sessionFilePath, parseOpenClawSessionRecord);
}
export function readOpenClawAcpStreamFile(streamFilePath) {
    return readJsonlFile(streamFilePath, parseOpenClawAcpStreamRecord);
}
export function discoverOpenClawMainSessionStores(options = {}) {
    const candidateRoots = options.profileRoots !== undefined
        ? [...new Set(options.profileRoots.map((root) => path.resolve(root)))]
        : discoverOpenClawProfileRoots(options.homeDir);
    return candidateRoots
        .map((profileRoot) => {
        const sessionsDir = path.join(profileRoot, "agents", "main", "sessions");
        const indexPath = path.join(sessionsDir, "sessions.json");
        if (!existsSync(indexPath)) {
            return null;
        }
        return {
            profileRoot,
            agentId: "main",
            sessionsDir,
            indexPath
        };
    })
        .filter((store) => store !== null)
        .sort((left, right) => left.indexPath.localeCompare(right.indexPath));
}
/**
 * Discover session stores for ALL agents under each profile root.
 * Scans every directory under `agents/` (not just `agents/main/`),
 * finding subagent, ACP, and any other agent session stores.
 */
export function discoverOpenClawSessionStores(options = {}) {
    const candidateRoots = options.profileRoots !== undefined
        ? [...new Set(options.profileRoots.map((root) => path.resolve(root)))]
        : discoverOpenClawProfileRoots(options.homeDir);
    const stores = [];
    for (const profileRoot of candidateRoots) {
        const agentsDir = path.join(profileRoot, "agents");
        if (!existsSync(agentsDir)) {
            continue;
        }
        let agentDirs;
        try {
            agentDirs = readdirSync(agentsDir, { withFileTypes: true })
                .filter((entry) => entry.isDirectory())
                .map((entry) => entry.name);
        }
        catch {
            continue;
        }
        for (const agentName of agentDirs) {
            const sessionsDir = path.join(agentsDir, agentName, "sessions");
            const indexPath = path.join(sessionsDir, "sessions.json");
            if (!existsSync(indexPath)) {
                continue;
            }
            stores.push({
                profileRoot,
                agentId: agentName,
                sessionsDir,
                indexPath
            });
        }
    }
    return stores.sort((left, right) => left.indexPath.localeCompare(right.indexPath));
}
function discoverOpenClawProfileRoots(homeDir) {
    const rootDir = path.resolve(homeDir ?? process.env.HOME ?? process.env.USERPROFILE ?? homedir());
    if (!existsSync(rootDir)) {
        return [];
    }
    const configuredRoots = discoverOpenClawHomes(rootDir).map((inspection) => inspection.openclawHome);
    // Session-store discovery must still see legacy per-profile homes that have live
    // session data but do not yet carry an openclaw.json layout marker.
    let legacyRoots = [];
    try {
        legacyRoots = readdirSync(rootDir, { withFileTypes: true })
            .filter((entry) => entry.isDirectory() && entry.name.startsWith(".openclaw-"))
            .map((entry) => path.join(rootDir, entry.name))
            .filter((profileRoot) => existsSync(path.join(profileRoot, "agents")));
    }
    catch {
        legacyRoots = [];
    }
    return [...new Set([...configuredRoots, ...legacyRoots])].sort((left, right) => left.localeCompare(right));
}
function parseJsonFile(filePath) {
    return JSON.parse(readFileSync(filePath, "utf8"));
}
function readJsonlFile(filePath, parseRecord) {
    const text = readFileSync(filePath, "utf8");
    if (text.trim() === "") {
        return [];
    }
    return text
        .split(/\r?\n/)
        .filter((line) => line.trim().length > 0)
        .map((line, index) => {
        try {
            return parseRecord(JSON.parse(line), index + 1);
        }
        catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new Error(`Could not parse JSONL record at ${filePath}:${index + 1}: ${message}`);
        }
    });
}
function parseOpenClawSessionRecord(value, lineNumber) {
    const record = expectRecord(value, lineNumber);
    const type = expectString(record.type, `${lineNumber}.type`);
    switch (type) {
        case "session":
            return {
                type,
                version: expectNumber(record.version, `${lineNumber}.version`),
                id: expectString(record.id, `${lineNumber}.id`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`),
                cwd: expectString(record.cwd, `${lineNumber}.cwd`)
            };
        case "model_change":
            return {
                type,
                id: expectString(record.id, `${lineNumber}.id`),
                parentId: expectNullableString(record.parentId, `${lineNumber}.parentId`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`),
                provider: expectString(record.provider, `${lineNumber}.provider`),
                modelId: expectString(record.modelId, `${lineNumber}.modelId`)
            };
        case "thinking_level_change":
            return {
                type,
                id: expectString(record.id, `${lineNumber}.id`),
                parentId: expectNullableString(record.parentId, `${lineNumber}.parentId`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`),
                thinkingLevel: expectString(record.thinkingLevel, `${lineNumber}.thinkingLevel`)
            };
        case "custom":
            return {
                type,
                customType: expectString(record.customType, `${lineNumber}.customType`),
                data: expectRecord(record.data, `${lineNumber}.data`),
                id: expectString(record.id, `${lineNumber}.id`),
                parentId: expectNullableString(record.parentId, `${lineNumber}.parentId`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`)
            };
        case "custom_message": {
            const data = {};
            if (record.content !== undefined) {
                data.content = record.content;
            }
            if (record.display !== undefined) {
                data.display = record.display;
            }
            if (record.details !== undefined) {
                data.details = record.details;
            }
            return {
                type: "custom",
                customType: expectString(record.customType, `${lineNumber}.customType`),
                data,
                id: expectString(record.id, `${lineNumber}.id`),
                parentId: expectNullableString(record.parentId, `${lineNumber}.parentId`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`)
            };
        }
        case "compaction": {
            const data = {};
            if (record.summary !== undefined) {
                data.summary = expectString(record.summary, `${lineNumber}.summary`);
            }
            if (record.firstKeptEntryId !== undefined) {
                data.firstKeptEntryId = expectString(record.firstKeptEntryId, `${lineNumber}.firstKeptEntryId`);
            }
            if (record.tokensBefore !== undefined) {
                data.tokensBefore = expectNumber(record.tokensBefore, `${lineNumber}.tokensBefore`);
            }
            if (record.details !== undefined) {
                data.details = expectRecord(record.details, `${lineNumber}.details`);
            }
            if (record.fromHook !== undefined) {
                if (typeof record.fromHook !== "boolean") {
                    throw new Error(`${lineNumber}.fromHook must be a boolean`);
                }
                data.fromHook = record.fromHook;
            }
            return {
                type: "custom",
                customType: "openclaw.compaction",
                data,
                id: expectString(record.id, `${lineNumber}.id`),
                parentId: expectNullableString(record.parentId, `${lineNumber}.parentId`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`)
            };
        }
        case "message":
            return {
                type,
                id: expectString(record.id, `${lineNumber}.id`),
                parentId: expectNullableString(record.parentId, `${lineNumber}.parentId`),
                timestamp: expectString(record.timestamp, `${lineNumber}.timestamp`),
                message: parseMessagePayload(record.message, `${lineNumber}.message`)
            };
        default:
            throw new Error(`Unknown OpenClaw session record type: ${type}`);
    }
}
function parseMessagePayload(value, path) {
    const payload = expectRecord(value, path);
    const content = typeof payload.content === "string"
        ? [{ type: "text", text: payload.content }]
        : expectArray(payload.content, `${path}.content`).map((entry, index) => parseContentPart(entry, `${path}.content[${index}]`));
    return {
        ...payload,
        role: expectString(payload.role, `${path}.role`),
        content,
        timestamp: expectNumber(payload.timestamp, `${path}.timestamp`)
    };
}
function parseContentPart(value, path) {
    const content = expectRecord(value, path);
    const type = expectString(content.type, `${path}.type`);
    if (type === "text") {
        return {
            ...content,
            type,
            text: expectString(content.text, `${path}.text`)
        };
    }
    if (type === "thinking") {
        return {
            ...content,
            type,
            thinking: expectString(content.thinking, `${path}.thinking`)
        };
    }
    if (type === "toolCall") {
        return {
            ...content,
            type,
            id: expectString(content.id, `${path}.id`),
            name: expectString(content.name, `${path}.name`),
            arguments: expectRecord(content.arguments, `${path}.arguments`)
        };
    }
    return {
        ...content,
        type
    };
}
function parseOpenClawAcpStreamRecord(value, lineNumber) {
    const record = expectRecord(value, lineNumber);
    return {
        ...record,
        ts: expectString(record.ts, `${lineNumber}.ts`),
        epochMs: expectNumber(record.epochMs, `${lineNumber}.epochMs`),
        runId: expectString(record.runId, `${lineNumber}.runId`),
        parentSessionKey: expectString(record.parentSessionKey, `${lineNumber}.parentSessionKey`),
        childSessionKey: expectString(record.childSessionKey, `${lineNumber}.childSessionKey`),
        agentId: expectString(record.agentId, `${lineNumber}.agentId`),
        kind: expectString(record.kind, `${lineNumber}.kind`)
    };
}
function expectRecord(value, path) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        throw new Error(`${path} must be an object`);
    }
    return value;
}
function expectArray(value, path) {
    if (!Array.isArray(value)) {
        throw new Error(`${path} must be an array`);
    }
    return value;
}
function expectString(value, path) {
    if (typeof value !== "string") {
        throw new Error(`${path} must be a string`);
    }
    return value;
}
function expectNullableString(value, path) {
    if (value === null) {
        return null;
    }
    return expectString(value, path);
}
function expectNumber(value, path) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${path} must be a finite number`);
    }
    return value;
}
//# sourceMappingURL=session-store.js.map
