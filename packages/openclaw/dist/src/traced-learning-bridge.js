import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const TRACED_LEARNING_BRIDGE_CONTRACT = "openclawbrain.traced-learning-bridge.v1";
const TRACED_LEARNING_BRIDGE_FILENAME = "traced-learning-state.json";

function normalizeCount(value) {
    return Number.isFinite(value) && value >= 0 ? Math.trunc(value) : 0;
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
function normalizeSource(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
}
function normalizeBridgePayload(payload) {
    if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("expected traced-learning bridge payload object");
    }
    return {
        contract: TRACED_LEARNING_BRIDGE_CONTRACT,
        updatedAt: normalizeOptionalString(payload.updatedAt) ?? new Date().toISOString(),
        routeTraceCount: normalizeCount(payload.routeTraceCount),
        supervisionCount: normalizeCount(payload.supervisionCount),
        routerUpdateCount: normalizeCount(payload.routerUpdateCount),
        teacherArtifactCount: normalizeCount(payload.teacherArtifactCount),
        pgVersionRequested: normalizeOptionalString(payload.pgVersionRequested),
        pgVersionUsed: normalizeOptionalString(payload.pgVersionUsed),
        decisionLogCount: normalizeCount(payload.decisionLogCount),
        fallbackReason: normalizeOptionalString(payload.fallbackReason),
        routerNoOpReason: normalizeOptionalString(payload.routerNoOpReason),
        materializedPackId: normalizeOptionalString(payload.materializedPackId),
        promoted: payload.promoted === true,
        baselinePersisted: payload.baselinePersisted === true,
        source: normalizeSource(payload.source)
    };
}
function defaultSurface(pathname, detail, error = null) {
    return {
        path: pathname,
        present: false,
        updatedAt: null,
        routeTraceCount: 0,
        supervisionCount: 0,
        routerUpdateCount: 0,
        teacherArtifactCount: 0,
        pgVersionRequested: null,
        pgVersionUsed: null,
        decisionLogCount: 0,
        materializedPackId: null,
        promoted: false,
        baselinePersisted: false,
        source: null,
        detail,
        error
    };
}
function resolveBrainRoot(env = process.env) {
    const explicit = normalizeOptionalString(env.OPENCLAWBRAIN_ROOT);
    if (explicit !== null) {
        return path.resolve(explicit);
    }
    const lcmDatabasePath = normalizeOptionalString(env.LCM_DATABASE_PATH);
    if (lcmDatabasePath !== null) {
        return path.join(path.dirname(path.resolve(lcmDatabasePath)), "openclawbrain");
    }
    return path.join(homedir(), ".openclaw", "openclawbrain");
}
function loadTrainingStateValue(db, key) {
    const row = db.prepare(`SELECT value FROM brain_training_state WHERE key = ?`).get(key);
    return row !== undefined && typeof row.value === "string" ? row.value : null;
}
function countRows(db, tableName) {
    const row = db.prepare(`SELECT COUNT(*) as count FROM ${tableName}`).get();
    return normalizeCount(row?.count);
}
function toIsoTimestamp(value) {
    return Number.isFinite(value) && value > 0 ? new Date(value).toISOString() : null;
}
export function resolveTracedLearningBridgePath(activationRoot) {
    return path.join(path.resolve(activationRoot), "watch", TRACED_LEARNING_BRIDGE_FILENAME);
}
export function writeTracedLearningBridge(activationRoot, payload) {
    const bridgePath = resolveTracedLearningBridgePath(activationRoot);
    const bridge = normalizeBridgePayload(payload);
    mkdirSync(path.dirname(bridgePath), { recursive: true });
    writeFileSync(bridgePath, `${JSON.stringify(bridge, null, 2)}\n`, "utf8");
    return bridgePath;
}
export function loadTracedLearningBridge(activationRoot) {
    const bridgePath = resolveTracedLearningBridgePath(activationRoot);
    if (!existsSync(bridgePath)) {
        return {
            path: bridgePath,
            bridge: null,
            error: null
        };
    }
    try {
        const parsed = JSON.parse(readFileSync(bridgePath, "utf8"));
        return {
            path: bridgePath,
            bridge: normalizeBridgePayload(parsed),
            error: null
        };
    }
    catch (error) {
        return {
            path: bridgePath,
            bridge: null,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}
export function loadBrainStoreTracedLearningBridge(options = {}) {
    const brainRoot = resolveBrainRoot(options.env ?? process.env);
    const dbPath = path.join(brainRoot, "state.db");
    if (!existsSync(dbPath)) {
        return {
            path: dbPath,
            bridge: null,
            error: null
        };
    }
    const sqlite = typeof process.getBuiltinModule === "function"
        ? process.getBuiltinModule("node:sqlite")
        : null;
    if (sqlite === null || typeof sqlite.DatabaseSync !== "function") {
        return {
            path: dbPath,
            bridge: null,
            error: null
        };
    }
    let db;
    try {
        db = new sqlite.DatabaseSync(dbPath, { readOnly: true });
        const routeTraceCount = countRows(db, "brain_traces");
        const supervisionCount = countRows(db, "brain_trace_supervision");
        const candidateUpdateRaw = loadTrainingStateValue(db, "last_pg_candidate_update_json");
        const candidatePackVersionRaw = loadTrainingStateValue(db, "last_pg_candidate_pack_version");
        const candidateUpdate = candidateUpdateRaw === null || candidateUpdateRaw.trim().length === 0
            ? null
            : JSON.parse(candidateUpdateRaw);
        const candidatePackVersion = Number.parseInt(candidatePackVersionRaw ?? "", 10);
        const bridge = normalizeBridgePayload({
            updatedAt: toIsoTimestamp(candidateUpdate?.generatedAt),
            routeTraceCount,
            supervisionCount,
            routerUpdateCount: candidateUpdate?.routeUpdateCount,
            teacherArtifactCount: candidateUpdate?.teacherLabelCount,
            pgVersionRequested: null,
            pgVersionUsed: null,
            decisionLogCount: 0,
            fallbackReason: null,
            routerNoOpReason: null,
            materializedPackId: null,
            promoted: false,
            baselinePersisted: false,
            source: {
                command: "brain-store",
                bridge: "brain_store_state",
                brainRoot,
                stateDbPath: dbPath,
                candidatePackVersion: Number.isFinite(candidatePackVersion) ? candidatePackVersion : null,
                candidateUpdateCount: normalizeCount(candidateUpdate?.updateCount)
            }
        });
        return {
            path: dbPath,
            bridge,
            error: null
        };
    }
    catch (error) {
        return {
            path: dbPath,
            bridge: null,
            error: error instanceof Error ? error.message : String(error)
        };
    }
    finally {
        if (db && typeof db.close === "function") {
            db.close();
        }
    }
}
function describeBridgeRuntimeState(loaded) {
    return loaded.bridge === null ? (loaded.error === null ? "missing" : "unreadable") : "present";
}
function buildStatusSurface(pathname, bridge, options = {}) {
    const detailParts = [
        `source=${bridge.source?.command === undefined ? "learn" : String(bridge.source.command)}`,
        `promoted=${bridge.promoted ? "yes" : "no"}`
    ];
    if (bridge.source?.bridge === "brain_store_state") {
        detailParts.push("bridge=brain_store_state");
    }
    if (options.runtimeState !== undefined) {
        detailParts.push(`runtime=${options.runtimeState}`);
    }
    if (bridge.fallbackReason !== null) {
        detailParts.push(`fallback=${bridge.fallbackReason}`);
    }
    if (bridge.routerNoOpReason !== null) {
        detailParts.push(`noOp=${bridge.routerNoOpReason}`);
    }
    return {
        path: pathname,
        present: true,
        updatedAt: bridge.updatedAt,
        routeTraceCount: bridge.routeTraceCount,
        supervisionCount: bridge.supervisionCount,
        routerUpdateCount: bridge.routerUpdateCount,
        teacherArtifactCount: bridge.teacherArtifactCount,
        pgVersionRequested: bridge.pgVersionRequested,
        pgVersionUsed: bridge.pgVersionUsed,
        decisionLogCount: bridge.decisionLogCount,
        materializedPackId: bridge.materializedPackId,
        promoted: bridge.promoted,
        baselinePersisted: bridge.baselinePersisted,
        source: bridge.source,
        detail: detailParts.join(" "),
        error: options.error ?? null
    };
}
function buildRuntimeMaterializationMetadata(loaded) {
    if (loaded.bridge === null) {
        return null;
    }
    return {
        path: loaded.path,
        updatedAt: loaded.bridge.updatedAt,
        routeTraceCount: loaded.bridge.routeTraceCount,
        supervisionCount: loaded.bridge.supervisionCount,
        routerUpdateCount: loaded.bridge.routerUpdateCount,
        teacherArtifactCount: loaded.bridge.teacherArtifactCount,
        pgVersionRequested: loaded.bridge.pgVersionRequested,
        pgVersionUsed: loaded.bridge.pgVersionUsed,
        decisionLogCount: loaded.bridge.decisionLogCount,
        materializedPackId: loaded.bridge.materializedPackId,
        promoted: loaded.bridge.promoted,
        baselinePersisted: loaded.bridge.baselinePersisted,
        fallbackReason: loaded.bridge.fallbackReason,
        routerNoOpReason: loaded.bridge.routerNoOpReason,
        source: loaded.bridge.source
    };
}
function mergeCanonicalStatusBridge(canonicalBridge, runtimeLoaded) {
    const runtimeBridge = runtimeLoaded.bridge;
    const runtimeMaterialized = buildRuntimeMaterializationMetadata(runtimeLoaded);
    return {
        updatedAt: canonicalBridge.updatedAt ?? runtimeBridge?.updatedAt ?? null,
        routeTraceCount: canonicalBridge.routeTraceCount,
        supervisionCount: canonicalBridge.supervisionCount,
        routerUpdateCount: canonicalBridge.routerUpdateCount,
        teacherArtifactCount: canonicalBridge.teacherArtifactCount,
        pgVersionRequested: runtimeBridge?.pgVersionRequested ?? null,
        pgVersionUsed: runtimeBridge?.pgVersionUsed ?? null,
        decisionLogCount: runtimeBridge?.decisionLogCount ?? 0,
        materializedPackId: runtimeBridge?.materializedPackId ?? null,
        promoted: runtimeBridge?.promoted ?? false,
        baselinePersisted: runtimeBridge?.baselinePersisted ?? false,
        fallbackReason: runtimeBridge?.fallbackReason ?? null,
        routerNoOpReason: runtimeBridge?.routerNoOpReason ?? null,
        source: runtimeMaterialized === null
            ? canonicalBridge.source
            : {
                ...(canonicalBridge.source ?? {}),
                runtimeMaterialized
            }
    };
}
export function mergeTracedLearningBridgePayload(payload, persisted) {
    const current = normalizeBridgePayload(payload);
    const persistedBridge = persisted?.bridge ?? null;
    if (persistedBridge === null) {
        return current;
    }
    const routeTraceCount = Math.max(current.routeTraceCount, persistedBridge.routeTraceCount);
    const supervisionCount = Math.max(current.supervisionCount, persistedBridge.supervisionCount);
    const routerUpdateCount = Math.max(current.routerUpdateCount, persistedBridge.routerUpdateCount);
    const teacherArtifactCount = Math.max(current.teacherArtifactCount, persistedBridge.teacherArtifactCount);
    const usedBridge = routeTraceCount !== current.routeTraceCount ||
        supervisionCount !== current.supervisionCount ||
        routerUpdateCount !== current.routerUpdateCount ||
        teacherArtifactCount !== current.teacherArtifactCount;
    if (!usedBridge) {
        return current;
    }
    return normalizeBridgePayload({
        ...current,
        routeTraceCount,
        supervisionCount,
        routerUpdateCount,
        teacherArtifactCount,
        routerNoOpReason: supervisionCount > 0 || routerUpdateCount > 0 ? null : current.routerNoOpReason,
        source: {
            ...(current.source ?? {}),
            bridge: "brain_store_state",
            bridgedRuntime: {
                path: persisted?.path ?? null,
                updatedAt: persistedBridge.updatedAt,
                routeTraceCount: persistedBridge.routeTraceCount,
                supervisionCount: persistedBridge.supervisionCount,
                routerUpdateCount: persistedBridge.routerUpdateCount,
                teacherArtifactCount: persistedBridge.teacherArtifactCount,
                source: persistedBridge.source
            }
        }
    });
}
export function buildTracedLearningStatusSurface(activationRoot, options = {}) {
    const persisted = loadBrainStoreTracedLearningBridge(options);
    const runtime = loadTracedLearningBridge(activationRoot);
    if (persisted.bridge !== null) {
        return buildStatusSurface(persisted.path, mergeCanonicalStatusBridge(persisted.bridge, runtime), {
            runtimeState: describeBridgeRuntimeState(runtime)
        });
    }
    if (runtime.bridge !== null) {
        return buildStatusSurface(runtime.path, runtime.bridge);
    }
    if (persisted.error !== null) {
        return defaultSurface(persisted.path, "brain_store_unreadable", persisted.error);
    }
    return defaultSurface(runtime.path, runtime.error === null ? "bridge_missing" : "bridge_unreadable", runtime.error);
}
