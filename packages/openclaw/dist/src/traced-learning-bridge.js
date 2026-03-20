import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
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
export function buildTracedLearningStatusSurface(activationRoot) {
    const loaded = loadTracedLearningBridge(activationRoot);
    if (loaded.bridge === null) {
        return defaultSurface(loaded.path, loaded.error === null ? "bridge_missing" : `bridge_unreadable`, loaded.error);
    }
    const detailParts = [
        `source=${loaded.bridge.source?.command === undefined ? "learn" : String(loaded.bridge.source.command)}`,
        `promoted=${loaded.bridge.promoted ? "yes" : "no"}`
    ];
    if (loaded.bridge.fallbackReason !== null) {
        detailParts.push(`fallback=${loaded.bridge.fallbackReason}`);
    }
    if (loaded.bridge.routerNoOpReason !== null) {
        detailParts.push(`noOp=${loaded.bridge.routerNoOpReason}`);
    }
    return {
        path: loaded.path,
        present: true,
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
        source: loaded.bridge.source,
        detail: detailParts.join(" "),
        error: null
    };
}
