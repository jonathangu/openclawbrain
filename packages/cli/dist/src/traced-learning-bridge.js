import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const TRACED_LEARNING_BRIDGE_CONTRACT = "openclawbrain.traced-learning-bridge.v1";
const TRACED_LEARNING_BRIDGE_FILENAME = "traced-learning-state.json";
// Canonical split-package learn/status summary persisted under brain_training_state.
const TRACED_LEARNING_STATUS_SURFACE_STATE_KEY = "traced_learning_status_surface_json";
const TRACED_LEARNING_STATUS_SURFACE_CONTRACT = "openclawbrain.traced-learning-status-surface.v1";
const TRACED_LEARNING_STATUS_SURFACE_BRIDGE = "brain_store_traced_learning_status_surface";

function normalizeCount(value) {
    return Number.isFinite(value) && value >= 0 ? Math.trunc(value) : 0;
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
function normalizeUnitInterval(value) {
    return Number.isFinite(value) ? Math.max(0, Math.min(1, Number(value))) : 0;
}
function normalizeSource(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
}
function normalizeLastInterruptionSummary(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    const normalized = {
        reason: normalizeOptionalString(value.reason),
        stage: normalizeOptionalString(value.stage),
        servedPartial: value.servedPartial === true,
        droppedFrontierCount: normalizeCount(value.droppedFrontierCount),
        droppedProposalCount: normalizeCount(value.droppedProposalCount),
        budgetUtilization: normalizeUnitInterval(value.budgetUtilization)
    };
    return normalized.reason !== null ||
        normalized.stage !== null ||
        normalized.servedPartial ||
        normalized.droppedFrontierCount > 0 ||
        normalized.droppedProposalCount > 0 ||
        normalized.budgetUtilization > 0
        ? normalized
        : null;
}
function formatLastInterruptionDetail(value) {
    const summary = normalizeLastInterruptionSummary(value);
    if (summary === null) {
        return null;
    }
    return [
        `interrupt=${summary.reason ?? summary.stage ?? "unknown"}`,
        `partial=${summary.servedPartial ? "yes" : "no"}`,
        `frontier=${summary.droppedFrontierCount}`,
        `proposals=${summary.droppedProposalCount}`,
        `budget=${Math.round(summary.budgetUtilization * 100)}%`
    ].join(" ");
}
function buildLastInterruptionSummaryFromAssemblyDecision(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    const accounting = value.interruptionAccounting !== null &&
        typeof value.interruptionAccounting === "object" &&
        !Array.isArray(value.interruptionAccounting)
        ? value.interruptionAccounting
        : null;
    return normalizeLastInterruptionSummary({
        reason: normalizeOptionalString(value.brainDropReason) ?? normalizeOptionalString(value.interruptionReason),
        stage: normalizeOptionalString(value.interruptionStage),
        servedPartial: value.servedPartial === true,
        droppedFrontierCount: Array.isArray(accounting?.droppedFrontierNodeIds)
            ? accounting.droppedFrontierNodeIds.filter((entry) => typeof entry === "string" && entry.trim().length > 0).length
            : normalizeCount(accounting?.droppedFrontierCount),
        droppedProposalCount: normalizeCount(accounting?.droppedProposalCount),
        budgetUtilization: accounting?.budgetUtilization
    });
}
function summarizeBridgeSource(value) {
    const source = normalizeSource(value);
    if (source === null) {
        return null;
    }
    const summarized = {
        command: normalizeOptionalString(source.command),
        bridge: normalizeOptionalString(source.bridge),
        brainRoot: normalizeOptionalString(source.brainRoot),
        stateDbPath: normalizeOptionalString(source.stateDbPath),
        persistedKey: normalizeOptionalString(source.persistedKey),
        candidatePackVersion: Number.isFinite(source.candidatePackVersion) ? Math.trunc(source.candidatePackVersion) : undefined,
        candidateUpdateCount: normalizeCount(source.candidateUpdateCount)
    };
    return Object.fromEntries(Object.entries(summarized).filter(([, candidate]) => candidate !== null && candidate !== undefined));
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
        lastInterruptionSummary: normalizeLastInterruptionSummary(payload.lastInterruptionSummary),
        source: normalizeSource(payload.source)
    };
}
function normalizePersistedStatusSurface(payload) {
    if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("expected traced-learning status surface payload object");
    }
    const source = normalizeSource(payload.source);
    if (source === null) {
        throw new Error("expected traced-learning status surface source");
    }
    return {
        contract: TRACED_LEARNING_STATUS_SURFACE_CONTRACT,
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
        lastInterruptionSummary: normalizeLastInterruptionSummary(payload.lastInterruptionSummary),
        source
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
        lastInterruptionSummary: null,
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
function loadTrainingStateJson(db, key) {
    const raw = loadTrainingStateValue(db, key);
    if (typeof raw !== "string") {
        return {
            value: null,
            error: null
        };
    }
    const trimmed = raw.trim();
    if (trimmed.length === 0) {
        return {
            value: null,
            error: null
        };
    }
    try {
        return {
            value: JSON.parse(trimmed),
            error: null
        };
    }
    catch (error) {
        return {
            value: null,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}
function loadLastAssemblyInterruptionSummary(db) {
    const loaded = loadTrainingStateJson(db, "last_assembly_decision_json");
    return loaded.value === null ? null : buildLastInterruptionSummaryFromAssemblyDecision(loaded.value);
}
function writeTrainingStateJson(db, key, value) {
    db.prepare(`INSERT OR REPLACE INTO brain_training_state (key, value) VALUES (?, ?)`).run(key, JSON.stringify(value));
}
function buildTracedLearningRoutingBuildFromServeTime(serveTimeLearning) {
    return {
        learnedRoutingPath: serveTimeLearning.pgVersion === "v2" ? "policy_gradient_v2" : "policy_gradient_v1",
        pgVersionRequested: serveTimeLearning.pgVersion,
        pgVersionUsed: serveTimeLearning.pgVersion,
        decisionLogCount: normalizeCount(serveTimeLearning.decisionLogCount),
        fallbackReason: serveTimeLearning.pgVersion === "v1"
            ? normalizeOptionalString(serveTimeLearning.fallbackReason) ?? "no_serve_time_decisions"
            : null,
        updatedBaseline: null
    };
}
export function buildTracedLearningBridgePayloadFromRuntime(input) {
    const lastMaterialization = input?.lastMaterialization ?? null;
    const serveTimeLearning = input?.serveTimeLearning ?? {
        pgVersion: "v1",
        decisionLogCount: 0,
        fallbackReason: null
    };
    const learnedRouter = lastMaterialization?.candidate?.summary?.learnedRouter ?? null;
    const routingBuild = lastMaterialization?.candidate?.routingBuild ?? buildTracedLearningRoutingBuildFromServeTime(serveTimeLearning);
    const fallbackReason = routingBuild.fallbackReason ??
        (routingBuild.pgVersionUsed === "v1"
            ? normalizeOptionalString(serveTimeLearning.fallbackReason) ?? "no_serve_time_decisions"
            : null);
    return normalizeBridgePayload({
        updatedAt: input?.updatedAt,
        routeTraceCount: learnedRouter?.routeTraceCount ?? serveTimeLearning.decisionLogCount,
        supervisionCount: learnedRouter?.supervisionCount ?? 0,
        routerUpdateCount: learnedRouter?.updateCount ?? 0,
        teacherArtifactCount: input?.teacherArtifactCount ?? 0,
        pgVersionRequested: routingBuild.pgVersionRequested,
        pgVersionUsed: routingBuild.pgVersionUsed,
        decisionLogCount: routingBuild.decisionLogCount,
        fallbackReason,
        routerNoOpReason: learnedRouter?.noOpReason ?? null,
        materializedPackId: input?.materializedPackId ?? lastMaterialization?.candidate?.summary?.packId ?? null,
        promoted: input?.promoted === true,
        baselinePersisted: input?.baselinePersisted === true,
        lastInterruptionSummary: input?.lastInterruptionSummary ?? null,
        source: input?.source
    });
}
function countRows(db, tableName) {
    const row = db.prepare(`SELECT COUNT(*) as count FROM ${tableName}`).get();
    return normalizeCount(row?.count);
}
function toIsoTimestamp(value) {
    return Number.isFinite(value) && value > 0 ? new Date(value).toISOString() : null;
}
function buildPersistedStatusSurfaceBridge(summary, context) {
    return normalizeBridgePayload({
        updatedAt: summary.updatedAt,
        routeTraceCount: summary.routeTraceCount,
        supervisionCount: summary.supervisionCount,
        routerUpdateCount: summary.routerUpdateCount,
        teacherArtifactCount: summary.teacherArtifactCount,
        pgVersionRequested: summary.pgVersionRequested,
        pgVersionUsed: summary.pgVersionUsed,
        decisionLogCount: summary.decisionLogCount,
        fallbackReason: summary.fallbackReason,
        routerNoOpReason: summary.routerNoOpReason,
        materializedPackId: summary.materializedPackId,
        promoted: summary.promoted,
        baselinePersisted: summary.baselinePersisted,
        lastInterruptionSummary: summary.lastInterruptionSummary,
        source: {
            command: "brain-store",
            bridge: TRACED_LEARNING_STATUS_SURFACE_BRIDGE,
            brainRoot: context.brainRoot,
            stateDbPath: context.dbPath,
            persistedKey: TRACED_LEARNING_STATUS_SURFACE_STATE_KEY,
            surfacedFrom: summary.source
        }
    });
}
function loadPersistedStatusSurface(db, context) {
    const loaded = loadTrainingStateJson(db, TRACED_LEARNING_STATUS_SURFACE_STATE_KEY);
    if (loaded.value === null) {
        return {
            bridge: null,
            error: loaded.error
        };
    }
    try {
        if (normalizeOptionalString(loaded.value.contract) !== TRACED_LEARNING_STATUS_SURFACE_CONTRACT) {
            throw new Error("unexpected traced-learning status surface contract");
        }
        return {
            bridge: buildPersistedStatusSurfaceBridge(normalizePersistedStatusSurface(loaded.value), context),
            error: null
        };
    }
    catch (error) {
        return {
            bridge: null,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}
function buildDerivedBrainStoreBridge(db, context, lastInterruptionSummary = null) {
    const routeTraceCount = countRows(db, "brain_traces");
    const supervisionCount = countRows(db, "brain_trace_supervision");
    const candidateUpdateRaw = loadTrainingStateValue(db, "last_pg_candidate_update_json");
    const candidatePackVersionRaw = loadTrainingStateValue(db, "last_pg_candidate_pack_version");
    const candidateUpdate = candidateUpdateRaw === null || candidateUpdateRaw.trim().length === 0
        ? null
        : JSON.parse(candidateUpdateRaw);
    const candidatePackVersion = Number.parseInt(candidatePackVersionRaw ?? "", 10);
    return normalizeBridgePayload({
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
        lastInterruptionSummary,
        source: {
            command: "brain-store",
            bridge: "brain_store_state",
            brainRoot: context.brainRoot,
            stateDbPath: context.dbPath,
            candidatePackVersion: Number.isFinite(candidatePackVersion) ? candidatePackVersion : null,
            candidateUpdateCount: normalizeCount(candidateUpdate?.updateCount)
        }
    });
}
function hasMeaningfulTracedLearningSignal(bridge) {
    return bridge.routeTraceCount > 0 ||
        bridge.supervisionCount > 0 ||
        bridge.routerUpdateCount > 0 ||
        bridge.teacherArtifactCount > 0 ||
        bridge.decisionLogCount > 0 ||
        bridge.materializedPackId !== null ||
        bridge.promoted ||
        bridge.baselinePersisted ||
        bridge.lastInterruptionSummary !== null ||
        bridge.pgVersionRequested !== null ||
        bridge.pgVersionUsed !== null ||
        bridge.fallbackReason !== null ||
        bridge.routerNoOpReason !== null ||
        Number.isFinite(bridge.source?.candidatePackVersion) ||
        normalizeCount(bridge.source?.candidateUpdateCount) > 0;
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
export function persistBrainStoreTracedLearningBridge(payload, options = {}) {
    const brainRoot = resolveBrainRoot(options.env ?? process.env);
    const dbPath = path.join(brainRoot, "state.db");
    if (!existsSync(dbPath)) {
        return {
            path: dbPath,
            bridge: null,
            persisted: false,
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
            persisted: false,
            error: null
        };
    }
    let db;
    try {
        db = new sqlite.DatabaseSync(dbPath);
        const summary = normalizePersistedStatusSurface(payload);
        writeTrainingStateJson(db, TRACED_LEARNING_STATUS_SURFACE_STATE_KEY, summary);
        return {
            path: dbPath,
            bridge: buildPersistedStatusSurfaceBridge(summary, {
                brainRoot,
                dbPath
            }),
            persisted: true,
            error: null
        };
    }
    catch (error) {
        return {
            path: dbPath,
            bridge: null,
            persisted: false,
            error: error instanceof Error ? error.message : String(error)
        };
    }
    finally {
        if (db && typeof db.close === "function") {
            db.close();
        }
    }
}
export function persistTracedLearningBridgeState(activationRoot, payload, options = {}) {
    const bridge = mergeTracedLearningBridgePayload(payload, loadBrainStoreTracedLearningBridge(options));
    persistBrainStoreTracedLearningBridge(bridge, options);
    writeTracedLearningBridge(activationRoot, bridge);
    return bridge;
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
        const lastInterruptionSummary = loadLastAssemblyInterruptionSummary(db);
        const persisted = loadPersistedStatusSurface(db, {
            brainRoot,
            dbPath
        });
        if (persisted.bridge !== null) {
            const bridge = lastInterruptionSummary === null
                ? persisted.bridge
                : normalizeBridgePayload({
                    ...persisted.bridge,
                    lastInterruptionSummary
                });
            return {
                path: dbPath,
                bridge,
                error: null
            };
        }
        const bridge = buildDerivedBrainStoreBridge(db, {
            brainRoot,
            dbPath
        }, lastInterruptionSummary);
        if (!hasMeaningfulTracedLearningSignal(bridge)) {
            return {
                path: dbPath,
                bridge: null,
                error: persisted.error
            };
        }
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
    if (typeof bridge.source?.bridge === "string") {
        detailParts.push(`bridge=${bridge.source.bridge}`);
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
    const interruptionDetail = formatLastInterruptionDetail(bridge.lastInterruptionSummary);
    if (interruptionDetail !== null) {
        detailParts.push(interruptionDetail);
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
        lastInterruptionSummary: bridge.lastInterruptionSummary,
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
        lastInterruptionSummary: loaded.bridge.lastInterruptionSummary,
        fallbackReason: loaded.bridge.fallbackReason,
        routerNoOpReason: loaded.bridge.routerNoOpReason,
        source: loaded.bridge.source
    };
}
function mergeCanonicalStatusBridge(canonicalBridge, runtimeLoaded) {
    const runtimeBridge = runtimeLoaded.bridge;
    const runtimeMaterialized = buildRuntimeMaterializationMetadata(runtimeLoaded);
    const hasPersistedSurface = canonicalBridge.source?.bridge === TRACED_LEARNING_STATUS_SURFACE_BRIDGE;
    if (hasPersistedSurface) {
        return {
            updatedAt: canonicalBridge.updatedAt,
            routeTraceCount: canonicalBridge.routeTraceCount,
            supervisionCount: canonicalBridge.supervisionCount,
            routerUpdateCount: canonicalBridge.routerUpdateCount,
            teacherArtifactCount: canonicalBridge.teacherArtifactCount,
            pgVersionRequested: canonicalBridge.pgVersionRequested,
            pgVersionUsed: canonicalBridge.pgVersionUsed,
            decisionLogCount: canonicalBridge.decisionLogCount,
            materializedPackId: canonicalBridge.materializedPackId,
            promoted: canonicalBridge.promoted,
            baselinePersisted: canonicalBridge.baselinePersisted,
            lastInterruptionSummary: canonicalBridge.lastInterruptionSummary ?? runtimeBridge?.lastInterruptionSummary ?? null,
            fallbackReason: canonicalBridge.fallbackReason,
            routerNoOpReason: canonicalBridge.routerNoOpReason,
            source: runtimeMaterialized === null
                ? canonicalBridge.source
                : {
                    ...(canonicalBridge.source ?? {}),
                    runtimeMaterialized
                }
        };
    }
    return {
        updatedAt: canonicalBridge.updatedAt ?? runtimeBridge?.updatedAt ?? null,
        routeTraceCount: canonicalBridge.routeTraceCount,
        supervisionCount: canonicalBridge.supervisionCount,
        routerUpdateCount: canonicalBridge.routerUpdateCount,
        teacherArtifactCount: canonicalBridge.teacherArtifactCount,
        pgVersionRequested: runtimeBridge?.pgVersionRequested ?? canonicalBridge.pgVersionRequested ?? null,
        pgVersionUsed: runtimeBridge?.pgVersionUsed ?? canonicalBridge.pgVersionUsed ?? null,
        decisionLogCount: runtimeBridge?.decisionLogCount ?? canonicalBridge.decisionLogCount ?? 0,
        materializedPackId: runtimeBridge?.materializedPackId ?? canonicalBridge.materializedPackId ?? null,
        promoted: runtimeBridge?.promoted ?? canonicalBridge.promoted,
        baselinePersisted: runtimeBridge?.baselinePersisted ?? canonicalBridge.baselinePersisted,
        lastInterruptionSummary: canonicalBridge.lastInterruptionSummary ?? runtimeBridge?.lastInterruptionSummary ?? null,
        fallbackReason: runtimeBridge?.fallbackReason ?? canonicalBridge.fallbackReason ?? null,
        routerNoOpReason: runtimeBridge?.routerNoOpReason ?? canonicalBridge.routerNoOpReason ?? null,
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
    const lastInterruptionSummary = current.lastInterruptionSummary ?? persistedBridge.lastInterruptionSummary ?? null;
    const usedBridge = routeTraceCount !== current.routeTraceCount ||
        supervisionCount !== current.supervisionCount ||
        routerUpdateCount !== current.routerUpdateCount ||
        teacherArtifactCount !== current.teacherArtifactCount ||
        lastInterruptionSummary !== current.lastInterruptionSummary;
    if (!usedBridge) {
        return current;
    }
    return normalizeBridgePayload({
        ...current,
        routeTraceCount,
        supervisionCount,
        routerUpdateCount,
        teacherArtifactCount,
        lastInterruptionSummary,
        routerNoOpReason: supervisionCount > 0 || routerUpdateCount > 0 ? null : current.routerNoOpReason,
        source: {
            ...(current.source ?? {}),
            bridge: normalizeOptionalString(persistedBridge.source?.bridge) ?? "brain_store_state",
            bridgedRuntime: {
                path: persisted?.path ?? null,
                updatedAt: persistedBridge.updatedAt,
                routeTraceCount: persistedBridge.routeTraceCount,
                supervisionCount: persistedBridge.supervisionCount,
                routerUpdateCount: persistedBridge.routerUpdateCount,
                teacherArtifactCount: persistedBridge.teacherArtifactCount,
                source: summarizeBridgeSource(persistedBridge.source)
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
