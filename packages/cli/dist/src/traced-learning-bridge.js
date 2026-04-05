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
function parseJsonValue(value, fallback) {
    if (typeof value !== "string" || value.trim().length === 0) {
        return fallback;
    }
    try {
        return JSON.parse(value);
    }
    catch {
        return fallback;
    }
}
function normalizeUnitInterval(value) {
    return Number.isFinite(value) ? Math.max(0, Math.min(1, Number(value))) : 0;
}
function normalizeSource(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
}
function toRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
}
function normalizeAgentIdentity(value) {
    const record = toRecord(value);
    const agentId = normalizeOptionalString(record?.agentId);
    const lane = normalizeOptionalString(record?.lane);
    return agentId === null || lane === null ? null : { agentId, lane };
}
function formatAgentIdentity(identity) {
    if (identity === null) {
        return null;
    }
    return identity.lane === "main" ? identity.agentId : `${identity.agentId}:${identity.lane}`;
}
function defaultFeedbackSummary(routeTraceCount = 0, supervisedTraceCount = 0, detail = "feedback truth is not visible in the current status surface") {
    return {
        visible: false,
        helpfulCount: 0,
        irrelevantCount: 0,
        harmfulCount: 0,
        supervisedTraceCount,
        routeTraceCount,
        latestAgentIdentity: null,
        latestLabel: null,
        detail
    };
}
function normalizeFeedbackSummary(value, counts = {}) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return defaultFeedbackSummary(normalizeCount(counts.routeTraceCount), normalizeCount(counts.supervisedTraceCount));
    }
    const routeTraceCount = normalizeCount(value.routeTraceCount ?? counts.routeTraceCount);
    const supervisedTraceCount = normalizeCount(value.supervisedTraceCount ?? counts.supervisedTraceCount);
    const latestAgentIdentity = normalizeAgentIdentity(value.latestAgentIdentity);
    return {
        visible: value.visible === true,
        helpfulCount: normalizeCount(value.helpfulCount),
        irrelevantCount: normalizeCount(value.irrelevantCount),
        harmfulCount: normalizeCount(value.harmfulCount),
        supervisedTraceCount,
        routeTraceCount,
        latestAgentIdentity,
        latestLabel: normalizeOptionalString(value.latestLabel) ?? formatAgentIdentity(latestAgentIdentity),
        detail: normalizeOptionalString(value.detail)
            ?? (routeTraceCount === 0
                ? "no traced routes recorded yet"
                : `${supervisedTraceCount}/${routeTraceCount} traced routes are covered by live verdicts`)
    };
}
function defaultAttributionCoverage(detail = "teacher gating truth is not visible in the current status surface") {
    return {
        visible: false,
        gatingVisible: false,
        completedWithoutEvaluationCount: 0,
        readyCount: 0,
        delayedCount: 0,
        budgetDeferredCount: 0,
        detail
    };
}
function normalizeAttributionCoverage(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return defaultAttributionCoverage();
    }
    return {
        visible: value.visible === true,
        gatingVisible: value.gatingVisible === true,
        completedWithoutEvaluationCount: normalizeCount(value.completedWithoutEvaluationCount),
        readyCount: normalizeCount(value.readyCount),
        delayedCount: normalizeCount(value.delayedCount),
        budgetDeferredCount: normalizeCount(value.budgetDeferredCount),
        detail: normalizeOptionalString(value.detail)
            ?? "teacher gating truth is not visible in the current status surface"
    };
}
function hasNonEmptyToolResults(value) {
    if (Array.isArray(value)) {
        return value.length > 0;
    }
    return typeof value === "string" && value.trim().length > 0 && value.trim() !== "[]";
}
function hasTeacherBindingMode(rawEvaluation) {
    const evaluation = parseJsonValue(rawEvaluation, null);
    switch (evaluation?.bindingMode) {
        case "exact_decision_id":
        case "exact_selection_digest":
        case "turn_compile_event_id":
        case "trace_id":
        case "legacy_heuristic":
        case "unbound":
            return true;
        default:
            return false;
    }
}
function classifyContextFeedbackVerdict(score) {
    if (Number(score) >= 0.25) {
        return "helpful";
    }
    if (Number(score) <= -0.25) {
        return "harmful";
    }
    return "irrelevant";
}
function isObservationReadyForTeacher(row, readyBefore) {
    return row.status === "pending_teacher"
        || normalizeOptionalString(row.follow_up_text) !== null
        || hasNonEmptyToolResults(row.tool_results_json)
        || Number(row.created_at ?? 0) <= readyBefore;
}
function buildDerivedFeedbackSummary(db, routeTraceCount, defaultSupervisionCount) {
    const traceRows = db.prepare(`
      SELECT id, route_trace_json
      FROM brain_traces
    `).all();
    const traceAgentIdentityById = new Map();
    for (const row of traceRows) {
        const routeTrace = parseJsonValue(row?.route_trace_json, null);
        const agentIdentity = normalizeAgentIdentity(routeTrace?.agentIdentity);
        if (agentIdentity !== null && typeof row?.id === "string") {
            traceAgentIdentityById.set(row.id, agentIdentity);
        }
    }
    const verdictCounts = {
        helpfulCount: 0,
        irrelevantCount: 0,
        harmfulCount: 0
    };
    const latestTraceIds = new Set();
    let latestAgentIdentity = null;
    const supervisionRows = db.prepare(`
      SELECT trace_id, metadata, value
      FROM brain_trace_supervision
      WHERE resolution = 'promoted_to_label'
      ORDER BY created_at DESC
    `).all();
    for (const row of supervisionRows) {
        const traceId = normalizeOptionalString(row?.trace_id);
        if (traceId === null || latestTraceIds.has(traceId)) {
            continue;
        }
        latestTraceIds.add(traceId);
        const metadata = parseJsonValue(row?.metadata, {});
        const agentIdentity = normalizeAgentIdentity(metadata?.agentIdentity)
            ?? traceAgentIdentityById.get(traceId)
            ?? null;
        if (latestAgentIdentity === null) {
            latestAgentIdentity = agentIdentity;
        }
        const verdict = classifyContextFeedbackVerdict(Number(row?.value ?? 0));
        if (verdict === "helpful") {
            verdictCounts.helpfulCount += 1;
        }
        else if (verdict === "harmful") {
            verdictCounts.harmfulCount += 1;
        }
        else {
            verdictCounts.irrelevantCount += 1;
        }
    }
    const supervisedTraceCount = latestTraceIds.size || normalizeCount(defaultSupervisionCount);
    return {
        visible: true,
        ...verdictCounts,
        supervisedTraceCount,
        routeTraceCount,
        latestAgentIdentity,
        latestLabel: formatAgentIdentity(latestAgentIdentity),
        detail: routeTraceCount === 0
            ? "no traced routes recorded yet"
            : `${verdictCounts.helpfulCount} helpful, ${verdictCounts.irrelevantCount} irrelevant, ${verdictCounts.harmfulCount} harmful; ${supervisedTraceCount}/${routeTraceCount} traced routes are supervised`
    };
}
function buildDerivedAttributionCoverage(db) {
    const completedRows = db.prepare(`
      SELECT teacher_evaluation_json
      FROM brain_observations
      WHERE status = 'completed'
    `).all();
    const completedWithoutEvaluationCount = completedRows.reduce((sum, row) => sum + (hasTeacherBindingMode(row?.teacher_evaluation_json) ? 0 : 1), 0);
    const evaluationCycle = loadTrainingStateJson(db, "last_teacher_evaluation_cycle_json");
    const budgetPerTick = Number.isFinite(evaluationCycle.value?.budgetPerTick)
        ? Math.max(0, Math.trunc(evaluationCycle.value.budgetPerTick))
        : null;
    const delayMs = Number.isFinite(evaluationCycle.value?.delayMs)
        ? Math.max(0, Math.trunc(evaluationCycle.value.delayMs))
        : null;
    if (budgetPerTick === null || delayMs === null) {
        return {
            visible: true,
            gatingVisible: false,
            completedWithoutEvaluationCount,
            readyCount: 0,
            delayedCount: 0,
            budgetDeferredCount: 0,
            detail: "teacher gating truth is not visible in the current status surface"
        };
    }
    const pendingRows = db.prepare(`
      SELECT status, follow_up_text, tool_results_json, created_at
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
      ORDER BY created_at ASC
    `).all();
    const readyBefore = Date.now() - delayMs;
    let readyCount = 0;
    for (const row of pendingRows) {
        if (isObservationReadyForTeacher(row, readyBefore)) {
            readyCount += 1;
        }
    }
    const delayedCount = Math.max(0, pendingRows.length - readyCount);
    const budgetDeferredCount = Math.max(0, readyCount - budgetPerTick);
    return {
        visible: true,
        gatingVisible: true,
        completedWithoutEvaluationCount,
        readyCount,
        delayedCount,
        budgetDeferredCount,
        detail: pendingRows.length === 0
            ? "no teacher observations are pending"
            : `completed_without_evaluation=${completedWithoutEvaluationCount}; ready=${readyCount}, delayed=${delayedCount}, budget_deferred=${budgetDeferredCount}`
    };
}
function loadJsonFile(pathname) {
    if (!existsSync(pathname)) {
        return null;
    }
    try {
        return JSON.parse(readFileSync(pathname, "utf8"));
    }
    catch {
        return null;
    }
}
function resolveActivePackPaths(activationRoot) {
    const pointers = toRecord(loadJsonFile(path.join(path.resolve(activationRoot), "activation-pointers.json")));
    const active = toRecord(pointers?.active);
    const packId = normalizeOptionalString(active?.packId);
    const packRootDir = normalizeOptionalString(active?.packRootDir)
        ?? (packId === null
            ? null
            : path.join(path.resolve(activationRoot), "packs", String(packId)));
    const manifestPath = normalizeOptionalString(active?.manifestPath)
        ?? (packRootDir === null ? null : path.join(packRootDir, "manifest.json"));
    return {
        packId,
        packRootDir,
        manifestPath
    };
}
function buildActivePackFeedbackSummary(activationRoot) {
    const active = resolveActivePackPaths(activationRoot);
    if (active.packRootDir === null || active.manifestPath === null) {
        return null;
    }
    const manifest = toRecord(loadJsonFile(active.manifestPath));
    const runtimeAssets = toRecord(manifest?.runtimeAssets);
    const router = toRecord(runtimeAssets?.router);
    const routerArtifactPath = normalizeOptionalString(router?.artifactPath);
    if (routerArtifactPath === null) {
        return null;
    }
    const routerPath = path.isAbsolute(routerArtifactPath)
        ? routerArtifactPath
        : path.join(active.packRootDir, routerArtifactPath);
    const routerArtifact = toRecord(loadJsonFile(routerPath));
    const traces = Array.isArray(routerArtifact?.traces) ? routerArtifact.traces : [];
    const verdictCounts = {
        helpfulCount: 0,
        irrelevantCount: 0,
        harmfulCount: 0
    };
    for (const trace of traces) {
        const traceRecord = toRecord(trace);
        if (normalizeOptionalString(traceRecord?.supervisionKind) === "route_trace") {
            continue;
        }
        const verdict = classifyContextFeedbackVerdict(Number(traceRecord?.reward ?? 0));
        if (verdict === "helpful") {
            verdictCounts.helpfulCount += 1;
        }
        else if (verdict === "harmful") {
            verdictCounts.harmfulCount += 1;
        }
        else {
            verdictCounts.irrelevantCount += 1;
        }
    }
    const routeTraceCount = normalizeCount(routerArtifact?.training?.routeTraceCount) || traces.length;
    const supervisedTraceCount = verdictCounts.helpfulCount + verdictCounts.irrelevantCount + verdictCounts.harmfulCount;
    if (routeTraceCount === 0 && supervisedTraceCount === 0) {
        return null;
    }
    return {
        visible: true,
        ...verdictCounts,
        supervisedTraceCount,
        routeTraceCount,
        latestAgentIdentity: null,
        latestLabel: null,
        detail: routeTraceCount === 0
            ? "no active-pack traced routes are visible"
            : `${verdictCounts.helpfulCount} helpful, ${verdictCounts.irrelevantCount} irrelevant, ${verdictCounts.harmfulCount} harmful; ${supervisedTraceCount}/${routeTraceCount} active-pack traced routes are supervised`
    };
}
function readIntegerNote(notes, prefix) {
    if (!Array.isArray(notes)) {
        return null;
    }
    const entry = notes.find((candidate) => typeof candidate === "string" && candidate.startsWith(prefix));
    if (typeof entry !== "string") {
        return null;
    }
    const parsed = Number.parseInt(entry.slice(prefix.length), 10);
    return Number.isFinite(parsed) ? parsed : null;
}
function buildWatchSnapshotAttributionCoverage(activationRoot) {
    const snapshot = toRecord(loadJsonFile(path.join(path.resolve(activationRoot), "watch", "teacher-snapshot.json")));
    const notes = Array.isArray(snapshot?.notes)
        ? snapshot.notes
        : Array.isArray(snapshot?.snapshot?.diagnostics?.notes)
            ? snapshot.snapshot.diagnostics.notes
            : Array.isArray(snapshot?.diagnostics?.notes)
                ? snapshot.diagnostics.notes
                : [];
    const readyCount = readIntegerNote(notes, "teacher_feedback_eligible=");
    const delayedCount = readIntegerNote(notes, "teacher_feedback_delayed=");
    const budgetDeferredCount = readIntegerNote(notes, "teacher_feedback_budgeted_out=");
    const budgetPerTick = readIntegerNote(notes, "teacher_budget=");
    const delayMs = readIntegerNote(notes, "teacher_delay_ms=");
    if (readyCount === null && delayedCount === null && budgetDeferredCount === null && budgetPerTick === null && delayMs === null) {
        return null;
    }
    return {
        visible: true,
        gatingVisible: budgetPerTick !== null || delayMs !== null,
        completedWithoutEvaluationCount: 0,
        readyCount: normalizeCount(readyCount),
        delayedCount: normalizeCount(delayedCount),
        budgetDeferredCount: normalizeCount(budgetDeferredCount),
        detail: `watch sparse-feedback queue: completed_without_evaluation=0, ready=${normalizeCount(readyCount)}, delayed=${normalizeCount(delayedCount)}, budget_deferred=${normalizeCount(budgetDeferredCount)}`
    };
}
function readWatchTeacherSnapshotPackTruth(activationRoot) {
    const snapshot = toRecord(loadJsonFile(path.join(path.resolve(activationRoot), "watch", "teacher-snapshot.json")));
    const learning = toRecord(snapshot?.learning);
    const nestedSnapshot = toRecord(snapshot?.snapshot);
    const nestedLearning = toRecord(nestedSnapshot?.learning);
    return {
        lastHandledMaterializationPackId: normalizeOptionalString(snapshot?.lastHandledMaterializationPackId)
            ?? normalizeOptionalString(learning?.lastHandledMaterializationPackId)
            ?? normalizeOptionalString(nestedLearning?.lastHandledMaterializationPackId),
        lastMaterializationPackId: normalizeOptionalString(snapshot?.lastMaterializationPackId)
            ?? normalizeOptionalString(learning?.lastMaterializationPackId)
            ?? normalizeOptionalString(nestedLearning?.lastMaterializationPackId)
    };
}
function buildActivationPackTruth(activationRoot) {
    const active = resolveActivePackPaths(activationRoot);
    const activePackId = active.packId;
    if (activePackId === null) {
        return null;
    }
    const watchTruth = readWatchTeacherSnapshotPackTruth(activationRoot);
    const handledPackId = watchTruth.lastHandledMaterializationPackId ?? watchTruth.lastMaterializationPackId;
    if (handledPackId === null) {
        return null;
    }
    return {
        activePackId,
        handledPackId,
        materializedPackId: handledPackId,
        promoted: handledPackId === activePackId
    };
}
function shouldPreferActivationFeedbackSummary(current, fallback) {
    if (fallback === null) {
        return false;
    }
    if (current.visible !== true) {
        return true;
    }
    return normalizeCount(current.supervisedTraceCount) === 0 && normalizeCount(fallback.supervisedTraceCount) > 0;
}
function shouldPreferWatchAttributionCoverage(current, fallback) {
    if (fallback === null) {
        return false;
    }
    if (current.visible !== true || current.gatingVisible !== true) {
        return normalizeCount(fallback.readyCount) > 0
            || normalizeCount(fallback.delayedCount) > 0
            || normalizeCount(fallback.budgetDeferredCount) > 0;
    }
    const currentKnown = normalizeCount(current.completedWithoutEvaluationCount)
        + normalizeCount(current.readyCount)
        + normalizeCount(current.delayedCount)
        + normalizeCount(current.budgetDeferredCount);
    const fallbackKnown = normalizeCount(fallback.completedWithoutEvaluationCount)
        + normalizeCount(fallback.readyCount)
        + normalizeCount(fallback.delayedCount)
        + normalizeCount(fallback.budgetDeferredCount);
    return currentKnown === 0 && fallbackKnown > 0;
}
function enrichBridgeWithActivationTruth(activationRoot, bridge) {
    const feedbackSummary = buildActivePackFeedbackSummary(activationRoot);
    const attributionCoverage = buildWatchSnapshotAttributionCoverage(activationRoot);
    const packTruth = buildActivationPackTruth(activationRoot);
    const preferFeedbackSummary = shouldPreferActivationFeedbackSummary(bridge.feedbackSummary, feedbackSummary);
    const preferAttributionCoverage = shouldPreferWatchAttributionCoverage(bridge.attributionCoverage, attributionCoverage);
    const preferPackTruth = packTruth !== null
        && (bridge.materializedPackId === null || (bridge.promoted !== true && packTruth.promoted === true))
        && (packTruth.promoted === true || bridge.materializedPackId === null);
    if (!preferFeedbackSummary && !preferAttributionCoverage && !preferPackTruth) {
        return bridge;
    }
    return normalizeBridgePayload({
        ...bridge,
        materializedPackId: preferPackTruth ? packTruth.materializedPackId : bridge.materializedPackId,
        promoted: preferPackTruth ? packTruth.promoted : bridge.promoted,
        feedbackSummary: preferFeedbackSummary
            ? feedbackSummary
            : bridge.feedbackSummary,
        attributionCoverage: preferAttributionCoverage
            ? attributionCoverage
            : bridge.attributionCoverage,
        source: preferPackTruth
            ? {
                ...(bridge.source ?? {}),
                activationPackTruth: {
                    activePackId: packTruth.activePackId,
                    handledPackId: packTruth.handledPackId,
                    source: "active_pack_plus_watch_snapshot"
                }
            }
            : bridge.source
    });
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
    const routeTraceCount = normalizeCount(payload.routeTraceCount);
    const supervisionCount = normalizeCount(payload.supervisionCount);
    return {
        contract: TRACED_LEARNING_BRIDGE_CONTRACT,
        updatedAt: normalizeOptionalString(payload.updatedAt) ?? new Date().toISOString(),
        routeTraceCount,
        supervisionCount,
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
        feedbackSummary: normalizeFeedbackSummary(payload.feedbackSummary, {
            routeTraceCount,
            supervisedTraceCount: supervisionCount
        }),
        attributionCoverage: normalizeAttributionCoverage(payload.attributionCoverage),
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
    const routeTraceCount = normalizeCount(payload.routeTraceCount);
    const supervisionCount = normalizeCount(payload.supervisionCount);
    return {
        contract: TRACED_LEARNING_STATUS_SURFACE_CONTRACT,
        updatedAt: normalizeOptionalString(payload.updatedAt) ?? new Date().toISOString(),
        routeTraceCount,
        supervisionCount,
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
        feedbackSummary: normalizeFeedbackSummary(payload.feedbackSummary, {
            routeTraceCount,
            supervisedTraceCount: supervisionCount
        }),
        attributionCoverage: normalizeAttributionCoverage(payload.attributionCoverage),
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
        feedbackSummary: defaultFeedbackSummary(),
        attributionCoverage: defaultAttributionCoverage(),
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
        feedbackSummary: summary.feedbackSummary,
        attributionCoverage: summary.attributionCoverage,
        source: {
            command: "brain-store",
            bridge: TRACED_LEARNING_STATUS_SURFACE_BRIDGE,
            brainRoot: context.brainRoot,
            stateDbPath: context.dbPath,
            persistedKey: TRACED_LEARNING_STATUS_SURFACE_STATE_KEY,
            surfacedFrom: summarizeBridgeSource(summary.source)
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
    let feedbackSummary = defaultFeedbackSummary(routeTraceCount, supervisionCount);
    try {
        feedbackSummary = buildDerivedFeedbackSummary(db, routeTraceCount, supervisionCount);
    }
    catch {
        feedbackSummary = defaultFeedbackSummary(routeTraceCount, supervisionCount);
    }
    let attributionCoverage = defaultAttributionCoverage();
    try {
        attributionCoverage = buildDerivedAttributionCoverage(db);
    }
    catch {
        attributionCoverage = defaultAttributionCoverage();
    }
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
        feedbackSummary,
        attributionCoverage,
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
        bridge.feedbackSummary.helpfulCount > 0 ||
        bridge.feedbackSummary.irrelevantCount > 0 ||
        bridge.feedbackSummary.harmfulCount > 0 ||
        bridge.attributionCoverage.completedWithoutEvaluationCount > 0 ||
        bridge.attributionCoverage.readyCount > 0 ||
        bridge.attributionCoverage.delayedCount > 0 ||
        bridge.attributionCoverage.budgetDeferredCount > 0 ||
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
        const existingSummaryLoaded = loadTrainingStateJson(db, TRACED_LEARNING_STATUS_SURFACE_STATE_KEY);
        if (existingSummaryLoaded.value !== null) {
            const existingSummary = normalizePersistedStatusSurface(existingSummaryLoaded.value);
            if (JSON.stringify(existingSummary) === JSON.stringify(summary)) {
                return {
                    path: dbPath,
                    bridge: buildPersistedStatusSurfaceBridge(existingSummary, {
                        brainRoot,
                        dbPath
                    }),
                    persisted: false,
                    error: null
                };
            }
        }
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
        let derived = null;
        try {
            derived = buildDerivedBrainStoreBridge(db, {
                brainRoot,
                dbPath
            }, lastInterruptionSummary);
        }
        catch {
            derived = null;
        }
        const persisted = loadPersistedStatusSurface(db, {
            brainRoot,
            dbPath
        });
        if (persisted.bridge !== null) {
            const bridge = normalizeBridgePayload({
                ...persisted.bridge,
                lastInterruptionSummary: lastInterruptionSummary ?? persisted.bridge.lastInterruptionSummary,
                feedbackSummary: derived?.feedbackSummary ?? persisted.bridge.feedbackSummary,
                attributionCoverage: derived?.attributionCoverage ?? persisted.bridge.attributionCoverage
            });
            return {
                path: dbPath,
                bridge,
                error: null
            };
        }
        const bridge = derived;
        if (bridge === null) {
            return {
                path: dbPath,
                bridge: null,
                error: persisted.error
            };
        }
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
        feedbackSummary: bridge.feedbackSummary,
        attributionCoverage: bridge.attributionCoverage,
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
        source: summarizeBridgeSource(loaded.bridge.source)
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
            feedbackSummary: canonicalBridge.feedbackSummary,
            attributionCoverage: canonicalBridge.attributionCoverage,
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
        feedbackSummary: canonicalBridge.feedbackSummary,
        attributionCoverage: canonicalBridge.attributionCoverage,
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
    const feedbackSummary = current.feedbackSummary.visible ? current.feedbackSummary : persistedBridge.feedbackSummary;
    const attributionCoverage = current.attributionCoverage.visible ? current.attributionCoverage : persistedBridge.attributionCoverage;
    const usedBridge = routeTraceCount !== current.routeTraceCount ||
        supervisionCount !== current.supervisionCount ||
        routerUpdateCount !== current.routerUpdateCount ||
        teacherArtifactCount !== current.teacherArtifactCount ||
        lastInterruptionSummary !== current.lastInterruptionSummary ||
        feedbackSummary.visible !== current.feedbackSummary.visible ||
        attributionCoverage.visible !== current.attributionCoverage.visible ||
        attributionCoverage.gatingVisible !== current.attributionCoverage.gatingVisible;
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
        feedbackSummary,
        attributionCoverage,
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
        return buildStatusSurface(persisted.path, enrichBridgeWithActivationTruth(activationRoot, mergeCanonicalStatusBridge(persisted.bridge, runtime)), {
            runtimeState: describeBridgeRuntimeState(runtime)
        });
    }
    if (runtime.bridge !== null) {
        return buildStatusSurface(runtime.path, enrichBridgeWithActivationTruth(activationRoot, runtime.bridge));
    }
    if (persisted.error !== null) {
        return defaultSurface(persisted.path, "brain_store_unreadable", persisted.error);
    }
    return defaultSurface(runtime.path, runtime.error === null ? "bridge_missing" : "bridge_unreadable", runtime.error);
}
