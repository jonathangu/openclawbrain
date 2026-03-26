import path from "node:path";
import { rankContextBlocks } from "@openclawbrain/compiler";
import { describeSparseFeedbackEventDispositions } from "@openclawbrain/learner";
import { CONTRACT_IDS } from "@openclawbrain/contracts";
import { appendLearningSpineLogEntry, buildLearningSpineLogId, loadActivationPointers, loadPack, loadPackFromActivation, readLearningSpineLogEntries } from "@openclawbrain/pack-format";
function noteValue(notes, prefix) {
    const note = notes.find((entry) => entry.startsWith(prefix));
    return note === undefined ? null : note.slice(prefix.length);
}
function roundNumber(value) {
    return Math.round(value * 10_000) / 10_000;
}
const MAX_PERSISTED_ROUTE_DECISION_CANDIDATE_SET_IDS = 256;
function softmax(values) {
    if (values.length === 0) {
        return [];
    }
    const maxValue = Math.max(...values);
    const numerators = values.map((value) => Math.exp(value - maxValue));
    const denominator = numerators.reduce((sum, value) => sum + value, 0);
    if (denominator === 0) {
        return values.map(() => roundNumber(1 / values.length));
    }
    return numerators.map((value) => roundNumber(value / denominator));
}
function isPlainObject(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return false;
    }
    const prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
}
function compactStructuralSignalValue(value, depth = 0) {
    if (value === null || typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return Number.isFinite(value) ? roundNumber(value) : undefined;
    }
    if (typeof value === "string") {
        return value.length <= 256 ? value : `${value.slice(0, 256)}...`;
    }
    if (depth >= 3) {
        return undefined;
    }
    if (Array.isArray(value)) {
        const compacted = value
            .slice(0, 32)
            .map((entry) => compactStructuralSignalValue(entry, depth + 1))
            .filter((entry) => entry !== undefined);
        return compacted;
    }
    if (!isPlainObject(value)) {
        return undefined;
    }
    const compactedEntries = Object.entries(value)
        .slice(0, 32)
        .flatMap(([key, entry]) => {
        const compacted = compactStructuralSignalValue(entry, depth + 1);
        return compacted === undefined ? [] : [[key, compacted]];
    });
    return compactedEntries.length > 0 ? Object.fromEntries(compactedEntries) : undefined;
}
function compactStructuralSignals(value) {
    const compacted = compactStructuralSignalValue(value);
    return compacted !== undefined && isPlainObject(compacted) ? compacted : null;
}
function isStableKernelContextBlock(block) {
    if (block.id.includes(":event:") || block.id.includes(":teacher:")) {
        return false;
    }
    if (block.source.startsWith("split:") || block.source.startsWith("merge:")) {
        return false;
    }
    return true;
}
function buildActivationPackSnapshot(rootDir, slot) {
    const pointers = loadActivationPointers(rootDir).pointers;
    const record = pointers[slot];
    if (record === null) {
        return null;
    }
    let graphChecksum = null;
    let routerChecksum = null;
    let routerIdentity = record.routerIdentity;
    try {
        const pack = loadPack(path.resolve(record.packRootDir));
        graphChecksum = pack.manifest.payloadChecksums.graph;
        routerChecksum = pack.manifest.payloadChecksums.router;
        routerIdentity = pack.router?.routerIdentity ?? record.routerIdentity;
    }
    catch {
        graphChecksum = null;
        routerChecksum = null;
    }
    return {
        slot,
        packId: record.packId,
        routePolicy: record.routePolicy,
        routerIdentity,
        routerChecksum,
        graphChecksum,
        eventExportDigest: record.eventExportDigest,
        builtAt: record.builtAt,
        updatedAt: record.updatedAt
    };
}
function buildActivationPointersSnapshot(rootDir) {
    return {
        active: buildActivationPackSnapshot(rootDir, "active"),
        candidate: buildActivationPackSnapshot(rootDir, "candidate"),
        previous: buildActivationPackSnapshot(rootDir, "previous")
    };
}
function rewardForEvent(event) {
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
        switch (event.kind) {
            case "correction":
                return 4;
            case "teaching":
                return 3;
            case "approval":
                return 2;
            case "suppression":
                return -3;
        }
    }
    switch (event.kind) {
        case "operator_override":
            return 4;
        case "memory_compiled":
            return 2;
        default:
            return 0;
    }
}
function defaultTargetBlockIds(packId, event) {
    const targetBlockIds = new Set([`${packId}:event:${event.eventId}`]);
    if (event.contract === CONTRACT_IDS.feedbackEvents && event.relatedInteractionId !== undefined) {
        targetBlockIds.add(`${packId}:event:${event.relatedInteractionId}`);
    }
    return [...targetBlockIds].sort((left, right) => left.localeCompare(right));
}
function collectBoundEdgeIds(pack, targetIds) {
    const targetSet = new Set(targetIds);
    const edgeIds = new Set();
    for (const block of pack.blocks) {
        if (!targetSet.has(block.id)) {
            continue;
        }
        for (const edge of block.edges ?? []) {
            if (!targetSet.has(edge.targetBlockId)) {
                continue;
            }
            edgeIds.add(`${block.id}->${edge.targetBlockId}:${edge.kind}`);
        }
    }
    return [...edgeIds].sort((left, right) => left.localeCompare(right));
}
function findPromotionLink(activationRoot, activePackId, routerChecksum) {
    if (activePackId === null) {
        return null;
    }
    const entries = readLearningSpineLogEntries(activationRoot, "promotionActivationEvents");
    const matched = [...entries].reverse().find((entry) => {
        if (entry.operation !== "promote_candidate_pack") {
            return false;
        }
        if (entry.after.active?.packId !== activePackId) {
            return false;
        }
        return routerChecksum === null || entry.after.active?.routerChecksum === routerChecksum;
    });
    return matched === undefined
        ? null
        : {
            matchedPromotionEventId: matched.recordId,
            servedAfterPromotion: true,
            matchedPackId: matched.after.active?.packId ?? null,
            matchedRouterChecksum: matched.after.active?.routerChecksum ?? null
        };
}
function appendServedAfterPromotionLog(input) {
    if (input.promotionLink?.servedAfterPromotion !== true) {
        return;
    }
    const snapshot = buildActivationPointersSnapshot(input.activationRoot);
    const entryBase = {
        recordType: "promotion_activation",
        operation: "served_after_promotion",
        occurredAt: input.recordedAt,
        activationRoot: path.resolve(input.activationRoot),
        reason: "later_served_turn_used_promoted_pack",
        promotionReason: null,
        rollbackTarget: null,
        before: snapshot,
        after: snapshot,
        laterServedTurn: {
            turnCompileEventId: input.compileEventId,
            sessionId: input.sessionId,
            channel: input.channel,
            servedAt: input.recordedAt,
            servedArtifact: input.servedArtifact,
            servedPackId: input.activePackId,
            servedRouterChecksum: input.routerChecksum,
            matchedPromotionEventId: input.promotionLink.matchedPromotionEventId,
            switchedToPromotedPack: true
        }
    };
    const entry = {
        ...entryBase,
        recordId: buildLearningSpineLogId("promotion-activation", entryBase)
    };
    appendLearningSpineLogEntry(input.activationRoot, "promotionActivationEvents", entry);
}
function serveFallbackReason(result) {
    if (!result.ok) {
        return result.error;
    }
    const notes = result.compileResponse.diagnostics.notes;
    const enforced = noteValue(notes, "learned_required_enforced=");
    if (enforced !== null) {
        return enforced;
    }
    const selectionMode = noteValue(notes, "selection_mode=");
    if (selectionMode === "priority_fallback") {
        return "priority_fallback";
    }
    const selectionTiers = noteValue(notes, "selection_tiers=");
    if (selectionTiers === "token_match+priority_fallback") {
        return selectionTiers;
    }
    return result.compileResponse.diagnostics.usedLearnedRouteFn ? null : `mode_effective=${result.compileResponse.diagnostics.modeEffective}`;
}
export function appendServeTimeRouteDecisionLog(input) {
    const activationRoot = path.resolve(input.activationRoot);
    const compileEvent = input.normalizedEventExport?.interactionEvents.find((event) => event.kind === "memory_compiled") ?? null;
    const activePack = input.compileResult.ok ? loadPackFromActivation(activationRoot, "active") : null;
    const resolvedMaxContextBlocks = input.compileResult.ok
        ? Number.parseInt(noteValue(input.compileResult.compileResponse.diagnostics.notes, "resolved_max_context_blocks=") ?? "", 10)
        : Number.NaN;
    const effectiveMaxContextBlocks = Number.isFinite(resolvedMaxContextBlocks)
        ? resolvedMaxContextBlocks
        : (input.turn.maxContextBlocks ?? 4);
    const ranked = input.compileResult.ok && activePack !== null
        ? rankContextBlocks(activePack, {
            contract: CONTRACT_IDS.runtimeCompile,
            agentId: "openclaw-learning-spine-log",
            userMessage: input.turn.userMessage,
            maxContextBlocks: effectiveMaxContextBlocks,
            modeRequested: input.turn.mode ?? "heuristic",
            ...(input.turn.runtimeHints !== undefined ? { runtimeHints: [...input.turn.runtimeHints] } : {})
        })
        : [];
    const probabilities = softmax(ranked.map((entry) => entry.score));
    const selectedIds = new Set(input.compileResult.ok ? input.compileResult.compileResponse.selectedContext.map((block) => block.id) : []);
    const selectedKernelContextIds = input.compileResult.ok
        ? input.compileResult.compileResponse.selectedContext.filter((block) => isStableKernelContextBlock(block)).map((block) => block.id)
        : [];
    const selectedBrainContextIds = input.compileResult.ok
        ? input.compileResult.compileResponse.selectedContext.filter((block) => !isStableKernelContextBlock(block)).map((block) => block.id)
        : [];
    const promotionLink = findPromotionLink(activationRoot, input.compileResult.ok ? input.compileResult.activePackId : null, activePack?.manifest.payloadChecksums.router ?? null);
    const entryBase = {
        recordType: "serve_time_route_decision",
        recordedAt: input.recordedAt,
        activationRoot,
        breadcrumbs: input.breadcrumbs ?? {
            entrypoint: "compileRuntimeContext",
            invocationSurface: "direct_compile_call",
            hostEvent: null,
            installedEntryPath: null,
            syntheticTurn: true
        },
        sessionId: input.turn.sessionId,
        channel: input.turn.channel,
        userMessage: input.turn.userMessage,
        turnSequenceStart: input.turn.sequenceStart ?? null,
        turnCompileEventId: compileEvent?.eventId ?? null,
        turnCreatedAt: compileEvent?.createdAt ?? input.turn.compile?.createdAt ?? input.turn.createdAt ?? null,
        activePackId: input.compileResult.ok ? input.compileResult.activePackId : null,
        activePackBuiltAt: activePack?.manifest.provenance.builtAt ?? null,
        activePackEventExportDigest: activePack?.manifest.provenance.eventExports?.exportDigest ?? null,
        activePackRouterChecksum: activePack?.manifest.payloadChecksums.router ?? null,
        activePackGraphChecksum: activePack?.manifest.payloadChecksums.graph ?? null,
        routerIdentity: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.routerIdentity : null,
        usedLearnedRouteFn: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.usedLearnedRouteFn : null,
        servedArtifact: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.servedArtifact : null,
        selectionDigest: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.selectionDigest : null,
        requestedBudget: {
            modeRequested: input.turn.mode ?? "heuristic",
            maxContextBlocks: effectiveMaxContextBlocks,
            maxContextChars: input.turn.maxContextChars ?? null
        },
        actualBudget: {
            modeEffective: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.modeEffective : null,
            selectedCount: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.selectedCount : 0,
            selectedCharCount: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.selectedCharCount : 0,
            selectedTokenCount: input.compileResult.ok ? input.compileResult.compileResponse.diagnostics.selectedTokenCount : 0
        },
        candidateSetIds: ranked.slice(0, MAX_PERSISTED_ROUTE_DECISION_CANDIDATE_SET_IDS).map((entry) => entry.blockId),
        chosenContextIds: input.compileResult.ok ? input.compileResult.compileResponse.selectedContext.map((block) => block.id) : [],
        candidateScores: ranked.map((entry, index) => ({
            blockId: entry.blockId,
            selected: selectedIds.has(entry.blockId),
            actionScore: roundNumber(entry.score),
            actionProbability: probabilities[index] ?? 0
        })),
        structuralSignals: compactStructuralSignals(input.compileResult.ok ? input.compileResult.compileResponse.structuralSignals : null),
        fallbackReason: serveFallbackReason(input.compileResult),
        hotPathTiming: input.compileResult.timing,
        kernelContextCount: selectedKernelContextIds.length,
        brainContextCount: selectedBrainContextIds.length,
        selectedKernelContextIds,
        selectedBrainContextIds,
        promotionLink
    };
    const entry = {
        ...entryBase,
        recordId: buildLearningSpineLogId("serve-route-decision", entryBase)
    };
    appendLearningSpineLogEntry(activationRoot, "serveTimeRouteDecisions", entry);
    appendServedAfterPromotionLog({
        activationRoot,
        recordedAt: input.recordedAt,
        compileEventId: compileEvent?.eventId ?? null,
        sessionId: input.turn.sessionId,
        channel: input.turn.channel,
        servedArtifact: entry.servedArtifact,
        activePackId: entry.activePackId,
        routerChecksum: entry.activePackRouterChecksum,
        promotionLink
    });
    return entry;
}
function eventSourceCounts(materialization) {
    const counts = {
        routeTrace: 0,
        humanFeedback: 0,
        operatorOverride: 0,
        selfMemory: 0,
        teacherSupervision: 0
    };
    for (const trace of materialization?.traces ?? []) {
        switch (trace.supervisionKind) {
            case "route_trace":
                counts.routeTrace += 1;
                break;
            case "human_feedback":
                counts.humanFeedback += 1;
                break;
            case "operator_override":
                counts.operatorOverride += 1;
                break;
            case "self_memory":
                counts.selfMemory += 1;
                break;
            case "teacher_supervision":
                counts.teacherSupervision += 1;
                break;
        }
    }
    return counts;
}
export function appendLearningUpdateLogs(input) {
    const activationRoot = path.resolve(input.activationRoot);
    const candidate = input.materialization.candidate;
    const router = candidate.payloads.router;
    const packId = candidate.summary.packId;
    const recordedAt = candidate.manifest.provenance.builtAt;
    const updateKey = {
        activationRoot,
        previousPackId: input.activeBeforePack?.manifest.packId ?? null,
        nextPackId: packId,
        nextRouterChecksum: candidate.manifest.payloadChecksums.router,
        recordedAt
    };
    const updateId = buildLearningSpineLogId("pg-update", updateKey);
    const graph = candidate.payloads.graph;
    const blockIds = new Set(graph.blocks.map((block) => block.id));
    const tracesBySourceEventId = new Map((router?.traces ?? []).map((trace) => [trace.sourceEventId, trace]));
    const feedbackDispositions = describeSparseFeedbackEventDispositions(input.materialization.normalizedEventExport.feedbackEvents, recordedAt, input.materialization.candidateInput.sparseFeedback);
    const feedbackDispositionById = new Map(feedbackDispositions.map((entry) => [entry.eventId, entry]));
    const labelEvents = [
        ...input.materialization.normalizedEventExport.feedbackEvents,
        ...input.materialization.normalizedEventExport.interactionEvents.filter((event) => event.kind === "operator_override" || event.kind === "memory_compiled")
    ];
    const bindings = labelEvents.map((event) => {
        const trace = tracesBySourceEventId.get(event.eventId) ?? null;
        const targetIds = trace?.targetBlockIds ?? defaultTargetBlockIds(packId, event);
        const boundBlockIds = targetIds.filter((targetId) => blockIds.has(targetId));
        const discardReason = trace !== null
            ? null
            : router === null
                ? "learned_routing_disabled"
                : event.contract === CONTRACT_IDS.feedbackEvents
                    ? feedbackDispositionById.get(event.eventId)?.reason ?? "excluded_from_pg_route_update"
                    : "excluded_from_pg_route_update";
        const entryBase = {
            recordType: "supervision_label_binding",
            recordedAt,
            activationRoot,
            updateId,
            packId,
            routerIdentity: router?.routerIdentity ?? null,
            eventExportDigest: candidate.summary.eventExportDigest,
            sourceEventId: event.eventId,
            feedbackEventId: event.contract === CONTRACT_IDS.feedbackEvents ? event.eventId : null,
            linkedTurnId: event.contract === CONTRACT_IDS.feedbackEvents ? event.relatedInteractionId ?? null : event.eventId,
            sourceContract: event.contract,
            sourceKind: event.kind,
            supervisionKind: trace?.supervisionKind ??
                (event.contract === CONTRACT_IDS.feedbackEvents
                    ? "human_feedback"
                    : event.kind === "operator_override"
                        ? "operator_override"
                        : "self_memory"),
            rewardAssigned: trace?.reward ?? rewardForEvent(event),
            boundBlockIds,
            boundEdgeIds: collectBoundEdgeIds(graph, boundBlockIds),
            boundTargetIds: targetIds,
            dedupOrDiscardReason: discardReason
        };
        return {
            ...entryBase,
            recordId: buildLearningSpineLogId("supervision-binding", entryBase)
        };
    });
    for (const binding of bindings) {
        appendLearningSpineLogEntry(activationRoot, "supervisionLabelBindings", binding);
    }
    const pgBase = {
        recordType: "pg_route_update",
        updateId,
        recordedAt,
        activationRoot,
        previousPackId: input.activeBeforePack?.manifest.packId ?? null,
        nextPackId: packId,
        previousRouterIdentity: input.activeBeforePack?.router?.routerIdentity ?? null,
        nextRouterIdentity: router?.routerIdentity ?? null,
        previousRouterChecksum: input.activeBeforePack?.manifest.payloadChecksums.router ?? null,
        nextRouterChecksum: candidate.manifest.payloadChecksums.router,
        previousUpdateCount: input.activeBeforePack?.router?.training.updateCount ?? null,
        nextUpdateCount: router?.training.updateCount ?? null,
        updateCountIncrement: input.activeBeforePack?.router?.training.updateCount === undefined || router?.training.updateCount === undefined
            ? null
            : router.training.updateCount - input.activeBeforePack.router.training.updateCount,
        routeTraceCount: router?.training.routeTraceCount ?? 0,
        supervisionCount: router?.training.supervisionCount ?? 0,
        labelSourceCounts: router?.training.collectedLabels ?? {
            total: 0,
            humanFeedback: 0,
            operatorOverride: 0,
            selfMemory: 0
        },
        sourceCounts: eventSourceCounts(router),
        updateOutcome: router === null ? "disabled" : router.training.status,
        noOpReason: router === null ? "learned routing disabled for candidate build" : router.training.noOpReason,
        objectiveSummary: {
            updateMechanism: router?.training.objective.updateMechanism ?? null,
            updateVersion: router?.training.objective.updateVersion ?? null,
            objective: router?.training.objective.objective ?? null,
            objectiveChecksum: router?.training.objective.objectiveChecksum ?? null,
            profile: router?.training.objective.profile ?? null,
            materializedLoss: false,
            lossSummary: router === null
                ? "learned routing disabled for candidate build"
                : "candidate pack materializes policy-gradient deltas and checksums; scalar loss is not stored"
        },
        resultingArtifacts: {
            packId,
            routerIdentity: router?.routerIdentity ?? null,
            manifestPath: input.candidateDescriptor.manifestPath,
            routerPath: input.candidateDescriptor.routerPath,
            weightsChecksum: router?.training.weightsChecksum ?? null,
            freshnessChecksum: router?.training.freshnessChecksum ?? null,
            eventExportDigest: candidate.summary.eventExportDigest
        }
    };
    const pgRouteUpdate = {
        ...pgBase,
        recordId: buildLearningSpineLogId("pg-route-update", pgBase)
    };
    appendLearningSpineLogEntry(activationRoot, "pgRouteUpdates", pgRouteUpdate);
    return {
        updateId,
        bindingCount: bindings.length,
        pgRouteUpdate
    };
}
//# sourceMappingURL=learning-spine.js.map
