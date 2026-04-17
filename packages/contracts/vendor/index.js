import { createHash } from "node:crypto";
export const CONTRACT_IDS = {
    activationPointers: "activation_pointers.v1",
    artifactManifest: "artifact_manifest.v1",
    brainAttachmentPolicy: "brain_attachment_policy.v1",
    currentProfileBrainStatus: "current_profile_brain_status.v1",
    feedbackEvents: "feedback_events.v1",
    interactionEvents: "interaction_events.v1",
    profileTurnAttribution: "profile_turn_attribution.v1",
    runtimeCompile: "runtime_compile.v1",
    teacherSupervisionArtifact: "teacher_supervision_artifact.v1"
};
export const PACK_GRAPH_SCHEMAS = {
    openclawInit: "openclaw_init_graph.v1"
};
export const ROUTER_PG_PROFILE_V1 = {
    traceSource: "event_reconstruction",
    actionSpace: "pack_block_softmax",
    targetConstruction: "event_block_plus_related_interaction",
    rewardSignal: "explicit_label_reward_table_v1",
    baseline: "none",
    offPolicyCorrection: "none",
    updateCadence: "candidate_pack_refresh"
};
export const ROUTER_PG_PROFILE_V2 = {
    traceSource: "serve_time_decision_log",
    actionSpace: "graph_local_neighbor_softmax",
    targetConstruction: "trajectory_reconstruction",
    rewardSignal: "explicit_label_reward_table_v1",
    baseline: "exponential_moving_average",
    offPolicyCorrection: "none",
    updateCadence: "candidate_pack_refresh",
    trajectorySource: "serve_decision_reconstruction"
};
function isIsoDate(value) {
    return !Number.isNaN(Date.parse(value));
}
function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(value, key);
}
function pushWhenMissing(errors, condition, message) {
    if (!condition) {
        errors.push(message);
    }
}
function uniqueInOrder(values) {
    const seen = new Set();
    const result = [];
    for (const value of values) {
        if (seen.has(value)) {
            continue;
        }
        seen.add(value);
        result.push(value);
    }
    return result;
}
export function describeRetrievalSemanticClass(value) {
    switch (value) {
        case "answer_bearing":
            return {
                semanticClass: value,
                answerRole: "answer_bearing",
                instructionRole: "non_scaffolding",
                detail: "Carries answer-domain content that should compete as primary retrieval material."
            };
        case "instructional_scaffolding":
            return {
                semanticClass: value,
                answerRole: "non_answer_bearing",
                instructionRole: "instructional_scaffolding",
                detail: "Carries meta-instruction, prompting, or questionnaire scaffolding rather than the underlying answer payload."
            };
        case "observability_support":
            return {
                semanticClass: value,
                answerRole: "non_answer_bearing",
                instructionRole: "non_scaffolding",
                detail: "Carries operator or diagnostic support that is useful for observability asks but is not answer-bearing domain content."
            };
        case "transport_residue":
            return {
                semanticClass: value,
                answerRole: "non_answer_bearing",
                instructionRole: "non_scaffolding",
                detail: "Carries transport or delivery residue that should normally stay out of semantic answer retrieval."
            };
    }
}
export function buildEventSemanticSurface(events) {
    const semanticTypes = [];
    const sourceKinds = [];
    const diagnosticIntents = [];
    for (const event of events) {
        if (event.semantic === undefined) {
            continue;
        }
        semanticTypes.push(event.semantic.semanticType);
        sourceKinds.push(event.semantic.sourceKind);
        if (event.semantic.diagnosticIntent !== undefined) {
            diagnosticIntents.push(event.semantic.diagnosticIntent);
        }
    }
    return {
        semanticTypes: uniqueInOrder(semanticTypes),
        sourceKinds: uniqueInOrder(sourceKinds),
        diagnosticIntents: uniqueInOrder(diagnosticIntents)
    };
}
function eventSequenceErrors(events) {
    const errors = [];
    let previous = null;
    for (const event of sortNormalizedEvents(events)) {
        if (previous !== null) {
            if (event.sequence === previous.sequence && event.eventId === previous.eventId) {
                errors.push(`duplicate normalized event identity: ${event.contract}:${event.eventId}`);
            }
            if (event.sequence < previous.sequence) {
                errors.push(`normalized events must be sorted by sequence: ${event.eventId}`);
            }
            if (event.sequence === previous.sequence && event.createdAt < previous.createdAt) {
                errors.push(`normalized events with sequence ${event.sequence} must be ordered by createdAt`);
            }
        }
        previous = event;
    }
    return errors;
}
export function createDefaultLearningSurface(scanSurfaces = ["workspace_snapshot"]) {
    const surfaces = uniqueInOrder(scanSurfaces.map((surface) => surface.trim()).filter((surface) => surface.length > 0));
    return {
        bootProfile: "fast_boot_defaults",
        learningCadence: "passive_background",
        scanPolicy: "always_on",
        scanSurfaces: surfaces.length === 0 ? ["workspace_snapshot"] : surfaces,
        labelSources: {
            human: [CONTRACT_IDS.feedbackEvents, `${CONTRACT_IDS.interactionEvents}:operator_override`],
            self: [`${CONTRACT_IDS.interactionEvents}:memory_compiled`]
        },
        labelHarvest: {
            humanLabels: 0,
            selfLabels: 0,
            corrections: 0,
            teachings: 0,
            approvals: 0,
            suppressions: 0,
            operatorOverrideLabels: 0,
            memoryCompileLabels: 0
        },
        principalSummary: {
            teacherIdentities: [],
            teacherRoles: [],
            teacherAuthorities: [],
            priorityClasses: [],
            scopedEventCount: 0,
            supersedingEventCount: 0
        }
    };
}
export function buildLearningSurface(events) {
    if (events.length === 0) {
        return createDefaultLearningSurface(["event_export:empty"]);
    }
    const sorted = sortNormalizedEvents(events);
    const scanSurfaces = uniqueInOrder(sorted.map((event) => `${event.source.stream}:${event.kind}`));
    const humanSources = [];
    const selfSources = [];
    const teacherIdentities = [];
    const teacherRoles = [];
    const teacherAuthorities = [];
    const priorityClasses = [];
    let corrections = 0;
    let teachings = 0;
    let approvals = 0;
    let suppressions = 0;
    let operatorOverrideLabels = 0;
    let memoryCompileLabels = 0;
    let scopedEventCount = 0;
    let supersedingEventCount = 0;
    for (const event of sorted) {
        const surface = `${event.source.stream}:${event.kind}`;
        if (event.principal !== undefined) {
            teacherIdentities.push(event.principal.teacherIdentity);
            teacherRoles.push(event.principal.teacherRole);
            teacherAuthorities.push(event.principal.teacherAuthority);
            priorityClasses.push(event.principal.priorityClass);
            if (event.principal.principalScope.kind !== "global") {
                scopedEventCount += 1;
            }
            if ((event.principal.supersedes?.length ?? 0) > 0) {
                supersedingEventCount += 1;
            }
        }
        if (event.contract === CONTRACT_IDS.feedbackEvents) {
            humanSources.push(surface);
            switch (event.kind) {
                case "correction":
                    corrections += 1;
                    break;
                case "teaching":
                    teachings += 1;
                    break;
                case "approval":
                    approvals += 1;
                    break;
                case "suppression":
                    suppressions += 1;
                    break;
            }
            continue;
        }
        if (event.kind === "operator_override") {
            humanSources.push(surface);
            operatorOverrideLabels += 1;
            continue;
        }
        if (event.kind === "memory_compiled") {
            selfSources.push(surface);
            memoryCompileLabels += 1;
        }
    }
    return {
        bootProfile: "fast_boot_defaults",
        learningCadence: "passive_background",
        scanPolicy: "always_on",
        scanSurfaces,
        labelSources: {
            human: uniqueInOrder(humanSources),
            self: uniqueInOrder(selfSources)
        },
        labelHarvest: {
            humanLabels: corrections + teachings + approvals + suppressions + operatorOverrideLabels,
            selfLabels: memoryCompileLabels,
            corrections,
            teachings,
            approvals,
            suppressions,
            operatorOverrideLabels,
            memoryCompileLabels
        },
        principalSummary: {
            teacherIdentities: uniqueInOrder(teacherIdentities),
            teacherRoles: uniqueInOrder(teacherRoles),
            teacherAuthorities: uniqueInOrder(teacherAuthorities),
            priorityClasses: uniqueInOrder(priorityClasses),
            scopedEventCount,
            supersedingEventCount
        }
    };
}
export function canonicalJson(value) {
    return `${JSON.stringify(value, null, 2)}\n`;
}
export function checksumJsonPayload(value) {
    return `sha256-${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}
export function computeRouterQueryChecksum(traces) {
    return checksumJsonPayload(traces.map((trace) => ({
        traceId: trace.traceId,
        sourceEventId: trace.sourceEventId,
        sourceContract: trace.sourceContract,
        sourceKind: trace.sourceKind,
        supervisionKind: trace.supervisionKind,
        targetBlockIds: [...trace.targetBlockIds],
        reward: trace.reward,
        queryTokens: [...trace.queryTokens],
        queryVector: trace.queryVector
    })));
}
export function computeRouterWeightsChecksum(policyUpdates) {
    return checksumJsonPayload(policyUpdates.map((update) => ({
        blockId: update.blockId,
        delta: update.delta,
        evidenceCount: update.evidenceCount,
        rewardSum: update.rewardSum,
        tokenWeights: update.tokenWeights,
        traceIds: [...update.traceIds]
    })));
}
export function computeRouterCollectedLabelCounts(traces) {
    let humanFeedback = 0;
    let operatorOverride = 0;
    let selfMemory = 0;
    for (const trace of traces) {
        if (trace.reward === 0) {
            continue;
        }
        switch (trace.supervisionKind) {
            case "human_feedback":
                humanFeedback += 1;
                break;
            case "operator_override":
                operatorOverride += 1;
                break;
            case "self_memory":
                selfMemory += 1;
                break;
        }
    }
    return {
        total: humanFeedback + operatorOverride + selfMemory,
        humanFeedback,
        operatorOverride,
        selfMemory
    };
}
export function computeRouterObjectiveChecksum(input) {
    return checksumJsonPayload({
        updateMechanism: input.updateMechanism,
        updateVersion: input.updateVersion,
        objective: input.objective,
        profile: input.profile,
        eventExportDigest: input.eventExportDigest,
        routeTraceCount: input.routeTraceCount,
        supervisionCount: input.supervisionCount,
        collectedLabels: input.collectedLabels,
        queryChecksum: input.queryChecksum
    });
}
export function computeRouterFreshnessChecksum(input) {
    return checksumJsonPayload({
        method: input.method,
        trainedAt: input.trainedAt,
        status: input.status,
        eventExportDigest: input.eventExportDigest,
        routeTraceCount: input.routeTraceCount,
        supervisionCount: input.supervisionCount,
        updateCount: input.updateCount
    });
}
export function buildRouteArtifactReference(input) {
    return {
        assetKind: input.routerAssetKind,
        routerIdentity: input.routerIdentity,
        routerChecksum: input.routerChecksum,
        routeFnVersion: input.router?.strategy ?? null,
        trainingMethod: input.router?.training.method ?? null,
        trainedAt: input.router?.trainedAt ?? null,
        eventExportDigest: input.router?.training.eventExportDigest ?? input.eventExportDigest ?? null,
        updateCount: input.router?.training.updateCount ?? null,
        objective: input.router?.training.objective.objective ?? null,
        objectiveChecksum: input.router?.training.objective.objectiveChecksum ?? null,
        freshnessChecksum: input.router?.training.freshnessChecksum ?? null
    };
}
export function buildServedArtifactProof(target, routeArtifact) {
    return {
        packId: target.packId,
        routePolicy: target.routePolicy,
        workspaceSnapshot: target.workspaceSnapshot,
        workspaceRevision: target.workspaceRevision,
        eventRange: {
            start: target.eventRange.start,
            end: target.eventRange.end,
            count: target.eventRange.count
        },
        eventExportDigest: target.eventExportDigest,
        builtAt: target.builtAt,
        routeArtifact: {
            ...routeArtifact
        }
    };
}
export function createInteractionEvent(value) {
    return {
        contract: CONTRACT_IDS.interactionEvents,
        ...value
    };
}
export function createFeedbackEvent(value) {
    return {
        contract: CONTRACT_IDS.feedbackEvents,
        ...value
    };
}
export function sortNormalizedEvents(events) {
    return [...events].sort((left, right) => {
        if (left.sequence !== right.sequence) {
            return left.sequence - right.sequence;
        }
        if (left.createdAt !== right.createdAt) {
            return left.createdAt.localeCompare(right.createdAt);
        }
        if (left.contract !== right.contract) {
            return left.contract.localeCompare(right.contract);
        }
        return left.eventId.localeCompare(right.eventId);
    });
}
export function createExplicitEventRange(value) {
    return {
        start: value.start,
        end: value.end,
        count: value.end >= value.start ? value.end - value.start + 1 : 0,
        firstEventId: null,
        lastEventId: null,
        firstCreatedAt: null,
        lastCreatedAt: null
    };
}
export function buildNormalizedEventRange(events) {
    const sorted = sortNormalizedEvents(events);
    if (sorted.length === 0) {
        return createExplicitEventRange({
            start: 0,
            end: -1
        });
    }
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    // Derive firstCreatedAt/lastCreatedAt from the chronological min/max across
    // all events, not just the first/last by sequence order. Sequence numbers
    // and timestamps can diverge when events span multiple session stores or
    // when quality filters remove intermediate events.
    let minCreatedAt = first.createdAt;
    let maxCreatedAt = first.createdAt;
    for (const event of sorted) {
        if (event.createdAt < minCreatedAt)
            minCreatedAt = event.createdAt;
        if (event.createdAt > maxCreatedAt)
            maxCreatedAt = event.createdAt;
    }
    return {
        start: first.sequence,
        end: last.sequence,
        count: sorted.length,
        firstEventId: first.eventId,
        lastEventId: last.eventId,
        firstCreatedAt: minCreatedAt,
        lastCreatedAt: maxCreatedAt
    };
}
export function buildNormalizedEventExport(value) {
    const interactionEvents = [...value.interactionEvents];
    const feedbackEvents = [...value.feedbackEvents];
    const events = sortNormalizedEvents([...interactionEvents, ...feedbackEvents]);
    const contracts = uniqueInOrder(events.map((event) => event.contract));
    const sessionIds = uniqueInOrder(events.map((event) => event.sessionId));
    const channels = uniqueInOrder(events.map((event) => event.channel));
    const sourceStreams = uniqueInOrder(events.map((event) => event.source.stream));
    return {
        interactionEvents,
        feedbackEvents,
        range: buildNormalizedEventRange(events),
        provenance: {
            runtimeOwner: "openclaw",
            sessionId: sessionIds.length === 1 ? (sessionIds[0] ?? null) : null,
            channel: channels.length === 1 ? (channels[0] ?? null) : null,
            interactionCount: interactionEvents.length,
            feedbackCount: feedbackEvents.length,
            sourceStreams,
            contracts,
            exportDigest: checksumJsonPayload({
                interactionEvents: sortNormalizedEvents(interactionEvents),
                feedbackEvents: sortNormalizedEvents(feedbackEvents)
            }),
            semanticSurface: buildEventSemanticSurface(events),
            learningSurface: buildLearningSurface(events)
        }
    };
}
function validateRuntimeContextBlock(value, label) {
    const errors = [];
    pushWhenMissing(errors, value.id.length > 0, `${label} id is required`);
    pushWhenMissing(errors, value.source.length > 0, `${label} source is required`);
    pushWhenMissing(errors, value.text.length > 0, `${label} text is required`);
    if (value.tokenCount !== undefined) {
        pushWhenMissing(errors, value.tokenCount >= 0, `${label} tokenCount must be non-negative`);
    }
    if (value.compactedFrom !== undefined) {
        pushWhenMissing(errors, value.compactedFrom.length > 0, `${label} compactedFrom must not be empty`);
        const uniqueIds = new Set(value.compactedFrom.filter((id) => id.length > 0));
        if (uniqueIds.size !== value.compactedFrom.length) {
            errors.push(`${label} compactedFrom must contain unique non-empty ids`);
        }
    }
    return errors;
}
function validatePackGraphEdge(value, label) {
    const errors = [];
    pushWhenMissing(errors, value.targetBlockId.length > 0, `${label} targetBlockId is required`);
    pushWhenMissing(errors, value.kind === "split" || value.kind === "merge" || value.kind === "connect" || value.kind === "feedback", `${label} kind must be split, merge, connect, or feedback`);
    pushWhenMissing(errors, Number.isFinite(value.weight) && value.weight >= 0, `${label} weight must be non-negative`);
    return errors;
}
function validatePackBlockState(value, label) {
    const errors = [];
    pushWhenMissing(errors, Number.isFinite(value.strength) && value.strength >= 0, `${label} strength must be non-negative`);
    pushWhenMissing(errors, Number.isFinite(value.freshness) && value.freshness >= 0 && value.freshness <= 1, `${label} freshness must be between 0 and 1`);
    pushWhenMissing(errors, Number.isFinite(value.traversalBias) && value.traversalBias >= 0, `${label} traversalBias must be non-negative`);
    pushWhenMissing(errors, Number.isInteger(value.evidenceCount) && value.evidenceCount >= 0, `${label} evidenceCount must be a non-negative integer`);
    pushWhenMissing(errors, Number.isInteger(value.splitDepth) && value.splitDepth >= 0, `${label} splitDepth must be a non-negative integer`);
    pushWhenMissing(errors, Number.isInteger(value.mergedFromCount) && value.mergedFromCount >= 0, `${label} mergedFromCount must be a non-negative integer`);
    return errors;
}
function validatePackBlockRoutingHints(value, label) {
    const errors = [];
    pushWhenMissing(errors, value.channels.length > 0, `${label} channels must not be empty`);
    const uniqueChannels = new Set(value.channels);
    if (uniqueChannels.size !== value.channels.length) {
        errors.push(`${label} channels must be unique`);
    }
    for (const channel of value.channels) {
        pushWhenMissing(errors, channel === "graph" || channel === "short_term" || channel === "vector", `${label} channel must be graph, short_term, or vector`);
    }
    for (const [name, numericValue] of [
        ["graphBias", value.graphBias],
        ["shortTermBias", value.shortTermBias],
        ["vectorBias", value.vectorBias],
        ["backgroundLabelAmplification", value.backgroundLabelAmplification]
    ]) {
        if (numericValue === undefined) {
            continue;
        }
        pushWhenMissing(errors, Number.isFinite(numericValue), `${label} ${name} must be finite when set`);
        if (name === "backgroundLabelAmplification") {
            pushWhenMissing(errors, numericValue >= 1, `${label} backgroundLabelAmplification must be >= 1 when set`);
        }
    }
    return errors;
}
function validatePackBlockInitScoreBreakdown(value, label) {
    const errors = [];
    for (const [name, numericValue] of Object.entries(value)) {
        pushWhenMissing(errors, Number.isFinite(numericValue), `${label} ${name} must be finite`);
        if (name !== "total") {
            pushWhenMissing(errors, numericValue >= 0, `${label} ${name} must be non-negative`);
        }
    }
    return errors;
}
function validatePackBlockInitSignals(value, label) {
    const errors = [];
    const allowedNodeTypes = ["file", "section", "task", "rule", "person", "project", "pointer", "event", "entity"];
    const allowedFileRoles = [
        "anchor",
        "working_set",
        "pointer_index",
        "recent_memory",
        "archived_memory",
        "correction_log",
        "reference",
        "workspace",
        "event_stream",
        "synthetic"
    ];
    pushWhenMissing(errors, value.mode === "heuristic_seed_v1", `${label} mode must be heuristic_seed_v1`);
    pushWhenMissing(errors, allowedNodeTypes.includes(value.nodeType), `${label} nodeType must be explicit`);
    pushWhenMissing(errors, allowedFileRoles.includes(value.fileRole), `${label} fileRole must be explicit`);
    pushWhenMissing(errors, Number.isFinite(value.score) && value.score >= 0, `${label} score must be non-negative`);
    pushWhenMissing(errors, value.seededChannels.length > 0, `${label} seededChannels must not be empty`);
    const uniqueChannels = new Set(value.seededChannels);
    if (uniqueChannels.size !== value.seededChannels.length) {
        errors.push(`${label} seededChannels must be unique`);
    }
    for (const channel of value.seededChannels) {
        pushWhenMissing(errors, channel === "graph" || channel === "short_term" || channel === "vector", `${label} seededChannels must contain only graph, short_term, or vector`);
    }
    errors.push(...validatePackBlockInitScoreBreakdown(value.scoreBreakdown, `${label} scoreBreakdown`));
    return errors;
}
function validateRoutingChannelScoreSummary(value, label) {
    const errors = [];
    pushWhenMissing(errors, Number.isInteger(value.graph) && value.graph >= 0, `${label} graph must be a non-negative integer`);
    pushWhenMissing(errors, Number.isInteger(value.shortTerm) && value.shortTerm >= 0, `${label} shortTerm must be a non-negative integer`);
    pushWhenMissing(errors, Number.isInteger(value.vector) && value.vector >= 0, `${label} vector must be a non-negative integer`);
    return errors;
}
function validatePackGraphEvolution(value) {
    const errors = [];
    pushWhenMissing(errors, isIsoDate(value.builtAt), "graph evolution builtAt must be an ISO timestamp");
    pushWhenMissing(errors, Number.isInteger(value.structuralOps.split) && value.structuralOps.split >= 0, "graph evolution split count must be non-negative");
    pushWhenMissing(errors, Number.isInteger(value.structuralOps.merge) && value.structuralOps.merge >= 0, "graph evolution merge count must be non-negative");
    pushWhenMissing(errors, Number.isInteger(value.structuralOps.prune) && value.structuralOps.prune >= 0, "graph evolution prune count must be non-negative");
    pushWhenMissing(errors, Number.isInteger(value.structuralOps.connect) && value.structuralOps.connect >= 0, "graph evolution connect count must be non-negative");
    if (value.connectDiagnostics !== undefined) {
        pushWhenMissing(errors, Number.isInteger(value.connectDiagnostics.requestedBudget) && value.connectDiagnostics.requestedBudget >= 0, "graph evolution connectDiagnostics requestedBudget must be non-negative");
        pushWhenMissing(errors, Number.isInteger(value.connectDiagnostics.scoreThreshold) && value.connectDiagnostics.scoreThreshold >= 0, "graph evolution connectDiagnostics scoreThreshold must be non-negative");
        pushWhenMissing(errors, Number.isInteger(value.connectDiagnostics.candidatePairCount) && value.connectDiagnostics.candidatePairCount >= 0, "graph evolution connectDiagnostics candidatePairCount must be non-negative");
        pushWhenMissing(errors, Number.isInteger(value.connectDiagnostics.appliedPairCount) && value.connectDiagnostics.appliedPairCount >= 0, "graph evolution connectDiagnostics appliedPairCount must be non-negative");
        pushWhenMissing(errors, Number.isInteger(value.connectDiagnostics.createdEdgeCount) && value.connectDiagnostics.createdEdgeCount >= 0, "graph evolution connectDiagnostics createdEdgeCount must be non-negative");
    }
    const prunedIds = new Set(value.prunedBlockIds.filter((blockId) => blockId.length > 0));
    if (prunedIds.size !== value.prunedBlockIds.length) {
        errors.push("graph evolution prunedBlockIds must contain unique non-empty ids");
    }
    if (value.strongestBlockId !== null && value.strongestBlockId.length === 0) {
        errors.push("graph evolution strongestBlockId must be null or a non-empty id");
    }
    return errors;
}
function flattenRuntimeContextCoverageIds(value) {
    return value.compactedFrom ?? [value.id];
}
export function validateRuntimeCompileRequest(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.runtimeCompile, "runtime_compile.v1 contract is required");
    pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
    pushWhenMissing(errors, value.userMessage.length > 0, "userMessage is required");
    pushWhenMissing(errors, value.maxContextBlocks >= 0, "maxContextBlocks must be non-negative");
    pushWhenMissing(errors, value.modeRequested === "heuristic" || value.modeRequested === "learned", "modeRequested must be heuristic or learned");
    if (value.activePackId !== undefined) {
        pushWhenMissing(errors, value.activePackId.length > 0, "activePackId must be non-empty when set");
    }
    if (value.maxContextChars !== undefined) {
        pushWhenMissing(errors, value.maxContextChars >= 0, "maxContextChars must be non-negative");
    }
    if (value.compactionMode !== undefined) {
        pushWhenMissing(errors, value.compactionMode === "none" || value.compactionMode === "native", "compactionMode must be none or native");
    }
    return errors;
}
export function validateRuntimeCompileExpectation(value) {
    const errors = [];
    if (value.packId !== undefined) {
        pushWhenMissing(errors, value.packId.length > 0, "runtime compile expectation packId must be non-empty when set");
    }
    if (value.routePolicy !== undefined) {
        pushWhenMissing(errors, value.routePolicy === "heuristic_allowed" || value.routePolicy === "requires_learned_routing", "runtime compile expectation routePolicy must be explicit when set");
    }
    if (value.routerIdentity !== undefined && value.routerIdentity !== null) {
        pushWhenMissing(errors, value.routerIdentity.length > 0, "runtime compile expectation routerIdentity must be non-empty when set");
    }
    if (value.workspaceSnapshot !== undefined) {
        pushWhenMissing(errors, value.workspaceSnapshot.length > 0, "runtime compile expectation workspaceSnapshot must be non-empty when set");
    }
    if (value.workspaceRevision !== undefined && value.workspaceRevision !== null) {
        pushWhenMissing(errors, value.workspaceRevision.length > 0, "runtime compile expectation workspaceRevision must be non-empty when set");
    }
    if (value.eventRange !== undefined) {
        pushWhenMissing(errors, value.eventRange.count >= 0, "runtime compile expectation eventRange.count must be non-negative");
        pushWhenMissing(errors, value.eventRange.start >= 0, "runtime compile expectation eventRange.start must be non-negative");
        pushWhenMissing(errors, value.eventRange.end >= value.eventRange.start, "runtime compile expectation eventRange.end must be >= start");
    }
    if (value.eventExportDigest !== undefined && value.eventExportDigest !== null) {
        pushWhenMissing(errors, value.eventExportDigest.length > 0, "runtime compile expectation eventExportDigest must be non-empty when set");
    }
    if (value.builtAt !== undefined) {
        pushWhenMissing(errors, isIsoDate(value.builtAt), "runtime compile expectation builtAt must be an ISO timestamp when set");
    }
    return errors;
}
export function validateRuntimeCompileTargetExpectation(target, expectation) {
    const errors = validateRuntimeCompileExpectation(expectation);
    if (expectation.packId !== undefined && target.packId !== expectation.packId) {
        errors.push(`runtime compile target packId ${target.packId} does not match expected ${expectation.packId}`);
    }
    if (expectation.routePolicy !== undefined && target.routePolicy !== expectation.routePolicy) {
        errors.push(`runtime compile target routePolicy ${target.routePolicy} does not match expected ${expectation.routePolicy}`);
    }
    if (expectation.routerIdentity !== undefined && target.routerIdentity !== expectation.routerIdentity) {
        errors.push(`runtime compile target routerIdentity ${target.routerIdentity ?? "null"} does not match expected ${expectation.routerIdentity ?? "null"}`);
    }
    if (expectation.workspaceSnapshot !== undefined && target.workspaceSnapshot !== expectation.workspaceSnapshot) {
        errors.push(`runtime compile target workspaceSnapshot ${target.workspaceSnapshot} does not match expected ${expectation.workspaceSnapshot}`);
    }
    if (expectation.workspaceRevision !== undefined && target.workspaceRevision !== expectation.workspaceRevision) {
        errors.push(`runtime compile target workspaceRevision ${target.workspaceRevision ?? "null"} does not match expected ${expectation.workspaceRevision ?? "null"}`);
    }
    if (expectation.eventRange !== undefined) {
        if (target.eventRange.start !== expectation.eventRange.start) {
            errors.push(`runtime compile target eventRange.start ${target.eventRange.start} does not match expected ${expectation.eventRange.start}`);
        }
        if (target.eventRange.end !== expectation.eventRange.end) {
            errors.push(`runtime compile target eventRange.end ${target.eventRange.end} does not match expected ${expectation.eventRange.end}`);
        }
        if (target.eventRange.count !== expectation.eventRange.count) {
            errors.push(`runtime compile target eventRange.count ${target.eventRange.count} does not match expected ${expectation.eventRange.count}`);
        }
    }
    if (expectation.eventExportDigest !== undefined && target.eventExportDigest !== expectation.eventExportDigest) {
        errors.push(`runtime compile target eventExportDigest ${target.eventExportDigest ?? "null"} does not match expected ${expectation.eventExportDigest ?? "null"}`);
    }
    if (expectation.builtAt !== undefined && target.builtAt !== expectation.builtAt) {
        errors.push(`runtime compile target builtAt ${target.builtAt} does not match expected ${expectation.builtAt}`);
    }
    return errors;
}
export function validateRouteArtifactReference(value, options = {}) {
    const label = options.label ?? "routeArtifact";
    const errors = [];
    pushWhenMissing(errors, value.assetKind === "none" || value.assetKind === "stub" || value.assetKind === "artifact", `${label} assetKind must be none, stub, or artifact`);
    if (value.routerIdentity !== null) {
        pushWhenMissing(errors, value.routerIdentity.length > 0, `${label} routerIdentity must be non-empty when set`);
    }
    if (value.routerChecksum !== null) {
        pushWhenMissing(errors, value.routerChecksum.length > 0, `${label} routerChecksum must be non-empty when set`);
    }
    if (value.trainedAt !== null) {
        pushWhenMissing(errors, isIsoDate(value.trainedAt), `${label} trainedAt must be an ISO timestamp when set`);
    }
    if (value.eventExportDigest !== null) {
        pushWhenMissing(errors, value.eventExportDigest.length > 0, `${label} eventExportDigest must be non-empty when set`);
    }
    if (value.updateCount !== null) {
        pushWhenMissing(errors, value.updateCount >= 0, `${label} updateCount must be non-negative when set`);
    }
    if (value.objectiveChecksum !== null) {
        pushWhenMissing(errors, value.objectiveChecksum.length > 0, `${label} objectiveChecksum must be non-empty when set`);
    }
    if (value.freshnessChecksum !== null) {
        pushWhenMissing(errors, value.freshnessChecksum.length > 0, `${label} freshnessChecksum must be non-empty when set`);
    }
    if (value.assetKind === "none") {
        const nullableFields = [
            value.routerIdentity,
            value.routerChecksum,
            value.routeFnVersion,
            value.trainingMethod,
            value.trainedAt,
            value.eventExportDigest,
            value.updateCount,
            value.objective,
            value.objectiveChecksum,
            value.freshnessChecksum
        ];
        if (nullableFields.some((field) => field !== null)) {
            errors.push(`${label} fields must be null when assetKind=none`);
        }
    }
    if (value.assetKind !== "none" && value.routerIdentity === null) {
        errors.push(`${label} routerIdentity is required when a route artifact exists`);
    }
    if (options.routePolicy === "requires_learned_routing") {
        if (value.assetKind === "none") {
            errors.push(`${label} assetKind must not be none when routePolicy requires learned routing`);
        }
        if (value.routerChecksum === null) {
            errors.push(`${label} routerChecksum is required when routePolicy requires learned routing`);
        }
    }
    return errors;
}
export function validateServedArtifactProof(value, label = "servedArtifact") {
    const errors = [];
    pushWhenMissing(errors, value.packId.length > 0, `${label} packId is required`);
    pushWhenMissing(errors, value.routePolicy === "heuristic_allowed" || value.routePolicy === "requires_learned_routing", `${label} routePolicy must be explicit`);
    pushWhenMissing(errors, value.workspaceSnapshot.length > 0, `${label} workspaceSnapshot is required`);
    pushWhenMissing(errors, isIsoDate(value.builtAt), `${label} builtAt must be an ISO timestamp`);
    if (value.workspaceRevision !== null) {
        pushWhenMissing(errors, value.workspaceRevision.length > 0, `${label} workspaceRevision must be non-empty when set`);
    }
    if (value.eventExportDigest !== null) {
        pushWhenMissing(errors, value.eventExportDigest.length > 0, `${label} eventExportDigest must be non-empty when set`);
    }
    pushWhenMissing(errors, value.eventRange.count >= 0, `${label} eventRange.count must be non-negative`);
    pushWhenMissing(errors, value.eventRange.start >= 0, `${label} eventRange.start must be non-negative`);
    if (value.eventRange.count === 0) {
        pushWhenMissing(errors, value.eventRange.end === value.eventRange.start - 1, `${label} empty eventRange must use end=start-1`);
    }
    else {
        pushWhenMissing(errors, value.eventRange.end >= value.eventRange.start, `${label} eventRange.end must be >= start`);
    }
    errors.push(...validateRouteArtifactReference(value.routeArtifact, { routePolicy: value.routePolicy, label: `${label}.routeArtifact` }));
    return errors;
}
function validateRuntimeCompileStructuralCandidateSignal(value, label) {
    const errors = [];
    pushWhenMissing(errors, value.blockId.length > 0, `${label} blockId is required`);
    pushWhenMissing(errors, Number.isInteger(value.rank) && value.rank >= 1, `${label} rank must be a positive integer`);
    pushWhenMissing(errors, Number.isFinite(value.score), `${label} score must be finite`);
    pushWhenMissing(errors, value.selectedBy === null || value.selectedBy === "token_match" || value.selectedBy === "priority_fallback", `${label} selectedBy must be token_match, priority_fallback, or null`);
    pushWhenMissing(errors, Number.isFinite(value.traversalScore) && value.traversalScore >= 0, `${label} traversalScore must be non-negative`);
    pushWhenMissing(errors, new Set(value.matchedTokens).size === value.matchedTokens.length, `${label} matchedTokens must be unique`);
    pushWhenMissing(errors, new Set(value.directMatchedTokens).size === value.directMatchedTokens.length, `${label} directMatchedTokens must be unique`);
    pushWhenMissing(errors, new Set(value.compactedFrom).size === value.compactedFrom.length, `${label} compactedFrom must be unique`);
    if (value.selectedBy !== null && !value.selected) {
        errors.push(`${label} selectedBy requires selected=true`);
    }
    if (value.directMatchedTokens.some((token) => !value.matchedTokens.includes(token))) {
        errors.push(`${label} directMatchedTokens must be a subset of matchedTokens`);
    }
    if (value.traversalActivated && value.traversalScore <= 0) {
        errors.push(`${label} traversalActivated=true requires traversalScore > 0`);
    }
    if (!value.traversalActivated && value.traversalScore !== 0) {
        errors.push(`${label} traversalActivated=false requires traversalScore = 0`);
    }
    return errors;
}
function validateRuntimeCompileStructuralSignals(value, label) {
    const errors = [];
    pushWhenMissing(errors, value.matchedCandidateCount >= 0, `${label} matchedCandidateCount must be non-negative`);
    pushWhenMissing(errors, value.selectedMatchedCount >= 0, `${label} selectedMatchedCount must be non-negative`);
    pushWhenMissing(errors, value.selectedPriorityFallbackCount >= 0, `${label} selectedPriorityFallbackCount must be non-negative`);
    pushWhenMissing(errors, value.overlapPrunedCount >= 0, `${label} overlapPrunedCount must be non-negative`);
    pushWhenMissing(errors, value.traversalActivatedCount >= 0, `${label} traversalActivatedCount must be non-negative`);
    pushWhenMissing(errors, new Set(value.selectedBlockIds).size === value.selectedBlockIds.length, `${label} selectedBlockIds must be unique`);
    pushWhenMissing(errors, new Set(value.overlapPrunedBlockIds).size === value.overlapPrunedBlockIds.length, `${label} overlapPrunedBlockIds must be unique`);
    pushWhenMissing(errors, new Set(value.traversalActivatedBlockIds).size === value.traversalActivatedBlockIds.length, `${label} traversalActivatedBlockIds must be unique`);
    const selectedCandidates = value.candidates.filter((candidate) => candidate.selected);
    const matchedCandidates = value.candidates.filter((candidate) => candidate.matchedTokens.length > 0);
    const selectedMatchedCandidates = selectedCandidates.filter((candidate) => candidate.selectedBy === "token_match");
    const selectedFallbackCandidates = selectedCandidates.filter((candidate) => candidate.selectedBy === "priority_fallback");
    const overlapPrunedCandidates = value.candidates.filter((candidate) => candidate.overlapPruned);
    const traversalCandidates = value.candidates.filter((candidate) => candidate.traversalActivated);
    pushWhenMissing(errors, value.matchedCandidateCount === matchedCandidates.length, `${label} matchedCandidateCount must match candidates`);
    pushWhenMissing(errors, value.selectedMatchedCount === selectedMatchedCandidates.length, `${label} selectedMatchedCount must match candidates`);
    pushWhenMissing(errors, value.selectedPriorityFallbackCount === selectedFallbackCandidates.length, `${label} selectedPriorityFallbackCount must match candidates`);
    pushWhenMissing(errors, value.overlapPrunedCount === overlapPrunedCandidates.length, `${label} overlapPrunedCount must match candidates`);
    pushWhenMissing(errors, value.traversalActivatedCount === traversalCandidates.length, `${label} traversalActivatedCount must match candidates`);
    pushWhenMissing(errors, checksumJsonPayload([...value.selectedBlockIds]) === checksumJsonPayload(selectedCandidates.map((candidate) => candidate.blockId)), `${label} selectedBlockIds must match candidates`);
    pushWhenMissing(errors, checksumJsonPayload([...value.overlapPrunedBlockIds]) === checksumJsonPayload(overlapPrunedCandidates.map((candidate) => candidate.blockId)), `${label} overlapPrunedBlockIds must match candidates`);
    pushWhenMissing(errors, checksumJsonPayload([...value.traversalActivatedBlockIds]) === checksumJsonPayload(traversalCandidates.map((candidate) => candidate.blockId)), `${label} traversalActivatedBlockIds must match candidates`);
    value.candidates.forEach((candidate, index) => {
        errors.push(...validateRuntimeCompileStructuralCandidateSignal(candidate, `${label}.candidates[${index}]`));
        if (candidate.rank !== index + 1) {
            errors.push(`${label}.candidates[${index}] rank must equal list position`);
        }
    });
    return errors;
}
export function validateRuntimeCompileResponse(value) {
    const errors = [];
    const selectedCoverage = new Map();
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.runtimeCompile, "runtime_compile.v1 contract is required");
    pushWhenMissing(errors, value.packId.length > 0, "packId is required");
    pushWhenMissing(errors, value.diagnostics.modeRequested === "heuristic" || value.diagnostics.modeRequested === "learned", "modeRequested must be explicit");
    pushWhenMissing(errors, value.diagnostics.modeEffective === "heuristic" || value.diagnostics.modeEffective === "learned", "modeEffective must be explicit");
    pushWhenMissing(errors, value.diagnostics.candidateCount >= 0, "candidateCount must be non-negative");
    pushWhenMissing(errors, value.diagnostics.selectedCount >= 0, "selectedCount must be non-negative");
    pushWhenMissing(errors, value.diagnostics.selectedCount === value.selectedContext.length, "selectedCount must match selectedContext length");
    pushWhenMissing(errors, value.diagnostics.candidateCount >= value.diagnostics.selectedCount, "candidateCount must be >= selectedCount");
    pushWhenMissing(errors, value.diagnostics.selectedCharCount >= 0, "selectedCharCount must be non-negative");
    pushWhenMissing(errors, value.diagnostics.selectedTokenCount >= 0, "selectedTokenCount must be non-negative");
    pushWhenMissing(errors, value.diagnostics.selectionStrategy === "pack_route_fn_selection_v1", "selectionStrategy must be pack_route_fn_selection_v1");
    pushWhenMissing(errors, value.diagnostics.selectionDigest.length > 0, "selectionDigest is required");
    errors.push(...validateRuntimeCompileStructuralSignals(value.diagnostics.structuralSignals, "diagnostics.structuralSignals"));
    pushWhenMissing(errors, value.diagnostics.compactionMode === "none" || value.diagnostics.compactionMode === "native", "compactionMode must be none or native");
    errors.push(...validateServedArtifactProof(value.diagnostics.servedArtifact, "diagnostics.servedArtifact"));
    errors.push(...validateRoutingChannelScoreSummary(value.diagnostics.routingChannels.candidates, "routingChannels.candidates"));
    errors.push(...validateRoutingChannelScoreSummary(value.diagnostics.routingChannels.selected, "routingChannels.selected"));
    if (value.diagnostics.servedArtifact.packId !== value.packId) {
        errors.push(`diagnostics.servedArtifact packId ${value.diagnostics.servedArtifact.packId} does not match response packId ${value.packId}`);
    }
    const replayLearnedRouteOverrideActive = value.diagnostics.notes.some((note) => note.startsWith("replay_learned_route_override="));
    if (!replayLearnedRouteOverrideActive && value.diagnostics.servedArtifact.routeArtifact.routerIdentity !== value.diagnostics.routerIdentity) {
        errors.push(`diagnostics.servedArtifact routeArtifact.routerIdentity ${value.diagnostics.servedArtifact.routeArtifact.routerIdentity ?? "null"} does not match diagnostics routerIdentity ${value.diagnostics.routerIdentity ?? "null"}`);
    }
    if (value.diagnostics.modeEffective === "learned" && !value.diagnostics.usedLearnedRouteFn && !replayLearnedRouteOverrideActive) {
        errors.push("learned mode requires usedLearnedRouteFn=true");
    }
    if (value.diagnostics.usedLearnedRouteFn && value.diagnostics.modeEffective !== "learned") {
        errors.push("usedLearnedRouteFn=true requires modeEffective=learned");
    }
    if (value.diagnostics.usedLearnedRouteFn && value.diagnostics.routerIdentity === null) {
        errors.push("learned routing requires routerIdentity");
    }
    value.selectedContext.forEach((block, index) => {
        errors.push(...validateRuntimeContextBlock(block, `selectedContext[${index}]`));
        for (const coverageId of flattenRuntimeContextCoverageIds(block)) {
            const existingBlockId = selectedCoverage.get(coverageId);
            if (existingBlockId !== undefined) {
                errors.push(`selectedContext[${index}] overlaps block ${existingBlockId} via ${coverageId}`);
                continue;
            }
            selectedCoverage.set(coverageId, block.id);
        }
    });
    return errors;
}
export function validateBrainServeHotPathTiming(value, label = "brain serve hot-path timing") {
    const errors = [];
    pushWhenMissing(errors, value.scope === "brain_serve_hot_path_only", `${label} scope must be brain_serve_hot_path_only`);
    pushWhenMissing(errors, value.backgroundWorkIncluded === false, `${label} backgroundWorkIncluded must be false`);
    pushWhenMissing(errors, value.detail.length > 0, `${label} detail is required`);
    const numericFields = [
        ["totalMs", value.totalMs],
        ["routeSelectionMs", value.routeSelectionMs],
        ["promptAssemblyMs", value.promptAssemblyMs],
        ["otherMs", value.otherMs]
    ];
    for (const [field, candidate] of numericFields) {
        if (candidate === null) {
            continue;
        }
        pushWhenMissing(errors, Number.isFinite(candidate) && candidate >= 0, `${label} ${field} must be null or a non-negative finite number`);
    }
    if (value.totalMs === null) {
        pushWhenMissing(errors, value.routeSelectionMs === null && value.promptAssemblyMs === null && value.otherMs === null, `${label} totalMs=null requires routeSelectionMs, promptAssemblyMs, and otherMs to be null`);
        return errors;
    }
    pushWhenMissing(errors, value.otherMs !== null, `${label} otherMs is required when totalMs is present`);
    if (value.otherMs === null) {
        return errors;
    }
    const accountedMs = (value.routeSelectionMs ?? 0) + (value.promptAssemblyMs ?? 0) + value.otherMs;
    if (Math.abs(accountedMs - value.totalMs) > 0.01) {
        errors.push(`${label} totalMs must equal routeSelectionMs + promptAssemblyMs + otherMs within 0.01ms`);
    }
    return errors;
}
export function validateBrainAttachmentPolicySemantics(value, label = "brain attachment policy") {
    const errors = [];
    pushWhenMissing(errors, value.mode === "dedicated" || value.mode === "shared", `${label} mode must be dedicated or shared`);
    pushWhenMissing(errors, value.readScope === "current_profile_only" || value.readScope === "attached_profiles", `${label} readScope must be current_profile_only or attached_profiles`);
    pushWhenMissing(errors, value.writeScope === "current_profile_only" || value.writeScope === "attached_profiles", `${label} writeScope must be current_profile_only or attached_profiles`);
    pushWhenMissing(errors, value.requiresProfileAttribution, `${label} requiresProfileAttribution must be true`);
    pushWhenMissing(errors, value.detail.length > 0, `${label} detail is required`);
    if (value.mode === "dedicated") {
        pushWhenMissing(errors, value.readScope === "current_profile_only", `${label} dedicated mode readScope must be current_profile_only`);
        pushWhenMissing(errors, value.writeScope === "current_profile_only", `${label} dedicated mode writeScope must be current_profile_only`);
        pushWhenMissing(errors, value.currentProfileExclusive, `${label} dedicated mode must be currentProfileExclusive`);
    }
    if (value.mode === "shared") {
        pushWhenMissing(errors, value.readScope === "attached_profiles", `${label} shared mode readScope must be attached_profiles`);
        pushWhenMissing(errors, value.writeScope === "attached_profiles", `${label} shared mode writeScope must be attached_profiles`);
        pushWhenMissing(errors, !value.currentProfileExclusive, `${label} shared mode must not be currentProfileExclusive`);
    }
    return errors;
}
export function validateBrainAttachmentPolicy(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.brainAttachmentPolicy, "brain_attachment_policy.v1 contract is required");
    errors.push(...validateBrainAttachmentPolicySemantics(value.policy));
    return errors;
}
function isSha256Digest(value) {
    return value.startsWith("sha256-") && value.length > "sha256-".length;
}
function validateFingerprintEntries(label, values) {
    const errors = [];
    pushWhenMissing(errors, values.length > 0, `${label} must not be empty`);
    pushWhenMissing(errors, new Set(values.filter((value) => value.length > 0)).size === values.length, `${label} must contain unique non-empty entries`);
    for (const value of values) {
        pushWhenMissing(errors, value.length > 0, `${label} must contain unique non-empty entries`);
    }
    return errors;
}
export function validateRuntimeContextFingerprint(value) {
    const errors = [];
    pushWhenMissing(errors, isSha256Digest(value.digest), "runtime context fingerprint digest must be a sha256 digest");
    if (value.selectionDigest !== null) {
        pushWhenMissing(errors, isSha256Digest(value.selectionDigest), "runtime context fingerprint selectionDigest must be a sha256 digest");
    }
    if (value.promptContextDigest !== null) {
        pushWhenMissing(errors, isSha256Digest(value.promptContextDigest), "runtime context fingerprint promptContextDigest must be a sha256 digest");
    }
    if (value.workspaceInjectionSurfaceDigest !== null) {
        pushWhenMissing(errors, isSha256Digest(value.workspaceInjectionSurfaceDigest), "runtime context fingerprint workspaceInjectionSurfaceDigest must be a sha256 digest");
    }
    if (value.runtimeHintsDigest !== null) {
        pushWhenMissing(errors, isSha256Digest(value.runtimeHintsDigest), "runtime context fingerprint runtimeHintsDigest must be a sha256 digest");
    }
    pushWhenMissing(errors, isSha256Digest(value.profileLineageDigest), "runtime context fingerprint profileLineageDigest must be a sha256 digest");
    pushWhenMissing(errors, isSha256Digest(value.sessionLineageDigest), "runtime context fingerprint sessionLineageDigest must be a sha256 digest");
    pushWhenMissing(errors, isSha256Digest(value.brainLineageDigest), "runtime context fingerprint brainLineageDigest must be a sha256 digest");
    if (value.promptContextFingerprints.length > 0) {
        errors.push(...validateFingerprintEntries("runtime context fingerprint promptContextFingerprints", value.promptContextFingerprints));
        for (const fingerprint of value.promptContextFingerprints) {
            pushWhenMissing(errors, isSha256Digest(fingerprint), "runtime context fingerprint promptContextFingerprints must be sha256 digests");
        }
    }
    if (value.promptContextDigest === null) {
        pushWhenMissing(errors, value.promptContextFingerprints.length === 0 && value.workspaceInjectionSurfaceDigest === null, "runtime context fingerprint promptContextDigest is required when prompt/context inputs are present");
    }
    if (value.runtimeHints.length > 0) {
        errors.push(...validateFingerprintEntries("runtime context fingerprint runtimeHints", value.runtimeHints));
    }
    if (value.runtimeHintsDigest === null) {
        pushWhenMissing(errors, value.runtimeHints.length === 0, "runtime context fingerprint runtimeHintsDigest is required when runtimeHints are present");
    }
    errors.push(...validateFingerprintEntries("runtime context fingerprint profileLineage", value.profileLineage));
    errors.push(...validateFingerprintEntries("runtime context fingerprint sessionLineage", value.sessionLineage));
    errors.push(...validateFingerprintEntries("runtime context fingerprint brainLineage", value.brainLineage));
    return errors;
}
export function validateProfileTurnAttribution(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.profileTurnAttribution, "profile_turn_attribution.v1 contract is required");
    pushWhenMissing(errors, value.hostRuntimeOwner === "openclaw", "profile turn attribution hostRuntimeOwner must be openclaw");
    pushWhenMissing(errors, value.profileSelector.length > 0, "profile turn attribution profileSelector must be non-empty");
    if (value.profileId !== null) {
        pushWhenMissing(errors, value.profileId.length > 0, "profile turn attribution profileId must be null or non-empty");
    }
    pushWhenMissing(errors, value.brainAttachmentPolicy === "undeclared" ||
        value.brainAttachmentPolicy === "dedicated" ||
        value.brainAttachmentPolicy === "shared", "profile turn attribution brainAttachmentPolicy must be undeclared, dedicated, or shared");
    pushWhenMissing(errors, value.brainStatus === "serving_active_pack" ||
        value.brainStatus === "fail_open_static_context" ||
        value.brainStatus === "hard_fail", "profile turn attribution brainStatus must be serving_active_pack, fail_open_static_context, or hard_fail");
    pushWhenMissing(errors, value.sessionId.length > 0, "profile turn attribution sessionId is required");
    pushWhenMissing(errors, value.channel.length > 0, "profile turn attribution channel is required");
    pushWhenMissing(errors, value.interactionEventId.length > 0, "profile turn attribution interactionEventId is required");
    pushWhenMissing(errors, isIsoDate(value.createdAt), "profile turn attribution createdAt must be an ISO timestamp");
    pushWhenMissing(errors, value.selectedContextCount >= 0, "profile turn attribution selectedContextCount must be non-negative");
    pushWhenMissing(errors, value.stableKernelBlockCount >= 0, "profile turn attribution stableKernelBlockCount must be non-negative");
    pushWhenMissing(errors, value.brainCompiledBlockCount >= 0, "profile turn attribution brainCompiledBlockCount must be non-negative");
    pushWhenMissing(errors, value.selectedContextCount === value.stableKernelBlockCount + value.brainCompiledBlockCount, "profile turn attribution selectedContextCount must equal stableKernelBlockCount + brainCompiledBlockCount");
    pushWhenMissing(errors, value.detail.length > 0, "profile turn attribution detail is required");
    if (value.packId !== null) {
        pushWhenMissing(errors, value.packId.length > 0, "profile turn attribution packId must be null or non-empty");
    }
    if (value.routerIdentity !== null) {
        pushWhenMissing(errors, value.routerIdentity.length > 0, "profile turn attribution routerIdentity must be null or non-empty");
    }
    if (value.selectionMode !== null) {
        pushWhenMissing(errors, value.selectionMode.length > 0, "profile turn attribution selectionMode must be null or non-empty");
    }
    if (value.selectionTiers !== null) {
        pushWhenMissing(errors, value.selectionTiers.length > 0, "profile turn attribution selectionTiers must be null or non-empty");
    }
    if (value.selectionDigest !== null) {
        pushWhenMissing(errors, value.selectionDigest.length > 0, "profile turn attribution selectionDigest must be null or non-empty");
    }
    errors.push(...validateRuntimeContextFingerprint(value.contextFingerprint));
    if (value.selectionDigest !== value.contextFingerprint.selectionDigest) {
        errors.push("profile turn attribution contextFingerprint.selectionDigest must match selectionDigest");
    }
    if (value.usedLearnedRouteFn === true && value.routerIdentity === null) {
        errors.push("profile turn attribution usedLearnedRouteFn=true requires routerIdentity");
    }
    const evidenceStates = [
        "route_fn_and_brain_context",
        "brain_context_only",
        "route_fn_only",
        "stable_kernel_only",
        "fail_open_static_context",
        "hard_fail",
        "unprobed"
    ];
    pushWhenMissing(errors, evidenceStates.includes(value.contextEvidence), "profile turn attribution contextEvidence must be explicit");
    if ((value.contextEvidence === "route_fn_and_brain_context" || value.contextEvidence === "route_fn_only") && value.usedLearnedRouteFn !== true) {
        errors.push("profile turn attribution route_fn evidence requires usedLearnedRouteFn=true");
    }
    for (const source of value.stableKernelSources) {
        pushWhenMissing(errors, source.length > 0, "profile turn attribution stableKernelSources must be non-empty");
    }
    for (const source of value.brainCompiledSources) {
        pushWhenMissing(errors, source.length > 0, "profile turn attribution brainCompiledSources must be non-empty");
    }
    return errors;
}
export function validateCurrentProfileBrainStatus(value) {
    const errors = [];
    const brainStates = ["missing", "seed_state_authoritative", "pg_promoted_pack_authoritative", "no_active_pack"];
    const serveStates = ["serving_active_pack", "fail_open_static_context", "hard_fail", "unprobed"];
    const statusLevels = ["ok", "warn", "fail"];
    const activationStates = [
        "healthy_seed",
        "awaiting_first_export",
        "active_promoted",
        "stale_incomplete",
        "broken_install",
        "detached"
    ];
    const budgetStrategies = ["fixed_v1", "empirical_v1"];
    const structuralDecisionOrigins = [
        "manual_caller_shape",
        "empirical_control",
        "default_path_control",
        "unknown"
    ];
    const structuralDecisionBases = [
        "caller_override",
        "compile_structural_signals",
        "graph_evolution",
        "fixed_default",
        "fixed_fallback",
        "no_evidence_fallback",
        "no_compile_signal_evidence_fallback",
        "unknown"
    ];
    const passiveLearningWatchStates = [
        "watching",
        "snapshot_only",
        "stale_snapshot",
        "not_visible"
    ];
    const attachmentStates = ["attached", "not_attached", "unknown"];
    const attachmentProofStates = ["self_proving", "activation_root_only"];
    const hookScopes = ["exact_openclaw_home", "activation_root_only"];
    const hookInstallStates = [
        "installed",
        "not_installed",
        "blocked_by_allowlist",
        "unverified"
    ];
    const hookLoadabilities = ["loadable", "blocked", "not_installed", "unverified"];
    const hookLoadProofs = ["status_probe_ready", "not_ready"];
    const passiveLearningExportStates = [
        "awaiting_first_export",
        "latest_export_visible",
        "history_only"
    ];
    const passiveLearningBacklogStates = [
        "unknown",
        "awaiting_first_export",
        "principal_live_priority",
        "principal_backfill_priority",
        "live_priority",
        "backfill_only",
        "caught_up"
    ];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.currentProfileBrainStatus, "current_profile_brain_status.v1 contract is required");
    pushWhenMissing(errors, isIsoDate(value.generatedAt), "current profile brain status generatedAt must be an ISO timestamp");
    pushWhenMissing(errors, value.host.noun === "Host", "current profile brain status host.noun must be Host");
    pushWhenMissing(errors, value.host.runtimeOwner === "openclaw", "current profile brain status host.runtimeOwner must be openclaw");
    pushWhenMissing(errors, value.host.activationRoot.length > 0, "current profile brain status host.activationRoot is required");
    pushWhenMissing(errors, value.profile.noun === "Profile", "current profile brain status profile.noun must be Profile");
    pushWhenMissing(errors, value.profile.selector === "current_profile", "current profile brain status profile.selector must be current_profile");
    if (value.profile.profileId !== null) {
        pushWhenMissing(errors, value.profile.profileId.length > 0, "current profile brain status profile.profileId must be null or non-empty");
    }
    pushWhenMissing(errors, value.profile.detail.length > 0, "current profile brain status profile.detail is required");
    pushWhenMissing(errors, value.brain.noun === "Brain", "current profile brain status brain.noun must be Brain");
    pushWhenMissing(errors, value.brain.initMode === null || value.brain.initMode === "fast_boot_defaults", "current profile brain status brain.initMode must be null or fast_boot_defaults");
    pushWhenMissing(errors, brainStates.includes(value.brain.state), "current profile brain status brain.state must be explicit");
    pushWhenMissing(errors, value.brain.routeFreshness === "updated" || value.brain.routeFreshness === "no_supervision" || value.brain.routeFreshness === "unknown", "current profile brain status brain.routeFreshness must be updated, no_supervision, or unknown");
    if (value.brain.activePackId !== null) {
        pushWhenMissing(errors, value.brain.activePackId.length > 0, "current profile brain status brain.activePackId must be null or non-empty");
    }
    if (value.brain.activationRoot !== null) {
        pushWhenMissing(errors, value.brain.activationRoot.length > 0, "current profile brain status brain.activationRoot must be null or non-empty");
    }
    if (value.brain.logRoot !== null) {
        pushWhenMissing(errors, value.brain.logRoot.length > 0, "current profile brain status brain.logRoot must be null or non-empty");
    }
    if (value.brain.routerIdentity !== null) {
        pushWhenMissing(errors, value.brain.routerIdentity.length > 0, "current profile brain status brain.routerIdentity must be null or non-empty");
    }
    if (value.brain.routerChecksum !== null) {
        pushWhenMissing(errors, value.brain.routerChecksum.length > 0, "current profile brain status brain.routerChecksum must be null or non-empty");
    }
    if (value.brain.lastExportAt !== null && !isIsoDate(value.brain.lastExportAt)) {
        errors.push("current profile brain status brain.lastExportAt must be null or an ISO timestamp");
    }
    if (value.brain.lastLearningUpdateAt !== null && !isIsoDate(value.brain.lastLearningUpdateAt)) {
        errors.push("current profile brain status brain.lastLearningUpdateAt must be null or an ISO timestamp");
    }
    if (value.brain.lastPromotionAt !== null && !isIsoDate(value.brain.lastPromotionAt)) {
        errors.push("current profile brain status brain.lastPromotionAt must be null or an ISO timestamp");
    }
    pushWhenMissing(errors, value.brain.summary.length > 0, "current profile brain status brain.summary is required");
    pushWhenMissing(errors, value.brain.detail.length > 0, "current profile brain status brain.detail is required");
    pushWhenMissing(errors, value.hook.noun === "Hook", "current profile brain status hook.noun must be Hook");
    pushWhenMissing(errors, hookScopes.includes(value.hook.scope), "current profile brain status hook.scope must be explicit");
    pushWhenMissing(errors, hookInstallStates.includes(value.hook.installState), "current profile brain status hook.installState must be explicit");
    pushWhenMissing(errors, hookLoadabilities.includes(value.hook.loadability), "current profile brain status hook.loadability must be explicit");
    pushWhenMissing(errors, hookLoadProofs.includes(value.hook.loadProof), "current profile brain status hook.loadProof must be explicit");
    pushWhenMissing(errors, typeof value.hook.desynced === "boolean", "current profile brain status hook.desynced must be a boolean");
    pushWhenMissing(errors, value.hook.detail.length > 0, "current profile brain status hook.detail is required");
    for (const [field, fieldValue] of [
        ["openclawHome", value.hook.openclawHome],
        ["hookPath", value.hook.hookPath],
        ["runtimeGuardPath", value.hook.runtimeGuardPath],
        ["manifestPath", value.hook.manifestPath]
    ]) {
        if (fieldValue !== null) {
            pushWhenMissing(errors, fieldValue.length > 0, `current profile brain status hook.${field} must be null or non-empty`);
        }
    }
    pushWhenMissing(errors, value.attachment.noun === "Attachment", "current profile brain status attachment.noun must be Attachment");
    pushWhenMissing(errors, attachmentStates.includes(value.attachment.state), "current profile brain status attachment.state must be explicit");
    pushWhenMissing(errors, value.attachment.servingSlot === "active" || value.attachment.servingSlot === "none", "current profile brain status attachment.servingSlot must be active or none");
    pushWhenMissing(errors, value.attachment.policyMode === "undeclared" ||
        value.attachment.policyMode === "dedicated" ||
        value.attachment.policyMode === "shared", "current profile brain status attachment.policyMode must be undeclared, dedicated, or shared");
    pushWhenMissing(errors, value.attachment.detail.length > 0, "current profile brain status attachment.detail is required");
    if (value.attachment.policyMode === "undeclared") {
        pushWhenMissing(errors, value.attachment.policy === null, "current profile brain status attachment.policy must be null when policyMode=undeclared");
    }
    else if (value.attachment.policy === null) {
        errors.push("current profile brain status attachment.policy is required when policyMode is declared");
    }
    else {
        pushWhenMissing(errors, value.attachment.policy.mode === value.attachment.policyMode, "current profile brain status attachment.policy.mode must match attachment.policyMode");
        errors.push(...validateBrainAttachmentPolicySemantics(value.attachment.policy, "current profile brain status attachment.policy"));
    }
    pushWhenMissing(errors, attachmentProofStates.includes(value.attachment.proofState), "current profile brain status attachment.proofState must be explicit");
    pushWhenMissing(errors, typeof value.attachment.watchOnly === "boolean", "current profile brain status attachment.watchOnly must be a boolean");
    pushWhenMissing(errors, statusLevels.includes(value.brainStatus.status), "current profile brain status brainStatus.status must be ok, warn, or fail");
    pushWhenMissing(errors, brainStates.includes(value.brainStatus.brainState), "current profile brain status brainStatus.brainState must be explicit");
    pushWhenMissing(errors, serveStates.includes(value.brainStatus.serveState), "current profile brain status brainStatus.serveState must be explicit");
    pushWhenMissing(errors, activationStates.includes(value.brainStatus.activationState), "current profile brain status brainStatus.activationState must be explicit");
    pushWhenMissing(errors, typeof value.brainStatus.failOpen === "boolean", "current profile brain status brainStatus.failOpen must be a boolean");
    pushWhenMissing(errors, typeof value.brainStatus.awaitingFirstExport === "boolean", "current profile brain status brainStatus.awaitingFirstExport must be a boolean");
    errors.push(...validateBrainServeHotPathTiming(value.brainStatus.timing, "current profile brain status brainStatus.timing"));
    pushWhenMissing(errors, structuralDecisionOrigins.includes(value.brainStatus.structuralDecision.origin), "current profile brain status brainStatus.structuralDecision.origin must be manual_caller_shape, empirical_control, default_path_control, or unknown");
    pushWhenMissing(errors, structuralDecisionBases.includes(value.brainStatus.structuralDecision.basis), "current profile brain status brainStatus.structuralDecision.basis must be explicit");
    pushWhenMissing(errors, value.brainStatus.structuralDecision.requestedBudgetStrategy === null ||
        budgetStrategies.includes(value.brainStatus.structuralDecision.requestedBudgetStrategy), "current profile brain status brainStatus.structuralDecision.requestedBudgetStrategy must be fixed_v1, empirical_v1, or null");
    pushWhenMissing(errors, value.brainStatus.structuralDecision.resolvedBudgetStrategy === null ||
        budgetStrategies.includes(value.brainStatus.structuralDecision.resolvedBudgetStrategy), "current profile brain status brainStatus.structuralDecision.resolvedBudgetStrategy must be fixed_v1, empirical_v1, or null");
    if (value.brainStatus.structuralDecision.resolvedMaxContextBlocks !== null &&
        (!Number.isInteger(value.brainStatus.structuralDecision.resolvedMaxContextBlocks) ||
            value.brainStatus.structuralDecision.resolvedMaxContextBlocks < 0)) {
        errors.push("current profile brain status brainStatus.structuralDecision.resolvedMaxContextBlocks must be null or a non-negative integer");
    }
    pushWhenMissing(errors, value.brainStatus.structuralDecision.detail.length > 0, "current profile brain status brainStatus.structuralDecision.detail is required");
    pushWhenMissing(errors, value.brainStatus.detail.length > 0, "current profile brain status brainStatus.detail is required");
    pushWhenMissing(errors, passiveLearningWatchStates.includes(value.passiveLearning.watchState), "current profile brain status passiveLearning.watchState must be explicit");
    pushWhenMissing(errors, typeof value.passiveLearning.learnerRunning === "boolean", "current profile brain status passiveLearning.learnerRunning must be a boolean");
    pushWhenMissing(errors, typeof value.passiveLearning.firstExportOccurred === "boolean", "current profile brain status passiveLearning.firstExportOccurred must be a boolean");
    pushWhenMissing(errors, passiveLearningExportStates.includes(value.passiveLearning.exportState), "current profile brain status passiveLearning.exportState must be explicit");
    pushWhenMissing(errors, passiveLearningBacklogStates.includes(value.passiveLearning.backlogState), "current profile brain status passiveLearning.backlogState must be explicit");
    if (value.passiveLearning.pendingLive !== null) {
        pushWhenMissing(errors, Number.isInteger(value.passiveLearning.pendingLive) && value.passiveLearning.pendingLive >= 0, "current profile brain status passiveLearning.pendingLive must be null or a non-negative integer");
    }
    if (value.passiveLearning.pendingBackfill !== null) {
        pushWhenMissing(errors, Number.isInteger(value.passiveLearning.pendingBackfill) && value.passiveLearning.pendingBackfill >= 0, "current profile brain status passiveLearning.pendingBackfill must be null or a non-negative integer");
    }
    if (value.passiveLearning.lastWatchHeartbeatAt !== null && !isIsoDate(value.passiveLearning.lastWatchHeartbeatAt)) {
        errors.push("current profile brain status passiveLearning.lastWatchHeartbeatAt must be null or an ISO timestamp");
    }
    if (value.passiveLearning.watchIntervalSeconds !== null) {
        pushWhenMissing(errors, Number.isInteger(value.passiveLearning.watchIntervalSeconds) && value.passiveLearning.watchIntervalSeconds > 0, "current profile brain status passiveLearning.watchIntervalSeconds must be null or a positive integer");
    }
    if (value.passiveLearning.lastExportAt !== null && !isIsoDate(value.passiveLearning.lastExportAt)) {
        errors.push("current profile brain status passiveLearning.lastExportAt must be null or an ISO timestamp");
    }
    if (value.passiveLearning.lastPromotionAt !== null && !isIsoDate(value.passiveLearning.lastPromotionAt)) {
        errors.push("current profile brain status passiveLearning.lastPromotionAt must be null or an ISO timestamp");
    }
    if (value.passiveLearning.currentServingPackId !== null) {
        pushWhenMissing(errors, value.passiveLearning.currentServingPackId.length > 0, "current profile brain status passiveLearning.currentServingPackId must be null or non-empty");
    }
    if (value.passiveLearning.lastMaterializedPackId !== null) {
        pushWhenMissing(errors, value.passiveLearning.lastMaterializedPackId.length > 0, "current profile brain status passiveLearning.lastMaterializedPackId must be null or non-empty");
    }
    pushWhenMissing(errors, typeof value.passiveLearning.lastObservedDelta.available === "boolean", "current profile brain status passiveLearning.lastObservedDelta.available must be a boolean");
    if (value.passiveLearning.lastObservedDelta.observedAt !== null && !isIsoDate(value.passiveLearning.lastObservedDelta.observedAt)) {
        errors.push("current profile brain status passiveLearning.lastObservedDelta.observedAt must be null or an ISO timestamp");
    }
    for (const [field, fieldValue] of [
        ["exported", value.passiveLearning.lastObservedDelta.exported],
        ["labeled", value.passiveLearning.lastObservedDelta.labeled],
        ["promoted", value.passiveLearning.lastObservedDelta.promoted],
        ["served", value.passiveLearning.lastObservedDelta.served]
    ]) {
        if (fieldValue !== null && typeof fieldValue !== "boolean") {
            errors.push(`current profile brain status passiveLearning.lastObservedDelta.${field} must be null or a boolean`);
        }
    }
    const lastObservedDeltaTransitionKinds = [
        "staged_candidate",
        "promoted_active"
    ];
    if (value.passiveLearning.lastObservedDelta.latestPackTransition !== null) {
        pushWhenMissing(errors, lastObservedDeltaTransitionKinds.includes(value.passiveLearning.lastObservedDelta.latestPackTransition.kind), "current profile brain status passiveLearning.lastObservedDelta.latestPackTransition.kind must be explicit");
        if (value.passiveLearning.lastObservedDelta.latestPackTransition.fromPackId !== null) {
            pushWhenMissing(errors, value.passiveLearning.lastObservedDelta.latestPackTransition.fromPackId.length > 0, "current profile brain status passiveLearning.lastObservedDelta.latestPackTransition.fromPackId must be null or non-empty");
        }
        pushWhenMissing(errors, value.passiveLearning.lastObservedDelta.latestPackTransition.toPackId.length > 0, "current profile brain status passiveLearning.lastObservedDelta.latestPackTransition.toPackId must be non-empty");
    }
    pushWhenMissing(errors, value.passiveLearning.lastObservedDelta.explanation.length > 0, "current profile brain status passiveLearning.lastObservedDelta.explanation is required");
    pushWhenMissing(errors, value.passiveLearning.detail.length > 0, "current profile brain status passiveLearning.detail is required");
    pushWhenMissing(errors, value.brainStatus.brainState === value.brain.state, "current profile brain status brainStatus.brainState must match brain.state");
    pushWhenMissing(errors, value.brainStatus.failOpen === (value.brainStatus.serveState === "fail_open_static_context"), "current profile brain status brainStatus.failOpen must match brainStatus.serveState");
    pushWhenMissing(errors, value.passiveLearning.learnerRunning === (value.passiveLearning.watchState === "watching"), "current profile brain status passiveLearning.learnerRunning must match passiveLearning.watchState");
    pushWhenMissing(errors, value.passiveLearning.firstExportOccurred === (value.passiveLearning.exportState !== "awaiting_first_export"), "current profile brain status passiveLearning.firstExportOccurred must match passiveLearning.exportState");
    pushWhenMissing(errors, value.passiveLearning.currentServingPackId === value.brain.activePackId, "current profile brain status passiveLearning.currentServingPackId must match brain.activePackId");
    pushWhenMissing(errors, value.passiveLearning.lastExportAt === value.brain.lastExportAt, "current profile brain status passiveLearning.lastExportAt must match brain.lastExportAt");
    pushWhenMissing(errors, value.passiveLearning.lastPromotionAt === value.brain.lastPromotionAt, "current profile brain status passiveLearning.lastPromotionAt must match brain.lastPromotionAt");
    if (value.passiveLearning.lastObservedDelta.available === false) {
        for (const fieldValue of [
            value.passiveLearning.lastObservedDelta.exported,
            value.passiveLearning.lastObservedDelta.labeled,
            value.passiveLearning.lastObservedDelta.promoted,
            value.passiveLearning.lastObservedDelta.served
        ]) {
            pushWhenMissing(errors, fieldValue === null, "current profile brain status passiveLearning.lastObservedDelta boolean fields must be null when lastObservedDelta.available=false");
        }
    }
    if (value.passiveLearning.lastObservedDelta.served === true) {
        pushWhenMissing(errors, value.passiveLearning.lastObservedDelta.promoted === true, "current profile brain status passiveLearning.lastObservedDelta.served requires promoted=true");
    }
    if (value.passiveLearning.lastObservedDelta.latestPackTransition?.kind === "promoted_active") {
        pushWhenMissing(errors, value.passiveLearning.lastObservedDelta.promoted === true, "current profile brain status passiveLearning.lastObservedDelta.latestPackTransition.kind=promoted_active requires promoted=true");
    }
    if (value.passiveLearning.lastObservedDelta.latestPackTransition?.kind === "staged_candidate") {
        pushWhenMissing(errors, value.passiveLearning.lastObservedDelta.promoted !== true, "current profile brain status passiveLearning.lastObservedDelta.latestPackTransition.kind=staged_candidate must not imply promoted=true");
    }
    const expectedStructuralDecisionOrigin = value.brainStatus.structuralDecision.basis === "caller_override"
        ? "manual_caller_shape"
        : value.brainStatus.structuralDecision.basis === "compile_structural_signals" ||
            value.brainStatus.structuralDecision.basis === "graph_evolution"
            ? "empirical_control"
            : value.brainStatus.structuralDecision.basis === "unknown"
                ? "unknown"
                : "default_path_control";
    pushWhenMissing(errors, value.brainStatus.structuralDecision.origin === expectedStructuralDecisionOrigin, "current profile brain status brainStatus.structuralDecision.origin must match structuralDecision.basis");
    const expectedStatus = value.hook.desynced ||
        value.attachment.state === "not_attached" ||
        value.brainStatus.serveState === "fail_open_static_context" ||
        value.brainStatus.serveState === "hard_fail"
        ? "fail"
        : value.attachment.state === "unknown" ||
            value.brainStatus.awaitingFirstExport ||
            value.brainStatus.serveState === "unprobed" ||
            value.brain.routeFreshness !== "updated"
            ? "warn"
            : "ok";
    pushWhenMissing(errors, value.brainStatus.status === expectedStatus, `current profile brain status brainStatus.status must be ${expectedStatus} for the reported attachment/serve/freshness state`);
    if (value.attachment.activationRoot !== null) {
        pushWhenMissing(errors, value.attachment.activationRoot.length > 0, "current profile brain status attachment.activationRoot must be null or non-empty");
    }
    if (value.attachment.watchOnly) {
        pushWhenMissing(errors, value.attachment.state !== "attached", "current profile brain status attachment.watchOnly must not be true when state=attached");
    }
    if (value.attachment.state === "attached") {
        pushWhenMissing(errors, value.attachment.activationRoot !== null, "attached current profile brain status requires attachment.activationRoot");
        pushWhenMissing(errors, value.attachment.proofState === "self_proving", "attached current profile brain status must be self-proving");
        pushWhenMissing(errors, value.hook.scope === "exact_openclaw_home" &&
            (value.hook.installState === "installed" || value.hook.installState === "blocked_by_allowlist"), "attached current profile brain status requires an exact-home installed hook surface");
    }
    if (value.attachment.state === "not_attached") {
        pushWhenMissing(errors, value.attachment.proofState === "self_proving", "detached current profile brain status must be self-proving");
        pushWhenMissing(errors, value.hook.scope === "exact_openclaw_home" && value.hook.installState === "not_installed", "detached current profile brain status requires an exact-home not_installed hook surface");
    }
    if (value.attachment.state === "unknown") {
        pushWhenMissing(errors, value.attachment.proofState === "activation_root_only", "unknown current profile attachment state requires activation_root_only proof");
        pushWhenMissing(errors, value.hook.scope === "activation_root_only", "unknown current profile attachment state requires activation-root-only hook scope");
    }
    if (value.hook.scope === "exact_openclaw_home") {
        pushWhenMissing(errors, value.hook.openclawHome !== null, "exact-home hook scope requires hook.openclawHome");
    }
    else {
        pushWhenMissing(errors, value.hook.openclawHome === null, "activation-root-only hook scope requires hook.openclawHome=null");
        pushWhenMissing(errors, value.hook.installState === "unverified", "activation-root-only hook scope requires installState=unverified");
        pushWhenMissing(errors, value.hook.loadability === "unverified", "activation-root-only hook scope requires loadability=unverified");
    }
    if (value.hook.installState === "blocked_by_allowlist") {
        pushWhenMissing(errors, value.hook.desynced === true, "blocked hook installState requires hook.desynced=true");
        pushWhenMissing(errors, value.hook.loadability === "blocked", "blocked hook installState requires hook.loadability=blocked");
    }
    if (value.hook.loadProof === "status_probe_ready") {
        pushWhenMissing(errors, value.hook.loadability === "loadable", "hook.loadProof=status_probe_ready requires hook.loadability=loadable");
        pushWhenMissing(errors, value.brainStatus.serveState === "serving_active_pack", "hook.loadProof=status_probe_ready requires brainStatus.serveState=serving_active_pack");
    }
    if (value.brainStatus.serveState === "serving_active_pack") {
        pushWhenMissing(errors, value.attachment.state !== "not_attached", "serving current profile brain status must not claim not_attached");
        pushWhenMissing(errors, value.brain.activePackId !== null, "serving current profile brain status requires brain.activePackId");
    }
    if (value.brainStatus.awaitingFirstExport) {
        pushWhenMissing(errors, value.brainStatus.serveState === "serving_active_pack", "current profile brain status brainStatus.awaitingFirstExport requires serveState=serving_active_pack");
    }
    if (value.brainStatus.activationState === "healthy_seed") {
        pushWhenMissing(errors, value.attachment.state !== "not_attached" &&
            value.brainStatus.serveState === "serving_active_pack" &&
            value.brain.state === "seed_state_authoritative" &&
            value.brain.activePackId !== null &&
            value.brainStatus.awaitingFirstExport === false, "current profile brain status brainStatus.activationState=healthy_seed requires a non-detached seed-state serving pack after the first export");
    }
    if (value.brainStatus.activationState === "awaiting_first_export") {
        pushWhenMissing(errors, value.attachment.state !== "not_attached" &&
            value.brainStatus.serveState === "serving_active_pack" &&
            value.brain.state === "seed_state_authoritative" &&
            value.brain.activePackId !== null &&
            value.brainStatus.awaitingFirstExport === true, "current profile brain status brainStatus.activationState=awaiting_first_export requires a non-detached seed-state serving pack with awaitingFirstExport=true");
    }
    if (value.brainStatus.activationState === "active_promoted") {
        pushWhenMissing(errors, value.attachment.state !== "not_attached" &&
            value.brainStatus.serveState === "serving_active_pack" &&
            value.brain.state === "pg_promoted_pack_authoritative" &&
            value.brain.activePackId !== null &&
            value.brainStatus.awaitingFirstExport === false, "current profile brain status brainStatus.activationState=active_promoted requires a non-detached promoted serving pack");
    }
    if (value.brainStatus.activationState === "stale_incomplete") {
        pushWhenMissing(errors, value.attachment.state !== "attached" &&
            value.brain.activePackId === null &&
            value.brainStatus.awaitingFirstExport === false, "current profile brain status brainStatus.activationState=stale_incomplete requires a non-serving activation root with no active pack");
    }
    if (value.brainStatus.activationState === "detached") {
        pushWhenMissing(errors, value.attachment.state !== "attached" &&
            value.brain.activePackId === null &&
            value.brainStatus.awaitingFirstExport === false, "current profile brain status brainStatus.activationState=detached requires a detached non-serving state");
    }
    if (value.currentTurnAttribution !== null) {
        errors.push(...validateProfileTurnAttribution(value.currentTurnAttribution));
        pushWhenMissing(errors, value.currentTurnAttribution.profileSelector === value.profile.selector, "current profile brain status currentTurnAttribution.profileSelector must match profile.selector");
        pushWhenMissing(errors, value.currentTurnAttribution.profileId === value.profile.profileId, "current profile brain status currentTurnAttribution.profileId must match profile.profileId");
        pushWhenMissing(errors, value.currentTurnAttribution.hostRuntimeOwner === value.host.runtimeOwner, "current profile brain status currentTurnAttribution.hostRuntimeOwner must match host.runtimeOwner");
        pushWhenMissing(errors, value.currentTurnAttribution.brainAttachmentPolicy === value.attachment.policyMode, "current profile brain status currentTurnAttribution.brainAttachmentPolicy must match attachment.policyMode");
    }
    return errors;
}
export function validateNormalizedEventSource(value) {
    const errors = [];
    pushWhenMissing(errors, value.runtimeOwner === "openclaw", "normalized events must declare runtimeOwner=openclaw");
    pushWhenMissing(errors, value.stream.length > 0, "normalized events require a source stream");
    return errors;
}
export function validateEventSemanticMetadata(value) {
    const errors = [];
    pushWhenMissing(errors, value.semanticType === "memory_candidate" ||
        value.semanticType === "teacher_signal" ||
        value.semanticType === "instructional_scaffolding" ||
        value.semanticType === "control_signal" ||
        value.semanticType === "delivery_residue" ||
        value.semanticType === "observability_residue", "event semanticType must be memory_candidate, teacher_signal, instructional_scaffolding, control_signal, delivery_residue, or observability_residue");
    pushWhenMissing(errors, value.sourceKind === "runtime_turn" ||
        value.sourceKind === "session_store" ||
        value.sourceKind === "scanner_export" ||
        value.sourceKind === "recorded_session_seed", "event sourceKind must be runtime_turn, session_store, scanner_export, or recorded_session_seed");
    if (value.diagnosticIntent !== undefined) {
        pushWhenMissing(errors, value.diagnosticIntent === "compile_observability" || value.diagnosticIntent === "delivery_observability", "event diagnosticIntent must be compile_observability or delivery_observability when set");
    }
    return errors;
}
export function validateEventSemanticSurface(value) {
    const errors = [];
    const uniqueSemanticTypes = new Set(value.semanticTypes);
    if (uniqueSemanticTypes.size !== value.semanticTypes.length) {
        errors.push("event semantic surface semanticTypes must be unique");
    }
    for (const semanticType of value.semanticTypes) {
        errors.push(...validateEventSemanticMetadata({ semanticType, sourceKind: "runtime_turn" }).filter((message) => message.startsWith("event semanticType")));
    }
    const uniqueSourceKinds = new Set(value.sourceKinds);
    if (uniqueSourceKinds.size !== value.sourceKinds.length) {
        errors.push("event semantic surface sourceKinds must be unique");
    }
    for (const sourceKind of value.sourceKinds) {
        errors.push(...validateEventSemanticMetadata({ semanticType: "memory_candidate", sourceKind }).filter((message) => message.startsWith("event sourceKind")));
    }
    const uniqueDiagnosticIntents = new Set(value.diagnosticIntents);
    if (uniqueDiagnosticIntents.size !== value.diagnosticIntents.length) {
        errors.push("event semantic surface diagnosticIntents must be unique");
    }
    for (const diagnosticIntent of value.diagnosticIntents) {
        errors.push(...validateEventSemanticMetadata({
            semanticType: "memory_candidate",
            sourceKind: "runtime_turn",
            diagnosticIntent
        }).filter((message) => message.startsWith("event diagnosticIntent")));
    }
    return errors;
}
export function validatePrincipalScope(value) {
    const errors = [];
    pushWhenMissing(errors, value.kind === "global" ||
        value.kind === "profile" ||
        value.kind === "session" ||
        value.kind === "interaction" ||
        value.kind === "message", "principal scope kind must be global, profile, session, interaction, or message");
    if (value.profileSelector !== undefined) {
        pushWhenMissing(errors, value.profileSelector.length > 0, "principal scope profileSelector must be non-empty when set");
    }
    if (value.sessionId !== undefined) {
        pushWhenMissing(errors, value.sessionId.length > 0, "principal scope sessionId must be non-empty when set");
    }
    if (value.interactionId !== undefined) {
        pushWhenMissing(errors, value.interactionId.length > 0, "principal scope interactionId must be non-empty when set");
    }
    if (value.messageId !== undefined) {
        pushWhenMissing(errors, value.messageId.length > 0, "principal scope messageId must be non-empty when set");
    }
    if (value.scopeKey !== undefined) {
        pushWhenMissing(errors, value.scopeKey.length > 0, "principal scope scopeKey must be non-empty when set");
    }
    return errors;
}
export function validatePrincipalMetadata(value) {
    const errors = [];
    pushWhenMissing(errors, value.teacherIdentity.length > 0, "principal teacherIdentity is required");
    pushWhenMissing(errors, value.teacherRole === "principal" ||
        value.teacherRole === "admin" ||
        value.teacherRole === "operator" ||
        value.teacherRole === "user" ||
        value.teacherRole === "assistant" ||
        value.teacherRole === "system", "principal teacherRole must be principal, admin, operator, user, assistant, or system");
    pushWhenMissing(errors, value.teacherAuthority === "binding" ||
        value.teacherAuthority === "primary_human" ||
        value.teacherAuthority === "high" ||
        value.teacherAuthority === "normal" ||
        value.teacherAuthority === "background", "principal teacherAuthority must be binding, primary_human, high, normal, or background");
    pushWhenMissing(errors, value.priorityClass === "critical" || value.priorityClass === "high" || value.priorityClass === "normal" || value.priorityClass === "low", "principal priorityClass must be critical, high, normal, or low");
    errors.push(...validatePrincipalScope(value.principalScope));
    if (value.supersedes !== undefined) {
        pushWhenMissing(errors, value.supersedes.length > 0, "principal supersedes must not be empty when set");
        pushWhenMissing(errors, new Set(value.supersedes.filter((entry) => entry.length > 0)).size === value.supersedes.length, "principal supersedes must contain unique non-empty entries");
    }
    return errors;
}
export function validateRuntimeTurnAttribution(value) {
    const errors = [];
    pushWhenMissing(errors, value.hostRuntimeOwner === "openclaw", "runtime turn attribution hostRuntimeOwner must be openclaw");
    pushWhenMissing(errors, value.profileSelector.length > 0, "runtime turn attribution profileSelector must be non-empty");
    if (value.profileId !== null) {
        pushWhenMissing(errors, value.profileId.length > 0, "runtime turn attribution profileId must be null or non-empty when set");
    }
    pushWhenMissing(errors, value.brainAttachmentPolicy === "undeclared" ||
        value.brainAttachmentPolicy === "dedicated" ||
        value.brainAttachmentPolicy === "shared", "runtime turn attribution brainAttachmentPolicy must be undeclared, dedicated, or shared");
    pushWhenMissing(errors, value.brainStatus === "serving_active_pack" ||
        value.brainStatus === "fail_open_static_context" ||
        value.brainStatus === "hard_fail", "runtime turn attribution brainStatus must be serving_active_pack, fail_open_static_context, or hard_fail");
    if (value.activePackId !== null) {
        pushWhenMissing(errors, value.activePackId.length > 0, "runtime turn attribution activePackId must be non-empty when set");
    }
    if (value.routerIdentity !== null) {
        pushWhenMissing(errors, value.routerIdentity.length > 0, "runtime turn attribution routerIdentity must be non-empty when set");
    }
    if (value.selectionDigest !== null) {
        pushWhenMissing(errors, value.selectionDigest.length > 0, "runtime turn attribution selectionDigest must be non-empty when set");
    }
    if (value.selectionTiers !== null) {
        pushWhenMissing(errors, value.selectionTiers.length > 0, "runtime turn attribution selectionTiers must be non-empty when set");
    }
    errors.push(...validateRuntimeContextFingerprint(value.contextFingerprint));
    if (value.selectionDigest !== value.contextFingerprint.selectionDigest) {
        errors.push("runtime turn attribution contextFingerprint.selectionDigest must match selectionDigest");
    }
    if (value.contextEvidence !== null) {
        pushWhenMissing(errors, value.contextEvidence === "route_fn_and_brain_context" ||
            value.contextEvidence === "brain_context_only" ||
            value.contextEvidence === "route_fn_only" ||
            value.contextEvidence === "stable_kernel_only" ||
            value.contextEvidence === "fail_open_static_context" ||
            value.contextEvidence === "hard_fail", "runtime turn attribution contextEvidence must be an explicit served-turn evidence state when set");
    }
    return errors;
}
export function validateInteractionEvent(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.interactionEvents, "interaction_events.v1 contract is required");
    pushWhenMissing(errors, value.eventId.length > 0, "eventId is required");
    pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
    pushWhenMissing(errors, value.sessionId.length > 0, "sessionId is required");
    pushWhenMissing(errors, value.channel.length > 0, "channel is required");
    pushWhenMissing(errors, value.sequence >= 0, "sequence must be non-negative");
    pushWhenMissing(errors, value.kind === "memory_compiled" || value.kind === "message_delivered" || value.kind === "operator_override", "interaction kind must be explicit");
    pushWhenMissing(errors, isIsoDate(value.createdAt), "createdAt must be an ISO timestamp");
    errors.push(...validateNormalizedEventSource(value.source));
    if (value.semantic !== undefined) {
        errors.push(...validateEventSemanticMetadata(value.semantic));
    }
    if (value.principal !== undefined) {
        errors.push(...validatePrincipalMetadata(value.principal));
    }
    if (value.attribution !== undefined) {
        errors.push(...validateRuntimeTurnAttribution(value.attribution));
    }
    return errors;
}
export function validateFeedbackEvent(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.feedbackEvents, "feedback_events.v1 contract is required");
    pushWhenMissing(errors, value.eventId.length > 0, "eventId is required");
    pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
    pushWhenMissing(errors, value.sessionId.length > 0, "sessionId is required");
    pushWhenMissing(errors, value.channel.length > 0, "channel is required");
    pushWhenMissing(errors, value.sequence >= 0, "sequence must be non-negative");
    pushWhenMissing(errors, value.kind === "correction" || value.kind === "teaching" || value.kind === "approval" || value.kind === "suppression", "feedback kind must be explicit");
    pushWhenMissing(errors, value.content.length > 0, "content is required");
    pushWhenMissing(errors, isIsoDate(value.createdAt), "createdAt must be an ISO timestamp");
    errors.push(...validateNormalizedEventSource(value.source));
    if (value.semantic !== undefined) {
        errors.push(...validateEventSemanticMetadata(value.semantic));
    }
    if (value.principal !== undefined) {
        errors.push(...validatePrincipalMetadata(value.principal));
    }
    if (value.attribution !== undefined) {
        errors.push(...validateRuntimeTurnAttribution(value.attribution));
    }
    return errors;
}
export function validateNormalizedEventRange(value) {
    const errors = [];
    pushWhenMissing(errors, value.start >= 0 || value.count === 0, "eventRange.start must be non-negative when events exist");
    pushWhenMissing(errors, value.count >= 0, "eventRange.count must be non-negative");
    if (value.count === 0) {
        if (value.end !== value.start - 1) {
            errors.push("empty event ranges must use end=start-1");
        }
        if (value.firstEventId !== null || value.lastEventId !== null) {
            errors.push("empty event ranges must not set event ids");
        }
        if (value.firstCreatedAt !== null || value.lastCreatedAt !== null) {
            errors.push("empty event ranges must not set timestamps");
        }
        return errors;
    }
    pushWhenMissing(errors, value.end >= value.start, "eventRange.end must be >= start");
    // `count` is event cardinality, not inclusive sequence width. Distinct
    // normalized events may legitimately share the same numeric sequence.
    const hasExplicitBoundaryMetadata = value.firstEventId !== null || value.lastEventId !== null || value.firstCreatedAt !== null || value.lastCreatedAt !== null;
    if (hasExplicitBoundaryMetadata) {
        pushWhenMissing(errors, value.firstEventId !== null, "eventRange.firstEventId is required when boundary metadata is present");
        pushWhenMissing(errors, value.lastEventId !== null, "eventRange.lastEventId is required when boundary metadata is present");
        pushWhenMissing(errors, value.firstCreatedAt !== null, "eventRange.firstCreatedAt is required when boundary metadata is present");
        pushWhenMissing(errors, value.lastCreatedAt !== null, "eventRange.lastCreatedAt is required when boundary metadata is present");
    }
    if (value.firstCreatedAt !== null && !isIsoDate(value.firstCreatedAt)) {
        errors.push("eventRange.firstCreatedAt must be an ISO timestamp");
    }
    if (value.lastCreatedAt !== null && !isIsoDate(value.lastCreatedAt)) {
        errors.push("eventRange.lastCreatedAt must be an ISO timestamp");
    }
    if (value.firstCreatedAt !== null && value.lastCreatedAt !== null && value.lastCreatedAt < value.firstCreatedAt) {
        errors.push("eventRange timestamps must be monotonic");
    }
    return errors;
}
export function validateTeacherSupervisionArtifact(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.teacherSupervisionArtifact, "teacher_supervision_artifact.v1 contract is required");
    pushWhenMissing(errors, value.artifactId.length > 0, "teacher supervision artifactId is required");
    pushWhenMissing(errors, value.dedupId.length > 0, "teacher supervision dedupId is required");
    pushWhenMissing(errors, value.kind === "correction" ||
        value.kind === "teaching" ||
        value.kind === "approval" ||
        value.kind === "suppression" ||
        value.kind === "operator_override", "teacher supervision kind must be explicit");
    pushWhenMissing(errors, isIsoDate(value.createdAt), "teacher supervision createdAt must be an ISO timestamp");
    pushWhenMissing(errors, value.source.runtimeOwner === "openclaw", "teacher supervision source runtimeOwner must be openclaw");
    pushWhenMissing(errors, value.source.sessionId.length > 0, "teacher supervision source sessionId is required");
    pushWhenMissing(errors, value.source.channel.length > 0, "teacher supervision source channel is required");
    pushWhenMissing(errors, value.source.sourceStreams.length > 0, "teacher supervision sourceStreams must not be empty");
    pushWhenMissing(errors, new Set(value.source.sourceStreams.filter((stream) => stream.length > 0)).size === value.source.sourceStreams.length, "teacher supervision sourceStreams must contain unique non-empty entries");
    pushWhenMissing(errors, value.source.eventRange.count > 0, "teacher supervision eventRange.count must be positive");
    pushWhenMissing(errors, value.source.eventRange.start >= 0, "teacher supervision eventRange.start must be non-negative");
    pushWhenMissing(errors, value.source.eventRange.end >= value.source.eventRange.start, "teacher supervision eventRange.end must be >= start");
    pushWhenMissing(errors, value.source.eventExportDigest.length > 0, "teacher supervision eventExportDigest is required");
    pushWhenMissing(errors, value.sourceEventIds.length > 0, "teacher supervision sourceEventIds must not be empty");
    pushWhenMissing(errors, new Set(value.sourceEventIds.filter((eventId) => eventId.length > 0)).size === value.sourceEventIds.length, "teacher supervision sourceEventIds must contain unique non-empty entries");
    if (value.relatedInteractionId !== null) {
        pushWhenMissing(errors, value.relatedInteractionId.length > 0, "teacher supervision relatedInteractionId must be non-empty when set");
    }
    if (value.principal !== undefined) {
        errors.push(...validatePrincipalMetadata(value.principal));
    }
    pushWhenMissing(errors, value.content.length > 0, "teacher supervision content is required");
    pushWhenMissing(errors, value.freshness.status === "fresh" || value.freshness.status === "stale", "teacher supervision freshness.status must be fresh or stale");
    pushWhenMissing(errors, isIsoDate(value.freshness.observedAt), "teacher supervision freshness.observedAt must be an ISO timestamp");
    pushWhenMissing(errors, isIsoDate(value.freshness.newestSourceCreatedAt), "teacher supervision freshness.newestSourceCreatedAt must be an ISO timestamp");
    pushWhenMissing(errors, value.freshness.ageMs >= 0, "teacher supervision freshness.ageMs must be non-negative");
    pushWhenMissing(errors, value.freshness.staleAfterMs > 0, "teacher supervision freshness.staleAfterMs must be positive");
    return errors;
}
export function validateWorkspaceMetadata(value) {
    const errors = [];
    pushWhenMissing(errors, value.workspaceId.length > 0, "workspaceId is required");
    pushWhenMissing(errors, value.snapshotId.length > 0, "snapshotId is required");
    pushWhenMissing(errors, isIsoDate(value.capturedAt), "workspace capturedAt must be an ISO timestamp");
    pushWhenMissing(errors, value.rootDir.length > 0, "workspace rootDir is required");
    if (value.branch !== null && value.branch.length === 0) {
        errors.push("workspace branch must be null or non-empty");
    }
    if (value.revision !== null && value.revision.length === 0) {
        errors.push("workspace revision must be null or non-empty");
    }
    if (value.manifestDigest !== null && value.manifestDigest.length === 0) {
        errors.push("workspace manifestDigest must be null or non-empty");
    }
    if (value.labels.some((label) => label.length === 0)) {
        errors.push("workspace labels must be non-empty");
    }
    if (value.files.some((file) => file.length === 0)) {
        errors.push("workspace files must be non-empty");
    }
    return errors;
}
export function validateLearningSurface(value) {
    const errors = [];
    pushWhenMissing(errors, value.bootProfile === "fast_boot_defaults", "learning surface bootProfile must be fast_boot_defaults");
    pushWhenMissing(errors, value.learningCadence === "passive_background", "learning surface learningCadence must be passive_background");
    pushWhenMissing(errors, value.scanPolicy === "always_on", "learning surface scanPolicy must be always_on");
    pushWhenMissing(errors, value.scanSurfaces.length > 0, "learning surface requires at least one scan surface");
    for (const surface of value.scanSurfaces) {
        pushWhenMissing(errors, surface.length > 0, "learning surface scan surfaces must be non-empty");
    }
    for (const source of value.labelSources.human) {
        pushWhenMissing(errors, source.length > 0, "learning surface human label sources must be non-empty");
    }
    for (const source of value.labelSources.self) {
        pushWhenMissing(errors, source.length > 0, "learning surface self label sources must be non-empty");
    }
    pushWhenMissing(errors, value.labelHarvest.humanLabels >= 0, "learning surface humanLabels must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.selfLabels >= 0, "learning surface selfLabels must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.corrections >= 0, "learning surface corrections must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.teachings >= 0, "learning surface teachings must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.approvals >= 0, "learning surface approvals must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.suppressions >= 0, "learning surface suppressions must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.operatorOverrideLabels >= 0, "learning surface operatorOverrideLabels must be non-negative");
    pushWhenMissing(errors, value.labelHarvest.memoryCompileLabels >= 0, "learning surface memoryCompileLabels must be non-negative");
    const humanLabels = value.labelHarvest.corrections +
        value.labelHarvest.teachings +
        value.labelHarvest.approvals +
        value.labelHarvest.suppressions +
        value.labelHarvest.operatorOverrideLabels;
    const selfLabels = value.labelHarvest.memoryCompileLabels;
    if (value.labelHarvest.humanLabels !== humanLabels) {
        errors.push("learning surface humanLabels must equal feedback and operatorOverride labels");
    }
    if (value.labelHarvest.selfLabels !== selfLabels) {
        errors.push("learning surface selfLabels must equal memoryCompileLabels");
    }
    for (const identity of value.principalSummary.teacherIdentities) {
        pushWhenMissing(errors, identity.length > 0, "learning surface principal teacherIdentities must be non-empty");
    }
    for (const role of value.principalSummary.teacherRoles) {
        pushWhenMissing(errors, role === "principal" || role === "admin" || role === "operator" || role === "user" || role === "assistant" || role === "system", "learning surface principal teacherRoles must be explicit");
    }
    for (const authority of value.principalSummary.teacherAuthorities) {
        pushWhenMissing(errors, authority === "binding" ||
            authority === "primary_human" ||
            authority === "high" ||
            authority === "normal" ||
            authority === "background", "learning surface principal teacherAuthorities must be explicit");
    }
    for (const priorityClass of value.principalSummary.priorityClasses) {
        pushWhenMissing(errors, priorityClass === "critical" || priorityClass === "high" || priorityClass === "normal" || priorityClass === "low", "learning surface principal priorityClasses must be explicit");
    }
    pushWhenMissing(errors, value.principalSummary.scopedEventCount >= 0, "learning surface principal scopedEventCount must be non-negative");
    pushWhenMissing(errors, value.principalSummary.supersedingEventCount >= 0, "learning surface principal supersedingEventCount must be non-negative");
    return errors;
}
export function validatePackBlockLearningSignals(value, blockId) {
    const errors = [];
    const prefix = blockId === undefined ? "pack block learning" : `pack block ${blockId}`;
    const allowedRoles = [
        "boot_default",
        "background_expectation",
        "label_surface",
        "workspace",
        "structural",
        "interaction",
        "feedback",
        "teacher_supervision"
    ];
    pushWhenMissing(errors, allowedRoles.includes(value.role), `${prefix} role must be explicit`);
    pushWhenMissing(errors, value.humanLabels >= 0, `${prefix} humanLabels must be non-negative`);
    pushWhenMissing(errors, value.selfLabels >= 0, `${prefix} selfLabels must be non-negative`);
    pushWhenMissing(errors, value.hebbianPulse >= 0, `${prefix} hebbianPulse must be non-negative`);
    if (value.decayHalfLifeDays !== null) {
        pushWhenMissing(errors, value.decayHalfLifeDays >= 0, `${prefix} decayHalfLifeDays must be non-negative when set`);
    }
    return errors;
}
export function validateEventExportProvenance(value, eventRange) {
    const errors = [];
    const emptyExportAllowed = value.sourceStreams.length === 0 &&
        value.interactionCount === 0 &&
        value.feedbackCount === 0 &&
        (eventRange?.count ?? 0) === 0;
    pushWhenMissing(errors, value.runtimeOwner === "openclaw", "event export provenance requires runtimeOwner=openclaw");
    pushWhenMissing(errors, value.interactionCount >= 0, "interactionCount must be non-negative");
    pushWhenMissing(errors, value.feedbackCount >= 0, "feedbackCount must be non-negative");
    pushWhenMissing(errors, value.exportDigest.length > 0, "event export provenance requires exportDigest");
    pushWhenMissing(errors, value.sourceStreams.length > 0 || emptyExportAllowed, "event export provenance requires at least one source stream unless the export is empty");
    if (value.semanticSurface !== undefined) {
        errors.push(...validateEventSemanticSurface(value.semanticSurface));
    }
    errors.push(...validateLearningSurface(value.learningSurface));
    if (emptyExportAllowed) {
        pushWhenMissing(errors, value.learningSurface.scanSurfaces.includes("event_export:empty"), "empty event export provenance must include event_export:empty scan surface");
    }
    for (const stream of value.sourceStreams) {
        pushWhenMissing(errors, stream.length > 0, "event export provenance source streams must be non-empty");
        if (!value.learningSurface.scanSurfaces.some((surface) => surface.startsWith(`${stream}:`))) {
            errors.push(`event export provenance learningSurface must include a scan surface for ${stream}`);
        }
    }
    for (const contract of value.contracts) {
        pushWhenMissing(errors, contract === CONTRACT_IDS.interactionEvents || contract === CONTRACT_IDS.feedbackEvents, "event export provenance contracts must be event contracts");
    }
    if (value.sessionId !== null) {
        pushWhenMissing(errors, value.sessionId.length > 0, "event export provenance sessionId must be non-empty when set");
    }
    if (value.channel !== null) {
        pushWhenMissing(errors, value.channel.length > 0, "event export provenance channel must be non-empty when set");
    }
    if (eventRange !== undefined) {
        if (value.interactionCount + value.feedbackCount !== eventRange.count) {
            errors.push("event export provenance counts must match eventRange.count");
        }
        if (value.learningSurface.labelHarvest.humanLabels + value.learningSurface.labelHarvest.selfLabels > eventRange.count) {
            errors.push("learning surface labels cannot exceed eventRange.count");
        }
    }
    return errors;
}
export function validateNormalizedEventExport(value) {
    const errors = [
        ...value.interactionEvents.flatMap((event) => validateInteractionEvent(event)),
        ...value.feedbackEvents.flatMap((event) => validateFeedbackEvent(event)),
        ...validateNormalizedEventRange(value.range),
        ...validateEventExportProvenance(value.provenance, value.range),
        ...eventSequenceErrors([...value.interactionEvents, ...value.feedbackEvents])
    ];
    const rebuilt = buildNormalizedEventExport(value);
    if (canonicalJson(rebuilt.range) !== canonicalJson(value.range)) {
        errors.push("normalized event export range does not match the supplied events");
    }
    const expectedProvenance = value.provenance.semanticSurface === undefined
        ? { ...rebuilt.provenance, semanticSurface: undefined }
        : rebuilt.provenance;
    if (canonicalJson(expectedProvenance) !== canonicalJson(value.provenance)) {
        errors.push("normalized event export provenance does not match the supplied events");
    }
    return errors;
}
export function validateArtifactManifest(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.artifactManifest, "artifact_manifest.v1 contract is required");
    pushWhenMissing(errors, value.packId.length > 0, "packId is required");
    pushWhenMissing(errors, value.immutable === true, "pack manifests must be immutable");
    pushWhenMissing(errors, value.runtimeAssets.graphPath.length > 0, "graphPath is required");
    pushWhenMissing(errors, value.runtimeAssets.vectorPath.length > 0, "vectorPath is required");
    pushWhenMissing(errors, value.payloadChecksums.graph.length > 0, "graph checksum is required");
    pushWhenMissing(errors, value.payloadChecksums.vector.length > 0, "vector checksum is required");
    pushWhenMissing(errors, isIsoDate(value.provenance.builtAt), "builtAt must be an ISO timestamp");
    pushWhenMissing(errors, value.provenance.workspaceSnapshot.length > 0, "workspaceSnapshot is required");
    errors.push(...validateRouteArtifactReference(value.routeArtifact, { routePolicy: value.routePolicy, label: "routeArtifact" }));
    errors.push(...validateWorkspaceMetadata(value.provenance.workspace));
    errors.push(...validateLearningSurface(value.provenance.learningSurface));
    pushWhenMissing(errors, value.graphDynamics.bootstrapping.fastBootDefaults === true, "graph bootstrapping fastBootDefaults must stay enabled");
    pushWhenMissing(errors, value.graphDynamics.bootstrapping.passiveBackgroundLearning === true, "graph bootstrapping passiveBackgroundLearning must stay enabled");
    pushWhenMissing(errors, value.graphDynamics.runtimePlasticitySource === "candidate_build" || value.graphDynamics.runtimePlasticitySource === "live_loop", "graph runtimePlasticitySource must be candidate_build or live_loop");
    pushWhenMissing(errors, value.graphDynamics.hebbian.learningRate >= 0, "hebbian learningRate must be non-negative");
    pushWhenMissing(errors, value.graphDynamics.decay.halfLifeDays >= 0, "decay halfLifeDays must be non-negative");
    errors.push(...validateNormalizedEventRange(value.provenance.eventRange));
    if (value.provenance.workspace.snapshotId !== value.provenance.workspaceSnapshot) {
        errors.push("workspaceSnapshot must match provenance.workspace.snapshotId");
    }
    if (value.routeArtifact.assetKind !== value.runtimeAssets.router.kind) {
        errors.push(`routeArtifact assetKind ${value.routeArtifact.assetKind} does not match runtimeAssets.router.kind ${value.runtimeAssets.router.kind}`);
    }
    if ((value.routeArtifact.routerIdentity ?? null) !== (value.runtimeAssets.router.identity ?? null)) {
        errors.push(`routeArtifact routerIdentity ${value.routeArtifact.routerIdentity ?? "null"} does not match runtimeAssets.router.identity ${value.runtimeAssets.router.identity ?? "null"}`);
    }
    if ((value.routeArtifact.routerChecksum ?? null) !== (value.payloadChecksums.router ?? null)) {
        errors.push(`routeArtifact routerChecksum ${value.routeArtifact.routerChecksum ?? "null"} does not match payloadChecksums.router ${value.payloadChecksums.router ?? "null"}`);
    }
    if (value.provenance.eventExports !== null) {
        errors.push(...validateEventExportProvenance(value.provenance.eventExports, value.provenance.eventRange));
        if (canonicalJson(value.provenance.eventExports.learningSurface) !== canonicalJson(value.provenance.learningSurface)) {
            errors.push("artifact provenance learningSurface must match event export learningSurface");
        }
        if (value.routeArtifact.assetKind !== "none" &&
            (value.routeArtifact.eventExportDigest ?? null) !== value.provenance.eventExports.exportDigest) {
            errors.push(`routeArtifact eventExportDigest ${value.routeArtifact.eventExportDigest ?? "null"} does not match provenance event export digest ${value.provenance.eventExports.exportDigest}`);
        }
    }
    if (value.routePolicy === "requires_learned_routing") {
        pushWhenMissing(errors, value.runtimeAssets.router.kind !== "none", "learned-routing packs require a router asset");
        pushWhenMissing(errors, value.runtimeAssets.router.identity !== null, "learned-routing packs require router identity");
        pushWhenMissing(errors, value.payloadChecksums.router !== null, "learned-routing packs require a router checksum");
    }
    if (value.runtimeAssets.router.kind === "none" && value.payloadChecksums.router !== null) {
        errors.push("router checksum must be null when no router asset exists");
    }
    return errors;
}
export function validateActivationPointerRecord(value, expectedSlot) {
    const errors = [];
    const allowedSlots = ["active", "candidate", "previous"];
    pushWhenMissing(errors, allowedSlots.includes(value.slot), "activation pointer slot must be active, candidate, or previous");
    pushWhenMissing(errors, value.packId.length > 0, "activation pointer packId is required");
    pushWhenMissing(errors, value.packRootDir.length > 0, "activation pointer packRootDir is required");
    pushWhenMissing(errors, value.manifestPath.length > 0, "activation pointer manifestPath is required");
    pushWhenMissing(errors, value.manifestPath.endsWith(".json"), "activation pointer manifestPath must target json");
    pushWhenMissing(errors, value.manifestDigest.startsWith("sha256-"), "activation pointer manifestDigest must be a sha256 digest");
    pushWhenMissing(errors, value.routePolicy === "heuristic_allowed" || value.routePolicy === "requires_learned_routing", "activation pointer routePolicy must be explicit");
    pushWhenMissing(errors, value.workspaceSnapshot.length > 0, "activation pointer workspaceSnapshot is required");
    pushWhenMissing(errors, isIsoDate(value.builtAt), "activation pointer builtAt must be an ISO timestamp");
    pushWhenMissing(errors, isIsoDate(value.updatedAt), "activation pointer updatedAt must be an ISO timestamp");
    pushWhenMissing(errors, value.eventRange.count >= 0, "activation pointer eventRange.count must be non-negative");
    pushWhenMissing(errors, value.eventRange.start >= 0, "activation pointer eventRange.start must be non-negative");
    if (value.eventRange.count === 0) {
        pushWhenMissing(errors, value.eventRange.end === value.eventRange.start - 1, "empty activation pointer eventRange must use end=start-1");
    }
    else {
        pushWhenMissing(errors, value.eventRange.end >= value.eventRange.start, "activation pointer eventRange.end must be >= start");
    }
    if (expectedSlot !== undefined && value.slot !== expectedSlot) {
        errors.push(`activation pointer slot ${value.slot} does not match field ${expectedSlot}`);
    }
    if (value.routePolicy === "requires_learned_routing" && value.routerIdentity === null) {
        errors.push("learned-routing activation pointers require routerIdentity");
    }
    return errors;
}
export function validateActivationPointers(value) {
    const errors = [];
    pushWhenMissing(errors, value.contract === CONTRACT_IDS.activationPointers, "activation_pointers.v1 contract is required");
    const seenPackIds = new Set();
    for (const slot of ["active", "candidate", "previous"]) {
        const record = value[slot];
        if (record === null) {
            continue;
        }
        errors.push(...validateActivationPointerRecord(record, slot));
        if (seenPackIds.has(record.packId)) {
            errors.push(`activation pointers must not reuse packId across slots: ${record.packId}`);
        }
        seenPackIds.add(record.packId);
    }
    return errors;
}
function validateOpenClawMarkdownFileRoleBinding(value, label) {
    const errors = [];
    const allowedRoles = [
        "repo_boundary",
        "claims_boundary",
        "contracts_reference",
        "glossary",
        "learning_policy",
        "agent_sop",
        "attach_quickstart",
        "integration_guide",
        "operator_guide",
        "operator_observability",
        "ops_recipe",
        "session_replay_proof",
        "release_guide",
        "evaluation_reproduction",
        "setup_guide",
        "worked_example"
    ];
    const allowedAudiences = ["runtime", "integrator", "operator", "proof"];
    const allowedTiers = ["core", "supporting"];
    pushWhenMissing(errors, value.path.length > 0, `${label} path is required`);
    pushWhenMissing(errors, allowedRoles.includes(value.role), `${label} role must be explicit`);
    pushWhenMissing(errors, allowedAudiences.includes(value.audience), `${label} audience must be explicit`);
    pushWhenMissing(errors, allowedTiers.includes(value.tier), `${label} tier must be core or supporting`);
    return errors;
}
export function validateOpenClawInitBlockMetadata(value, label = "graph init metadata") {
    const errors = [];
    const allowedNodeKinds = [
        "markdown_ontology",
        "product_invariant",
        "workspace_snapshot",
        "event_export",
        "event",
        "teacher_supervision",
        "synthetic_topology"
    ];
    const allowedSourceKinds = [
        "markdown",
        "product",
        "workspace",
        "event_export",
        "event",
        "teacher_supervision",
        "synthetic"
    ];
    pushWhenMissing(errors, allowedNodeKinds.includes(value.nodeKind), `${label} nodeKind must be explicit`);
    pushWhenMissing(errors, allowedSourceKinds.includes(value.sourceKind), `${label} sourceKind must be explicit`);
    pushWhenMissing(errors, value.fastBootRequired === true, `${label} fastBootRequired must stay enabled`);
    pushWhenMissing(errors, value.passiveBackgroundLearningRequired === true, `${label} passiveBackgroundLearningRequired must stay enabled`);
    pushWhenMissing(errors, value.heuristicScope === "init_priors_and_topology_only", `${label} heuristicScope must stay init_priors_and_topology_only`);
    pushWhenMissing(errors, value.learnedLabelPolicy === "explicit_collected_labels_only", `${label} learnedLabelPolicy must stay explicit_collected_labels_only`);
    if (value.fileRole !== undefined) {
        errors.push(...validateOpenClawMarkdownFileRoleBinding(value.fileRole, `${label} fileRole`));
        if (value.sourceKind !== "markdown") {
            errors.push(`${label} fileRole requires sourceKind=markdown`);
        }
    }
    return errors;
}
export function validateOpenClawInitGraphOntology(value) {
    const errors = [];
    const seenPaths = new Set();
    pushWhenMissing(errors, value.schema === PACK_GRAPH_SCHEMAS.openclawInit, `graph ontology schema must be ${PACK_GRAPH_SCHEMAS.openclawInit}`);
    pushWhenMissing(errors, value.typedMarkdownSurface === true, "graph ontology must treat markdown as typed");
    pushWhenMissing(errors, value.fileRoles.length > 0, "graph ontology requires fileRoles");
    pushWhenMissing(errors, value.fastBootRequired === true, "graph ontology fastBootRequired must stay enabled");
    pushWhenMissing(errors, value.passiveBackgroundLearningRequired === true, "graph ontology passiveBackgroundLearningRequired must stay enabled");
    pushWhenMissing(errors, value.heuristicScope === "init_priors_and_topology_only", "graph ontology heuristicScope must stay init_priors_and_topology_only");
    pushWhenMissing(errors, value.learnedLabelPolicy === "explicit_collected_labels_only", "graph ontology learnedLabelPolicy must stay explicit_collected_labels_only");
    for (const [index, fileRole] of value.fileRoles.entries()) {
        errors.push(...validateOpenClawMarkdownFileRoleBinding(fileRole, `graph ontology fileRoles[${index}]`));
        if (seenPaths.has(fileRole.path)) {
            errors.push(`graph ontology fileRoles must be unique per path: ${fileRole.path}`);
        }
        seenPaths.add(fileRole.path);
    }
    return errors;
}
export function validatePackGraphPayload(value, expectedPackId) {
    const errors = [];
    pushWhenMissing(errors, value.packId.length > 0, "graph packId is required");
    pushWhenMissing(errors, value.blocks.length > 0, "graph must contain at least one context block");
    if (expectedPackId !== undefined && value.packId !== expectedPackId) {
        errors.push(`graph packId ${value.packId} does not match manifest packId ${expectedPackId}`);
    }
    if (value.schema !== undefined) {
        pushWhenMissing(errors, value.schema === PACK_GRAPH_SCHEMAS.openclawInit, `graph schema must be ${PACK_GRAPH_SCHEMAS.openclawInit}`);
    }
    if (value.ontology !== undefined) {
        errors.push(...validateOpenClawInitGraphOntology(value.ontology));
        if (value.schema === undefined) {
            errors.push("graph ontology requires graph schema");
        }
        else if (value.ontology.schema !== value.schema) {
            errors.push(`graph ontology schema ${value.ontology.schema} does not match graph schema ${value.schema}`);
        }
    }
    const seen = new Set();
    const ontologyBindings = new Map((value.ontology?.fileRoles ?? []).map((fileRole) => [fileRole.path, fileRole]));
    for (const block of value.blocks) {
        pushWhenMissing(errors, block.id.length > 0, "graph blocks require id");
        pushWhenMissing(errors, block.source.length > 0, `graph block ${block.id || "<unknown>"} requires source`);
        pushWhenMissing(errors, block.text.length > 0, `graph block ${block.id || "<unknown>"} requires text`);
        pushWhenMissing(errors, block.priority >= 0, `graph block ${block.id || "<unknown>"} priority must be non-negative`);
        pushWhenMissing(errors, block.keywords.length > 0, `graph block ${block.id || "<unknown>"} requires keywords`);
        errors.push(...validatePackBlockLearningSignals(block.learning, block.id));
        if (block.semantic !== undefined) {
            errors.push(...validateEventSemanticMetadata(block.semantic));
        }
        if (block.initSeed !== undefined) {
            errors.push(...validatePackBlockInitSignals(block.initSeed, `graph block ${block.id || "<unknown>"} initSeed`));
        }
        if (block.tokenCount !== undefined) {
            pushWhenMissing(errors, block.tokenCount >= 0, `graph block ${block.id || "<unknown>"} tokenCount must be non-negative`);
        }
        if (block.compactedFrom !== undefined) {
            pushWhenMissing(errors, block.compactedFrom.length > 0, `graph block ${block.id || "<unknown>"} compactedFrom must not be empty`);
            const compactedFrom = new Set(block.compactedFrom.filter((id) => id.length > 0));
            if (compactedFrom.size !== block.compactedFrom.length) {
                errors.push(`graph block ${block.id || "<unknown>"} compactedFrom must contain unique non-empty ids`);
            }
        }
        if (block.state !== undefined) {
            errors.push(...validatePackBlockState(block.state, `graph block ${block.id || "<unknown>"} state`));
        }
        if (block.routing !== undefined) {
            errors.push(...validatePackBlockRoutingHints(block.routing, `graph block ${block.id || "<unknown>"} routing`));
        }
        if (block.edges !== undefined) {
            const seenEdges = new Set();
            for (const [index, edge] of block.edges.entries()) {
                errors.push(...validatePackGraphEdge(edge, `graph block ${block.id || "<unknown>"} edge[${index}]`));
                const edgeIdentity = `${edge.kind}:${edge.targetBlockId}`;
                if (seenEdges.has(edgeIdentity)) {
                    errors.push(`graph block ${block.id || "<unknown>"} must not repeat edge ${edgeIdentity}`);
                }
                seenEdges.add(edgeIdentity);
            }
        }
        if (block.init !== undefined) {
            errors.push(...validateOpenClawInitBlockMetadata(block.init, `graph block ${block.id || "<unknown>"} init`));
            if (value.schema === undefined) {
                errors.push(`graph block ${block.id || "<unknown>"} init metadata requires graph schema`);
            }
            if (block.init.sourceKind === "markdown" && block.init.fileRole === undefined) {
                errors.push(`graph block ${block.id || "<unknown>"} markdown init metadata requires fileRole`);
            }
            if (block.init.fileRole !== undefined) {
                if (value.ontology === undefined) {
                    errors.push(`graph block ${block.id || "<unknown>"} fileRole requires graph ontology`);
                }
                if (block.source !== block.init.fileRole.path) {
                    errors.push(`graph block ${block.id || "<unknown>"} source ${block.source} must match init fileRole path ${block.init.fileRole.path}`);
                }
                const ontologyBinding = ontologyBindings.get(block.init.fileRole.path);
                if (ontologyBinding === undefined) {
                    errors.push(`graph block ${block.id || "<unknown>"} fileRole path ${block.init.fileRole.path} is not declared in graph ontology`);
                }
                else if (ontologyBinding.role !== block.init.fileRole.role ||
                    ontologyBinding.audience !== block.init.fileRole.audience ||
                    ontologyBinding.tier !== block.init.fileRole.tier) {
                    errors.push(`graph block ${block.id || "<unknown>"} fileRole ${block.init.fileRole.path} must match graph ontology classification`);
                }
            }
        }
        if (seen.has(block.id)) {
            errors.push(`graph block ids must be unique: ${block.id}`);
        }
        seen.add(block.id);
    }
    if (value.evolution !== undefined) {
        errors.push(...validatePackGraphEvolution(value.evolution));
    }
    return errors;
}
export function validatePackVectorsPayload(value, graph) {
    const errors = [];
    pushWhenMissing(errors, value.packId.length > 0, "vector packId is required");
    const seen = new Set();
    const knownBlockIds = graph === undefined ? null : new Set(graph.blocks.map((block) => block.id));
    if (graph !== undefined && value.packId !== graph.packId) {
        errors.push(`vector packId ${value.packId} does not match graph packId ${graph.packId}`);
    }
    for (const entry of value.entries) {
        pushWhenMissing(errors, entry.blockId.length > 0, "vector entries require blockId");
        pushWhenMissing(errors, entry.keywords.length > 0, `vector entry ${entry.blockId || "<unknown>"} requires keywords`);
        pushWhenMissing(errors, entry.boost >= 0, `vector entry ${entry.blockId || "<unknown>"} boost must be non-negative`);
        if (entry.embedding !== undefined) {
            pushWhenMissing(errors, entry.embedding.model.length > 0, `vector entry ${entry.blockId || "<unknown>"} embedding model is required`);
            pushWhenMissing(errors, entry.embedding.values.length > 0, `vector entry ${entry.blockId || "<unknown>"} embedding values must be non-empty`);
            if (!entry.embedding.values.every((candidate) => Number.isFinite(candidate))) {
                errors.push(`vector entry ${entry.blockId || "<unknown>"} embedding values must be finite numbers`);
            }
        }
        if (seen.has(entry.blockId)) {
            errors.push(`vector entries must be unique per blockId: ${entry.blockId}`);
        }
        seen.add(entry.blockId);
        if (knownBlockIds !== null && !knownBlockIds.has(entry.blockId)) {
            errors.push(`vector entry references unknown blockId ${entry.blockId}`);
        }
    }
    return errors;
}
export function validateRouterArtifact(value, manifest) {
    const errors = [];
    const queryChecksum = computeRouterQueryChecksum(value.traces);
    const traceCollectedLabels = computeRouterCollectedLabelCounts(value.traces);
    const isV2 = value.training.method === "policy_gradient_v2" ||
        value.training.objective.updateVersion === "route_pg_update_v2" ||
        value.training.objective.objective === "supervised_route_pg_v2";
    const collectedLabels = isV2 ? value.training.collectedLabels : traceCollectedLabels;
    pushWhenMissing(errors, value.routerIdentity.length > 0, "routerIdentity is required");
    pushWhenMissing(errors, value.strategy === "learned_route_fn_v1", "router strategy must be learned_route_fn_v1");
    pushWhenMissing(errors, isIsoDate(value.trainedAt), "router trainedAt must be an ISO timestamp");
    pushWhenMissing(errors, value.training.status === "updated" || value.training.status === "no_supervision", "router training status must be updated or no_supervision");
    pushWhenMissing(errors, value.training.method === "policy_gradient_v1" || value.training.method === "policy_gradient_v2", "router training method must be policy_gradient_v1 or policy_gradient_v2");
    if (!isV2) {
        pushWhenMissing(errors, value.training.routeTraceCount === value.traces.length, "router routeTraceCount must match trace count");
    }
    pushWhenMissing(errors, value.training.updateCount === value.policyUpdates.length, "router updateCount must match policyUpdates length");
    const supervisionCount = isV2
        ? value.training.supervisionCount
        : value.traces.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length;
    if (!isV2) {
        pushWhenMissing(errors, value.training.supervisionCount === supervisionCount, "router supervisionCount must match supervised traces");
        pushWhenMissing(errors, value.training.collectedLabels.total === collectedLabels.total, "router collectedLabels.total must match traces");
        pushWhenMissing(errors, value.training.collectedLabels.humanFeedback === collectedLabels.humanFeedback, "router collectedLabels.humanFeedback must match traces");
        pushWhenMissing(errors, value.training.collectedLabels.operatorOverride === collectedLabels.operatorOverride, "router collectedLabels.operatorOverride must match traces");
        pushWhenMissing(errors, value.training.collectedLabels.selfMemory === collectedLabels.selfMemory, "router collectedLabels.selfMemory must match traces");
    }
    pushWhenMissing(errors, value.training.queryChecksum === queryChecksum, "router queryChecksum does not match traces");
    pushWhenMissing(errors, value.training.objective.updateMechanism === "policy_gradient", "router objective updateMechanism must be policy_gradient");
    pushWhenMissing(errors, value.training.objective.updateVersion === "route_pg_update_v1" || value.training.objective.updateVersion === "route_pg_update_v2", "router objective updateVersion must be route_pg_update_v1 or route_pg_update_v2");
    pushWhenMissing(errors, value.training.objective.objective === "supervised_route_pg_v1" || value.training.objective.objective === "supervised_route_pg_v2", "router objective must be supervised_route_pg_v1 or supervised_route_pg_v2");
    pushWhenMissing(errors, value.training.objective.profile.traceSource === "event_reconstruction" ||
        value.training.objective.profile.traceSource === "serve_time_decision_log", "router objective profile traceSource must be event_reconstruction or serve_time_decision_log");
    pushWhenMissing(errors, value.training.objective.profile.actionSpace === "pack_block_softmax" ||
        value.training.objective.profile.actionSpace === "graph_local_neighbor_softmax", "router objective profile actionSpace must be pack_block_softmax or graph_local_neighbor_softmax");
    pushWhenMissing(errors, value.training.objective.profile.targetConstruction === "event_block_plus_related_interaction" ||
        value.training.objective.profile.targetConstruction === "trajectory_reconstruction", "router objective profile targetConstruction must be event_block_plus_related_interaction or trajectory_reconstruction");
    pushWhenMissing(errors, value.training.objective.profile.rewardSignal === "explicit_label_reward_table_v1", "router objective profile rewardSignal must be explicit_label_reward_table_v1");
    pushWhenMissing(errors, value.training.objective.profile.baseline === "none" ||
        value.training.objective.profile.baseline === "exponential_moving_average", "router objective profile baseline must be none or exponential_moving_average");
    pushWhenMissing(errors, value.training.objective.profile.offPolicyCorrection === "none", "router objective profile offPolicyCorrection must be none");
    pushWhenMissing(errors, value.training.objective.profile.updateCadence === "candidate_pack_refresh", "router objective profile updateCadence must be candidate_pack_refresh");
    pushWhenMissing(errors, value.training.objective.objectiveChecksum ===
        computeRouterObjectiveChecksum({
            updateMechanism: value.training.objective.updateMechanism,
            updateVersion: value.training.objective.updateVersion,
            objective: value.training.objective.objective,
            profile: value.training.objective.profile,
            eventExportDigest: value.training.eventExportDigest,
            routeTraceCount: value.training.routeTraceCount,
            supervisionCount: value.training.supervisionCount,
            collectedLabels,
            queryChecksum
        }), "router objectiveChecksum does not match router objective metadata");
    pushWhenMissing(errors, value.training.weightsChecksum === computeRouterWeightsChecksum(value.policyUpdates), "router weightsChecksum does not match policyUpdates");
    pushWhenMissing(errors, value.training.freshnessChecksum ===
        computeRouterFreshnessChecksum({
            method: value.training.method,
            trainedAt: value.trainedAt,
            status: value.training.status,
            eventExportDigest: value.training.eventExportDigest,
            routeTraceCount: value.training.routeTraceCount,
            supervisionCount: value.training.supervisionCount,
            updateCount: value.training.updateCount
        }), "router freshnessChecksum does not match router freshness metadata");
    if (value.training.status === "updated") {
        pushWhenMissing(errors, value.training.updateCount > 0, "updated routers must record at least one policy update");
        pushWhenMissing(errors, value.training.noOpReason === null, "updated routers must not set noOpReason");
    }
    if (value.training.status === "no_supervision") {
        pushWhenMissing(errors, value.training.updateCount === 0, "no-supervision routers must not record policy updates");
        pushWhenMissing(errors, typeof value.training.noOpReason === "string" && value.training.noOpReason.length > 0, "no-supervision routers must expose a non-empty noOpReason");
    }
    const seenTraceIds = new Set();
    for (const trace of value.traces) {
        pushWhenMissing(errors, trace.traceId.length > 0, "router traces require traceId");
        pushWhenMissing(errors, trace.sourceEventId.length > 0, "router traces require sourceEventId");
        pushWhenMissing(errors, trace.sourceKind.length > 0, "router traces require sourceKind");
        pushWhenMissing(errors, trace.supervisionKind === "route_trace" ||
            trace.supervisionKind === "human_feedback" ||
            trace.supervisionKind === "operator_override" ||
            trace.supervisionKind === "self_memory" ||
            trace.supervisionKind === "teacher_supervision", "router traces require a supported supervisionKind");
        pushWhenMissing(errors, Number.isFinite(trace.reward), "router traces require finite reward values");
        if (seenTraceIds.has(trace.traceId)) {
            errors.push(`duplicate router traceId ${trace.traceId}`);
        }
        seenTraceIds.add(trace.traceId);
        for (const weight of Object.values(trace.queryVector)) {
            pushWhenMissing(errors, Number.isFinite(weight), "router trace queryVector values must be finite");
        }
    }
    const seenBlockIds = new Set();
    for (const update of value.policyUpdates) {
        pushWhenMissing(errors, update.blockId.length > 0, "router policyUpdates require blockId");
        pushWhenMissing(errors, Number.isFinite(update.delta), "router policyUpdates require finite delta values");
        pushWhenMissing(errors, Number.isFinite(update.rewardSum), "router policyUpdates require finite rewardSum values");
        pushWhenMissing(errors, update.evidenceCount > 0, "router policyUpdates require evidenceCount > 0");
        if (seenBlockIds.has(update.blockId)) {
            errors.push(`duplicate router policy update blockId ${update.blockId}`);
        }
        seenBlockIds.add(update.blockId);
        for (const weight of Object.values(update.tokenWeights)) {
            pushWhenMissing(errors, Number.isFinite(weight), "router policy update tokenWeights must be finite");
        }
    }
    if (manifest !== undefined) {
        if (manifest.routePolicy === "requires_learned_routing") {
            pushWhenMissing(errors, value.requiresLearnedRouting === true, "learned-routing manifests require router requiresLearnedRouting=true");
        }
        if (manifest.runtimeAssets.router.identity !== null && value.routerIdentity !== manifest.runtimeAssets.router.identity) {
            errors.push(`router identity ${value.routerIdentity} does not match manifest router identity ${manifest.runtimeAssets.router.identity}`);
        }
        if (manifest.provenance.eventExports?.exportDigest !== undefined &&
            manifest.provenance.eventExports !== null &&
            value.training.eventExportDigest !== manifest.provenance.eventExports.exportDigest) {
            errors.push(`router eventExportDigest ${value.training.eventExportDigest ?? "null"} does not match manifest event export digest ${manifest.provenance.eventExports.exportDigest}`);
        }
        if (manifest.routeArtifact.routeFnVersion !== value.strategy) {
            errors.push(`router strategy ${value.strategy} does not match manifest routeArtifact routeFnVersion ${manifest.routeArtifact.routeFnVersion ?? "null"}`);
        }
        if (manifest.routeArtifact.trainingMethod !== value.training.method) {
            errors.push(`router training method ${value.training.method} does not match manifest routeArtifact trainingMethod ${manifest.routeArtifact.trainingMethod ?? "null"}`);
        }
        if (manifest.routeArtifact.trainedAt !== value.trainedAt) {
            errors.push(`router trainedAt ${value.trainedAt} does not match manifest routeArtifact trainedAt ${manifest.routeArtifact.trainedAt ?? "null"}`);
        }
        if ((manifest.routeArtifact.eventExportDigest ?? null) !== (value.training.eventExportDigest ?? null)) {
            errors.push(`router eventExportDigest ${value.training.eventExportDigest ?? "null"} does not match manifest routeArtifact eventExportDigest ${manifest.routeArtifact.eventExportDigest ?? "null"}`);
        }
        if (manifest.routeArtifact.updateCount !== value.training.updateCount) {
            errors.push(`router updateCount ${value.training.updateCount} does not match manifest routeArtifact updateCount ${manifest.routeArtifact.updateCount ?? null}`);
        }
        if (manifest.routeArtifact.objective !== value.training.objective.objective) {
            errors.push(`router objective ${value.training.objective.objective} does not match manifest routeArtifact objective ${manifest.routeArtifact.objective ?? "null"}`);
        }
        if (manifest.routeArtifact.objectiveChecksum !== value.training.objective.objectiveChecksum) {
            errors.push(`router objectiveChecksum ${value.training.objective.objectiveChecksum} does not match manifest routeArtifact objectiveChecksum ${manifest.routeArtifact.objectiveChecksum ?? "null"}`);
        }
        if (manifest.routeArtifact.freshnessChecksum !== value.training.freshnessChecksum) {
            errors.push(`router freshnessChecksum ${value.training.freshnessChecksum} does not match manifest routeArtifact freshnessChecksum ${manifest.routeArtifact.freshnessChecksum ?? "null"}`);
        }
    }
    return errors;
}
export const FIXTURE_PACK_GRAPH = {
    packId: "pack-fixture",
    schema: PACK_GRAPH_SCHEMAS.openclawInit,
    ontology: {
        schema: PACK_GRAPH_SCHEMAS.openclawInit,
        typedMarkdownSurface: true,
        fileRoles: [
            { path: "README.md", role: "repo_boundary", audience: "runtime", tier: "core" },
            { path: "CLAIMS.md", role: "claims_boundary", audience: "proof", tier: "core" },
            { path: "docs/internal/contracts-v1.md", role: "contracts_reference", audience: "integrator", tier: "core" },
            { path: "docs/glossary.md", role: "glossary", audience: "integrator", tier: "supporting" },
            { path: "docs/internal/learning-first-convergence.md", role: "learning_policy", audience: "runtime", tier: "core" },
            { path: "docs/new-agent-sop.md", role: "agent_sop", audience: "operator", tier: "supporting" },
            { path: "docs/openclaw-attach-quickstart.md", role: "attach_quickstart", audience: "integrator", tier: "core" },
            { path: "docs/openclaw-integration.md", role: "integration_guide", audience: "integrator", tier: "core" },
            { path: "docs/operator-guide.md", role: "operator_guide", audience: "operator", tier: "core" },
            { path: "docs/operator-observability.md", role: "operator_observability", audience: "operator", tier: "core" },
            { path: "docs/ops-recipes.md", role: "ops_recipe", audience: "operator", tier: "core" },
            { path: "docs/internal/recorded-session-replay.md", role: "session_replay_proof", audience: "proof", tier: "supporting" },
            { path: "docs/internal/release.md", role: "release_guide", audience: "operator", tier: "supporting" },
            { path: "docs/reproduce-eval.md", role: "evaluation_reproduction", audience: "proof", tier: "supporting" },
            { path: "docs/setup-guide.md", role: "setup_guide", audience: "integrator", tier: "core" },
            { path: "docs/worked-example.md", role: "worked_example", audience: "integrator", tier: "supporting" }
        ],
        fastBootRequired: true,
        passiveBackgroundLearningRequired: true,
        heuristicScope: "init_priors_and_topology_only",
        learnedLabelPolicy: "explicit_collected_labels_only"
    },
    blocks: [
        {
            id: "ctx-feedback-scanner",
            source: "docs/openclaw-attach-quickstart.md",
            text: "Always-on feedback scanner harvests human labels from local session logs with Ollama qwen3.5:9b-q4_K_M and checkpointed replay.",
            keywords: ["feedback", "scanner", "always-on", "session", "logs", "ollama", "qwen", "checkpoint"],
            priority: 5,
            tokenCount: 16,
            routing: {
                channels: ["short_term", "vector"],
                shortTermBias: 2,
                vectorBias: 1,
                backgroundLabelAmplification: 2
            },
            learning: {
                role: "label_surface",
                humanLabels: 2,
                selfLabels: 0,
                decayHalfLifeDays: 30,
                hebbianPulse: 5
            },
            init: {
                nodeKind: "markdown_ontology",
                sourceKind: "markdown",
                fastBootRequired: true,
                passiveBackgroundLearningRequired: true,
                heuristicScope: "init_priors_and_topology_only",
                learnedLabelPolicy: "explicit_collected_labels_only",
                fileRole: {
                    path: "docs/openclaw-attach-quickstart.md",
                    role: "attach_quickstart",
                    audience: "integrator",
                    tier: "core"
                }
            }
        },
        {
            id: "ctx-runtime-compile",
            source: "docs/internal/contracts-v1.md",
            text: "runtime_compile.v1 keeps fast boot defaults available while passive background learning hydrates promoted packs, explicit budgets, and manifest-gated routing.",
            keywords: ["runtime", "compile", "fast", "boot", "passive", "background", "pack", "manifest", "routing", "openclaw", "budget"],
            priority: 4,
            tokenCount: 19,
            routing: {
                channels: ["vector"],
                vectorBias: 2,
                backgroundLabelAmplification: 2
            },
            learning: {
                role: "boot_default",
                humanLabels: 0,
                selfLabels: 0,
                decayHalfLifeDays: null,
                hebbianPulse: 2
            },
            init: {
                nodeKind: "markdown_ontology",
                sourceKind: "markdown",
                fastBootRequired: true,
                passiveBackgroundLearningRequired: true,
                heuristicScope: "init_priors_and_topology_only",
                learnedLabelPolicy: "explicit_collected_labels_only",
                fileRole: {
                    path: "docs/internal/contracts-v1.md",
                    role: "contracts_reference",
                    audience: "integrator",
                    tier: "core"
                }
            }
        },
        {
            id: "ctx-structural-ops",
            source: "docs/internal/learning-first-convergence.md",
            text: "Structural graph operations like split, merge, prune, and connect stay first-class beside Hebbian reinforcement and decay.",
            keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory", "hebbian", "decay"],
            priority: 3,
            tokenCount: 15,
            routing: {
                channels: ["graph", "vector"],
                graphBias: 2,
                vectorBias: 1
            },
            learning: {
                role: "structural",
                humanLabels: 0,
                selfLabels: 0,
                decayHalfLifeDays: 30,
                hebbianPulse: 4
            },
            init: {
                nodeKind: "markdown_ontology",
                sourceKind: "markdown",
                fastBootRequired: true,
                passiveBackgroundLearningRequired: true,
                heuristicScope: "init_priors_and_topology_only",
                learnedLabelPolicy: "explicit_collected_labels_only",
                fileRole: {
                    path: "docs/internal/learning-first-convergence.md",
                    role: "learning_policy",
                    audience: "runtime",
                    tier: "core"
                }
            }
        },
        {
            id: "ctx-context-compact",
            source: "pack/pack-fixture:structural-compaction",
            text: "Compacted pack context keeps fast boot defaults and passive background learning deterministic across human label, self label, and structural graph sources.",
            keywords: ["pack", "structural", "compaction", "context", "deterministic", "fast", "boot", "background", "labels"],
            priority: 4,
            tokenCount: 18,
            compactedFrom: ["ctx-feedback-scanner", "ctx-runtime-compile", "ctx-structural-ops"],
            routing: {
                channels: ["graph", "vector"],
                graphBias: 1,
                vectorBias: 2,
                backgroundLabelAmplification: 2
            },
            learning: {
                role: "background_expectation",
                humanLabels: 2,
                selfLabels: 1,
                decayHalfLifeDays: 30,
                hebbianPulse: 4
            },
            init: {
                nodeKind: "synthetic_topology",
                sourceKind: "synthetic",
                fastBootRequired: true,
                passiveBackgroundLearningRequired: true,
                heuristicScope: "init_priors_and_topology_only",
                learnedLabelPolicy: "explicit_collected_labels_only"
            }
        }
    ]
};
export const FIXTURE_PACK_VECTORS = {
    packId: FIXTURE_PACK_GRAPH.packId,
    entries: [
        {
            blockId: "ctx-feedback-scanner",
            keywords: ["feedback", "scanner", "human_label", "always_on", "ollama", "qwen", "checkpoint", "sessions"],
            boost: 4,
            weights: {
                feedback: 5,
                scanner: 5,
                human_label: 6,
                always_on: 4,
                ollama: 4,
                qwen: 6,
                checkpoint: 3
            }
        },
        {
            blockId: "ctx-runtime-compile",
            keywords: ["runtime", "compile", "fast_boot", "passive_background", "pack", "manifest", "routing", "openclaw", "budget"],
            boost: 3,
            weights: {
                runtime: 5,
                compile: 5,
                fast_boot: 5,
                passive_background: 4,
                manifest: 4,
                pack: 3,
                routing: 3,
                budget: 3
            }
        },
        {
            blockId: "ctx-structural-ops",
            keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory", "hebbian", "decay"],
            boost: 2,
            weights: {
                structural: 5,
                split: 4,
                merge: 4,
                prune: 3,
                connect: 3,
                hebbian: 4,
                decay: 3,
                memory: 2
            }
        },
        {
            blockId: "ctx-context-compact",
            keywords: ["pack", "structural", "compaction", "context", "deterministic", "fast_boot", "passive_background", "human_label", "self_label"],
            boost: 4,
            weights: {
                structural: 5,
                compaction: 5,
                context: 4,
                deterministic: 4,
                fast_boot: 4,
                passive_background: 4,
                human_label: 3,
                self_label: 3
            }
        }
    ]
};
const FIXTURE_PROMPT_CONTEXT_FINGERPRINTS = ["sha256-system-prompt-fixture", "sha256-context-injection-fixture"];
const FIXTURE_WORKSPACE_INJECTION_SURFACE_DIGEST = "sha256-workspace-injection-surface-fixture";
const FIXTURE_RUNTIME_HINTS = ["feedback scanner", "scanner attribution"];
function buildFixtureRuntimeContextFingerprint(input) {
    const promptContextDigest = checksumJsonPayload({
        promptContextFingerprints: [...FIXTURE_PROMPT_CONTEXT_FINGERPRINTS],
        workspaceInjectionSurfaceDigest: FIXTURE_WORKSPACE_INJECTION_SURFACE_DIGEST
    });
    const runtimeHintsDigest = checksumJsonPayload([...FIXTURE_RUNTIME_HINTS]);
    const profileLineage = [...input.profileLineage];
    const sessionLineage = [`session:${input.sessionId}`, `channel:${input.channel}`, `source_stream:${input.sourceStream}`];
    const brainLineage = [
        `brain_status:${input.brainStatus}`,
        `active_pack:${input.activePackId ?? "none"}`,
        `router:${input.routerIdentity ?? "none"}`,
        `used_learned_route_fn:${input.usedLearnedRouteFn === null ? "unknown" : String(input.usedLearnedRouteFn)}`
    ];
    const profileLineageDigest = checksumJsonPayload(profileLineage);
    const sessionLineageDigest = checksumJsonPayload(sessionLineage);
    const brainLineageDigest = checksumJsonPayload(brainLineage);
    return {
        selectionDigest: input.selectionDigest,
        promptContextDigest,
        promptContextFingerprints: [...FIXTURE_PROMPT_CONTEXT_FINGERPRINTS],
        workspaceInjectionSurfaceDigest: FIXTURE_WORKSPACE_INJECTION_SURFACE_DIGEST,
        runtimeHintsDigest,
        runtimeHints: [...FIXTURE_RUNTIME_HINTS],
        profileLineageDigest,
        profileLineage,
        sessionLineageDigest,
        sessionLineage,
        brainLineageDigest,
        brainLineage,
        digest: checksumJsonPayload({
            selectionDigest: input.selectionDigest,
            promptContextDigest,
            runtimeHintsDigest,
            profileLineageDigest,
            sessionLineageDigest,
            brainLineageDigest
        })
    };
}
export const FIXTURE_RUNTIME_TURN_ATTRIBUTION = {
    hostRuntimeOwner: "openclaw",
    profileSelector: "current_profile",
    profileId: null,
    brainAttachmentPolicy: "dedicated",
    brainStatus: "serving_active_pack",
    activePackId: FIXTURE_PACK_GRAPH.packId,
    usedLearnedRouteFn: true,
    routerIdentity: `${FIXTURE_PACK_GRAPH.packId}:route_fn`,
    selectionDigest: "sha256-selection-fixture",
    selectionTiers: "route_fn>teacher>workspace",
    contextFingerprint: buildFixtureRuntimeContextFingerprint({
        selectionDigest: "sha256-selection-fixture",
        sessionId: "session-fixture",
        channel: "whatsapp",
        sourceStream: "openclaw/runtime/whatsapp",
        brainStatus: "serving_active_pack",
        activePackId: FIXTURE_PACK_GRAPH.packId,
        routerIdentity: `${FIXTURE_PACK_GRAPH.packId}:route_fn`,
        usedLearnedRouteFn: true,
        profileLineage: ["host:openclaw", "profile:current_profile", "attachment_policy:dedicated"]
    }),
    contextEvidence: "route_fn_and_brain_context"
};
export const FIXTURE_INTERACTION_EVENTS = [
    createInteractionEvent({
        eventId: "evt-interaction-fixture-1",
        agentId: "agent-fixture",
        sessionId: "session-fixture",
        channel: "whatsapp",
        sequence: 101,
        kind: "memory_compiled",
        createdAt: "2026-03-06T00:00:00.000Z",
        source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/runtime/whatsapp"
        },
        semantic: {
            semanticType: "observability_residue",
            sourceKind: "runtime_turn",
            diagnosticIntent: "compile_observability"
        },
        packId: FIXTURE_PACK_GRAPH.packId,
        principal: {
            teacherIdentity: "openclaw/self",
            teacherRole: "assistant",
            teacherAuthority: "background",
            principalScope: {
                kind: "session",
                sessionId: "session-fixture",
                scopeKey: "session-fixture"
            },
            priorityClass: "low"
        },
        attribution: FIXTURE_RUNTIME_TURN_ATTRIBUTION
    }),
    createInteractionEvent({
        eventId: "evt-interaction-fixture-2",
        agentId: "agent-fixture",
        sessionId: "session-fixture",
        channel: "whatsapp",
        sequence: 103,
        kind: "message_delivered",
        createdAt: "2026-03-06T00:02:00.000Z",
        source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/runtime/whatsapp"
        },
        semantic: {
            semanticType: "delivery_residue",
            sourceKind: "runtime_turn",
            diagnosticIntent: "delivery_observability"
        },
        packId: FIXTURE_PACK_GRAPH.packId,
        messageId: "msg-fixture-1",
        attribution: FIXTURE_RUNTIME_TURN_ATTRIBUTION
    })
];
export const FIXTURE_FEEDBACK_EVENTS = [
    createFeedbackEvent({
        eventId: "evt-feedback-fixture-1",
        agentId: "agent-fixture",
        sessionId: "session-fixture",
        channel: "whatsapp",
        sequence: 102,
        kind: "teaching",
        createdAt: "2026-03-06T00:01:00.000Z",
        source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/runtime/whatsapp"
        },
        semantic: {
            semanticType: "teacher_signal",
            sourceKind: "runtime_turn"
        },
        content: "Use the unified feedback scanner before enabling default loop scans.",
        messageId: "msg-fixture-1",
        principal: {
            teacherIdentity: "bihua",
            teacherRole: "principal",
            teacherAuthority: "binding",
            principalScope: {
                kind: "interaction",
                profileSelector: "current_profile",
                sessionId: "session-fixture",
                interactionId: FIXTURE_INTERACTION_EVENTS[0].eventId,
                scopeKey: "session-fixture:evt-interaction-fixture-1"
            },
            priorityClass: "critical",
            supersedes: ["evt-feedback-legacy-0"]
        },
        attribution: FIXTURE_RUNTIME_TURN_ATTRIBUTION,
        relatedInteractionId: FIXTURE_INTERACTION_EVENTS[0].eventId
    }),
    createFeedbackEvent({
        eventId: "evt-feedback-fixture-2",
        agentId: "agent-fixture",
        sessionId: "session-fixture",
        channel: "whatsapp",
        sequence: 104,
        kind: "approval",
        createdAt: "2026-03-06T00:03:00.000Z",
        source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/runtime/whatsapp"
        },
        semantic: {
            semanticType: "teacher_signal",
            sourceKind: "runtime_turn"
        },
        content: "Learned routing promotion is approved after compile diagnostics stay stable.",
        principal: {
            teacherIdentity: "jonathan",
            teacherRole: "admin",
            teacherAuthority: "high",
            principalScope: {
                kind: "profile",
                profileSelector: "current_profile",
                scopeKey: "current_profile"
            },
            priorityClass: "high"
        },
        attribution: FIXTURE_RUNTIME_TURN_ATTRIBUTION,
        relatedInteractionId: FIXTURE_INTERACTION_EVENTS[1].eventId
    })
];
export const FIXTURE_NORMALIZED_EVENT_EXPORT = buildNormalizedEventExport({
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS
});
const FIXTURE_ROUTER_TRACES = [
    {
        traceId: "trace-fixture-memory-compiled",
        sourceEventId: "evt-interaction-fixture-1",
        sourceContract: CONTRACT_IDS.interactionEvents,
        sourceKind: "memory_compiled",
        supervisionKind: "self_memory",
        targetBlockIds: ["ctx-runtime-compile", "ctx-context-compact"],
        reward: 2,
        queryTokens: ["memory", "compiled", "runtime", "pack"],
        queryVector: {
            memory: 2,
            compiled: 1,
            runtime: 1,
            pack: 1
        }
    },
    {
        traceId: "trace-fixture-feedback-scanner",
        sourceEventId: "evt-feedback-fixture-1",
        sourceContract: CONTRACT_IDS.feedbackEvents,
        sourceKind: "teaching",
        supervisionKind: "human_feedback",
        targetBlockIds: ["ctx-feedback-scanner", "ctx-runtime-compile"],
        reward: 4,
        queryTokens: ["feedback", "scanner", "default", "loop", "scans"],
        queryVector: {
            feedback: 3,
            scanner: 3,
            default: 1,
            loop: 1,
            scans: 1
        }
    },
    {
        traceId: "trace-fixture-approval-route",
        sourceEventId: "evt-feedback-fixture-2",
        sourceContract: CONTRACT_IDS.feedbackEvents,
        sourceKind: "approval",
        supervisionKind: "human_feedback",
        targetBlockIds: ["ctx-runtime-compile", "ctx-context-compact"],
        reward: 2,
        queryTokens: ["learned", "routing", "promotion", "compile", "diagnostics"],
        queryVector: {
            learned: 1,
            routing: 2,
            promotion: 1,
            compile: 2,
            diagnostics: 1
        }
    },
    {
        traceId: "trace-fixture-message-route",
        sourceEventId: "evt-interaction-fixture-2",
        sourceContract: CONTRACT_IDS.interactionEvents,
        sourceKind: "message_delivered",
        supervisionKind: "route_trace",
        targetBlockIds: ["ctx-context-compact"],
        reward: 0,
        queryTokens: ["message", "delivered", "route"],
        queryVector: {
            message: 1,
            delivered: 1,
            route: 1
        }
    }
];
const FIXTURE_ROUTER_POLICY_UPDATES = [
    {
        blockId: "ctx-feedback-scanner",
        delta: 11,
        evidenceCount: 2,
        rewardSum: 6,
        tokenWeights: {
            feedback: 5,
            scanner: 5,
            default: 1
        },
        traceIds: ["trace-fixture-feedback-scanner", "trace-fixture-approval-route"]
    },
    {
        blockId: "ctx-runtime-compile",
        delta: 8,
        evidenceCount: 3,
        rewardSum: 8,
        tokenWeights: {
            runtime: 2,
            compile: 3,
            routing: 2,
            promotion: 1
        },
        traceIds: ["trace-fixture-memory-compiled", "trace-fixture-feedback-scanner", "trace-fixture-approval-route"]
    },
    {
        blockId: "ctx-context-compact",
        delta: 5,
        evidenceCount: 2,
        rewardSum: 4,
        tokenWeights: {
            pack: 1,
            compile: 1,
            diagnostics: 1,
            route: 2
        },
        traceIds: ["trace-fixture-memory-compiled", "trace-fixture-approval-route"]
    }
];
export const FIXTURE_ROUTER_ARTIFACT = {
    routerIdentity: "pack-fixture:route_fn",
    strategy: "learned_route_fn_v1",
    trainedAt: "2026-03-06T00:00:00.000Z",
    requiresLearnedRouting: true,
    training: {
        method: "policy_gradient_v1",
        status: "updated",
        eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest,
        routeTraceCount: FIXTURE_ROUTER_TRACES.length,
        supervisionCount: FIXTURE_ROUTER_TRACES.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length,
        updateCount: FIXTURE_ROUTER_POLICY_UPDATES.length,
        collectedLabels: computeRouterCollectedLabelCounts(FIXTURE_ROUTER_TRACES),
        objective: {
            updateMechanism: "policy_gradient",
            updateVersion: "route_pg_update_v1",
            objective: "supervised_route_pg_v1",
            profile: ROUTER_PG_PROFILE_V1,
            objectiveChecksum: computeRouterObjectiveChecksum({
                updateMechanism: "policy_gradient",
                updateVersion: "route_pg_update_v1",
                objective: "supervised_route_pg_v1",
                profile: ROUTER_PG_PROFILE_V1,
                eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest,
                routeTraceCount: FIXTURE_ROUTER_TRACES.length,
                supervisionCount: FIXTURE_ROUTER_TRACES.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length,
                collectedLabels: computeRouterCollectedLabelCounts(FIXTURE_ROUTER_TRACES),
                queryChecksum: computeRouterQueryChecksum(FIXTURE_ROUTER_TRACES)
            })
        },
        queryChecksum: computeRouterQueryChecksum(FIXTURE_ROUTER_TRACES),
        weightsChecksum: computeRouterWeightsChecksum(FIXTURE_ROUTER_POLICY_UPDATES),
        freshnessChecksum: computeRouterFreshnessChecksum({
            method: "policy_gradient_v1",
            trainedAt: "2026-03-06T00:00:00.000Z",
            status: "updated",
            eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest,
            routeTraceCount: FIXTURE_ROUTER_TRACES.length,
            supervisionCount: FIXTURE_ROUTER_TRACES.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length,
            updateCount: FIXTURE_ROUTER_POLICY_UPDATES.length
        }),
        noOpReason: null
    },
    traces: FIXTURE_ROUTER_TRACES,
    policyUpdates: FIXTURE_ROUTER_POLICY_UPDATES
};
export const FIXTURE_TEACHER_SUPERVISION_ARTIFACT = {
    contract: CONTRACT_IDS.teacherSupervisionArtifact,
    artifactId: "teacher-sha256-6f0f75f1f8a145dbe4a7fa8d2b8ad9a70c1eb9c5f2451d3dfd1ef540bd692a86",
    dedupId: "sha256-6f0f75f1f8a145dbe4a7fa8d2b8ad9a70c1eb9c5f2451d3dfd1ef540bd692a86",
    kind: "teaching",
    createdAt: FIXTURE_FEEDBACK_EVENTS[0].createdAt,
    source: {
        runtimeOwner: "openclaw",
        sessionId: FIXTURE_FEEDBACK_EVENTS[0].sessionId,
        channel: FIXTURE_FEEDBACK_EVENTS[0].channel,
        sourceStreams: [FIXTURE_FEEDBACK_EVENTS[0].source.stream],
        eventRange: {
            start: FIXTURE_NORMALIZED_EVENT_EXPORT.range.start,
            end: FIXTURE_NORMALIZED_EVENT_EXPORT.range.end,
            count: FIXTURE_NORMALIZED_EVENT_EXPORT.range.count
        },
        eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest
    },
    sourceEventIds: [FIXTURE_FEEDBACK_EVENTS[0].eventId, FIXTURE_INTERACTION_EVENTS[0].eventId],
    relatedInteractionId: FIXTURE_FEEDBACK_EVENTS[0].relatedInteractionId ?? null,
    ...(FIXTURE_FEEDBACK_EVENTS[0].principal === undefined
        ? {}
        : { principal: FIXTURE_FEEDBACK_EVENTS[0].principal }),
    content: FIXTURE_FEEDBACK_EVENTS[0].content,
    freshness: {
        status: "fresh",
        observedAt: "2026-03-06T00:00:15.000Z",
        newestSourceCreatedAt: FIXTURE_FEEDBACK_EVENTS[0].createdAt,
        ageMs: 15_000,
        staleAfterMs: 300_000
    }
};
export const FIXTURE_WORKSPACE_METADATA = {
    workspaceId: "workspace-fixture",
    snapshotId: "workspace-fixture@snapshot-2026-03-06",
    capturedAt: "2026-03-06T00:00:00.000Z",
    rootDir: "/workspace/openclawbrain",
    branch: "main",
    revision: "fixture-rev-20260306",
    dirty: false,
    manifestDigest: "sha256-workspace-fixture",
    labels: ["learning-first", "public-surface"],
    files: ["README.md", "packages/contracts/src/index.ts", "packages/learner/src/index.ts"]
};
export const FIXTURE_ARTIFACT_MANIFEST = {
    contract: CONTRACT_IDS.artifactManifest,
    packId: FIXTURE_PACK_GRAPH.packId,
    immutable: true,
    routePolicy: "requires_learned_routing",
    runtimeAssets: {
        graphPath: "graph.json",
        vectorPath: "vectors.json",
        router: {
            kind: "artifact",
            identity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
            artifactPath: "router/model.json"
        }
    },
    payloadChecksums: {
        graph: checksumJsonPayload(FIXTURE_PACK_GRAPH),
        vector: checksumJsonPayload(FIXTURE_PACK_VECTORS),
        router: checksumJsonPayload(FIXTURE_ROUTER_ARTIFACT)
    },
    routeArtifact: buildRouteArtifactReference({
        routerAssetKind: "artifact",
        routerIdentity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
        routerChecksum: checksumJsonPayload(FIXTURE_ROUTER_ARTIFACT),
        router: FIXTURE_ROUTER_ARTIFACT,
        eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest
    }),
    modelFingerprints: ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", FIXTURE_ROUTER_ARTIFACT.routerIdentity],
    provenance: {
        workspace: FIXTURE_WORKSPACE_METADATA,
        workspaceSnapshot: FIXTURE_WORKSPACE_METADATA.snapshotId,
        eventRange: FIXTURE_NORMALIZED_EVENT_EXPORT.range,
        eventExports: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance,
        learningSurface: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.learningSurface,
        builtAt: "2026-03-06T00:00:00.000Z",
        offlineArtifacts: ["feedback_events.v1", "runtime_compile.v1"]
    },
    graphDynamics: {
        bootstrapping: {
            fastBootDefaults: true,
            passiveBackgroundLearning: true
        },
        runtimePlasticitySource: "candidate_build",
        hebbian: {
            enabled: true,
            learningRate: 0.2
        },
        decay: {
            enabled: true,
            halfLifeDays: 30
        },
        structuralOps: {
            split: 1,
            merge: 0,
            prune: 2,
            connect: 3
        }
    }
};
export const FIXTURE_ACTIVATION_POINTERS = {
    contract: CONTRACT_IDS.activationPointers,
    active: {
        slot: "active",
        packId: "pack-active",
        packRootDir: "/packs/pack-active",
        manifestPath: "/packs/pack-active/manifest.json",
        manifestDigest: "sha256-pack-active-manifest",
        routePolicy: "heuristic_allowed",
        routerIdentity: null,
        workspaceSnapshot: "workspace-active@snapshot-2026-03-06",
        workspaceRevision: "workspace-active-rev",
        eventRange: {
            start: 1,
            end: 25,
            count: 25
        },
        eventExportDigest: null,
        builtAt: "2026-03-06T00:00:00.000Z",
        updatedAt: "2026-03-06T00:00:00.000Z"
    },
    candidate: {
        slot: "candidate",
        packId: "pack-candidate",
        packRootDir: "/packs/pack-candidate",
        manifestPath: "/packs/pack-candidate/manifest.json",
        manifestDigest: "sha256-pack-candidate-manifest",
        routePolicy: "requires_learned_routing",
        routerIdentity: "pack-candidate:route_fn",
        workspaceSnapshot: FIXTURE_WORKSPACE_METADATA.snapshotId,
        workspaceRevision: FIXTURE_WORKSPACE_METADATA.revision,
        eventRange: {
            start: 26,
            end: 40,
            count: 15
        },
        eventExportDigest: "sha256-candidate-events",
        builtAt: "2026-03-06T00:05:00.000Z",
        updatedAt: "2026-03-06T00:05:00.000Z"
    },
    previous: {
        slot: "previous",
        packId: "pack-previous",
        packRootDir: "/packs/pack-previous",
        manifestPath: "/packs/pack-previous/manifest.json",
        manifestDigest: "sha256-pack-previous-manifest",
        routePolicy: "heuristic_allowed",
        routerIdentity: null,
        workspaceSnapshot: "workspace-previous@snapshot-2026-03-05",
        workspaceRevision: "workspace-previous-rev",
        eventRange: {
            start: 0,
            end: 0,
            count: 1
        },
        eventExportDigest: null,
        builtAt: "2026-03-05T23:55:00.000Z",
        updatedAt: "2026-03-06T00:10:00.000Z"
    }
};
export const FIXTURE_RUNTIME_COMPILE_REQUEST = {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-fixture",
    userMessage: "Compile scanner and manifest context for this turn.",
    maxContextBlocks: 3,
    maxContextChars: 240,
    modeRequested: "heuristic",
    runtimeHints: ["feedback scanner", "manifest", "structural compaction"],
    compactionMode: "native"
};
const FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT = [
    {
        id: "ctx-context-compact",
        source: "pack/pack-fixture:structural-compaction",
        text: "Pack-backed structural compaction keeps larger context windows deterministic across feedback, runtime compile, and structural graph sources.",
        tokenCount: 16,
        compactedFrom: ["ctx-feedback-scanner", "ctx-runtime-compile", "ctx-structural-ops"]
    }
];
export const FIXTURE_RUNTIME_COMPILE_RESPONSE = {
    contract: CONTRACT_IDS.runtimeCompile,
    packId: FIXTURE_ARTIFACT_MANIFEST.packId,
    selectedContext: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT,
    diagnostics: {
        modeRequested: "heuristic",
        modeEffective: "learned",
        usedLearnedRouteFn: true,
        routerIdentity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
        servedArtifact: buildServedArtifactProof({
            packId: FIXTURE_ARTIFACT_MANIFEST.packId,
            routePolicy: FIXTURE_ARTIFACT_MANIFEST.routePolicy,
            routerIdentity: FIXTURE_ARTIFACT_MANIFEST.runtimeAssets.router.identity,
            workspaceSnapshot: FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot,
            workspaceRevision: FIXTURE_ARTIFACT_MANIFEST.provenance.workspace.revision,
            eventRange: {
                start: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.start,
                end: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.end,
                count: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.count
            },
            eventExportDigest: FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null,
            builtAt: FIXTURE_ARTIFACT_MANIFEST.provenance.builtAt
        }, FIXTURE_ARTIFACT_MANIFEST.routeArtifact),
        candidateCount: FIXTURE_PACK_GRAPH.blocks.length,
        selectedCount: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT.length,
        selectedCharCount: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT.reduce((sum, block) => sum + block.text.length, 0),
        selectedTokenCount: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT.reduce((sum, block) => sum + (block.tokenCount ?? 0), 0),
        selectionStrategy: "pack_route_fn_selection_v1",
        selectionDigest: checksumJsonPayload({
            packId: FIXTURE_ARTIFACT_MANIFEST.packId,
            selectedContext: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT
        }),
        structuralSignals: {
            matchedCandidateCount: 2,
            selectedMatchedCount: 1,
            selectedPriorityFallbackCount: 0,
            overlapPrunedCount: 1,
            traversalActivatedCount: 1,
            selectedBlockIds: ["ctx-context-compact"],
            overlapPrunedBlockIds: ["ctx-feedback-scanner"],
            traversalActivatedBlockIds: ["ctx-context-compact"],
            candidates: [
                {
                    blockId: "ctx-context-compact",
                    rank: 1,
                    score: 14,
                    selected: true,
                    selectedBy: "token_match",
                    matchedTokens: ["feedback", "scanner", "manifest", "structural", "compaction"],
                    directMatchedTokens: ["feedback", "scanner", "manifest", "structural", "compaction"],
                    traversalActivated: true,
                    traversalScore: 2,
                    overlapPruned: false,
                    compactedFrom: ["ctx-feedback-scanner", "ctx-runtime-compile", "ctx-structural-ops"]
                },
                {
                    blockId: "ctx-feedback-scanner",
                    rank: 2,
                    score: 8,
                    selected: false,
                    selectedBy: null,
                    matchedTokens: ["feedback", "scanner"],
                    directMatchedTokens: ["feedback", "scanner"],
                    traversalActivated: false,
                    traversalScore: 0,
                    overlapPruned: true,
                    compactedFrom: []
                }
            ]
        },
        compactionMode: "native",
        compactionApplied: false,
        routingChannels: {
            candidates: {
                graph: 2,
                shortTerm: 1,
                vector: 4
            },
            selected: {
                graph: 1,
                shortTerm: 0,
                vector: 1
            }
        },
        notes: [
            "selected_context_ids=ctx-context-compact",
            "selection_mode=token_match(feedback,scanner,manifest,structural,compaction)",
            "selection_tiers=token_match_only",
            "selection_strategy=pack_route_fn_selection_v1",
            "selection_compaction_deduped=3",
            "router_strategy=learned_route_fn_v1"
        ]
    }
};
const FIXTURE_DEDICATED_BRAIN_ATTACHMENT_POLICY_SEMANTICS = {
    mode: "dedicated",
    readScope: "current_profile_only",
    writeScope: "current_profile_only",
    currentProfileExclusive: true,
    requiresProfileAttribution: true,
    detail: "dedicated brains are exclusive to the current profile and must keep profile attribution explicit on every served turn"
};
const FIXTURE_SHARED_BRAIN_ATTACHMENT_POLICY_SEMANTICS = {
    mode: "shared",
    readScope: "attached_profiles",
    writeScope: "attached_profiles",
    currentProfileExclusive: false,
    requiresProfileAttribution: true,
    detail: "shared brains may serve multiple attached profiles, so status and per-turn attribution must stay profile-explicit"
};
export const FIXTURE_DEDICATED_BRAIN_ATTACHMENT_POLICY = {
    contract: CONTRACT_IDS.brainAttachmentPolicy,
    policy: FIXTURE_DEDICATED_BRAIN_ATTACHMENT_POLICY_SEMANTICS
};
export const FIXTURE_SHARED_BRAIN_ATTACHMENT_POLICY = {
    contract: CONTRACT_IDS.brainAttachmentPolicy,
    policy: FIXTURE_SHARED_BRAIN_ATTACHMENT_POLICY_SEMANTICS
};
export const FIXTURE_PROFILE_TURN_ATTRIBUTION = {
    contract: CONTRACT_IDS.profileTurnAttribution,
    hostRuntimeOwner: "openclaw",
    profileSelector: "current_profile",
    profileId: null,
    brainAttachmentPolicy: "dedicated",
    brainStatus: "serving_active_pack",
    sessionId: FIXTURE_INTERACTION_EVENTS[0].sessionId,
    channel: FIXTURE_INTERACTION_EVENTS[0].channel,
    interactionEventId: FIXTURE_INTERACTION_EVENTS[0].eventId,
    createdAt: FIXTURE_INTERACTION_EVENTS[0].createdAt,
    packId: FIXTURE_RUNTIME_COMPILE_RESPONSE.packId,
    routerIdentity: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.routerIdentity,
    usedLearnedRouteFn: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.usedLearnedRouteFn,
    selectionMode: "token_match(feedback,scanner,manifest,structural,compaction)",
    selectionTiers: "token_match_only",
    selectionDigest: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.selectionDigest,
    contextFingerprint: buildFixtureRuntimeContextFingerprint({
        selectionDigest: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.selectionDigest,
        sessionId: FIXTURE_INTERACTION_EVENTS[0].sessionId,
        channel: FIXTURE_INTERACTION_EVENTS[0].channel,
        sourceStream: FIXTURE_INTERACTION_EVENTS[0].source.stream,
        brainStatus: "serving_active_pack",
        activePackId: FIXTURE_RUNTIME_COMPILE_RESPONSE.packId,
        routerIdentity: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.routerIdentity,
        usedLearnedRouteFn: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.usedLearnedRouteFn,
        profileLineage: ["host:openclaw", "profile:current_profile", "attachment_policy:dedicated"]
    }),
    selectedContextCount: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.selectedCount,
    stableKernelBlockCount: 0,
    brainCompiledBlockCount: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.selectedCount,
    stableKernelSources: [],
    brainCompiledSources: FIXTURE_RUNTIME_COMPILE_RESPONSE.selectedContext.map((block) => block.source),
    contextEvidence: "route_fn_and_brain_context",
    detail: "current profile turn was served from the active promoted pack with explicit route-function and block-source attribution"
};
export const FIXTURE_CURRENT_PROFILE_BRAIN_STATUS = {
    contract: CONTRACT_IDS.currentProfileBrainStatus,
    generatedAt: "2026-03-07T20:00:00.000Z",
    host: {
        noun: "Host",
        runtimeOwner: "openclaw",
        activationRoot: "/activation/openclaw/current-profile"
    },
    profile: {
        noun: "Profile",
        selector: "current_profile",
        profileId: null,
        detail: "The Host resolves the current Profile through the active Attachment boundary only."
    },
    brain: {
        noun: "Brain",
        activationRoot: "/activation/openclaw/current-profile",
        logRoot: "/activation/openclaw/current-profile/logs/learning-spine",
        activePackId: FIXTURE_RUNTIME_COMPILE_RESPONSE.packId,
        initMode: "fast_boot_defaults",
        state: "seed_state_authoritative",
        routeFreshness: "updated",
        routerIdentity: FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics.routerIdentity,
        routerChecksum: FIXTURE_ARTIFACT_MANIFEST.payloadChecksums.router,
        lastExportAt: FIXTURE_NORMALIZED_EVENT_EXPORT.range.lastCreatedAt,
        lastLearningUpdateAt: FIXTURE_ROUTER_ARTIFACT.trainedAt,
        lastPromotionAt: "2026-03-07T19:59:00.000Z",
        summary: "Brain is serving active pack pack-live-learning; learned routing is live, but authority is still seed-state.",
        detail: "The Brain is serving pack-live-learning from the active Brain slot behind the Host boundary."
    },
    hook: {
        noun: "Hook",
        scope: "exact_openclaw_home",
        openclawHome: "/Users/example/.openclaw-example",
        hookPath: "/Users/example/.openclaw-example/extensions/openclawbrain/index.ts",
        runtimeGuardPath: "/Users/example/.openclaw-example/extensions/openclawbrain/runtime-guard.js",
        manifestPath: "/Users/example/.openclaw-example/extensions/openclawbrain/openclaw.plugin.json",
        installState: "installed",
        loadability: "loadable",
        loadProof: "status_probe_ready",
        desynced: false,
        detail: "profile hook is installed at ~/.openclaw-example/extensions/openclawbrain"
    },
    attachment: {
        noun: "Attachment",
        state: "attached",
        activationRoot: "/activation/openclaw/current-profile",
        servingSlot: "active",
        policyMode: "dedicated",
        policy: FIXTURE_DEDICATED_BRAIN_ATTACHMENT_POLICY.policy,
        proofState: "self_proving",
        watchOnly: false,
        detail: "current profile is attached to a dedicated OpenClawBrain activation boundary"
    },
    brainStatus: {
        status: "ok",
        brainState: "seed_state_authoritative",
        serveState: "serving_active_pack",
        activationState: "healthy_seed",
        usedLearnedRouteFn: true,
        failOpen: false,
        awaitingFirstExport: false,
        structuralDecision: {
            origin: "unknown",
            basis: "unknown",
            requestedBudgetStrategy: null,
            resolvedBudgetStrategy: null,
            resolvedMaxContextBlocks: null,
            detail: "structural decision attribution is unavailable in this fixture answer"
        },
        timing: {
            scope: "brain_serve_hot_path_only",
            totalMs: 3.421,
            routeSelectionMs: 2.918,
            promptAssemblyMs: 0.207,
            otherMs: 0.296,
            backgroundWorkIncluded: false,
            detail: "Measured inside compileRuntimeContext before serve-route logging; includes serve-path normalization, active-pack lookup, structural-budget resolution, route/candidate selection, and prompt assembly when run; excludes background scanner/embedder/teacher work, promotion, and runtime event-export writes."
        },
        detail: "current profile is serving compiled context from the active promoted pack"
    },
    passiveLearning: {
        learnerRunning: true,
        firstExportOccurred: true,
        watchState: "watching",
        exportState: "latest_export_visible",
        backlogState: "caught_up",
        pendingLive: 0,
        pendingBackfill: 0,
        lastWatchHeartbeatAt: "2026-03-07T20:00:00.000Z",
        watchIntervalSeconds: 30,
        lastExportAt: FIXTURE_NORMALIZED_EVENT_EXPORT.range.lastCreatedAt,
        lastPromotionAt: "2026-03-07T19:59:00.000Z",
        currentServingPackId: FIXTURE_RUNTIME_COMPILE_RESPONSE.packId,
        lastMaterializedPackId: FIXTURE_RUNTIME_COMPILE_RESPONSE.packId,
        lastObservedDelta: {
            available: true,
            observedAt: "2026-03-07T19:59:00.000Z",
            exported: true,
            labeled: true,
            promoted: true,
            served: true,
            latestPackTransition: {
                kind: "promoted_active",
                fromPackId: "pack-seed-fixture",
                toPackId: FIXTURE_RUNTIME_COMPILE_RESPONSE.packId
            },
            explanation: "Latest user message was exported, labeled, promoted into the active pack, and is now served."
        },
        detail: "watch heartbeat is fresh, first export is proven, and passive backlog is caught up"
    },
    currentTurnAttribution: FIXTURE_PROFILE_TURN_ATTRIBUTION
};
export const FIXTURE_INTERACTION_EVENT = FIXTURE_INTERACTION_EVENTS[0];
/**
 * Validates a proposed `WorkspaceInjectionSurfaceV1` for obvious
 * misclassifications and missing required kernel kinds.
 *
 * Checks performed:
 * - Brain-eligible section roles must be compatible with their kind.
 * - Safety constraint and identity anchor are recommended as kernel.
 * - Ambiguous sections produce WARN findings.
 * - No brain-eligible content should be classified as kernel (heuristic check
 *   based on kind: `teaching_example` and `workspace_state` in kernel → FAIL).
 *
 * Returns a `KernelSurfaceValidationResultV1` with all findings. The caller
 * decides whether to act on WARN vs FAIL findings.
 *
 * Note: this validates the *descriptor* — it does not parse system prompt text.
 * Automatic text extraction is not yet supported.
 */
export function validateKernelSurface(surface) {
    const findings = [];
    // Kernel must have at least one identity_anchor or safety_constraint to be
    // useful; neither being present is a warning.
    const kernelKinds = new Set(surface.kernelSections.map((s) => s.kind));
    if (!kernelKinds.has("identity_anchor") && !kernelKinds.has("safety_constraint")) {
        findings.push({
            severity: "WARN",
            code: "KERNEL_NO_IDENTITY_OR_SAFETY",
            message: "Kernel surface has neither an identity_anchor nor a safety_constraint. " +
                "If this deployment has no safety or identity requirements, this is intentional. " +
                "Otherwise, verify that these are present in the system prompt."
        });
    }
    // Empty kernel is suspicious but not an immediate FAIL — some operators may
    // have a minimal system prompt.
    if (surface.kernelSections.length === 0) {
        findings.push({
            severity: "WARN",
            code: "KERNEL_EMPTY",
            message: "No kernel sections defined. If the system prompt is intentionally empty, " +
                "this is fine. Otherwise, add at least one kernel section."
        });
    }
    // Brain-eligible role compatibility check.
    const ROLE_COMPAT = {
        domain_knowledge: ["boot_default", "background_expectation"],
        workspace_state: ["workspace"],
        soft_behavioral_pref: ["label_surface"],
        project_context: ["structural"],
        teaching_example: ["teacher_supervision"]
    };
    for (const section of surface.brainEligibleSections) {
        const compatible = ROLE_COMPAT[section.kind];
        if (!compatible.includes(section.mappedRole)) {
            findings.push({
                severity: "WARN",
                code: "BRAIN_ROLE_MISMATCH",
                message: `Brain-eligible section "${section.description}" has kind "${section.kind}" ` +
                    `but mappedRole "${section.mappedRole}". ` +
                    `Expected one of: ${compatible.join(", ")}.`
            });
        }
    }
    // Ambiguous sections are always WARN.
    for (const section of surface.ambiguous ?? []) {
        findings.push({
            severity: "WARN",
            code: "AMBIGUOUS_SECTION",
            message: `Section "${section.description}" is ambiguous (tentative: ${section.tentativeCategory}). ` +
                "Resolve classification before using this surface in production."
        });
    }
    const severity = findings.reduce((worst, f) => {
        if (f.severity === "FAIL")
            return "FAIL";
        if (f.severity === "WARN" && worst === "PASS")
            return "WARN";
        return worst;
    }, "PASS");
    return { severity, findings };
}
export const FIXTURE_FEEDBACK_EVENT = FIXTURE_FEEDBACK_EVENTS[0];
//# sourceMappingURL=index.js.map
