import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { buildRouteArtifactReference, CONTRACT_IDS, PACK_GRAPH_SCHEMAS, ROUTER_PG_PROFILE_V1, ROUTER_PG_PROFILE_V2, checksumJsonPayload, computeRouterCollectedLabelCounts, computeRouterFreshnessChecksum, computeRouterObjectiveChecksum, computeRouterQueryChecksum, computeRouterWeightsChecksum, sortNormalizedEvents, validateTeacherSupervisionArtifact } from "@openclawbrain/contracts";
import { buildNormalizedEventDedupId, buildNormalizedEventExport, buildNormalizedEventExportBridge, createDefaultLearningSurface, createEventExportCursor, createExplicitEventRange, validateNormalizedEventExport, validateNormalizedEventExportBridge, validateNormalizedEventExportSlice } from "@openclawbrain/event-export";
import { computePayloadChecksum, loadPack, PACK_LAYOUT, summarizeStructuralGraphEvolution, writePackFile } from "@openclawbrain/pack-format";
import { buildArtifactProvenance } from "@openclawbrain/provenance";
import { createWorkspaceMetadata } from "@openclawbrain/workspace-metadata";
export const DEFAULT_ALWAYS_ON_LEARNING_LIVE_SLICES_PER_CYCLE = 1;
export const DEFAULT_ALWAYS_ON_LEARNING_BACKFILL_SLICES_PER_CYCLE = 1;
export const DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS = 5 * 60 * 1_000;
export const DEFAULT_POINTER_AWARE_WORKING_SET_LIMIT = 6;
export const DEFAULT_POINTER_AWARE_PASSIVE_EXPANSION_LIMIT = 12;
export const DEFAULT_ALWAYS_ON_STRUCTURAL_PLASTICITY_OPS = {
    split: 1,
    merge: 1,
    prune: 1,
    connect: 3
};
export const ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING = {
    split: 1,
    merge: 1,
    prune: 1,
    connect: 4
};
export const ALWAYS_ON_STRUCTURAL_PLASTICITY_MIN_INTERACTIONS = 2;
export const ALWAYS_ON_STRUCTURAL_PLASTICITY_MIN_FEEDBACK = 1;
const CONNECT_PAIR_SCORE_THRESHOLD = 2;
export const DEFAULT_SPARSE_FEEDBACK_POLICY = {
    teacherBudget: 32,
    teacherDelayMs: 0,
    feedbackMask: {
        correction: true,
        teaching: true,
        approval: true,
        suppression: true
    },
    backgroundLabelAmplification: 1
};
const IMPLICIT_POSITIVE_SOURCE_SUFFIX = "/implicit-positive";
const OPENCLAW_INIT_HEURISTIC_SCOPE = "init_priors_and_topology_only";
const OPENCLAW_INIT_LEARNED_LABEL_POLICY = "explicit_collected_labels_only";
const OPENCLAW_MARKDOWN_FILE_ROLES = [
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
];
function cloneOpenClawMarkdownFileRole(pathname) {
    const fileRole = OPENCLAW_MARKDOWN_FILE_ROLES.find((candidate) => candidate.path === pathname);
    if (fileRole === undefined) {
        throw new Error(`openclaw init graph requires an explicit file role for ${pathname}`);
    }
    return { ...fileRole };
}
function createOpenClawInitBlockMetadata(input) {
    return {
        nodeKind: input.nodeKind,
        sourceKind: input.sourceKind,
        fastBootRequired: true,
        passiveBackgroundLearningRequired: true,
        heuristicScope: OPENCLAW_INIT_HEURISTIC_SCOPE,
        learnedLabelPolicy: OPENCLAW_INIT_LEARNED_LABEL_POLICY,
        ...(input.sourceKind === "markdown" ? { fileRole: cloneOpenClawMarkdownFileRole(input.source) } : {})
    };
}
function createOpenClawInitGraphOntology() {
    return {
        schema: PACK_GRAPH_SCHEMAS.openclawInit,
        typedMarkdownSurface: true,
        fileRoles: OPENCLAW_MARKDOWN_FILE_ROLES.map((fileRole) => ({ ...fileRole })),
        fastBootRequired: true,
        passiveBackgroundLearningRequired: true,
        heuristicScope: OPENCLAW_INIT_HEURISTIC_SCOPE,
        learnedLabelPolicy: OPENCLAW_INIT_LEARNED_LABEL_POLICY
    };
}
function stableHash(value) {
    let hash = 0;
    for (const char of value) {
        hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
}
function normalizeSparseFeedbackPolicy(value) {
    const merged = {
        teacherBudget: value?.teacherBudget ?? DEFAULT_SPARSE_FEEDBACK_POLICY.teacherBudget,
        teacherDelayMs: value?.teacherDelayMs ?? DEFAULT_SPARSE_FEEDBACK_POLICY.teacherDelayMs,
        feedbackMask: {
            correction: value?.feedbackMask?.correction ?? DEFAULT_SPARSE_FEEDBACK_POLICY.feedbackMask.correction,
            teaching: value?.feedbackMask?.teaching ?? DEFAULT_SPARSE_FEEDBACK_POLICY.feedbackMask.teaching,
            approval: value?.feedbackMask?.approval ?? DEFAULT_SPARSE_FEEDBACK_POLICY.feedbackMask.approval,
            suppression: value?.feedbackMask?.suppression ?? DEFAULT_SPARSE_FEEDBACK_POLICY.feedbackMask.suppression
        },
        backgroundLabelAmplification: value?.backgroundLabelAmplification ?? DEFAULT_SPARSE_FEEDBACK_POLICY.backgroundLabelAmplification
    };
    assertNonNegativeInteger("sparseFeedback.teacherBudget", merged.teacherBudget);
    assertNonNegativeInteger("sparseFeedback.teacherDelayMs", merged.teacherDelayMs);
    if (!Number.isFinite(merged.backgroundLabelAmplification) || merged.backgroundLabelAmplification < 1) {
        throw new Error("sparseFeedback.backgroundLabelAmplification must be a finite number >= 1");
    }
    return merged;
}
function clampAlwaysOnStructuralOps(requested) {
    const requestedOrDefault = {
        split: requested?.split ?? DEFAULT_ALWAYS_ON_STRUCTURAL_PLASTICITY_OPS.split,
        merge: requested?.merge ?? DEFAULT_ALWAYS_ON_STRUCTURAL_PLASTICITY_OPS.merge,
        prune: requested?.prune ?? DEFAULT_ALWAYS_ON_STRUCTURAL_PLASTICITY_OPS.prune,
        connect: requested?.connect ?? DEFAULT_ALWAYS_ON_STRUCTURAL_PLASTICITY_OPS.connect
    };
    return {
        split: Math.min(requestedOrDefault.split, ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING.split),
        merge: Math.min(requestedOrDefault.merge, ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING.merge),
        prune: Math.min(requestedOrDefault.prune, ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING.prune),
        connect: Math.min(requestedOrDefault.connect, ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING.connect)
    };
}
function normalizeAlwaysOnLearningStructuralControlStrategy(value) {
    return value ?? "empirical_v1";
}
function normalizeAlwaysOnLearningCompileStructuralSignals(value) {
    if (value === undefined || value === null) {
        return null;
    }
    return {
        matchedCandidateCount: Math.max(0, value.matchedCandidateCount),
        selectedMatchedCount: Math.max(0, value.selectedMatchedCount),
        overlapPrunedCount: Math.max(0, value.overlapPrunedCount),
        traversalActivatedCount: Math.max(0, value.traversalActivatedCount)
    };
}
function resolveEmpiricalConnectBudget(traversalActivatedCount) {
    return Math.min(Math.max(0, traversalActivatedCount), ALWAYS_ON_STRUCTURAL_PLASTICITY_OP_CEILING.connect);
}
function createDefaultAlwaysOnLearningStructuralControllerState() {
    return {
        requestedStrategy: "empirical_v1",
        effectiveStrategy: "fixed_v1",
        source: "no_compile_signal_evidence_fallback",
        compileSignals: null,
        structuralOps: clampAlwaysOnStructuralOps(undefined)
    };
}
function resolveAlwaysOnLearningStructuralController(current, input) {
    if (input.structuralOps === undefined &&
        input.structuralControlStrategy === undefined &&
        input.compileStructuralSignals === undefined) {
        return structuredClone(current);
    }
    const requestedStrategy = normalizeAlwaysOnLearningStructuralControlStrategy(input.structuralControlStrategy ?? current.requestedStrategy);
    const compileSignals = normalizeAlwaysOnLearningCompileStructuralSignals(input.compileStructuralSignals);
    if (input.structuralOps !== undefined) {
        return {
            requestedStrategy,
            effectiveStrategy: requestedStrategy,
            source: "caller_override",
            compileSignals,
            structuralOps: clampAlwaysOnStructuralOps(input.structuralOps)
        };
    }
    if (requestedStrategy !== "empirical_v1") {
        return {
            requestedStrategy,
            effectiveStrategy: requestedStrategy,
            source: "fixed_default",
            compileSignals,
            structuralOps: clampAlwaysOnStructuralOps(undefined)
        };
    }
    const expansionCandidates = Math.max(0, (compileSignals?.matchedCandidateCount ?? 0) - (compileSignals?.selectedMatchedCount ?? 0));
    const overlapPrunedCount = Math.max(0, compileSignals?.overlapPrunedCount ?? 0);
    const traversalActivatedCount = Math.max(0, compileSignals?.traversalActivatedCount ?? 0);
    const evidenceTotal = expansionCandidates + overlapPrunedCount + traversalActivatedCount;
    if (evidenceTotal === 0) {
        return {
            requestedStrategy,
            effectiveStrategy: "fixed_v1",
            source: "no_compile_signal_evidence_fallback",
            compileSignals,
            structuralOps: clampAlwaysOnStructuralOps(undefined)
        };
    }
    return {
        requestedStrategy,
        effectiveStrategy: requestedStrategy,
        source: "compile_structural_signals_empirical_v1",
        compileSignals,
        structuralOps: clampAlwaysOnStructuralOps({
            split: expansionCandidates > 0 ? 1 : 0,
            merge: overlapPrunedCount > 0 ? 1 : 0,
            prune: overlapPrunedCount > 0 ? 1 : 0,
            connect: resolveEmpiricalConnectBudget(traversalActivatedCount)
        })
    };
}
function createSparseFeedbackRuntimeDiagnostics(policy, overrides = {}) {
    return {
        ...policy,
        eligibleFeedbackCount: overrides.eligibleFeedbackCount ?? 0,
        maskedFeedbackCount: overrides.maskedFeedbackCount ?? 0,
        delayedFeedbackCount: overrides.delayedFeedbackCount ?? 0,
        budgetedOutFeedbackCount: overrides.budgetedOutFeedbackCount ?? 0,
        amplifiedBackgroundLabelCount: overrides.amplifiedBackgroundLabelCount ?? 0
    };
}
function normalizeAlwaysOnStructuralOps(requested, normalizedEventExport) {
    const interactionCount = normalizedEventExport.interactionEvents.length;
    const feedbackCount = normalizedEventExport.feedbackEvents.length;
    const clamped = clampAlwaysOnStructuralOps(requested);
    const preserveSuppressionPrune = normalizedEventExport.feedbackEvents.some((event) => event.kind === "suppression" && event.relatedInteractionId !== undefined);
    if (interactionCount < ALWAYS_ON_STRUCTURAL_PLASTICITY_MIN_INTERACTIONS) {
        return {
            split: 0,
            merge: 0,
            prune: preserveSuppressionPrune ? clamped.prune : 0,
            connect: 0
        };
    }
    if (feedbackCount < ALWAYS_ON_STRUCTURAL_PLASTICITY_MIN_FEEDBACK) {
        return {
            split: 0,
            merge: 0,
            prune: preserveSuppressionPrune ? clamped.prune : 0,
            connect: clamped.connect
        };
    }
    return clamped;
}
function cloneCursor(value) {
    return {
        runtimeOwner: value.runtimeOwner,
        live: {
            after: value.live.after === null ? null : { ...value.live.after },
            exhausted: value.live.exhausted
        },
        backfill: {
            before: value.backfill.before === null ? null : { ...value.backfill.before },
            exhausted: value.backfill.exhausted
        }
    };
}
function cloneSliceWatermark(value) {
    return {
        first: value.first === null ? null : { ...value.first },
        last: value.last === null ? null : { ...value.last }
    };
}
function cloneNormalizedEventExportSlice(value) {
    return structuredClone(value);
}
function cloneAlwaysOnLearningRuntimeState(value) {
    const cloned = structuredClone(value);
    return {
        ...cloned,
        structuralController: cloned.structuralController ?? createDefaultAlwaysOnLearningStructuralControllerState()
    };
}
function cloneAlwaysOnLearningMaterializationJob(value) {
    return structuredClone(value);
}
function compareSliceRecency(left, right) {
    if (left.export.range.end !== right.export.range.end) {
        return right.export.range.end - left.export.range.end;
    }
    if (left.export.range.start !== right.export.range.start) {
        return right.export.range.start - left.export.range.start;
    }
    return left.sliceId.localeCompare(right.sliceId);
}
function sortPendingSlicesByRecency(slices) {
    return [...slices].sort(compareSliceRecency);
}
function materializeCandidatePackResult(rootDir, result) {
    rmSync(rootDir, { recursive: true, force: true });
    mkdirSync(rootDir, { recursive: true });
    writePackFile(rootDir, PACK_LAYOUT.graph, result.payloads.graph);
    writePackFile(rootDir, PACK_LAYOUT.vectors, result.payloads.vectors);
    if (result.payloads.router !== null) {
        writePackFile(rootDir, PACK_LAYOUT.router, result.payloads.router);
    }
    writePackFile(rootDir, PACK_LAYOUT.manifest, result.manifest);
    return loadPack(path.resolve(rootDir));
}
function buildBridgePackLabel(basePackLabel, slice, index) {
    return `${basePackLabel}-${String(index + 1).padStart(2, "0")}-${slice.lane}-${slice.export.range.start}-${slice.export.range.end}`;
}
function buildBundleEntryRootDir(rootDir, entry, index) {
    const digestSuffix = entry.sliceId.replace(/^sha256-/u, "").slice(0, 12);
    return path.join(rootDir, `${String(index + 1).padStart(2, "0")}-${entry.lane}-${entry.normalizedEventExport.range.start}-${entry.normalizedEventExport.range.end}-${digestSuffix}`);
}
function assertPositiveInteger(label, value) {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`${label} must be a positive integer`);
    }
}
function assertNonNegativeInteger(label, value) {
    if (!Number.isInteger(value) || value < 0) {
        throw new Error(`${label} must be a non-negative integer`);
    }
}
function newestIsoTimestamp(values) {
    return [...values].sort((left, right) => Date.parse(right) - Date.parse(left))[0] ?? values[0] ?? "1970-01-01T00:00:00.000Z";
}
function teacherAuthorityRank(artifact) {
    switch (artifact.principal?.teacherAuthority) {
        case "binding":
            return 5;
        case "primary_human":
            return 4;
        case "high":
            return 3;
        case "normal":
            return 2;
        case "background":
            return 1;
        default:
            return 0;
    }
}
function principalPriorityRank(artifact) {
    switch (artifact.principal?.priorityClass) {
        case "critical":
            return 4;
        case "high":
            return 3;
        case "normal":
            return 2;
        case "low":
            return 1;
        default:
            return 0;
    }
}
function principalRoleRank(artifact) {
    switch (artifact.principal?.teacherRole) {
        case "principal":
            return 4;
        case "admin":
            return 3;
        case "operator":
            return 2;
        case "user":
            return 1;
        default:
            return 0;
    }
}
function principalMetadataRouteRewardBoost(principal) {
    if (principal === null || principal === undefined) {
        return 0;
    }
    const priorityBoost = principal.priorityClass === "critical" ? 2 : principal.priorityClass === "high" ? 1.5 : principal.priorityClass === "normal" ? 1 : 0.5;
    const authorityBoost = principal.teacherAuthority === "binding"
        ? 1.5
        : principal.teacherAuthority === "primary_human"
            ? 1
            : principal.teacherAuthority === "high"
                ? 0.75
                : principal.teacherAuthority === "normal"
                    ? 0.5
                    : 0.25;
    const roleBoost = principal.teacherRole === "principal" ? 1 : principal.teacherRole === "admin" ? 0.5 : principal.teacherRole === "operator" ? 0.25 : 0;
    return roundMetric(priorityBoost + authorityBoost + roleBoost);
}
function principalBacklogCheckpoint(principal, principalBacklog) {
    if (principal === null || principal === undefined || principalBacklog === undefined) {
        return null;
    }
    return principalBacklog.checkpoints.find((checkpoint) => checkpoint.teacherIdentity === principal.teacherIdentity) ?? null;
}
function principalBacklogRouteRewardBoost(principal, principalBacklog) {
    const checkpoint = principalBacklogCheckpoint(principal, principalBacklog);
    if (checkpoint === null) {
        return principal !== null && principal !== undefined && principal.priorityClass === "critical" && (principalBacklog?.pendingEventCount ?? 0) > 0 ? 0.5 : 0;
    }
    return roundMetric(Math.min(3, checkpoint.pendingLiveEventCount + checkpoint.pendingBackfillEventCount * 0.5 + Math.max(0, checkpoint.pendingEventCount - 1) * 0.25));
}
function principalBacklogRoutingBoost(principal, principalBacklog) {
    const checkpoint = principalBacklogCheckpoint(principal, principalBacklog);
    if (checkpoint === null) {
        return {
            priority: 0,
            shortTermBias: 0,
            vectorBias: 0
        };
    }
    return {
        priority: roundMetric(Math.min(2.5, checkpoint.pendingEventCount * 0.5 + checkpoint.pendingLiveEventCount * 0.5)),
        shortTermBias: roundMetric(Math.min(2, checkpoint.pendingLiveEventCount * 0.75 + checkpoint.pendingBackfillEventCount * 0.25)),
        vectorBias: roundMetric(Math.min(1.5, checkpoint.pendingEventCount * 0.2 + checkpoint.pendingBackfillEventCount * 0.2))
    };
}
function compareTeacherSupervisionArtifacts(left, right) {
    if (left.freshness.status !== right.freshness.status) {
        return left.freshness.status === "fresh" ? -1 : 1;
    }
    const authorityDelta = teacherAuthorityRank(right) - teacherAuthorityRank(left);
    if (authorityDelta !== 0) {
        return authorityDelta;
    }
    const priorityDelta = principalPriorityRank(right) - principalPriorityRank(left);
    if (priorityDelta !== 0) {
        return priorityDelta;
    }
    const roleDelta = principalRoleRank(right) - principalRoleRank(left);
    if (roleDelta !== 0) {
        return roleDelta;
    }
    if (left.createdAt !== right.createdAt) {
        return Date.parse(right.createdAt) - Date.parse(left.createdAt);
    }
    return left.artifactId.localeCompare(right.artifactId);
}
function cloneTeacherSupervisionArtifact(value) {
    return structuredClone(value);
}
function normalizeTeacherSupervisionArtifacts(artifacts) {
    if (artifacts === undefined) {
        return [];
    }
    const deduped = new Map();
    for (const artifact of artifacts) {
        const validationErrors = validateTeacherSupervisionArtifact(artifact);
        if (validationErrors.length > 0) {
            throw new Error(`teacher supervision artifact is invalid: ${validationErrors.join("; ")}`);
        }
        const current = deduped.get(artifact.dedupId);
        if (current === undefined ||
            Date.parse(artifact.freshness.observedAt) > Date.parse(current.freshness.observedAt) ||
            (artifact.freshness.observedAt === current.freshness.observedAt && compareTeacherSupervisionArtifacts(artifact, current) < 0)) {
            deduped.set(artifact.dedupId, cloneTeacherSupervisionArtifact(artifact));
        }
    }
    const supersededEventIds = new Set();
    for (const artifact of deduped.values()) {
        for (const superseded of artifact.principal?.supersedes ?? []) {
            supersededEventIds.add(superseded);
        }
    }
    return [...deduped.values()]
        .filter((artifact) => !artifact.sourceEventIds.some((eventId) => supersededEventIds.has(eventId)))
        .sort(compareTeacherSupervisionArtifacts);
}
function teacherSupervisionContentForInteraction(event) {
    const message = event.messageId === undefined ? "" : ` Message: ${event.messageId}.`;
    const pack = event.packId === undefined ? "" : ` Pack: ${event.packId}.`;
    return `Operator override on ${event.channel} session ${event.sessionId}.${pack}${message}`;
}
export function buildTeacherSupervisionArtifactsFromNormalizedEventExport(input) {
    const validationErrors = validateNormalizedEventExport(input.normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    const staleAfterMs = input.staleAfterMs ?? DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS;
    assertPositiveInteger("staleAfterMs", staleAfterMs);
    const observedAt = input.observedAt ??
        input.normalizedEventExport.range.lastCreatedAt ??
        input.normalizedEventExport.range.firstCreatedAt ??
        "1970-01-01T00:00:00.000Z";
    const interactionsById = new Map(input.normalizedEventExport.interactionEvents.map((event) => [event.eventId, event]));
    const artifacts = [];
    const sparseFeedback = evaluateSparseFeedback(input.normalizedEventExport.feedbackEvents, observedAt, input.sparseFeedback);
    for (const feedback of input.normalizedEventExport.feedbackEvents) {
        if (!sparseFeedback.selectedEventIds.has(feedback.eventId)) {
            continue;
        }
        const relatedInteraction = feedback.relatedInteractionId === undefined ? undefined : interactionsById.get(feedback.relatedInteractionId);
        const sourceEvents = [feedback, ...(relatedInteraction === undefined ? [] : [relatedInteraction])];
        const newestSourceCreatedAt = newestIsoTimestamp(sourceEvents.map((event) => event.createdAt));
        const ageMs = Math.max(0, Date.parse(observedAt) - Date.parse(newestSourceCreatedAt));
        const dedupId = checksumJsonPayload({
            kind: feedback.kind,
            feedbackDedupId: buildNormalizedEventDedupId(feedback),
            relatedInteractionDedupId: relatedInteraction === undefined ? null : buildNormalizedEventDedupId(relatedInteraction),
            content: feedback.content,
            relatedInteractionId: feedback.relatedInteractionId ?? null,
            principal: feedback.principal ?? null
        });
        artifacts.push({
            contract: CONTRACT_IDS.teacherSupervisionArtifact,
            artifactId: `teacher-${dedupId}`,
            dedupId,
            kind: feedback.kind,
            createdAt: feedback.createdAt,
            source: {
                runtimeOwner: "openclaw",
                sessionId: feedback.sessionId,
                channel: feedback.channel,
                sourceStreams: [...new Set(sourceEvents.map((event) => event.source.stream))],
                eventRange: {
                    start: input.normalizedEventExport.range.start,
                    end: input.normalizedEventExport.range.end,
                    count: input.normalizedEventExport.range.count
                },
                eventExportDigest: input.normalizedEventExport.provenance.exportDigest
            },
            sourceEventIds: [...new Set(sourceEvents.map((event) => event.eventId))],
            relatedInteractionId: feedback.relatedInteractionId ?? null,
            ...(feedback.principal === undefined ? {} : { principal: feedback.principal }),
            content: feedback.content,
            freshness: {
                status: ageMs <= staleAfterMs ? "fresh" : "stale",
                observedAt,
                newestSourceCreatedAt,
                ageMs,
                staleAfterMs
            }
        });
    }
    for (const interaction of input.normalizedEventExport.interactionEvents) {
        if (interaction.kind !== "operator_override") {
            continue;
        }
        const ageMs = Math.max(0, Date.parse(observedAt) - Date.parse(interaction.createdAt));
        const dedupId = checksumJsonPayload({
            kind: interaction.kind,
            interactionDedupId: buildNormalizedEventDedupId(interaction),
            principal: interaction.principal ?? null
        });
        artifacts.push({
            contract: CONTRACT_IDS.teacherSupervisionArtifact,
            artifactId: `teacher-${dedupId}`,
            dedupId,
            kind: "operator_override",
            createdAt: interaction.createdAt,
            source: {
                runtimeOwner: "openclaw",
                sessionId: interaction.sessionId,
                channel: interaction.channel,
                sourceStreams: [interaction.source.stream],
                eventRange: {
                    start: input.normalizedEventExport.range.start,
                    end: input.normalizedEventExport.range.end,
                    count: input.normalizedEventExport.range.count
                },
                eventExportDigest: input.normalizedEventExport.provenance.exportDigest
            },
            sourceEventIds: [interaction.eventId],
            relatedInteractionId: null,
            ...(interaction.principal === undefined ? {} : { principal: interaction.principal }),
            content: teacherSupervisionContentForInteraction(interaction),
            freshness: {
                status: ageMs <= staleAfterMs ? "fresh" : "stale",
                observedAt,
                newestSourceCreatedAt: interaction.createdAt,
                ageMs,
                staleAfterMs
            }
        });
    }
    return normalizeTeacherSupervisionArtifacts(artifacts);
}
function normalizeAlwaysOnLearningCadence(input) {
    const cadence = {
        liveSlicesPerCycle: input?.liveSlicesPerCycle ?? DEFAULT_ALWAYS_ON_LEARNING_LIVE_SLICES_PER_CYCLE,
        backfillSlicesPerCycle: input?.backfillSlicesPerCycle ?? DEFAULT_ALWAYS_ON_LEARNING_BACKFILL_SLICES_PER_CYCLE
    };
    assertPositiveInteger("cadence.liveSlicesPerCycle", cadence.liveSlicesPerCycle);
    assertNonNegativeInteger("cadence.backfillSlicesPerCycle", cadence.backfillSlicesPerCycle);
    return cadence;
}
function mergePendingSlices(pending, discovered) {
    const live = pending.live.map(cloneNormalizedEventExportSlice);
    const backfill = pending.backfill.map(cloneNormalizedEventExportSlice);
    const seenSliceIds = new Set([...live, ...backfill].map((slice) => slice.sliceId));
    for (const slice of discovered) {
        if (seenSliceIds.has(slice.sliceId)) {
            continue;
        }
        if (slice.lane === "live") {
            live.push(cloneNormalizedEventExportSlice(slice));
        }
        else {
            backfill.push(cloneNormalizedEventExportSlice(slice));
        }
        seenSliceIds.add(slice.sliceId);
    }
    return {
        live: sortPendingSlicesByRecency(live),
        backfill: sortPendingSlicesByRecency(backfill)
    };
}
function mergeNormalizedEventExports(current, additions) {
    if (current === null && additions.length === 0) {
        return null;
    }
    const mergedEvents = sortNormalizedEvents([
        ...(current?.interactionEvents ?? []),
        ...(current?.feedbackEvents ?? []),
        ...additions.flatMap((slice) => [...slice.export.interactionEvents, ...slice.export.feedbackEvents])
    ]);
    const deduped = [];
    const seenDedupIds = new Set();
    for (const event of mergedEvents) {
        const dedupId = buildNormalizedEventDedupId(event);
        if (seenDedupIds.has(dedupId)) {
            continue;
        }
        seenDedupIds.add(dedupId);
        deduped.push(event);
    }
    return buildNormalizedEventExport({
        interactionEvents: deduped.filter((event) => event.contract === CONTRACT_IDS.interactionEvents),
        feedbackEvents: deduped.filter((event) => event.contract === CONTRACT_IDS.feedbackEvents)
    });
}
function createAlwaysOnLearningPendingSlices() {
    return {
        live: [],
        backfill: []
    };
}
const ALWAYS_ON_LEARNING_SCHEDULER_BUCKET_ORDER = [
    "principal_immediate",
    "principal_backfill",
    "live",
    "backfill"
];
function classifyPendingSliceBucket(slice, context, current) {
    const sliceEvents = [...slice.export.interactionEvents, ...slice.export.feedbackEvents];
    const sliceKeywords = uniqueKeywords(sliceEvents.flatMap((event) => keywordTokens(`${event.kind} ${event.channel} ${event.sessionId} ${event.source.stream} ${event.contract === CONTRACT_IDS.feedbackEvents ? event.content : event.messageId ?? ""}`)));
    const principalMetadata = sliceEvents.some((event) => event.principal !== undefined);
    const principalFeedback = slice.export.feedbackEvents.some((event) => event.kind === "correction" || event.kind === "teaching");
    const principalSession = sliceEvents.some((event) => context.sessionIds.includes(event.sessionId));
    const principalSource = slice.export.provenance.sourceStreams.some((stream) => context.sourceStreams.includes(stream));
    const frontierOverlap = context.frontierKeywords.length === 0 ? 0 : keywordOverlap(sliceKeywords, context.frontierKeywords);
    if (slice.lane === "live") {
        if (current === null || principalMetadata || principalFeedback || frontierOverlap >= 3) {
            return "principal_immediate";
        }
        return "live";
    }
    const backfillScore = scoreBackfillRouteValue(slice, context);
    if (principalMetadata || principalFeedback || principalSession || principalSource || backfillScore >= 6) {
        return "principal_backfill";
    }
    return "backfill";
}
function bucketPendingSlices(pending, current) {
    const context = buildBackfillPrioritizationContext(current);
    const buckets = {
        principal_immediate: [],
        principal_backfill: [],
        live: [],
        backfill: []
    };
    for (const slice of sortPendingSlicesByRecency(pending.live)) {
        buckets[classifyPendingSliceBucket(slice, context, current)].push(cloneNormalizedEventExportSlice(slice));
    }
    for (const slice of sortPendingBackfillSlices(pending.backfill, current)) {
        buckets[classifyPendingSliceBucket(slice, context, current)].push(cloneNormalizedEventExportSlice(slice));
    }
    return buckets;
}
function selectScheduledSlices(pending, current, cadence) {
    const buckets = bucketPendingSlices(pending, current);
    const pendingByBucket = {
        principal_immediate: buckets.principal_immediate.length,
        principal_backfill: buckets.principal_backfill.length,
        live: buckets.live.length,
        backfill: buckets.backfill.length
    };
    const selectedLive = [...buckets.principal_immediate, ...buckets.live]
        .slice(0, cadence.liveSlicesPerCycle)
        .map(cloneNormalizedEventExportSlice);
    const bootstrapLiveFirst = current === null && selectedLive.length > 0;
    const selectedBackfill = bootstrapLiveFirst
        ? []
        : [...buckets.principal_backfill, ...buckets.backfill]
            .slice(0, cadence.backfillSlicesPerCycle)
            .map(cloneNormalizedEventExportSlice);
    const selected = [...selectedLive, ...selectedBackfill];
    const selectedIds = new Set(selected.map((slice) => slice.sliceId));
    const remaining = {
        live: sortPendingSlicesByRecency(pending.live.filter((slice) => !selectedIds.has(slice.sliceId))),
        backfill: sortPendingBackfillSlices(pending.backfill.filter((slice) => !selectedIds.has(slice.sliceId)), current)
    };
    const remainingBuckets = bucketPendingSlices(remaining, current);
    return {
        selected,
        selectedBucket: selected.length === 0
            ? "none"
            : classifyPendingSliceBucket(selected[0], buildBackfillPrioritizationContext(current), current),
        remaining,
        nextPriorityBucket: ALWAYS_ON_LEARNING_SCHEDULER_BUCKET_ORDER.find((bucket) => remainingBuckets[bucket].length > 0) ?? "none",
        pendingByBucket
    };
}
export function createAlwaysOnLearningRuntimeState() {
    const sparseFeedback = normalizeSparseFeedbackPolicy(undefined);
    return {
        runtimeOwner: "openclaw",
        hotPathLearning: false,
        attachBlocksOnFullReplay: false,
        cursor: createEventExportCursor(),
        pending: createAlwaysOnLearningPendingSlices(),
        learnedEventExport: null,
        runtimeGraph: null,
        runtimePlasticity: null,
        learnedGraph: null,
        structuralController: createDefaultAlwaysOnLearningStructuralControllerState(),
        sparseFeedback: createSparseFeedbackRuntimeDiagnostics(sparseFeedback),
        lastMaterializedAt: null,
        materializationCount: 0
    };
}
function summarizePrincipalBacklog(state) {
    const checkpointByIdentity = new Map();
    const pendingEvents = [];
    for (const event of sortNormalizedEvents([
        ...(state.learnedEventExport?.interactionEvents ?? []),
        ...(state.learnedEventExport?.feedbackEvents ?? [])
    ])) {
        if (event.principal === undefined) {
            continue;
        }
        const current = checkpointByIdentity.get(event.principal.teacherIdentity);
        checkpointByIdentity.set(event.principal.teacherIdentity, {
            teacherIdentity: event.principal.teacherIdentity,
            teacherRole: event.principal.teacherRole,
            priorityClass: event.principal.priorityClass,
            learnedThroughSequence: event.sequence,
            learnedThroughCreatedAt: event.createdAt,
            pendingEventCount: current?.pendingEventCount ?? 0,
            pendingLiveEventCount: current?.pendingLiveEventCount ?? 0,
            pendingBackfillEventCount: current?.pendingBackfillEventCount ?? 0,
            oldestPendingSequence: current?.oldestPendingSequence ?? null,
            oldestPendingCreatedAt: current?.oldestPendingCreatedAt ?? null,
            newestPendingSequence: current?.newestPendingSequence ?? null,
            newestPendingCreatedAt: current?.newestPendingCreatedAt ?? null
        });
    }
    const pendingSlices = [...state.pending.live, ...state.pending.backfill];
    for (const slice of pendingSlices) {
        for (const event of sortNormalizedEvents([...slice.export.interactionEvents, ...slice.export.feedbackEvents])) {
            if (event.principal === undefined) {
                continue;
            }
            const pendingEvent = {
                teacherIdentity: event.principal.teacherIdentity,
                teacherRole: event.principal.teacherRole,
                priorityClass: event.principal.priorityClass,
                eventId: event.eventId,
                kind: event.kind,
                sequence: event.sequence,
                createdAt: event.createdAt,
                lane: slice.lane,
                sourceStream: event.source.stream
            };
            pendingEvents.push(pendingEvent);
            const current = checkpointByIdentity.get(event.principal.teacherIdentity) ?? {
                teacherIdentity: event.principal.teacherIdentity,
                teacherRole: event.principal.teacherRole,
                priorityClass: event.principal.priorityClass,
                learnedThroughSequence: null,
                learnedThroughCreatedAt: null,
                pendingEventCount: 0,
                pendingLiveEventCount: 0,
                pendingBackfillEventCount: 0,
                oldestPendingSequence: null,
                oldestPendingCreatedAt: null,
                newestPendingSequence: null,
                newestPendingCreatedAt: null
            };
            checkpointByIdentity.set(event.principal.teacherIdentity, {
                ...current,
                pendingEventCount: current.pendingEventCount + 1,
                pendingLiveEventCount: current.pendingLiveEventCount + (slice.lane === "live" ? 1 : 0),
                pendingBackfillEventCount: current.pendingBackfillEventCount + (slice.lane === "backfill" ? 1 : 0),
                oldestPendingSequence: current.oldestPendingSequence === null ? event.sequence : Math.min(current.oldestPendingSequence, event.sequence),
                oldestPendingCreatedAt: current.oldestPendingCreatedAt === null || Date.parse(event.createdAt) < Date.parse(current.oldestPendingCreatedAt)
                    ? event.createdAt
                    : current.oldestPendingCreatedAt,
                newestPendingSequence: current.newestPendingSequence === null ? event.sequence : Math.max(current.newestPendingSequence, event.sequence),
                newestPendingCreatedAt: current.newestPendingCreatedAt === null || Date.parse(event.createdAt) > Date.parse(current.newestPendingCreatedAt)
                    ? event.createdAt
                    : current.newestPendingCreatedAt
            });
        }
    }
    pendingEvents.sort((left, right) => {
        if (left.sequence !== right.sequence) {
            return left.sequence - right.sequence;
        }
        if (left.createdAt !== right.createdAt) {
            return Date.parse(left.createdAt) - Date.parse(right.createdAt);
        }
        return left.eventId.localeCompare(right.eventId);
    });
    const checkpoints = [...checkpointByIdentity.values()].sort((left, right) => {
        const priority = (value) => value === "critical" ? 0 : value === "high" ? 1 : value === "normal" ? 2 : value === "low" ? 3 : 4;
        const pendingDelta = right.pendingEventCount - left.pendingEventCount;
        if (pendingDelta !== 0) {
            return pendingDelta;
        }
        const priorityDelta = priority(left.priorityClass) - priority(right.priorityClass);
        if (priorityDelta !== 0) {
            return priorityDelta;
        }
        return left.teacherIdentity.localeCompare(right.teacherIdentity);
    });
    return {
        principalCount: checkpoints.length,
        pendingEventCount: pendingEvents.length,
        checkpoints,
        oldestUnlearnedEvent: pendingEvents[0] ?? null,
        newestPendingEvent: pendingEvents.at(-1) ?? null
    };
}
export function describeAlwaysOnLearningRuntimeState(state, lastMaterialization = null) {
    const livePending = state.pending.live.length;
    const backfillPending = state.pending.backfill.length;
    const pendingBuckets = bucketPendingSlices(state.pending, state.learnedEventExport);
    const pendingByBucket = {
        principal_immediate: pendingBuckets.principal_immediate.length,
        principal_backfill: pendingBuckets.principal_backfill.length,
        live: pendingBuckets.live.length,
        backfill: pendingBuckets.backfill.length
    };
    const nextPriorityBucket = ALWAYS_ON_LEARNING_SCHEDULER_BUCKET_ORDER.find((bucket) => pendingByBucket[bucket] > 0) ?? "none";
    const bootstrapped = state.learnedEventExport !== null;
    const principalBacklog = summarizePrincipalBacklog(state);
    const mode = !bootstrapped && state.materializationCount === 0
        ? "cold_start"
        : livePending > 0
            ? "live_priority"
            : backfillPending > 0
                ? "background_catchup"
                : "caught_up";
    return {
        runtimeOwner: state.runtimeOwner,
        hotPathLearning: state.hotPathLearning,
        attachBlocksOnFullReplay: state.attachBlocksOnFullReplay,
        bootstrapped,
        mode,
        nextPriorityLane: nextPriorityBucket === "principal_backfill" || nextPriorityBucket === "backfill" ? "backfill" : nextPriorityBucket === "none" ? "none" : "live",
        nextPriorityBucket,
        pending: {
            live: livePending,
            backfill: backfillPending,
            total: livePending + backfillPending,
            freshLivePriority: pendingByBucket.principal_immediate > 0 || pendingByBucket.live > 0,
            byBucket: pendingByBucket
        },
        principalBacklog,
        learnedRange: state.learnedEventExport === null ? null : { ...state.learnedEventExport.range },
        materialization: {
            count: state.materializationCount,
            lastMaterializedAt: state.lastMaterializedAt,
            lastJobId: lastMaterialization?.jobId ?? null,
            lastReason: lastMaterialization?.reason ?? null,
            lastLane: lastMaterialization?.lane ?? null,
            lastPriority: lastMaterialization?.priority ?? null,
            lastSchedulerBucket: lastMaterialization?.schedulerBucket ?? null
        }
    };
}
function hasPendingSlices(pending) {
    return pending.live.length > 0 || pending.backfill.length > 0;
}
function buildAlwaysOnLearningMaterializationJob(input, current, selectedSlices, normalizedEventExport, runtimeGraph, structuralController, schedulerBucket, principalBacklog) {
    const lane = selectedSlices.some((slice) => slice.lane === "live") ? "live" : "backfill";
    const reason = lane === "live"
        ? current.learnedEventExport === null
            ? "attach_bootstrap"
            : "fresh_live_events"
        : "passive_history_catchup";
    const structuralOps = normalizeAlwaysOnStructuralOps(structuralController.structuralOps, normalizedEventExport);
    const candidateInput = {
        packLabel: input.packLabel,
        workspace: input.workspace,
        normalizedEventExport,
        ...(input.teacherSupervisionArtifacts !== undefined ? { teacherSupervisionArtifacts: input.teacherSupervisionArtifacts } : {}),
        learnedRouting: input.learnedRouting,
        ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
        ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
        structuralOps,
        ...(runtimeGraph !== null ? { runtimeGraph } : {}),
        ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {}),
        principalBacklog,
        ...(input.pgVersion !== undefined ? { pgVersion: input.pgVersion } : {}),
        ...(input.serveTimeDecisions !== undefined ? { serveTimeDecisions: [...input.serveTimeDecisions] } : {}),
        ...(input.baselineState !== undefined ? { baselineState: { ...input.baselineState } } : {})
    };
    const candidate = buildCandidatePackFromNormalizedEventExport(candidateInput);
    const selectedSliceIds = selectedSlices.map((slice) => slice.sliceId);
    return {
        jobId: `learning-${stableHash(checksumJsonPayload({ lane, schedulerBucket, reason, exportDigest: normalizedEventExport.provenance.exportDigest, selectedSliceIds }))}`,
        lane,
        priority: lane === "live" ? "immediate" : "background",
        schedulerBucket,
        reason,
        selectedSliceIds,
        selectedEventRange: normalizedEventExport.range,
        normalizedEventExport,
        candidateInput,
        candidate
    };
}
export function advanceAlwaysOnLearningRuntime(input) {
    const cadence = normalizeAlwaysOnLearningCadence(input.cadence);
    const current = cloneAlwaysOnLearningRuntimeState(input.state ?? createAlwaysOnLearningRuntimeState());
    const structuralController = resolveAlwaysOnLearningStructuralController(current.structuralController, input);
    const sparseFeedback = normalizeSparseFeedbackPolicy(input.sparseFeedback ?? current.sparseFeedback);
    const bridge = buildNormalizedEventExportBridge({
        interactionEvents: [...input.interactionEvents],
        feedbackEvents: [...input.feedbackEvents],
        cursor: current.cursor,
        ...(input.liveSliceSize !== undefined ? { liveSliceSize: input.liveSliceSize } : {}),
        ...(input.backfillSliceSize !== undefined ? { backfillSliceSize: input.backfillSliceSize } : {})
    });
    const pending = mergePendingSlices(current.pending, bridge.slices);
    const schedule = selectScheduledSlices(pending, current.learnedEventExport, cadence);
    const selectedSlices = schedule.selected;
    const learnedEventExport = mergeNormalizedEventExports(current.learnedEventExport, selectedSlices);
    const runtimeGraphSnapshot = buildRuntimeGraphSnapshot({
        ...input,
        state: {
            ...current,
            structuralController
        }
    });
    const runtimeGraph = runtimeGraphSnapshot?.graph ?? current.runtimeGraph;
    const runtimePlasticity = runtimeGraphSnapshot?.plasticity ?? current.runtimePlasticity;
    const sparseFeedbackObservedAt = input.builtAt ?? learnedEventExport?.range.lastCreatedAt ?? learnedEventExport?.range.firstCreatedAt ?? current.lastMaterializedAt ?? "1970-01-01T00:00:00.000Z";
    const sparseFeedbackDiagnostics = createSparseFeedbackRuntimeDiagnostics(sparseFeedback);
    const nextSparseFeedback = learnedEventExport === null
        ? sparseFeedbackDiagnostics
        : evaluateSparseFeedback(learnedEventExport.feedbackEvents, sparseFeedbackObservedAt, sparseFeedback).diagnostics;
    const materialization = learnedEventExport === null || selectedSlices.length === 0
        ? null
        : buildAlwaysOnLearningMaterializationJob(input, current, selectedSlices, learnedEventExport, runtimeGraph, structuralController, schedule.selectedBucket === "none" ? "live" : schedule.selectedBucket, summarizePrincipalBacklog({
            ...current,
            pending: {
                live: pending.live.map(cloneNormalizedEventExportSlice),
                backfill: pending.backfill.map(cloneNormalizedEventExportSlice)
            }
        }));
    const nextState = {
        runtimeOwner: "openclaw",
        hotPathLearning: false,
        attachBlocksOnFullReplay: false,
        cursor: bridge.cursor,
        pending: {
            live: schedule.remaining.live.map(cloneNormalizedEventExportSlice),
            backfill: schedule.remaining.backfill.map(cloneNormalizedEventExportSlice)
        },
        learnedEventExport,
        runtimeGraph: runtimeGraph === null ? null : structuredClone(runtimeGraph),
        runtimePlasticity: runtimePlasticity === null ? null : structuredClone(runtimePlasticity),
        learnedGraph: materialization?.candidate.payloads.graph ?? current.learnedGraph,
        structuralController: structuredClone(structuralController),
        sparseFeedback: nextSparseFeedback,
        lastMaterializedAt: materialization?.candidate.manifest.provenance.builtAt ?? current.lastMaterializedAt,
        materializationCount: current.materializationCount + (materialization === null ? 0 : 1)
    };
    return {
        runtimeOwner: "openclaw",
        hotPathLearning: false,
        attachBlocksOnFullReplay: false,
        bridge: structuredClone(bridge),
        selectedSlices: selectedSlices.map(cloneNormalizedEventExportSlice),
        deferred: {
            live: nextState.pending.live.length,
            backfill: nextState.pending.backfill.length
        },
        materialization: materialization === null ? null : cloneAlwaysOnLearningMaterializationJob(materialization),
        state: cloneAlwaysOnLearningRuntimeState(nextState)
    };
}
export function drainAlwaysOnLearningRuntime(input) {
    const maxCycles = input.maxCycles ?? 64;
    assertPositiveInteger("maxCycles", maxCycles);
    const cycles = [];
    const materializations = [];
    let state = cloneAlwaysOnLearningRuntimeState(input.state ?? createAlwaysOnLearningRuntimeState());
    let stopReason = "max_cycles";
    for (let cycle = 1; cycle <= maxCycles; cycle += 1) {
        const result = advanceAlwaysOnLearningRuntime({
            ...input,
            state
        });
        cycles.push({
            cycle,
            ...structuredClone(result)
        });
        if (result.materialization !== null) {
            materializations.push(cloneAlwaysOnLearningMaterializationJob(result.materialization));
        }
        state = cloneAlwaysOnLearningRuntimeState(result.state);
        const idle = result.selectedSlices.length === 0 && result.bridge.slices.length === 0 && !hasPendingSlices(state.pending);
        if (idle) {
            stopReason = "idle";
            break;
        }
        if (result.selectedSlices.length === 0) {
            stopReason = "no_progress";
            break;
        }
    }
    return {
        runtimeOwner: "openclaw",
        drained: stopReason === "idle",
        stopReason,
        cycles,
        materializations,
        state
    };
}
export function materializeAlwaysOnLearningCandidatePack(rootDir, job) {
    return materializeCandidatePackResult(rootDir, job.candidate);
}
export async function materializeAlwaysOnLearningCandidatePackWithEmbedder(rootDir, job, embedder) {
    const result = await reindexCandidatePackBuildResultWithEmbedder(job.candidate, embedder);
    return materializeCandidatePackResult(rootDir, result);
}
export function buildCandidatePackFromNormalizedEventExportSlice(input) {
    const validationErrors = validateNormalizedEventExportSlice(input.normalizedEventExportSlice);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export slice is invalid: ${validationErrors.join("; ")}`);
    }
    return buildCandidatePackFromNormalizedEventExport({
        packLabel: input.packLabel,
        workspace: input.workspace,
        normalizedEventExport: input.normalizedEventExportSlice.export,
        ...(input.teacherSupervisionArtifacts !== undefined ? { teacherSupervisionArtifacts: input.teacherSupervisionArtifacts } : {}),
        learnedRouting: input.learnedRouting,
        ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
        ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
        ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {}),
        ...(input.runtimeGraph !== undefined ? { runtimeGraph: input.runtimeGraph } : {}),
        ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {}),
        ...(input.principalBacklog !== undefined ? { principalBacklog: input.principalBacklog } : {}),
        ...(input.pgVersion !== undefined ? { pgVersion: input.pgVersion } : {}),
        ...(input.serveTimeDecisions !== undefined ? { serveTimeDecisions: [...input.serveTimeDecisions] } : {}),
        ...(input.baselineState !== undefined ? { baselineState: { ...input.baselineState } } : {})
    });
}
export function buildCandidatePackBundleFromNormalizedEventExportBridge(input) {
    const validationErrors = validateNormalizedEventExportBridge(input.normalizedEventExportBridge);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export bridge is invalid: ${validationErrors.join("; ")}`);
    }
    const entries = input.normalizedEventExportBridge.slices.map((slice, index) => {
        const packLabel = buildBridgePackLabel(input.packLabel, slice, index);
        return {
            lane: slice.lane,
            sliceId: slice.sliceId,
            packLabel,
            normalizedEventExport: slice.export,
            nextCursor: cloneCursor(slice.nextCursor),
            watermark: cloneSliceWatermark(slice.watermark),
            build: buildCandidatePackFromNormalizedEventExportSlice({
                packLabel,
                workspace: input.workspace,
                normalizedEventExportSlice: slice,
                ...(input.teacherSupervisionArtifacts !== undefined ? { teacherSupervisionArtifacts: input.teacherSupervisionArtifacts } : {}),
                learnedRouting: input.learnedRouting,
                ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
                ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
                ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {}),
                ...(input.runtimeGraph !== undefined ? { runtimeGraph: input.runtimeGraph } : {}),
                ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {}),
                ...(input.principalBacklog !== undefined ? { principalBacklog: input.principalBacklog } : {}),
                ...(input.pgVersion !== undefined ? { pgVersion: input.pgVersion } : {}),
                ...(input.serveTimeDecisions !== undefined ? { serveTimeDecisions: [...input.serveTimeDecisions] } : {}),
                ...(input.baselineState !== undefined ? { baselineState: { ...input.baselineState } } : {})
            })
        };
    });
    const bundleDigest = checksumJsonPayload({
        runtimeOwner: input.normalizedEventExportBridge.runtimeOwner,
        bridgeDigest: input.normalizedEventExportBridge.bridgeDigest,
        cursor: input.normalizedEventExportBridge.cursor,
        entries: entries.map((entry) => ({
            lane: entry.lane,
            sliceId: entry.sliceId,
            packLabel: entry.packLabel,
            packId: entry.build.summary.packId,
            nextCursor: entry.nextCursor
        })),
        dedupedInputCount: input.normalizedEventExportBridge.dedupedInputCount,
        duplicateIdentityCount: input.normalizedEventExportBridge.duplicateIdentityCount
    });
    return {
        runtimeOwner: input.normalizedEventExportBridge.runtimeOwner,
        bridgeDigest: input.normalizedEventExportBridge.bridgeDigest,
        bundleDigest,
        cursor: cloneCursor(input.normalizedEventExportBridge.cursor),
        dedupedInputCount: input.normalizedEventExportBridge.dedupedInputCount,
        duplicateIdentityCount: input.normalizedEventExportBridge.duplicateIdentityCount,
        entries
    };
}
function structuralOpsSummary(input) {
    return {
        split: input.structuralOps?.split ?? 0,
        merge: input.structuralOps?.merge ?? 0,
        prune: input.structuralOps?.prune ?? 0,
        connect: input.structuralOps?.connect ?? 0
    };
}
const POINTER_AWARE_ANCHOR_BASENAMES = [
    "MEMORY.md",
    "AGENTS.md",
    "SOUL.md",
    "USER.md",
    "TOOLS.md",
    "IDENTITY.md",
    "HEARTBEAT.md"
];
const POINTER_AWARE_MARKDOWN_EXTENSIONS = new Set([".md", ".markdown", ".txt", ".rst"]);
const POINTER_AWARE_MAX_GRAPH_FILES = 24;
const POINTER_AWARE_MAX_GRAPH_DEPTH = 2;
const POINTER_AWARE_MAX_FILE_BYTES = 256_000;
const POINTER_AWARE_MAX_EXCERPT_CHARS = 320;
function uniqueInOrder(values) {
    return [...new Set(values)];
}
function emptyPointerAwareWorkingSetResult(rootDir, observedAt) {
    return {
        rootDir,
        observedAt,
        memoryPath: null,
        activeTasksPath: null,
        todaysMemoryPath: null,
        anchorPaths: [],
        bootInputs: [],
        workingSet: [],
        passiveExpansion: [],
        files: [],
        pointers: [],
        graphDigest: null
    };
}
function safeReadWorkspaceTextFile(rootDir, relativePath) {
    const resolvedPath = path.join(rootDir, relativePath.split("/").join(path.sep));
    try {
        const stats = statSync(resolvedPath);
        if (!stats.isFile() || stats.size > POINTER_AWARE_MAX_FILE_BYTES) {
            return null;
        }
        return readFileSync(resolvedPath, "utf8");
    }
    catch {
        return null;
    }
}
function normalizeWorkspaceRelativePath(rootDir, sourcePath, candidate) {
    const cleaned = candidate
        .trim()
        .replace(/^[<([{'"`]+/u, "")
        .replace(/[>)}'"`,.;:]+$/u, "")
        .split("#")[0]
        ?.split("?")[0]
        ?? "";
    if (cleaned.length === 0 || /^(?:[a-z]+:|\/\/)/iu.test(cleaned) || cleaned.startsWith("/")) {
        return null;
    }
    const normalizeResolvedPath = (resolvedPath) => {
        const relativePath = path.relative(rootDir, resolvedPath);
        if (relativePath.length === 0 || relativePath.startsWith("..") || path.isAbsolute(relativePath)) {
            return null;
        }
        const normalizedPath = relativePath.split(path.sep).join("/");
        return safeReadWorkspaceTextFile(rootDir, normalizedPath) === null ? null : normalizedPath;
    };
    const sourceRelative = normalizeResolvedPath(path.resolve(rootDir, path.dirname(sourcePath), cleaned));
    if (sourceRelative !== null) {
        return sourceRelative;
    }
    if (!cleaned.startsWith(".")) {
        return normalizeResolvedPath(path.resolve(rootDir, cleaned));
    }
    return null;
}
function strongerPointerHint(current, candidate) {
    const rank = {
        memory_index: 3,
        markdown_link: 2,
        bare_path: 1
    };
    return rank[candidate] > rank[current] ? candidate : current;
}
function extractPointerCandidates(rootDir, sourcePath, text) {
    const candidates = new Map();
    const noteCandidate = (rawValue, hint) => {
        const normalizedPath = normalizeWorkspaceRelativePath(rootDir, sourcePath, rawValue);
        if (normalizedPath === null || normalizedPath === sourcePath) {
            return;
        }
        const existingHint = candidates.get(normalizedPath);
        candidates.set(normalizedPath, existingHint === undefined ? hint : strongerPointerHint(existingHint, hint));
    };
    for (const match of text.matchAll(/\[[^\]]+\]\(([^)]+)\)/gu)) {
        const rawValue = match[1];
        if (rawValue !== undefined) {
            noteCandidate(rawValue, "markdown_link");
        }
    }
    const pathPattern = /(?:\.{0,2}\/)?(?:[A-Za-z0-9._-]+\/)*[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+/gu;
    for (const line of text.split(/\r?\n/u)) {
        const hint = /^\s*(?:[-*+]|\d+\.)\s+/u.test(line) ? "memory_index" : "bare_path";
        for (const match of line.matchAll(pathPattern)) {
            noteCandidate(match[0], hint);
        }
    }
    return [...candidates.entries()]
        .sort((left, right) => left[0].localeCompare(right[0]))
        .map(([candidatePath, hint]) => ({
        path: candidatePath,
        hint
    }));
}
function isPointerGraphFile(relativePath) {
    return POINTER_AWARE_MARKDOWN_EXTENSIONS.has(path.extname(relativePath).toLowerCase());
}
function listWorkspaceMemoryFiles(rootDir) {
    const memoryRoot = path.join(rootDir, "memory");
    if (!existsSync(memoryRoot)) {
        return [];
    }
    try {
        return readdirSync(memoryRoot, { withFileTypes: true })
            .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".md"))
            .map((entry) => `memory/${entry.name}`)
            .sort((left, right) => left.localeCompare(right));
    }
    catch {
        return [];
    }
}
function selectTodaysMemoryPath(memoryFiles, observedAt) {
    const datePrefix = observedAt.slice(0, 10);
    const matches = memoryFiles.filter((filePath) => path.posix.basename(filePath).startsWith(`${datePrefix}-`));
    return matches[0] ?? null;
}
function summarizeWorkspaceExcerpt(content) {
    const normalized = content
        .replace(/```[\s\S]*?```/gu, " ")
        .replace(/`([^`]+)`/gu, "$1")
        .replace(/\[[^\]]+\]\(([^)]+)\)/gu, "$1")
        .replace(/^#+\s+/gmu, "")
        .replace(/\s+/gu, " ")
        .trim();
    if (normalized.length <= POINTER_AWARE_MAX_EXCERPT_CHARS) {
        return normalized;
    }
    return `${normalized.slice(0, POINTER_AWARE_MAX_EXCERPT_CHARS - 1).trimEnd()}…`;
}
function pointerPriorityScore(relativePath, discovery, memoryPath, activeTasksPath, todaysMemoryPath) {
    const basename = path.posix.basename(relativePath);
    let score = discovery.inboundCount * 35 + discovery.outboundCount * 12 - discovery.depth * 20;
    if (relativePath === activeTasksPath) {
        score += 1_000;
    }
    if (relativePath === todaysMemoryPath) {
        score += 950;
    }
    if (relativePath === memoryPath) {
        score += 900;
    }
    if (POINTER_AWARE_ANCHOR_BASENAMES.includes(basename)) {
        score += 800;
    }
    if (relativePath.startsWith("memory/")) {
        score += 120;
    }
    if (discovery.directFromMemory) {
        score += 180;
    }
    if (discovery.directFromHotBoot) {
        score += 160;
    }
    if (discovery.directFromAnchor) {
        score += 120;
    }
    return score;
}
function previewPaths(paths, count) {
    if (paths.length === 0) {
        return "none";
    }
    const head = paths.slice(0, count).join(", ");
    return paths.length <= count ? head : `${head}, +${paths.length - count} more`;
}
export function buildPointerAwareWorkingSet(input) {
    const observedAt = input.observedAt ?? "1970-01-01T00:00:00.000Z";
    const rootDir = path.resolve(input.rootDir);
    const empty = emptyPointerAwareWorkingSetResult(rootDir, observedAt);
    if (!existsSync(rootDir)) {
        return empty;
    }
    const memoryPath = safeReadWorkspaceTextFile(rootDir, "MEMORY.md") === null ? null : "MEMORY.md";
    const activeTasksPath = safeReadWorkspaceTextFile(rootDir, "active-tasks.md") === null ? null : "active-tasks.md";
    const anchorPaths = POINTER_AWARE_ANCHOR_BASENAMES.filter((filePath) => filePath !== memoryPath && safeReadWorkspaceTextFile(rootDir, filePath) !== null);
    const todaysMemoryPath = selectTodaysMemoryPath(listWorkspaceMemoryFiles(rootDir), observedAt);
    const bootInputs = uniqueInOrder([activeTasksPath, todaysMemoryPath, memoryPath, ...anchorPaths].filter((value) => value !== null));
    if (bootInputs.length === 0) {
        return empty;
    }
    const discoveries = new Map();
    const pointers = [];
    const queued = new Set();
    const visited = new Set();
    const queue = bootInputs
        .filter((filePath) => isPointerGraphFile(filePath))
        .map((filePath) => ({ filePath, depth: 0 }));
    const ensureDiscovery = (filePath, depth) => {
        const current = discoveries.get(filePath);
        if (current !== undefined) {
            current.depth = Math.min(current.depth, depth);
            return current;
        }
        const created = {
            depth,
            inboundCount: 0,
            outboundCount: 0,
            directFromMemory: false,
            directFromHotBoot: false,
            directFromAnchor: false
        };
        discoveries.set(filePath, created);
        return created;
    };
    for (const filePath of bootInputs) {
        ensureDiscovery(filePath, 0);
    }
    while (queue.length > 0 && visited.size < POINTER_AWARE_MAX_GRAPH_FILES) {
        const current = queue.shift();
        if (current === undefined || visited.has(current.filePath)) {
            continue;
        }
        visited.add(current.filePath);
        const content = safeReadWorkspaceTextFile(rootDir, current.filePath);
        if (content === null) {
            continue;
        }
        const sourceDiscovery = ensureDiscovery(current.filePath, current.depth);
        const outgoingTargets = new Set();
        for (const candidate of extractPointerCandidates(rootDir, current.filePath, content)) {
            if (outgoingTargets.has(candidate.path)) {
                continue;
            }
            outgoingTargets.add(candidate.path);
            const targetDiscovery = ensureDiscovery(candidate.path, current.depth + 1);
            targetDiscovery.inboundCount += 1;
            if (current.filePath === memoryPath) {
                targetDiscovery.directFromMemory = true;
            }
            if (current.filePath === activeTasksPath || current.filePath === todaysMemoryPath) {
                targetDiscovery.directFromHotBoot = true;
            }
            if (anchorPaths.includes(current.filePath)) {
                targetDiscovery.directFromAnchor = true;
            }
            sourceDiscovery.outboundCount += 1;
            pointers.push({
                sourcePath: current.filePath,
                targetPath: candidate.path,
                depth: current.depth + 1,
                hint: candidate.hint
            });
            if (current.depth < POINTER_AWARE_MAX_GRAPH_DEPTH &&
                isPointerGraphFile(candidate.path) &&
                !visited.has(candidate.path) &&
                !queued.has(candidate.path)) {
                queue.push({ filePath: candidate.path, depth: current.depth + 1 });
                queued.add(candidate.path);
            }
        }
    }
    const rankedPaths = [...discoveries.entries()]
        .map(([filePath, discovery]) => ({
        filePath,
        priority: pointerPriorityScore(filePath, discovery, memoryPath, activeTasksPath, todaysMemoryPath),
        discovery
    }))
        .sort((left, right) => {
        if (right.priority !== left.priority) {
            return right.priority - left.priority;
        }
        return left.filePath.localeCompare(right.filePath);
    });
    const bootInputSet = new Set(bootInputs);
    const workingSet = rankedPaths
        .filter((entry) => !bootInputSet.has(entry.filePath))
        .slice(0, input.workingSetLimit ?? DEFAULT_POINTER_AWARE_WORKING_SET_LIMIT)
        .map((entry) => entry.filePath);
    const workingSetSet = new Set(workingSet);
    const passiveExpansion = rankedPaths
        .filter((entry) => !bootInputSet.has(entry.filePath) && !workingSetSet.has(entry.filePath))
        .slice(0, input.passiveExpansionLimit ?? DEFAULT_POINTER_AWARE_PASSIVE_EXPANSION_LIMIT)
        .map((entry) => entry.filePath);
    const selectedFiles = [...bootInputs, ...workingSet].map((filePath) => {
        const discovery = discoveries.get(filePath) ?? ensureDiscovery(filePath, 0);
        const content = safeReadWorkspaceTextFile(rootDir, filePath) ?? "";
        return {
            path: filePath,
            layer: bootInputSet.has(filePath) ? "anchor" : "working_set",
            priority: pointerPriorityScore(filePath, discovery, memoryPath, activeTasksPath, todaysMemoryPath),
            excerpt: summarizeWorkspaceExcerpt(content),
            inboundCount: discovery.inboundCount,
            outboundCount: discovery.outboundCount
        };
    });
    const graphDigest = checksumJsonPayload({
        bootInputs,
        workingSet,
        passiveExpansion,
        pointers
    });
    return {
        rootDir,
        observedAt,
        memoryPath,
        activeTasksPath,
        todaysMemoryPath,
        anchorPaths,
        bootInputs,
        workingSet,
        passiveExpansion,
        files: selectedFiles,
        pointers,
        graphDigest
    };
}
function summarizePointerAwareWorkingSet(result) {
    return {
        pointerAware: result.graphDigest !== null,
        memoryPath: result.memoryPath,
        bootInputs: [...result.bootInputs],
        workingSet: [...result.workingSet],
        passiveExpansion: [...result.passiveExpansion],
        graphDigest: result.graphDigest
    };
}
function buildWorkspaceInitGraphSeed(packId, workspace, observedAt, result) {
    if (result.graphDigest === null || result.bootInputs.length === 0) {
        return {
            blocks: [],
            metadata: new Map(),
            seedEdges: new Map()
        };
    }
    const blocks = [];
    const metadata = new Map();
    const seedEdges = new Map();
    const blockIds = new Map();
    const summaryBlockId = `${packId}:pointer-aware-init`;
    const passiveBlockId = result.passiveExpansion.length === 0 ? null : `${packId}:pointer-passive-expansion`;
    blocks.push({
        id: summaryBlockId,
        source: result.memoryPath ?? workspace.rootDir,
        text: `Pointer-aware init keeps fast boot first with anchors ${previewPaths(result.bootInputs, 4)}, working set ${previewPaths(result.workingSet, 4)}, and passive expansion ${previewPaths(result.passiveExpansion, 4)} from the MEMORY index graph.`,
        keywords: uniqueKeywords([
            "pointer",
            "init",
            "working",
            "set",
            "anchor",
            "passive",
            ...keywordTokens(result.bootInputs.join(" ")),
            ...keywordTokens(result.workingSet.join(" "))
        ]),
        priority: 6,
        routing: routingHints(["graph", "short_term", "vector"], {
            graphBias: 2,
            shortTermBias: 2,
            vectorBias: 2
        }),
        learning: learningSignals({
            role: "boot_default",
            decayHalfLifeDays: null,
            hebbianPulse: 3
        })
    });
    metadata.set(summaryBlockId, {
        createdAt: observedAt,
        sourceStream: result.memoryPath ?? workspace.rootDir,
        syntheticRole: "base",
        splitDepth: 0
    });
    seedEdges.set(summaryBlockId, []);
    for (const file of result.files) {
        const blockId = `${packId}:workspace-init:${stableHash(file.path)}`;
        blockIds.set(file.path, blockId);
        const topPriority = file.path === result.activeTasksPath || file.path === result.todaysMemoryPath;
        const priority = file.layer === "anchor" ? (topPriority ? 7 : 6) : 5;
        const layerLabel = file.layer === "anchor" ? "Top-priority boot input" : "Working-set input";
        blocks.push({
            id: blockId,
            source: file.path,
            text: `${layerLabel} ${file.path} selected by pointer-aware init (inbound=${file.inboundCount}, outbound=${file.outboundCount}). ${file.excerpt}`.trim(),
            keywords: uniqueKeywords([
                ...(file.layer === "anchor" ? ["anchor", "boot", "pointer"] : ["working", "set", "pointer"]),
                ...keywordTokens(file.path),
                ...keywordTokens(file.excerpt)
            ]),
            priority,
            routing: routingHints(file.layer === "anchor" ? ["graph", "short_term", "vector"] : ["graph", "vector"], {
                graphBias: file.layer === "anchor" ? 2 : 1,
                shortTermBias: file.layer === "anchor" ? 2 : 0,
                vectorBias: 2
            }),
            learning: learningSignals({
                role: "workspace",
                decayHalfLifeDays: null,
                hebbianPulse: file.layer === "anchor" ? 3 : 2
            })
        });
        metadata.set(blockId, {
            createdAt: observedAt,
            sourceStream: file.path,
            syntheticRole: "base",
            splitDepth: 0
        });
        seedEdges.set(blockId, []);
        addEdge(seedEdges, summaryBlockId, {
            targetBlockId: blockId,
            kind: "connect",
            weight: priority
        });
        addEdge(seedEdges, blockId, {
            targetBlockId: summaryBlockId,
            kind: "connect",
            weight: Math.max(2, priority - 1)
        });
    }
    if (passiveBlockId !== null) {
        blocks.push({
            id: passiveBlockId,
            source: result.memoryPath ?? workspace.rootDir,
            text: `Passive expansion keeps ${result.passiveExpansion.length} pointer-reachable files off the hot path after handoff: ${previewPaths(result.passiveExpansion, 6)}.`,
            keywords: uniqueKeywords([
                "passive",
                "expansion",
                "pointer",
                "background",
                ...keywordTokens(result.passiveExpansion.join(" "))
            ]),
            priority: 4,
            routing: routingHints(["graph", "vector"], {
                graphBias: 1,
                vectorBias: 2
            }),
            learning: learningSignals({
                role: "background_expectation",
                decayHalfLifeDays: 30,
                hebbianPulse: Math.max(1, Math.min(4, result.passiveExpansion.length))
            })
        });
        metadata.set(passiveBlockId, {
            createdAt: observedAt,
            sourceStream: result.memoryPath ?? workspace.rootDir,
            syntheticRole: "base",
            splitDepth: 0
        });
        seedEdges.set(passiveBlockId, []);
        addEdge(seedEdges, summaryBlockId, {
            targetBlockId: passiveBlockId,
            kind: "connect",
            weight: 4
        });
        addEdge(seedEdges, passiveBlockId, {
            targetBlockId: summaryBlockId,
            kind: "connect",
            weight: 3
        });
    }
    const passiveExpansionSet = new Set(result.passiveExpansion);
    for (const pointer of result.pointers) {
        const sourceBlockId = blockIds.get(pointer.sourcePath);
        const targetBlockId = blockIds.get(pointer.targetPath) ?? (passiveBlockId !== null && passiveExpansionSet.has(pointer.targetPath) ? passiveBlockId : undefined);
        if (sourceBlockId === undefined || targetBlockId === undefined || sourceBlockId === targetBlockId) {
            continue;
        }
        addEdge(seedEdges, sourceBlockId, {
            targetBlockId,
            kind: "connect",
            weight: Math.max(2, 6 - pointer.depth)
        });
    }
    return {
        blocks,
        metadata,
        seedEdges
    };
}
function topFrequencyKeywords(values, limit) {
    const counts = new Map();
    for (const value of values) {
        for (const token of keywordTokens(value)) {
            counts.set(token, (counts.get(token) ?? 0) + 1);
        }
    }
    return [...counts.entries()]
        .sort((left, right) => {
        if (right[1] !== left[1]) {
            return right[1] - left[1];
        }
        return left[0].localeCompare(right[0]);
    })
        .slice(0, limit)
        .map(([token]) => token);
}
function datedSourceTimestamp(source) {
    const match = source.match(/(20\d{2}-\d{2}-\d{2})/u);
    if (match === null) {
        return null;
    }
    const iso = `${match[1]}T12:00:00.000Z`;
    return Number.isNaN(Date.parse(iso)) ? null : iso;
}
function classifyInitFileRole(source, builtAt) {
    const normalized = source.toLowerCase();
    const datedSource = datedSourceTimestamp(source);
    if (/(^|\/)(agents|soul|user|tools)\.md$/u.test(normalized)) {
        return "anchor";
    }
    if (/(^|\/)memory\.md$/u.test(normalized)) {
        return "pointer_index";
    }
    if (/(^|\/)active-tasks\.md$/u.test(normalized)) {
        return "working_set";
    }
    if (/memory\/20\d{2}-\d{2}-\d{2}.*\.md$/u.test(normalized)) {
        if (datedSource !== null && daysBetween(datedSource, builtAt) <= 14) {
            return "recent_memory";
        }
        return "archived_memory";
    }
    if (/correction|teaching|lesson|teach|fix/u.test(normalized)) {
        return "correction_log";
    }
    if (normalized.startsWith("docs/") || normalized.startsWith("contracts/") || normalized.endsWith("readme.md")) {
        return "reference";
    }
    if (normalized.startsWith("openclaw/runtime/") || normalized.includes(":memory_compiled") || normalized.endsWith(".v1")) {
        return "event_stream";
    }
    if (normalized.startsWith("/workspace/") || normalized.includes("workspace")) {
        return "workspace";
    }
    return "synthetic";
}
function inferInitNodeType(source, keywords, fileRole) {
    const keywordSet = new Set(keywords);
    switch (fileRole) {
        case "anchor":
            return "rule";
        case "pointer_index":
            return "pointer";
        case "working_set":
            return "task";
        case "recent_memory":
        case "archived_memory":
        case "event_stream":
            return "event";
        case "workspace":
            return "project";
        case "correction_log":
            return keywordSet.has("correction") || keywordSet.has("teaching") ? "rule" : "event";
        case "reference":
            if (keywordSet.has("policy") || keywordSet.has("rule")) {
                return "rule";
            }
            if (keywordSet.has("project")) {
                return "project";
            }
            return "file";
        case "synthetic":
        default:
            if (keywordSet.has("pointer") || keywordSet.has("index")) {
                return "pointer";
            }
            if (keywordSet.has("task")) {
                return "task";
            }
            if (keywordSet.has("project")) {
                return "project";
            }
            if (keywordSet.has("person") || keywordSet.has("user")) {
                return "person";
            }
            if (keywordSet.has("entity")) {
                return "entity";
            }
            return "file";
    }
}
function initRoleScore(nodeType, fileRole) {
    const nodeScore = {
        file: 2.2,
        section: 2,
        task: 3.6,
        rule: 3.5,
        person: 2.4,
        project: 2.7,
        pointer: 3.8,
        event: 3,
        entity: 2.5
    };
    const roleBonus = {
        anchor: 1.4,
        working_set: 1.3,
        pointer_index: 1.5,
        recent_memory: 0.9,
        archived_memory: 0.1,
        correction_log: 1.2,
        reference: 0.7,
        workspace: 1,
        event_stream: 0.8,
        synthetic: 0.4
    };
    return roundMetric(nodeScore[nodeType] + roleBonus[fileRole]);
}
function initAuthorityScore(fileRole, source) {
    switch (fileRole) {
        case "anchor":
            return 4;
        case "pointer_index":
            return 3.8;
        case "working_set":
            return 3.5;
        case "correction_log":
            return 3.4;
        case "recent_memory":
            return 3;
        case "workspace":
            return 2.8;
        case "reference":
            return 2.4;
        case "event_stream":
            return /:correction|:teaching|:operator_override/u.test(source) ? 3.2 : 2.5;
        case "archived_memory":
            return 1.5;
        case "synthetic":
        default:
            return 1.6;
    }
}
function initRecencyScore(createdAt, builtAt, fileRole) {
    const halfLifeDays = fileRole === "working_set" ? 3 : fileRole === "recent_memory" ? 7 : fileRole === "archived_memory" ? 21 : 14;
    return roundMetric(decayFreshness(createdAt, builtAt, halfLifeDays) * 4);
}
function initStalenessPenalty(createdAt, builtAt, fileRole) {
    const freshness = decayFreshness(createdAt, builtAt, 14);
    const fileRolePenalty = fileRole === "archived_memory" ? 2 : fileRole === "reference" ? 0.75 : 0;
    return roundMetric((1 - freshness) * 3 + fileRolePenalty);
}
function correctionDensityScore(block, correctionKeywords) {
    const keywordHits = keywordOverlap(block.keywords, correctionKeywords);
    const sourceBoost = /:correction|:teaching|:approval|:suppression|teacher/u.test(block.source) ? 1.5 : 0;
    const labelBoost = block.learning.humanLabels > 0 ? 1 : 0;
    return roundMetric(Math.min(4, keywordHits * 0.5 + sourceBoost + labelBoost));
}
function seededRoutingHintsForNodeType(nodeType, fileRole, score) {
    const dominantBias = score >= 11 ? 4 : score >= 8 ? 3 : 2;
    const anchorGraphBoost = fileRole === "anchor" || fileRole === "pointer_index" ? 1 : 0;
    const workingSetShortTermBoost = fileRole === "working_set" ? 1 : 0;
    const correctionBoost = fileRole === "correction_log" ? 1 : 0;
    switch (nodeType) {
        case "pointer":
            return routingHints(["graph", "vector"], {
                graphBias: dominantBias + 1 + anchorGraphBoost,
                vectorBias: 2
            });
        case "task":
            return routingHints(["short_term", "graph"], {
                shortTermBias: dominantBias + 1 + workingSetShortTermBoost,
                graphBias: 2
            });
        case "rule":
            return routingHints(["graph", "short_term", "vector"], {
                graphBias: dominantBias + anchorGraphBoost,
                shortTermBias: 2 + correctionBoost,
                vectorBias: 1
            });
        case "event":
            return routingHints(["short_term", "vector"], {
                shortTermBias: dominantBias + workingSetShortTermBoost + correctionBoost,
                vectorBias: 2 + correctionBoost
            });
        case "project":
            return routingHints(["graph", "vector"], {
                graphBias: 2,
                vectorBias: dominantBias
            });
        case "person":
        case "entity":
            return routingHints(["vector", "graph"], {
                vectorBias: dominantBias + 1,
                graphBias: 1
            });
        case "section":
        case "file":
        default:
            return routingHints(fileRole === "reference" ? ["vector"] : ["graph", "vector"], {
                graphBias: fileRole === "reference" ? 0 : 1 + anchorGraphBoost,
                vectorBias: dominantBias
            });
    }
}
function learningRoleForInitFileRole(fileRole) {
    switch (fileRole) {
        case "anchor":
        case "working_set":
            return "boot_default";
        case "workspace":
            return "workspace";
        case "correction_log":
            return "label_surface";
        case "pointer_index":
        case "recent_memory":
        case "archived_memory":
            return "background_expectation";
        case "reference":
        case "event_stream":
        case "synthetic":
        default:
            return "structural";
    }
}
function initPriorityForFileRole(fileRole) {
    switch (fileRole) {
        case "anchor":
        case "working_set":
        case "pointer_index":
            return 5;
        case "recent_memory":
        case "workspace":
        case "correction_log":
            return 4;
        case "reference":
        case "event_stream":
        case "synthetic":
            return 3;
        case "archived_memory":
        default:
            return 2;
    }
}
function initDecayHalfLifeDays(fileRole) {
    switch (fileRole) {
        case "anchor":
        case "working_set":
            return null;
        case "recent_memory":
            return 14;
        case "archived_memory":
            return 45;
        default:
            return 30;
    }
}
function buildInitSeedContext(input, workspace, eventExport, teacherArtifacts) {
    const offlineArtifacts = input.offlineArtifacts ?? [];
    const feedbackTexts = eventExport?.feedbackEvents.map((event) => event.content) ?? [];
    const teacherTexts = teacherArtifacts.map((artifact) => artifact.content);
    const activeTaskSources = [
        ...offlineArtifacts.filter((artifact) => /active-tasks\.md|memory\/20\d{2}-\d{2}-\d{2}/u.test(artifact)),
        ...feedbackTexts,
        ...teacherArtifacts.filter((artifact) => artifact.freshness.status === "fresh").map((artifact) => artifact.content)
    ];
    const pointerSources = [
        ...offlineArtifacts.filter((artifact) => /(^|\/)memory\.md$|pointer|index/u.test(artifact)),
        ...offlineArtifacts.filter((artifact) => /(^|\/)(agents|soul|user|tools)\.md$/u.test(artifact)),
        workspace.snapshotId,
        workspace.workspaceId
    ];
    const entitySources = [
        workspace.workspaceId,
        workspace.snapshotId,
        workspace.revision ?? "",
        ...workspace.labels,
        ...offlineArtifacts,
        ...feedbackTexts,
        ...teacherTexts
    ];
    const correctionSources = [
        ...((eventExport?.feedbackEvents.filter((event) => event.kind === "correction" || event.kind === "teaching").map((event) => event.content)) ?? []),
        ...teacherArtifacts
            .filter((artifact) => artifact.kind === "correction" || artifact.kind === "teaching" || artifact.kind === "operator_override")
            .map((artifact) => artifact.content),
        ...offlineArtifacts.filter((artifact) => /correction|teaching|lesson|teach/u.test(artifact))
    ];
    const entityKeywords = topFrequencyKeywords(entitySources, 16);
    return {
        activeTaskKeywords: topFrequencyKeywords(activeTaskSources.length > 0 ? activeTaskSources : entitySources, 12),
        pointerKeywords: topFrequencyKeywords(pointerSources.length > 0 ? pointerSources : entitySources, 12),
        entityKeywords,
        correctionKeywords: topFrequencyKeywords(correctionSources.length > 0 ? correctionSources : entityKeywords, 12)
    };
}
function buildBlockInitSignals(block, metadata, context, builtAt) {
    const createdAt = metadata?.createdAt ?? builtAt;
    const fileRole = classifyInitFileRole(block.source, builtAt);
    const nodeType = inferInitNodeType(block.source, block.keywords, fileRole);
    const role = initRoleScore(nodeType, fileRole);
    const authority = initAuthorityScore(fileRole, block.source);
    const recency = initRecencyScore(createdAt, builtAt, fileRole);
    const activeTaskOverlap = roundMetric(Math.min(4, keywordOverlap(block.keywords, context.activeTaskKeywords)));
    const pointerCentrality = roundMetric(Math.min(4, (nodeType === "pointer" ? 2 : 0) + (fileRole === "pointer_index" ? 1.5 : 0) + keywordOverlap(block.keywords, context.pointerKeywords) * 0.5));
    const correctionDensity = correctionDensityScore(block, context.correctionKeywords);
    const entityOverlap = roundMetric(Math.min(4, keywordOverlap(topFocusKeywords(block), context.entityKeywords)));
    const staleness = initStalenessPenalty(createdAt, builtAt, fileRole);
    const score = roundMetric(clamp(role + authority + recency + activeTaskOverlap + pointerCentrality + correctionDensity + entityOverlap - staleness, 0, 24));
    const seededRouting = seededRoutingHintsForNodeType(nodeType, fileRole, score);
    return {
        mode: "heuristic_seed_v1",
        nodeType,
        fileRole,
        seededChannels: seededRouting.channels,
        score,
        scoreBreakdown: {
            role,
            authority,
            recency,
            activeTaskOverlap,
            pointerCentrality,
            correctionDensity,
            entityOverlap,
            staleness,
            total: score
        }
    };
}
function applyInitSignalsToBlock(block, metadata, context, builtAt) {
    const init = buildBlockInitSignals(block, metadata, context, builtAt);
    const seededRouting = seededRoutingHintsForNodeType(init.nodeType, init.fileRole, init.score);
    return {
        ...block,
        priority: Math.max(block.priority, Math.max(1, Math.ceil(init.score / 3.5))),
        keywords: uniqueKeywords([...block.keywords, "init_seed", init.nodeType, init.fileRole, ...seededRouting.channels]),
        initSeed: init,
        routing: mergeRoutingHints(block.routing, seededRouting)
    };
}
function offlineArtifactBlocks(packId, offlineArtifacts, builtAt) {
    return [...new Set(offlineArtifacts)]
        .filter((artifact) => artifact.length > 0)
        .map((artifact) => {
        const fileRole = classifyInitFileRole(artifact, builtAt);
        const keywords = uniqueKeywords([...keywordTokens(artifact), fileRole, inferInitNodeType(artifact, keywordTokens(artifact), fileRole), "init", "seed"]);
        const nodeType = inferInitNodeType(artifact, keywords, fileRole);
        const text = `Deterministic init seed from ${artifact} stays heuristic-only for boot route priors before PG-authoritative handoff.`;
        return {
            id: `${packId}:offline:${stableHash(artifact)}`,
            source: artifact,
            text,
            tokenCount: estimateTokenCount(text),
            keywords,
            priority: initPriorityForFileRole(fileRole),
            routing: seededRoutingHintsForNodeType(nodeType, fileRole, initPriorityForFileRole(fileRole) * 2),
            learning: learningSignals({
                role: learningRoleForInitFileRole(fileRole),
                decayHalfLifeDays: initDecayHalfLifeDays(fileRole),
                hebbianPulse: Math.max(1, initPriorityForFileRole(fileRole) - 1)
            })
        };
    });
}
function buildBackfillPrioritizationContext(current) {
    if (current === null) {
        return {
            frontierKeywords: [],
            sourceStreams: [],
            sessionIds: [],
            channels: [],
            observedAt: null
        };
    }
    const recentEvents = sortNormalizedEvents([...current.interactionEvents, ...current.feedbackEvents]).slice(-6);
    return {
        frontierKeywords: uniqueKeywords(recentEvents.flatMap((event) => keywordTokens(`${event.kind} ${event.channel} ${event.sessionId} ${event.source.stream} ${event.contract === CONTRACT_IDS.feedbackEvents ? event.content : event.messageId ?? ""}`))),
        sourceStreams: [...new Set(recentEvents.map((event) => event.source.stream))],
        sessionIds: [...new Set(recentEvents.map((event) => event.sessionId))],
        channels: [...new Set(recentEvents.map((event) => event.channel))],
        observedAt: recentEvents[recentEvents.length - 1]?.createdAt ?? current.range.lastCreatedAt
    };
}
function scoreBackfillRouteValue(slice, context) {
    const sliceEvents = [...slice.export.interactionEvents, ...slice.export.feedbackEvents];
    const sliceKeywords = uniqueKeywords(sliceEvents.flatMap((event) => keywordTokens(`${event.kind} ${event.channel} ${event.sessionId} ${event.source.stream} ${event.contract === CONTRACT_IDS.feedbackEvents ? event.content : event.messageId ?? ""}`)));
    const feedbackSignal = slice.export.feedbackEvents.reduce((sum, event) => {
        switch (event.kind) {
            case "correction":
                return sum + 5;
            case "teaching":
                return sum + 4;
            case "approval":
                return sum + 2;
            case "suppression":
            default:
                return sum + 1;
        }
    }, 0);
    const sourceOverlap = slice.export.provenance.sourceStreams.filter((stream) => context.sourceStreams.includes(stream)).length * 1.5;
    const sessionOverlap = sliceEvents.some((event) => context.sessionIds.includes(event.sessionId)) ? 1.5 : 0;
    const channelOverlap = sliceEvents.some((event) => context.channels.includes(event.channel)) ? 1 : 0;
    const frontierOverlap = context.frontierKeywords.length === 0 ? 0 : keywordOverlap(sliceKeywords, context.frontierKeywords) * 1.35;
    const mixedBonus = slice.export.feedbackEvents.length > 0 && slice.export.interactionEvents.length > 0 ? 1 : 0;
    const lastCreatedAt = slice.export.range.lastCreatedAt ?? slice.export.range.firstCreatedAt;
    const recency = context.observedAt === null || lastCreatedAt === null ? 0 : roundMetric(clamp(3 - daysBetween(lastCreatedAt, context.observedAt), 0, 3));
    return roundMetric(feedbackSignal + sourceOverlap + sessionOverlap + channelOverlap + frontierOverlap + mixedBonus + recency);
}
function sortPendingBackfillSlices(slices, current) {
    const context = buildBackfillPrioritizationContext(current);
    return [...slices].sort((left, right) => {
        const leftScore = scoreBackfillRouteValue(left, context);
        const rightScore = scoreBackfillRouteValue(right, context);
        if (rightScore !== leftScore) {
            return rightScore - leftScore;
        }
        return compareSliceRecency(left, right);
    });
}
function roundMetric(value) {
    return Math.round(value * 1_000) / 1_000;
}
function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}
function estimateTokenCount(text) {
    return Math.max(1, keywordTokens(text).length);
}
function uniqueKeywords(values) {
    return [...new Set(values)].slice(0, 16);
}
function genericKeyword(token) {
    return ["feedback", "interaction", "session", "openclaw", "runtime", "message", "memory", "pack"].includes(token);
}
function daysBetween(fromIso, toIso) {
    return Math.max(0, Date.parse(toIso) - Date.parse(fromIso)) / 86_400_000;
}
function compareIsoDates(left, right) {
    return Date.parse(left) - Date.parse(right);
}
function decayFreshness(createdAt, builtAt, halfLifeDays) {
    if (halfLifeDays === null) {
        return 1;
    }
    return roundMetric(clamp(Math.pow(0.5, daysBetween(createdAt, builtAt) / halfLifeDays), 0.05, 1));
}
function keywordOverlap(left, right) {
    const rightSet = new Set(right);
    return left.reduce((count, keyword) => count + (rightSet.has(keyword) ? 1 : 0), 0);
}
function cloneGraphBlock(block) {
    return structuredClone(block);
}
function addEdge(edgesById, fromId, edge) {
    const edges = edgesById.get(fromId);
    if (edges === undefined) {
        edgesById.set(fromId, [structuredClone(edge)]);
        return true;
    }
    const existing = edges.find((candidate) => candidate.kind === edge.kind && candidate.targetBlockId === edge.targetBlockId);
    if (existing !== undefined) {
        existing.weight = Math.max(existing.weight, edge.weight);
        return false;
    }
    edges.push(structuredClone(edge));
    return true;
}
function topFocusKeywords(block) {
    const focused = block.keywords.filter((keyword) => !genericKeyword(keyword));
    return (focused.length > 0 ? focused : block.keywords).slice(0, 4);
}
function splitBlockText(parent) {
    const focusKeywords = topFocusKeywords(parent);
    const focus = focusKeywords.length === 0 ? "learned memory" : focusKeywords.join(", ");
    const content = parent.text.split(/(?<=[.!?])/u).map((part) => part.trim()).find((part) => part.length >= 24) ?? parent.text;
    return `Focused memory on ${focus}: ${content}`;
}
function mergeBlockText(left, right) {
    return `Merged memory path: ${left.text} ${right.text}`;
}
function buildBlockMetadata(packId, workspace, builtAt, blocks, eventExport, teacherArtifacts, workspaceInitGraphSeed) {
    const metadata = new Map();
    const eventsByBlockId = new Map();
    const teacherByBlockId = new Map();
    const teacherSummaryCreatedAt = teacherArtifacts.reduce((latest, artifact) => {
        if (latest === null || compareIsoDates(artifact.createdAt, latest) > 0) {
            return artifact.createdAt;
        }
        return latest;
    }, null);
    for (const event of [...(eventExport?.interactionEvents ?? []), ...(eventExport?.feedbackEvents ?? [])]) {
        eventsByBlockId.set(`${packId}:event:${event.eventId}`, event);
    }
    for (const artifact of teacherArtifacts) {
        teacherByBlockId.set(`${packId}:teacher:${artifact.artifactId}`, artifact);
    }
    for (const [blockId, blockMeta] of workspaceInitGraphSeed.metadata) {
        metadata.set(blockId, { ...blockMeta });
    }
    for (const block of blocks) {
        const event = eventsByBlockId.get(block.id);
        const teacherArtifact = teacherByBlockId.get(block.id);
        const seedMeta = metadata.get(block.id);
        const datedSource = datedSourceTimestamp(block.source);
        const createdAt = event?.createdAt ??
            teacherArtifact?.createdAt ??
            seedMeta?.createdAt ??
            (block.id === `${packId}:teacher-supervision-summary` ? teacherSummaryCreatedAt : null) ??
            datedSource ??
            workspace.capturedAt ??
            builtAt;
        const nextMeta = {
            createdAt,
            sourceStream: event?.source.stream ?? teacherArtifact?.source.sourceStreams.join("+") ?? seedMeta?.sourceStream ?? block.source,
            syntheticRole: seedMeta?.syntheticRole ?? "base",
            splitDepth: seedMeta?.splitDepth ?? 0
        };
        if (event !== undefined) {
            nextMeta.sessionId = event.sessionId;
            nextMeta.channel = event.channel;
            if (event.contract === CONTRACT_IDS.feedbackEvents && event.relatedInteractionId !== undefined) {
                nextMeta.relatedInteractionId = event.relatedInteractionId;
            }
        }
        else if (teacherArtifact !== undefined) {
            nextMeta.sessionId = teacherArtifact.source.sessionId;
            nextMeta.channel = teacherArtifact.source.channel;
            if (teacherArtifact.relatedInteractionId !== null && teacherArtifact.relatedInteractionId !== undefined) {
                nextMeta.relatedInteractionId = teacherArtifact.relatedInteractionId;
            }
        }
        else {
            if (seedMeta?.sessionId !== undefined) {
                nextMeta.sessionId = seedMeta.sessionId;
            }
            if (seedMeta?.channel !== undefined) {
                nextMeta.channel = seedMeta.channel;
            }
            if (seedMeta?.relatedInteractionId !== undefined) {
                nextMeta.relatedInteractionId = seedMeta.relatedInteractionId;
            }
        }
        metadata.set(block.id, nextMeta);
    }
    return metadata;
}
function keywordTokens(value) {
    return [...new Set(value.toLowerCase().split(/[^a-z0-9]+/u).filter((token) => token.length >= 3 && /[a-z]/u.test(token)))].slice(0, 16);
}
function routingHints(channels, overrides = {}) {
    return {
        channels: [...new Set(channels)],
        ...overrides
    };
}
function mergeRoutingHints(...values) {
    const channels = [];
    let graphBias = 0;
    let shortTermBias = 0;
    let vectorBias = 0;
    let backgroundLabelAmplification = 1;
    for (const value of values) {
        if (value === undefined) {
            continue;
        }
        channels.push(...value.channels);
        graphBias = Math.max(graphBias, value.graphBias ?? 0);
        shortTermBias = Math.max(shortTermBias, value.shortTermBias ?? 0);
        vectorBias = Math.max(vectorBias, value.vectorBias ?? 0);
        backgroundLabelAmplification = Math.max(backgroundLabelAmplification, value.backgroundLabelAmplification ?? 1);
    }
    return routingHints(channels.length > 0 ? channels : ["vector"], {
        ...(graphBias > 0 ? { graphBias } : {}),
        ...(shortTermBias > 0 ? { shortTermBias } : {}),
        ...(vectorBias > 0 ? { vectorBias } : {}),
        ...(backgroundLabelAmplification > 1 ? { backgroundLabelAmplification } : {})
    });
}
function isImplicitPositiveApproval(event) {
    return event.kind === "approval" && event.source.stream.endsWith(IMPLICIT_POSITIVE_SOURCE_SUFFIX);
}
function isImplicitPositiveTeacherApproval(artifact) {
    return artifact.kind === "approval" && artifact.source.sourceStreams.some((stream) => stream.endsWith(IMPLICIT_POSITIVE_SOURCE_SUFFIX));
}
function feedbackPriority(event) {
    switch (event.kind) {
        case "correction":
            return 4;
        case "teaching":
            return 3;
        case "approval":
            return isImplicitPositiveApproval(event) ? 0.5 : 2;
        case "suppression":
        default:
            return 1;
    }
}
function sparseFeedbackMaskAllows(policy, kind) {
    return policy.feedbackMask[kind];
}
function evaluateSparseFeedback(feedbackEvents, observedAt, sparseFeedback) {
    const policy = normalizeSparseFeedbackPolicy(sparseFeedback);
    const eligible = [];
    let maskedFeedbackCount = 0;
    let delayedFeedbackCount = 0;
    for (const feedbackEvent of feedbackEvents) {
        if (!sparseFeedbackMaskAllows(policy, feedbackEvent.kind)) {
            maskedFeedbackCount += 1;
            continue;
        }
        const ageMs = Math.max(0, Date.parse(observedAt) - Date.parse(feedbackEvent.createdAt));
        if (ageMs < policy.teacherDelayMs) {
            delayedFeedbackCount += 1;
            continue;
        }
        eligible.push(feedbackEvent);
    }
    eligible.sort((left, right) => {
        const priorityDelta = feedbackPriority(right) - feedbackPriority(left);
        if (priorityDelta !== 0) {
            return priorityDelta;
        }
        const createdAtDelta = Date.parse(right.createdAt) - Date.parse(left.createdAt);
        if (createdAtDelta !== 0) {
            return createdAtDelta;
        }
        return right.sequence - left.sequence;
    });
    const selected = eligible.slice(0, policy.teacherBudget);
    const selectedEventIds = new Set(selected.map((event) => event.eventId));
    const amplifiedBackgroundLabelCount = Math.max(0, Math.round(selected.filter((event) => event.kind !== "suppression").length * Math.max(0, policy.backgroundLabelAmplification - 1)));
    return {
        selectedEventIds,
        diagnostics: createSparseFeedbackRuntimeDiagnostics(policy, {
            eligibleFeedbackCount: eligible.length,
            maskedFeedbackCount,
            delayedFeedbackCount,
            budgetedOutFeedbackCount: Math.max(0, eligible.length - selected.length),
            amplifiedBackgroundLabelCount
        })
    };
}
export function describeSparseFeedbackEventDispositions(feedbackEvents, observedAt, sparseFeedback) {
    const policy = normalizeSparseFeedbackPolicy(sparseFeedback);
    const eligible = [];
    const reasonByEventId = new Map();
    for (const feedbackEvent of feedbackEvents) {
        if (!sparseFeedbackMaskAllows(policy, feedbackEvent.kind)) {
            reasonByEventId.set(feedbackEvent.eventId, "masked");
            continue;
        }
        const ageMs = Math.max(0, Date.parse(observedAt) - Date.parse(feedbackEvent.createdAt));
        if (ageMs < policy.teacherDelayMs) {
            reasonByEventId.set(feedbackEvent.eventId, "delayed");
            continue;
        }
        eligible.push(feedbackEvent);
    }
    eligible.sort((left, right) => {
        const priorityDelta = feedbackPriority(right) - feedbackPriority(left);
        if (priorityDelta !== 0) {
            return priorityDelta;
        }
        const createdAtDelta = Date.parse(right.createdAt) - Date.parse(left.createdAt);
        if (createdAtDelta !== 0) {
            return createdAtDelta;
        }
        return right.sequence - left.sequence;
    });
    const selectedEventIds = new Set(eligible.slice(0, policy.teacherBudget).map((event) => event.eventId));
    for (const event of eligible) {
        if (!selectedEventIds.has(event.eventId)) {
            reasonByEventId.set(event.eventId, "budgeted_out");
        }
    }
    return feedbackEvents.map((event) => ({
        eventId: event.eventId,
        selected: selectedEventIds.has(event.eventId),
        reason: selectedEventIds.has(event.eventId) ? null : reasonByEventId.get(event.eventId) ?? null
    }));
}
function eventPriority(event) {
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
        switch (event.kind) {
            case "correction":
            case "teaching":
                return 5;
            case "approval":
                return isImplicitPositiveApproval(event) ? 3 : 4;
            case "suppression":
                return 4;
            default:
                return 4;
        }
    }
    switch (event.kind) {
        case "operator_override":
            return 5;
        case "memory_compiled":
            return 4;
        case "message_delivered":
            return 3;
        default:
            return 3;
    }
}
function sentenceCase(value) {
    return value.length === 0 ? value : `${value[0]?.toUpperCase() ?? ""}${value.slice(1)}`;
}
function learningSignals(input) {
    return {
        role: input.role,
        humanLabels: input.humanLabels ?? 0,
        selfLabels: input.selfLabels ?? 0,
        decayHalfLifeDays: input.decayHalfLifeDays ?? null,
        hebbianPulse: input.hebbianPulse ?? 0
    };
}
function summarizeEvent(event) {
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
        const relation = event.relatedInteractionId === undefined ? "" : ` Related interaction: ${event.relatedInteractionId}.`;
        return `${sentenceCase(event.kind)} feedback on ${event.channel} session ${event.sessionId}: ${event.content}.${relation}`;
    }
    const messagePart = event.messageId === undefined ? "" : ` Message: ${event.messageId}.`;
    const packPart = event.packId === undefined ? "" : ` Pack: ${event.packId}.`;
    return `Interaction ${event.kind} on ${event.channel} session ${event.sessionId}.${packPart}${messagePart}`;
}
function eventLearningSignals(event) {
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
        return learningSignals({
            role: "feedback",
            humanLabels: 1,
            decayHalfLifeDays: 30,
            hebbianPulse: eventPriority(event)
        });
    }
    if (event.kind === "operator_override") {
        return learningSignals({
            role: "interaction",
            humanLabels: 1,
            decayHalfLifeDays: 30,
            hebbianPulse: eventPriority(event)
        });
    }
    if (event.kind === "memory_compiled") {
        return learningSignals({
            role: "interaction",
            selfLabels: 1,
            decayHalfLifeDays: 30,
            hebbianPulse: eventPriority(event)
        });
    }
    return learningSignals({
        role: "interaction",
        decayHalfLifeDays: 14,
        hebbianPulse: 1
    });
}
function eventBlock(packId, event) {
    const text = summarizeEvent(event);
    const source = `${event.source.stream}:${event.kind}`;
    return {
        id: `${packId}:event:${event.eventId}`,
        source,
        text,
        keywords: keywordTokens(`${event.kind} ${event.channel} ${event.source.stream} ${text}`),
        priority: eventPriority(event),
        routing: routingHints(event.contract === CONTRACT_IDS.feedbackEvents ? ["short_term", "vector"] : ["short_term"], event.contract === CONTRACT_IDS.feedbackEvents
            ? {
                shortTermBias: 2,
                vectorBias: 1
            }
            : {
                shortTermBias: 2
            }),
        learning: eventLearningSignals(event),
        ...(event.semantic === undefined ? {} : { semantic: event.semantic }),
        init: createOpenClawInitBlockMetadata({
            nodeKind: "event",
            sourceKind: "event",
            source
        })
    };
}
function teacherSupervisionPriority(artifact) {
    const freshnessBoost = artifact.freshness.status === "fresh" ? 1 : 0;
    const principalBoost = teacherAuthorityRank(artifact) + principalPriorityRank(artifact);
    switch (artifact.kind) {
        case "correction":
        case "teaching":
        case "operator_override":
            return 5 + freshnessBoost + principalBoost;
        case "approval":
            return (isImplicitPositiveTeacherApproval(artifact) ? 3 : 4) + freshnessBoost + principalBoost;
        case "suppression":
        default:
            return 4 + freshnessBoost + principalBoost;
    }
}
function teacherSupervisionLearningSignals(artifact) {
    return learningSignals({
        role: "teacher_supervision",
        humanLabels: 1,
        decayHalfLifeDays: artifact.freshness.status === "fresh" ? 30 : 14,
        hebbianPulse: teacherSupervisionPriority(artifact)
    });
}
function summarizeTeacherSupervisionArtifact(artifact) {
    const related = artifact.relatedInteractionId === null ? "" : ` Related interaction: ${artifact.relatedInteractionId}.`;
    const principal = artifact.principal === undefined
        ? ""
        : ` Principal ${artifact.principal.teacherRole}/${artifact.principal.teacherAuthority}/${artifact.principal.priorityClass} from ${artifact.principal.teacherIdentity}.${(artifact.principal.supersedes?.length ?? 0) > 0 ? ` Supersedes: ${artifact.principal.supersedes?.join(", ")}.` : ""}`;
    return `Teacher ${artifact.kind} from ${artifact.source.channel} session ${artifact.source.sessionId} (${artifact.freshness.status}, ageMs=${artifact.freshness.ageMs}) observed at ${artifact.freshness.observedAt}: ${artifact.content}.${principal}${related}`;
}
function teacherSupervisionBlocks(packId, artifacts, principalBacklog) {
    if (artifacts.length === 0) {
        return [];
    }
    const freshCount = artifacts.filter((artifact) => artifact.freshness.status === "fresh").length;
    const staleCount = artifacts.length - freshCount;
    return [
        {
            id: `${packId}:teacher-supervision-summary`,
            source: `contracts/${CONTRACT_IDS.teacherSupervisionArtifact}`,
            text: `Teacher supervision artifacts stay canonical with ${artifacts.length} deduplicated records (fresh=${freshCount}, stale=${staleCount}) flowing into future candidate packs.`,
            keywords: ["teacher", "supervision", "dedup", "fresh", "stale", "candidate", "pack"],
            priority: 5,
            routing: routingHints(["short_term", "vector"], {
                shortTermBias: 2,
                vectorBias: 1
            }),
            learning: learningSignals({
                role: "teacher_supervision",
                humanLabels: artifacts.length,
                decayHalfLifeDays: 30,
                hebbianPulse: 5
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "teacher_supervision",
                sourceKind: "teacher_supervision",
                source: `contracts/${CONTRACT_IDS.teacherSupervisionArtifact}`
            })
        },
        ...artifacts.map((artifact) => {
            const text = summarizeTeacherSupervisionArtifact(artifact);
            const source = `${artifact.source.sourceStreams.join("+")}:${artifact.kind}`;
            const backlogBoost = principalBacklogRoutingBoost(artifact.principal, principalBacklog);
            return {
                id: `${packId}:teacher:${artifact.artifactId}`,
                source,
                text,
                keywords: keywordTokens(`${artifact.kind} ${artifact.source.channel} ${artifact.source.sessionId} ${artifact.freshness.status} ${artifact.source.sourceStreams.join(" ")} ${text}`),
                priority: roundMetric(teacherSupervisionPriority(artifact) + backlogBoost.priority),
                routing: routingHints(["short_term", "vector"], {
                    shortTermBias: roundMetric((artifact.freshness.status === "fresh" ? 2 : 1) + backlogBoost.shortTermBias),
                    vectorBias: roundMetric(1 + backlogBoost.vectorBias)
                }),
                learning: teacherSupervisionLearningSignals(artifact),
                init: createOpenClawInitBlockMetadata({
                    nodeKind: "teacher_supervision",
                    sourceKind: "teacher_supervision",
                    source
                })
            };
        })
    ];
}
function staticLifecycleBlocks(packId, input, workspace, learningSurface) {
    const structuralOps = structuralOpsSummary(input);
    const humanSources = learningSurface.labelSources.human.join(", ") || "feedback_events.v1";
    const selfSources = learningSurface.labelSources.self.join(", ") || "interaction_events.v1:memory_compiled";
    const sparseFeedback = evaluateSparseFeedback(input.eventExports?.feedbackEvents ?? [], input.builtAt ?? workspace.capturedAt, input.sparseFeedback).diagnostics;
    const amplifiedBackgroundLabels = sparseFeedback.amplifiedBackgroundLabelCount;
    return [
        {
            id: `${packId}:feedback-scanner`,
            source: "docs/openclaw-attach-quickstart.md",
            text: "Always-on feedback scanner harvests human labels from local session logs with Ollama qwen3.5:9b-q4_K_M, checkpointed resumes, and deduplicated background scans.",
            keywords: ["feedback", "scanner", "always", "background", "labels", "ollama", "qwen", "checkpoint", "dedup"],
            priority: 5,
            routing: routingHints(["short_term", "vector"], {
                shortTermBias: 2,
                vectorBias: 1,
                backgroundLabelAmplification: sparseFeedback.backgroundLabelAmplification
            }),
            learning: learningSignals({
                role: "label_surface",
                humanLabels: learningSurface.labelHarvest.humanLabels,
                decayHalfLifeDays: 30,
                hebbianPulse: 5
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "markdown_ontology",
                sourceKind: "markdown",
                source: "docs/openclaw-attach-quickstart.md"
            })
        },
        {
            id: `${packId}:fast-boot-defaults`,
            source: "product/always-on-learning",
            text: "Fast boot defaults stay live at startup so OpenClaw can answer immediately while passive background learning hydrates richer graph state.",
            keywords: ["fast", "boot", "defaults", "startup", "background", "learning", "openclaw"],
            priority: 5,
            routing: routingHints(["vector"], {
                vectorBias: 2
            }),
            learning: learningSignals({
                role: "boot_default",
                decayHalfLifeDays: null,
                hebbianPulse: 2
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "product_invariant",
                sourceKind: "product",
                source: "product/always-on-learning"
            })
        },
        {
            id: `${packId}:passive-background-learning`,
            source: "product/always-on-learning",
            text: `Learning cadence is ${learningSurface.learningCadence} with ${learningSurface.scanPolicy} scans across ${learningSurface.scanSurfaces.join(", ")}; sparse teacher gating budget=${sparseFeedback.teacherBudget}, delayMs=${sparseFeedback.teacherDelayMs}, background_amplification=${sparseFeedback.backgroundLabelAmplification}, amplified_labels=${amplifiedBackgroundLabels}.`,
            keywords: [
                "passive",
                "background",
                "always",
                "scan",
                "learning",
                "sparse",
                "teacher",
                "budget",
                "delay",
                "amplification",
                ...keywordTokens(learningSurface.scanSurfaces.join(" "))
            ],
            priority: 5,
            routing: routingHints(["graph", "vector"], {
                graphBias: 1,
                vectorBias: 2,
                backgroundLabelAmplification: sparseFeedback.backgroundLabelAmplification
            }),
            learning: learningSignals({
                role: "background_expectation",
                humanLabels: amplifiedBackgroundLabels,
                decayHalfLifeDays: 30,
                hebbianPulse: 3 + amplifiedBackgroundLabels
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "product_invariant",
                sourceKind: "product",
                source: "product/always-on-learning"
            })
        },
        {
            id: `${packId}:human-label-harvest`,
            source: humanSources,
            text: `Human label harvest is first-class with ${learningSurface.labelHarvest.humanLabels} labels sourced from ${humanSources}.`,
            keywords: ["human", "labels", "harvest", "feedback", "approval", "teaching", "correction", "suppression"],
            priority: 4,
            routing: routingHints(["short_term", "vector"], {
                shortTermBias: 1,
                vectorBias: 1,
                backgroundLabelAmplification: sparseFeedback.backgroundLabelAmplification
            }),
            learning: learningSignals({
                role: "label_surface",
                humanLabels: learningSurface.labelHarvest.humanLabels + amplifiedBackgroundLabels,
                decayHalfLifeDays: 30,
                hebbianPulse: Math.max(1, learningSurface.labelHarvest.humanLabels + amplifiedBackgroundLabels)
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "event_export",
                sourceKind: "event_export",
                source: humanSources
            })
        },
        {
            id: `${packId}:self-label-harvest`,
            source: selfSources,
            text: `Self label harvest stays visible with ${learningSurface.labelHarvest.selfLabels} memory-side labels sourced from ${selfSources}.`,
            keywords: ["self", "labels", "harvest", "memory", "compiled", "graph"],
            priority: 4,
            routing: routingHints(["short_term", "vector"], {
                shortTermBias: 1,
                vectorBias: 1
            }),
            learning: learningSignals({
                role: "label_surface",
                selfLabels: learningSurface.labelHarvest.selfLabels,
                decayHalfLifeDays: 30,
                hebbianPulse: Math.max(1, learningSurface.labelHarvest.selfLabels)
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "event_export",
                sourceKind: "event_export",
                source: selfSources
            })
        },
        {
            id: `${packId}:workspace`,
            source: workspace.rootDir,
            text: `Workspace snapshot ${workspace.snapshotId} for ${workspace.workspaceId} captured at ${workspace.capturedAt} with revision ${workspace.revision ?? "unversioned"}.`,
            keywords: keywordTokens(`workspace ${workspace.workspaceId} ${workspace.snapshotId} ${workspace.branch ?? ""} ${workspace.revision ?? ""} ${workspace.labels.join(" ")}`),
            priority: 4,
            routing: routingHints(["vector"], {
                vectorBias: 1
            }),
            learning: learningSignals({
                role: "workspace",
                decayHalfLifeDays: null,
                hebbianPulse: 1
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "workspace_snapshot",
                sourceKind: "workspace",
                source: workspace.rootDir
            })
        },
        {
            id: `${packId}:structural-ops`,
            source: "docs/internal/learning-first-convergence.md",
            text: `Structural graph learning stays first-class with Hebbian reinforcement, decay half-life 30 days, and ops split=${structuralOps.split}, merge=${structuralOps.merge}, prune=${structuralOps.prune}, connect=${structuralOps.connect}.`,
            keywords: ["structural", "hebbian", "decay", "split", "merge", "prune", "connect", "graph", "memory"],
            priority: 4,
            routing: routingHints(["graph", "vector"], {
                graphBias: 2,
                vectorBias: 1
            }),
            learning: learningSignals({
                role: "structural",
                decayHalfLifeDays: 30,
                hebbianPulse: 4
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "markdown_ontology",
                sourceKind: "markdown",
                source: "docs/internal/learning-first-convergence.md"
            })
        }
    ];
}
function eventExportBlocks(packId, eventExport) {
    const allEvents = [...eventExport.interactionEvents, ...eventExport.feedbackEvents];
    const learningSurface = eventExport.provenance.learningSurface;
    const summaryKeywords = keywordTokens(`normalized event export ${eventExport.provenance.sourceStreams.join(" ")} interaction ${eventExport.provenance.interactionCount} feedback ${eventExport.provenance.feedbackCount} human ${learningSurface.labelHarvest.humanLabels} self ${learningSurface.labelHarvest.selfLabels}`);
    const source = `contracts/${CONTRACT_IDS.interactionEvents}+${CONTRACT_IDS.feedbackEvents}`;
    return [
        {
            id: `${packId}:event-export`,
            source,
            text: `Normalized event export covers ${eventExport.provenance.interactionCount} interaction events and ${eventExport.provenance.feedbackCount} feedback events across sequences ${eventExport.range.start}-${eventExport.range.end}; harvested labels human=${learningSurface.labelHarvest.humanLabels}, self=${learningSurface.labelHarvest.selfLabels}.`,
            keywords: summaryKeywords,
            priority: 5,
            routing: routingHints(["short_term", "vector"], {
                shortTermBias: 2,
                vectorBias: 1
            }),
            learning: learningSignals({
                role: "label_surface",
                humanLabels: learningSurface.labelHarvest.humanLabels,
                selfLabels: learningSurface.labelHarvest.selfLabels,
                decayHalfLifeDays: 30,
                hebbianPulse: 5
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "event_export",
                sourceKind: "event_export",
                source
            })
        },
        ...allEvents.map((event) => eventBlock(packId, event))
    ];
}
function addFeedbackEdges(edgesById, packId, eventExport) {
    if (eventExport === null) {
        return;
    }
    for (const event of eventExport.feedbackEvents) {
        if (event.relatedInteractionId === undefined) {
            continue;
        }
        const feedbackBlockId = `${packId}:event:${event.eventId}`;
        const interactionBlockId = `${packId}:event:${event.relatedInteractionId}`;
        addEdge(edgesById, feedbackBlockId, {
            targetBlockId: interactionBlockId,
            kind: "feedback",
            weight: Math.max(2, eventPriority(event))
        });
        addEdge(edgesById, interactionBlockId, {
            targetBlockId: feedbackBlockId,
            kind: "feedback",
            weight: Math.max(2, eventPriority(event) - 1)
        });
    }
}
function connectPairs(blocks, metadataById, edgesById, connectLimit) {
    const candidates = [];
    for (let index = 0; index < blocks.length; index += 1) {
        const left = blocks[index];
        if (left === undefined) {
            continue;
        }
        for (let peerIndex = index + 1; peerIndex < blocks.length; peerIndex += 1) {
            const right = blocks[peerIndex];
            if (right === undefined) {
                continue;
            }
            const leftMeta = metadataById.get(left.id);
            const rightMeta = metadataById.get(right.id);
            const overlap = keywordOverlap(left.keywords, right.keywords);
            const sameStream = leftMeta?.sourceStream !== undefined && leftMeta.sourceStream === rightMeta?.sourceStream ? 2 : 0;
            const sameSession = leftMeta?.sessionId !== undefined && leftMeta.sessionId === rightMeta?.sessionId ? 1 : 0;
            const organicSignal = overlap > 0 || sameSession > 0;
            const score = overlap + sameStream + sameSession;
            if (!organicSignal || score < CONNECT_PAIR_SCORE_THRESHOLD) {
                continue;
            }
            candidates.push({
                leftId: left.id,
                rightId: right.id,
                score
            });
        }
    }
    candidates.sort((left, right) => {
        if (right.score !== left.score) {
            return right.score - left.score;
        }
        if (left.leftId !== right.leftId) {
            return left.leftId.localeCompare(right.leftId);
        }
        return left.rightId.localeCompare(right.rightId);
    });
    let appliedPairCount = 0;
    let createdEdgeCount = 0;
    for (const candidate of candidates) {
        if (appliedPairCount >= connectLimit) {
            break;
        }
        const weight = Math.max(2, candidate.score);
        const createdLeft = addEdge(edgesById, candidate.leftId, {
            targetBlockId: candidate.rightId,
            kind: "connect",
            weight
        });
        const createdRight = addEdge(edgesById, candidate.rightId, {
            targetBlockId: candidate.leftId,
            kind: "connect",
            weight
        });
        if (createdLeft || createdRight) {
            appliedPairCount += 1;
            createdEdgeCount += (createdLeft ? 1 : 0) + (createdRight ? 1 : 0);
        }
    }
    return {
        appliedPairCount,
        candidatePairCount: candidates.length,
        createdEdgeCount
    };
}
function splitBlocks(packId, blocks, metadataById, edgesById, splitLimit) {
    const candidates = blocks
        .filter((block) => block.id.includes(":event:") &&
        (block.learning.humanLabels > 0 ||
            block.learning.hebbianPulse >= 4 ||
            block.source.includes(":teaching") ||
            block.source.includes(":correction")))
        .sort((left, right) => {
        const leftScore = left.learning.hebbianPulse + left.learning.humanLabels + left.priority;
        const rightScore = right.learning.hebbianPulse + right.learning.humanLabels + right.priority;
        if (rightScore !== leftScore) {
            return rightScore - leftScore;
        }
        return left.id.localeCompare(right.id);
    });
    let applied = 0;
    for (const parent of candidates) {
        if (applied >= splitLimit) {
            break;
        }
        const meta = metadataById.get(parent.id);
        if (meta === undefined) {
            continue;
        }
        const blockId = `${parent.id}:split:${applied + 1}`;
        const text = splitBlockText(parent);
        const splitBlock = {
            id: blockId,
            source: `split:${parent.source}`,
            text,
            tokenCount: estimateTokenCount(text),
            compactedFrom: [parent.id],
            keywords: uniqueKeywords([...topFocusKeywords(parent), ...keywordTokens(text), "split", "focused"]),
            priority: parent.priority + 1,
            routing: mergeRoutingHints(parent.routing, routingHints(["graph"], { graphBias: 2, shortTermBias: 1 })),
            learning: learningSignals({
                role: parent.learning.role,
                humanLabels: parent.learning.humanLabels,
                selfLabels: parent.learning.selfLabels,
                decayHalfLifeDays: parent.learning.decayHalfLifeDays,
                hebbianPulse: parent.learning.hebbianPulse + 2
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "synthetic_topology",
                sourceKind: "synthetic",
                source: `split:${parent.source}`
            })
        };
        blocks.push(splitBlock);
        metadataById.set(blockId, {
            createdAt: meta.createdAt,
            sourceStream: meta.sourceStream,
            ...(meta.sessionId !== undefined ? { sessionId: meta.sessionId } : {}),
            ...(meta.channel !== undefined ? { channel: meta.channel } : {}),
            ...(meta.relatedInteractionId !== undefined ? { relatedInteractionId: meta.relatedInteractionId } : {}),
            syntheticRole: "split",
            splitDepth: meta.splitDepth + 1
        });
        edgesById.set(blockId, []);
        addEdge(edgesById, parent.id, { targetBlockId: blockId, kind: "split", weight: parent.learning.hebbianPulse + 1 });
        addEdge(edgesById, blockId, { targetBlockId: parent.id, kind: "merge", weight: Math.max(1, parent.priority - 1) });
        applied += 1;
    }
    return applied;
}
function mergeBlocks(packId, blocks, metadataById, edgesById, mergeLimit) {
    const candidates = [];
    for (let index = 0; index < blocks.length; index += 1) {
        const left = blocks[index];
        if (left === undefined) {
            continue;
        }
        if (left.compactedFrom !== undefined) {
            continue;
        }
        for (let peerIndex = index + 1; peerIndex < blocks.length; peerIndex += 1) {
            const right = blocks[peerIndex];
            if (right === undefined || right.compactedFrom !== undefined) {
                continue;
            }
            const leftMeta = metadataById.get(left.id);
            const rightMeta = metadataById.get(right.id);
            const overlap = keywordOverlap(left.keywords, right.keywords);
            const related = leftMeta?.relatedInteractionId === right.id.replace(`${packId}:event:`, "") || rightMeta?.relatedInteractionId === left.id.replace(`${packId}:event:`, "") ? 3 : 0;
            const score = overlap + related + (leftMeta?.sourceStream === rightMeta?.sourceStream ? 1 : 0);
            if (score < 3) {
                continue;
            }
            candidates.push({ leftId: left.id, rightId: right.id, score });
        }
    }
    candidates.sort((left, right) => {
        if (right.score !== left.score) {
            return right.score - left.score;
        }
        if (left.leftId !== right.leftId) {
            return left.leftId.localeCompare(right.leftId);
        }
        return left.rightId.localeCompare(right.rightId);
    });
    const used = new Set();
    let applied = 0;
    for (const candidate of candidates) {
        if (applied >= mergeLimit) {
            break;
        }
        if (used.has(candidate.leftId) || used.has(candidate.rightId)) {
            continue;
        }
        const left = blocks.find((block) => block.id === candidate.leftId);
        const right = blocks.find((block) => block.id === candidate.rightId);
        if (left === undefined || right === undefined) {
            continue;
        }
        const leftMeta = metadataById.get(left.id);
        const rightMeta = metadataById.get(right.id);
        if (leftMeta === undefined || rightMeta === undefined) {
            continue;
        }
        const blockId = `${packId}:merge:${applied + 1}`;
        const text = mergeBlockText(left, right);
        const mergedBlock = {
            id: blockId,
            source: `merge:${left.source}+${right.source}`,
            text,
            tokenCount: estimateTokenCount(text),
            compactedFrom: [left.id, right.id],
            keywords: uniqueKeywords([...left.keywords, ...right.keywords, "merge", "path", "connected"]),
            priority: Math.max(left.priority, right.priority) + 1,
            routing: mergeRoutingHints(left.routing, right.routing, routingHints(["graph"], { graphBias: 2 })),
            learning: learningSignals({
                role: left.learning.role === right.learning.role ? left.learning.role : "structural",
                humanLabels: left.learning.humanLabels + right.learning.humanLabels,
                selfLabels: left.learning.selfLabels + right.learning.selfLabels,
                decayHalfLifeDays: left.learning.decayHalfLifeDays ?? right.learning.decayHalfLifeDays,
                hebbianPulse: left.learning.hebbianPulse + right.learning.hebbianPulse
            }),
            init: createOpenClawInitBlockMetadata({
                nodeKind: "synthetic_topology",
                sourceKind: "synthetic",
                source: `merge:${left.source}+${right.source}`
            })
        };
        blocks.push(mergedBlock);
        metadataById.set(blockId, {
            createdAt: compareIsoDates(leftMeta.createdAt, rightMeta.createdAt) >= 0 ? leftMeta.createdAt : rightMeta.createdAt,
            sourceStream: `${leftMeta.sourceStream}|${rightMeta.sourceStream}`,
            ...(leftMeta.sessionId !== undefined ? { sessionId: leftMeta.sessionId } : rightMeta.sessionId !== undefined ? { sessionId: rightMeta.sessionId } : {}),
            ...(leftMeta.channel !== undefined ? { channel: leftMeta.channel } : rightMeta.channel !== undefined ? { channel: rightMeta.channel } : {}),
            syntheticRole: "merge",
            splitDepth: 0
        });
        edgesById.set(blockId, []);
        addEdge(edgesById, blockId, { targetBlockId: left.id, kind: "merge", weight: Math.max(2, candidate.score) });
        addEdge(edgesById, blockId, { targetBlockId: right.id, kind: "merge", weight: Math.max(2, candidate.score) });
        addEdge(edgesById, left.id, { targetBlockId: blockId, kind: "connect", weight: Math.max(2, candidate.score - 1) });
        addEdge(edgesById, right.id, { targetBlockId: blockId, kind: "connect", weight: Math.max(2, candidate.score - 1) });
        used.add(left.id);
        used.add(right.id);
        applied += 1;
    }
    return applied;
}
function assignGraphState(blocks, metadataById, edgesById, builtAt) {
    for (const block of blocks) {
        const metadata = metadataById.get(block.id);
        const freshness = decayFreshness(metadata?.createdAt ?? builtAt, builtAt, block.learning.decayHalfLifeDays);
        const edgeCount = edgesById.get(block.id)?.length ?? 0;
        const mergedFromCount = block.compactedFrom?.length ?? 0;
        const splitDepth = metadata?.splitDepth ?? 0;
        const hebbianGain = 1 + block.learning.humanLabels * 0.35 + block.learning.selfLabels * 0.2 + block.learning.hebbianPulse * 0.12;
        const structuralGain = 1 + mergedFromCount * 0.1 + splitDepth * 0.12 + edgeCount * 0.06;
        const initGain = 1 + (block.initSeed?.score ?? 0) * 0.04;
        const initTraversalBias = (block.initSeed?.scoreBreakdown.activeTaskOverlap ?? 0) * 0.4 +
            (block.initSeed?.scoreBreakdown.pointerCentrality ?? 0) * 0.45 +
            (block.initSeed?.scoreBreakdown.entityOverlap ?? 0) * 0.2;
        const evidenceCount = Math.max(1, block.learning.humanLabels + block.learning.selfLabels + mergedFromCount + (splitDepth > 0 ? 1 : 0) + (block.initSeed === undefined ? 0 : 1));
        const state = {
            strength: roundMetric(Math.max(0.25, block.priority * freshness * hebbianGain * structuralGain * initGain)),
            freshness,
            traversalBias: roundMetric(Math.max(0, freshness * 2 + edgeCount * 1.15 + mergedFromCount * 0.75 + splitDepth * 0.5 + initTraversalBias)),
            evidenceCount,
            splitDepth,
            mergedFromCount,
            pruned: false
        };
        block.state = state;
    }
}
function pruneBlocks(blocks, metadataById, eventExport, pruneLimit) {
    if (pruneLimit === 0) {
        return [];
    }
    const suppressedInteractionIds = new Set(eventExport?.feedbackEvents
        .filter((event) => event.kind === "suppression" && event.relatedInteractionId !== undefined)
        .map((event) => `${event.relatedInteractionId}`) ?? []);
    const candidates = blocks
        .filter((block) => {
        const meta = metadataById.get(block.id);
        const eventId = block.id.includes(":event:") ? block.id.split(":event:")[1] : undefined;
        const suppressed = eventId !== undefined && suppressedInteractionIds.has(eventId);
        const lowStrength = (block.state?.strength ?? block.priority) <= 3.5;
        const labelFree = block.learning.humanLabels === 0 && block.learning.selfLabels === 0;
        const eventLike = meta?.syntheticRole === "base" && block.learning.role === "interaction";
        return suppressed || (eventLike && labelFree && lowStrength);
    })
        .sort((left, right) => {
        const leftEventId = left.id.includes(":event:") ? left.id.split(":event:")[1] : undefined;
        const rightEventId = right.id.includes(":event:") ? right.id.split(":event:")[1] : undefined;
        const leftSuppressed = leftEventId !== undefined && suppressedInteractionIds.has(leftEventId);
        const rightSuppressed = rightEventId !== undefined && suppressedInteractionIds.has(rightEventId);
        if (leftSuppressed !== rightSuppressed) {
            return leftSuppressed ? -1 : 1;
        }
        const leftStrength = left.state?.strength ?? left.priority;
        const rightStrength = right.state?.strength ?? right.priority;
        if (leftStrength !== rightStrength) {
            return leftStrength - rightStrength;
        }
        return left.id.localeCompare(right.id);
    });
    return candidates.slice(0, pruneLimit).map((block) => block.id);
}
function applyGraphEvolution(packId, builtAt, blocks, metadataById, structuralOps, eventExport, seedEdges) {
    const workingBlocks = blocks.map((block) => cloneGraphBlock(block));
    const edgesById = new Map(workingBlocks.map((block) => [block.id, []]));
    for (const [blockId, blockEdges] of seedEdges) {
        for (const edge of blockEdges) {
            addEdge(edgesById, blockId, edge);
        }
    }
    addFeedbackEdges(edgesById, packId, eventExport);
    const appliedSplit = splitBlocks(packId, workingBlocks, metadataById, edgesById, structuralOps.split);
    const appliedMerge = mergeBlocks(packId, workingBlocks, metadataById, edgesById, structuralOps.merge);
    const connectResult = connectPairs(workingBlocks, metadataById, edgesById, structuralOps.connect);
    const connectDiagnostics = {
        requestedBudget: structuralOps.connect,
        scoreThreshold: CONNECT_PAIR_SCORE_THRESHOLD,
        candidatePairCount: connectResult.candidatePairCount,
        appliedPairCount: connectResult.appliedPairCount,
        createdEdgeCount: connectResult.createdEdgeCount
    };
    assignGraphState(workingBlocks, metadataById, edgesById, builtAt);
    const prunedBlockIds = pruneBlocks(workingBlocks, metadataById, eventExport, structuralOps.prune);
    const pruned = new Set(prunedBlockIds);
    const survivors = workingBlocks.filter((block) => !pruned.has(block.id));
    const survivorIds = new Set(survivors.map((block) => block.id));
    for (const block of survivors) {
        const edges = (edgesById.get(block.id) ?? [])
            .filter((edge) => survivorIds.has(edge.targetBlockId))
            .sort((left, right) => {
            if (right.weight !== left.weight) {
                return right.weight - left.weight;
            }
            if (left.kind !== right.kind) {
                return left.kind.localeCompare(right.kind);
            }
            return left.targetBlockId.localeCompare(right.targetBlockId);
        });
        if (edges.length > 0) {
            block.edges = edges;
        }
    }
    assignGraphState(survivors, metadataById, new Map(survivors.map((block) => [block.id, block.edges ?? []])), builtAt);
    survivors.sort((left, right) => {
        const leftStrength = left.state?.strength ?? left.priority;
        const rightStrength = right.state?.strength ?? right.priority;
        if (rightStrength !== leftStrength) {
            return rightStrength - leftStrength;
        }
        if (right.priority !== left.priority) {
            return right.priority - left.priority;
        }
        return left.id.localeCompare(right.id);
    });
    const strongestBlockId = survivors[0]?.id ?? null;
    return {
        blocks: survivors,
        evolution: {
            builtAt,
            hebbianApplied: true,
            decayApplied: true,
            structuralOps: {
                split: appliedSplit,
                merge: appliedMerge,
                prune: prunedBlockIds.length,
                connect: connectResult.appliedPairCount
            },
            connectDiagnostics,
            prunedBlockIds,
            strongestBlockId
        }
    };
}
function createGraphPayload(packId, input, workspace, eventExport, learningSurface, builtAt, workspaceInit) {
    const teacherSupervisionArtifacts = normalizeTeacherSupervisionArtifacts(input.teacherSupervisionArtifacts);
    const workspaceInitGraphSeed = buildWorkspaceInitGraphSeed(packId, workspace, builtAt, workspaceInit);
    const lifecycleBlocks = staticLifecycleBlocks(packId, input, workspace, learningSurface);
    const workspaceBlocks = workspaceInitGraphSeed.blocks;
    const seedArtifactPaths = Array.from(new Set([...(input.offlineArtifacts ?? []), ...workspaceInit.bootInputs, ...workspaceInit.workingSet, ...workspaceInit.passiveExpansion]));
    const artifactBlocks = offlineArtifactBlocks(packId, seedArtifactPaths, builtAt);
    const eventBlocks = eventExport === null ? [] : eventExportBlocks(packId, eventExport);
    const teacherBlocks = teacherSupervisionBlocks(packId, teacherSupervisionArtifacts, input.principalBacklog);
    const graphSeedBlocks = [...lifecycleBlocks, ...workspaceBlocks, ...artifactBlocks, ...eventBlocks, ...teacherBlocks];
    const metadataById = buildBlockMetadata(packId, workspace, builtAt, graphSeedBlocks, eventExport, teacherSupervisionArtifacts, workspaceInitGraphSeed);
    const initContext = buildInitSeedContext({
        ...input,
        offlineArtifacts: seedArtifactPaths
    }, workspace, eventExport, teacherSupervisionArtifacts);
    const seededBlocks = graphSeedBlocks.map((block) => applyInitSignalsToBlock(block, metadataById.get(block.id), initContext, builtAt));
    const evolved = applyGraphEvolution(packId, builtAt, seededBlocks, metadataById, structuralOpsSummary(input), eventExport, workspaceInitGraphSeed.seedEdges);
    return {
        packId,
        schema: PACK_GRAPH_SCHEMAS.openclawInit,
        ontology: createOpenClawInitGraphOntology(),
        blocks: evolved.blocks,
        evolution: evolved.evolution
    };
}
function learningVectorKeywords(block) {
    const keywords = [block.learning.role];
    for (const channel of block.routing?.channels ?? []) {
        keywords.push(channel === "short_term" ? "short_term" : channel);
    }
    if (block.init !== undefined) {
        keywords.push(block.init.nodeKind, block.init.sourceKind, block.init.heuristicScope, block.init.learnedLabelPolicy);
        if (block.init.fileRole !== undefined) {
            keywords.push(block.init.fileRole.role, `${block.init.fileRole.audience}_audience`, `${block.init.fileRole.tier}_tier`);
        }
    }
    if (block.learning.role === "boot_default") {
        keywords.push("fast_boot");
    }
    if (block.learning.role === "background_expectation") {
        keywords.push("passive_background", "always_on");
    }
    if (block.learning.role === "teacher_supervision") {
        keywords.push("teacher", "supervision");
    }
    if (block.initSeed !== undefined) {
        keywords.push("init_seed", block.initSeed.nodeType, block.initSeed.fileRole, ...block.initSeed.seededChannels.map((channel) => `seed_${channel}`));
    }
    if (block.learning.humanLabels > 0) {
        keywords.push("human_label");
    }
    if (block.learning.selfLabels > 0) {
        keywords.push("self_label");
    }
    if (block.learning.hebbianPulse > 0) {
        keywords.push("hebbian");
    }
    if (block.learning.decayHalfLifeDays !== null) {
        keywords.push("decay");
    }
    if ((block.compactedFrom?.length ?? 0) > 1) {
        keywords.push("merged");
    }
    if ((block.state?.splitDepth ?? 0) > 0) {
        keywords.push("split");
    }
    if ((block.edges?.length ?? 0) > 0) {
        keywords.push("connected");
    }
    if ((block.routing?.backgroundLabelAmplification ?? 1) > 1) {
        keywords.push("background_amplified");
    }
    if ((block.state?.strength ?? 0) >= 6) {
        keywords.push("reinforced");
    }
    if ((block.state?.freshness ?? 1) < 0.5) {
        keywords.push("decayed");
    }
    for (const edge of block.edges ?? []) {
        keywords.push(edge.kind);
    }
    return keywords;
}
function vectorEntryFromBlock(block) {
    const keywords = [...new Set([...block.keywords, ...learningVectorKeywords(block)])];
    const weights = Object.fromEntries(keywords.map((keyword, index) => [keyword, Math.max(1, block.priority - Math.min(index, Math.max(0, block.priority - 1)))]));
    if (block.learning.humanLabels > 0) {
        weights.human_label = Math.max(weights.human_label ?? 0, block.priority + block.learning.humanLabels);
    }
    if (block.learning.selfLabels > 0) {
        weights.self_label = Math.max(weights.self_label ?? 0, block.priority + block.learning.selfLabels);
    }
    if (block.learning.hebbianPulse > 0) {
        weights.hebbian = Math.max(weights.hebbian ?? 0, block.learning.hebbianPulse + 1);
    }
    if (block.learning.role === "boot_default") {
        weights.fast_boot = Math.max(weights.fast_boot ?? 0, block.priority + 1);
    }
    if (block.learning.role === "background_expectation") {
        weights.passive_background = Math.max(weights.passive_background ?? 0, block.priority + 1);
    }
    if (block.initSeed !== undefined) {
        weights.init_seed = Math.max(weights.init_seed ?? 0, Math.max(1, Math.ceil(block.initSeed.score / 3)));
        weights[block.initSeed.nodeType] = Math.max(weights[block.initSeed.nodeType] ?? 0, Math.max(1, Math.ceil(block.initSeed.score / 4)));
        weights[block.initSeed.fileRole] = Math.max(weights[block.initSeed.fileRole] ?? 0, Math.max(1, Math.ceil(block.initSeed.score / 4)));
        for (const channel of block.initSeed.seededChannels) {
            const seededKeyword = `seed_${channel}`;
            weights[seededKeyword] = Math.max(weights[seededKeyword] ?? 0, Math.max(1, Math.ceil(block.initSeed.score / 5)));
        }
    }
    if ((block.state?.strength ?? 0) > 0) {
        weights.reinforced = Math.max(weights.reinforced ?? 0, Math.ceil(block.state?.strength ?? 0));
    }
    if ((block.state?.freshness ?? 1) < 1) {
        weights.decayed = Math.max(weights.decayed ?? 0, Math.ceil((1 - (block.state?.freshness ?? 1)) * 4));
    }
    if ((block.edges?.length ?? 0) > 0) {
        weights.connected = Math.max(weights.connected ?? 0, block.edges?.length ?? 0);
        for (const edge of block.edges ?? []) {
            weights[edge.kind] = Math.max(weights[edge.kind] ?? 0, Math.ceil(edge.weight));
        }
    }
    return {
        blockId: block.id,
        keywords,
        boost: Math.max(1, Math.ceil(block.priority / 2)) +
            Math.min(3, block.learning.humanLabels + block.learning.selfLabels) +
            Math.min(2, Math.ceil(block.learning.hebbianPulse / 3)) +
            Math.min(3, Math.ceil((block.state?.traversalBias ?? 0) / 3)) +
            Math.min(2, block.edges?.length ?? 0) -
            ((block.state?.freshness ?? 1) < 0.35 ? 1 : 0),
        weights
    };
}
function createVectorsPayload(graph) {
    return {
        packId: graph.packId,
        entries: graph.blocks.map((block) => vectorEntryFromBlock(block))
    };
}
function embeddingTextForBlock(block) {
    return [block.source, block.text, ...block.keywords].filter((candidate) => candidate.length > 0).join("\n\n");
}
function isFiniteEmbeddingResult(value) {
    return value !== undefined && value.model.length > 0 && value.values.length > 0 && value.values.every((candidate) => Number.isFinite(candidate));
}
async function collectEmbeddingsWithRetry(targets, embedder, embeddingsByBlockId) {
    if (targets.length === 0) {
        return;
    }
    try {
        const embeddings = await embedder.embed(targets.map((target) => target.text));
        if (embeddings.length !== targets.length) {
            throw new Error("embedding batch length mismatch");
        }
        for (const [index, target] of targets.entries()) {
            const embedding = embeddings[index];
            if (!isFiniteEmbeddingResult(embedding)) {
                continue;
            }
            embeddingsByBlockId.set(target.blockId, {
                model: embedding.model,
                values: [...embedding.values]
            });
        }
        return;
    }
    catch {
        if (targets.length === 1) {
            return;
        }
    }
    const midpoint = Math.ceil(targets.length / 2);
    await collectEmbeddingsWithRetry(targets.slice(0, midpoint), embedder, embeddingsByBlockId);
    await collectEmbeddingsWithRetry(targets.slice(midpoint), embedder, embeddingsByBlockId);
}
async function enrichVectorsPayloadWithEmbeddings(graph, vectors, embedder) {
    const embeddingsByBlockId = new Map();
    try {
        await collectEmbeddingsWithRetry(graph.blocks.map((block) => ({
            blockId: block.id,
            text: embeddingTextForBlock(block)
        })), embedder, embeddingsByBlockId);
    }
    catch {
        return vectors;
    }
    if (embeddingsByBlockId.size === 0) {
        return vectors;
    }
    return {
        ...vectors,
        entries: vectors.entries.map((entry) => {
            const embedding = embeddingsByBlockId.get(entry.blockId);
            return embedding === undefined
                ? entry
                : {
                    ...entry,
                    embedding: {
                        model: embedding.model,
                        values: [...embedding.values]
                    }
                };
        })
    };
}
function hasStoredEmbeddings(vectors) {
    return vectors.entries.some((entry) => entry.embedding !== undefined);
}
function needsEmbeddingReindex(vectors, embedder) {
    if (vectors.entries.length === 0) {
        return false;
    }
    return vectors.entries.some((entry) => entry.embedding === undefined || entry.embedding.model !== embedder.model);
}
function embeddingModelFingerprints(vectors) {
    return [...new Set(vectors.entries.flatMap((entry) => (entry.embedding === undefined ? [] : [entry.embedding.model])))];
}
function withEmbeddedVectors(result, vectors) {
    if (!hasStoredEmbeddings(vectors)) {
        return result;
    }
    return {
        ...result,
        payloads: {
            ...result.payloads,
            vectors
        },
        manifest: {
            ...result.manifest,
            payloadChecksums: {
                ...result.manifest.payloadChecksums,
                vector: computePayloadChecksum(vectors)
            },
            modelFingerprints: [...new Set([...result.manifest.modelFingerprints, ...embeddingModelFingerprints(vectors)])]
        }
    };
}
export async function reindexCandidatePackBuildResultWithEmbedder(result, embedder) {
    if (!needsEmbeddingReindex(result.payloads.vectors, embedder)) {
        return result;
    }
    const vectors = await enrichVectorsPayloadWithEmbeddings(result.payloads.graph, result.payloads.vectors, embedder);
    return withEmbeddedVectors(result, vectors);
}
function countKeywordWeights(value) {
    const counts = new Map();
    for (const token of value.toLowerCase().split(/[^a-z0-9]+/u)) {
        if (token.length < 3 || !/[a-z]/u.test(token)) {
            continue;
        }
        counts.set(token, (counts.get(token) ?? 0) + 1);
    }
    return Object.fromEntries(counts.entries());
}
function lifecycleBlockIds(packId) {
    return {
        feedbackScanner: `${packId}:feedback-scanner`,
        fastBootDefaults: `${packId}:fast-boot-defaults`,
        passiveBackgroundLearning: `${packId}:passive-background-learning`,
        humanLabelHarvest: `${packId}:human-label-harvest`,
        selfLabelHarvest: `${packId}:self-label-harvest`,
        workspace: `${packId}:workspace`,
        structuralOps: `${packId}:structural-ops`
    };
}
function eventQueryTokens(event) {
    const base = event.contract === CONTRACT_IDS.feedbackEvents
        ? `${event.content} ${summarizeEvent(event)} ${event.relatedInteractionId ?? ""} ${event.messageId ?? ""}`
        : `${summarizeEvent(event)} ${event.packId ?? ""} ${event.messageId ?? ""}`;
    return keywordTokens(`${event.kind} ${event.channel} ${event.source.stream} ${base}`);
}
function supervisionKindForEvent(event) {
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
        return "human_feedback";
    }
    if (event.kind === "operator_override") {
        return "operator_override";
    }
    if (event.kind === "memory_compiled") {
        return "self_memory";
    }
    return "route_trace";
}
function rewardForEvent(event, principalBacklog) {
    let reward = 0;
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
        switch (event.kind) {
            case "correction":
                reward = 4;
                break;
            case "teaching":
                reward = 3;
                break;
            case "approval":
                reward = isImplicitPositiveApproval(event) ? 1 : 2;
                break;
            case "suppression":
                reward = -3;
                break;
        }
    }
    if (reward === 0) {
        switch (event.kind) {
            case "operator_override":
                reward = 4;
                break;
            case "memory_compiled":
                reward = 2;
                break;
            default:
                reward = 0;
                break;
        }
    }
    if (reward === 0) {
        return 0;
    }
    return roundMetric(reward + principalMetadataRouteRewardBoost(event.principal) + principalBacklogRouteRewardBoost(event.principal, principalBacklog));
}
function targetBlockIdsForEvent(packId, event) {
    const targetBlockIds = new Set([`${packId}:event:${event.eventId}`]);
    if (event.contract === CONTRACT_IDS.feedbackEvents && event.relatedInteractionId !== undefined) {
        targetBlockIds.add(`${packId}:event:${event.relatedInteractionId}`);
    }
    return [...targetBlockIds].sort();
}
function buildRouterTrace(packId, event, principalBacklog) {
    const queryTokens = eventQueryTokens(event);
    const traceId = `trace-${stableHash(checksumJsonPayload({ packId, eventId: event.eventId, kind: event.kind, queryTokens }))}`;
    return {
        traceId,
        sourceEventId: event.eventId,
        sourceContract: event.contract,
        sourceKind: event.kind,
        supervisionKind: supervisionKindForEvent(event),
        targetBlockIds: targetBlockIdsForEvent(packId, event),
        reward: rewardForEvent(event, principalBacklog),
        queryTokens,
        queryVector: countKeywordWeights(event.contract === CONTRACT_IDS.feedbackEvents ? `${event.content} ${summarizeEvent(event)}` : summarizeEvent(event))
    };
}
function buildBlockTokenWeights(block, vectorEntry) {
    const weights = new Map();
    const assign = (keyword, weight) => {
        for (const token of keywordTokens(keyword)) {
            weights.set(token, Math.max(weights.get(token) ?? 0, weight));
        }
    };
    for (const keyword of block.keywords) {
        assign(keyword, 1);
    }
    if (vectorEntry !== undefined) {
        for (const keyword of vectorEntry.keywords) {
            assign(keyword, 2);
        }
        for (const [keyword, weight] of Object.entries(vectorEntry.weights ?? {})) {
            assign(keyword, Math.max(1, Math.round(weight)));
        }
    }
    return weights;
}
function routingChannelsForRouterFeatures(block, vectorEntry) {
    if (block.routing !== undefined) {
        return [...block.routing.channels];
    }
    const channels = new Set();
    if ((block.edges?.length ?? 0) > 0 || (block.state?.traversalBias ?? 0) > 0 || block.learning.role === "structural") {
        channels.add("graph");
    }
    if (["interaction", "feedback", "teacher_supervision", "label_surface"].includes(block.learning.role)) {
        channels.add("short_term");
    }
    if (vectorEntry !== undefined || block.keywords.length > 0 || ["background_expectation", "boot_default", "workspace"].includes(block.learning.role)) {
        channels.add("vector");
    }
    return channels.size === 0 ? ["vector"] : [...channels];
}
function routerFeatureTokensForBlock(block, vectorEntry) {
    const features = new Set();
    features.add(`feature:role:${block.learning.role}`);
    for (const channel of routingChannelsForRouterFeatures(block, vectorEntry)) {
        features.add(`feature:channel:${channel}`);
    }
    if (block.learning.humanLabels > 0) {
        features.add("feature:labels:human");
    }
    if (block.learning.selfLabels > 0) {
        features.add("feature:labels:self");
    }
    if (block.initSeed !== undefined) {
        features.add(`feature:init_role:${block.initSeed.fileRole}`);
        features.add(`feature:init_node:${block.initSeed.nodeType}`);
        for (const channel of block.initSeed.seededChannels) {
            features.add(`feature:seed_channel:${channel}`);
        }
    }
    if ((block.state?.freshness ?? 1) >= 0.85) {
        features.add("feature:freshness:fresh");
    }
    else if ((block.state?.freshness ?? 1) < 0.5) {
        features.add("feature:freshness:stale");
    }
    if ((block.state?.strength ?? 0) >= 6) {
        features.add("feature:strength:reinforced");
    }
    if ((block.edges?.length ?? 0) > 0) {
        features.add("feature:topology:connected");
    }
    return [...features].sort();
}
/**
 * Build an adjacency map from the graph's block edges.
 *
 * For each block, collects the targetBlockIds from its edges, filtering out:
 * - Self-loops (edges pointing back to the same block)
 * - Edges targeting blocks that don't exist in the graph
 *
 * Blocks with no (valid) outgoing edges get an empty neighbor array.
 */
export function buildAdjacencyMap(graph) {
    const blockIds = new Set(graph.blocks.map((block) => block.id));
    const adjacency = new Map();
    for (const block of graph.blocks) {
        const neighbors = [];
        const seen = new Set();
        for (const edge of block.edges ?? []) {
            const target = edge.targetBlockId;
            // Exclude self-loops and targets not present in the graph; deduplicate
            if (target !== block.id && blockIds.has(target) && !seen.has(target)) {
                seen.add(target);
                neighbors.push(target);
            }
        }
        adjacency.set(block.id, neighbors);
    }
    return adjacency;
}
/**
 * Look up the neighbor IDs for a given block from a pre-built adjacency map.
 * Returns an empty array if the block is not in the map.
 */
function getNeighborIds(blockId, adjacency) {
    return adjacency.get(blockId) ?? [];
}
function softmax(values) {
    if (values.length === 0) {
        return [];
    }
    const maxValue = Math.max(...values);
    const exponents = values.map((value) => Math.exp(value - maxValue));
    const denominator = exponents.reduce((sum, value) => sum + value, 0);
    if (denominator === 0) {
        return values.map(() => 1 / values.length);
    }
    return exponents.map((value) => value / denominator);
}
function policyGradientBaseLogit(block, vectorEntry, trace) {
    const blockWeights = buildBlockTokenWeights(block, vectorEntry);
    const overlapTokens = trace.queryTokens.filter((token) => blockWeights.has(token));
    const overlapScore = overlapTokens.reduce((sum, token) => sum + Math.max(1, Math.min(4, blockWeights.get(token) ?? 1)), 0);
    return {
        logit: overlapScore +
            (block.priority ?? 0) * 0.15 +
            (block.state?.strength ?? 0) * 0.05 +
            (block.state?.freshness ?? 1) * 0.05 +
            (block.routing?.graphBias ?? 0) * 0.2 +
            (block.routing?.vectorBias ?? 0) * 0.2 +
            (block.routing?.shortTermBias ?? 0) * 0.1 +
            (block.learning.humanLabels ?? 0) * 0.1 +
            (block.learning.selfLabels ?? 0) * 0.05,
        overlapTokens
    };
}
export function buildGraphLocalActionSet(nodeBlockId, neighborBlockIds, graph, vectors, queryContext, tau, stopBias = 0.0) {
    // Build lookup maps for blocks and vector entries
    const blockById = new Map(graph.blocks.map((b) => [b.id, b]));
    const vectorById = new Map(vectors.entries.map((e) => [e.blockId, e]));
    // Construct a minimal trace-compatible object for policyGradientBaseLogit
    const syntheticTrace = {
        queryTokens: queryContext.queryTokens,
        queryVector: queryContext.queryVector
    };
    // Compute logit for each neighbor: base logit / τ
    const logits = new Map();
    for (const neighborId of neighborBlockIds) {
        const block = blockById.get(neighborId);
        if (block === undefined) {
            continue; // skip neighbors not present in graph
        }
        const vectorEntry = vectorById.get(neighborId);
        const { logit } = policyGradientBaseLogit(block, vectorEntry, syntheticTrace);
        logits.set(neighborId, logit / tau);
    }
    // Add STOP action
    logits.set("STOP", stopBias / tau);
    // Compute softmax over all logits (neighbors + STOP)
    const keys = Array.from(logits.keys());
    const values = keys.map((k) => logits.get(k));
    const probs = softmax(values);
    const probabilities = new Map();
    for (let i = 0; i < keys.length; i++) {
        probabilities.set(keys[i], probs[i]);
    }
    // Filter neighborBlockIds to only those actually present in graph
    const validNeighborIds = neighborBlockIds.filter((id) => logits.has(id));
    return {
        nodeBlockId,
        neighborBlockIds: validNeighborIds,
        includesStop: true,
        logits,
        probabilities
    };
}
// ---------------------------------------------------------------------------
// Trajectory types and corrected tail-sum policy gradient (Gu 2016)
// ---------------------------------------------------------------------------
// TrajectoryStepV1, TrajectoryV1 defined below in PG5 section
export const STOP_ACTION_ID = "__STOP__";
const STOP_ACTION = STOP_ACTION_ID;
const DEFAULT_STOP_BIAS = 0.0;
const DEFAULT_TAU = 1.0;
const DEFAULT_BASELINE_ALPHA = 0.05;
const MAX_TRAJECTORY_LENGTH = 20;
function initBaseline(alpha = 0.05) {
    return { movingAverage: 0, count: 0, alpha, lastUpdatedAt: new Date().toISOString() };
}
export function createDefaultBaselineState(alpha = 0.05) {
    return initBaseline(alpha);
}
/**
 * EMA update: on first observation use the raw outcome; thereafter blend.
 */
export function updateBaseline(current, outcome) {
    const newAvg = current.count === 0
        ? outcome
        : current.movingAverage * (1 - current.alpha) + outcome * current.alpha;
    return {
        movingAverage: newAvg,
        count: current.count + 1,
        alpha: current.alpha,
        lastUpdatedAt: new Date().toISOString()
    };
}
/**
 * Corrected tail-sum policy gradient update for a single trajectory.
 *
 * From Gu (2016):
 *   ∂v(s_t)/∂W = E[ z_T · Σ_{l=t}^{T-1} ∇_W log π(a_l | s_l) ]
 *
 * At each node i, ∇_{w_i} log π(a|i) = (1/τ)(e_a − π_i),
 * so Σ_j gradient_j = 0 (mass redistribution, not inflation).
 */
export function computeTrajectoryPolicyGradient(trajectory, adjacency, graph, vectors, tau, pgScale) {
    const updates = new Map();
    const T = trajectory.steps.length;
    if (T === 0) {
        return updates;
    }
    const advantage = trajectory.outcome - trajectory.baselineValue;
    if (advantage === 0) {
        return updates;
    }
    // For each starting step t, compute the tail sum of score gradients l=t..T-1
    for (let t = 0; t < T; t++) {
        const tailGradient = new Map();
        for (let l = t; l < T; l++) {
            const step = trajectory.steps[l];
            // Build the set of candidate actions (neighbors + STOP)
            const candidates = step.candidateNeighborIds;
            // ∇_W log π(a_l | s_l) = (1/τ)(e_{a_l} - π_{s_l})
            for (const j of candidates) {
                const prob = step.candidateProbabilities[j] ?? 0;
                let grad;
                if (step.actionBlockId === null) {
                    // STOP was chosen
                    if (j === STOP_ACTION_ID) {
                        grad = (1 / tau) * (1 - prob);
                    }
                    else {
                        grad = (1 / tau) * (-prob);
                    }
                }
                else {
                    if (j === step.actionBlockId) {
                        grad = (1 / tau) * (1 - prob);
                    }
                    else {
                        grad = (1 / tau) * (-prob);
                    }
                }
                tailGradient.set(j, (tailGradient.get(j) ?? 0) + grad);
            }
        }
        // Weight by advantage and pgScale, accumulate into updates
        for (const [blockId, grad] of tailGradient) {
            // Skip STOP action — it has no learnable weight
            if (blockId === STOP_ACTION_ID) {
                continue;
            }
            const scaledDelta = advantage * grad * pgScale;
            const current = updates.get(blockId);
            if (current === undefined) {
                updates.set(blockId, {
                    delta: scaledDelta,
                    evidenceCount: 1,
                    rewardSum: trajectory.outcome
                });
            }
            else {
                current.delta += scaledDelta;
                current.evidenceCount += 1;
                current.rewardSum += trajectory.outcome;
            }
        }
    }
    return updates;
}
/**
 * Aggregate policy gradient updates from multiple trajectories into
 * RouterPolicyUpdateV1[] format.
 */
export function aggregateTrajectoryUpdates(trajectories, adjacency, graph, vectors, tau, pgScale) {
    const aggregated = new Map();
    for (const trajectory of trajectories) {
        const trajectoryUpdates = computeTrajectoryPolicyGradient(trajectory, adjacency, graph, vectors, tau, pgScale);
        for (const [blockId, update] of trajectoryUpdates) {
            const current = aggregated.get(blockId);
            if (current === undefined) {
                aggregated.set(blockId, { ...update });
            }
            else {
                current.delta += update.delta;
                current.evidenceCount += update.evidenceCount;
                current.rewardSum += update.rewardSum;
            }
        }
    }
    return [...aggregated.entries()]
        .map(([blockId, update]) => ({
        blockId,
        delta: roundPolicyGradientValue(update.delta),
        evidenceCount: update.evidenceCount,
        rewardSum: roundPolicyGradientValue(update.rewardSum),
        tokenWeights: {},
        traceIds: []
    }))
        .filter((update) => update.delta !== 0)
        .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta) ||
        right.evidenceCount - left.evidenceCount ||
        left.blockId.localeCompare(right.blockId));
}
// ---------------------------------------------------------------------------
function roundPolicyGradientValue(value) {
    const rounded = Math.round(value * 100) / 100;
    return Math.abs(rounded) < 0.01 ? 0 : rounded;
}
function summarizeVisibleLearnedDelta(router) {
    return {
        routerIdentity: router?.routerIdentity ?? null,
        trainingMethod: router?.training.method ?? null,
        refreshStatus: router?.training.status ?? null,
        updateCount: router?.training.updateCount ?? 0,
        supervisionCount: router?.training.supervisionCount ?? 0,
        weightsChecksum: router?.training.weightsChecksum ?? null,
        visibleDelta: router?.policyUpdates.slice(0, 3).map((update) => `${update.blockId}:${update.delta}`) ?? [],
        noOpReason: router?.training.noOpReason ?? null
    };
}
function createRouterArtifact(packId, builtAt, graph, vectors, eventExport, sparseFeedback, principalBacklog) {
    const sparseFeedbackEvaluation = eventExport === null ? null : evaluateSparseFeedback(eventExport.feedbackEvents, builtAt, sparseFeedback);
    const selectedFeedbackIds = sparseFeedbackEvaluation?.selectedEventIds ?? new Set();
    const traces = eventExport === null
        ? []
        : sortNormalizedEvents([
            ...eventExport.interactionEvents,
            ...eventExport.feedbackEvents.filter((event) => selectedFeedbackIds.has(event.eventId))
        ]).map((event) => buildRouterTrace(packId, event, principalBacklog));
    const blockIds = new Set(graph.blocks.map((block) => block.id));
    // Build reverse index: original block ID → current graph block ID.
    // When blocks are merged/split, the original event block IDs (e.g. packId:event:eventId)
    // disappear and get replaced by new IDs (e.g. packId:merge:1). The original IDs are
    // stored in each block's `compactedFrom` array. Without this mapping, targetBlockIds
    // from traces (which reference the original event IDs) never match any graph block,
    // causing zero gradient updates.
    const compactedFromIndex = new Map();
    for (const block of graph.blocks) {
        if (block.compactedFrom !== undefined) {
            for (const originId of block.compactedFrom) {
                compactedFromIndex.set(originId, block.id);
            }
        }
    }
    const vectorEntries = new Map(vectors.entries.map((entry) => [entry.blockId, entry]));
    const policyUpdatesByBlock = new Map();
    const pgScale = 8;
    for (const trace of traces) {
        if (trace.reward === 0) {
            continue;
        }
        // Remap targetBlockIds through the compactedFrom index: if an original event block
        // was merged/split into a new block, resolve to that block's current ID.
        const remappedTargetIds = trace.targetBlockIds.map((blockId) => blockIds.has(blockId) ? blockId : compactedFromIndex.get(blockId) ?? blockId);
        let targetIds = [...new Set(remappedTargetIds)].filter((blockId) => blockIds.has(blockId));
        // Suffix-based fallback: when the graph was cloned from a runtime graph, block IDs
        // have a different prefix (e.g. "runtime-graph-xxx:event:evt-*" vs "pack-yyy:event:evt-*").
        // Match by stripping the prefix and comparing the suffix after the first colon group.
        if (targetIds.length === 0) {
            const suffixIndex = new Map();
            for (const block of graph.blocks) {
                // Extract the suffix after the first ":" segment (e.g. "event:evt-feedback-extract-abc")
                const colonIdx = block.id.indexOf(":");
                if (colonIdx >= 0) {
                    suffixIndex.set(block.id.slice(colonIdx + 1), block.id);
                }
            }
            const suffixMatched = [];
            for (const targetId of trace.targetBlockIds) {
                const colonIdx = targetId.indexOf(":");
                if (colonIdx >= 0) {
                    const suffix = targetId.slice(colonIdx + 1);
                    const match = suffixIndex.get(suffix);
                    if (match !== undefined) {
                        suffixMatched.push(match);
                    }
                }
            }
            if (suffixMatched.length > 0) {
                targetIds = [...new Set(suffixMatched)];
            }
        }
        // Content-based fallback: when synthetic event block IDs don't match any graph block
        // (e.g. live_loop path where runtime graph blocks have different ID conventions),
        // find the best-matching graph blocks by keyword overlap with the trace's queryTokens.
        if (targetIds.length === 0 && trace.queryTokens.length > 0) {
            const queryTokenSet = new Set(trace.queryTokens);
            const scored = [];
            for (const block of graph.blocks) {
                const blockTokens = new Set([
                    ...keywordTokens(block.text),
                    ...block.keywords.flatMap((keyword) => keywordTokens(keyword)),
                    ...keywordTokens(block.source)
                ]);
                let overlap = 0;
                for (const token of queryTokenSet) {
                    if (blockTokens.has(token)) {
                        overlap++;
                    }
                }
                if (overlap >= 2) {
                    scored.push({ blockId: block.id, overlap });
                }
            }
            if (scored.length > 0) {
                scored.sort((left, right) => right.overlap - left.overlap);
                const topOverlap = scored[0].overlap;
                // Take all blocks within 70% of the best match score, up to 3
                targetIds = scored
                    .filter((entry) => entry.overlap >= topOverlap * 0.7)
                    .slice(0, 3)
                    .map((entry) => entry.blockId);
            }
        }
        if (targetIds.length === 0) {
            continue;
        }
        const perBlock = graph.blocks.map((block) => {
            const vectorEntry = vectorEntries.get(block.id);
            const base = policyGradientBaseLogit(block, vectorEntry, trace);
            return {
                block,
                featureTokens: routerFeatureTokensForBlock(block, vectorEntry),
                overlapTokens: base.overlapTokens,
                targeted: targetIds.includes(block.id),
                logit: base.logit
            };
        });
        const probabilities = softmax(perBlock.map((entry) => entry.logit));
        const targetProbability = 1 / targetIds.length;
        for (const [index, entry] of perBlock.entries()) {
            if (!entry.targeted && entry.overlapTokens.length === 0) {
                continue;
            }
            const expectedProbability = probabilities[index] ?? 0;
            const gradient = trace.reward * ((entry.targeted ? targetProbability : 0) - expectedProbability);
            const delta = roundPolicyGradientValue(gradient * pgScale);
            if (delta === 0) {
                continue;
            }
            const current = policyUpdatesByBlock.get(entry.block.id) ?? {
                blockId: entry.block.id,
                delta: 0,
                evidenceCount: 0,
                rewardSum: 0,
                tokenWeights: new Map(),
                traceIds: new Set()
            };
            current.delta += delta;
            current.evidenceCount += 1;
            current.rewardSum += trace.reward;
            current.traceIds.add(trace.traceId);
            const weightedTokens = entry.targeted ? [...new Set([...trace.queryTokens, ...entry.featureTokens])] : entry.overlapTokens;
            for (const token of weightedTokens) {
                const tokenGradient = roundPolicyGradientValue(gradient * Math.max(1, trace.queryVector[token] ?? 1));
                if (tokenGradient === 0) {
                    continue;
                }
                current.tokenWeights.set(token, roundPolicyGradientValue((current.tokenWeights.get(token) ?? 0) + tokenGradient));
            }
            policyUpdatesByBlock.set(entry.block.id, current);
        }
    }
    const policyUpdates = [...policyUpdatesByBlock.values()]
        .map((update) => ({
        blockId: update.blockId,
        delta: update.delta,
        evidenceCount: update.evidenceCount,
        rewardSum: update.rewardSum,
        tokenWeights: Object.fromEntries([...update.tokenWeights.entries()].sort(([left], [right]) => left.localeCompare(right))),
        traceIds: [...update.traceIds].sort()
    }))
        .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta) || right.evidenceCount - left.evidenceCount || left.blockId.localeCompare(right.blockId));
    const supervisionCount = traces.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length;
    const collectedLabels = computeRouterCollectedLabelCounts(traces);
    const queryChecksum = computeRouterQueryChecksum(traces);
    const status = policyUpdates.length > 0 ? "updated" : "no_supervision";
    const noOpReason = policyUpdates.length > 0
        ? null
        : eventExport === null
            ? "no normalized event export supplied for learned routing refresh"
            : supervisionCount === 0
                ? "no canonical supervision found in normalized event export"
                : "supervision produced no learned routing delta";
    return {
        routerIdentity: `${packId}:route_fn`,
        strategy: "learned_route_fn_v1",
        trainedAt: builtAt,
        requiresLearnedRouting: true,
        training: {
            method: "policy_gradient_v1",
            status,
            eventExportDigest: eventExport?.provenance.exportDigest ?? null,
            routeTraceCount: traces.length,
            supervisionCount,
            updateCount: policyUpdates.length,
            collectedLabels,
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
                    eventExportDigest: eventExport?.provenance.exportDigest ?? null,
                    routeTraceCount: traces.length,
                    supervisionCount,
                    collectedLabels,
                    queryChecksum
                })
            },
            queryChecksum,
            weightsChecksum: computeRouterWeightsChecksum(policyUpdates),
            freshnessChecksum: computeRouterFreshnessChecksum({
                method: "policy_gradient_v1",
                trainedAt: builtAt,
                status,
                eventExportDigest: eventExport?.provenance.exportDigest ?? null,
                routeTraceCount: traces.length,
                supervisionCount,
                updateCount: policyUpdates.length
            }),
            noOpReason
        },
        traces,
        policyUpdates
    };
}
/**
 * Join serve-time decisions with feedback events to assign outcome rewards.
 * Returns decisionRecordId → outcome (z_T).
 */
export function joinDecisionsWithFeedback(decisions, eventExport, maxDelayMs = 300_000) {
    const outcomes = new Map();
    if (eventExport === null) {
        // No feedback available — all decisions get 0 (no update)
        for (const decision of decisions) {
            outcomes.set(decision.recordId, 0);
        }
        return outcomes;
    }
    // Index feedback events by sessionId and time for joining
    const feedbackBySession = new Map();
    for (const event of eventExport.feedbackEvents) {
        const sessionId = event.sessionId ?? "__none__";
        const entries = feedbackBySession.get(sessionId) ?? [];
        entries.push({
            createdAt: new Date(event.createdAt).getTime(),
            reward: rewardForEvent(event, undefined)
        });
        feedbackBySession.set(sessionId, entries);
    }
    for (const decision of decisions) {
        const sessionId = decision.sessionId ?? "__none__";
        const decisionTime = new Date(decision.recordedAt).getTime();
        const sessionFeedback = feedbackBySession.get(sessionId) ?? [];
        // Find feedback within the time window after this decision
        let bestReward = 0;
        let bestDelay = Infinity;
        for (const fb of sessionFeedback) {
            const delay = fb.createdAt - decisionTime;
            if (delay >= 0 && delay <= maxDelayMs && delay < bestDelay) {
                bestReward = fb.reward;
                bestDelay = delay;
            }
        }
        outcomes.set(decision.recordId, bestReward);
    }
    return outcomes;
}
/**
 * Build a GraphLocalActionSet using pre-computed candidate scores (for V2 trajectory reconstruction).
 */
function buildGraphLocalActionSetFromScores(nodeBlockId, neighborBlockIds, graph, vectors, candidateScores, tau) {
    const logits = new Map();
    for (const neighborId of neighborBlockIds) {
        const score = candidateScores.get(neighborId) ?? 0;
        logits.set(neighborId, score / tau);
    }
    logits.set(STOP_ACTION, DEFAULT_STOP_BIAS / tau);
    const keys = Array.from(logits.keys());
    const values = keys.map((k) => logits.get(k));
    const probs = softmax(values);
    const probabilities = new Map();
    for (let i = 0; i < keys.length; i++) {
        probabilities.set(keys[i], probs[i]);
    }
    return {
        nodeBlockId,
        neighborBlockIds: neighborBlockIds.filter((id) => logits.has(id)),
        includesStop: true,
        logits,
        probabilities
    };
}
/**
 * Reconstruct a trajectory from a serve-time decision log entry.
 * Traces through the graph starting from the highest-scoring selected block.
 */
export function reconstructTrajectoryFromServeDecision(decision, graph, vectors, adjacency, tau, outcome, baselineValue) {
    const chosenSet = new Set(decision.chosenContextIds);
    const candidateScoresMap = new Map();
    for (const cs of decision.candidateScores) {
        candidateScoresMap.set(cs.blockId, cs.actionScore);
    }
    // Sort chosen blocks by score descending to find entry point
    const chosenWithScores = decision.chosenContextIds
        .map((blockId) => ({ blockId, score: candidateScoresMap.get(blockId) ?? 0 }))
        .sort((a, b) => b.score - a.score);
    const steps = [];
    const visited = new Set();
    // BFS/DFS through edges from highest-scoring chosen block
    const queue = chosenWithScores.length > 0 ? [chosenWithScores[0].blockId] : [];
    while (queue.length > 0 && steps.length < MAX_TRAJECTORY_LENGTH) {
        const nodeBlockId = queue.shift();
        if (visited.has(nodeBlockId))
            continue;
        visited.add(nodeBlockId);
        const neighbors = adjacency.get(nodeBlockId) ?? [];
        const localActionSet = buildGraphLocalActionSetFromScores(nodeBlockId, neighbors, graph, vectors, candidateScoresMap, tau);
        // Find the next chosen neighbor (action taken)
        let actionBlockId = null;
        for (const neighborId of neighbors) {
            if (chosenSet.has(neighborId) && !visited.has(neighborId)) {
                actionBlockId = neighborId;
                break;
            }
        }
        const actionKey = actionBlockId ?? STOP_ACTION;
        const actionProb = localActionSet.probabilities.get(actionKey) ?? 0;
        const actionLogProb = actionProb > 0 ? Math.log(actionProb) : -Infinity;
        steps.push({
            stepIndex: steps.length,
            nodeBlockId,
            actionBlockId,
            actionScore: candidateScoresMap.get(actionKey) ?? 0,
            actionLogProbability: actionLogProb,
            candidateNeighborIds: [...localActionSet.neighborBlockIds, STOP_ACTION],
            candidateScores: Object.fromEntries(localActionSet.logits),
            candidateProbabilities: Object.fromEntries(localActionSet.probabilities)
        });
        if (actionBlockId !== null) {
            queue.unshift(actionBlockId); // DFS: go deeper first
        }
    }
    // Add remaining chosen blocks not reachable via edges as 1-step sub-trajectories
    for (const { blockId } of chosenWithScores) {
        if (visited.has(blockId))
            continue;
        if (steps.length >= MAX_TRAJECTORY_LENGTH)
            break;
        visited.add(blockId);
        const localActionSet = buildGraphLocalActionSetFromScores(blockId, adjacency.get(blockId) ?? [], graph, vectors, candidateScoresMap, tau);
        const stopProb = localActionSet.probabilities.get(STOP_ACTION) ?? 0;
        const stopLogProb = stopProb > 0 ? Math.log(stopProb) : -Infinity;
        steps.push({
            stepIndex: steps.length,
            nodeBlockId: blockId,
            actionBlockId: null, // STOP immediately
            actionScore: 0,
            actionLogProbability: stopLogProb,
            candidateNeighborIds: [...localActionSet.neighborBlockIds, STOP_ACTION],
            candidateScores: Object.fromEntries(localActionSet.logits),
            candidateProbabilities: Object.fromEntries(localActionSet.probabilities)
        });
    }
    const trajectoryId = `traj-${stableHash(checksumJsonPayload({
        decisionId: decision.recordId,
        steps: steps.map((s) => s.nodeBlockId)
    }))}`;
    return {
        trajectoryId,
        sessionId: decision.sessionId,
        turnId: decision.turnCompileEventId,
        createdAt: decision.recordedAt,
        steps,
        outcome,
        baselineValue
    };
}
/**
 * Compute the corrected tail-sum PG update for a single trajectory.
 * Per the Gu (2016) paper: ∂v(s_t)/∂W = E[ z_T · Σ_{l=t}^{T} ∇_W log π_W(a_l | s_l) ]
 */
function computeTrajectoryPolicyGradientV2(trajectory, adjacency, graph, vectors, candidateScoresMap, tau, pgScale) {
    const updates = new Map();
    const advantage = trajectory.outcome - trajectory.baselineValue;
    if (advantage === 0) {
        return updates;
    }
    const T = trajectory.steps.length;
    for (let t = 0; t < T; t++) {
        // Compute tail sum of score gradients from step t to T
        const tailGradient = new Map();
        for (let l = t; l < T; l++) {
            const step = trajectory.steps[l];
            const actionKey = step.actionBlockId ?? STOP_ACTION;
            // ∇_W log π(a_l | s_l) = (1/τ)(e_{a_l} - π_{s_l})
            for (const neighborId of step.candidateNeighborIds) {
                const prob = step.candidateProbabilities[neighborId] ?? 0;
                const grad = neighborId === actionKey
                    ? (1 / tau) * (1 - prob)
                    : (1 / tau) * (-prob);
                tailGradient.set(neighborId, (tailGradient.get(neighborId) ?? 0) + grad);
            }
        }
        // Weight by advantage and pgScale
        for (const [blockId, grad] of tailGradient) {
            if (blockId === STOP_ACTION)
                continue; // don't update virtual STOP
            const delta = roundPolicyGradientValue(advantage * grad * pgScale);
            if (delta === 0)
                continue;
            const current = updates.get(blockId) ?? { delta: 0, evidenceCount: 0, rewardSum: 0 };
            current.delta += delta;
            current.evidenceCount += 1;
            current.rewardSum += trajectory.outcome;
            updates.set(blockId, current);
        }
    }
    return updates;
}
/**
 * Create a V2 (paper-aligned) router artifact using trajectory-based PG updates.
 *
 * This function:
 * 1. Builds adjacency map from graph
 * 2. Joins serve-time decisions with feedback events (to get outcomes)
 * 3. Reconstructs trajectories from serve-time decisions
 * 4. For each trajectory, updates baseline and computes advantage
 * 5. Runs corrected tail-sum PG update across all trajectories
 * 6. Aggregates into RouterPolicyUpdateV1[] format
 * 7. Builds the router artifact with ROUTER_PG_PROFILE_V2
 */
function createRouterArtifactV2(packId, builtAt, graph, vectors, eventExport, serveTimeDecisions, baselineState, sparseFeedback, principalBacklog) {
    const tau = DEFAULT_TAU;
    const pgScale = 8;
    // 1. Build adjacency map from graph
    const adjacency = buildAdjacencyMap(graph);
    // 2. Join serve-time decisions with feedback events to get outcomes
    const outcomeMap = joinDecisionsWithFeedback(serveTimeDecisions, eventExport);
    // 3 & 4. Reconstruct trajectories and update baseline
    let currentBaseline = { ...baselineState };
    const trajectories = [];
    for (const decision of serveTimeDecisions) {
        const outcome = outcomeMap.get(decision.recordId) ?? 0;
        const trajectory = reconstructTrajectoryFromServeDecision(decision, graph, vectors, adjacency, tau, outcome, currentBaseline.movingAverage);
        trajectories.push(trajectory);
        // Update baseline after each trajectory
        if (outcome !== 0) {
            currentBaseline = updateBaseline(currentBaseline, outcome);
        }
    }
    // 5 & 6. Run corrected tail-sum PG update across all trajectories and aggregate
    const aggregatedUpdates = new Map();
    for (const trajectory of trajectories) {
        if (trajectory.outcome === 0)
            continue;
        // Build candidate scores map from the decision's recorded scores
        const decision = serveTimeDecisions.find((d) => d.recordId === trajectory.turnId ||
            `traj-${stableHash(checksumJsonPayload({ decisionId: d.recordId, steps: trajectory.steps.map((s) => s.nodeBlockId) }))}` === trajectory.trajectoryId);
        const candidateScoresMap = new Map();
        if (decision !== undefined) {
            for (const cs of decision.candidateScores) {
                candidateScoresMap.set(cs.blockId, cs.actionScore);
            }
        }
        const trajectoryUpdates = computeTrajectoryPolicyGradientV2(trajectory, adjacency, graph, vectors, candidateScoresMap, tau, pgScale);
        for (const [blockId, update] of trajectoryUpdates) {
            const current = aggregatedUpdates.get(blockId) ?? {
                blockId,
                delta: 0,
                evidenceCount: 0,
                rewardSum: 0,
                traceIds: new Set()
            };
            current.delta += update.delta;
            current.evidenceCount += update.evidenceCount;
            current.rewardSum += update.rewardSum;
            current.traceIds.add(trajectory.trajectoryId);
            aggregatedUpdates.set(blockId, current);
        }
    }
    // Format as RouterPolicyUpdateV1[]
    const policyUpdates = [...aggregatedUpdates.values()]
        .map((update) => ({
        blockId: update.blockId,
        delta: roundPolicyGradientValue(update.delta),
        evidenceCount: update.evidenceCount,
        rewardSum: update.rewardSum,
        tokenWeights: {}, // V2 does not use token-level weights (graph-local softmax instead)
        traceIds: [...update.traceIds].sort()
    }))
        .filter((update) => update.delta !== 0)
        .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta) ||
        right.evidenceCount - left.evidenceCount ||
        left.blockId.localeCompare(right.blockId));
    // V2 learns from reconstructed trajectories rather than RouterTraceV1 records, so
    // the router artifact keeps an empty trace list and reports trace-derived counts as 0.
    const traces = [];
    const supervisedTrajectoryCount = trajectories.filter((t) => t.outcome !== 0).length;
    const routeTraceCount = traces.length;
    const supervisionCount = traces.filter((trace) => trace.reward !== 0).length;
    const queryChecksum = computeRouterQueryChecksum(traces);
    const collectedLabels = computeRouterCollectedLabelCounts(traces);
    const status = policyUpdates.length > 0 ? "updated" : "no_supervision";
    const noOpReason = policyUpdates.length > 0
        ? null
        : serveTimeDecisions.length === 0
            ? "no serve-time decisions supplied for V2 learned routing refresh"
            : supervisedTrajectoryCount === 0
                ? "no outcomes found for serve-time decisions"
                : "trajectory updates produced no learned routing delta";
    const artifact = {
        routerIdentity: `${packId}:route_fn`,
        strategy: "learned_route_fn_v1",
        trainedAt: builtAt,
        requiresLearnedRouting: true,
        training: {
            method: "policy_gradient_v1",
            status,
            eventExportDigest: eventExport?.provenance.exportDigest ?? null,
            routeTraceCount,
            supervisionCount,
            updateCount: policyUpdates.length,
            collectedLabels,
            objective: {
                updateMechanism: "policy_gradient",
                updateVersion: "route_pg_update_v1",
                objective: "supervised_route_pg_v1",
                profile: ROUTER_PG_PROFILE_V2,
                objectiveChecksum: computeRouterObjectiveChecksum({
                    updateMechanism: "policy_gradient",
                    updateVersion: "route_pg_update_v1",
                    objective: "supervised_route_pg_v1",
                    profile: ROUTER_PG_PROFILE_V2,
                    eventExportDigest: eventExport?.provenance.exportDigest ?? null,
                    routeTraceCount,
                    supervisionCount,
                    collectedLabels,
                    queryChecksum
                })
            },
            queryChecksum,
            weightsChecksum: computeRouterWeightsChecksum(policyUpdates),
            freshnessChecksum: computeRouterFreshnessChecksum({
                method: "policy_gradient_v1",
                trainedAt: builtAt,
                status,
                eventExportDigest: eventExport?.provenance.exportDigest ?? null,
                routeTraceCount,
                supervisionCount,
                updateCount: policyUpdates.length
            }),
            noOpReason
        },
        traces,
        policyUpdates
    };
    return { artifact, updatedBaseline: currentBaseline };
}
function resolveEventExport(input) {
    if (input.eventExports === undefined) {
        return null;
    }
    const eventExport = buildNormalizedEventExport(input.eventExports);
    const validationErrors = validateNormalizedEventExport(eventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    if (eventExport.range.start !== input.eventRange.start || eventExport.range.end !== input.eventRange.end) {
        throw new Error(`event export range ${eventExport.range.start}-${eventExport.range.end} does not match requested range ${input.eventRange.start}-${input.eventRange.end}`);
    }
    return eventExport;
}
function defaultLearningSurface(workspace, offlineArtifacts, workspaceInit) {
    return createDefaultLearningSurface(uniqueInOrder([
        `workspace:${workspace.workspaceId}`,
        ...workspaceInit.bootInputs.map((filePath) => `workspace_init:${filePath}`),
        ...offlineArtifacts.map((artifact) => `offline:${artifact}`)
    ]));
}
function cloneRuntimeGraphForPack(packId, runtimeGraph, builtAt) {
    const cloned = structuredClone(runtimeGraph);
    cloned.packId = packId;
    if (cloned.evolution !== undefined) {
        cloned.evolution = {
            ...cloned.evolution,
            builtAt
        };
    }
    return cloned;
}
function summarizeRuntimeGraphPlasticity(source, graph, builtAt, sourcePackId, eventExport) {
    return {
        source,
        graphChecksum: computePayloadChecksum(graph),
        builtAt,
        sourcePackId,
        blockCount: graph.blocks.length,
        strongestBlockId: graph.evolution?.strongestBlockId ?? null,
        eventRange: eventExport === null
            ? null
            : {
                start: eventExport.range.start,
                end: eventExport.range.end,
                count: eventExport.range.count
            },
        eventExportDigest: eventExport?.provenance.exportDigest ?? null,
        evolution: graph.evolution ?? null
    };
}
function buildRuntimeGraphSnapshot(input) {
    if (input.interactionEvents.length === 0 && input.feedbackEvents.length === 0) {
        return null;
    }
    const normalizedEventExport = buildNormalizedEventExport({
        interactionEvents: [...input.interactionEvents],
        feedbackEvents: [...input.feedbackEvents]
    });
    const validationErrors = validateNormalizedEventExport(normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    const builtAt = input.builtAt ?? normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? "2026-03-06T00:00:00.000Z";
    const workspace = createWorkspaceMetadata(input.workspace);
    const structuralOps = normalizeAlwaysOnStructuralOps(input.state?.structuralController.structuralOps, normalizedEventExport);
    const candidateInput = {
        packLabel: `${input.packLabel}-runtime-plasticity`,
        workspace: input.workspace,
        eventRange: {
            start: normalizedEventExport.range.start,
            end: normalizedEventExport.range.end
        },
        eventExports: {
            interactionEvents: [...normalizedEventExport.interactionEvents],
            feedbackEvents: [...normalizedEventExport.feedbackEvents]
        },
        ...(input.teacherSupervisionArtifacts !== undefined ? { teacherSupervisionArtifacts: input.teacherSupervisionArtifacts } : {}),
        learnedRouting: false,
        builtAt,
        ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
        structuralOps
    };
    const runtimePackId = `runtime-graph-${stableHash(checksumJsonPayload({
        packLabel: input.packLabel,
        workspaceSnapshot: workspace.snapshotId,
        eventExportDigest: normalizedEventExport.provenance.exportDigest,
        builtAt,
        structuralOps: structuralOpsSummary(candidateInput)
    }))}`;
    const graph = createGraphPayload(runtimePackId, candidateInput, workspace, normalizedEventExport, normalizedEventExport.provenance.learningSurface, builtAt, buildPointerAwareWorkingSet({
        rootDir: workspace.rootDir,
        observedAt: builtAt
    }));
    return {
        graph,
        plasticity: summarizeRuntimeGraphPlasticity("live_loop", graph, builtAt, null, normalizedEventExport)
    };
}
export function buildCandidatePack(input) {
    const builtAt = input.builtAt ?? "2026-03-06T00:00:00.000Z";
    const routePolicy = input.learnedRouting ? "requires_learned_routing" : "heuristic_allowed";
    const eventExport = resolveEventExport(input);
    const eventRange = eventExport?.range ?? createExplicitEventRange(input.eventRange);
    const workspace = createWorkspaceMetadata(input.workspace);
    const offlineArtifacts = input.offlineArtifacts ?? [];
    const teacherSupervisionArtifacts = normalizeTeacherSupervisionArtifacts(input.teacherSupervisionArtifacts);
    const workspaceInit = buildPointerAwareWorkingSet({
        rootDir: workspace.rootDir,
        observedAt: builtAt
    });
    const learningSurface = eventExport?.provenance.learningSurface ?? defaultLearningSurface(workspace, offlineArtifacts, workspaceInit);
    const sparseFeedback = normalizeSparseFeedbackPolicy(input.sparseFeedback);
    const decisionLogCount = input.serveTimeDecisions?.length ?? 0;
    const useV2 = input.pgVersion === "v2" && decisionLogCount > 0;
    const fallbackReason = !input.learnedRouting
        ? null
        : input.pgVersion === "v2" && decisionLogCount === 0
            ? "no_serve_time_decisions"
            : null;
    const routingSeed = input.learnedRouting && (input.pgVersion === "v2" || decisionLogCount > 0 || input.baselineState !== undefined)
        ? {
            pgVersionRequested: input.pgVersion ?? "v1",
            serveTimeDecisionDigest: input.serveTimeDecisions === undefined ? null : checksumJsonPayload(input.serveTimeDecisions),
            baselineState: input.baselineState ?? null
        }
        : null;
    const seed = JSON.stringify({
        packLabel: input.packLabel,
        workspace,
        eventRange,
        learnedRouting: input.learnedRouting,
        builtAt,
        offlineArtifacts,
        structuralOps: structuralOpsSummary(input),
        sparseFeedback,
        eventExportDigest: eventExport?.provenance.exportDigest ?? null,
        learningSurface,
        workspaceInitDigest: workspaceInit.graphDigest,
        workspaceInitBootInputs: workspaceInit.bootInputs,
        workspaceInitWorkingSet: workspaceInit.workingSet,
        teacherSupervisionArtifacts,
        runtimeGraphChecksum: input.runtimeGraph === undefined ? null : computePayloadChecksum(input.runtimeGraph),
        runtimePlasticitySource: input.runtimeGraph === undefined ? "candidate_build" : "live_loop",
        principalBacklog: input.principalBacklog ?? null,
        routingSeed
    });
    const packId = `pack-${stableHash(seed)}`;
    const graph = input.runtimeGraph === undefined
        ? createGraphPayload(packId, input, workspace, eventExport, learningSurface, builtAt, workspaceInit)
        : cloneRuntimeGraphForPack(packId, input.runtimeGraph, builtAt);
    const runtimePlasticity = summarizeRuntimeGraphPlasticity(input.runtimeGraph === undefined ? "candidate_build" : "live_loop", graph, builtAt, input.runtimeGraph?.packId ?? null, eventExport);
    const vectors = createVectorsPayload(graph);
    let router = null;
    let updatedBaseline = null;
    if (input.learnedRouting) {
        if (useV2) {
            const v2Result = createRouterArtifactV2(packId, builtAt, graph, vectors, eventExport, input.serveTimeDecisions, input.baselineState ?? initBaseline(), input.sparseFeedback, input.principalBacklog);
            router = v2Result.artifact;
            updatedBaseline = v2Result.updatedBaseline;
        }
        else {
            router = createRouterArtifact(packId, builtAt, graph, vectors, eventExport, input.sparseFeedback, input.principalBacklog);
        }
    }
    const payloads = {
        graph,
        vectors,
        router
    };
    const manifest = {
        contract: CONTRACT_IDS.artifactManifest,
        packId,
        immutable: true,
        routePolicy,
        runtimeAssets: {
            graphPath: PACK_LAYOUT.graph,
            vectorPath: PACK_LAYOUT.vectors,
            router: input.learnedRouting
                ? {
                    kind: "artifact",
                    identity: payloads.router?.routerIdentity ?? null,
                    artifactPath: PACK_LAYOUT.router
                }
                : {
                    kind: "none",
                    identity: null,
                    artifactPath: null
                }
        },
        payloadChecksums: {
            graph: computePayloadChecksum(payloads.graph),
            vector: computePayloadChecksum(payloads.vectors),
            router: payloads.router === null ? null : computePayloadChecksum(payloads.router)
        },
        routeArtifact: buildRouteArtifactReference({
            routerAssetKind: input.learnedRouting ? "artifact" : "none",
            routerIdentity: payloads.router?.routerIdentity ?? null,
            routerChecksum: payloads.router === null ? null : computePayloadChecksum(payloads.router),
            router: payloads.router,
            eventExportDigest: input.learnedRouting ? eventExport?.provenance.exportDigest ?? null : null
        }),
        modelFingerprints: input.learnedRouting
            ? ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", payloads.router?.routerIdentity ?? "router:missing"]
            : ["BAAI/bge-large-en-v1.5"],
        provenance: buildArtifactProvenance({
            workspace,
            eventRange,
            eventExports: eventExport?.provenance ?? null,
            learningSurface,
            builtAt,
            offlineArtifacts
        }),
        graphDynamics: {
            bootstrapping: {
                fastBootDefaults: true,
                passiveBackgroundLearning: true
            },
            runtimePlasticitySource: runtimePlasticity.source,
            hebbian: {
                enabled: true,
                learningRate: 0.1
            },
            decay: {
                enabled: true,
                halfLifeDays: 30
            },
            structuralOps: graph.evolution?.structuralOps ?? structuralOpsSummary(input)
        }
    };
    const structuralOps = manifest.graphDynamics.structuralOps;
    const prunedBlockCount = graph.evolution?.prunedBlockIds.length ?? 0;
    const liveBlockCount = graph.blocks.length;
    const strongestBlockId = runtimePlasticity.strongestBlockId;
    const connectDiagnostics = graph.evolution?.connectDiagnostics ?? null;
    const graphEvolutionLog = {
        packId,
        provenance: runtimePlasticity.source,
        builtAt,
        graphChecksum: manifest.payloadChecksums.graph,
        blockCount: liveBlockCount,
        structuralOps: { ...structuralOps },
        connectDiagnostics,
        structuralEvolutionSummary: summarizeStructuralGraphEvolution({
            blockCount: liveBlockCount,
            strongestBlockId,
            structuralOps,
            prunedBlockCount,
            connectDiagnostics
        }),
        prunedBlockIds: graph.evolution?.prunedBlockIds ?? [],
        hebbianSummary: {
            applied: graph.evolution?.hebbianApplied ?? manifest.graphDynamics.hebbian.enabled,
            learningRate: manifest.graphDynamics.hebbian.learningRate
        },
        decaySummary: {
            applied: graph.evolution?.decayApplied ?? manifest.graphDynamics.decay.enabled,
            halfLifeDays: manifest.graphDynamics.decay.halfLifeDays
        },
        strongestBlockId,
        eventExportDigest: runtimePlasticity.eventExportDigest
    };
    return {
        manifest,
        payloads,
        routingBuild: {
            learnedRoutingPath: !input.learnedRouting
                ? "disabled"
                : useV2
                    ? "policy_gradient_v2"
                    : "policy_gradient_v1",
            pgVersionRequested: input.learnedRouting ? input.pgVersion ?? "v1" : null,
            pgVersionUsed: input.learnedRouting ? (useV2 ? "v2" : "v1") : null,
            decisionLogCount,
            fallbackReason,
            updatedBaseline
        },
        summary: {
            packId,
            immutable: true,
            routePolicy,
            workspaceSnapshot: workspace.snapshotId,
            eventRange,
            eventExportDigest: eventExport?.provenance.exportDigest ?? null,
            learningSurface: manifest.provenance.learningSurface,
            bootstrapping: manifest.graphDynamics.bootstrapping,
            workspaceInit: summarizePointerAwareWorkingSet(workspaceInit),
            runtimePlasticity,
            graphEvolutionLog,
            learnedRouter: summarizeVisibleLearnedDelta(router)
        }
    };
}
export function buildCandidatePackFromNormalizedEventExport(input) {
    const validationErrors = validateNormalizedEventExport(input.normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    const candidateInput = {
        packLabel: input.packLabel,
        workspace: input.workspace,
        eventRange: {
            start: input.normalizedEventExport.range.start,
            end: input.normalizedEventExport.range.end
        },
        eventExports: {
            interactionEvents: [...input.normalizedEventExport.interactionEvents],
            feedbackEvents: [...input.normalizedEventExport.feedbackEvents]
        },
        ...(input.teacherSupervisionArtifacts !== undefined ? { teacherSupervisionArtifacts: input.teacherSupervisionArtifacts } : {}),
        learnedRouting: input.learnedRouting,
        ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
        ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
        ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {}),
        ...(input.runtimeGraph !== undefined ? { runtimeGraph: input.runtimeGraph } : {}),
        ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {}),
        ...(input.principalBacklog !== undefined ? { principalBacklog: input.principalBacklog } : {}),
        ...(input.pgVersion !== undefined ? { pgVersion: input.pgVersion } : {}),
        ...(input.serveTimeDecisions !== undefined ? { serveTimeDecisions: [...input.serveTimeDecisions] } : {}),
        ...(input.baselineState !== undefined ? { baselineState: { ...input.baselineState } } : {})
    };
    return buildCandidatePack(candidateInput);
}
export async function buildCandidatePackWithEmbedder(input, embedder) {
    const result = buildCandidatePack(input);
    return reindexCandidatePackBuildResultWithEmbedder(result, embedder);
}
export async function buildCandidatePackFromNormalizedEventExportWithEmbedder(input, embedder) {
    const result = buildCandidatePackFromNormalizedEventExport(input);
    return reindexCandidatePackBuildResultWithEmbedder(result, embedder);
}
export function materializeCandidatePack(rootDir, input) {
    const result = buildCandidatePack(input);
    return materializeCandidatePackResult(rootDir, result);
}
export async function materializeCandidatePackWithEmbedder(rootDir, input, embedder) {
    const result = await buildCandidatePackWithEmbedder(input, embedder);
    return materializeCandidatePackResult(rootDir, result);
}
export function materializeCandidatePackFromNormalizedEventExport(rootDir, input) {
    const result = buildCandidatePackFromNormalizedEventExport(input);
    return materializeCandidatePackResult(rootDir, result);
}
export async function materializeCandidatePackFromNormalizedEventExportWithEmbedder(rootDir, input, embedder) {
    const result = await buildCandidatePackFromNormalizedEventExportWithEmbedder(input, embedder);
    return materializeCandidatePackResult(rootDir, result);
}
export function materializeCandidatePackFromNormalizedEventExportSlice(rootDir, input) {
    const result = buildCandidatePackFromNormalizedEventExportSlice(input);
    return materializeCandidatePackResult(rootDir, result);
}
export function materializeCandidatePackBundleFromNormalizedEventExportBridge(rootDir, input) {
    const bundle = buildCandidatePackBundleFromNormalizedEventExportBridge(input);
    rmSync(rootDir, { recursive: true, force: true });
    mkdirSync(rootDir, { recursive: true });
    const entries = bundle.entries.map((entry, index) => {
        const entryRootDir = buildBundleEntryRootDir(rootDir, entry, index);
        return {
            ...entry,
            rootDir: path.resolve(entryRootDir),
            descriptor: materializeCandidatePackResult(entryRootDir, entry.build)
        };
    });
    return {
        runtimeOwner: bundle.runtimeOwner,
        bridgeDigest: bundle.bridgeDigest,
        bundleDigest: bundle.bundleDigest,
        cursor: cloneCursor(bundle.cursor),
        dedupedInputCount: bundle.dedupedInputCount,
        duplicateIdentityCount: bundle.duplicateIdentityCount,
        entries
    };
}
// ---------------------------------------------------------------------------
// Baseline: EMA of trajectory returns (§3.4 of PG design doc)
// ---------------------------------------------------------------------------
const BASELINE_STATE_FILENAME = "baseline-state.json";
/**
 * Load baseline state from `<activationRoot>/baseline-state.json`.
 * If the file is missing or unparseable, returns a fresh zero-initialised state.
 */
export function loadOrInitBaseline(activationRoot) {
    const filePath = path.join(activationRoot, BASELINE_STATE_FILENAME);
    try {
        if (existsSync(filePath)) {
            const raw = readFileSync(filePath, "utf8");
            const parsed = JSON.parse(raw);
            // Validate minimally – fall through to default on bad shape
            if (typeof parsed.movingAverage === "number" &&
                typeof parsed.count === "number" &&
                typeof parsed.alpha === "number" &&
                typeof parsed.lastUpdatedAt === "string") {
                return {
                    movingAverage: parsed.movingAverage,
                    count: parsed.count,
                    alpha: parsed.alpha,
                    lastUpdatedAt: parsed.lastUpdatedAt
                };
            }
        }
    }
    catch {
        // File missing, corrupt, or unreadable – fall through to default
    }
    return {
        movingAverage: 0,
        count: 0,
        alpha: DEFAULT_BASELINE_ALPHA,
        lastUpdatedAt: new Date().toISOString()
    };
}
/**
 * Persist baseline state to `<activationRoot>/baseline-state.json`.
 * Creates the directory tree if it doesn't already exist.
 */
export function persistBaseline(activationRoot, state) {
    mkdirSync(activationRoot, { recursive: true });
    const filePath = path.join(activationRoot, BASELINE_STATE_FILENAME);
    writeFileSync(filePath, JSON.stringify(state, null, 2) + "\n", "utf8");
}
//# sourceMappingURL=index.js.map