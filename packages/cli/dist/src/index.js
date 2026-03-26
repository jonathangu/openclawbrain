import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
import { CONTRACT_IDS, buildEventSemanticSurface, buildNormalizedEventExport, canonicalJson, checksumJsonPayload, createFeedbackEvent, createInteractionEvent, sortNormalizedEvents, validateKernelSurface, validateNormalizedEventExport } from "@openclawbrain/contracts";
import { classifyFeedbackSignalContent, describeNormalizedEventExportObservability } from "@openclawbrain/event-export";
import { DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS, advanceAlwaysOnLearningRuntime, buildTeacherSupervisionArtifactsFromNormalizedEventExport, createAlwaysOnLearningRuntimeState, describeAlwaysOnLearningRuntimeState, materializeAlwaysOnLearningCandidatePack, materializeCandidatePackFromNormalizedEventExport } from "./local-learner.js";
import { LEARNING_SPINE_LOG_LAYOUT, activatePack, describeActivationObservability, describeActivationTarget, describePackCompileTarget, inspectActivationState, loadPackFromActivation, promoteCandidatePack, readLearningSpineLogEntries, rollbackActivePack, stageCandidatePack } from "@openclawbrain/pack-format";
import { inspectOpenClawBrainHookStatus, summarizeOpenClawBrainHookLoad } from "./openclaw-hook-truth.js";
import { appendLearningUpdateLogs, appendServeTimeRouteDecisionLog } from "./learning-spine.js";
import { buildFeedbackSemanticMetadata, buildInteractionSemanticMetadata } from "./semantic-metadata.js";
export { clearOpenClawProfileRuntimeLoadProof, listOpenClawProfileRuntimeLoadProofs, recordOpenClawProfileRuntimeLoadProof, resolveAttachmentRuntimeLoadProofsPath } from "./attachment-truth.js";
import { createTeacherLabeler } from "./teacher-labeler.js";
export { createHttpOllamaTeacherLabelerClient, createOllamaTeacherLabeler, createTeacherLabeler } from "./teacher-labeler.js";
const DEFAULT_AGENT_ID = "openclaw-runtime";
const FEEDBACK_KINDS = new Set(["correction", "teaching", "approval", "suppression"]);
export const DEFAULT_ASYNC_TEACHER_QUEUE_CAPACITY = 8;
const RECORDED_SESSION_TRACE_CONTRACT = "recorded_session_trace.v1";
const RECORDED_SESSION_FIXTURE_CONTRACT = "recorded_session_replay_fixture.v1";
const RECORDED_SESSION_BUNDLE_CONTRACT = "recorded_session_replay_bundle.v1";
const RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT = "normalized_event_export_bundle.v1";
const DEFAULT_ATTACH_STATUS_MESSAGE = "openclaw attach status probe";
const DEFAULT_ATTACH_STATUS_RUNTIME_HINTS = ["attach", "status", "probe"];
const BRAIN_SERVE_HOT_PATH_TIMING_DETAIL = "Measured inside compileRuntimeContext before serve-route logging; includes serve-path normalization, active-pack lookup, structural-budget resolution, route/candidate selection, and prompt assembly when run; excludes background scanner/embedder/teacher work, promotion, and runtime event-export writes.";
export const RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT = {
    manifest: "manifest.json",
    payload: "normalized-event-export.json"
};
function normalizeRuntimeProfileSelector(value, fieldName, fallback = "current_profile") {
    if (value === undefined || value === null) {
        return fallback;
    }
    return normalizeNonEmptyString(value, fieldName);
}
function normalizeBootstrapProfileSelector(value) {
    return normalizeRuntimeProfileSelector(value, "profileSelector");
}
function quoteShellArg(value) {
    return `'${value.replace(/'/g, `'"'"'`)}'`;
}
function detectBootstrapOperatorCliPrefix() {
    const npmExecPath = (process.env.npm_execpath ?? "").toLowerCase();
    const userAgent = process.env.npm_config_user_agent ?? "";
    if (npmExecPath.includes("npm-cli.js")) {
        return "npm exec openclawbrain --";
    }
    if (npmExecPath.includes("pnpm")) {
        return "pnpm exec openclawbrain";
    }
    if (userAgent.startsWith("npm/")) {
        return "npm exec openclawbrain --";
    }
    if (userAgent.startsWith("pnpm/")) {
        return "pnpm exec openclawbrain";
    }
    return "npm exec openclawbrain --";
}
function buildBootstrapAttachStatusCommand(activationRoot) {
    return `${detectBootstrapOperatorCliPrefix()} status --activation-root ${quoteShellArg(activationRoot)}`;
}
function buildBootstrapAttachStatusJsonCommand(activationRoot) {
    return `${buildBootstrapAttachStatusCommand(activationRoot)} --json`;
}
function buildBootstrapAttachRollbackDryRunCommand(activationRoot) {
    return `${detectBootstrapOperatorCliPrefix()} rollback --activation-root ${quoteShellArg(activationRoot)} --dry-run`;
}
function buildBootstrapRuntimeAttachNextSteps(input) {
    const profileSpecific = input.profileSelector !== "current_profile";
    const nextSteps = [
        {
            id: "inspect_current_profile_status",
            detail: profileSpecific
                ? `Inspect the canonical current-profile status for this activation root; explicit attach/export attribution for profileSelector="${input.profileSelector}" stays on the runtime path.`
                : 'Ask the attached gateway "How\'s the brain?" with the compact current-profile status summary; add --json for the canonical object.',
            command: buildBootstrapAttachStatusCommand(input.activationRoot)
        },
        {
            id: "preview_rollback_readiness",
            detail: "Preview rollback before any pointer move so the retained previous pack path stays explicit from the first attach.",
            command: buildBootstrapAttachRollbackDryRunCommand(input.activationRoot)
        }
    ];
    if (input.currentProfile.brainStatus.awaitingFirstExport) {
        nextSteps.push({
            id: "record_next_current_profile_turn",
            detail: profileSpecific
                ? `Export the next live turn with profileSelector="${input.profileSelector}" so refresh/promote can advance beyond seed_state_authoritative without widening the operator read scope.`
                : "Export the next live current-profile turn so refresh/promote can advance beyond seed_state_authoritative.",
            command: null
        });
    }
    else {
        nextSteps.push({
            id: "continue_current_profile_learning_loop",
            detail: profileSpecific
                ? `Keep exporting turns with profileSelector="${input.profileSelector}" and refresh/promote newer activation-ready packs off the hot path.`
                : "Keep exporting current-profile turns and refresh/promote newer activation-ready packs off the hot path.",
            command: null
        });
    }
    return nextSteps;
}
export function formatBootstrapRuntimeAttachReport(result) {
    const [nextStep, ...followUps] = result.nextSteps;
    const profileIdSuffix = result.currentProfile.profile.profileId === null ? "" : ` id=${result.currentProfile.profile.profileId}`;
    const lines = [
        `ATTACH ${result.currentProfile.brainStatus.status}`,
        `profile     selector=${result.profileSelector}${profileIdSuffix} attachment=${result.currentProfile.attachment.state} policy=${result.currentProfile.attachment.policyMode} operator_scope=${result.operatorReadScope}`,
        `activation  root=${result.activationRoot}`,
        `brain       pack=${result.packId} state=${result.currentProfile.brain.state} serve=${result.currentProfile.brainStatus.serveState}`
    ];
    if (result.profileSelector !== "current_profile") {
        lines.push("boundary    canonical status remains current_profile-only; explicit attached profile attribution stays on the runtime attach/export path");
    }
    if (nextStep !== undefined) {
        lines.push(`next        ${nextStep.detail}`);
        if (nextStep.command !== null) {
            lines.push(`command     ${nextStep.command}`);
            if (nextStep.id === "inspect_current_profile_status") {
                lines.push(`proof       ${buildBootstrapAttachStatusJsonCommand(result.activationRoot)}`);
            }
        }
    }
    lines.push(`help        ${detectBootstrapOperatorCliPrefix()} --help`);
    for (const followUp of followUps) {
        lines.push(`follow-up   ${followUp.detail}`);
        if (followUp.command !== null) {
            lines.push(`command     ${followUp.command}`);
        }
    }
    return lines.join("\n");
}
function normalizeBrainAttachmentPolicy(value) {
    if (value === "dedicated" || value === "shared") {
        return value;
    }
    return "undeclared";
}
function buildCurrentProfileAttachmentPolicy(policyMode) {
    if (policyMode === "dedicated") {
        return {
            mode: "dedicated",
            readScope: "current_profile_only",
            writeScope: "current_profile_only",
            currentProfileExclusive: true,
            requiresProfileAttribution: true,
            detail: "dedicated brains are exclusive to the current profile and must keep profile attribution explicit on every served turn"
        };
    }
    if (policyMode === "shared") {
        return {
            mode: "shared",
            readScope: "attached_profiles",
            writeScope: "attached_profiles",
            currentProfileExclusive: false,
            requiresProfileAttribution: true,
            detail: "shared brains may serve multiple attached profiles, so status and per-turn attribution must stay profile-explicit"
        };
    }
    return null;
}
const OPENCLAW_LANDING_BOUNDARIES_V1 = {
    compileBoundary: {
        contract: CONTRACT_IDS.runtimeCompile,
        activationSlot: "active",
        entrypoint: "compileRuntimeContext",
        servedFromCandidateBeforePromotion: false,
        learnedRouteEvidenceRequiredWhenManifestRequiresIt: true
    },
    eventExportBoundary: {
        emittedContracts: [CONTRACT_IDS.interactionEvents, CONTRACT_IDS.feedbackEvents],
        entrypoint: "runRuntimeTurn",
        bundleWriteOptional: true,
        writeFailuresEraseSuccessfulCompile: false,
        learningHandoffStaysOffHotPath: true
    },
    activePackBoundary: {
        servingSlot: "active",
        inspectableSlots: ["active", "candidate", "previous"],
        candidateServedBeforePromotion: false,
        previousSlotUsedForRollback: true
    },
    promotionBoundary: {
        candidateSlot: "candidate",
        activeSlot: "active",
        previousSlot: "previous",
        requiresActivationReadyCandidate: true,
        compileSeesCandidateOnlyAfterPromotion: true,
        promotionHappensOffHotPath: true
    },
    failOpenSemantics: {
        missingActivePackFallsBackToStaticContext: true,
        learnedRequiredRouteArtifactDriftHardFails: true,
        hardFailuresDisableStaticFallback: true,
        eventExportWriteFailurePreservesCompile: true
    },
    runtimeResponsibilities: [
        "runtime orchestration and session flow",
        "prompt assembly and response delivery",
        "guarded serve-path fail-open decisions"
    ],
    brainResponsibilities: [
        "normalized event emission and export handoff",
        "candidate pack materialization and learned route refresh",
        "activation staging promotion and rollback",
        "promoted-pack compilation diagnostics"
    ]
};
function toErrorMessage(error) {
    return error instanceof Error ? error.message : String(error);
}
function buildAsyncTeacherLoopNotes(input) {
    return [
        `teacher_queue_depth=${input.queueDepth}`,
        `teacher_freshness=${input.latestFreshness}`,
        `teacher_artifacts_total=${input.artifactCount}`,
        `teacher_artifacts_emitted=${input.emittedArtifactCount}`,
        `teacher_artifacts_deduped=${input.dedupedArtifactCount}`,
        `teacher_budget=${input.sparseFeedback.teacherBudget}`,
        `teacher_delay_ms=${input.sparseFeedback.teacherDelayMs}`,
        `teacher_feedback_mask=correction:${input.sparseFeedback.feedbackMask.correction},teaching:${input.sparseFeedback.feedbackMask.teaching},approval:${input.sparseFeedback.feedbackMask.approval},suppression:${input.sparseFeedback.feedbackMask.suppression}`,
        `teacher_feedback_eligible=${input.sparseFeedback.eligibleFeedbackCount}`,
        `teacher_feedback_masked=${input.sparseFeedback.maskedFeedbackCount}`,
        `teacher_feedback_delayed=${input.sparseFeedback.delayedFeedbackCount}`,
        `teacher_feedback_budgeted_out=${input.sparseFeedback.budgetedOutFeedbackCount}`,
        `teacher_background_amplified=${input.sparseFeedback.amplifiedBackgroundLabelCount}`,
        `teacher_noop=${input.noOpReason}`,
        `teacher_labeler=${input.teacherLabeler?.status ?? "disabled"}`,
        `teacher_labeler_detail=${input.teacherLabeler?.detail ?? "disabled"}`,
        input.materialization === null ? "teacher_materialization=noop" : `teacher_materialized_pack=${input.materialization.candidate.summary.packId}`
    ];
}
function cloneAlwaysOnLearningMaterializationJobOrNull(value) {
    return value === null ? null : structuredClone(value);
}
function cloneTeacherSupervisionArtifacts(value) {
    return [...structuredClone(value)];
}
function cloneAsyncTeacherSnapshotState(value) {
    if (value === undefined) {
        return undefined;
    }
    const interactionEvents = Array.isArray(value.interactionEvents) ? value.interactionEvents : [];
    const feedbackEvents = Array.isArray(value.feedbackEvents) ? value.feedbackEvents : [];
    const seenExportDigests = Array.isArray(value.seenExportDigests) ? value.seenExportDigests : [];
    return {
        interactionEvents: [...structuredClone(interactionEvents)],
        feedbackEvents: [...structuredClone(feedbackEvents)],
        seenExportDigests: [...seenExportDigests]
    };
}
function cloneAsyncTeacherSnapshotRuntime(value) {
    return value === undefined ? undefined : { ...value };
}
function cloneCanonicalSupervision(value) {
    return structuredClone(value);
}
function cloneContinuousProductLoopPackVersion(value) {
    return structuredClone(value);
}
function cloneContinuousProductLoopState(value) {
    return structuredClone(value);
}
function buildNormalizedEventDedupId(event) {
    return checksumJsonPayload({
        contract: event.contract,
        eventId: event.eventId,
        agentId: event.agentId,
        sessionId: event.sessionId,
        channel: event.channel,
        sequence: event.sequence,
        kind: event.kind,
        createdAt: event.createdAt,
        source: event.source,
        packId: "packId" in event ? event.packId ?? null : null,
        content: "content" in event ? event.content : null,
        messageId: event.messageId ?? null,
        relatedInteractionId: "relatedInteractionId" in event ? event.relatedInteractionId ?? null : null
    });
}
function mergeRuntimeEventHistory(current, incoming) {
    const merged = sortNormalizedEvents([
        ...current.interactionEvents,
        ...current.feedbackEvents,
        ...incoming.interactionEvents,
        ...incoming.feedbackEvents
    ]);
    const deduped = [];
    const seen = new Set();
    for (const event of merged) {
        const dedupId = buildNormalizedEventDedupId(event);
        if (seen.has(dedupId)) {
            continue;
        }
        seen.add(dedupId);
        deduped.push(event);
    }
    return {
        interactionEvents: deduped.filter((event) => event.contract === CONTRACT_IDS.interactionEvents),
        feedbackEvents: deduped.filter((event) => event.contract === CONTRACT_IDS.feedbackEvents)
    };
}
function buildContinuousTurnExport(turn, loopRoot) {
    const exportSeed = checksumJsonPayload({
        sessionId: turn.sessionId,
        channel: turn.channel,
        userId: turn.userId ?? null,
        sourceStream: turn.sourceStream ?? null,
        userMessage: turn.userMessage,
        createdAt: turn.createdAt ?? null,
        sequenceStart: turn.sequenceStart ?? null,
        compileCreatedAt: turn.compile?.createdAt ?? null,
        delivery: turn.delivery === false
            ? false
            : turn.delivery === undefined || turn.delivery === null
                ? null
                : turn.delivery === true
                    ? true
                    : {
                        createdAt: turn.delivery.createdAt ?? null,
                        messageId: turn.delivery.messageId ?? null,
                        sequence: turn.delivery.sequence ?? null
                    },
        feedback: (turn.feedback ?? [])
            .filter((item) => item !== null)
            .map((item) => ({
            content: item.content,
            actorName: item.actorName ?? null,
            createdAt: item.createdAt ?? null,
            priorityHint: item.priorityHint ?? null,
            sequence: item.sequence ?? null,
            kind: item.kind ?? null,
            messageId: item.messageId ?? null,
            relatedInteractionId: item.relatedInteractionId ?? null
        }))
    })
        .replace(/^sha256-/u, "")
        .slice(0, 12);
    const exportName = `${turn.sessionId}-${exportSeed}`;
    return {
        rootDir: path.join(loopRoot, "event-exports", exportName),
        exportName
    };
}
function withContinuousTurnExport(turn, loopRoot) {
    if (turn.export !== undefined && turn.export !== null) {
        return {
            ...turn,
            export: {
                ...turn.export
            }
        };
    }
    return {
        ...turn,
        export: buildContinuousTurnExport(turn, loopRoot)
    };
}
function buildPackVersion(version, target) {
    return {
        version,
        packId: target.packId,
        routePolicy: target.routePolicy,
        routerIdentity: target.routerIdentity,
        workspaceSnapshot: target.workspaceSnapshot,
        workspaceRevision: target.workspaceRevision,
        eventRange: {
            start: target.eventRange.start,
            end: target.eventRange.end,
            count: target.eventRange.count
        },
        eventExportDigest: target.eventExportDigest,
        builtAt: target.builtAt
    };
}
function buildLearningCandidateTarget(candidate) {
    return {
        packId: candidate.summary.packId,
        routePolicy: candidate.summary.routePolicy,
        routerIdentity: candidate.payloads.router?.routerIdentity ?? null,
        workspaceSnapshot: candidate.summary.workspaceSnapshot,
        workspaceRevision: candidate.manifest.provenance.workspace.revision,
        eventRange: {
            start: candidate.summary.eventRange.start,
            end: candidate.summary.eventRange.end,
            count: candidate.summary.eventRange.count
        },
        eventExportDigest: candidate.summary.eventExportDigest,
        builtAt: candidate.manifest.provenance.builtAt
    };
}
function registerPackVersion(state, target) {
    const existing = state.packLineage.find((entry) => entry.packId === target.packId);
    if (existing !== undefined) {
        return cloneContinuousProductLoopPackVersion(existing);
    }
    const created = buildPackVersion(state.nextPackVersion, target);
    state.packLineage.push(cloneContinuousProductLoopPackVersion(created));
    state.nextPackVersion += 1;
    return created;
}
function tryReadActivePackTarget(rootDir) {
    try {
        return describeActivationTarget(rootDir, "active", { requireActivationReady: true });
    }
    catch {
        return null;
    }
}
function syncContinuousActivePack(state) {
    const activeTarget = tryReadActivePackTarget(state.activationRoot);
    if (activeTarget === null) {
        state.currentActivePack = null;
        state.activePackVersion = 0;
        return null;
    }
    const activePack = registerPackVersion(state, activeTarget);
    state.currentActivePack = cloneContinuousProductLoopPackVersion(activePack);
    state.activePackVersion = activePack.version;
    return activePack;
}
function buildContinuousPackRoot(loopRoot, packVersion) {
    return path.join(loopRoot, "packs", `v${String(packVersion.version).padStart(4, "0")}-${packVersion.packId}`);
}
export function buildCanonicalSupervision(normalizedEventExport) {
    const feedback = normalizedEventExport.feedbackEvents.map((event) => ({
        eventId: event.eventId,
        kind: event.kind,
        sequence: event.sequence,
        createdAt: event.createdAt,
        content: event.content,
        relatedInteractionId: event.relatedInteractionId ?? null
    }));
    const compilePackIds = [
        ...new Set(normalizedEventExport.interactionEvents.flatMap((event) => event.kind === "memory_compiled" && event.packId ? [event.packId] : []))
    ];
    const relatedInteractionIds = [...new Set(feedback.flatMap((event) => (event.relatedInteractionId ? [event.relatedInteractionId] : [])))];
    const feedbackCounts = {
        corrections: feedback.filter((event) => event.kind === "correction").length,
        teachings: feedback.filter((event) => event.kind === "teaching").length,
        approvals: feedback.filter((event) => event.kind === "approval").length,
        suppressions: feedback.filter((event) => event.kind === "suppression").length
    };
    const supervisionDigest = checksumJsonPayload({
        exportDigest: normalizedEventExport.provenance.exportDigest,
        eventRange: {
            start: normalizedEventExport.range.start,
            end: normalizedEventExport.range.end,
            count: normalizedEventExport.range.count
        },
        sourceStreams: normalizedEventExport.provenance.sourceStreams,
        humanLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.humanLabels,
        selfLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.selfLabels,
        feedback,
        compilePackIds,
        relatedInteractionIds
    });
    return {
        runtimeOwner: "openclaw",
        exportDigest: normalizedEventExport.provenance.exportDigest,
        supervisionDigest,
        sessionId: normalizedEventExport.provenance.sessionId,
        channel: normalizedEventExport.provenance.channel,
        eventRange: {
            start: normalizedEventExport.range.start,
            end: normalizedEventExport.range.end,
            count: normalizedEventExport.range.count
        },
        sourceStreams: [...normalizedEventExport.provenance.sourceStreams],
        humanLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.humanLabels,
        selfLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.selfLabels,
        feedbackCounts,
        compilePackIds,
        relatedInteractionIds,
        feedback
    };
}
export function createContinuousProductLoopState(input) {
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const loopRoot = path.resolve(normalizeNonEmptyString(input.loopRoot, "loopRoot"));
    const activeTarget = tryReadActivePackTarget(activationRoot);
    const activePack = activeTarget === null ? null : buildPackVersion(1, activeTarget);
    return {
        runtimeOwner: "openclaw",
        activationRoot,
        loopRoot,
        interactionEvents: [],
        feedbackEvents: [],
        learner: createAlwaysOnLearningRuntimeState(),
        runtimePlasticity: null,
        activePackVersion: activePack?.version ?? 0,
        currentActivePack: activePack === null ? null : cloneContinuousProductLoopPackVersion(activePack),
        candidatePack: null,
        packLineage: activePack === null ? [] : [cloneContinuousProductLoopPackVersion(activePack)],
        nextPackVersion: activePack === null ? 1 : 2,
        promotionCount: 0,
        lastSupervision: null
    };
}
function mergeUniqueEvents(current, additions) {
    const merged = new Map();
    for (const event of [...current, ...additions]) {
        merged.set(buildNormalizedEventDedupId(event), structuredClone(event));
    }
    return [...merged.values()].sort((left, right) => left.sequence - right.sequence || left.createdAt.localeCompare(right.createdAt));
}
function mergeTeacherArtifacts(current, additions) {
    const merged = new Map();
    for (const artifact of [...current, ...additions]) {
        const existing = merged.get(artifact.dedupId);
        if (existing === undefined ||
            Date.parse(artifact.freshness.observedAt) > Date.parse(existing.freshness.observedAt) ||
            (artifact.freshness.observedAt === existing.freshness.observedAt && artifact.artifactId.localeCompare(existing.artifactId) < 0)) {
            merged.set(artifact.dedupId, structuredClone(artifact));
        }
    }
    return [...merged.values()].sort((left, right) => {
        if (left.freshness.status !== right.freshness.status) {
            return left.freshness.status === "fresh" ? -1 : 1;
        }
        if (left.createdAt !== right.createdAt) {
            return Date.parse(right.createdAt) - Date.parse(left.createdAt);
        }
        return left.artifactId.localeCompare(right.artifactId);
    });
}
function latestTeacherFreshness(artifacts) {
    return artifacts[0]?.freshness.status ?? "none";
}
export class AsyncTeacherLiveLoop {
    input;
    queueCapacity;
    staleAfterMs;
    teacherLabeler;
    queuedExportDigests = new Set();
    seenExportDigests = new Set();
    queue = [];
    drainPromise = null;
    interactionEvents = [];
    feedbackEvents = [];
    teacherArtifacts = [];
    learnerState = createAlwaysOnLearningRuntimeState();
    lastMaterialization = null;
    lastTeacherLabelerResult = null;
    diagnostics = {
        acceptedExportCount: 0,
        processedExportCount: 0,
        duplicateExportCount: 0,
        droppedExportCount: 0,
        emittedArtifactCount: 0,
        dedupedArtifactCount: 0,
        lastProcessedAt: null,
        latestFreshness: "none",
        lastNoOpReason: "none",
        notes: buildAsyncTeacherLoopNotes({
            queueDepth: 0,
            latestFreshness: "none",
            artifactCount: 0,
            emittedArtifactCount: 0,
            dedupedArtifactCount: 0,
            sparseFeedback: this.learnerState.sparseFeedback,
            noOpReason: "none",
            materialization: null,
            teacherLabeler: null
        })
    };
    constructor(input) {
        this.input = input;
        this.queueCapacity = input.maxQueuedExports ?? DEFAULT_ASYNC_TEACHER_QUEUE_CAPACITY;
        this.staleAfterMs = input.staleAfterMs ?? DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS;
        this.teacherLabeler = createTeacherLabeler(input.teacherLabeler);
        if (!Number.isInteger(this.queueCapacity) || this.queueCapacity <= 0) {
            throw new Error("maxQueuedExports must be a positive integer");
        }
        if (!Number.isInteger(this.staleAfterMs) || this.staleAfterMs <= 0) {
            throw new Error("staleAfterMs must be a positive integer");
        }
        const resumedSnapshot = input.resumeFromSnapshot;
        if (resumedSnapshot !== undefined && resumedSnapshot !== null) {
            if (resumedSnapshot.runtimeOwner !== "openclaw") {
                throw new Error("async teacher resume snapshot runtimeOwner must be openclaw");
            }
            this.interactionEvents = [...structuredClone(resumedSnapshot.state?.interactionEvents ?? [])];
            this.feedbackEvents = [...structuredClone(resumedSnapshot.state?.feedbackEvents ?? [])];
            this.teacherArtifacts = cloneTeacherSupervisionArtifacts(resumedSnapshot.teacher.artifacts);
            this.learnerState = structuredClone(resumedSnapshot.learner.state);
            this.lastMaterialization = cloneAlwaysOnLearningMaterializationJobOrNull(resumedSnapshot.learner.lastMaterialization);
            this.diagnostics = {
                ...structuredClone(resumedSnapshot.diagnostics),
                notes: [...resumedSnapshot.diagnostics.notes]
            };
            for (const exportDigest of resumedSnapshot.state?.seenExportDigests ?? []) {
                this.seenExportDigests.add(exportDigest);
            }
            this.refreshNotes();
        }
    }
    enqueueNormalizedEventExport(normalizedEventExport, options = {}) {
        const validationErrors = validateNormalizedEventExport(normalizedEventExport);
        if (validationErrors.length > 0) {
            throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
        }
        const exportDigest = normalizedEventExport.provenance.exportDigest;
        if (this.seenExportDigests.has(exportDigest) || this.queuedExportDigests.has(exportDigest)) {
            this.diagnostics.duplicateExportCount += 1;
            this.diagnostics.lastNoOpReason = "duplicate_export";
            this.refreshNotes();
            return {
                accepted: false,
                exportDigest,
                queueDepth: this.queue.length,
                notes: [...this.diagnostics.notes],
                reason: "duplicate_export"
            };
        }
        if (this.queue.length >= this.queueCapacity) {
            this.diagnostics.droppedExportCount += 1;
            this.diagnostics.lastNoOpReason = "queue_full";
            this.refreshNotes();
            return {
                accepted: false,
                exportDigest,
                queueDepth: this.queue.length,
                notes: [...this.diagnostics.notes],
                reason: "queue_full"
            };
        }
        const observedAt = options.observedAt ?? normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString();
        this.queue.push({
            jobId: `teacher-loop-${createHash("sha256").update(`${exportDigest}:${observedAt}`).digest("hex")}`,
            exportDigest,
            observedAt,
            normalizedEventExport: structuredClone(normalizedEventExport)
        });
        this.queuedExportDigests.add(exportDigest);
        this.diagnostics.acceptedExportCount += 1;
        this.refreshNotes();
        void this.ensureDrain().catch(() => {
            // Explicit flush()/ingest callers observe the failure; the background kickoff must not leak an unhandled rejection.
        });
        return {
            accepted: true,
            exportDigest,
            queueDepth: this.queue.length,
            notes: [...this.diagnostics.notes],
            reason: null
        };
    }
    enqueueScannedEventExport(scannedEventExport, options = {}) {
        const built = buildNormalizedEventExportFromScannedEvents(scannedEventExport);
        if (!built.ok) {
            return {
                accepted: false,
                exportDigest: null,
                queueDepth: this.queue.length,
                notes: [...this.diagnostics.notes],
                reason: built.reason,
                warnings: [...built.warnings],
                error: built.error,
                scanner: cloneScannerExportManifest(built.scanner)
            };
        }
        const enqueue = this.enqueueNormalizedEventExport(built.normalizedEventExport, {
            observedAt: options.observedAt ?? built.scanner.producedAt
        });
        return {
            accepted: enqueue.accepted,
            exportDigest: enqueue.exportDigest,
            queueDepth: enqueue.queueDepth,
            notes: [...enqueue.notes],
            reason: enqueue.reason,
            warnings: [...built.warnings],
            error: null,
            scanner: cloneScannerExportManifest(built.scanner)
        };
    }
    async ingestRuntimeEventExportScannerScan(scan) {
        if (scan.selected.length === 0) {
            this.diagnostics.lastNoOpReason = "empty_scan";
            this.refreshNotes();
            const snapshot = this.snapshot();
            return {
                runtimeOwner: "openclaw",
                scanRoot: scan.scanRoot,
                scannedAt: scan.scannedAt,
                selectedCount: 0,
                acceptedCount: 0,
                duplicateCount: 0,
                droppedCount: 0,
                liveAcceptedCount: 0,
                backfillAcceptedCount: 0,
                duplicateScannerDigestCount: scan.duplicateExportDigests.length,
                staleSkippedCount: scan.staleSkippedExportDigests.length,
                invalidBundleCount: scan.invalidBundles.length,
                noOpReason: "empty_scan",
                notes: [...snapshot.diagnostics.notes],
                results: [],
                snapshot
            };
        }
        const results = [];
        let acceptedCount = 0;
        let duplicateCount = 0;
        let droppedCount = 0;
        let liveAcceptedCount = 0;
        let backfillAcceptedCount = 0;
        for (const hit of scan.selected) {
            let enqueue = this.enqueueNormalizedEventExport(hit.normalizedEventExport, {
                observedAt: hit.exportedAt
            });
            if (!enqueue.accepted && enqueue.reason === "queue_full") {
                await this.flush();
                enqueue = this.enqueueNormalizedEventExport(hit.normalizedEventExport, {
                    observedAt: hit.exportedAt
                });
            }
            if (enqueue.accepted) {
                acceptedCount += 1;
                if (hit.lane === "live") {
                    liveAcceptedCount += 1;
                }
                else {
                    backfillAcceptedCount += 1;
                }
            }
            else if (enqueue.reason === "duplicate_export") {
                duplicateCount += 1;
            }
            else if (enqueue.reason === "queue_full") {
                droppedCount += 1;
            }
            results.push({
                lane: hit.lane,
                exportDigest: hit.exportDigest,
                exportName: hit.exportName,
                exportedAt: hit.exportedAt,
                eventRange: {
                    start: hit.eventRange.start,
                    end: hit.eventRange.end,
                    count: hit.eventRange.count
                },
                accepted: enqueue.accepted,
                queueDepth: enqueue.queueDepth,
                reason: enqueue.reason
            });
        }
        const snapshot = acceptedCount > 0 ? await this.flush() : this.snapshot();
        const noOpReason = acceptedCount > 0
            ? "none"
            : duplicateCount === scan.selected.length
                ? "duplicate_exports"
                : "queue_full";
        return {
            runtimeOwner: "openclaw",
            scanRoot: scan.scanRoot,
            scannedAt: scan.scannedAt,
            selectedCount: scan.selected.length,
            acceptedCount,
            duplicateCount,
            droppedCount,
            liveAcceptedCount,
            backfillAcceptedCount,
            duplicateScannerDigestCount: scan.duplicateExportDigests.length,
            staleSkippedCount: scan.staleSkippedExportDigests.length,
            invalidBundleCount: scan.invalidBundles.length,
            noOpReason,
            notes: [...snapshot.diagnostics.notes],
            results,
            snapshot
        };
    }
    async flush() {
        await this.ensureDrain();
        return this.snapshot();
    }
    snapshot() {
        return {
            runtimeOwner: "openclaw",
            queue: {
                capacity: this.queueCapacity,
                depth: this.queue.length,
                running: this.drainPromise !== null
            },
            teacher: {
                artifactCount: this.teacherArtifacts.length,
                artifacts: cloneTeacherSupervisionArtifacts(this.teacherArtifacts),
                latestFreshness: this.diagnostics.latestFreshness
            },
            learner: {
                state: structuredClone(this.learnerState),
                lastMaterialization: cloneAlwaysOnLearningMaterializationJobOrNull(this.lastMaterialization)
            },
            diagnostics: {
                ...this.diagnostics,
                notes: [...this.diagnostics.notes]
            },
            state: {
                interactionEvents: [...structuredClone(this.interactionEvents)],
                feedbackEvents: [...structuredClone(this.feedbackEvents)],
                seenExportDigests: [...this.seenExportDigests].sort()
            }
        };
    }
    async ensureDrain() {
        if (this.drainPromise === null) {
            this.drainPromise = this.drain().finally(() => {
                this.drainPromise = null;
            });
        }
        await this.drainPromise;
        if (this.queue.length > 0) {
            await this.ensureDrain();
        }
    }
    async drain() {
        while (this.queue.length > 0) {
            const job = this.queue.shift();
            this.queuedExportDigests.delete(job.exportDigest);
            const previousInteractionEvents = [...structuredClone(this.interactionEvents)];
            const previousFeedbackEvents = [...structuredClone(this.feedbackEvents)];
            const previousTeacherArtifacts = cloneTeacherSupervisionArtifacts(this.teacherArtifacts);
            const previousLearnerState = structuredClone(this.learnerState);
            const previousLastMaterialization = cloneAlwaysOnLearningMaterializationJobOrNull(this.lastMaterialization);
            const previousDiagnostics = {
                ...structuredClone(this.diagnostics),
                notes: [...this.diagnostics.notes]
            };
            const previousSeenExportDigests = [...this.seenExportDigests];
            try {
                this.seenExportDigests.add(job.exportDigest);
                this.interactionEvents = mergeUniqueEvents(this.interactionEvents, job.normalizedEventExport.interactionEvents);
                this.feedbackEvents = mergeUniqueEvents(this.feedbackEvents, job.normalizedEventExport.feedbackEvents);
                const mergedNormalizedEventExport = buildNormalizedEventExport({
                    interactionEvents: this.interactionEvents,
                    feedbackEvents: this.feedbackEvents
                });
                const learnedRoutingState = this.input.resolveLearnedRoutingState?.() ?? {};
                const builtArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
                    normalizedEventExport: mergedNormalizedEventExport,
                    observedAt: job.observedAt,
                    staleAfterMs: this.staleAfterMs,
                    ...(this.input.sparseFeedback !== undefined ? { sparseFeedback: this.input.sparseFeedback } : {})
                });
                let generatedTeacherArtifacts = [];
                if (this.teacherLabeler !== null) {
                    try {
                        this.lastTeacherLabelerResult = await this.teacherLabeler.label({
                            normalizedEventExport: mergedNormalizedEventExport,
                            observedAt: job.observedAt,
                            staleAfterMs: this.staleAfterMs,
                            existingArtifacts: [...this.teacherArtifacts, ...builtArtifacts],
                            ...(learnedRoutingState.serveTimeDecisions !== undefined
                                ? { serveTimeDecisions: learnedRoutingState.serveTimeDecisions }
                                : {})
                        });
                        generatedTeacherArtifacts = this.lastTeacherLabelerResult.artifacts;
                    }
                    catch (error) {
                        this.lastTeacherLabelerResult = {
                            artifacts: [],
                            status: "fail_open",
                            detail: toErrorMessage(error)
                        };
                    }
                }
                const nextBuiltArtifacts = mergeTeacherArtifacts([], [...builtArtifacts, ...generatedTeacherArtifacts]);
                const currentDedupIds = new Set(this.teacherArtifacts.map((artifact) => artifact.dedupId));
                const nextTeacherArtifacts = mergeTeacherArtifacts(this.teacherArtifacts, nextBuiltArtifacts);
                const emittedArtifactCount = nextBuiltArtifacts.filter((artifact) => !currentDedupIds.has(artifact.dedupId)).length;
                const dedupedArtifactCount = nextBuiltArtifacts.length - emittedArtifactCount;
                this.teacherArtifacts = nextTeacherArtifacts;
                const learnerResult = advanceAlwaysOnLearningRuntime({
                    packLabel: this.input.packLabel,
                    workspace: this.input.workspace,
                    interactionEvents: this.interactionEvents,
                    feedbackEvents: this.feedbackEvents,
                    teacherSupervisionArtifacts: this.teacherArtifacts,
                    learnedRouting: this.input.learnedRouting,
                    state: this.learnerState,
                    builtAt: this.input.builtAt ?? job.observedAt,
                    ...(this.input.offlineArtifacts !== undefined ? { offlineArtifacts: this.input.offlineArtifacts } : {}),
                    ...(this.input.structuralOps !== undefined ? { structuralOps: this.input.structuralOps } : {}),
                    ...(this.input.sparseFeedback !== undefined ? { sparseFeedback: this.input.sparseFeedback } : {}),
                    ...(this.input.liveSliceSize !== undefined ? { liveSliceSize: this.input.liveSliceSize } : {}),
                    ...(this.input.backfillSliceSize !== undefined ? { backfillSliceSize: this.input.backfillSliceSize } : {}),
                    ...(this.input.cadence !== undefined ? { cadence: this.input.cadence } : {}),
                    ...(learnedRoutingState.pgVersion !== undefined ? { pgVersion: learnedRoutingState.pgVersion } : {}),
                    ...(learnedRoutingState.serveTimeDecisions !== undefined ? { serveTimeDecisions: learnedRoutingState.serveTimeDecisions } : {}),
                    ...(learnedRoutingState.baselineState !== undefined ? { baselineState: learnedRoutingState.baselineState } : {}),
                    ...(this.input.activationRoot !== undefined ? { activationRoot: this.input.activationRoot } : {})
                });
                this.learnerState = structuredClone(learnerResult.state);
                this.lastMaterialization = cloneAlwaysOnLearningMaterializationJobOrNull(learnerResult.materialization);
                const updatedBaseline = learnerResult.materialization?.candidate.routingBuild.updatedBaseline ?? null;
                if (updatedBaseline !== null) {
                    this.input.persistUpdatedBaseline?.(structuredClone(updatedBaseline));
                }
                this.diagnostics.processedExportCount += 1;
                this.diagnostics.emittedArtifactCount += emittedArtifactCount;
                this.diagnostics.dedupedArtifactCount += dedupedArtifactCount;
                this.diagnostics.lastProcessedAt = job.observedAt;
                this.diagnostics.latestFreshness = latestTeacherFreshness(this.teacherArtifacts);
                this.diagnostics.lastNoOpReason = emittedArtifactCount === 0 ? "no_teacher_artifacts" : "none";
                this.refreshNotes();
            }
            catch (error) {
                this.interactionEvents = previousInteractionEvents;
                this.feedbackEvents = previousFeedbackEvents;
                this.teacherArtifacts = previousTeacherArtifacts;
                this.learnerState = previousLearnerState;
                this.lastMaterialization = previousLastMaterialization;
                this.diagnostics = previousDiagnostics;
                this.seenExportDigests.clear();
                for (const exportDigest of previousSeenExportDigests) {
                    this.seenExportDigests.add(exportDigest);
                }
                this.refreshNotes();
                throw error;
            }
        }
    }
    refreshNotes() {
        this.diagnostics.notes = buildAsyncTeacherLoopNotes({
            queueDepth: this.queue.length,
            latestFreshness: this.diagnostics.latestFreshness,
            artifactCount: this.teacherArtifacts.length,
            emittedArtifactCount: this.diagnostics.emittedArtifactCount,
            dedupedArtifactCount: this.diagnostics.dedupedArtifactCount,
            sparseFeedback: this.learnerState.sparseFeedback,
            noOpReason: this.diagnostics.lastNoOpReason,
            materialization: this.lastMaterialization,
            teacherLabeler: this.lastTeacherLabelerResult
        });
    }
}
export function createAsyncTeacherLiveLoop(input) {
    return new AsyncTeacherLiveLoop(input);
}
function normalizeScannerStringArray(value, fieldName) {
    if (!Array.isArray(value)) {
        throw new Error(`${fieldName} must be an array of strings`);
    }
    return value.map((entry, index) => normalizeNonEmptyString(entry, `${fieldName}[${index}]`));
}
function normalizeScannerLiveWorkspaceInput(workspace) {
    const normalized = {
        workspaceId: normalizeNonEmptyString(workspace.workspaceId, "workspace.workspaceId"),
        snapshotId: normalizeNonEmptyString(workspace.snapshotId, "workspace.snapshotId"),
        capturedAt: normalizeIsoTimestamp(workspace.capturedAt, "workspace.capturedAt"),
        rootDir: path.resolve(normalizeNonEmptyString(workspace.rootDir, "workspace.rootDir")),
        branch: normalizeOptionalString(workspace.branch) ?? null,
        revision: normalizeOptionalString(workspace.revision) ?? null,
        dirty: workspace.dirty === true,
        manifestDigest: normalizeOptionalString(workspace.manifestDigest) ?? null
    };
    if (workspace.labels !== undefined) {
        normalized.labels = normalizeScannerStringArray(workspace.labels, "workspace.labels");
    }
    if (workspace.files !== undefined) {
        normalized.files = normalizeScannerStringArray(workspace.files, "workspace.files");
    }
    return normalized;
}
export function scanRecordedSession(input) {
    const rootDir = path.resolve(normalizeNonEmptyString(input.rootDir, "rootDir"));
    const fixture = buildRecordedSessionReplayFixture(structuredClone(input.trace));
    const bundle = runRecordedSessionReplay(rootDir, fixture);
    return {
        runtimeOwner: "openclaw",
        scanMode: "session",
        rootDir,
        fixtureHash: fixture.fixtureHash,
        bundle
    };
}
export function scanLiveEventExport(input) {
    const normalizedEventExport = structuredClone(input.normalizedEventExport);
    const validationErrors = validateNormalizedEventExport(normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    const workspace = normalizeScannerLiveWorkspaceInput(input.workspace);
    const packLabel = normalizeOptionalString(input.packLabel) ?? "scanner-live-cli";
    const observedAt = normalizeIsoTimestamp(input.observedAt, "observedAt", normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString());
    const teacherArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
        normalizedEventExport,
        observedAt,
        staleAfterMs: input.staleAfterMs ?? DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS
    });
    const learnerResult = advanceAlwaysOnLearningRuntime({
        packLabel,
        workspace,
        interactionEvents: normalizedEventExport.interactionEvents,
        feedbackEvents: normalizedEventExport.feedbackEvents,
        teacherSupervisionArtifacts: teacherArtifacts,
        learnedRouting: input.learnedRouting ?? true,
        state: createAlwaysOnLearningRuntimeState(),
        builtAt: normalizeIsoTimestamp(input.builtAt, "builtAt", observedAt),
        ...(input.activationRoot !== undefined ? { activationRoot: input.activationRoot } : {}),
        ...(input.liveSliceSize !== undefined ? { liveSliceSize: input.liveSliceSize } : {}),
        ...(input.backfillSliceSize !== undefined ? { backfillSliceSize: input.backfillSliceSize } : {})
    });
    const latestFreshness = latestTeacherFreshness(teacherArtifacts);
    const lastNoOpReason = teacherArtifacts.length === 0 ? "no_teacher_artifacts" : "none";
    const snapshot = {
        runtimeOwner: "openclaw",
        queue: {
            capacity: 1,
            depth: 0,
            running: false
        },
        teacher: {
            artifactCount: teacherArtifacts.length,
            artifacts: cloneTeacherSupervisionArtifacts(teacherArtifacts),
            latestFreshness
        },
        learner: {
            state: structuredClone(learnerResult.state),
            lastMaterialization: cloneAlwaysOnLearningMaterializationJobOrNull(learnerResult.materialization)
        },
        diagnostics: {
            acceptedExportCount: 1,
            processedExportCount: 1,
            duplicateExportCount: 0,
            droppedExportCount: 0,
            emittedArtifactCount: teacherArtifacts.length,
            dedupedArtifactCount: 0,
            lastProcessedAt: observedAt,
            latestFreshness,
            lastNoOpReason,
            notes: buildAsyncTeacherLoopNotes({
                queueDepth: 0,
                latestFreshness,
                artifactCount: teacherArtifacts.length,
                emittedArtifactCount: teacherArtifacts.length,
                dedupedArtifactCount: 0,
                sparseFeedback: learnerResult.state.sparseFeedback,
                noOpReason: lastNoOpReason,
                materialization: learnerResult.materialization,
                teacherLabeler: null
            })
        }
    };
    const labelFlow = summarizeNormalizedEventExportLabelFlow(normalizedEventExport, teacherArtifacts.length);
    const learningPath = summarizeLearningPathFromMaterialization(learnerResult.materialization);
    return {
        runtimeOwner: "openclaw",
        scanMode: "live",
        observedAt,
        packLabel,
        supervision: buildCanonicalSupervision(normalizedEventExport),
        snapshot,
        labelFlow,
        learningPath
    };
}
function readJsonFile(filePath) {
    return JSON.parse(readFileSync(filePath, "utf8"));
}
export function resolveAsyncTeacherLiveLoopSnapshotPath(activationRoot) {
    return path.join(path.resolve(normalizeNonEmptyString(activationRoot, "activationRoot")), "async-teacher-live-loop.snapshot.json");
}
export const WATCH_STATE_DIRNAME = "watch";
export const WATCH_SESSION_TAIL_CURSOR_BASENAME = "session-tail-cursor.json";
export const WATCH_TEACHER_SNAPSHOT_BASENAME = "teacher-snapshot.json";
export const DEFAULT_WATCH_POLL_INTERVAL_SECONDS = 30;
function isAsyncTeacherLiveLoopSnapshot(value) {
    if (value === null || typeof value !== "object") {
        return false;
    }
    const candidate = value;
    return (candidate.runtimeOwner === "openclaw" &&
        candidate.queue !== undefined &&
        candidate.teacher !== undefined &&
        candidate.learner !== undefined &&
        candidate.diagnostics !== undefined);
}
function isWatchTeacherSnapshot(value) {
    if (value === null || typeof value !== "object") {
        return false;
    }
    const candidate = value;
    return (candidate.contract === "openclaw_watch_teacher_snapshot.v1" &&
        candidate.runtimeOwner === "openclaw" &&
        isAsyncTeacherLiveLoopSnapshot(candidate.snapshot));
}
function cloneWatchTeacherSnapshotFailure(value) {
    if (value === null ||
        value === undefined ||
        !["materialization_failed", "teacher_fail_open"].includes(value.mode) ||
        typeof value.detail !== "string" ||
        typeof value.at !== "string") {
        return null;
    }
    return {
        mode: value.mode,
        detail: value.detail,
        at: value.at
    };
}
function buildUnavailableLastObservedDelta(explanation) {
    return {
        available: false,
        observedAt: null,
        exported: null,
        labeled: null,
        promoted: null,
        served: null,
        latestPackTransition: null,
        explanation
    };
}
function cloneLastObservedDelta(value) {
    if (value === null || value === undefined || typeof value !== "object") {
        return buildUnavailableLastObservedDelta("last observed delta is unavailable");
    }
    const transition = value.latestPackTransition;
    const latestPackTransition = transition !== null &&
        transition !== undefined &&
        (transition.kind === "staged_candidate" || transition.kind === "promoted_active") &&
        typeof transition.toPackId === "string" &&
        transition.toPackId.length > 0 &&
        (transition.fromPackId === null || (typeof transition.fromPackId === "string" && transition.fromPackId.length > 0))
        ? {
            kind: transition.kind,
            fromPackId: transition.fromPackId,
            toPackId: transition.toPackId
        }
        : null;
    const available = value.available === true;
    return {
        available,
        observedAt: typeof value.observedAt === "string" ? value.observedAt : null,
        exported: available && typeof value.exported === "boolean" ? value.exported : null,
        labeled: available && typeof value.labeled === "boolean" ? value.labeled : null,
        promoted: available && typeof value.promoted === "boolean" ? value.promoted : null,
        served: available && typeof value.served === "boolean" ? value.served : null,
        latestPackTransition,
        explanation: typeof value.explanation === "string" && value.explanation.trim().length > 0
            ? value.explanation
            : "last observed delta is unavailable"
    };
}
function cloneWatchEmbedInstrumentationPoint(value) {
    if (value === null || value === undefined || typeof value !== "object") {
        return null;
    }
    const candidate = value;
    if (candidate.slot !== null &&
        candidate.slot !== "candidate" &&
        candidate.slot !== "active") {
        return null;
    }
    return {
        slot: candidate.slot ?? null,
        packId: typeof candidate.packId === "string" ? candidate.packId : null,
        runtimeEmbedderPresent: candidate.runtimeEmbedderPresent === true,
        runtimeEmbedderModel: typeof candidate.runtimeEmbedderModel === "string" ? candidate.runtimeEmbedderModel : null,
        vectorEntryCount: typeof candidate.vectorEntryCount === "number" ? candidate.vectorEntryCount : null,
        numericEmbeddingEntryCount: typeof candidate.numericEmbeddingEntryCount === "number" ? candidate.numericEmbeddingEntryCount : null,
        embeddingModels: Array.isArray(candidate.embeddingModels)
            ? candidate.embeddingModels.filter((model) => typeof model === "string")
            : [],
        error: typeof candidate.error === "string" ? candidate.error : null
    };
}
function cloneWatchEmbedInstrumentationTrace(value) {
    if (value === null || value === undefined || typeof value !== "object") {
        return null;
    }
    const candidate = value;
    const beforeCandidateMaterialization = cloneWatchEmbedInstrumentationPoint(candidate.beforeCandidateMaterialization);
    if (beforeCandidateMaterialization === null || typeof candidate.observedAt !== "string") {
        return null;
    }
    return {
        observedAt: candidate.observedAt,
        candidatePackId: typeof candidate.candidatePackId === "string" ? candidate.candidatePackId : null,
        promotionAllowed: typeof candidate.promotionAllowed === "boolean" ? candidate.promotionAllowed : null,
        promotionFindings: Array.isArray(candidate.promotionFindings)
            ? candidate.promotionFindings.filter((finding) => typeof finding === "string")
            : [],
        beforeCandidateMaterialization,
        afterCandidateMaterialization: cloneWatchEmbedInstrumentationPoint(candidate.afterCandidateMaterialization),
        afterStage: cloneWatchEmbedInstrumentationPoint(candidate.afterStage),
        afterPromote: cloneWatchEmbedInstrumentationPoint(candidate.afterPromote)
    };
}
function cloneRuntimeEventExportScannerCheckpoint(value) {
    return structuredClone(value);
}
function buildWatchTeacherSnapshotTeacherSummary(snapshot) {
    return {
        artifactCount: snapshot.teacher.artifactCount,
        latestFreshness: snapshot.teacher.latestFreshness,
        acceptedExportCount: snapshot.diagnostics.acceptedExportCount,
        processedExportCount: snapshot.diagnostics.processedExportCount,
        duplicateExportCount: snapshot.diagnostics.duplicateExportCount,
        droppedExportCount: snapshot.diagnostics.droppedExportCount,
        emittedArtifactCount: snapshot.diagnostics.emittedArtifactCount,
        dedupedArtifactCount: snapshot.diagnostics.dedupedArtifactCount,
        lastProcessedAt: snapshot.diagnostics.lastProcessedAt,
        lastNoOpReason: snapshot.diagnostics.lastNoOpReason,
        queueDepth: snapshot.queue.depth,
        queueCapacity: snapshot.queue.capacity,
        running: snapshot.queue.running,
        lastAppliedMaterializationJobId: snapshot.runtime?.lastAppliedMaterializationJobId ?? snapshot.learner.lastMaterialization?.jobId ?? null,
        lastMaterializedPackId: snapshot.learner.lastMaterialization?.candidate.summary.packId ?? null
    };
}
function buildWatchTeacherSnapshotLearningSummary(snapshot, lastHandledMaterializationPackId) {
    const plan = describeAlwaysOnLearningRuntimeState(snapshot.learner.state, snapshot.learner.lastMaterialization);
    return {
        bootstrapped: plan.bootstrapped,
        mode: plan.mode,
        nextPriorityLane: plan.nextPriorityLane,
        nextPriorityBucket: plan.nextPriorityBucket,
        pendingLive: plan.pending.live,
        pendingBackfill: plan.pending.backfill,
        pendingTotal: plan.pending.total,
        pendingByBucket: { ...plan.pending.byBucket },
        materializationCount: plan.materialization.count,
        lastMaterializedAt: plan.materialization.lastMaterializedAt,
        lastMaterializationReason: plan.materialization.lastReason,
        lastMaterializationLane: plan.materialization.lastLane,
        lastMaterializedPackId: snapshot.learner.lastMaterialization?.candidate.summary.packId ?? null,
        lastHandledMaterializationPackId
    };
}
function buildWatchTeacherSnapshotLabelingSummary(snapshot) {
    const learningSurface = snapshot.learner.lastMaterialization?.candidate.summary.learningSurface ??
        snapshot.learner.state.learnedEventExport?.provenance.learningSurface ??
        null;
    return {
        learningCadence: learningSurface?.learningCadence ?? "passive_background",
        scanPolicy: learningSurface?.scanPolicy ?? "always_on",
        liveSlicesPerCycle: 1,
        backfillSlicesPerCycle: 1,
        teacherBudget: snapshot.learner.state.sparseFeedback.teacherBudget,
        teacherDelayMs: snapshot.learner.state.sparseFeedback.teacherDelayMs,
        backgroundLabelAmplification: snapshot.learner.state.sparseFeedback.backgroundLabelAmplification
    };
}
function normalizeWatchTeacherSnapshotFromValue(value, snapshot, sourcePath) {
    const activationRoot = path.dirname(path.dirname(sourcePath));
    const defaultScanRoot = path.join(activationRoot, "event-exports");
    return {
        contract: "openclaw_watch_teacher_snapshot.v1",
        runtimeOwner: "openclaw",
        updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : new Date(0).toISOString(),
        lastRunAt: typeof value.lastRunAt === "string"
            ? value.lastRunAt
            : typeof value.updatedAt === "string"
                ? value.updatedAt
                : snapshot.diagnostics.lastProcessedAt ?? new Date(0).toISOString(),
        pollIntervalSeconds: typeof value.pollIntervalSeconds === "number" &&
            Number.isInteger(value.pollIntervalSeconds) &&
            value.pollIntervalSeconds > 0
            ? value.pollIntervalSeconds
            : DEFAULT_WATCH_POLL_INTERVAL_SECONDS,
        scanRoot: typeof value.scanRoot === "string" ? value.scanRoot : defaultScanRoot,
        sessionTailCursorPath: typeof value.sessionTailCursorPath === "string"
            ? value.sessionTailCursorPath
            : resolveWatchSessionTailCursorPath(activationRoot),
        sessionTailCursorUpdatedAt: typeof value.sessionTailCursorUpdatedAt === "string"
            ? value.sessionTailCursorUpdatedAt
            : typeof value.updatedAt === "string"
                ? value.updatedAt
                : new Date(0).toISOString(),
        sessionTailSessionsTracked: typeof value.sessionTailSessionsTracked === "number" ? value.sessionTailSessionsTracked : 0,
        sessionTailBridgedEventCount: typeof value.sessionTailBridgedEventCount === "number" ? value.sessionTailBridgedEventCount : 0,
        scannerCheckpointPath: typeof value.scannerCheckpointPath === "string"
            ? value.scannerCheckpointPath
            : path.join(typeof value.scanRoot === "string" ? value.scanRoot : defaultScanRoot, ".openclawbrain-scanner-checkpoint.json"),
        scannerCheckpoint: value.scannerCheckpoint !== undefined
            ? cloneRuntimeEventExportScannerCheckpoint(value.scannerCheckpoint)
            : createRuntimeEventExportScannerCheckpoint({
                scanRoot: typeof value.scanRoot === "string" ? value.scanRoot : defaultScanRoot
            }),
        replayedBundleCount: typeof value.replayedBundleCount === "number" ? value.replayedBundleCount : 0,
        replayedEventCount: typeof value.replayedEventCount === "number" ? value.replayedEventCount : 0,
        exportedBundleCount: typeof value.exportedBundleCount === "number" ? value.exportedBundleCount : 0,
        exportedEventCount: typeof value.exportedEventCount === "number" ? value.exportedEventCount : 0,
        startupWarnings: Array.isArray(value.startupWarnings)
            ? value.startupWarnings.filter((warning) => typeof warning === "string")
            : [],
        lastTeacherError: typeof value.lastTeacherError === "string"
            ? value.lastTeacherError
            : null,
        localSessionTailNoopReason: typeof value.localSessionTailNoopReason === "string" ? value.localSessionTailNoopReason : null,
        lastHandledMaterializationPackId: typeof value.lastHandledMaterializationPackId === "string" ? value.lastHandledMaterializationPackId : null,
        teacher: value.teacher ?? buildWatchTeacherSnapshotTeacherSummary(snapshot),
        learning: value.learning ?? buildWatchTeacherSnapshotLearningSummary(snapshot, typeof value.lastHandledMaterializationPackId === "string" ? value.lastHandledMaterializationPackId : null),
        labeling: value.labeling ?? buildWatchTeacherSnapshotLabelingSummary(snapshot),
        lastObservedDelta: cloneLastObservedDelta(value.lastObservedDelta),
        embedInstrumentation: cloneWatchEmbedInstrumentationTrace(value.embedInstrumentation),
        failure: cloneWatchTeacherSnapshotFailure(value.failure),
        snapshot
    };
}
export function resolveWatchStateRoot(activationRoot) {
    return path.resolve(normalizeNonEmptyString(activationRoot, "activationRoot"), WATCH_STATE_DIRNAME);
}
export function resolveWatchSessionTailCursorPath(activationRoot) {
    return path.join(resolveWatchStateRoot(activationRoot), WATCH_SESSION_TAIL_CURSOR_BASENAME);
}
export function resolveWatchTeacherSnapshotPath(activationRoot) {
    return path.join(resolveWatchStateRoot(activationRoot), WATCH_TEACHER_SNAPSHOT_BASENAME);
}
export function resolveOperatorTeacherSnapshotPath(activationRoot, explicitPath) {
    if (explicitPath !== null && explicitPath !== undefined) {
        return explicitPath;
    }
    const canonicalWatchSnapshotPath = resolveWatchTeacherSnapshotPath(activationRoot);
    if (existsSync(canonicalWatchSnapshotPath)) {
        return canonicalWatchSnapshotPath;
    }
    const asyncSnapshotPath = resolveAsyncTeacherLiveLoopSnapshotPath(activationRoot);
    return existsSync(asyncSnapshotPath) ? asyncSnapshotPath : null;
}
export function loadTeacherSurface(snapshotPath) {
    const resolvedPath = path.resolve(snapshotPath);
    let parsed;
    try {
        parsed = readJsonFile(resolvedPath);
    }
    catch {
        return null;
    }
    if (isWatchTeacherSnapshot(parsed)) {
        const snapshot = loadAsyncTeacherLiveLoopSnapshotFromValue(parsed.snapshot);
        return {
            sourcePath: resolvedPath,
            sourceKind: "watch_snapshot",
            snapshot,
            watchSnapshot: normalizeWatchTeacherSnapshotFromValue(parsed, snapshot, resolvedPath)
        };
    }
    if (isAsyncTeacherLiveLoopSnapshot(parsed)) {
        return {
            sourcePath: resolvedPath,
            sourceKind: "async_snapshot",
            snapshot: loadAsyncTeacherLiveLoopSnapshotFromValue(parsed),
            watchSnapshot: null
        };
    }
    return null;
}
export function loadWatchTeacherSnapshotState(snapshotPath) {
    const resolvedPath = path.resolve(snapshotPath);
    if (!existsSync(resolvedPath)) {
        return {
            lastHandledMaterializationPackId: null,
            lastObservedDelta: buildUnavailableLastObservedDelta("no watch teacher snapshot is visible for the latest observed cycle"),
            embedInstrumentation: null,
            snapshot: null,
            error: null
        };
    }
    let parsed;
    try {
        parsed = readJsonFile(resolvedPath);
    }
    catch (error) {
        return {
            lastHandledMaterializationPackId: null,
            lastObservedDelta: buildUnavailableLastObservedDelta("watch teacher snapshot could not be loaded"),
            embedInstrumentation: null,
            snapshot: null,
            error: error instanceof Error ? error.message : String(error)
        };
    }
    if (isWatchTeacherSnapshot(parsed)) {
        return {
            lastHandledMaterializationPackId: parsed.lastHandledMaterializationPackId,
            lastObservedDelta: cloneLastObservedDelta(parsed.lastObservedDelta),
            embedInstrumentation: cloneWatchEmbedInstrumentationTrace(parsed.embedInstrumentation),
            snapshot: loadAsyncTeacherLiveLoopSnapshotFromValue(parsed.snapshot),
            error: null
        };
    }
    if (isAsyncTeacherLiveLoopSnapshot(parsed)) {
        return {
            lastHandledMaterializationPackId: null,
            lastObservedDelta: buildUnavailableLastObservedDelta("raw async teacher snapshots do not record the last observed export/label/promotion delta"),
            embedInstrumentation: null,
            snapshot: loadAsyncTeacherLiveLoopSnapshotFromValue(parsed),
            error: null
        };
    }
    return {
        lastHandledMaterializationPackId: null,
        lastObservedDelta: buildUnavailableLastObservedDelta("watch teacher snapshot is invalid"),
        embedInstrumentation: null,
        snapshot: null,
        error: `watch teacher snapshot is invalid: ${resolvedPath}`
    };
}
export function persistWatchTeacherSnapshot(snapshotPath, input) {
    const persistedAt = new Date().toISOString();
    const canonicalSnapshot = loadAsyncTeacherLiveLoopSnapshotFromValue(input.snapshot);
    const resolvedScanRoot = path.resolve(input.scanRoot);
    const canonicalRuntime = canonicalSnapshot.runtime ?? {
        startedAt: input.lastRunAt,
        lastHeartbeatAt: input.lastRunAt,
        lastScanAt: input.lastRunAt,
        scanRoot: resolvedScanRoot,
        lastAppliedMaterializationJobId: null
    };
    canonicalSnapshot.runtime = {
        ...canonicalRuntime,
        startedAt: canonicalRuntime.startedAt ?? input.lastRunAt,
        lastHeartbeatAt: input.lastRunAt,
        lastScanAt: input.lastRunAt,
        scanRoot: typeof canonicalRuntime.scanRoot === "string" && canonicalRuntime.scanRoot.trim().length > 0
            ? canonicalRuntime.scanRoot
            : resolvedScanRoot
    };
    const payload = {
        contract: "openclaw_watch_teacher_snapshot.v1",
        runtimeOwner: "openclaw",
        updatedAt: persistedAt,
        lastRunAt: input.lastRunAt,
        pollIntervalSeconds: input.pollIntervalSeconds,
        scanRoot: resolvedScanRoot,
        sessionTailCursorPath: path.resolve(input.sessionTailCursorPath),
        sessionTailCursorUpdatedAt: input.sessionTailCursorUpdatedAt,
        sessionTailSessionsTracked: input.sessionTailSessionsTracked,
        sessionTailBridgedEventCount: input.sessionTailBridgedEventCount,
        scannerCheckpointPath: path.resolve(input.scannerCheckpointPath),
        scannerCheckpoint: cloneRuntimeEventExportScannerCheckpoint(input.scannerCheckpoint),
        replayedBundleCount: input.replayedBundleCount,
        replayedEventCount: input.replayedEventCount,
        exportedBundleCount: input.exportedBundleCount,
        exportedEventCount: input.exportedEventCount,
        startupWarnings: [...input.startupWarnings],
        lastTeacherError: input.lastTeacherError,
        localSessionTailNoopReason: input.localSessionTailNoopReason,
        lastHandledMaterializationPackId: input.lastHandledMaterializationPackId,
        teacher: buildWatchTeacherSnapshotTeacherSummary(canonicalSnapshot),
        learning: buildWatchTeacherSnapshotLearningSummary(canonicalSnapshot, input.lastHandledMaterializationPackId),
        labeling: buildWatchTeacherSnapshotLabelingSummary(canonicalSnapshot),
        lastObservedDelta: cloneLastObservedDelta(input.lastObservedDelta),
        embedInstrumentation: cloneWatchEmbedInstrumentationTrace(input.embedInstrumentation),
        failure: cloneWatchTeacherSnapshotFailure(input.failure),
        snapshot: canonicalSnapshot
    };
    mkdirSync(path.dirname(snapshotPath), { recursive: true });
    writeFileSync(snapshotPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    return payload;
}
function loadAsyncTeacherLiveLoopSnapshotFromValue(snapshot) {
    if (snapshot.runtimeOwner !== "openclaw") {
        throw new Error("async teacher snapshot runtimeOwner must be openclaw");
    }
    const cloned = {
        ...snapshot,
        diagnostics: {
            ...snapshot.diagnostics,
            notes: [...snapshot.diagnostics.notes]
        },
        teacher: {
            ...snapshot.teacher,
            artifacts: cloneTeacherSupervisionArtifacts(snapshot.teacher.artifacts)
        },
        learner: {
            state: structuredClone(snapshot.learner.state),
            lastMaterialization: cloneAlwaysOnLearningMaterializationJobOrNull(snapshot.learner.lastMaterialization)
        }
    };
    if (snapshot.state !== undefined) {
        cloned.state = cloneAsyncTeacherSnapshotState(snapshot.state);
    }
    if (snapshot.runtime !== undefined) {
        cloned.runtime = cloneAsyncTeacherSnapshotRuntime(snapshot.runtime);
    }
    return cloned;
}
export function loadAsyncTeacherLiveLoopSnapshot(snapshotPath) {
    const resolvedPath = path.resolve(snapshotPath);
    const parsed = readJsonFile(resolvedPath);
    if (isWatchTeacherSnapshot(parsed)) {
        return loadAsyncTeacherLiveLoopSnapshotFromValue(parsed.snapshot);
    }
    if (isAsyncTeacherLiveLoopSnapshot(parsed)) {
        return loadAsyncTeacherLiveLoopSnapshotFromValue(parsed);
    }
    throw new Error(`async teacher snapshot is invalid: ${resolvedPath}`);
}
function resolveBundlePayloadPath(rootDir, payloadPath) {
    const resolved = path.resolve(rootDir, payloadPath);
    const relative = path.relative(rootDir, resolved);
    if (path.isAbsolute(payloadPath) || relative.startsWith("..") || relative === "") {
        throw new Error("event export bundle payloadPath must stay within the bundle root");
    }
    return resolved;
}
export function buildRuntimeEventExportBundleManifest(input) {
    return {
        contract: RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT,
        exportName: input.exportName,
        exportedAt: input.exportedAt,
        payloadPath: input.payloadPath,
        payloadDigest: checksumJsonPayload(input.normalizedEventExport),
        summary: {
            runtimeOwner: input.normalizedEventExport.provenance.runtimeOwner,
            sessionId: input.normalizedEventExport.provenance.sessionId,
            channel: input.normalizedEventExport.provenance.channel,
            eventRange: {
                start: input.normalizedEventExport.range.start,
                end: input.normalizedEventExport.range.end,
                count: input.normalizedEventExport.range.count
            },
            interactionCount: input.normalizedEventExport.provenance.interactionCount,
            feedbackCount: input.normalizedEventExport.provenance.feedbackCount,
            sourceStreams: [...input.normalizedEventExport.provenance.sourceStreams],
            contracts: [...input.normalizedEventExport.provenance.contracts],
            semanticSurface: structuredClone(input.normalizedEventExport.provenance.semanticSurface ??
                buildEventSemanticSurface([
                    ...input.normalizedEventExport.interactionEvents,
                    ...input.normalizedEventExport.feedbackEvents
                ]))
        },
        ...(input.scanner !== undefined ? { scanner: cloneScannerExportManifestOrNull(input.scanner) } : {})
    };
}
export function validateRuntimeEventExportBundleManifest(value, normalizedEventExport) {
    const errors = [];
    if (value.contract !== RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT) {
        errors.push("normalized_event_export_bundle.v1 contract is required");
    }
    if (value.exportName.length === 0) {
        errors.push("exportName is required");
    }
    if (Number.isNaN(Date.parse(value.exportedAt))) {
        errors.push("exportedAt must be an ISO timestamp");
    }
    if (value.payloadPath.length === 0) {
        errors.push("payloadPath is required");
    }
    if (value.payloadDigest.length === 0) {
        errors.push("payloadDigest is required");
    }
    if (value.scanner !== undefined && value.scanner !== null) {
        errors.push(...validateScannerExportManifest(value.scanner).map((message) => `scanner ${message}`));
        if (normalizedEventExport !== undefined && normalizedEventExport.range.count === 0) {
            errors.push("scanner bundles require at least one exported event");
        }
    }
    if (normalizedEventExport !== undefined) {
        const exportErrors = validateNormalizedEventExport(normalizedEventExport);
        if (exportErrors.length > 0) {
            errors.push(...exportErrors);
        }
        const rebuilt = buildRuntimeEventExportBundleManifest({
            exportName: value.exportName,
            exportedAt: value.exportedAt,
            payloadPath: value.payloadPath,
            normalizedEventExport,
            ...(value.scanner !== undefined ? { scanner: value.scanner } : {})
        });
        if (rebuilt.payloadDigest !== value.payloadDigest) {
            errors.push("event export bundle payloadDigest does not match the supplied normalized event export");
        }
        const expectedSummary = value.summary.semanticSurface === undefined ? { ...rebuilt.summary, semanticSurface: undefined } : rebuilt.summary;
        if (canonicalJson(expectedSummary) !== canonicalJson(value.summary)) {
            errors.push("event export bundle summary does not match the supplied normalized event export");
        }
        if (canonicalJson(rebuilt.scanner ?? null) !== canonicalJson(value.scanner ?? null)) {
            errors.push("event export bundle scanner metadata does not match the supplied normalized event export");
        }
    }
    return errors;
}
export function loadRuntimeEventExportBundle(rootDir) {
    const resolvedRoot = path.resolve(rootDir);
    const manifestPath = path.join(resolvedRoot, RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.manifest);
    const manifest = readJsonFile(manifestPath);
    const payloadPath = resolveBundlePayloadPath(resolvedRoot, manifest.payloadPath);
    const normalizedEventExport = readJsonFile(payloadPath);
    const validationErrors = validateRuntimeEventExportBundleManifest(manifest, normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`event export bundle is invalid: ${validationErrors.join("; ")}`);
    }
    return {
        rootDir: resolvedRoot,
        manifestPath,
        payloadPath,
        manifest,
        normalizedEventExport
    };
}
function cloneScannerExportManifest(value) {
    return {
        scannerId: value.scannerId,
        lane: value.lane,
        status: value.status,
        producedAt: value.producedAt,
        sourceManifestPath: value.sourceManifestPath,
        sourceManifestDigest: value.sourceManifestDigest,
        warnings: [...value.warnings],
        failures: [...value.failures]
    };
}
function cloneScannerExportManifestOrNull(value) {
    return value === null ? null : cloneScannerExportManifest(value);
}
function validateScannerExportManifest(value) {
    const errors = [];
    if (normalizeOptionalString(value.scannerId) === undefined) {
        errors.push("scannerId is required");
    }
    if (normalizeOptionalString(value.lane) === undefined) {
        errors.push("lane is required");
    }
    if (!["complete", "partial", "failed"].includes(value.status)) {
        errors.push("status must be complete, partial, or failed");
    }
    if (Number.isNaN(Date.parse(value.producedAt))) {
        errors.push("producedAt must be an ISO timestamp");
    }
    if (value.sourceManifestPath !== null && normalizeOptionalString(value.sourceManifestPath) === undefined) {
        errors.push("sourceManifestPath must be null or a non-empty string");
    }
    if (value.sourceManifestDigest !== null && normalizeOptionalString(value.sourceManifestDigest) === undefined) {
        errors.push("sourceManifestDigest must be null or a non-empty string");
    }
    if (!Array.isArray(value.warnings) || value.warnings.some((warning) => normalizeOptionalString(warning) === undefined)) {
        errors.push("warnings must contain only non-empty strings");
    }
    if (!Array.isArray(value.failures) || value.failures.some((failure) => normalizeOptionalString(failure) === undefined)) {
        errors.push("failures must contain only non-empty strings");
    }
    if (value.status === "failed" && value.failures.length === 0) {
        errors.push("failed scanner manifests require at least one failure");
    }
    return errors;
}
function normalizeScannerExportWarnings(value) {
    const warnings = new Set();
    for (const warning of value.warnings) {
        const normalized = normalizeOptionalString(warning);
        if (normalized !== undefined) {
            warnings.add(normalized);
        }
    }
    for (const failure of value.failures) {
        const normalized = normalizeOptionalString(failure);
        if (normalized !== undefined) {
            warnings.add(`scanner_failure:${normalized}`);
        }
    }
    return [...warnings];
}
export function buildNormalizedEventExportFromScannedEvents(input) {
    const scanner = cloneScannerExportManifest(input.scanner);
    const scannerErrors = validateScannerExportManifest(scanner);
    if (scannerErrors.length > 0) {
        return {
            ok: false,
            normalizedEventExport: null,
            scanner,
            warnings: normalizeScannerExportWarnings(scanner),
            reason: "invalid_scanner_manifest",
            error: `scanner manifest is invalid: ${scannerErrors.join("; ")}`
        };
    }
    const eventCount = input.interactionEvents.length + input.feedbackEvents.length;
    if (eventCount === 0) {
        return {
            ok: false,
            normalizedEventExport: null,
            scanner,
            warnings: normalizeScannerExportWarnings(scanner),
            reason: scanner.status === "failed" ? "scan_failed" : "no_events",
            error: scanner.status === "failed"
                ? "scanner failed before producing any interaction or feedback events"
                : "scanner output did not contain any interaction or feedback events"
        };
    }
    const normalizedEventExport = buildNormalizedEventExport({
        interactionEvents: input.interactionEvents.map((event) => ({
            ...event,
            semantic: event.semantic ?? buildInteractionSemanticMetadata("scanner_export", event.kind)
        })),
        feedbackEvents: input.feedbackEvents.map((event) => ({
            ...event,
            semantic: event.semantic ?? buildFeedbackSemanticMetadata("scanner_export", event.kind, event.content)
        }))
    });
    const exportErrors = validateNormalizedEventExport(normalizedEventExport);
    if (exportErrors.length > 0) {
        return {
            ok: false,
            normalizedEventExport: null,
            scanner,
            warnings: normalizeScannerExportWarnings(scanner),
            reason: "invalid_scanner_manifest",
            error: `scanner output produced an invalid normalized event export: ${exportErrors.join("; ")}`
        };
    }
    return {
        ok: true,
        normalizedEventExport,
        scanner,
        warnings: normalizeScannerExportWarnings(scanner)
    };
}
function writeNormalizedEventExportBundleFiles(input) {
    const resolvedRoot = path.resolve(input.rootDir);
    const manifestPath = path.join(resolvedRoot, RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.manifest);
    const payloadPath = path.join(resolvedRoot, input.manifest.payloadPath);
    mkdirSync(path.dirname(payloadPath), { recursive: true });
    writeFileSync(payloadPath, canonicalJson(input.normalizedEventExport), "utf8");
    writeFileSync(manifestPath, canonicalJson(input.manifest), "utf8");
    const descriptor = loadRuntimeEventExportBundle(resolvedRoot);
    return {
        ok: true,
        wroteBundle: true,
        normalizedEventExport: descriptor.normalizedEventExport,
        rootDir: descriptor.rootDir,
        manifestPath: descriptor.manifestPath,
        payloadPath: descriptor.payloadPath,
        manifest: descriptor.manifest
    };
}
const RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_CONTRACT = "runtime_event_export_scanner_checkpoint.v1";
export const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_LIVE_TAIL_BUNDLES = 2;
export const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_BACKFILL_BUNDLES_PER_PASS = 1;
export const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_STALE_HISTORY_MS = 1000 * 60 * 60 * 24 * 7;
export const DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_BASENAME = ".openclawbrain-scanner-checkpoint.json";
function validateRuntimeEventExportScannerBundleCursor(value, fieldName) {
    const errors = [];
    if (value.exportDigest.length === 0) {
        errors.push(`${fieldName}.exportDigest is required`);
    }
    if (value.exportName.length === 0) {
        errors.push(`${fieldName}.exportName is required`);
    }
    if (Number.isNaN(Date.parse(value.exportedAt))) {
        errors.push(`${fieldName}.exportedAt must be an ISO timestamp`);
    }
    if (!Number.isInteger(value.eventRange.start) || value.eventRange.start < 0) {
        errors.push(`${fieldName}.eventRange.start must be a non-negative integer`);
    }
    if (!Number.isInteger(value.eventRange.end) || value.eventRange.end < 0) {
        errors.push(`${fieldName}.eventRange.end must be a non-negative integer`);
    }
    if (!Number.isInteger(value.eventRange.count) || value.eventRange.count < 0) {
        errors.push(`${fieldName}.eventRange.count must be a non-negative integer`);
    }
    return errors;
}
export function createRuntimeEventExportScannerCheckpoint(input) {
    return {
        contract: RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_CONTRACT,
        runtimeOwner: "openclaw",
        scanRoot: path.resolve(normalizeNonEmptyString(input.scanRoot, "scanRoot")),
        updatedAt: normalizeIsoTimestamp(input.updatedAt, "updatedAt", new Date().toISOString()),
        live: {
            after: null
        },
        backfill: {
            before: null,
            exhausted: false,
            staleBefore: null
        },
        processedExportDigests: [],
        stats: {
            scanPasses: 0,
            liveBundlesScanned: 0,
            backfillBundlesScanned: 0,
            duplicateBundlesSkipped: 0,
            staleBundlesSkipped: 0,
            invalidBundlesSkipped: 0
        }
    };
}
export function validateRuntimeEventExportScannerCheckpoint(value) {
    const errors = [];
    if (value.contract !== RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_CONTRACT) {
        errors.push("runtime_event_export_scanner_checkpoint.v1 contract is required");
    }
    if (value.runtimeOwner !== "openclaw") {
        errors.push("runtime event export scanner checkpoint runtimeOwner must be openclaw");
    }
    if (value.scanRoot.length === 0) {
        errors.push("runtime event export scanner checkpoint scanRoot is required");
    }
    if (Number.isNaN(Date.parse(value.updatedAt))) {
        errors.push("runtime event export scanner checkpoint updatedAt must be an ISO timestamp");
    }
    if (value.live.after !== null) {
        errors.push(...validateRuntimeEventExportScannerBundleCursor(value.live.after, "live.after"));
    }
    if (value.backfill.before !== null) {
        errors.push(...validateRuntimeEventExportScannerBundleCursor(value.backfill.before, "backfill.before"));
    }
    if (value.backfill.staleBefore !== null && Number.isNaN(Date.parse(value.backfill.staleBefore))) {
        errors.push("runtime event export scanner checkpoint backfill.staleBefore must be null or an ISO timestamp");
    }
    if (new Set(value.processedExportDigests).size !== value.processedExportDigests.length) {
        errors.push("runtime event export scanner checkpoint processedExportDigests must be unique");
    }
    for (const [label, count] of Object.entries(value.stats)) {
        if (!Number.isInteger(count) || count < 0) {
            errors.push(`runtime event export scanner checkpoint stats.${label} must be a non-negative integer`);
        }
    }
    return errors;
}
export function loadRuntimeEventExportScannerCheckpoint(checkpointPath) {
    const resolvedPath = path.resolve(normalizeNonEmptyString(checkpointPath, "checkpointPath"));
    const checkpoint = readJsonFile(resolvedPath);
    const errors = validateRuntimeEventExportScannerCheckpoint(checkpoint);
    if (errors.length > 0) {
        throw new Error(`runtime event export scanner checkpoint is invalid: ${errors.join("; ")}`);
    }
    return structuredClone(checkpoint);
}
function writeRuntimeEventExportScannerCheckpoint(checkpointPath, checkpoint) {
    const resolvedPath = path.resolve(checkpointPath);
    const errors = validateRuntimeEventExportScannerCheckpoint(checkpoint);
    if (errors.length > 0) {
        throw new Error(`runtime event export scanner checkpoint is invalid: ${errors.join("; ")}`);
    }
    mkdirSync(path.dirname(resolvedPath), { recursive: true });
    writeFileSync(resolvedPath, canonicalJson(checkpoint), "utf8");
}
function normalizePositiveInteger(value, fieldName, fallbackValue) {
    const resolved = value ?? fallbackValue;
    if (!Number.isInteger(resolved) || resolved <= 0) {
        throw new Error(`${fieldName} must be a positive integer`);
    }
    return resolved;
}
function normalizeNonNegativeDurationMs(value, fieldName, fallbackValue) {
    const resolved = value ?? fallbackValue;
    if (!Number.isInteger(resolved) || resolved < 0) {
        throw new Error(`${fieldName} must be a non-negative integer`);
    }
    return resolved;
}
function buildRuntimeEventExportScannerBundleCursor(descriptor) {
    return {
        exportDigest: descriptor.normalizedEventExport.provenance.exportDigest,
        exportName: descriptor.manifest.exportName,
        exportedAt: descriptor.manifest.exportedAt,
        eventRange: {
            start: descriptor.normalizedEventExport.range.start,
            end: descriptor.normalizedEventExport.range.end,
            count: descriptor.normalizedEventExport.range.count
        }
    };
}
function compareRuntimeEventExportScannerBundleCursor(left, right) {
    if (left.eventRange.end !== right.eventRange.end) {
        return left.eventRange.end - right.eventRange.end;
    }
    if (left.eventRange.start !== right.eventRange.start) {
        return left.eventRange.start - right.eventRange.start;
    }
    if (left.exportedAt !== right.exportedAt) {
        return left.exportedAt.localeCompare(right.exportedAt);
    }
    return left.exportDigest.localeCompare(right.exportDigest);
}
function buildRuntimeEventExportScannerHit(bundle, queueEntry) {
    return {
        lane: queueEntry.lane,
        rootDir: queueEntry.rootDir,
        exportDigest: queueEntry.exportDigest,
        exportName: queueEntry.exportName,
        exportedAt: queueEntry.exportedAt,
        eventRange: {
            start: queueEntry.eventRange.start,
            end: queueEntry.eventRange.end,
            count: queueEntry.eventRange.count
        },
        priorityBucket: queueEntry.priorityBucket,
        priorityScore: queueEntry.priorityScore,
        priorityReasons: [...queueEntry.priorityReasons],
        humanLabelCount: queueEntry.humanLabelCount,
        feedbackCount: queueEntry.feedbackCount,
        teacherRoles: [...queueEntry.teacherRoles],
        teacherAuthorities: [...queueEntry.teacherAuthorities],
        priorityClasses: [...queueEntry.priorityClasses],
        scopedPrincipalEventCount: queueEntry.scopedPrincipalEventCount,
        supersedingPrincipalEventCount: queueEntry.supersedingPrincipalEventCount,
        staleHistory: queueEntry.staleHistory,
        ageMsFromLatest: queueEntry.ageMsFromLatest,
        normalizedEventExport: structuredClone(bundle.descriptor.normalizedEventExport)
    };
}
function scannerTeacherRoleWeight(role) {
    switch (role) {
        case "principal":
            return 4;
        case "admin":
            return 3;
        case "operator":
            return 1.5;
        case "user":
        case "assistant":
        case "system":
        default:
            return 0;
    }
}
function scannerTeacherAuthorityWeight(authority) {
    switch (authority) {
        case "binding":
            return 4;
        case "primary_human":
            return 3;
        case "high":
            return 2;
        case "normal":
            return 0.5;
        case "background":
        default:
            return 0;
    }
}
function scannerPrincipalPriorityWeight(priorityClass) {
    switch (priorityClass) {
        case "critical":
            return 4;
        case "high":
            return 3;
        case "normal":
            return 1;
        case "low":
        default:
            return 0;
    }
}
function scannerFeedbackPriorityWeight(kind) {
    switch (kind) {
        case "correction":
            return 5;
        case "teaching":
            return 4;
        case "approval":
            return 2;
        case "suppression":
        default:
            return 1;
    }
}
function isPrincipalHeavyScannerBundle(learningSurface) {
    return (learningSurface.principalSummary.teacherRoles.some((role) => role === "principal" || role === "admin") ||
        learningSurface.principalSummary.teacherAuthorities.some((authority) => authority === "binding" || authority === "primary_human" || authority === "high") ||
        learningSurface.principalSummary.priorityClasses.some((priorityClass) => priorityClass === "critical" || priorityClass === "high") ||
        learningSurface.principalSummary.supersedingEventCount > 0);
}
function buildRuntimeEventExportScannerQueueEntry(input) {
    const learningSurface = input.bundle.descriptor.normalizedEventExport.provenance.learningSurface;
    const ageMsFromLatest = input.latestExportedAt === null ? null : Math.max(0, Date.parse(input.latestExportedAt) - Date.parse(input.bundle.cursor.exportedAt));
    const recencyScore = ageMsFromLatest === null ? 0 : roundMetric(clamp(3 - ageMsFromLatest / 86_400_000, 0, 3));
    const feedbackScore = input.bundle.descriptor.normalizedEventExport.feedbackEvents.reduce((sum, event) => sum + scannerFeedbackPriorityWeight(event.kind), 0);
    const roleScore = learningSurface.principalSummary.teacherRoles.reduce((sum, role) => sum + scannerTeacherRoleWeight(role), 0);
    const authorityScore = learningSurface.principalSummary.teacherAuthorities.reduce((sum, authority) => sum + scannerTeacherAuthorityWeight(authority), 0);
    const priorityClassScore = learningSurface.principalSummary.priorityClasses.reduce((sum, priorityClass) => sum + scannerPrincipalPriorityWeight(priorityClass), 0);
    const principalHeavy = isPrincipalHeavyScannerBundle(learningSurface);
    const staleHistory = input.staleBefore !== null && input.bundle.cursor.exportedAt < input.staleBefore;
    const priorityBucket = staleHistory
        ? "stale_history"
        : input.lane === "live"
            ? "live"
            : principalHeavy
                ? "principal_backfill"
                : "backfill";
    const priorityScore = roundMetric(feedbackScore +
        roleScore +
        authorityScore +
        priorityClassScore +
        learningSurface.principalSummary.scopedEventCount +
        learningSurface.principalSummary.supersedingEventCount * 3 +
        learningSurface.labelHarvest.humanLabels +
        recencyScore +
        (principalHeavy ? 6 : 0));
    const priorityReasons = [];
    if (input.lane === "live") {
        priorityReasons.push("live_tail");
    }
    if (principalHeavy) {
        priorityReasons.push("principal_heavy");
    }
    for (const role of learningSurface.principalSummary.teacherRoles) {
        if (role === "principal" || role === "admin") {
            priorityReasons.push(`teacher_role=${role}`);
        }
    }
    for (const authority of learningSurface.principalSummary.teacherAuthorities) {
        if (authority === "binding" || authority === "primary_human" || authority === "high") {
            priorityReasons.push(`teacher_authority=${authority}`);
        }
    }
    for (const priorityClass of learningSurface.principalSummary.priorityClasses) {
        if (priorityClass === "critical" || priorityClass === "high") {
            priorityReasons.push(`priority_class=${priorityClass}`);
        }
    }
    if (learningSurface.principalSummary.supersedingEventCount > 0) {
        priorityReasons.push(`superseding_events=${learningSurface.principalSummary.supersedingEventCount}`);
    }
    if (learningSurface.labelHarvest.humanLabels > 0) {
        priorityReasons.push(`human_labels=${learningSurface.labelHarvest.humanLabels}`);
    }
    if (staleHistory) {
        priorityReasons.push("stale_history_floor");
    }
    return {
        lane: input.lane,
        rootDir: input.bundle.descriptor.rootDir,
        exportDigest: input.bundle.cursor.exportDigest,
        exportName: input.bundle.cursor.exportName,
        exportedAt: input.bundle.cursor.exportedAt,
        eventRange: {
            start: input.bundle.cursor.eventRange.start,
            end: input.bundle.cursor.eventRange.end,
            count: input.bundle.cursor.eventRange.count
        },
        priorityBucket,
        priorityScore,
        priorityReasons,
        humanLabelCount: learningSurface.labelHarvest.humanLabels,
        feedbackCount: input.bundle.descriptor.normalizedEventExport.feedbackEvents.length,
        teacherRoles: [...learningSurface.principalSummary.teacherRoles],
        teacherAuthorities: [...learningSurface.principalSummary.teacherAuthorities],
        priorityClasses: [...learningSurface.principalSummary.priorityClasses],
        scopedPrincipalEventCount: learningSurface.principalSummary.scopedEventCount,
        supersedingPrincipalEventCount: learningSurface.principalSummary.supersedingEventCount,
        staleHistory,
        ageMsFromLatest
    };
}
function compareRuntimeEventExportScannerQueueEntry(left, right) {
    const bucketOrder = {
        live: 0,
        principal_backfill: 1,
        backfill: 2,
        stale_history: 3
    };
    if (bucketOrder[left.priorityBucket] !== bucketOrder[right.priorityBucket]) {
        return bucketOrder[left.priorityBucket] - bucketOrder[right.priorityBucket];
    }
    if (right.priorityScore !== left.priorityScore) {
        return right.priorityScore - left.priorityScore;
    }
    if (right.eventRange.end !== left.eventRange.end) {
        return right.eventRange.end - left.eventRange.end;
    }
    if (right.eventRange.start !== left.eventRange.start) {
        return right.eventRange.start - left.eventRange.start;
    }
    if (right.exportedAt !== left.exportedAt) {
        return right.exportedAt.localeCompare(left.exportedAt);
    }
    return right.exportDigest.localeCompare(left.exportDigest);
}
function listRuntimeEventExportBundleRoots(scanRoot) {
    if (!existsSync(scanRoot)) {
        return [];
    }
    return readdirSync(scanRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => path.join(scanRoot, entry.name))
        .sort((left, right) => left.localeCompare(right));
}
function discoverRuntimeEventExportBundles(scanRoot) {
    const loaded = [];
    const invalidBundles = [];
    for (const rootDir of listRuntimeEventExportBundleRoots(scanRoot)) {
        try {
            const descriptor = loadRuntimeEventExportBundle(rootDir);
            loaded.push({
                descriptor,
                cursor: buildRuntimeEventExportScannerBundleCursor(descriptor)
            });
        }
        catch (error) {
            invalidBundles.push({
                rootDir,
                error: toErrorMessage(error)
            });
        }
    }
    loaded.sort((left, right) => compareRuntimeEventExportScannerBundleCursor(left.cursor, right.cursor));
    const deduped = [];
    const seenDigests = new Set();
    const duplicateExportDigests = [];
    for (const bundle of loaded) {
        if (seenDigests.has(bundle.cursor.exportDigest)) {
            duplicateExportDigests.push(bundle.cursor.exportDigest);
            continue;
        }
        seenDigests.add(bundle.cursor.exportDigest);
        deduped.push(bundle);
    }
    return {
        bundles: deduped,
        invalidBundles,
        duplicateExportDigests: [...new Set(duplicateExportDigests)]
    };
}
function delayRuntimeEventExportScannerPoll(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}
export class RuntimeEventExportScanner {
    scanRoot;
    checkpointPath;
    liveTailBundles;
    backfillBundlesPerPass;
    staleHistoryMs;
    checkpoint;
    constructor(input) {
        this.scanRoot = path.resolve(normalizeNonEmptyString(input.scanRoot, "scanRoot"));
        this.checkpointPath = path.resolve(input.checkpointPath ?? path.join(this.scanRoot, DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_BASENAME));
        this.liveTailBundles = normalizePositiveInteger(input.liveTailBundles, "liveTailBundles", DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_LIVE_TAIL_BUNDLES);
        this.backfillBundlesPerPass = normalizePositiveInteger(input.backfillBundlesPerPass, "backfillBundlesPerPass", DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_BACKFILL_BUNDLES_PER_PASS);
        this.staleHistoryMs = normalizeNonNegativeDurationMs(input.staleHistoryMs, "staleHistoryMs", DEFAULT_RUNTIME_EVENT_EXPORT_SCANNER_STALE_HISTORY_MS);
        this.checkpoint = existsSync(this.checkpointPath)
            ? loadRuntimeEventExportScannerCheckpoint(this.checkpointPath)
            : createRuntimeEventExportScannerCheckpoint({ scanRoot: this.scanRoot });
        if (this.checkpoint.scanRoot !== this.scanRoot) {
            throw new Error(`runtime event export scanner checkpoint scanRoot mismatch: checkpoint=${this.checkpoint.scanRoot} scanner=${this.scanRoot}`);
        }
    }
    snapshot() {
        return structuredClone(this.checkpoint);
    }
    restoreCheckpoint(checkpoint) {
        const restored = structuredClone(checkpoint);
        const errors = validateRuntimeEventExportScannerCheckpoint(restored);
        if (errors.length > 0) {
            throw new Error(`runtime event export scanner checkpoint is invalid: ${errors.join("; ")}`);
        }
        if (restored.scanRoot !== this.scanRoot) {
            throw new Error(`runtime event export scanner checkpoint scanRoot mismatch: checkpoint=${restored.scanRoot} scanner=${this.scanRoot}`);
        }
        this.checkpoint = restored;
        writeRuntimeEventExportScannerCheckpoint(this.checkpointPath, this.checkpoint);
    }
    scanOnce(options = {}) {
        const scannedAt = normalizeIsoTimestamp(options.scannedAt, "scannedAt", new Date().toISOString());
        const discovered = discoverRuntimeEventExportBundles(this.scanRoot);
        const processedDigests = new Set(this.checkpoint.processedExportDigests);
        const latestExportedAt = discovered.bundles[discovered.bundles.length - 1]?.cursor.exportedAt ?? null;
        const staleBefore = latestExportedAt === null ? null : new Date(Date.parse(latestExportedAt) - this.staleHistoryMs).toISOString();
        const staleSkipped = new Set();
        const isFreshEnough = (bundle) => {
            if (staleBefore === null) {
                return true;
            }
            return bundle.cursor.exportedAt >= staleBefore;
        };
        const live = this.checkpoint.live.after === null
            ? discovered.bundles.filter((bundle) => !processedDigests.has(bundle.cursor.exportDigest) && isFreshEnough(bundle)).slice(-this.liveTailBundles)
            : discovered.bundles
                .filter((bundle) => !processedDigests.has(bundle.cursor.exportDigest) &&
                compareRuntimeEventExportScannerBundleCursor(bundle.cursor, this.checkpoint.live.after) > 0 &&
                isFreshEnough(bundle))
                .slice(0, this.liveTailBundles);
        const liveQueue = live.map((bundle) => buildRuntimeEventExportScannerQueueEntry({
            lane: "live",
            bundle,
            latestExportedAt,
            staleBefore
        }));
        const nextLiveAfter = live.length > 0 ? live[live.length - 1]?.cursor ?? null : this.checkpoint.live.after;
        const backfillFrontier = nextLiveAfter;
        const liveDigests = new Set(live.map((bundle) => bundle.cursor.exportDigest));
        const prioritizedBackfillCandidates = [];
        const staleHistoryQueue = [];
        for (const bundle of discovered.bundles) {
            if (processedDigests.has(bundle.cursor.exportDigest) || liveDigests.has(bundle.cursor.exportDigest)) {
                continue;
            }
            const queueEntry = buildRuntimeEventExportScannerQueueEntry({
                lane: "backfill",
                bundle,
                latestExportedAt,
                staleBefore
            });
            if (queueEntry.staleHistory) {
                staleHistoryQueue.push(queueEntry);
                staleSkipped.add(bundle.cursor.exportDigest);
                processedDigests.add(bundle.cursor.exportDigest);
                continue;
            }
            if (backfillFrontier !== null &&
                compareRuntimeEventExportScannerBundleCursor(bundle.cursor, backfillFrontier) < 0) {
                prioritizedBackfillCandidates.push({
                    bundle,
                    queueEntry
                });
            }
        }
        prioritizedBackfillCandidates.sort((left, right) => compareRuntimeEventExportScannerQueueEntry(left.queueEntry, right.queueEntry));
        const selectedBackfillCandidates = prioritizedBackfillCandidates.slice(0, this.backfillBundlesPerPass);
        const backfill = selectedBackfillCandidates.map((candidate) => candidate.bundle);
        for (const bundle of [...live, ...backfill]) {
            processedDigests.add(bundle.cursor.exportDigest);
        }
        const backfillQueue = prioritizedBackfillCandidates.map((candidate) => candidate.queueEntry);
        const selectedLive = liveQueue.map((queueEntry, index) => buildRuntimeEventExportScannerHit(live[index], queueEntry));
        const selectedBackfill = selectedBackfillCandidates.map((candidate) => buildRuntimeEventExportScannerHit(candidate.bundle, candidate.queueEntry));
        const nextBackfillBefore = backfill.length > 0
            ? [...backfill].sort((left, right) => compareRuntimeEventExportScannerBundleCursor(left.cursor, right.cursor))[0]?.cursor ?? null
            : this.checkpoint.backfill.before;
        const exhausted = backfillFrontier === null
            ? true
            : !prioritizedBackfillCandidates.some((candidate) => !processedDigests.has(candidate.bundle.cursor.exportDigest));
        this.checkpoint = {
            contract: RUNTIME_EVENT_EXPORT_SCANNER_CHECKPOINT_CONTRACT,
            runtimeOwner: "openclaw",
            scanRoot: this.scanRoot,
            updatedAt: scannedAt,
            live: {
                after: nextLiveAfter === null ? null : { ...nextLiveAfter, eventRange: { ...nextLiveAfter.eventRange } }
            },
            backfill: {
                before: nextBackfillBefore === null ? null : { ...nextBackfillBefore, eventRange: { ...nextBackfillBefore.eventRange } },
                exhausted,
                staleBefore
            },
            processedExportDigests: [...processedDigests],
            stats: {
                scanPasses: this.checkpoint.stats.scanPasses + 1,
                liveBundlesScanned: this.checkpoint.stats.liveBundlesScanned + selectedLive.length,
                backfillBundlesScanned: this.checkpoint.stats.backfillBundlesScanned + selectedBackfill.length,
                duplicateBundlesSkipped: this.checkpoint.stats.duplicateBundlesSkipped + discovered.duplicateExportDigests.length,
                staleBundlesSkipped: this.checkpoint.stats.staleBundlesSkipped + staleSkipped.size,
                invalidBundlesSkipped: this.checkpoint.stats.invalidBundlesSkipped + discovered.invalidBundles.length
            }
        };
        writeRuntimeEventExportScannerCheckpoint(this.checkpointPath, this.checkpoint);
        return {
            runtimeOwner: "openclaw",
            scanRoot: this.scanRoot,
            checkpointPath: this.checkpointPath,
            scannedAt,
            live: selectedLive,
            backfill: selectedBackfill,
            selected: [...selectedLive, ...selectedBackfill],
            queue: {
                ageFloor: {
                    newestExportedAt: latestExportedAt,
                    staleBefore,
                    staleHistoryMs: this.staleHistoryMs
                },
                live: liveQueue,
                backfill: backfillQueue,
                staleHistory: staleHistoryQueue.sort(compareRuntimeEventExportScannerQueueEntry)
            },
            duplicateExportDigests: discovered.duplicateExportDigests,
            staleSkippedExportDigests: [...staleSkipped],
            invalidBundles: discovered.invalidBundles,
            idle: selectedLive.length === 0 && selectedBackfill.length === 0,
            checkpoint: this.snapshot()
        };
    }
    async runLoop(options = {}) {
        const pollIntervalMs = normalizeNonNegativeDurationMs(options.pollIntervalMs, "pollIntervalMs", 0);
        const maxPasses = normalizePositiveInteger(options.maxPasses, "maxPasses", 1);
        const stopWhenIdle = options.stopWhenIdle !== false;
        let passCount = 0;
        let liveBundlesScanned = 0;
        let backfillBundlesScanned = 0;
        let lastScan = null;
        let stoppedReason = "max_passes";
        while (passCount < maxPasses) {
            if (options.signal?.aborted) {
                stoppedReason = "aborted";
                break;
            }
            lastScan = this.scanOnce();
            passCount += 1;
            liveBundlesScanned += lastScan.live.length;
            backfillBundlesScanned += lastScan.backfill.length;
            if (options.onPass !== undefined) {
                await options.onPass(lastScan);
            }
            if (stopWhenIdle && lastScan.idle) {
                stoppedReason = "idle";
                break;
            }
            if (passCount >= maxPasses) {
                stoppedReason = "max_passes";
                break;
            }
            if (pollIntervalMs > 0) {
                await delayRuntimeEventExportScannerPoll(pollIntervalMs);
            }
        }
        return {
            runtimeOwner: "openclaw",
            passCount,
            liveBundlesScanned,
            backfillBundlesScanned,
            stoppedReason,
            lastScan,
            checkpoint: this.snapshot()
        };
    }
}
export function createRuntimeEventExportScanner(input) {
    return new RuntimeEventExportScanner(input);
}
function normalizeNonEmptyString(value, fieldName) {
    if (typeof value !== "string" || value.trim().length === 0) {
        throw new Error(`${fieldName} is required`);
    }
    return value.trim();
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}
function normalizeNonNegativeInteger(value, fieldName, fallbackValue) {
    if (value === undefined || value === null) {
        return fallbackValue;
    }
    if (!Number.isInteger(value) || value < 0) {
        throw new Error(`${fieldName} must be a non-negative integer`);
    }
    return value;
}
function normalizeIsoTimestamp(value, fieldName, fallbackValue) {
    const candidate = value ?? fallbackValue;
    if (candidate === undefined || candidate === null || candidate === "") {
        throw new Error(`${fieldName} is required`);
    }
    if (typeof candidate !== "string" || Number.isNaN(Date.parse(candidate))) {
        throw new Error(`${fieldName} must be an ISO timestamp`);
    }
    return new Date(candidate).toISOString();
}
function normalizeMode(value) {
    return value ?? "heuristic";
}
function normalizeCompileSelectionMode(value) {
    if (value === undefined) {
        return undefined;
    }
    if (value === "flat_rank_v1" || value === "graph_walk_v1") {
        return value;
    }
    throw new Error(`selectionMode must be flat_rank_v1 or graph_walk_v1, received ${String(value)}`);
}
function normalizeRuntimeHints(value) {
    if (value === undefined) {
        return [];
    }
    if (value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
        throw new Error("runtimeHints must be an array of non-empty strings");
    }
    return value.map((item) => item.trim());
}
function deterministicEventId(value) {
    const digest = createHash("sha256").update(JSON.stringify(value)).digest("hex");
    return `evt-${digest.slice(0, 16)}`;
}
function nextSequenceFactory(startValue = 1) {
    let cursor = normalizeNonNegativeInteger(startValue, "sequenceStart", 1);
    return (explicitValue) => {
        if (explicitValue !== undefined && explicitValue !== null) {
            const normalized = normalizeNonNegativeInteger(explicitValue, "sequence", explicitValue);
            cursor = Math.max(cursor, normalized + 1);
            return normalized;
        }
        const current = cursor;
        cursor += 1;
        return current;
    };
}
function isFeedbackKind(value) {
    return FEEDBACK_KINDS.has(value);
}
function isPresent(value) {
    return value !== null;
}
const KNOWN_SCANNER_REALITY_PRINCIPALS = {
    bihua: {
        teacherIdentity: "bihua",
        teacherRole: "principal",
        teacherAuthority: "binding",
        priorityClass: "critical"
    },
    jonathan: {
        teacherIdentity: "jonathan",
        teacherRole: "admin",
        teacherAuthority: "high",
        priorityClass: "high"
    },
    "jonathan gu": {
        teacherIdentity: "jonathan",
        teacherRole: "admin",
        teacherAuthority: "high",
        priorityClass: "high"
    }
};
function normalizePrincipalPriorityHint(value, fieldName) {
    if (value === undefined || value === null) {
        return undefined;
    }
    if (value === "critical" || value === "high" || value === "normal" || value === "low") {
        return value;
    }
    throw new Error(`${fieldName} must be critical, high, normal, or low`);
}
function normalizePrincipalActorName(value) {
    const normalized = normalizeOptionalString(value);
    return normalized === undefined ? undefined : normalized.toLowerCase().replace(/\s+/gu, " ");
}
function slugifyPrincipalIdentityFragment(value) {
    return value
        .toLowerCase()
        .replace(/[^a-z0-9]+/gu, "-")
        .replace(/^-+|-+$/gu, "");
}
function buildRuntimePrincipalScope(input) {
    const scopeKey = [
        `profile:${input.profileSelector}`,
        input.profileId === undefined ? null : `profile_id:${input.profileId}`,
        `session:${input.sessionId}`,
        input.userId === undefined ? null : `user:${input.userId}`,
        input.interactionId === undefined ? null : `interaction:${input.interactionId}`,
        input.messageId === undefined ? null : `message:${input.messageId}`
    ]
        .filter(isPresent)
        .join("|");
    if (input.messageId !== undefined) {
        return {
            kind: "message",
            profileSelector: input.profileSelector,
            sessionId: input.sessionId,
            ...(input.interactionId === undefined ? {} : { interactionId: input.interactionId }),
            messageId: input.messageId,
            scopeKey
        };
    }
    if (input.interactionId !== undefined) {
        return {
            kind: "interaction",
            profileSelector: input.profileSelector,
            sessionId: input.sessionId,
            interactionId: input.interactionId,
            scopeKey
        };
    }
    return {
        kind: "session",
        profileSelector: input.profileSelector,
        sessionId: input.sessionId,
        scopeKey
    };
}
function buildRuntimeSelfPrincipal(turn) {
    const profileSelector = normalizeRuntimeProfileSelector(turn.profileSelector, "profileSelector");
    const profileId = normalizeOptionalString(turn.profileId);
    const userId = normalizeOptionalString(turn.userId);
    return {
        teacherIdentity: "openclaw/self",
        teacherRole: "assistant",
        teacherAuthority: "background",
        priorityClass: "low",
        principalScope: buildRuntimePrincipalScope({
            profileSelector,
            ...(profileId === undefined ? {} : { profileId }),
            sessionId: turn.sessionId,
            ...(userId === undefined ? {} : { userId })
        })
    };
}
function resolveRuntimeFeedbackPrincipal(input) {
    const actorName = normalizePrincipalActorName(input.feedback.actorName);
    const knownPrincipal = actorName === undefined ? undefined : KNOWN_SCANNER_REALITY_PRINCIPALS[actorName];
    const profileSelector = normalizeRuntimeProfileSelector(input.turn.profileSelector, "profileSelector");
    const profileId = normalizeOptionalString(input.turn.profileId);
    const userId = normalizeOptionalString(input.turn.userId);
    const priorityHint = normalizePrincipalPriorityHint(input.feedback.priorityHint, "feedback.priorityHint");
    const userIdentityFragment = userId === undefined ? undefined : slugifyPrincipalIdentityFragment(userId);
    const actorIdentityFragment = actorName === undefined ? undefined : slugifyPrincipalIdentityFragment(actorName);
    const teacherIdentity = knownPrincipal?.teacherIdentity ??
        (userIdentityFragment === undefined || userIdentityFragment.length === 0
            ? actorIdentityFragment === undefined || actorIdentityFragment.length === 0
                ? undefined
                : `scanner/actor/${actorIdentityFragment}`
            : `scanner/user/${userIdentityFragment}`);
    if (teacherIdentity === undefined) {
        return undefined;
    }
    return {
        teacherIdentity,
        teacherRole: knownPrincipal?.teacherRole ?? "user",
        teacherAuthority: knownPrincipal?.teacherAuthority ?? "normal",
        priorityClass: priorityHint ?? knownPrincipal?.priorityClass ?? "normal",
        principalScope: buildRuntimePrincipalScope({
            profileSelector,
            ...(profileId === undefined ? {} : { profileId }),
            sessionId: input.turn.sessionId,
            ...(userId === undefined ? {} : { userId }),
            ...(input.relatedInteractionId === undefined ? {} : { interactionId: input.relatedInteractionId }),
            ...(input.messageId === undefined ? {} : { messageId: input.messageId })
        })
    };
}
export function classifyFeedbackKind(content) {
    return classifyFeedbackSignalContent(content)?.kind ?? "teaching";
}
export function formatPromptContext(compileResponse) {
    const lines = [
        "[BRAIN_CONTEXT v1]",
        `PACK_ID: ${compileResponse.packId}`,
        `MODE: ${compileResponse.diagnostics.modeEffective}`
    ];
    if (compileResponse.diagnostics.routerIdentity !== null) {
        lines.push(`ROUTER: ${compileResponse.diagnostics.routerIdentity}`);
    }
    if (compileResponse.selectedContext.length > 0) {
        lines.push("");
    }
    for (const block of compileResponse.selectedContext) {
        lines.push(`SOURCE: ${block.source}`);
        lines.push(`BLOCK_ID: ${block.id}`);
        lines.push(block.text.trim());
        lines.push("");
    }
    if (lines[lines.length - 1] === "") {
        lines.pop();
    }
    lines.push("[/BRAIN_CONTEXT]");
    return `${lines.join("\n")}\n`;
}
function resolveActivationRootForFailure(value) {
    return path.resolve(normalizeOptionalString(value) ?? ".");
}
function monotonicClockNs() {
    return process.hrtime.bigint();
}
function elapsedMsFrom(startedAtNs, endedAtNs = monotonicClockNs()) {
    return Number(endedAtNs - startedAtNs) / 1_000_000;
}
function roundHotPathTimingMs(value) {
    return Math.round(value * 1_000) / 1_000;
}
function buildUnavailableBrainServeHotPathTiming(detail) {
    return {
        scope: "brain_serve_hot_path_only",
        totalMs: null,
        routeSelectionMs: null,
        promptAssemblyMs: null,
        otherMs: null,
        backgroundWorkIncluded: false,
        detail
    };
}
function buildBrainServeHotPathTiming(input) {
    const totalMs = roundHotPathTimingMs(input.totalMs);
    const routeSelectionMs = input.routeSelectionMs === null ? null : roundHotPathTimingMs(input.routeSelectionMs);
    const promptAssemblyMs = input.promptAssemblyMs === null ? null : roundHotPathTimingMs(input.promptAssemblyMs);
    const otherMs = roundHotPathTimingMs(Math.max(0, input.totalMs - (input.routeSelectionMs ?? 0) - (input.promptAssemblyMs ?? 0)));
    return {
        scope: "brain_serve_hot_path_only",
        totalMs,
        routeSelectionMs,
        promptAssemblyMs,
        otherMs,
        backgroundWorkIncluded: false,
        detail: BRAIN_SERVE_HOT_PATH_TIMING_DETAIL
    };
}
function failOpenCompileResult(error, activationRoot, timing = buildUnavailableBrainServeHotPathTiming("serve-path timing was unavailable")) {
    return {
        ok: false,
        fallbackToStaticContext: true,
        hardRequirementViolated: false,
        activationRoot: path.resolve(activationRoot),
        error: toErrorMessage(error),
        brainContext: "",
        timing
    };
}
function classifyCompileFailure(error, activationRoot, timing = buildUnavailableBrainServeHotPathTiming("serve-path timing was unavailable")) {
    const resolvedActivationRoot = path.resolve(activationRoot);
    try {
        const inspection = inspectActivationState(resolvedActivationRoot);
        const active = inspection.active;
        if (active !== null && active.routePolicy === "requires_learned_routing") {
            const failureReason = active.findings.length > 0 ? active.findings.join("; ") : toErrorMessage(error);
            return {
                ok: false,
                fallbackToStaticContext: false,
                hardRequirementViolated: true,
                activationRoot: resolvedActivationRoot,
                error: `Learned-routing hotpath hard requirement violated for active pack ${active.packId} (routerIdentity=${active.routerIdentity ?? "null"}): ${failureReason}`,
                brainContext: "",
                timing
            };
        }
    }
    catch {
        return failOpenCompileResult(error, resolvedActivationRoot, timing);
    }
    return failOpenCompileResult(error, resolvedActivationRoot, timing);
}
function uniqueNotes(notes) {
    return [...new Set(notes.filter((note) => note.length > 0))];
}
function buildServeRouteLogFailOpenWarning(scope, error) {
    return `learning spine serve route log failed open (${scope}): ${toErrorMessage(error)}`;
}
function buildServeRouteLogFailOpenNotes(scope, error) {
    return [
        "serve_route_log_status=fail_open",
        `serve_route_log_scope=${scope}`,
        `serve_route_log_error=${toErrorMessage(error)}`
    ];
}
function normalizeServeRouteChannel(value) {
    if (typeof value !== "string") {
        return undefined;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
}
function normalizeServeRouteMessage(value) {
    return typeof value === "string" ? value.trim() : "";
}
function normalizeServeRouteInstalledEntryPath(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length === 0 ? null : path.resolve(trimmed);
}
function buildCompileServeRouteBreadcrumbs(input) {
    return {
        entrypoint: "compileRuntimeContext",
        invocationSurface: input._serveRouteBreadcrumbs?.invocationSurface ?? "direct_compile_call",
        hostEvent: input._serveRouteBreadcrumbs?.hostEvent ?? null,
        installedEntryPath: normalizeServeRouteInstalledEntryPath(input._serveRouteBreadcrumbs?.installedEntryPath),
        syntheticTurn: true
    };
}
function buildRunRuntimeTurnServeRouteBreadcrumbs() {
    return {
        entrypoint: "runRuntimeTurn",
        invocationSurface: "runtime_turn_helper",
        hostEvent: null,
        installedEntryPath: null,
        syntheticTurn: false
    };
}
function appendCompileServeRouteDecisionLog(input) {
    if (input.compileInput._suppressServeLog) {
        return;
    }
    if (!input.compileResult.ok && input.userMessage.length === 0) {
        // Invalid/partial invocation envelopes should fail open without paying for
        // serve-time route log writes on the live compile hot path.
        return;
    }
    const recordedAt = new Date().toISOString();
    const sessionId = normalizeServeRouteChannel(input.compileInput.sessionId) ?? `ext-compile-${Date.now()}`;
    const channel = normalizeServeRouteChannel(input.compileInput.channel) ?? "extension";
    const syntheticTurn = {
        sessionId,
        channel,
        userMessage: input.userMessage,
        createdAt: recordedAt
    };
    if (input.compileInput.maxContextBlocks !== undefined) {
        syntheticTurn.maxContextBlocks = input.compileInput.maxContextBlocks;
    }
    if (input.compileInput.budgetStrategy === "fixed_v1" || input.compileInput.budgetStrategy === "empirical_v1") {
        syntheticTurn.budgetStrategy = input.compileInput.budgetStrategy;
    }
    if (input.compileInput.mode === "heuristic" || input.compileInput.mode === "learned") {
        syntheticTurn.mode = input.compileInput.mode;
    }
    if (input.compileInput.runtimeHints !== undefined) {
        syntheticTurn.runtimeHints = input.compileInput.runtimeHints;
    }
    try {
        appendServeTimeRouteDecisionLog({
            activationRoot: input.activationRoot,
            turn: syntheticTurn,
            compileResult: input.compileResult,
            recordedAt,
            breadcrumbs: buildCompileServeRouteBreadcrumbs(input.compileInput)
        });
    }
    catch (error) {
        if (input.compileResult.ok) {
            input.compileResult.compileResponse.diagnostics.notes = uniqueNotes([
                ...input.compileResult.compileResponse.diagnostics.notes,
                ...buildServeRouteLogFailOpenNotes("compileRuntimeContext", error)
            ]);
        }
        console.warn(`[openclawbrain] ${buildServeRouteLogFailOpenWarning("compileRuntimeContext", error)} ` +
            `(activationRoot=${input.activationRoot}, sessionId=${sessionId}, channel=${channel})`);
    }
}
function roundMetric(value) {
    return Math.round(value * 100) / 100;
}
function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}
function clampInteger(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, Math.round(value)));
}
export function deriveEmpiricalStructuralBudget(input) {
    const requestedStrategy = input.requestedStrategy ?? "fixed_v1";
    const defaultMaxContextBlocks = input.defaultMaxContextBlocks ?? 4;
    const minimumMaxContextBlocks = input.minimumMaxContextBlocks ?? 2;
    const maximumMaxContextBlocks = input.maximumMaxContextBlocks ?? 6;
    if (input.requestedMaxContextBlocks !== undefined) {
        const maxContextBlocks = clampInteger(input.requestedMaxContextBlocks, 0, Number.MAX_SAFE_INTEGER);
        return {
            requestedStrategy,
            effectiveStrategy: requestedStrategy,
            maxContextBlocks,
            defaultMaxContextBlocks,
            evidence: { split: 0, merge: 0, prune: 0, connect: 0 },
            evidenceTotal: 0,
            tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
            notes: [
                `requested_budget_strategy=${requestedStrategy}`,
                `requested_max_context_blocks=${maxContextBlocks}`,
                `resolved_budget_strategy=${requestedStrategy}`,
                `resolved_max_context_blocks=${maxContextBlocks}`,
                "structural_budget_source=caller_override"
            ]
        };
    }
    if (requestedStrategy !== "empirical_v1") {
        return {
            requestedStrategy,
            effectiveStrategy: requestedStrategy,
            maxContextBlocks: defaultMaxContextBlocks,
            defaultMaxContextBlocks,
            evidence: { split: 0, merge: 0, prune: 0, connect: 0 },
            evidenceTotal: 0,
            tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
            notes: [
                `requested_budget_strategy=${requestedStrategy}`,
                `resolved_budget_strategy=${requestedStrategy}`,
                `resolved_max_context_blocks=${defaultMaxContextBlocks}`,
                `structural_budget_source=${requestedStrategy === "fixed_v1" ? "fixed_default" : "fixed_fallback"}`
            ]
        };
    }
    const evidence = {
        split: Math.max(0, input.evolution?.structuralOps.split ?? 0),
        merge: Math.max(0, input.evolution?.structuralOps.merge ?? 0),
        prune: Math.max(0, Math.max(input.evolution?.structuralOps.prune ?? 0, input.evolution?.prunedBlockIds.length ?? 0)),
        connect: Math.max(0, input.evolution?.structuralOps.connect ?? 0)
    };
    const evidenceTotal = evidence.split + evidence.merge + evidence.prune + evidence.connect;
    if (evidenceTotal === 0) {
        return {
            requestedStrategy,
            effectiveStrategy: "fixed_v1",
            maxContextBlocks: defaultMaxContextBlocks,
            defaultMaxContextBlocks,
            evidence,
            evidenceTotal,
            tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
            notes: [
                `requested_budget_strategy=${requestedStrategy}`,
                "resolved_budget_strategy=fixed_v1",
                `resolved_max_context_blocks=${defaultMaxContextBlocks}`,
                "structural_budget_source=no_evidence_fallback"
            ]
        };
    }
    const tendencies = {
        split: evidence.split / evidenceTotal,
        merge: evidence.merge / evidenceTotal,
        prune: evidence.prune / evidenceTotal,
        connect: evidence.connect / evidenceTotal
    };
    const expansionPressure = tendencies.split + tendencies.connect;
    const contractionPressure = tendencies.merge + tendencies.prune;
    const directionalPressure = expansionPressure - contractionPressure;
    const maxContextBlocks = clampInteger(defaultMaxContextBlocks + directionalPressure * 2, minimumMaxContextBlocks, maximumMaxContextBlocks);
    return {
        requestedStrategy,
        effectiveStrategy: requestedStrategy,
        maxContextBlocks,
        defaultMaxContextBlocks,
        evidence,
        evidenceTotal,
        tendencies,
        notes: [
            `requested_budget_strategy=${requestedStrategy}`,
            `resolved_budget_strategy=${requestedStrategy}`,
            `resolved_max_context_blocks=${maxContextBlocks}`,
            "structural_budget_source=graph_evolution_empirical_v1",
            `structural_budget_evidence=split:${evidence.split},merge:${evidence.merge},prune:${evidence.prune},connect:${evidence.connect},total:${evidenceTotal}`,
            `structural_budget_tendencies=split:${tendencies.split.toFixed(4)},merge:${tendencies.merge.toFixed(4)},prune:${tendencies.prune.toFixed(4)},connect:${tendencies.connect.toFixed(4)}`,
            `structural_budget_pressures=expand:${expansionPressure.toFixed(4)},contract:${contractionPressure.toFixed(4)},directional:${directionalPressure.toFixed(4)}`
        ]
    };
}
export function deriveEmpiricalStructuralBudgetFromCompileSignals(input) {
    const requestedStrategy = input.requestedStrategy ?? "fixed_v1";
    const defaultMaxContextBlocks = input.defaultMaxContextBlocks ?? 4;
    const minimumMaxContextBlocks = input.minimumMaxContextBlocks ?? 2;
    const maximumMaxContextBlocks = input.maximumMaxContextBlocks ?? 6;
    if (input.requestedMaxContextBlocks !== undefined) {
        const maxContextBlocks = clampInteger(input.requestedMaxContextBlocks, 0, Number.MAX_SAFE_INTEGER);
        return {
            requestedStrategy,
            effectiveStrategy: requestedStrategy,
            maxContextBlocks,
            defaultMaxContextBlocks,
            evidence: { split: 0, merge: 0, prune: 0, connect: 0 },
            evidenceTotal: 0,
            tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
            notes: [
                `requested_budget_strategy=${requestedStrategy}`,
                `requested_max_context_blocks=${maxContextBlocks}`,
                `resolved_budget_strategy=${requestedStrategy}`,
                `resolved_max_context_blocks=${maxContextBlocks}`,
                "structural_budget_source=caller_override"
            ]
        };
    }
    if (requestedStrategy !== "empirical_v1") {
        return {
            requestedStrategy,
            effectiveStrategy: requestedStrategy,
            maxContextBlocks: defaultMaxContextBlocks,
            defaultMaxContextBlocks,
            evidence: { split: 0, merge: 0, prune: 0, connect: 0 },
            evidenceTotal: 0,
            tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
            notes: [
                `requested_budget_strategy=${requestedStrategy}`,
                `resolved_budget_strategy=${requestedStrategy}`,
                `resolved_max_context_blocks=${defaultMaxContextBlocks}`,
                `structural_budget_source=${requestedStrategy === "fixed_v1" ? "fixed_default" : "fixed_fallback"}`
            ]
        };
    }
    const compileEvidence = {
        expansionCandidates: Math.max(0, (input.structuralSignals?.matchedCandidateCount ?? 0) - (input.structuralSignals?.selectedMatchedCount ?? 0)),
        traversalActivations: Math.max(0, input.structuralSignals?.traversalActivatedCount ?? 0),
        overlapPruned: Math.max(0, input.structuralSignals?.overlapPrunedCount ?? 0)
    };
    const evidence = {
        split: compileEvidence.expansionCandidates,
        merge: 0,
        prune: compileEvidence.overlapPruned,
        connect: compileEvidence.traversalActivations
    };
    const evidenceTotal = evidence.split + evidence.merge + evidence.prune + evidence.connect;
    if (evidenceTotal === 0) {
        return {
            requestedStrategy,
            effectiveStrategy: "fixed_v1",
            maxContextBlocks: defaultMaxContextBlocks,
            defaultMaxContextBlocks,
            evidence,
            evidenceTotal,
            tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
            notes: [
                `requested_budget_strategy=${requestedStrategy}`,
                "resolved_budget_strategy=fixed_v1",
                `resolved_max_context_blocks=${defaultMaxContextBlocks}`,
                "structural_budget_source=no_compile_signal_evidence_fallback"
            ]
        };
    }
    const tendencies = {
        split: evidence.split / evidenceTotal,
        merge: 0,
        prune: evidence.prune / evidenceTotal,
        connect: evidence.connect / evidenceTotal
    };
    const expansionPressure = tendencies.split + tendencies.connect;
    const contractionPressure = tendencies.prune;
    const directionalPressure = expansionPressure - contractionPressure;
    const maxContextBlocks = clampInteger(defaultMaxContextBlocks + directionalPressure * 2, minimumMaxContextBlocks, maximumMaxContextBlocks);
    return {
        requestedStrategy,
        effectiveStrategy: requestedStrategy,
        maxContextBlocks,
        defaultMaxContextBlocks,
        evidence,
        evidenceTotal,
        tendencies,
        notes: [
            `requested_budget_strategy=${requestedStrategy}`,
            `resolved_budget_strategy=${requestedStrategy}`,
            `resolved_max_context_blocks=${maxContextBlocks}`,
            "structural_budget_source=compile_structural_signals_empirical_v1",
            `structural_budget_compile_evidence=matched_unselected:${compileEvidence.expansionCandidates},traversal:${compileEvidence.traversalActivations},overlap_pruned:${compileEvidence.overlapPruned},total:${evidenceTotal}`,
            `structural_budget_tendencies=split:${tendencies.split.toFixed(4)},merge:${tendencies.merge.toFixed(4)},prune:${tendencies.prune.toFixed(4)},connect:${tendencies.connect.toFixed(4)}`,
            `structural_budget_pressures=expand:${expansionPressure.toFixed(4)},contract:${contractionPressure.toFixed(4)},directional:${directionalPressure.toFixed(4)}`
        ]
    };
}
function resolveCompileBudget(target, input) {
    const pack = loadPackFromActivation(target.activationRoot, "active");
    return deriveEmpiricalStructuralBudget({
        requestedStrategy: input.budgetStrategy ?? "empirical_v1",
        ...(input.maxContextBlocks !== undefined ? { requestedMaxContextBlocks: input.maxContextBlocks } : {}),
        ...(pack?.graph.evolution !== undefined ? { evolution: pack.graph.evolution } : {}),
        defaultMaxContextBlocks: 4,
        minimumMaxContextBlocks: 2,
        maximumMaxContextBlocks: 6
    });
}
export function resolveActivePackForCompile(activationRoot) {
    const resolvedActivationRoot = path.resolve(normalizeNonEmptyString(activationRoot, "activationRoot"));
    const inspection = inspectActivationState(resolvedActivationRoot);
    const activePointer = inspection.pointers.active;
    if (inspection.active === null || activePointer === null) {
        throw new Error(`No active pack pointer found in ${resolvedActivationRoot}`);
    }
    if (!inspection.active.activationReady) {
        throw new Error(`Active pack is not activation-ready: ${inspection.active.findings.join("; ")}`);
    }
    return {
        activationRoot: resolvedActivationRoot,
        activePointer,
        inspection: inspection.active
    };
}
export function compileRuntimeContext(input) {
    const totalStartedAtNs = monotonicClockNs();
    const fallbackActivationRoot = resolveActivationRootForFailure(input.activationRoot);
    let activationRoot = fallbackActivationRoot;
    let agentId = process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
    let runtimeHints = [];
    let selectionMode;
    let userMessage = "";
    let maxContextChars;
    let mode = "heuristic";
    let routeSelectionStartedAtNs = null;
    let routeSelectionMs = null;
    let promptAssemblyStartedAtNs = null;
    let promptAssemblyMs = null;
    let result;
    try {
        activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
        agentId = normalizeOptionalString(input.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
        runtimeHints = normalizeRuntimeHints(input.runtimeHints);
        selectionMode = normalizeCompileSelectionMode(input.selectionMode);
        userMessage = normalizeNonEmptyString(input.message, "message");
        maxContextChars =
            input.maxContextChars !== undefined
                ? normalizeNonNegativeInteger(input.maxContextChars, "maxContextChars", input.maxContextChars)
                : undefined;
        mode = normalizeMode(input.mode);
    }
    catch (error) {
        result = failOpenCompileResult(error, fallbackActivationRoot, buildBrainServeHotPathTiming({
            totalMs: elapsedMsFrom(totalStartedAtNs),
            routeSelectionMs,
            promptAssemblyMs
        }));
        appendCompileServeRouteDecisionLog({
            compileInput: input,
            activationRoot: result.activationRoot,
            compileResult: result,
            userMessage: normalizeServeRouteMessage(input.message)
        });
        return result;
    }
    try {
        const target = resolveActivePackForCompile(activationRoot);
        const resolvedBudget = resolveCompileBudget(target, input);
        routeSelectionStartedAtNs = monotonicClockNs();
        const compile = compileRuntimeFromActivation(activationRoot, {
            contract: CONTRACT_IDS.runtimeCompile,
            agentId,
            userMessage,
            maxContextBlocks: resolvedBudget.maxContextBlocks,
            ...(maxContextChars !== undefined ? { maxContextChars } : {}),
            modeRequested: mode,
            activePackId: target.activePointer.packId,
            ...(input.compactionMode !== undefined ? { compactionMode: input.compactionMode } : {}),
            ...(runtimeHints.length > 0 ? { runtimeHints } : {})
        }, {
            ...(selectionMode !== undefined ? { selectionMode } : {})
        });
        routeSelectionMs = elapsedMsFrom(routeSelectionStartedAtNs);
        const compileResponse = {
            ...compile.response,
            diagnostics: {
                ...compile.response.diagnostics,
                notes: uniqueNotes([...compile.response.diagnostics.notes, ...resolvedBudget.notes, "OpenClaw remains the runtime owner"])
            }
        };
        promptAssemblyStartedAtNs = monotonicClockNs();
        const brainContext = formatPromptContext(compileResponse);
        promptAssemblyMs = elapsedMsFrom(promptAssemblyStartedAtNs);
        result = {
            ok: true,
            fallbackToStaticContext: false,
            hardRequirementViolated: false,
            activationRoot,
            activePackId: compile.target.packId,
            packRootDir: path.resolve(target.activePointer.packRootDir),
            compileResponse,
            brainContext,
            timing: buildBrainServeHotPathTiming({
                totalMs: elapsedMsFrom(totalStartedAtNs),
                routeSelectionMs,
                promptAssemblyMs
            })
        };
    }
    catch (error) {
        if (routeSelectionStartedAtNs !== null && routeSelectionMs === null) {
            routeSelectionMs = elapsedMsFrom(routeSelectionStartedAtNs);
        }
        if (promptAssemblyStartedAtNs !== null && promptAssemblyMs === null) {
            promptAssemblyMs = elapsedMsFrom(promptAssemblyStartedAtNs);
        }
        result = classifyCompileFailure(error, activationRoot, buildBrainServeHotPathTiming({
            totalMs: elapsedMsFrom(totalStartedAtNs),
            routeSelectionMs,
            promptAssemblyMs
        }));
    }
    appendCompileServeRouteDecisionLog({
        compileInput: input,
        activationRoot: result.activationRoot,
        compileResult: result,
        userMessage
    });
    return result;
}
function readDiagnosticNoteValue(notes, prefix) {
    const note = notes.find((entry) => entry.startsWith(prefix));
    return note === undefined ? null : note.slice(prefix.length);
}
function readDiagnosticNoteList(notes, prefix) {
    const value = readDiagnosticNoteValue(notes, prefix);
    if (value === null || value.length === 0) {
        return [];
    }
    return value
        .split("|")
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
}
function parseDiagnosticInteger(value) {
    if (value === null) {
        return null;
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isInteger(parsed) ? parsed : null;
}
function parseStructuralBudgetStrategy(value) {
    return value === "fixed_v1" || value === "empirical_v1" ? value : null;
}
function summarizeStructuralDecisionFromNotes(notes) {
    const requestedBudgetStrategy = parseStructuralBudgetStrategy(readDiagnosticNoteValue(notes, "requested_budget_strategy="));
    const resolvedBudgetStrategy = parseStructuralBudgetStrategy(readDiagnosticNoteValue(notes, "resolved_budget_strategy="));
    const resolvedMaxContextBlocks = parseDiagnosticInteger(readDiagnosticNoteValue(notes, "resolved_max_context_blocks="));
    const requestedMaxContextBlocks = parseDiagnosticInteger(readDiagnosticNoteValue(notes, "requested_max_context_blocks="));
    const structuralBudgetSource = readDiagnosticNoteValue(notes, "structural_budget_source=");
    switch (structuralBudgetSource) {
        case "caller_override":
            return {
                origin: "manual_caller_shape",
                basis: "caller_override",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `manual caller shaping fixed the serve-path structural budget at ${requestedMaxContextBlocks ?? resolvedMaxContextBlocks ?? "unknown"} blocks; empirical and default-path control were bypassed`
            };
        case "compile_structural_signals_empirical_v1":
            return {
                origin: "empirical_control",
                basis: "compile_structural_signals",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `empirical control resolved the serve-path structural budget to ${resolvedMaxContextBlocks ?? "unknown"} blocks from prior compile structural signals; no manual caller shaping was applied`
            };
        case "graph_evolution_empirical_v1":
            return {
                origin: "empirical_control",
                basis: "graph_evolution",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `empirical control resolved the serve-path structural budget to ${resolvedMaxContextBlocks ?? "unknown"} blocks from active-pack graph evolution evidence; no manual caller shaping was applied`
            };
        case "fixed_default":
            return {
                origin: "default_path_control",
                basis: "fixed_default",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `default-path fixed-budget control set the serve-path structural budget to ${resolvedMaxContextBlocks ?? "unknown"} blocks because empirical control was not requested`
            };
        case "fixed_fallback":
            return {
                origin: "default_path_control",
                basis: "fixed_fallback",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `default-path fixed-budget fallback set the serve-path structural budget to ${resolvedMaxContextBlocks ?? "unknown"} blocks; no manual caller shaping was applied`
            };
        case "no_evidence_fallback":
            return {
                origin: "default_path_control",
                basis: "no_evidence_fallback",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `default-path fixed-budget fallback kept the serve-path structural budget at ${resolvedMaxContextBlocks ?? "unknown"} blocks because graph-evolution evidence was absent`
            };
        case "no_compile_signal_evidence_fallback":
            return {
                origin: "default_path_control",
                basis: "no_compile_signal_evidence_fallback",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: `default-path fixed-budget fallback kept the serve-path structural budget at ${resolvedMaxContextBlocks ?? "unknown"} blocks because compile-signal evidence was absent`
            };
        default:
            return {
                origin: "unknown",
                basis: "unknown",
                requestedBudgetStrategy,
                resolvedBudgetStrategy,
                resolvedMaxContextBlocks,
                detail: "structural decision attribution is unavailable from the compile notes"
            };
    }
}
function sortedUniqueStrings(values) {
    return [...new Set(values.filter((value) => value.length > 0))].sort((left, right) => left.localeCompare(right));
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
function buildContextAttributionSummary(input) {
    const selectionTiers = readDiagnosticNoteValue(input.notes ?? [], "selection_tiers=");
    const selectedContext = [...(input.selectedContext ?? [])];
    const stableKernelBlocks = selectedContext.filter((block) => isStableKernelContextBlock(block));
    const brainCompiledBlocks = selectedContext.filter((block) => !isStableKernelContextBlock(block));
    if (input.unprobed) {
        return {
            selectedContextCount: 0,
            stableKernelBlockCount: 0,
            brainCompiledBlockCount: 0,
            stableKernelSources: [],
            brainCompiledSources: [],
            selectionTiers,
            evidence: "unprobed",
            detail: "compile probe was not run, so kernel-vs-brain attribution is unknown"
        };
    }
    if (input.hardRequirementViolated) {
        return {
            selectedContextCount: 0,
            stableKernelBlockCount: 0,
            brainCompiledBlockCount: 0,
            stableKernelSources: [],
            brainCompiledSources: [],
            selectionTiers,
            evidence: "hard_fail",
            detail: "learned-routing hard requirement failed before any compiled brain context could be selected"
        };
    }
    if (input.fallbackToStaticContext) {
        return {
            selectedContextCount: 0,
            stableKernelBlockCount: 0,
            brainCompiledBlockCount: 0,
            stableKernelSources: [],
            brainCompiledSources: [],
            selectionTiers,
            evidence: "fail_open_static_context",
            detail: "serve probe fell back to static context, so no compiled brain context was selected"
        };
    }
    const evidence = brainCompiledBlocks.length > 0
        ? input.usedLearnedRouteFn === true
            ? "route_fn_and_brain_context"
            : "brain_context_only"
        : input.usedLearnedRouteFn === true
            ? "route_fn_only"
            : "stable_kernel_only";
    const detailPrefix = `selected=${selectedContext.length}; tiers=${selectionTiers ?? "unknown"}; kernel=${stableKernelBlocks.length}; brain=${brainCompiledBlocks.length}`;
    const detail = evidence === "route_fn_and_brain_context"
        ? `${detailPrefix}; learned route ran and non-seed brain-compiled context was selected`
        : evidence === "brain_context_only"
            ? `${detailPrefix}; non-seed brain-compiled context was selected without learned-route evidence`
            : evidence === "route_fn_only"
                ? `${detailPrefix}; learned route ran but selected context stayed inside the stable kernel`
                : `${detailPrefix}; selected context stayed inside the stable kernel`;
    return {
        selectedContextCount: selectedContext.length,
        stableKernelBlockCount: stableKernelBlocks.length,
        brainCompiledBlockCount: brainCompiledBlocks.length,
        stableKernelSources: sortedUniqueStrings(stableKernelBlocks.map((block) => block.source)),
        brainCompiledSources: sortedUniqueStrings(brainCompiledBlocks.map((block) => block.source)),
        selectionTiers,
        evidence,
        detail
    };
}
function buildAttachCompileStatus(result, observability, activePackId) {
    if (!result.ok) {
        return {
            ok: false,
            fallbackToStaticContext: result.fallbackToStaticContext,
            hardRequirementViolated: result.hardRequirementViolated,
            activePackId,
            usedLearnedRouteFn: null,
            routerIdentity: observability?.learnedRouteFn.routerIdentity ?? null,
            selectionDigest: null,
            initMode: observability?.initHandoff.initMode ?? null,
            handoffState: observability?.initHandoff.handoffState ?? null,
            seedSources: observability?.initHandoff.seedSources ?? [],
            contextAttribution: buildContextAttributionSummary({
                fallbackToStaticContext: result.fallbackToStaticContext,
                hardRequirementViolated: result.hardRequirementViolated,
                usedLearnedRouteFn: null
            }),
            timing: result.timing,
            notes: [],
            error: result.error
        };
    }
    const notes = [...result.compileResponse.diagnostics.notes];
    const contextAttribution = buildContextAttributionSummary({
        fallbackToStaticContext: false,
        hardRequirementViolated: false,
        usedLearnedRouteFn: result.compileResponse.diagnostics.usedLearnedRouteFn,
        selectedContext: result.compileResponse.selectedContext,
        notes
    });
    return {
        ok: true,
        fallbackToStaticContext: false,
        hardRequirementViolated: false,
        activePackId: result.activePackId,
        usedLearnedRouteFn: result.compileResponse.diagnostics.usedLearnedRouteFn,
        routerIdentity: result.compileResponse.diagnostics.routerIdentity,
        selectionDigest: result.compileResponse.diagnostics.selectionDigest,
        initMode: readDiagnosticNoteValue(notes, "init_mode=") ?? observability?.initHandoff.initMode ?? null,
        handoffState: readDiagnosticNoteValue(notes, "handoff_state=") ?? observability?.initHandoff.handoffState ?? null,
        seedSources: readDiagnosticNoteList(notes, "seed_sources=").length > 0
            ? readDiagnosticNoteList(notes, "seed_sources=")
            : observability?.initHandoff.seedSources ?? [],
        contextAttribution,
        timing: result.timing,
        notes,
        error: null
    };
}
function buildAttachSuccessSignals(input) {
    const signals = [];
    const activeTarget = input.observability?.target ?? null;
    if (input.inspection.active?.activationReady) {
        signals.push(`active_pack_ready:${input.inspection.active.packId}`);
    }
    if (activeTarget !== null) {
        signals.push(`active_workspace_snapshot:${activeTarget.workspaceSnapshot}`);
        if (activeTarget.eventRange.count === 0) {
            signals.push("awaiting_first_export");
        }
    }
    if (input.compile?.ok) {
        signals.push(`compile_ok:${input.compile.activePackId ?? "unknown"}`);
    }
    if (input.compile !== null) {
        signals.push(`context:${input.compile.contextAttribution.evidence}`);
        signals.push(`context_blocks:kernel=${input.compile.contextAttribution.stableKernelBlockCount},brain=${input.compile.contextAttribution.brainCompiledBlockCount}`);
    }
    if (input.compile?.handoffState !== null && input.compile?.handoffState !== undefined) {
        signals.push(`handoff:${input.compile.handoffState}`);
    }
    if (input.inspection.active?.routePolicy === "requires_learned_routing") {
        if (input.compile?.ok && input.compile.usedLearnedRouteFn === true && input.compile.routerIdentity !== null) {
            signals.push(`learned_route_compile_verified:${input.compile.routerIdentity}`);
        }
    }
    else if (input.compile?.ok) {
        signals.push("heuristic_compile_verified");
    }
    return signals;
}
function buildAttachStatusCompileInput(activationRoot, compile) {
    if (compile === false) {
        return null;
    }
    return {
        activationRoot,
        agentId: normalizeOptionalString(compile?.agentId) ?? `${DEFAULT_AGENT_ID}-attach-status`,
        message: normalizeOptionalString(compile?.message) ?? DEFAULT_ATTACH_STATUS_MESSAGE,
        ...(compile?.maxContextBlocks !== undefined ? { maxContextBlocks: compile.maxContextBlocks } : {}),
        ...(compile?.budgetStrategy !== undefined ? { budgetStrategy: compile.budgetStrategy } : {}),
        ...(compile?.maxContextChars !== undefined ? { maxContextChars: compile.maxContextChars } : {}),
        ...(compile?.mode !== undefined ? { mode: compile.mode } : {}),
        ...(compile?.compactionMode !== undefined ? { compactionMode: compile.compactionMode } : {}),
        runtimeHints: compile?.runtimeHints ?? [...DEFAULT_ATTACH_STATUS_RUNTIME_HINTS]
    };
}
export function describeAttachStatus(input) {
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const inspection = inspectActivationState(activationRoot);
    const activeObservability = inspection.active === null ? null : describeActivationObservability(activationRoot, "active");
    const compileInput = buildAttachStatusCompileInput(activationRoot, input.compile);
    const compile = compileInput === null
        ? null
        : buildAttachCompileStatus(compileRuntimeContext(compileInput), activeObservability, inspection.active?.packId ?? null);
    return {
        runtimeOwner: "openclaw",
        activationRoot,
        inspection,
        activeObservability,
        compile,
        landingBoundaries: structuredClone(OPENCLAW_LANDING_BOUNDARIES_V1),
        successSignals: buildAttachSuccessSignals({
            inspection,
            observability: activeObservability,
            compile
        })
    };
}
export function rollbackRuntimeAttach(input) {
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const updatedAt = normalizeIsoTimestamp(input.updatedAt, "updatedAt", new Date().toISOString());
    const dryRun = input.dryRun === true;
    const before = inspectActivationState(activationRoot, updatedAt);
    const findings = [...before.rollback.findings];
    const allowed = before.rollback.allowed;
    if (!allowed) {
        return {
            runtimeOwner: "openclaw",
            activationRoot,
            updatedAt,
            dryRun,
            allowed,
            findings,
            before: {
                activePackId: before.active?.packId ?? before.pointers.active?.packId ?? null,
                candidatePackId: before.candidate?.packId ?? before.pointers.candidate?.packId ?? null,
                previousPackId: before.previous?.packId ?? before.pointers.previous?.packId ?? null
            },
            after: null,
            restoredPackId: null,
            parkedCandidatePackId: null
        };
    }
    const after = dryRun
        ? before.rollback.nextPointers
        : rollbackActivePack(activationRoot, {
            updatedAt,
            reason: "runtime_attach_rollback"
        }).pointers;
    return {
        runtimeOwner: "openclaw",
        activationRoot,
        updatedAt,
        dryRun,
        allowed,
        findings,
        before: {
            activePackId: before.active?.packId ?? before.pointers.active?.packId ?? null,
            candidatePackId: before.candidate?.packId ?? before.pointers.candidate?.packId ?? null,
            previousPackId: before.previous?.packId ?? before.pointers.previous?.packId ?? null
        },
        after: after === null ? null : {
            activePackId: after.active?.packId ?? null,
            candidatePackId: after.candidate?.packId ?? null,
            previousPackId: after.previous?.packId ?? null
        },
        restoredPackId: after?.active?.packId ?? null,
        parkedCandidatePackId: after?.candidate?.packId ?? null
    };
}
function resolveBootstrapNormalizedEventExport(input) {
    const interactionEvents = [...(input.interactionEvents ?? [])];
    const feedbackEvents = [...(input.feedbackEvents ?? [])];
    if (input.normalizedEventExport !== undefined && (interactionEvents.length > 0 || feedbackEvents.length > 0)) {
        throw new Error("Provide normalizedEventExport or interactionEvents/feedbackEvents, not both");
    }
    const normalizedEventExport = input.normalizedEventExport !== undefined
        ? canonicalizeBootstrapNormalizedEventExport(input.normalizedEventExport)
        : buildNormalizedEventExport({
            interactionEvents,
            feedbackEvents
        });
    const validationErrors = validateNormalizedEventExport(normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(formatBootstrapNormalizedEventExportValidationError(normalizedEventExport, validationErrors));
    }
    return normalizedEventExport;
}
function canonicalizeBootstrapNormalizedEventExport(normalizedEventExport) {
    const interactionEvents = cloneBootstrapNormalizedEventArray(normalizedEventExport.interactionEvents, "interactionEvents");
    const feedbackEvents = cloneBootstrapNormalizedEventArray(normalizedEventExport.feedbackEvents, "feedbackEvents");
    try {
        return buildNormalizedEventExport({
            interactionEvents,
            feedbackEvents
        });
    }
    catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error("bootstrapRuntimeAttach could not reconstruct a safe normalized event export from the provided event arrays. " +
            "Repair the event payload or, for a first attach with no live events yet, pass empty arrays so bootstrap can self-boot. " +
            `Details: ${detail}`);
    }
}
function cloneBootstrapNormalizedEventArray(value, fieldName) {
    if (!Array.isArray(value)) {
        throw new Error(`bootstrapRuntimeAttach expected normalizedEventExport.${fieldName} to be an array. ` +
            "For a first attach with no live events yet, pass empty arrays or omit normalizedEventExport and use interactionEvents: [] / feedbackEvents: [].");
    }
    return structuredClone(value);
}
function formatBootstrapNormalizedEventExportValidationError(normalizedEventExport, validationErrors) {
    const details = validationErrors.join("; ");
    const zeroEventBootstrap = normalizedEventExport.interactionEvents.length === 0 &&
        normalizedEventExport.feedbackEvents.length === 0 &&
        normalizedEventExport.range.count === 0;
    if (zeroEventBootstrap) {
        return ("bootstrapRuntimeAttach could not derive a safe zero-event bootstrap export: " +
            `${details}. ` +
            "For a first attach with no live events yet, pass empty interaction/feedback arrays or an empty normalized export payload and bootstrap will self-boot.");
    }
    return ("bootstrapRuntimeAttach could not use the provided normalized event export: " +
        `${details}. ` +
        "Repair the event payload or pass raw interactionEvents/feedbackEvents so bootstrap can derive range and provenance itself.");
}
export function bootstrapRuntimeAttach(input) {
    const profileSelector = normalizeBootstrapProfileSelector(input.profileSelector);
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const packRoot = path.resolve(normalizeNonEmptyString(input.packRoot, "packRoot"));
    const normalizedEventExport = resolveBootstrapNormalizedEventExport(input);
    const builtAt = normalizeIsoTimestamp(input.builtAt, "builtAt", normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString());
    const activatedAt = normalizeIsoTimestamp(input.activatedAt, "activatedAt", builtAt);
    const teacherSupervisionArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
        normalizedEventExport,
        observedAt: activatedAt,
        ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {})
    });
    const descriptor = materializeCandidatePackFromNormalizedEventExport(packRoot, {
        packLabel: normalizeNonEmptyString(input.packLabel, "packLabel"),
        workspace: input.workspace,
        normalizedEventExport,
        teacherSupervisionArtifacts,
        learnedRouting: input.learnedRouting ?? true,
        builtAt,
        ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
        ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {}),
        ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {})
    });
    activatePack(activationRoot, packRoot, {
        updatedAt: activatedAt,
        reason: "attach_bootstrap"
    });
    const status = describeAttachStatus({
        activationRoot,
        ...(input.compile !== undefined ? { compile: input.compile } : {})
    });
    const currentProfile = describeCurrentProfileBrainStatus({
        activationRoot,
        updatedAt: activatedAt,
        ...(input.brainAttachmentPolicy !== undefined ? { brainAttachmentPolicy: input.brainAttachmentPolicy } : {}),
        ...(input.profileId !== undefined ? { profileId: input.profileId } : {})
    });
    return {
        runtimeOwner: "openclaw",
        profileSelector,
        operatorReadScope: "current_profile_only",
        activationRoot,
        packRoot,
        packId: descriptor.manifest.packId,
        normalizedEventExport,
        status,
        currentProfile,
        nextSteps: buildBootstrapRuntimeAttachNextSteps({
            activationRoot,
            profileSelector,
            currentProfile
        })
    };
}
function normalizeFingerprintEntries(values) {
    return uniqueStringsInOrder((values ?? [])
        .map((value) => normalizeOptionalString(value))
        .filter((value) => value !== undefined));
}
function buildRuntimeContextFingerprint(input) {
    const promptContextFingerprints = normalizeFingerprintEntries(input.turn.contextFingerprint?.promptContextFingerprints);
    const workspaceInjectionSurfaceDigest = input.turn.contextFingerprint?.workspaceInjectionSurface === undefined ||
        input.turn.contextFingerprint?.workspaceInjectionSurface === null
        ? null
        : checksumJsonPayload(input.turn.contextFingerprint.workspaceInjectionSurface);
    const promptContextDigest = promptContextFingerprints.length === 0 && workspaceInjectionSurfaceDigest === null
        ? null
        : checksumJsonPayload({
            promptContextFingerprints,
            workspaceInjectionSurfaceDigest
        });
    const runtimeHints = normalizeFingerprintEntries(input.turn.runtimeHints);
    const runtimeHintsDigest = runtimeHints.length === 0 ? null : checksumJsonPayload(runtimeHints);
    const profileId = normalizeOptionalString(input.turn.profileId);
    const profileLineage = uniqueStringsInOrder([
        "host:openclaw",
        `profile:${input.profileSelector}`,
        profileId === undefined ? undefined : `profile_id:${profileId}`,
        `attachment_policy:${input.brainAttachmentPolicy}`,
        ...normalizeFingerprintEntries(input.turn.contextFingerprint?.profileLineage)
    ].filter((value) => value !== undefined));
    const sessionLineage = uniqueStringsInOrder([
        `session:${input.turn.sessionId}`,
        `channel:${input.turn.channel}`,
        `source_stream:${input.sourceStream}`,
        ...normalizeFingerprintEntries(input.turn.contextFingerprint?.sessionLineage)
    ]);
    const brainLineage = uniqueStringsInOrder([
        `brain_status:${input.brainStatus}`,
        `active_pack:${input.activePackId ?? "none"}`,
        `router:${input.routerIdentity ?? "none"}`,
        `used_learned_route_fn:${input.usedLearnedRouteFn === null ? "unknown" : String(input.usedLearnedRouteFn)}`
    ]);
    const profileLineageDigest = checksumJsonPayload(profileLineage);
    const sessionLineageDigest = checksumJsonPayload(sessionLineage);
    const brainLineageDigest = checksumJsonPayload(brainLineage);
    return {
        selectionDigest: input.selectionDigest,
        promptContextDigest,
        promptContextFingerprints,
        workspaceInjectionSurfaceDigest,
        runtimeHintsDigest,
        runtimeHints,
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
function buildRuntimeTurnAttribution(input) {
    const profileSelector = normalizeRuntimeProfileSelector(input.turn.profileSelector, "profileSelector");
    const brainAttachmentPolicy = normalizeBrainAttachmentPolicy(input.turn.brainAttachmentPolicy);
    const profileId = normalizeOptionalString(input.turn.profileId) ?? null;
    if (input.compileResult.ok) {
        const notes = input.compileResult.compileResponse.diagnostics.notes;
        const contextAttribution = buildContextAttributionSummary({
            fallbackToStaticContext: false,
            hardRequirementViolated: false,
            usedLearnedRouteFn: input.compileResult.compileResponse.diagnostics.usedLearnedRouteFn,
            selectedContext: input.compileResult.compileResponse.selectedContext,
            notes
        });
        return {
            hostRuntimeOwner: "openclaw",
            profileSelector,
            profileId,
            brainAttachmentPolicy,
            brainStatus: "serving_active_pack",
            activePackId: input.compileResult.activePackId,
            usedLearnedRouteFn: input.compileResult.compileResponse.diagnostics.usedLearnedRouteFn,
            routerIdentity: input.compileResult.compileResponse.diagnostics.routerIdentity,
            selectionDigest: input.compileResult.compileResponse.diagnostics.selectionDigest,
            selectionTiers: readDiagnosticNoteValue(notes, "selection_tiers="),
            contextFingerprint: buildRuntimeContextFingerprint({
                turn: input.turn,
                sourceStream: input.sourceStream,
                profileSelector,
                brainAttachmentPolicy,
                brainStatus: "serving_active_pack",
                activePackId: input.compileResult.activePackId,
                usedLearnedRouteFn: input.compileResult.compileResponse.diagnostics.usedLearnedRouteFn,
                routerIdentity: input.compileResult.compileResponse.diagnostics.routerIdentity,
                selectionDigest: input.compileResult.compileResponse.diagnostics.selectionDigest
            }),
            contextEvidence: contextAttribution.evidence === "unprobed" ? null : contextAttribution.evidence
        };
    }
    const contextAttribution = buildContextAttributionSummary({
        fallbackToStaticContext: input.compileResult.fallbackToStaticContext,
        hardRequirementViolated: input.compileResult.hardRequirementViolated,
        usedLearnedRouteFn: null
    });
    return {
        hostRuntimeOwner: "openclaw",
        profileSelector,
        profileId,
        brainAttachmentPolicy,
        brainStatus: input.compileResult.hardRequirementViolated ? "hard_fail" : "fail_open_static_context",
        activePackId: null,
        usedLearnedRouteFn: null,
        routerIdentity: null,
        selectionDigest: null,
        selectionTiers: null,
        contextFingerprint: buildRuntimeContextFingerprint({
            turn: input.turn,
            sourceStream: input.sourceStream,
            profileSelector,
            brainAttachmentPolicy,
            brainStatus: input.compileResult.hardRequirementViolated ? "hard_fail" : "fail_open_static_context",
            activePackId: null,
            usedLearnedRouteFn: null,
            routerIdentity: null,
            selectionDigest: null
        }),
        contextEvidence: contextAttribution.evidence === "unprobed" ? null : contextAttribution.evidence
    };
}
function buildCompileInteractionEvent(input) {
    if (!input.compileResult.ok) {
        return null;
    }
    const sequence = input.nextSequence(input.turn.compile?.sequence);
    const eventId = normalizeOptionalString(input.turn.compile?.eventId) ?? deterministicEventId({
        channel: input.turn.channel,
        createdAt: input.createdAt,
        kind: "memory_compiled",
        packId: input.compileResult.compileResponse.packId,
        sequence,
        sessionId: input.turn.sessionId,
        source: input.sourceStream
    });
    return createInteractionEvent({
        eventId,
        agentId: input.agentId,
        sessionId: input.turn.sessionId,
        channel: input.turn.channel,
        sequence,
        kind: "memory_compiled",
        createdAt: input.createdAt,
        source: {
            runtimeOwner: "openclaw",
            stream: input.sourceStream
        },
        semantic: buildInteractionSemanticMetadata("runtime_turn", "memory_compiled"),
        packId: input.compileResult.compileResponse.packId,
        principal: buildRuntimeSelfPrincipal(input.turn),
        attribution: input.attribution
    });
}
function buildDeliveryInteractionEvent(input) {
    if (input.turn.delivery === undefined || input.turn.delivery === null || input.turn.delivery === false) {
        return null;
    }
    const delivery = typeof input.turn.delivery === "object" ? input.turn.delivery : {};
    const createdAt = normalizeIsoTimestamp(delivery.createdAt, "delivery.createdAt", input.defaultCreatedAt);
    const sequence = input.nextSequence(delivery.sequence);
    const messageId = normalizeOptionalString(delivery.messageId);
    const eventId = normalizeOptionalString(delivery.eventId) ?? deterministicEventId({
        channel: input.turn.channel,
        createdAt,
        kind: "message_delivered",
        messageId: messageId ?? null,
        packId: input.compileResult.ok ? input.compileResult.compileResponse.packId : null,
        sequence,
        sessionId: input.turn.sessionId,
        source: input.sourceStream
    });
    return createInteractionEvent({
        eventId,
        agentId: input.agentId,
        sessionId: input.turn.sessionId,
        channel: input.turn.channel,
        sequence,
        kind: "message_delivered",
        createdAt,
        source: {
            runtimeOwner: "openclaw",
            stream: input.sourceStream
        },
        semantic: buildInteractionSemanticMetadata("runtime_turn", "message_delivered"),
        attribution: input.attribution,
        ...(input.compileResult.ok ? { packId: input.compileResult.compileResponse.packId } : {}),
        ...(messageId !== undefined ? { messageId } : {})
    });
}
function buildFeedbackEvents(input) {
    const feedbackItems = input.turn.feedback ?? [];
    return feedbackItems.map((item, index) => {
        if (item === null) {
            throw new Error(`feedback[${index}] must be an object`);
        }
        const content = normalizeNonEmptyString(item.content, `feedback[${index}].content`);
        const createdAt = normalizeIsoTimestamp(item.createdAt, `feedback[${index}].createdAt`, input.defaultCreatedAt);
        const sequence = input.nextSequence(item.sequence);
        const kind = item.kind === undefined || item.kind === null ? classifyFeedbackKind(content) : item.kind;
        if (!isFeedbackKind(kind)) {
            throw new Error(`feedback[${index}].kind must be correction, teaching, approval, or suppression`);
        }
        const messageId = normalizeOptionalString(item.messageId);
        const eventId = normalizeOptionalString(item.eventId) ?? deterministicEventId({
            channel: input.turn.channel,
            content,
            createdAt,
            kind,
            messageId: messageId ?? null,
            sequence,
            sessionId: input.turn.sessionId,
            source: input.sourceStream
        });
        const relatedInteractionId = normalizeOptionalString(item.relatedInteractionId) ?? input.compileInteraction?.eventId;
        const principal = resolveRuntimeFeedbackPrincipal({
            turn: input.turn,
            feedback: item,
            ...(relatedInteractionId === undefined ? {} : { relatedInteractionId }),
            ...(messageId === undefined ? {} : { messageId })
        });
        return createFeedbackEvent({
            eventId,
            agentId: input.agentId,
            sessionId: input.turn.sessionId,
            channel: input.turn.channel,
            sequence,
            kind,
            createdAt,
            source: {
                runtimeOwner: "openclaw",
                stream: input.sourceStream
            },
            content,
            semantic: buildFeedbackSemanticMetadata("runtime_turn", kind, content),
            attribution: input.attribution,
            ...(messageId !== undefined ? { messageId } : {}),
            ...(principal === undefined ? {} : { principal }),
            ...(relatedInteractionId !== undefined ? { relatedInteractionId } : {})
        });
    });
}
export function buildNormalizedRuntimeEventExport(turn, compileResult) {
    const agentId = normalizeOptionalString(turn.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
    const profileSelector = normalizeRuntimeProfileSelector(turn.profileSelector, "profileSelector");
    const sessionId = normalizeNonEmptyString(turn.sessionId, "sessionId");
    const channel = normalizeNonEmptyString(turn.channel, "channel");
    const sourceStream = normalizeOptionalString(turn.sourceStream) ?? `openclaw/runtime/${channel}`;
    const compileCreatedAt = normalizeIsoTimestamp(turn.compile?.createdAt, "compile.createdAt", turn.createdAt);
    const nextSequence = nextSequenceFactory(turn.sequenceStart ?? 1);
    const normalizedTurn = {
        ...turn,
        agentId,
        profileSelector,
        channel,
        sessionId
    };
    const attribution = buildRuntimeTurnAttribution({
        turn: normalizedTurn,
        compileResult,
        sourceStream
    });
    const compileInteraction = buildCompileInteractionEvent({
        turn: normalizedTurn,
        compileResult,
        attribution,
        sourceStream,
        nextSequence,
        createdAt: compileCreatedAt,
        agentId
    });
    const feedbackEvents = buildFeedbackEvents({
        turn: normalizedTurn,
        attribution,
        sourceStream,
        nextSequence,
        defaultCreatedAt: compileCreatedAt,
        compileInteraction,
        compileResult,
        agentId
    });
    const deliveryInteraction = buildDeliveryInteractionEvent({
        turn: normalizedTurn,
        compileResult,
        attribution,
        sourceStream,
        nextSequence,
        defaultCreatedAt: compileCreatedAt,
        agentId
    });
    const interactionEvents = [compileInteraction, deliveryInteraction].filter(isPresent);
    if (interactionEvents.length === 0 && feedbackEvents.length === 0) {
        throw new Error("runtime turn did not produce any normalized events");
    }
    const normalizedEventExport = buildNormalizedEventExport({
        interactionEvents,
        feedbackEvents
    });
    const validationErrors = validateNormalizedEventExport(normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    return normalizedEventExport;
}
export function writeRuntimeEventExportBundle(turn, normalizedEventExport) {
    if (turn.export === undefined || turn.export === null) {
        return {
            ok: true,
            wroteBundle: false,
            normalizedEventExport
        };
    }
    const rootDir = normalizeNonEmptyString(turn.export.rootDir, "export.rootDir");
    const exportName = normalizeOptionalString(turn.export.exportName) ??
        `${turn.sessionId}-${normalizedEventExport.range.start}-${normalizedEventExport.range.end}`;
    const exportedAt = normalizeIsoTimestamp(turn.export.exportedAt, "export.exportedAt", normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString());
    const resolvedRoot = path.resolve(rootDir);
    const manifest = buildRuntimeEventExportBundleManifest({
        exportName,
        exportedAt,
        payloadPath: RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.payload,
        normalizedEventExport
    });
    return writeNormalizedEventExportBundleFiles({
        rootDir: resolvedRoot,
        manifest,
        normalizedEventExport
    });
}
export function writeScannedEventExportBundle(input) {
    const built = buildNormalizedEventExportFromScannedEvents(input.scannedEventExport);
    if (!built.ok) {
        return built;
    }
    const rootDir = normalizeNonEmptyString(input.rootDir, "rootDir");
    const exportName = normalizeOptionalString(input.exportName) ??
        `${built.scanner.scannerId}-${built.scanner.lane}-${built.normalizedEventExport.range.start}-${built.normalizedEventExport.range.end}`;
    const exportedAt = normalizeIsoTimestamp(input.exportedAt, "exportedAt", built.scanner.producedAt);
    const manifest = buildRuntimeEventExportBundleManifest({
        exportName,
        exportedAt,
        payloadPath: RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.payload,
        normalizedEventExport: built.normalizedEventExport,
        scanner: built.scanner
    });
    return writeNormalizedEventExportBundleFiles({
        rootDir,
        manifest,
        normalizedEventExport: built.normalizedEventExport
    });
}
export function runRuntimeTurn(turn, options = {}) {
    const warnings = [];
    const serveRouteBreadcrumbs = buildRunRuntimeTurnServeRouteBreadcrumbs();
    const agentId = normalizeOptionalString(turn.agentId);
    const compileInput = {
        activationRoot: (options.activationRoot ?? turn.activationRoot),
        message: turn.userMessage,
        ...(agentId !== undefined ? { agentId } : {}),
        ...(turn.maxContextBlocks !== undefined ? { maxContextBlocks: turn.maxContextBlocks } : {}),
        ...(turn.budgetStrategy !== undefined ? { budgetStrategy: turn.budgetStrategy } : {}),
        ...(turn.mode !== undefined ? { mode: turn.mode } : {}),
        ...(turn.selectionMode !== undefined ? { selectionMode: turn.selectionMode } : {}),
        ...(turn.runtimeHints !== undefined ? { runtimeHints: turn.runtimeHints } : {}),
        _suppressServeLog: true
    };
    const compileResult = compileRuntimeContext(compileInput);
    const serveLoggedAt = turn.compile?.createdAt ?? turn.createdAt ?? new Date().toISOString();
    if (!compileResult.ok && compileResult.hardRequirementViolated) {
        try {
            appendServeTimeRouteDecisionLog({
                activationRoot: compileResult.activationRoot,
                turn,
                compileResult,
                recordedAt: serveLoggedAt,
                breadcrumbs: serveRouteBreadcrumbs
            });
        }
        catch (error) {
            console.warn(`[openclawbrain] serve-time route decision log failed before hard-fail throw: ${toErrorMessage(error)} ` +
                `(activationRoot=${compileResult.activationRoot}, sessionId=${turn.sessionId}, channel=${turn.channel})`);
        }
        throw new Error(compileResult.error);
    }
    try {
        const normalizedEventExport = buildNormalizedRuntimeEventExport(turn, compileResult);
        try {
            const compileEvent = normalizedEventExport.interactionEvents.find((event) => event.kind === "memory_compiled");
            appendServeTimeRouteDecisionLog({
                activationRoot: compileResult.activationRoot,
                turn,
                compileResult,
                normalizedEventExport,
                recordedAt: compileEvent?.createdAt ?? serveLoggedAt,
                breadcrumbs: serveRouteBreadcrumbs
            });
        }
        catch (error) {
            warnings.push(buildServeRouteLogFailOpenWarning("runRuntimeTurn", error));
            if (compileResult.ok) {
                compileResult.compileResponse.diagnostics.notes = uniqueNotes([
                    ...compileResult.compileResponse.diagnostics.notes,
                    ...buildServeRouteLogFailOpenNotes("runRuntimeTurn", error)
                ]);
            }
        }
        try {
            const eventExport = writeRuntimeEventExportBundle(turn, normalizedEventExport);
            return {
                ...compileResult,
                eventExport,
                warnings
            };
        }
        catch (error) {
            if (options.failOpen === false) {
                throw error;
            }
            warnings.push(toErrorMessage(error));
            return {
                ...compileResult,
                eventExport: {
                    ok: false,
                    wroteBundle: false,
                    error: toErrorMessage(error)
                },
                warnings
            };
        }
    }
    catch (error) {
        warnings.push(toErrorMessage(error));
        return {
            ...compileResult,
            eventExport: {
                ok: false,
                wroteBundle: false,
                error: toErrorMessage(error)
            },
            warnings
        };
    }
}
export function runContinuousProductLoopTurn(input) {
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const loopRoot = path.resolve(normalizeNonEmptyString(input.loopRoot, "loopRoot"));
    const failOpen = input.failOpen !== false;
    const currentState = cloneContinuousProductLoopState(input.state ??
        createContinuousProductLoopState({
            activationRoot,
            loopRoot
        }));
    currentState.activationRoot = activationRoot;
    currentState.loopRoot = loopRoot;
    const activeBeforeTurn = syncContinuousActivePack(currentState);
    const compileActiveVersion = activeBeforeTurn?.version ?? 0;
    const compileActivePackId = activeBeforeTurn?.packId ?? null;
    const turn = withContinuousTurnExport(input.turn, loopRoot);
    const turnResult = runRuntimeTurn(turn, {
        activationRoot,
        failOpen
    });
    const learningWarnings = [];
    let supervision = null;
    const learning = {
        warnings: learningWarnings,
        supervisionDigest: null,
        bridgeDigest: null,
        selectedSliceIds: [],
        materializationJobId: null,
        materializationReason: null,
        materializationLane: null,
        candidateRootDir: null,
        candidatePack: currentState.candidatePack === null ? null : cloneContinuousProductLoopPackVersion(currentState.candidatePack),
        runtimePlasticity: currentState.runtimePlasticity === null ? null : structuredClone(currentState.runtimePlasticity),
        promotionAllowed: false,
        promotionFindings: [],
        promoted: false
    };
    if (!turnResult.eventExport.ok) {
        learningWarnings.push(`continuous learner skipped: ${turnResult.eventExport.error}`);
        return {
            runtimeOwner: "openclaw",
            compileActiveVersion,
            compileActivePackId,
            turn: turnResult,
            supervision,
            learning,
            state: cloneContinuousProductLoopState(currentState)
        };
    }
    const normalizedEventExport = turnResult.eventExport.normalizedEventExport;
    supervision = buildCanonicalSupervision(normalizedEventExport);
    learning.supervisionDigest = supervision.supervisionDigest;
    currentState.lastSupervision = cloneCanonicalSupervision(supervision);
    const mergedHistory = mergeRuntimeEventHistory(currentState, normalizedEventExport);
    currentState.interactionEvents = mergedHistory.interactionEvents;
    currentState.feedbackEvents = mergedHistory.feedbackEvents;
    const compileStructuralSignals = turnResult.ok ? turnResult.compileResponse.diagnostics.structuralSignals : undefined;
    try {
        let activeBeforePack = null;
        try {
            activeBeforePack = loadPackFromActivation(activationRoot, "active", { requireActivationReady: true });
        }
        catch {
            activeBeforePack = null;
        }
        const learnerResult = advanceAlwaysOnLearningRuntime({
            packLabel: input.packLabel,
            workspace: input.workspace,
            interactionEvents: currentState.interactionEvents,
            feedbackEvents: currentState.feedbackEvents,
            learnedRouting: input.learnedRouting ?? true,
            state: currentState.learner,
            builtAt: normalizeIsoTimestamp(input.candidateBuiltAt, "candidateBuiltAt", normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt),
            ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
            ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {}),
            ...(compileStructuralSignals !== undefined ? { compileStructuralSignals } : {}),
            ...(input.sparseFeedback !== undefined ? { sparseFeedback: input.sparseFeedback } : {}),
            ...(input.liveSliceSize !== undefined ? { liveSliceSize: input.liveSliceSize } : {}),
            ...(input.backfillSliceSize !== undefined ? { backfillSliceSize: input.backfillSliceSize } : {}),
            ...(input.cadence !== undefined ? { cadence: input.cadence } : {}),
            activationRoot
        });
        currentState.learner = structuredClone(learnerResult.state);
        currentState.runtimePlasticity = learnerResult.state.runtimePlasticity === null ? null : structuredClone(learnerResult.state.runtimePlasticity);
        learning.runtimePlasticity = currentState.runtimePlasticity === null ? null : structuredClone(currentState.runtimePlasticity);
        learning.bridgeDigest = learnerResult.bridge.bridgeDigest;
        learning.selectedSliceIds = learnerResult.selectedSlices.map((slice) => slice.sliceId);
        learning.materializationJobId = learnerResult.materialization?.jobId ?? null;
        learning.materializationReason = learnerResult.materialization?.reason ?? null;
        learning.materializationLane = learnerResult.materialization?.lane ?? null;
        if (learnerResult.materialization !== null) {
            const candidatePack = registerPackVersion(currentState, buildLearningCandidateTarget(learnerResult.materialization.candidate));
            const candidateRootDir = buildContinuousPackRoot(loopRoot, candidatePack);
            const descriptor = materializeAlwaysOnLearningCandidatePack(candidateRootDir, learnerResult.materialization);
            const candidateTarget = describePackCompileTarget(descriptor);
            learning.candidateRootDir = candidateRootDir;
            learning.candidatePack = cloneContinuousProductLoopPackVersion(candidatePack);
            currentState.candidatePack = cloneContinuousProductLoopPackVersion(candidatePack);
            const stagedAt = normalizeIsoTimestamp(input.stageUpdatedAt, "stageUpdatedAt", descriptor.manifest.provenance.builtAt);
            try {
                appendLearningUpdateLogs({
                    activationRoot,
                    materialization: learnerResult.materialization,
                    activeBeforePack,
                    candidateDescriptor: descriptor
                });
            }
            catch (error) {
                learningWarnings.push(`learning spine update logs failed: ${toErrorMessage(error)}`);
            }
            stageCandidatePack(activationRoot, candidateRootDir, {
                updatedAt: stagedAt,
                reason: `stage_candidate:${learnerResult.materialization.reason}:${learnerResult.materialization.lane}`
            });
            const stagedInspection = inspectActivationState(activationRoot, stagedAt);
            learning.promotionAllowed = stagedInspection.promotion.allowed;
            learning.promotionFindings = [...stagedInspection.promotion.findings];
            if ((input.autoPromote ?? true) && stagedInspection.promotion.allowed) {
                const promotedAt = normalizeIsoTimestamp(input.promoteUpdatedAt, "promoteUpdatedAt", stagedAt);
                promoteCandidatePack(activationRoot, {
                    updatedAt: promotedAt,
                    reason: `auto_promote:${learnerResult.materialization.reason}:${learnerResult.materialization.lane}`
                });
                currentState.promotionCount += 1;
                currentState.candidatePack = null;
                learning.promoted = true;
                const activePack = registerPackVersion(currentState, candidateTarget);
                currentState.currentActivePack = cloneContinuousProductLoopPackVersion(activePack);
                currentState.activePackVersion = activePack.version;
                syncContinuousActivePack(currentState);
            }
        }
    }
    catch (error) {
        if (!failOpen) {
            throw error;
        }
        learningWarnings.push(`continuous learner failed open: ${toErrorMessage(error)}`);
    }
    return {
        runtimeOwner: "openclaw",
        compileActiveVersion,
        compileActivePackId,
        turn: turnResult,
        supervision,
        learning,
        state: cloneContinuousProductLoopState(currentState)
    };
}
function ensureRecordedSessionTrace(trace) {
    if (trace.contract !== RECORDED_SESSION_TRACE_CONTRACT) {
        throw new Error(`${RECORDED_SESSION_TRACE_CONTRACT} contract is required`);
    }
    normalizeNonEmptyString(trace.traceId, "traceId");
    normalizeIsoTimestamp(trace.recordedAt, "recordedAt");
    normalizeIsoTimestamp(trace.bundleBuiltAt, "bundleBuiltAt");
    normalizeNonEmptyString(trace.sessionId, "sessionId");
    normalizeNonEmptyString(trace.channel, "channel");
    normalizeNonEmptyString(trace.sourceStream, "sourceStream");
    normalizeIsoTimestamp(trace.seedBuiltAt, "seedBuiltAt");
    normalizeIsoTimestamp(trace.seedActivatedAt, "seedActivatedAt");
    if (trace.privacy.sanitized !== true) {
        throw new Error("recorded session trace must be explicitly sanitized");
    }
    if (trace.seedCues.length === 0) {
        throw new Error("recorded session trace requires at least one seed cue");
    }
    if (trace.turns.length === 0) {
        throw new Error("recorded session trace requires at least one turn");
    }
    for (const [index, cue] of trace.seedCues.entries()) {
        normalizeNonEmptyString(cue.cueId, `seedCues[${index}].cueId`);
        normalizeIsoTimestamp(cue.createdAt, `seedCues[${index}].createdAt`);
        normalizeNonEmptyString(cue.content, `seedCues[${index}].content`);
    }
    for (const [index, turn] of trace.turns.entries()) {
        normalizeIsoTimestamp(turn.createdAt, `turns[${index}].createdAt`);
        normalizeNonEmptyString(turn.userMessage, `turns[${index}].userMessage`);
        if ((turn.expectedContextPhrases ?? []).length === 0) {
            throw new Error(`turns[${index}].expectedContextPhrases must include at least one phrase`);
        }
        for (const [feedbackIndex, feedback] of (turn.feedback ?? []).entries()) {
            normalizeIsoTimestamp(feedback.createdAt, `turns[${index}].feedback[${feedbackIndex}].createdAt`);
            normalizeNonEmptyString(feedback.content, `turns[${index}].feedback[${feedbackIndex}].content`);
        }
    }
}
function padReplayNumber(value) {
    return String(value).padStart(2, "0");
}
function replayTurnId(traceId, index, explicitValue) {
    return normalizeOptionalString(explicitValue) ?? `${traceId}-turn-${padReplayNumber(index + 1)}`;
}
function replaySequenceStart(index) {
    return 1_000 + index * 10;
}
function replayMessageId(turnId) {
    return `msg-${turnId}`;
}
function addMinutes(value, minutes) {
    return new Date(Date.parse(value) + minutes * 60_000).toISOString();
}
function normalizeReplayPhrase(value) {
    return value.toLowerCase().replace(/\s+/gu, " ").trim();
}
function hasReplayPhrase(texts, phrase) {
    const normalizedPhrase = normalizeReplayPhrase(phrase);
    return texts.some((text) => normalizeReplayPhrase(text).includes(normalizedPhrase));
}
function uniqueStringsInOrder(values) {
    const seen = new Set();
    const unique = [];
    for (const value of values) {
        if (seen.has(value)) {
            continue;
        }
        seen.add(value);
        unique.push(value);
    }
    return unique;
}
function buildRecordedSessionSeedExport(trace) {
    const agentId = normalizeOptionalString(trace.agentId) ?? DEFAULT_AGENT_ID;
    const seedSessionId = `${trace.sessionId}-seed`;
    const sourceStream = `${trace.sourceStream}/seed`;
    const interactionEvents = [];
    const feedbackEvents = [];
    let sequence = 1;
    for (const cue of trace.seedCues) {
        const interactionEventId = `${seedSessionId}:${cue.cueId}:interaction`;
        const feedbackEventId = `${seedSessionId}:${cue.cueId}:feedback`;
        const interaction = createInteractionEvent({
            eventId: interactionEventId,
            agentId,
            sessionId: seedSessionId,
            channel: trace.channel,
            sequence,
            kind: "operator_override",
            createdAt: cue.createdAt,
            source: {
                runtimeOwner: "openclaw",
                stream: sourceStream
            },
            semantic: buildInteractionSemanticMetadata("recorded_session_seed", "operator_override"),
            messageId: `${cue.cueId}-seed-message`
        });
        sequence += 1;
        const feedback = createFeedbackEvent({
            eventId: feedbackEventId,
            agentId,
            sessionId: seedSessionId,
            channel: trace.channel,
            sequence,
            kind: cue.kind ?? "teaching",
            createdAt: addMinutes(cue.createdAt, 1),
            source: {
                runtimeOwner: "openclaw",
                stream: sourceStream
            },
            content: cue.content,
            semantic: buildFeedbackSemanticMetadata("recorded_session_seed", cue.kind ?? "teaching", cue.content),
            relatedInteractionId: interaction.eventId
        });
        sequence += 1;
        interactionEvents.push(interaction);
        feedbackEvents.push(feedback);
    }
    return buildNormalizedEventExport({
        interactionEvents,
        feedbackEvents
    });
}
function recordedSessionFixtureBase(trace) {
    const traceHash = checksumJsonPayload(trace);
    const turns = trace.turns.map((turn, index) => {
        const turnId = replayTurnId(trace.traceId, index, turn.turnId);
        const sequenceStart = replaySequenceStart(index);
        return {
            turnId,
            turn: {
                ...(trace.agentId !== undefined && trace.agentId !== null ? { agentId: trace.agentId } : {}),
                sessionId: trace.sessionId,
                channel: trace.channel,
                sourceStream: trace.sourceStream,
                userMessage: turn.userMessage,
                createdAt: turn.createdAt,
                sequenceStart,
                maxContextBlocks: 3,
                mode: "heuristic",
                ...(turn.runtimeHints !== undefined ? { runtimeHints: [...turn.runtimeHints] } : {}),
                compile: {
                    createdAt: turn.createdAt,
                    sequence: sequenceStart
                },
                delivery: turn.deliveredAt === undefined || turn.deliveredAt === null
                    ? false
                    : {
                        createdAt: turn.deliveredAt,
                        sequence: sequenceStart + 1,
                        messageId: replayMessageId(turnId)
                    },
                feedback: (turn.feedback ?? []).map((feedback, feedbackIndex) => ({
                    createdAt: feedback.createdAt,
                    content: feedback.content,
                    sequence: sequenceStart + 2 + feedbackIndex,
                    kind: feedback.kind ?? null
                }))
            },
            expectedContextPhrases: [...turn.expectedContextPhrases],
            minimumPhraseHits: Math.max(1, turn.minimumPhraseHits ?? turn.expectedContextPhrases.length)
        };
    });
    return {
        contract: RECORDED_SESSION_FIXTURE_CONTRACT,
        traceId: trace.traceId,
        source: trace.source,
        recordedAt: trace.recordedAt,
        bundleBuiltAt: trace.bundleBuiltAt,
        traceHash,
        privacy: {
            sanitized: true,
            notes: [...trace.privacy.notes]
        },
        workspace: {
            workspaceId: trace.workspace.workspaceId,
            snapshotId: trace.workspace.snapshotId,
            capturedAt: trace.workspace.capturedAt,
            rootDir: trace.workspace.rootDir,
            ...(trace.workspace.branch !== undefined ? { branch: trace.workspace.branch } : {}),
            revision: trace.workspace.revision,
            ...(trace.workspace.labels !== undefined ? { labels: [...trace.workspace.labels] } : {})
        },
        seedBuiltAt: trace.seedBuiltAt,
        seedActivatedAt: trace.seedActivatedAt,
        seedExport: buildRecordedSessionSeedExport(trace),
        turns
    };
}
export function buildRecordedSessionReplayFixture(trace) {
    ensureRecordedSessionTrace(trace);
    const base = recordedSessionFixtureBase(trace);
    return {
        ...base,
        fixtureHash: checksumJsonPayload(base)
    };
}
function recordedSessionReplayFixtureBase(fixture) {
    return {
        contract: RECORDED_SESSION_FIXTURE_CONTRACT,
        traceId: fixture.traceId,
        source: fixture.source,
        recordedAt: fixture.recordedAt,
        bundleBuiltAt: fixture.bundleBuiltAt,
        traceHash: fixture.traceHash,
        privacy: {
            sanitized: true,
            notes: [...fixture.privacy.notes]
        },
        workspace: {
            workspaceId: fixture.workspace.workspaceId,
            snapshotId: fixture.workspace.snapshotId,
            capturedAt: fixture.workspace.capturedAt,
            rootDir: fixture.workspace.rootDir,
            ...(fixture.workspace.branch !== undefined ? { branch: fixture.workspace.branch } : {}),
            revision: fixture.workspace.revision,
            ...(fixture.workspace.labels !== undefined ? { labels: [...fixture.workspace.labels] } : {})
        },
        seedBuiltAt: fixture.seedBuiltAt,
        seedActivatedAt: fixture.seedActivatedAt,
        seedExport: fixture.seedExport,
        turns: fixture.turns.map((turn) => ({
            turnId: turn.turnId,
            turn: structuredClone(turn.turn),
            expectedContextPhrases: [...turn.expectedContextPhrases],
            minimumPhraseHits: turn.minimumPhraseHits
        }))
    };
}
function buildReplayTurnScore(input) {
    const phraseHits = input.expectedContextPhrases.filter((phrase) => hasReplayPhrase(input.texts, phrase));
    const missedPhrases = input.expectedContextPhrases.filter((phrase) => !phraseHits.includes(phrase));
    const compileScore = input.compileOk ? 40 : 0;
    const phraseScore = input.expectedContextPhrases.length === 0 ? 60 : Math.round((phraseHits.length / input.expectedContextPhrases.length) * 60);
    return {
        phraseHits,
        missedPhrases,
        qualityScore: Math.min(100, compileScore + phraseScore)
    };
}
function buildRecordedSessionTurnObservability(result) {
    if (!result.eventExport.ok) {
        return {
            scanPolicy: null,
            scanSurfaces: [],
            humanLabelCount: 0,
            selfLabelCount: 0,
            totalEventCount: 0,
            attributedEventCount: 0,
            selectionDigestCount: 0,
            freshestSourceStream: null,
            freshestCreatedAt: null
        };
    }
    const observability = describeNormalizedEventExportObservability(result.eventExport.normalizedEventExport);
    const freshestSource = observability.supervisionFreshnessBySource[0] ?? null;
    return {
        scanPolicy: observability.learningSurface.scanPolicy,
        scanSurfaces: [...observability.learningSurface.scanSurfaces],
        humanLabelCount: observability.learningSurface.humanLabelCount,
        selfLabelCount: observability.learningSurface.selfLabelCount,
        totalEventCount: observability.attributionCoverage.totalEventCount,
        attributedEventCount: observability.attributionCoverage.attributedEventCount,
        selectionDigestCount: observability.attributionCoverage.selectionDigestCount,
        freshestSourceStream: observability.teacherFreshness.sourceStream ?? freshestSource?.sourceStream ?? null,
        freshestCreatedAt: observability.teacherFreshness.freshestCreatedAt ?? freshestSource?.freshestCreatedAt ?? null
    };
}
function buildRecordedSessionTurnReport(turnFixture, result, options) {
    const compileOk = result.ok;
    const selectedContextTexts = compileOk ? result.compileResponse.selectedContext.map((block) => block.text) : [];
    const selectedContextIds = compileOk ? result.compileResponse.selectedContext.map((block) => block.id) : [];
    const scoring = buildReplayTurnScore({
        compileOk,
        texts: selectedContextTexts,
        expectedContextPhrases: turnFixture.expectedContextPhrases
    });
    const observability = buildRecordedSessionTurnObservability(result);
    const eventExportDigest = result.eventExport.ok === true ? result.eventExport.normalizedEventExport.provenance.exportDigest : null;
    return {
        turnId: turnFixture.turnId,
        compileOk,
        fallbackToStaticContext: result.fallbackToStaticContext,
        hardRequirementViolated: result.hardRequirementViolated,
        activePackId: result.ok ? result.activePackId : null,
        usedLearnedRouteFn: result.ok ? result.compileResponse.diagnostics.usedLearnedRouteFn : false,
        routerIdentity: result.ok ? result.compileResponse.diagnostics.routerIdentity : null,
        selectionDigest: result.ok ? result.compileResponse.diagnostics.selectionDigest : null,
        selectedContextIds,
        selectedContextTexts,
        eventExportDigest,
        expectedContextPhrases: [...turnFixture.expectedContextPhrases],
        minimumPhraseHits: turnFixture.minimumPhraseHits,
        phraseHits: scoring.phraseHits,
        missedPhrases: scoring.missedPhrases,
        qualityScore: scoring.qualityScore,
        compileActiveVersion: options.compileActiveVersion,
        promoted: options.promoted,
        observability,
        warnings: [...result.warnings]
    };
}
function countRecordedSessionActivePackChanges(turns) {
    let changes = 0;
    let previousPackId = null;
    for (const turn of turns) {
        if (turn.activePackId === null) {
            continue;
        }
        if (previousPackId !== null && previousPackId !== turn.activePackId) {
            changes += 1;
        }
        previousPackId = turn.activePackId;
    }
    return changes;
}
function buildRecordedSessionReplayScannerWarnings(mode, turns, activePackChangeCount) {
    const warnings = [];
    const exportTurnCount = turns.filter((turn) => turn.observability.totalEventCount > 0).length;
    const attributedTurnCount = turns.filter((turn) => turn.observability.attributedEventCount > 0).length;
    const selectionDigestTurnCount = turns.filter((turn) => turn.observability.selectionDigestCount > 0).length;
    const humanLabelCount = turns.reduce((sum, turn) => sum + turn.observability.humanLabelCount, 0);
    const scanSurfaceCount = new Set(turns.flatMap((turn) => turn.observability.scanSurfaces)).size;
    if (exportTurnCount === 0) {
        warnings.push("no_export_observability");
    }
    if (scanSurfaceCount === 0) {
        warnings.push("scan_surfaces_missing");
    }
    if (humanLabelCount === 0) {
        warnings.push("human_labels_missing");
    }
    if (attributedTurnCount === 0) {
        warnings.push("turn_attribution_missing");
    }
    if (turns.some((turn) => turn.compileOk) && selectionDigestTurnCount === 0) {
        warnings.push("selection_digest_missing");
    }
    if (mode === "learned_replay" && activePackChangeCount === 0) {
        warnings.push("active_pack_never_moved");
    }
    return warnings;
}
function buildRecordedSessionReplayScannerEvidence(mode, turns) {
    const exportTurnCount = turns.filter((turn) => turn.observability.totalEventCount > 0).length;
    const scanSurfaces = uniqueStringsInOrder(turns.flatMap((turn) => turn.observability.scanSurfaces));
    const attributedTurnCount = turns.filter((turn) => turn.observability.attributedEventCount > 0).length;
    const selectionDigestTurnCount = turns.filter((turn) => turn.observability.selectionDigestCount > 0).length;
    const activePackChangeCount = countRecordedSessionActivePackChanges(turns);
    const latestObservedTurn = [...turns]
        .filter((turn) => turn.observability.freshestCreatedAt !== null)
        .sort((left, right) => (right.observability.freshestCreatedAt ?? "").localeCompare(left.observability.freshestCreatedAt ?? ""))[0];
    return {
        exportTurnCount,
        scanPolicy: turns.find((turn) => turn.observability.scanPolicy !== null)?.observability.scanPolicy ?? null,
        scanSurfaceCount: scanSurfaces.length,
        scanSurfaces,
        humanLabelCount: turns.reduce((sum, turn) => sum + turn.observability.humanLabelCount, 0),
        selfLabelCount: turns.reduce((sum, turn) => sum + turn.observability.selfLabelCount, 0),
        totalEventCount: turns.reduce((sum, turn) => sum + turn.observability.totalEventCount, 0),
        attributedEventCount: turns.reduce((sum, turn) => sum + turn.observability.attributedEventCount, 0),
        attributedTurnCount,
        selectionDigestCount: turns.reduce((sum, turn) => sum + turn.observability.selectionDigestCount, 0),
        selectionDigestTurnCount,
        activePackChangeCount,
        freshestSourceStream: latestObservedTurn?.observability.freshestSourceStream ?? null,
        freshestCreatedAt: latestObservedTurn?.observability.freshestCreatedAt ?? null,
        warnings: buildRecordedSessionReplayScannerWarnings(mode, turns, activePackChangeCount)
    };
}
function buildRecordedSessionReplayModeSummary(mode, turns) {
    const compileOkCount = turns.filter((turn) => turn.compileOk).length;
    const phraseHitCount = turns.reduce((sum, turn) => sum + turn.phraseHits.length, 0);
    const phraseCount = turns.reduce((sum, turn) => sum + turn.expectedContextPhrases.length, 0);
    const usedLearnedRouteTurnCount = turns.filter((turn) => turn.usedLearnedRouteFn).length;
    const promotionCount = turns.filter((turn) => turn.promoted).length;
    const qualityScore = turns.length === 0 ? 0 : Math.round(turns.reduce((sum, turn) => sum + turn.qualityScore, 0) / turns.length);
    const packIds = uniqueStringsInOrder(turns.map((turn) => turn.activePackId).filter(isPresent));
    const scannerEvidence = buildRecordedSessionReplayScannerEvidence(mode, turns);
    const base = {
        mode,
        qualityScore,
        compileOkCount,
        phraseHitCount,
        phraseCount,
        usedLearnedRouteTurnCount,
        promotionCount,
        packIds,
        scannerEvidence
    };
    return {
        ...base,
        scoreHash: checksumJsonPayload({
            summary: base,
            turns: turns.map((turn) => ({
                turnId: turn.turnId,
                qualityScore: turn.qualityScore,
                phraseHits: turn.phraseHits,
                missedPhrases: turn.missedPhrases,
                compileOk: turn.compileOk,
                usedLearnedRouteFn: turn.usedLearnedRouteFn,
                activePackId: turn.activePackId,
                selectionDigest: turn.selectionDigest,
                promoted: turn.promoted,
                compileActiveVersion: turn.compileActiveVersion
            }))
        })
    };
}
function buildRecordedSessionReplayModeReport(mode, turns) {
    return {
        mode,
        summary: buildRecordedSessionReplayModeSummary(mode, turns),
        turns: [...turns]
    };
}
function buildRecordedSessionReplayScoreHash(modes) {
    return checksumJsonPayload(modes.map((mode) => ({
        mode: mode.mode,
        qualityScore: mode.summary.qualityScore,
        compileOkCount: mode.summary.compileOkCount,
        phraseHitCount: mode.summary.phraseHitCount,
        phraseCount: mode.summary.phraseCount,
        usedLearnedRouteTurnCount: mode.summary.usedLearnedRouteTurnCount,
        promotionCount: mode.summary.promotionCount,
        packIds: mode.summary.packIds,
        scannerEvidence: mode.summary.scannerEvidence,
        scoreHash: mode.summary.scoreHash
    })));
}
function recordedSessionReplayBundleBase(bundle) {
    return {
        contract: RECORDED_SESSION_BUNDLE_CONTRACT,
        traceId: bundle.traceId,
        source: bundle.source,
        recordedAt: bundle.recordedAt,
        generatedAt: bundle.generatedAt,
        traceHash: bundle.traceHash,
        fixtureHash: bundle.fixtureHash,
        scoreHash: bundle.scoreHash,
        privacy: {
            sanitized: true,
            notes: [...bundle.privacy.notes]
        },
        modes: bundle.modes.map((mode) => ({
            mode: mode.mode,
            summary: {
                ...mode.summary,
                packIds: [...mode.summary.packIds]
            },
            turns: mode.turns.map((turn) => ({
                ...turn,
                selectedContextIds: [...turn.selectedContextIds],
                selectedContextTexts: [...turn.selectedContextTexts],
                expectedContextPhrases: [...turn.expectedContextPhrases],
                phraseHits: [...turn.phraseHits],
                missedPhrases: [...turn.missedPhrases],
                observability: {
                    scanPolicy: turn.observability.scanPolicy,
                    scanSurfaces: [...turn.observability.scanSurfaces],
                    humanLabelCount: turn.observability.humanLabelCount,
                    selfLabelCount: turn.observability.selfLabelCount,
                    totalEventCount: turn.observability.totalEventCount,
                    attributedEventCount: turn.observability.attributedEventCount,
                    selectionDigestCount: turn.observability.selectionDigestCount,
                    freshestSourceStream: turn.observability.freshestSourceStream,
                    freshestCreatedAt: turn.observability.freshestCreatedAt
                },
                warnings: [...turn.warnings]
            }))
        })),
        summary: {
            winnerMode: bundle.summary.winnerMode,
            ranking: bundle.summary.ranking.map((entry) => ({ ...entry }))
        }
    };
}
function buildRecordedSessionTurnExportRoot(modeRoot, turnId) {
    return {
        rootDir: path.join(modeRoot, "exports", turnId),
        exportName: turnId
    };
}
function prepareReplayModeRoot(rootDir, mode) {
    const modeRoot = path.resolve(path.join(rootDir, mode));
    rmSync(modeRoot, { recursive: true, force: true });
    mkdirSync(modeRoot, { recursive: true });
    return modeRoot;
}
function prepareSeedActivation(rootDir, fixture) {
    const activationRoot = path.join(rootDir, "activation");
    const seedPackRoot = path.join(rootDir, "seed-pack");
    const seedPack = materializeCandidatePackFromNormalizedEventExport(seedPackRoot, {
        packLabel: `${fixture.traceId}-seed`,
        workspace: fixture.workspace,
        normalizedEventExport: fixture.seedExport,
        learnedRouting: false,
        builtAt: fixture.seedBuiltAt,
        offlineArtifacts: ["recorded-session-replay-seed"],
        structuralOps: {
            connect: 1
        }
    });
    activatePack(activationRoot, seedPackRoot, {
        updatedAt: fixture.seedActivatedAt,
        reason: "recorded_session_seed_activate"
    });
    return {
        activationRoot,
        seedPackId: seedPack.manifest.packId
    };
}
function runRecordedSessionNoBrainMode(rootDir, fixture) {
    const modeRoot = prepareReplayModeRoot(rootDir, "no_brain");
    const activationRoot = path.join(modeRoot, "activation");
    const turns = fixture.turns.map((turnFixture) => {
        const result = runRuntimeTurn({
            ...turnFixture.turn,
            export: buildRecordedSessionTurnExportRoot(modeRoot, turnFixture.turnId)
        }, {
            activationRoot,
            failOpen: true
        });
        return buildRecordedSessionTurnReport(turnFixture, result, {
            compileActiveVersion: null,
            promoted: false
        });
    });
    return buildRecordedSessionReplayModeReport("no_brain", turns);
}
function runRecordedSessionSeedPackMode(rootDir, fixture) {
    const modeRoot = prepareReplayModeRoot(rootDir, "seed_pack");
    const { activationRoot } = prepareSeedActivation(modeRoot, fixture);
    const turns = fixture.turns.map((turnFixture) => {
        const result = runRuntimeTurn({
            ...turnFixture.turn,
            export: buildRecordedSessionTurnExportRoot(modeRoot, turnFixture.turnId)
        }, {
            activationRoot,
            failOpen: false
        });
        return buildRecordedSessionTurnReport(turnFixture, result, {
            compileActiveVersion: 1,
            promoted: false
        });
    });
    return buildRecordedSessionReplayModeReport("seed_pack", turns);
}
function runRecordedSessionLearnedReplayMode(rootDir, fixture) {
    const modeRoot = prepareReplayModeRoot(rootDir, "learned_replay");
    const { activationRoot } = prepareSeedActivation(modeRoot, fixture);
    const loopRoot = path.join(modeRoot, "loop");
    let state;
    const turns = [];
    for (const turnFixture of fixture.turns) {
        const compileCreatedAt = normalizeIsoTimestamp(turnFixture.turn.compile?.createdAt, "turn.compile.createdAt", turnFixture.turn.createdAt);
        const result = runContinuousProductLoopTurn({
            activationRoot,
            loopRoot,
            packLabel: `${fixture.traceId}-learned`,
            workspace: fixture.workspace,
            ...(state !== undefined ? { state } : {}),
            learnedRouting: true,
            failOpen: false,
            turn: {
                ...turnFixture.turn,
                export: buildRecordedSessionTurnExportRoot(modeRoot, turnFixture.turnId)
            },
            candidateBuiltAt: addMinutes(compileCreatedAt, 2),
            stageUpdatedAt: addMinutes(compileCreatedAt, 3),
            promoteUpdatedAt: addMinutes(compileCreatedAt, 4)
        });
        state = result.state;
        turns.push(buildRecordedSessionTurnReport(turnFixture, result.turn, {
            compileActiveVersion: result.compileActiveVersion,
            promoted: result.learning.promoted
        }));
    }
    return buildRecordedSessionReplayModeReport("learned_replay", turns);
}
export function runRecordedSessionReplay(rootDir, fixture) {
    const resolvedRoot = path.resolve(normalizeNonEmptyString(rootDir, "rootDir"));
    const seedExportErrors = validateNormalizedEventExport(fixture.seedExport);
    if (seedExportErrors.length > 0) {
        throw new Error(`recorded session replay seed export is invalid: ${seedExportErrors.join("; ")}`);
    }
    const expectedFixtureHash = checksumJsonPayload(recordedSessionReplayFixtureBase(fixture));
    if (fixture.fixtureHash !== expectedFixtureHash) {
        throw new Error(`recorded session replay fixtureHash mismatch: expected ${expectedFixtureHash}, received ${fixture.fixtureHash}`);
    }
    const modes = [
        runRecordedSessionNoBrainMode(resolvedRoot, fixture),
        runRecordedSessionSeedPackMode(resolvedRoot, fixture),
        runRecordedSessionLearnedReplayMode(resolvedRoot, fixture)
    ];
    const ranking = modes
        .map((mode) => ({
        mode: mode.mode,
        qualityScore: mode.summary.qualityScore
    }))
        .sort((left, right) => right.qualityScore - left.qualityScore || left.mode.localeCompare(right.mode));
    const scoreHash = buildRecordedSessionReplayScoreHash(modes);
    const base = {
        contract: RECORDED_SESSION_BUNDLE_CONTRACT,
        traceId: fixture.traceId,
        source: fixture.source,
        recordedAt: fixture.recordedAt,
        generatedAt: fixture.bundleBuiltAt,
        traceHash: fixture.traceHash,
        fixtureHash: fixture.fixtureHash,
        scoreHash,
        privacy: {
            sanitized: true,
            notes: [...fixture.privacy.notes]
        },
        modes,
        summary: {
            winnerMode: ranking[0]?.mode ?? null,
            ranking
        }
    };
    return {
        ...base,
        bundleHash: checksumJsonPayload(base)
    };
}
function rescoreRecordedSessionReplayTurn(turn) {
    const scoring = buildReplayTurnScore({
        compileOk: turn.compileOk,
        texts: turn.selectedContextTexts,
        expectedContextPhrases: turn.expectedContextPhrases
    });
    return {
        ...turn,
        phraseHits: scoring.phraseHits,
        missedPhrases: scoring.missedPhrases,
        qualityScore: scoring.qualityScore,
        selectedContextIds: [...turn.selectedContextIds],
        selectedContextTexts: [...turn.selectedContextTexts],
        expectedContextPhrases: [...turn.expectedContextPhrases],
        warnings: [...turn.warnings]
    };
}
function rescoreRecordedSessionReplayMode(mode) {
    const turns = mode.turns.map((turn) => rescoreRecordedSessionReplayTurn(turn));
    return buildRecordedSessionReplayModeReport(mode.mode, turns);
}
export function rescoreRecordedSessionReplayBundle(bundle) {
    const modes = bundle.modes.map((mode) => rescoreRecordedSessionReplayMode(mode));
    return {
        scoreHash: buildRecordedSessionReplayScoreHash(modes),
        modes: modes.map((mode) => ({
            mode: mode.mode,
            qualityScore: mode.summary.qualityScore,
            scoreHash: mode.summary.scoreHash
        }))
    };
}
export function verifyRecordedSessionReplayBundleHashes(bundle) {
    const rescored = rescoreRecordedSessionReplayBundle(bundle);
    const rebuiltBundleHash = checksumJsonPayload(recordedSessionReplayBundleBase(bundle));
    return {
        bundleHashMatches: rebuiltBundleHash === bundle.bundleHash,
        scoreHashMatches: rescored.scoreHash === bundle.scoreHash
    };
}
export const OPERATOR_API_CONTRACT_ID = "openclaw_operator_api.v1";
export const SUPPORTED_OPERATOR_API_FAMILIES = [
    "bootstrap_attach",
    "status",
    "export",
    "refresh",
    "promote",
    "rollback",
    "proof_observability"
];
export const OPERATOR_API_CONTRACT_V1 = {
    contract: OPERATOR_API_CONTRACT_ID,
    runtimeOwner: "openclaw",
    scope: "narrow_supported_operator_surface",
    families: SUPPORTED_OPERATOR_API_FAMILIES,
    routes: [
        {
            family: "bootstrap_attach",
            scope: "programmatic",
            packageName: "@openclawbrain/openclaw",
            entrypoints: ["bootstrapRuntimeAttach", "formatBootstrapRuntimeAttachReport", "describeAttachStatus"],
            summary: "Bootstrap the first current-profile attach, print the next operator step cleanly, and prove the initial handoff state without pretending live learning has already run.",
            notes: [
                "Zero-event bootstrap is supported and stays explicit through awaiting_first_export.",
                "Attach serves only from activation's active slot after bootstrap completes.",
                "bootstrapRuntimeAttach() returns the canonical current-profile answer plus copy-paste-ready next-step output for the resolved activation root."
            ]
        },
        {
            family: "status",
            scope: "cli",
            packageName: "@openclawbrain/openclaw",
            entrypoints: ["openclawbrain status", "describeCurrentProfileBrainStatus"],
            summary: "Read the canonical current-profile brain-status object for the active Host/Profile/Brain/Attachment boundary.",
            notes: [
                "Status is the first operator read path.",
                "describeCurrentProfileBrainStatus() freezes the supported Host/Profile/Brain/Attachment answer shape for the current profile.",
                "Use activation and export observability proof helpers when you need candidate/previous or export-freshness detail."
            ]
        },
        {
            family: "export",
            scope: "programmatic",
            packageName: "@openclawbrain/openclaw",
            entrypoints: ["buildNormalizedRuntimeEventExport", "writeRuntimeEventExportBundle", "loadRuntimeEventExportBundle"],
            summary: "Emit the deterministic learner handoff artifact explicitly instead of folding export into a larger implicit runtime loop.",
            notes: [
                "Export is an off-hot-path operator handoff artifact, not proof of immediate active-pack mutation.",
                "Bundle roots and normalized payloads are both accepted downstream by observability surfaces."
            ]
        },
        {
            family: "refresh",
            scope: "programmatic",
            packageName: "@openclawbrain/learner",
            entrypoints: [
                "createAlwaysOnLearningRuntimeState",
                "advanceAlwaysOnLearningRuntime",
                "materializeAlwaysOnLearningCandidatePack"
            ],
            summary: "Refresh candidate learning state explicitly through the learner boundary before any activation-pointer move happens.",
            notes: [
                "Refresh is PG-only candidate-pack materialization in this repo.",
                "Refresh does not mutate the currently served active pack in place."
            ]
        },
        {
            family: "promote",
            scope: "programmatic",
            packageName: "@openclawbrain/pack-format",
            entrypoints: ["stageCandidatePack", "promoteCandidatePack"],
            summary: "Stage and promote activation-ready candidate packs through explicit pointer changes.",
            notes: [
                "Promotion is the only path that changes which pack is served.",
                "Candidate and previous remain inspectable around the pointer move."
            ]
        },
        {
            family: "rollback",
            scope: "cli",
            packageName: "@openclawbrain/openclaw",
            entrypoints: ["openclawbrain rollback", "rollbackRuntimeAttach", "formatOperatorRollbackReport"],
            summary: "Preview and apply the explicit active<-previous / active->candidate rollback move.",
            notes: [
                "Rollback is blocked when the previous pointer is unavailable.",
                "Dry-run is the required first read path for safe operator rollback."
            ]
        },
        {
            family: "proof_observability",
            scope: "programmatic",
            packageName: "@openclawbrain/openclaw",
            entrypoints: ["describeAttachStatus", "describeKernelBrainBoundary"],
            summary: "Prove the local attach and kernel-vs-brain boundary from the shipped bridge surface.",
            notes: [
                "Use these for repo-local or installed-package operator proof reads.",
                "These surfaces report the promoted artifact boundary, not full live runtime plasticity."
            ]
        },
        {
            family: "proof_observability",
            scope: "programmatic",
            packageName: "@openclawbrain/pack-format",
            entrypoints: ["describeActivationObservability"],
            summary: "Inspect activation health, freshness, route artifacts, rollback lineage, and slot readiness.",
            notes: ["Activation observability is the ground truth for active/candidate/previous slot inspection."]
        },
        {
            family: "proof_observability",
            scope: "programmatic",
            packageName: "@openclawbrain/event-export",
            entrypoints: ["describeNormalizedEventExportObservability"],
            summary: "Inspect supervision freshness and teacher freshness from the exported learner handoff artifact.",
            notes: ["Export observability is local-to-export proof only."]
        },
        {
            family: "proof_observability",
            scope: "proof_lane",
            packageName: "workspace",
            entrypoints: ["pnpm current-profile-lifecycle:smoke", "pnpm observability:smoke"],
            summary: "Run the repo-local proof lanes that derive operator truth from the canonical current-profile status object plus activation observability.",
            notes: ["These lanes are proof machinery, not a second semver-stable API."]
        }
    ],
    quarantinedSurface: [
        "openclawbrain-ops doctor was deleted; use the canonical current-profile status object plus proof helpers instead of a parallel troubleshooting surface.",
        "buildOperatorSurfaceReport / formatOperatorStatusReport / formatOperatorDoctorReport were historical parallel status surfaces and are not the supported operator API.",
        "runContinuousProductLoopTurn collapses export/refresh/promote into one proof helper and is not the supported operator API.",
        "runRecordedSessionReplay and recorded-session fixtures are proof helpers, not operator API.",
        "release scripts, root smoke plumbing, and workspace layout are proof-and-build machinery, not operator API.",
        "runRuntimeTurn is a runtime convenience wrapper and not the narrow operator export contract.",
        "createAsyncTeacherLiveLoop is supporting internals for refresh/teacher snapshots, not the narrow operator contract."
    ]
};
export const OPENCLAW_OPERATOR_NOUNS_V1 = ["Host", "Profile", "Brain", "Attachment"];
export const CURRENT_PROFILE_BRAIN_STATUS_CONTRACT = CONTRACT_IDS.currentProfileBrainStatus;
export const BRAIN_ATTACHMENT_POLICY_SEMANTICS_V1 = {
    undeclared: "The Host has not declared whether the current Profile's Brain attachment policy is shared or dedicated; do not infer profile exclusivity from activation state alone.",
    dedicated: "The Host declares a dedicated Brain attachment policy: one Profile is intentionally attached to one Brain activation root, and operators may treat the served Brain state as profile-specific until the attachment changes.",
    shared: "The Host declares a shared Brain attachment policy: multiple Profiles may intentionally attach to the same Brain activation root, attribution must stay current-profile explicit, and operators must not treat later served context as profile-exclusive."
};
function summarizeOperatorSlot(slot, updatedAt) {
    if (slot === null) {
        return null;
    }
    return {
        slot: slot.slot,
        packId: slot.packId,
        activationReady: slot.activationReady,
        routePolicy: slot.routePolicy,
        routerIdentity: slot.routerIdentity,
        workspaceSnapshot: slot.workspaceSnapshot,
        workspaceRevision: slot.workspaceRevision,
        eventRange: { ...slot.eventRange },
        eventExportDigest: slot.eventExportDigest,
        builtAt: slot.builtAt,
        updatedAt,
        findings: [...slot.findings]
    };
}
function buildMissingLabelFlowSummary(detail) {
    return {
        source: "missing",
        humanLabelCount: null,
        selfLabelCount: null,
        asyncTeacherArtifactCount: null,
        implicitPositiveCount: null,
        detail
    };
}
function buildMissingLearningPathSummary(detail) {
    return {
        available: false,
        source: "missing",
        policyGradientVersion: "unavailable",
        policyGradientMethod: null,
        objective: null,
        targetConstruction: null,
        connectOpsFired: null,
        reconstructedTrajectoryCount: null,
        detail
    };
}
function countTeacherArtifactBlocks(blocks) {
    return blocks.filter((block) => block.learning.role === "teacher_supervision" && !block.id.endsWith(":teacher-supervision-summary")).length;
}
function summarizePolicyGradientVersion(targetConstruction) {
    if (targetConstruction === "trajectory_reconstruction") {
        return "v2";
    }
    if (targetConstruction === "event_block_plus_related_interaction") {
        return "v1";
    }
    return "unavailable";
}
function summarizePackLabelFlow(source, pack) {
    const labelHarvest = pack.manifest.provenance.learningSurface.labelHarvest;
    return {
        source,
        humanLabelCount: labelHarvest.humanLabels,
        selfLabelCount: labelHarvest.selfLabels,
        asyncTeacherArtifactCount: countTeacherArtifactBlocks(pack.graph.blocks),
        implicitPositiveCount: labelHarvest.approvals,
        detail: source === "active_pack"
            ? "active pack label harvest and teacher artifacts are visible"
            : source === "materialized_candidate"
                ? "materialized candidate label harvest and teacher artifacts are visible"
                : "event export label harvest is visible"
    };
}
function summarizePackLearningPath(source, pack) {
    const targetConstruction = pack.router?.training.objective.profile.targetConstruction ?? null;
    const policyGradientVersion = summarizePolicyGradientVersion(targetConstruction);
    const reconstructedTrajectoryCount = policyGradientVersion === "v2"
        ? pack.router?.training.routeTraceCount ?? 0
        : pack.router === null
            ? null
            : 0;
    return {
        available: pack.router !== null,
        source,
        policyGradientVersion,
        policyGradientMethod: pack.router?.training.method ?? null,
        objective: pack.router?.training.objective.objective ?? null,
        targetConstruction,
        connectOpsFired: pack.manifest.graphDynamics.structuralOps.connect,
        reconstructedTrajectoryCount,
        detail: pack.router === null
            ? "pack has no learned router artifact"
            : policyGradientVersion === "v2"
                ? "learned routing uses trajectory-reconstruction PG"
                : policyGradientVersion === "v1"
                    ? "learned routing uses event-reconstruction PG"
                    : "learned routing is present but the PG profile is not recognizable"
    };
}
function summarizePackObservability(source, pack) {
    return {
        labelFlow: summarizePackLabelFlow(source, pack),
        learningPath: summarizePackLearningPath(source, pack)
    };
}
export function summarizeNormalizedEventExportLabelFlow(normalizedEventExport, asyncTeacherArtifactCount = 0) {
    const labelHarvest = normalizedEventExport.provenance.learningSurface.labelHarvest;
    return {
        source: "event_export",
        humanLabelCount: labelHarvest.humanLabels,
        selfLabelCount: labelHarvest.selfLabels,
        asyncTeacherArtifactCount,
        implicitPositiveCount: labelHarvest.approvals,
        detail: "event export label harvest is visible"
    };
}
export function summarizeLearningPathFromMaterialization(materialization) {
    if (materialization === null) {
        return buildMissingLearningPathSummary("no candidate pack materialized during this learning pass");
    }
    return summarizePackLearningPath("materialized_candidate", {
        manifest: materialization.candidate.manifest,
        graph: materialization.candidate.payloads.graph,
        router: materialization.candidate.payloads.router
    });
}
function summarizeActivePackObservability(activationRoot, active) {
    if (active === null) {
        return {
            labelFlow: buildMissingLabelFlowSummary("no active pack is attached"),
            learningPath: buildMissingLearningPathSummary("no active pack is attached")
        };
    }
    if (!active.activationReady) {
        return {
            labelFlow: buildMissingLabelFlowSummary(`active pack ${active.packId} is not activation-ready`),
            learningPath: buildMissingLearningPathSummary(`active pack ${active.packId} is not activation-ready`)
        };
    }
    try {
        const pack = loadPackFromActivation(activationRoot, "active", { requireActivationReady: true });
        if (pack === null) {
            return {
                labelFlow: buildMissingLabelFlowSummary("active pack payloads are unavailable"),
                learningPath: buildMissingLearningPathSummary("active pack payloads are unavailable")
            };
        }
        return summarizePackObservability("active_pack", {
            manifest: pack.manifest,
            graph: pack.graph,
            router: pack.router
        });
    }
    catch (error) {
        const detail = `active pack observability could not be loaded: ${toErrorMessage(error)}`;
        return {
            labelFlow: buildMissingLabelFlowSummary(detail),
            learningPath: buildMissingLearningPathSummary(detail)
        };
    }
}
function summarizeLastPromotion(inspection) {
    if (inspection.active === null) {
        return {
            known: false,
            at: null,
            confidence: "no_active_pack",
            note: "active slot is empty, so no promotion can be proven"
        };
    }
    if (inspection.previous !== null) {
        return {
            known: inspection.pointers.active?.updatedAt !== null,
            at: inspection.pointers.active?.updatedAt ?? null,
            confidence: "proven_from_previous_pointer",
            note: "previous pointer is retained, so the current active pack is the last promoted pack"
        };
    }
    return {
        known: false,
        at: null,
        confidence: "unknown_from_local_pointers",
        note: "no previous pointer is retained, so local activation pointers cannot prove the last promotion time"
    };
}
function summarizeCandidateAheadBy(candidateAheadBy) {
    if (candidateAheadBy === null) {
        return [];
    }
    return Object.entries(candidateAheadBy)
        .filter(([, changed]) => changed === true)
        .map(([field]) => field)
        .sort();
}
function summarizeManyProfileSupport(policyMode) {
    if (policyMode === "shared") {
        return {
            operatorSurface: "current_profile_only",
            declaredAttachmentPolicy: "shared",
            sameGatewayIntent: "shared_attachment_declared",
            checkedInProofTopology: "two_local_gateways_dedicated_only",
            sameGatewayProof: false,
            sharedWriteSafetyProof: false,
            detail: "The Host declares that multiple Profiles may intentionally attach to this Brain activation root, but the shipped operator read stays current-profile-only and this repo still does not prove same-gateway attachment behavior or shared write safety."
        };
    }
    if (policyMode === "dedicated") {
        return {
            operatorSurface: "current_profile_only",
            declaredAttachmentPolicy: "dedicated",
            sameGatewayIntent: "dedicated_current_profile_boundary",
            checkedInProofTopology: "two_local_gateways_dedicated_only",
            sameGatewayProof: false,
            sharedWriteSafetyProof: false,
            detail: "The Host declares one current Profile per Brain activation root. The checked-in many-profile proof is still narrower: two local gateways with dedicated brains, not same-gateway many-profile attachment inside one running host."
        };
    }
    return {
        operatorSurface: "current_profile_only",
        declaredAttachmentPolicy: "undeclared",
        sameGatewayIntent: "undeclared",
        checkedInProofTopology: "two_local_gateways_dedicated_only",
        sameGatewayProof: false,
        sharedWriteSafetyProof: false,
        detail: "The Host has not declared shared-vs-dedicated attachment policy. Keep the operator read current-profile-only, do not infer profile exclusivity from activation state alone, and do not claim same-gateway many-profile proof."
    };
}
function summarizeRetainedActivationSlots(input) {
    const retained = [];
    if (input.candidate !== null) {
        retained.push(`candidate=${input.candidate.packId}`);
    }
    if (input.previous !== null) {
        retained.push(`previous=${input.previous.packId}`);
    }
    return retained.length === 0 ? "none" : retained.join(", ");
}
function isAwaitingFirstExportSlot(slot) {
    return slot !== null && slot.eventRange.count === 0;
}
function summarizeOperatorActivationState(input) {
    if (input.inspectionError !== null) {
        return {
            state: "broken_install",
            detail: `activation root could not be inspected because activation pointers or pinned pack metadata are unreadable: ${input.inspectionError}`,
            inspectionError: input.inspectionError
        };
    }
    if (input.active !== null && !input.active.activationReady) {
        return {
            state: "broken_install",
            detail: input.active.findings.length > 0
                ? `active pack ${input.active.packId} is pinned but not activation-ready: ${input.active.findings.join("; ")}`
                : `active pack ${input.active.packId} is pinned but not activation-ready`,
            inspectionError: null
        };
    }
    if (input.active === null) {
        if (input.candidate !== null || input.previous !== null) {
            return {
                state: "stale_incomplete",
                detail: `activation root has retained non-serving state but no active pack (${summarizeRetainedActivationSlots(input)})`,
                inspectionError: null
            };
        }
        return {
            state: "detached",
            detail: "activation root has no active, candidate, or previous pack pinned",
            inspectionError: null
        };
    }
    if (input.observability === null) {
        return {
            state: "broken_install",
            detail: `active pack ${input.active.packId} is pinned, but activation observability could not be derived`,
            inspectionError: null
        };
    }
    if (input.observability.initHandoff.handoffState === "pg_promoted_pack_authoritative") {
        return {
            state: "active_promoted",
            detail: `active pack ${input.active.packId} is authoritative through promotion, not the seed handoff`,
            inspectionError: null
        };
    }
    if (isAwaitingFirstExportSlot(input.active)) {
        return {
            state: "awaiting_first_export",
            detail: `active seed-state pack ${input.active.packId} is healthy but still waiting for the first live export`,
            inspectionError: null
        };
    }
    if (input.observability.initHandoff.handoffState === "seed_state_authoritative") {
        return {
            state: "healthy_seed",
            detail: `active seed-state pack ${input.active.packId} is serving beyond the first export`,
            inspectionError: null
        };
    }
    return {
        state: "stale_incomplete",
        detail: `active pack ${input.active.packId} is pinned, but init handoff truth is incomplete (${input.observability.initHandoff.handoffState})`,
        inspectionError: null
    };
}
function summarizeBrainStateWithoutObservability(active, activation) {
    return {
        state: active === null ? "no_active_pack" : "missing",
        initMode: null,
        runtimePlasticitySource: null,
        seedStateVisible: false,
        seedBlockCount: 0,
        activePackId: active?.packId ?? null,
        activeWorkspaceSnapshot: active?.workspaceSnapshot ?? null,
        activeEventExportDigest: active?.eventExportDigest ?? null,
        detail: activation.detail
    };
}
function summarizeGraphWithoutObservability(input, active, activation) {
    const latestMaterialization = summarizeLatestGraphMaterialization(input, active, null);
    return {
        available: false,
        runtimePlasticitySource: null,
        structuralOps: null,
        connectDiagnostics: null,
        changed: null,
        blockCount: null,
        operationsApplied: [],
        liveBlockCount: null,
        prunedBlockCount: null,
        prePruneBlockCount: null,
        strongestBlockId: null,
        operatorSummary: null,
        latestMaterialization,
        detail: active === null
            ? activation.detail
            : `active pack ${active.packId} is pinned, but graph observability is unavailable`
    };
}
function summarizeLearnedRoutingWithoutObservability(active) {
    return {
        required: active?.routePolicy === "requires_learned_routing",
        available: false,
        routerIdentity: active?.routerIdentity ?? null,
        routeFnVersion: null,
        trainingMethod: null,
        routerTrainedAt: null,
        objective: null,
        pgProfile: null,
        routerChecksum: null,
        objectiveChecksum: null,
        updateMechanism: null,
        updateVersion: null,
        updateCount: null,
        supervisionCount: null,
        collectedLabelsTotal: null,
        freshnessChecksum: null,
        handoffState: "missing",
        initMode: null,
        seedStateVisible: false
    };
}
function summarizeBrainState(active, observability) {
    if (active === null) {
        return {
            state: "no_active_pack",
            initMode: null,
            runtimePlasticitySource: null,
            seedStateVisible: false,
            seedBlockCount: 0,
            activePackId: null,
            activeWorkspaceSnapshot: null,
            activeEventExportDigest: null,
            detail: "no active pack is pinned, so the serve path can only fail open or hard fail"
        };
    }
    const state = observability.initHandoff.handoffState;
    const detail = state === "pg_promoted_pack_authoritative"
        ? "serving is pinned to a PG-promoted pack rather than seed-state authority"
        : state === "seed_state_authoritative"
            ? "serving is still pinned to the current seed-state authority"
            : "init/handoff metadata is missing for the active pack";
    return {
        state,
        initMode: observability.initHandoff.initMode,
        runtimePlasticitySource: observability.graphDynamics.runtimePlasticitySource,
        seedStateVisible: observability.initHandoff.seedStateVisible,
        seedBlockCount: observability.initHandoff.seedBlockCount,
        activePackId: active.packId,
        activeWorkspaceSnapshot: active.workspaceSnapshot,
        activeEventExportDigest: active.eventExportDigest,
        detail
    };
}
function summarizeLatestGraphMaterialization(input, active, activeGraphEvolution) {
    const loadedTeacherSurface = loadTeacherSurfaceFromInput(input);
    const latestMaterialization = loadedTeacherSurface?.snapshot.learner.lastMaterialization ?? null;
    const latestGraph = latestMaterialization?.candidate.summary.graphEvolutionLog ?? null;
    if (latestGraph !== null) {
        const packId = latestMaterialization?.candidate.summary.packId ?? latestGraph.packId;
        return {
            known: true,
            packId,
            changed: latestGraph.structuralEvolutionSummary.changed,
            connectDiagnostics: latestGraph.connectDiagnostics === null ? null : { ...latestGraph.connectDiagnostics },
            operatorSummary: latestGraph.structuralEvolutionSummary.operatorSummary,
            detail: `latest known materialization is ${packId} from the teacher snapshot`
        };
    }
    if (active !== null && activeGraphEvolution !== null) {
        return {
            known: true,
            packId: active.packId,
            changed: activeGraphEvolution.structuralEvolutionSummary.changed,
            connectDiagnostics: activeGraphEvolution.connectDiagnostics === null ? null : { ...activeGraphEvolution.connectDiagnostics },
            operatorSummary: activeGraphEvolution.structuralEvolutionSummary.operatorSummary,
            detail: `latest known materialization is the active pack ${active.packId}`
        };
    }
    return {
        known: false,
        packId: null,
        changed: null,
        connectDiagnostics: null,
        operatorSummary: null,
        detail: active === null
            ? "no active pack or materialized learner snapshot is visible"
            : `active pack ${active.packId} is pinned, but no latest materialization snapshot is visible`
    };
}
function summarizeGraphObservability(input, active, observability) {
    const latestMaterialization = summarizeLatestGraphMaterialization(input, active, observability.graphEvolutionLog);
    if (active === null) {
        return {
            available: false,
            runtimePlasticitySource: null,
            structuralOps: null,
            connectDiagnostics: null,
            changed: null,
            blockCount: null,
            operationsApplied: [],
            liveBlockCount: null,
            prunedBlockCount: null,
            prePruneBlockCount: null,
            strongestBlockId: null,
            operatorSummary: null,
            latestMaterialization,
            detail: "no active pack is pinned, so there is no structural graph surface to inspect"
        };
    }
    const graphEvolution = observability.graphEvolutionLog;
    if (graphEvolution === null) {
        return {
            available: false,
            runtimePlasticitySource: observability.graphDynamics.runtimePlasticitySource,
            structuralOps: null,
            connectDiagnostics: null,
            changed: null,
            blockCount: null,
            operationsApplied: [],
            liveBlockCount: null,
            prunedBlockCount: null,
            prePruneBlockCount: null,
            strongestBlockId: null,
            operatorSummary: null,
            latestMaterialization,
            detail: "active pack is present, but the graph evolution log is missing from activation observability"
        };
    }
    return {
        available: true,
        runtimePlasticitySource: observability.graphDynamics.runtimePlasticitySource,
        structuralOps: { ...graphEvolution.structuralOps },
        connectDiagnostics: graphEvolution.connectDiagnostics === null ? null : { ...graphEvolution.connectDiagnostics },
        changed: graphEvolution.structuralEvolutionSummary.changed,
        blockCount: graphEvolution.blockCount,
        operationsApplied: [...graphEvolution.structuralEvolutionSummary.operationsApplied],
        liveBlockCount: graphEvolution.structuralEvolutionSummary.liveBlockCount,
        prunedBlockCount: graphEvolution.structuralEvolutionSummary.prunedBlockCount,
        prePruneBlockCount: graphEvolution.structuralEvolutionSummary.prePruneBlockCount,
        strongestBlockId: graphEvolution.strongestBlockId,
        operatorSummary: graphEvolution.structuralEvolutionSummary.operatorSummary,
        latestMaterialization,
        detail: graphEvolution.structuralEvolutionSummary.changed
            ? graphEvolution.structuralEvolutionSummary.operatorSummary
            : "active pack graph is stable with no structural evolution beyond the current promoted artifact"
    };
}
function summarizeServePath(compile) {
    const structuralDecision = summarizeStructuralDecisionFromNotes(compile?.notes ?? []);
    if (compile === null) {
        return {
            state: "unprobed",
            fallbackToStaticContext: false,
            hardRequirementViolated: false,
            activePackId: null,
            usedLearnedRouteFn: null,
            routerIdentity: null,
            selectionMode: null,
            selectionDigest: null,
            refreshStatus: null,
            freshnessChecksum: null,
            requestedBudgetStrategy: null,
            resolvedBudgetStrategy: null,
            resolvedMaxContextBlocks: null,
            structuralBudgetSource: null,
            structuralBudgetEvidence: null,
            structuralBudgetPressures: null,
            structuralDecision,
            contextAttribution: buildContextAttributionSummary({
                fallbackToStaticContext: false,
                hardRequirementViolated: false,
                usedLearnedRouteFn: null,
                unprobed: true
            }),
            timing: buildUnavailableBrainServeHotPathTiming("serve-path timing is unavailable because the hot path was not probed"),
            error: null
        };
    }
    if (!compile.ok) {
        return {
            state: compile.hardRequirementViolated ? "hard_fail" : "fail_open_static_context",
            fallbackToStaticContext: compile.fallbackToStaticContext,
            hardRequirementViolated: compile.hardRequirementViolated,
            activePackId: compile.activePackId,
            usedLearnedRouteFn: compile.usedLearnedRouteFn,
            routerIdentity: compile.routerIdentity,
            selectionMode: null,
            selectionDigest: null,
            refreshStatus: null,
            freshnessChecksum: null,
            requestedBudgetStrategy: null,
            resolvedBudgetStrategy: null,
            resolvedMaxContextBlocks: null,
            structuralBudgetSource: null,
            structuralBudgetEvidence: null,
            structuralBudgetPressures: null,
            structuralDecision,
            contextAttribution: compile.contextAttribution,
            timing: compile.timing,
            error: compile.error
        };
    }
    return {
        state: "serving_active_pack",
        fallbackToStaticContext: compile.fallbackToStaticContext,
        hardRequirementViolated: compile.hardRequirementViolated,
        activePackId: compile.activePackId,
        usedLearnedRouteFn: compile.usedLearnedRouteFn,
        routerIdentity: compile.routerIdentity,
        selectionMode: readDiagnosticNoteValue(compile.notes, "selection_mode="),
        selectionDigest: compile.selectionDigest,
        refreshStatus: readDiagnosticNoteValue(compile.notes, "router_refresh_status="),
        freshnessChecksum: readDiagnosticNoteValue(compile.notes, "router_freshness_checksum="),
        requestedBudgetStrategy: readDiagnosticNoteValue(compile.notes, "requested_budget_strategy="),
        resolvedBudgetStrategy: readDiagnosticNoteValue(compile.notes, "resolved_budget_strategy="),
        resolvedMaxContextBlocks: structuralDecision.resolvedMaxContextBlocks,
        structuralBudgetSource: readDiagnosticNoteValue(compile.notes, "structural_budget_source="),
        structuralBudgetEvidence: readDiagnosticNoteValue(compile.notes, "structural_budget_compile_evidence=") ??
            readDiagnosticNoteValue(compile.notes, "structural_budget_evidence="),
        structuralBudgetPressures: readDiagnosticNoteValue(compile.notes, "structural_budget_pressures="),
        structuralDecision,
        contextAttribution: compile.contextAttribution,
        timing: compile.timing,
        error: compile.error
    };
}
function probeOperatorServePath(activationRoot, observability, activePackId) {
    const compileInput = buildAttachStatusCompileInput(activationRoot, undefined);
    const compile = compileInput === null ? null : buildAttachCompileStatus(compileRuntimeContext(compileInput), observability, activePackId);
    return summarizeServePath(compile);
}
function loadOperatorEventExportFromPath(resolvedPath) {
    const stats = statSync(resolvedPath);
    if (stats.isDirectory()) {
        const bundle = loadRuntimeEventExportBundle(resolvedPath);
        return {
            sourcePath: resolvedPath,
            sourceKind: "bundle_root",
            normalizedEventExport: bundle.normalizedEventExport,
            exportedAt: bundle.manifest.exportedAt
        };
    }
    const normalizedEventExport = readJsonFile(resolvedPath);
    const validationErrors = validateNormalizedEventExport(normalizedEventExport);
    if (validationErrors.length > 0) {
        throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }
    return {
        sourcePath: resolvedPath,
        sourceKind: "payload",
        normalizedEventExport,
        exportedAt: null
    };
}
function resolveOperatorEventExportScanRoots(input) {
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const teacherSurface = loadTeacherSurfaceFromInput(input);
    const teacherScanRoot = normalizeOptionalString(teacherSurface?.watchSnapshot?.scanRoot);
    return uniqueStringsInOrder([
        teacherScanRoot === undefined ? undefined : path.resolve(teacherScanRoot),
        path.join(activationRoot, "event-exports")
    ].filter((value) => value !== undefined));
}
function loadLatestOperatorEventExportFromScanRoots(scanRoots) {
    let latest = null;
    for (const scanRoot of scanRoots) {
        const discovered = discoverRuntimeEventExportBundles(scanRoot);
        const candidate = discovered.bundles[discovered.bundles.length - 1] ?? null;
        if (candidate === null) {
            continue;
        }
        if (latest === null || compareRuntimeEventExportScannerBundleCursor(latest.cursor, candidate.cursor) < 0) {
            latest = candidate;
        }
    }
    if (latest === null) {
        return null;
    }
    return {
        sourcePath: latest.descriptor.rootDir,
        sourceKind: "bundle_root",
        normalizedEventExport: latest.descriptor.normalizedEventExport,
        exportedAt: latest.descriptor.manifest.exportedAt
    };
}
function loadOperatorEventExport(input) {
    const eventExportPath = normalizeOptionalString(input.eventExportPath);
    if (eventExportPath !== undefined) {
        return loadOperatorEventExportFromPath(path.resolve(eventExportPath));
    }
    return loadLatestOperatorEventExportFromScanRoots(resolveOperatorEventExportScanRoots(input));
}
function summarizePrincipalItem(event) {
    if (event.principal === undefined) {
        return null;
    }
    return {
        eventId: event.eventId,
        contract: event.contract,
        kind: event.kind,
        sequence: event.sequence,
        createdAt: event.createdAt,
        teacherIdentity: event.principal.teacherIdentity,
        teacherAuthority: event.principal.teacherAuthority,
        priorityClass: event.principal.priorityClass,
        scopeKind: event.principal.principalScope.kind,
        supersedes: [...(event.principal.supersedes ?? [])]
    };
}
function summarizePrincipalFeedback(event) {
    const base = summarizePrincipalItem(event);
    if (base === null) {
        return null;
    }
    return {
        ...base,
        content: event.content,
        relatedInteractionId: event.relatedInteractionId ?? null
    };
}
function isPrincipalEvent(event) {
    return event.principal?.teacherRole === "principal";
}
function summarizePrincipalObservability(input, active) {
    const loaded = loadOperatorEventExport(input);
    if (loaded === null) {
        return {
            available: false,
            sourcePath: null,
            sourceKind: "missing",
            latestFeedback: null,
            latestCorrection: null,
            pendingCount: null,
            pendingItems: [],
            latestPromotion: {
                known: false,
                at: null,
                activePackId: active?.packId ?? null,
                activeEventRangeEnd: active?.eventRange.end ?? null,
                includesLatestFeedback: null,
                includesLatestCorrection: null,
                note: "no event export path supplied"
            },
            servingDownstreamOfLatestCorrection: null,
            detail: "no event export path supplied"
        };
    }
    const allPrincipalEvents = sortNormalizedEvents([
        ...loaded.normalizedEventExport.interactionEvents,
        ...loaded.normalizedEventExport.feedbackEvents
    ]).filter(isPrincipalEvent);
    const principalFeedbackEvents = loaded.normalizedEventExport.feedbackEvents
        .filter((event) => isPrincipalEvent(event))
        .sort((left, right) => {
        if (left.sequence !== right.sequence) {
            return right.sequence - left.sequence;
        }
        return right.createdAt.localeCompare(left.createdAt);
    });
    const latestFeedbackEvent = principalFeedbackEvents[0] ?? null;
    const latestCorrectionEvent = principalFeedbackEvents.find((event) => event.kind === "correction") ?? null;
    const activeEventRangeEnd = active?.eventRange.end ?? null;
    const pendingEvents = activeEventRangeEnd === null ? allPrincipalEvents : allPrincipalEvents.filter((event) => event.sequence > activeEventRangeEnd);
    const includesLatestFeedback = latestFeedbackEvent === null ? null : activeEventRangeEnd !== null && latestFeedbackEvent.sequence <= activeEventRangeEnd;
    const includesLatestCorrection = latestCorrectionEvent === null ? null : activeEventRangeEnd !== null && latestCorrectionEvent.sequence <= activeEventRangeEnd;
    return {
        available: true,
        sourcePath: loaded.sourcePath,
        sourceKind: loaded.sourceKind,
        latestFeedback: latestFeedbackEvent === null ? null : summarizePrincipalFeedback(latestFeedbackEvent),
        latestCorrection: latestCorrectionEvent === null ? null : summarizePrincipalFeedback(latestCorrectionEvent),
        pendingCount: pendingEvents.length,
        pendingItems: pendingEvents.map((event) => summarizePrincipalItem(event)).filter((item) => item !== null),
        latestPromotion: {
            known: active !== null && active.eventRange.count > 0,
            at: active?.updatedAt ?? null,
            activePackId: active?.packId ?? null,
            activeEventRangeEnd,
            includesLatestFeedback,
            includesLatestCorrection,
            note: active === null
                ? "no active pack is serving, so principal promotion state is not observable"
                : active.eventRange.count === 0
                    ? "active pack is still awaiting the first export, so no principal event has been promoted into serving"
                    : includesLatestFeedback === true
                        ? "active serving range already covers the latest principal feedback"
                        : latestFeedbackEvent === null
                            ? "no principal feedback is present in the supplied export"
                            : "latest principal feedback sits ahead of the active serving range"
        },
        servingDownstreamOfLatestCorrection: includesLatestCorrection,
        detail: latestFeedbackEvent === null
            ? "the supplied export does not contain principal feedback"
            : pendingEvents.length === 0
                ? "principal feedback is visible and current serving is caught up to the supplied export"
                : `${pendingEvents.length} principal item(s) are newer than the active serving range`
    };
}
function summarizeSupervision(input) {
    const loaded = loadOperatorEventExport(input);
    if (loaded === null) {
        return {
            available: false,
            sourcePath: null,
            sourceKind: "missing",
            exportDigest: null,
            exportedAt: null,
            flowing: null,
            scanPolicy: null,
            scanSurfaceCount: 0,
            scanSurfaces: [],
            sourceCount: 0,
            freshestSourceStream: null,
            freshestCreatedAt: null,
            freshestKind: null,
            humanLabelCount: null,
            selfLabelCount: null,
            attributedEventCount: null,
            totalEventCount: null,
            selectionDigestCount: null,
            sources: [],
            detail: "no event export path supplied"
        };
    }
    const observability = describeNormalizedEventExportObservability(loaded.normalizedEventExport);
    const freshestSource = observability.supervisionFreshnessBySource[0] ?? null;
    const flowing = observability.teacherFreshness.freshestCreatedAt !== null && observability.teacherFreshness.humanLabelCount > 0;
    return {
        available: true,
        sourcePath: loaded.sourcePath,
        sourceKind: loaded.sourceKind,
        exportDigest: observability.exportDigest,
        exportedAt: loaded.exportedAt,
        flowing,
        scanPolicy: observability.learningSurface.scanPolicy,
        scanSurfaceCount: observability.learningSurface.scanSurfaces.length,
        scanSurfaces: [...observability.learningSurface.scanSurfaces],
        sourceCount: observability.supervisionFreshnessBySource.length,
        freshestSourceStream: observability.teacherFreshness.sourceStream ?? freshestSource?.sourceStream ?? null,
        freshestCreatedAt: observability.teacherFreshness.freshestCreatedAt ?? freshestSource?.freshestCreatedAt ?? null,
        freshestKind: observability.teacherFreshness.freshestKind ?? freshestSource?.freshestKind ?? null,
        humanLabelCount: observability.teacherFreshness.humanLabelCount,
        selfLabelCount: observability.learningSurface.selfLabelCount,
        attributedEventCount: observability.attributionCoverage.attributedEventCount,
        totalEventCount: observability.attributionCoverage.totalEventCount,
        selectionDigestCount: observability.attributionCoverage.selectionDigestCount,
        sources: [...observability.teacherFreshness.sources],
        detail: flowing
            ? "human supervision is visible in the supplied export"
            : "the supplied export does not yet show human supervision"
    };
}
function loadTeacherSurfaceFromInput(input) {
    const teacherSnapshotPath = resolveOperatorTeacherSnapshotPath(input.activationRoot, normalizeOptionalString(input.teacherSnapshotPath) ?? null);
    if (teacherSnapshotPath === null) {
        return null;
    }
    return loadTeacherSurface(teacherSnapshotPath);
}
function summarizeTeacherLoopWatchState(input) {
    if (input.sourceKind !== "watch_snapshot" || input.watchSnapshot === null) {
        return {
            snapshotUpdatedAt: null,
            lastWatchHeartbeatAt: null,
            pollIntervalSeconds: null,
            watchState: input.sourceKind === "async_snapshot" ? "snapshot_only" : "not_visible"
        };
    }
    const lastWatchHeartbeatAt = input.watchSnapshot.snapshot.runtime?.lastHeartbeatAt ?? input.watchSnapshot.lastRunAt;
    const pollIntervalSeconds = input.watchSnapshot.pollIntervalSeconds;
    if (lastWatchHeartbeatAt === null) {
        return {
            snapshotUpdatedAt: input.watchSnapshot.updatedAt,
            lastWatchHeartbeatAt: null,
            pollIntervalSeconds,
            watchState: "snapshot_only"
        };
    }
    const lagMs = Date.parse(input.observedAt) - Date.parse(lastWatchHeartbeatAt);
    const staleAfterMs = pollIntervalSeconds * 2000 + 15_000;
    return {
        snapshotUpdatedAt: input.watchSnapshot.updatedAt,
        lastWatchHeartbeatAt,
        pollIntervalSeconds,
        watchState: Number.isFinite(lagMs) && lagMs >= 0 && lagMs <= staleAfterMs ? "watching" : "stale_snapshot"
    };
}
function summarizeTeacherLoop(input) {
    const loaded = loadTeacherSurfaceFromInput(input);
    const teacherSnapshotPath = resolveOperatorTeacherSnapshotPath(input.activationRoot, normalizeOptionalString(input.teacherSnapshotPath) ?? null);
    const unavailableFromMissing = buildUnavailableLastObservedDelta("no watch teacher snapshot is visible for the latest observed cycle");
    const unavailableFromAsync = buildUnavailableLastObservedDelta("raw async teacher snapshots do not record the last observed export/label/promotion delta");
    if (loaded === null && teacherSnapshotPath === null) {
        return {
            available: false,
            sourcePath: null,
            sourceKind: "missing",
            snapshotUpdatedAt: null,
            lastRunAt: null,
            lastNoOpReason: "unavailable",
            latestFreshness: "unavailable",
            startedAt: null,
            lastHeartbeatAt: null,
            lastScanAt: null,
            pollIntervalSeconds: null,
            watchState: "not_visible",
            lastProcessedAt: null,
            artifactCount: null,
            queueDepth: null,
            queueCapacity: null,
            running: null,
            replayedBundleCount: null,
            replayedEventCount: null,
            exportedBundleCount: null,
            exportedEventCount: null,
            sessionTailSessionsTracked: null,
            sessionTailBridgedEventCount: null,
            localSessionTailNoopReason: null,
            learningCadence: "unavailable",
            scanPolicy: "unavailable",
            liveSlicesPerCycle: null,
            backfillSlicesPerCycle: null,
            failureMode: "unavailable",
            failureDetail: null,
            lastAppliedMaterializationJobId: null,
            lastMaterializedPackId: null,
            lastObservedDelta: unavailableFromMissing,
            notes: [],
            detail: "no teacher snapshot path supplied"
        };
    }
    if (loaded === null) {
        return {
            available: false,
            sourcePath: path.resolve(teacherSnapshotPath),
            sourceKind: "missing",
            snapshotUpdatedAt: null,
            lastRunAt: null,
            lastNoOpReason: "unavailable",
            latestFreshness: "unavailable",
            startedAt: null,
            lastHeartbeatAt: null,
            lastScanAt: null,
            pollIntervalSeconds: null,
            watchState: "not_visible",
            lastProcessedAt: null,
            artifactCount: null,
            queueDepth: null,
            queueCapacity: null,
            running: null,
            replayedBundleCount: null,
            replayedEventCount: null,
            exportedBundleCount: null,
            exportedEventCount: null,
            sessionTailSessionsTracked: null,
            sessionTailBridgedEventCount: null,
            localSessionTailNoopReason: null,
            learningCadence: "unavailable",
            scanPolicy: "unavailable",
            liveSlicesPerCycle: null,
            backfillSlicesPerCycle: null,
            failureMode: "unavailable",
            failureDetail: null,
            lastAppliedMaterializationJobId: null,
            lastMaterializedPackId: null,
            lastObservedDelta: unavailableFromMissing,
            notes: [],
            detail: "teacher snapshot could not be loaded"
        };
    }
    const snapshot = loaded.snapshot;
    const watchSnapshot = loaded.watchSnapshot;
    const watchState = summarizeTeacherLoopWatchState({
        observedAt: normalizeIsoTimestamp(input.updatedAt, "updatedAt", new Date().toISOString()),
        sourceKind: loaded.sourceKind,
        watchSnapshot
    });
    return {
        available: true,
        sourcePath: loaded.sourcePath,
        sourceKind: loaded.sourceKind,
        snapshotUpdatedAt: watchState.snapshotUpdatedAt,
        lastRunAt: watchSnapshot?.lastRunAt ?? snapshot.runtime?.lastHeartbeatAt ?? snapshot.diagnostics.lastProcessedAt ?? null,
        lastNoOpReason: snapshot.diagnostics.lastNoOpReason,
        latestFreshness: snapshot.diagnostics.latestFreshness,
        startedAt: snapshot.runtime?.startedAt ?? null,
        lastHeartbeatAt: watchState.lastWatchHeartbeatAt ?? snapshot.runtime?.lastHeartbeatAt ?? null,
        lastScanAt: snapshot.runtime?.lastScanAt ?? null,
        pollIntervalSeconds: watchState.pollIntervalSeconds,
        watchState: watchState.watchState,
        lastProcessedAt: snapshot.diagnostics.lastProcessedAt,
        artifactCount: watchSnapshot?.teacher.artifactCount ?? snapshot.teacher.artifactCount,
        queueDepth: snapshot.queue.depth,
        queueCapacity: snapshot.queue.capacity,
        running: snapshot.queue.running,
        replayedBundleCount: watchSnapshot?.replayedBundleCount ?? null,
        replayedEventCount: watchSnapshot?.replayedEventCount ?? null,
        exportedBundleCount: watchSnapshot?.exportedBundleCount ?? null,
        exportedEventCount: watchSnapshot?.exportedEventCount ?? null,
        sessionTailSessionsTracked: watchSnapshot?.sessionTailSessionsTracked ?? null,
        sessionTailBridgedEventCount: watchSnapshot?.sessionTailBridgedEventCount ?? null,
        localSessionTailNoopReason: watchSnapshot?.localSessionTailNoopReason ?? null,
        learningCadence: watchSnapshot?.labeling.learningCadence ?? "unavailable",
        scanPolicy: watchSnapshot?.labeling.scanPolicy ?? "unavailable",
        liveSlicesPerCycle: watchSnapshot?.labeling.liveSlicesPerCycle ?? null,
        backfillSlicesPerCycle: watchSnapshot?.labeling.backfillSlicesPerCycle ?? null,
        failureMode: watchSnapshot?.failure?.mode ?? "none",
        failureDetail: watchSnapshot?.failure?.detail ?? null,
        lastAppliedMaterializationJobId: watchSnapshot?.teacher.lastAppliedMaterializationJobId ??
            snapshot.runtime?.lastAppliedMaterializationJobId ??
            snapshot.learner.lastMaterialization?.jobId ??
            null,
        lastMaterializedPackId: snapshot.learner.lastMaterialization?.candidate.summary.packId ?? null,
        lastObservedDelta: loaded.sourceKind === "watch_snapshot" && watchSnapshot !== null
            ? cloneLastObservedDelta(watchSnapshot.lastObservedDelta)
            : unavailableFromAsync,
        notes: [...snapshot.diagnostics.notes],
        detail: loaded.sourceKind === "watch_snapshot"
            ? "canonical watch teacher snapshot loaded"
            : "raw async teacher snapshot loaded"
    };
}
function matchesActiveRouteFnLog(input) {
    if (input.activePackId !== null && input.entryPackId === input.activePackId) {
        return true;
    }
    if (input.routerChecksum !== null && input.entryRouterChecksum === input.routerChecksum) {
        return true;
    }
    return false;
}
function summarizeRouteFnFreshness(input) {
    if (input.activePackId === null) {
        return {
            available: false,
            activePackId: null,
            routerIdentity: input.learnedRouting.routerIdentity,
            routerChecksum: input.learnedRouting.routerChecksum,
            trainedAt: null,
            updatedAt: null,
            usedAt: null,
            lastDecisionAt: null,
            lastDecisionUsedLearnedRouteFn: null,
            detail: "no active pack is pinned, so there is no current route_fn to time"
        };
    }
    if (!input.learnedRouting.available) {
        return {
            available: false,
            activePackId: input.activePackId,
            routerIdentity: input.learnedRouting.routerIdentity,
            routerChecksum: input.learnedRouting.routerChecksum,
            trainedAt: input.learnedRouting.routerTrainedAt,
            updatedAt: null,
            usedAt: null,
            lastDecisionAt: null,
            lastDecisionUsedLearnedRouteFn: null,
            detail: input.learnedRouting.required
                ? `active pack ${input.activePackId} requires learned routing, but no route_fn artifact is available`
                : `active pack ${input.activePackId} does not require a learned route_fn`
        };
    }
    const updates = readLearningSpineLogEntries(input.activationRoot, "pgRouteUpdates");
    const decisions = readLearningSpineLogEntries(input.activationRoot, "serveTimeRouteDecisions");
    const updated = [...updates].reverse().find((entry) => matchesActiveRouteFnLog({
        activePackId: input.activePackId,
        routerChecksum: input.learnedRouting.routerChecksum,
        entryPackId: entry.nextPackId,
        entryRouterChecksum: entry.nextRouterChecksum
    }));
    const learnedUse = [...decisions].reverse().find((entry) => entry.usedLearnedRouteFn === true &&
        matchesActiveRouteFnLog({
            activePackId: input.activePackId,
            routerChecksum: input.learnedRouting.routerChecksum,
            entryPackId: entry.activePackId,
            entryRouterChecksum: entry.activePackRouterChecksum
        }));
    const lastDecision = [...decisions].reverse().find((entry) => matchesActiveRouteFnLog({
        activePackId: input.activePackId,
        routerChecksum: input.learnedRouting.routerChecksum,
        entryPackId: entry.activePackId,
        entryRouterChecksum: entry.activePackRouterChecksum
    }));
    return {
        available: true,
        activePackId: input.activePackId,
        routerIdentity: input.learnedRouting.routerIdentity,
        routerChecksum: input.learnedRouting.routerChecksum,
        trainedAt: input.learnedRouting.routerTrainedAt,
        updatedAt: updated?.recordedAt ?? null,
        usedAt: learnedUse?.recordedAt ?? null,
        lastDecisionAt: lastDecision?.recordedAt ?? null,
        lastDecisionUsedLearnedRouteFn: lastDecision?.usedLearnedRouteFn ?? null,
        detail: learnedUse !== undefined
            ? `active route_fn last served a learned turn at ${learnedUse.recordedAt}`
            : updated !== undefined
                ? `active route_fn was updated at ${updated.recordedAt}, but no learned serve decision for the current pack is visible yet`
                : input.learnedRouting.routerTrainedAt !== null
                    ? `active route_fn was trained at ${input.learnedRouting.routerTrainedAt}, but no matching update or serve-use log is visible yet`
                    : `active pack ${input.activePackId} exposes a route_fn, but the train/update timestamps are not visible`
    };
}
function summarizeLearningBacklogState(plan, principalLagStatus) {
    if (!plan.bootstrapped && plan.pending.total === 0) {
        return "awaiting_first_export";
    }
    if (plan.nextPriorityBucket === "principal_immediate") {
        return "principal_live_priority";
    }
    if (plan.nextPriorityBucket === "principal_backfill") {
        return "principal_backfill_priority";
    }
    if (plan.nextPriorityBucket === "live") {
        return "live_priority";
    }
    if (plan.nextPriorityBucket === "backfill") {
        return "backfill_only";
    }
    return principalLagStatus === "pending_promotion" ? "principal_live_priority" : "caught_up";
}
function summarizeLearningWarningStates(input) {
    const warnings = new Set();
    if (!input.plan.bootstrapped && input.plan.pending.total === 0) {
        warnings.add("awaiting_first_export");
    }
    if (input.plan.pending.byBucket.principal_immediate > 0) {
        warnings.add("principal_live_backlog");
    }
    if (input.plan.pending.byBucket.principal_backfill > 0) {
        warnings.add("principal_backfill_pending");
    }
    if (input.principalLagStatus === "pending_promotion") {
        warnings.add("active_pack_behind_latest_principal");
    }
    if (input.plan.pending.backfill > 0) {
        warnings.add("passive_backfill_pending");
    }
    if (input.teacherSnapshot.queue.capacity > 0 &&
        input.teacherSnapshot.queue.depth >= input.teacherSnapshot.queue.capacity) {
        warnings.add("teacher_queue_full");
    }
    if (input.teacherSnapshot.diagnostics.latestFreshness === "stale") {
        warnings.add("teacher_labels_stale");
    }
    if (input.teacherSnapshot.diagnostics.lastNoOpReason === "no_teacher_artifacts") {
        warnings.add("teacher_no_artifacts");
    }
    return [...warnings];
}
function summarizeAlwaysOnLearning(input, active) {
    const unavailableLag = {
        activeEventRangeEnd: active?.eventRange.end ?? null,
        latestPrincipalSequence: null,
        sequenceLag: null,
        status: "unavailable"
    };
    const loadedTeacherSurface = loadTeacherSurfaceFromInput(input);
    const teacherSnapshotPath = resolveOperatorTeacherSnapshotPath(input.activationRoot, normalizeOptionalString(input.teacherSnapshotPath) ?? null);
    if (loadedTeacherSurface === null && teacherSnapshotPath === null) {
        return {
            available: false,
            sourcePath: null,
            bootstrapped: null,
            mode: "unavailable",
            nextPriorityLane: "unavailable",
            nextPriorityBucket: "unavailable",
            backlogState: "unavailable",
            pendingLive: null,
            pendingBackfill: null,
            pendingTotal: null,
            pendingByBucket: null,
            freshLivePriority: null,
            principalCheckpointCount: null,
            pendingPrincipalCount: null,
            oldestUnlearnedPrincipalEvent: null,
            newestPendingPrincipalEvent: null,
            leadingPrincipalCheckpoint: null,
            principalCheckpoints: [],
            principalLagToPromotion: unavailableLag,
            warningStates: ["teacher_snapshot_unavailable"],
            learnedRange: null,
            materializationCount: null,
            lastMaterializedAt: null,
            lastMaterializationReason: null,
            lastMaterializationLane: null,
            lastMaterializationPriority: null,
            lastMaterializedPackId: null,
            detail: "no teacher snapshot path supplied"
        };
    }
    if (loadedTeacherSurface === null) {
        return {
            available: false,
            sourcePath: path.resolve(teacherSnapshotPath),
            bootstrapped: null,
            mode: "unavailable",
            nextPriorityLane: "unavailable",
            nextPriorityBucket: "unavailable",
            backlogState: "unavailable",
            pendingLive: null,
            pendingBackfill: null,
            pendingTotal: null,
            pendingByBucket: null,
            freshLivePriority: null,
            principalCheckpointCount: null,
            pendingPrincipalCount: null,
            oldestUnlearnedPrincipalEvent: null,
            newestPendingPrincipalEvent: null,
            leadingPrincipalCheckpoint: null,
            principalCheckpoints: [],
            principalLagToPromotion: unavailableLag,
            warningStates: ["teacher_snapshot_unavailable"],
            learnedRange: null,
            materializationCount: null,
            lastMaterializedAt: null,
            lastMaterializationReason: null,
            lastMaterializationLane: null,
            lastMaterializationPriority: null,
            lastMaterializedPackId: null,
            detail: "teacher snapshot could not be loaded"
        };
    }
    const snapshot = loadedTeacherSurface.snapshot;
    const plan = describeAlwaysOnLearningRuntimeState(snapshot.learner.state, snapshot.learner.lastMaterialization);
    const latestPrincipalSequence = plan.principalBacklog.checkpoints.reduce((latest, checkpoint) => {
        const candidate = checkpoint.newestPendingSequence ?? checkpoint.learnedThroughSequence;
        if (candidate === null) {
            return latest;
        }
        return latest === null ? candidate : Math.max(latest, candidate);
    }, null);
    const activeEventRangeEnd = active?.eventRange.end ?? null;
    const sequenceLag = latestPrincipalSequence === null || activeEventRangeEnd === null
        ? null
        : Math.max(latestPrincipalSequence - activeEventRangeEnd, 0);
    const principalLagStatus = sequenceLag === null
        ? "unavailable"
        : sequenceLag === 0
            ? "caught_up"
            : "pending_promotion";
    const backlogState = summarizeLearningBacklogState(plan, principalLagStatus);
    const warningStates = summarizeLearningWarningStates({
        plan,
        principalLagStatus,
        teacherSnapshot: snapshot
    });
    return {
        available: true,
        sourcePath: loadedTeacherSurface.sourcePath,
        bootstrapped: plan.bootstrapped,
        mode: plan.mode,
        nextPriorityLane: plan.nextPriorityLane,
        nextPriorityBucket: plan.nextPriorityBucket,
        backlogState,
        pendingLive: plan.pending.live,
        pendingBackfill: plan.pending.backfill,
        pendingTotal: plan.pending.total,
        pendingByBucket: { ...plan.pending.byBucket },
        freshLivePriority: plan.pending.freshLivePriority,
        principalCheckpointCount: plan.principalBacklog.principalCount,
        pendingPrincipalCount: plan.principalBacklog.pendingEventCount,
        oldestUnlearnedPrincipalEvent: plan.principalBacklog.oldestUnlearnedEvent,
        newestPendingPrincipalEvent: plan.principalBacklog.newestPendingEvent,
        leadingPrincipalCheckpoint: plan.principalBacklog.checkpoints[0] ?? null,
        principalCheckpoints: plan.principalBacklog.checkpoints,
        principalLagToPromotion: {
            activeEventRangeEnd,
            latestPrincipalSequence,
            sequenceLag,
            status: principalLagStatus
        },
        warningStates,
        learnedRange: plan.learnedRange === null ? null : { ...plan.learnedRange },
        materializationCount: plan.materialization.count,
        lastMaterializedAt: plan.materialization.lastMaterializedAt,
        lastMaterializationReason: plan.materialization.lastReason,
        lastMaterializationLane: plan.materialization.lastLane,
        lastMaterializationPriority: plan.materialization.lastPriority,
        lastMaterializedPackId: snapshot.learner.lastMaterialization?.candidate.summary.packId ?? null,
        detail: plan.nextPriorityBucket === "principal_immediate"
            ? "principal-priority live slices are next; passive backfill stays behind live intake"
            : plan.nextPriorityBucket === "live"
                ? "fresh live slices are next; passive backfill stays behind live intake"
                : plan.nextPriorityBucket === "principal_backfill"
                    ? "live intake is clear; principal-priority passive backfill is next"
                    : plan.nextPriorityBucket === "backfill"
                        ? "live intake is clear; passive backfill is next"
                        : plan.bootstrapped
                            ? "fast-init has handed off to the current learned export without queued backlog"
                            : "learner is waiting for the first export"
    };
}
function buildOperatorFindings(report) {
    const findings = [];
    const push = (severity, code, summary, detail) => {
        findings.push({ severity, code, summary, detail });
    };
    if (report.hook.desynced) {
        push("fail", "hook_desynced", "profile hook is present but not loadable", report.hook.detail);
    }
    else if (report.attachmentTruth.state === "not_attached") {
        push("fail", "current_profile_not_attached", "current profile is not attached", report.attachmentTruth.detail);
    }
    else if (report.attachmentTruth.state === "unknown") {
        push("warn", "attachment_scope_partial", "current-profile attachment is not self-proven", report.attachmentTruth.detail);
    }
    else {
        push("pass", "current_profile_attached", "current profile hook is present", report.attachmentTruth.detail);
    }
    if (report.activation.state === "broken_install") {
        push("fail", "activation_broken_install", "activation root is broken", report.activation.detail);
    }
    else if (report.activation.state === "stale_incomplete") {
        push("fail", "activation_stale_incomplete", "activation root is stale or incomplete", report.activation.detail);
    }
    else if (report.active === null) {
        push("fail", "active_missing", "active slot is empty", "no active pack found; this is the pre-bootstrap state — call `bootstrapRuntimeAttach()` to activate an initial pack before compiling or serving");
    }
    else if (!report.active.activationReady) {
        push("fail", "active_unhealthy", `active slot is not activation-ready: ${report.active.packId}`, report.active.findings.join("; ") || "inspect the active pack payloads and activation pointers");
    }
    else {
        push("pass", "active_ready", `active slot is ready: ${report.active.packId}`, "serving can inspect the active pack without activation drift");
        if (isAwaitingFirstExportSlot(report.active)) {
            push("warn", "bootstrap_waiting_for_first_export", "active pack bootstrapped without live exports yet", "serving is up from seed-state defaults; this is expected on first install — learning activates after the first turn is captured via `runRuntimeTurn()`");
        }
    }
    if (report.learnedRouting.required) {
        if (report.learnedRouting.available) {
            push("pass", "learned_route_ready", `learned routing is active: ${report.learnedRouting.routerIdentity ?? "unknown-router"}`, `updateCount=${report.learnedRouting.updateCount ?? 0}; labels=${report.learnedRouting.collectedLabelsTotal ?? 0}; handoff=${report.learnedRouting.handoffState}`);
        }
        else {
            push("fail", "learned_route_missing", "active pack requires learned routing but the learned route artifact is unavailable", "repair the router artifact or promote a healthy learned-routing candidate before serving");
        }
    }
    else {
        push("pass", "learned_route_optional", "active pack does not require learned routing", `handoff=${report.learnedRouting.handoffState}`);
    }
    if (report.servePath.state === "serving_active_pack") {
        push("pass", "serve_path_verified", `serve path compiles from active pack ${report.servePath.activePackId ?? "unknown-pack"}`, `selection=${report.servePath.selectionMode ?? "unknown"}; tiers=${report.servePath.contextAttribution.selectionTiers ?? "unknown"}; router=${report.servePath.routerIdentity ?? "none"}; routeFreshness=${report.servePath.freshnessChecksum ?? "unknown"}`);
        push("pass", "structural_budget_visible", `structural budget resolves to ${report.servePath.resolvedMaxContextBlocks ?? "unknown"} blocks`, `origin=${report.servePath.structuralDecision.origin}; basis=${report.servePath.structuralDecision.basis}; requested=${report.servePath.requestedBudgetStrategy ?? "unknown"}; resolved=${report.servePath.resolvedBudgetStrategy ?? "unknown"}; source=${report.servePath.structuralBudgetSource ?? "unknown"}; evidence=${report.servePath.structuralBudgetEvidence ?? "none"}; pressures=${report.servePath.structuralBudgetPressures ?? "none"}`);
    }
    else if (report.servePath.state === "fail_open_static_context") {
        push("warn", "serve_path_fail_open", "serve path would fail open to static context", report.servePath.error ?? "compile probe fell back to static context");
    }
    else if (report.servePath.state === "hard_fail") {
        push("fail", "serve_path_hard_fail", "serve path would hard fail on the learned-routing requirement", report.servePath.error ?? "compile probe violated a learned-routing hard requirement");
    }
    else {
        push("warn", "serve_path_unprobed", "serve path was not probed", "operator surface could not verify fail-open versus hard-fail behavior");
    }
    if (report.learnedRouting.required && report.servePath.state === "serving_active_pack" && report.servePath.usedLearnedRouteFn !== true) {
        push("fail", "serve_path_route_evidence_missing", "serve path compiled without learned-route evidence on a learned-routing pack", `router=${report.servePath.routerIdentity ?? report.learnedRouting.routerIdentity ?? "none"}; selection=${report.servePath.selectionMode ?? "unknown"}`);
    }
    if (report.servePath.state === "serving_active_pack") {
        if (report.servePath.contextAttribution.brainCompiledBlockCount > 0) {
            push("pass", "brain_context_visible", "serve probe selected brain-compiled context", `brainSources=${formatList(report.servePath.contextAttribution.brainCompiledSources)}; kernelSources=${formatList(report.servePath.contextAttribution.stableKernelSources)}; evidence=${report.servePath.contextAttribution.evidence}`);
        }
        else {
            push("warn", "brain_context_kernel_only", "serve probe stayed inside the stable kernel", `kernelSources=${formatList(report.servePath.contextAttribution.stableKernelSources)}; evidence=${report.servePath.contextAttribution.evidence}; ${report.servePath.contextAttribution.detail}`);
        }
    }
    if (report.candidate === null) {
        push("pass", "candidate_missing", "no candidate pack is currently staged", "steady state can legitimately run without a staged candidate until the next refresh lands");
    }
    else if (!report.candidate.activationReady) {
        push("warn", "candidate_unhealthy", `candidate slot is not activation-ready: ${report.candidate.packId}`, report.candidate.findings.join("; ") || "fix candidate pack payloads before promotion");
    }
    else if (report.promotion.allowed) {
        push("pass", "promotion_ready", `candidate is promotion-ready: ${report.candidate.packId}`, report.freshness.candidateAheadBy.length === 0
            ? "candidate is staged and promotion is allowed"
            : `candidate is ahead on ${report.freshness.candidateAheadBy.join(", ")}`);
    }
    else {
        push("warn", "promotion_blocked", `candidate is staged but promotion is blocked: ${report.candidate.packId}`, report.promotion.findings.join("; ") || "inspect freshness and activation findings before promotion");
    }
    if (report.promotion.lastPromotion.known) {
        push("pass", "last_promotion_known", `last promotion is proven at ${report.promotion.lastPromotion.at}`, report.promotion.lastPromotion.note);
    }
    else {
        push("warn", "last_promotion_unknown", "last promotion is not provable from local activation pointers", report.promotion.lastPromotion.note);
    }
    if (report.rollback.allowed) {
        push("pass", "rollback_ready", `rollback is ready to restore ${report.rollback.previousPackId ?? "the previous pack"}`, report.previous === null ? "previous pack is pointer-visible even if slot inspection is unavailable" : `previous pack=${report.previous.packId}`);
    }
    else {
        push("warn", "rollback_blocked", "rollback is not ready", report.rollback.findings.join("; ") || "previous pointer is missing or no rollback target is retained");
    }
    if (!report.supervision.available) {
        push("warn", "supervision_unavailable", "supervision flow is not inspectable yet", "pass `--event-export <bundle-root-or-payload>` to inspect local supervision freshness");
    }
    else if (report.supervision.flowing) {
        push("pass", "supervision_visible", `supervision is flowing through ${report.supervision.freshestSourceStream ?? "unknown-source"}`, `freshest=${report.supervision.freshestCreatedAt ?? "unknown"}; humanLabels=${report.supervision.humanLabelCount ?? 0}`);
    }
    else {
        push("warn", "supervision_not_flowing", "the supplied export does not yet show human supervision", `sourcePath=${report.supervision.sourcePath ?? "unknown"}; exportDigest=${report.supervision.exportDigest ?? "unknown"}`);
    }
    if (report.supervision.available) {
        if (report.supervision.scanSurfaceCount > 0) {
            push("pass", "scan_surfaces_visible", `scanner surfaces are visible: ${formatList(report.supervision.scanSurfaces)}`, `scanPolicy=${report.supervision.scanPolicy ?? "unknown"}; humanLabels=${report.supervision.humanLabelCount ?? 0}; selfLabels=${report.supervision.selfLabelCount ?? 0}`);
        }
        else {
            push("warn", "scan_surfaces_missing", "scanner surfaces are not visible in the supplied export", `exportDigest=${report.supervision.exportDigest ?? "unknown"}`);
        }
        if (report.supervision.totalEventCount !== null && report.supervision.totalEventCount > 0) {
            if (report.supervision.attributedEventCount === report.supervision.totalEventCount) {
                push("pass", "turn_attribution_visible", `all supplied events are attributable: ${report.supervision.attributedEventCount}/${report.supervision.totalEventCount}`, `selectionDigests=${report.supervision.selectionDigestCount ?? 0}`);
            }
            else {
                push("warn", "turn_attribution_partial", `some supplied events are unattributed: ${report.supervision.attributedEventCount ?? 0}/${report.supervision.totalEventCount}`, `selectionDigests=${report.supervision.selectionDigestCount ?? 0}`);
            }
        }
    }
    if (!report.teacherLoop.available) {
        push("warn", "teacher_snapshot_unavailable", "last async no-op reason is not inspectable yet", "pass `--teacher-snapshot <snapshot.json>` to inspect duplicate/no-op handling");
    }
    else {
        push("pass", "teacher_snapshot_loaded", `last async no-op reason is ${report.teacherLoop.lastNoOpReason}`, `latestFreshness=${report.teacherLoop.latestFreshness}; queue=${report.teacherLoop.queueDepth}/${report.teacherLoop.queueCapacity}`);
    }
    return findings;
}
function summarizeOperatorStatus(findings) {
    if (findings.some((finding) => finding.severity === "fail")) {
        return "fail";
    }
    if (findings.some((finding) => finding.severity === "warn")) {
        return "warn";
    }
    return "ok";
}
function yesNo(value) {
    if (value === null) {
        return "unknown";
    }
    return value ? "yes" : "no";
}
function formatList(values, empty = "none") {
    return values.length === 0 ? empty : values.join(",");
}
function formatCompactList(values, empty = "none", maxItems = 2, maxLength = 20) {
    if (values.length === 0) {
        return empty;
    }
    const visible = values.slice(0, maxItems).map((value) => formatCompactValue(value, empty, maxLength));
    return values.length > maxItems ? `${visible.join("|")}+${values.length - maxItems}more` : visible.join("|");
}
function formatCompactValue(value, empty = "none", maxLength = 24) {
    if (value === null || value === undefined || value.length === 0) {
        return empty;
    }
    return value.length <= maxLength ? value : `${value.slice(0, maxLength)}…`;
}
function summarizeCurrentProfileLogRoot(activationRoot) {
    const logRoot = path.join(path.resolve(activationRoot), LEARNING_SPINE_LOG_LAYOUT.dir);
    return existsSync(logRoot) ? logRoot : null;
}
function summarizeCurrentProfileLastLearningUpdateAt(activationRoot, learning, teacherLoop) {
    const updates = readLearningSpineLogEntries(activationRoot, "pgRouteUpdates");
    return updates.at(-1)?.recordedAt ?? teacherLoop.lastRunAt ?? learning.lastMaterializedAt ?? null;
}
function didCurrentProfileFirstExportOccur(report) {
    if ((report.active?.eventRange.count ?? 0) > 0) {
        return true;
    }
    if (report.supervision.exportedAt !== null) {
        return true;
    }
    if (report.learning.available && (report.learning.learnedRange?.count ?? 0) > 0) {
        return true;
    }
    if (!report.teacherLoop.available) {
        return false;
    }
    return ((report.teacherLoop.replayedBundleCount ?? 0) > 0 ||
        (report.teacherLoop.exportedBundleCount ?? 0) > 0 ||
        report.teacherLoop.lastProcessedAt !== null ||
        report.teacherLoop.lastMaterializedPackId !== null);
}
function summarizeCurrentProfilePassiveLearning(report, activePackId) {
    const firstExportOccurred = didCurrentProfileFirstExportOccur(report);
    const exportState = !firstExportOccurred
        ? "awaiting_first_export"
        : report.supervision.exportedAt !== null
            ? "latest_export_visible"
            : "history_only";
    const backlogState = report.learning.available && report.learning.backlogState !== "unavailable"
        ? report.learning.backlogState
        : "unknown";
    const detail = !firstExportOccurred
        ? report.teacherLoop.watchState === "watching"
            ? "watch heartbeat is fresh, but this activation root has not observed its first export yet"
            : "this activation root is still waiting for the first export before passive learning can advance"
        : backlogState === "unknown"
            ? "first export is proven, but passive backlog state is not visible from the current local artifacts"
            : report.teacherLoop.watchState === "watching"
                ? `watch heartbeat is fresh; passive backlog is ${backlogState} with live=${report.learning.pendingLive ?? 0} and backfill=${report.learning.pendingBackfill ?? 0}`
                : report.teacherLoop.watchState === "stale_snapshot"
                    ? `last saved watch snapshot is stale; latest known passive backlog is ${backlogState}`
                    : report.teacherLoop.watchState === "snapshot_only"
                        ? `passive backlog is visible from the last saved snapshot: ${backlogState}`
                        : `passive backlog is visible from the last known learner state: ${backlogState}`;
    return {
        learnerRunning: report.teacherLoop.watchState === "watching",
        firstExportOccurred,
        watchState: report.teacherLoop.watchState,
        exportState,
        backlogState,
        pendingLive: report.learning.available ? report.learning.pendingLive : null,
        pendingBackfill: report.learning.available ? report.learning.pendingBackfill : null,
        lastWatchHeartbeatAt: report.teacherLoop.lastHeartbeatAt,
        watchIntervalSeconds: report.teacherLoop.pollIntervalSeconds,
        lastExportAt: report.supervision.exportedAt,
        lastPromotionAt: report.promotion.lastPromotion.at,
        currentServingPackId: activePackId,
        lastMaterializedPackId: report.learning.lastMaterializedPackId ?? report.teacherLoop.lastMaterializedPackId,
        lastObservedDelta: cloneLastObservedDelta(report.teacherLoop.lastObservedDelta),
        detail
    };
}
function summarizeOperatorHook(report) {
    return summarizeOpenClawBrainHookLoad(inspectOpenClawBrainHookStatus(report.openclawHome ?? null), report.servePath.state === "serving_active_pack");
}
function summarizeOperatorAttachmentTruth(report) {
    const watchOnly = report.active !== null || report.teacherLoop.watchState !== "not_visible";
    if (report.hook.scope === "exact_openclaw_home") {
        if (report.hook.installState === "not_installed") {
            return {
                state: "not_attached",
                proofState: "self_proving",
                watchOnly,
                activationRoot: null,
                servingSlot: "none",
                detail: watchOnly
                    ? "the selected OpenClaw home has no OpenClawBrain hook, even though this activation root still shows serve/watch activity"
                    : "the selected OpenClaw home has no OpenClawBrain hook"
            };
        }
        return {
            state: "attached",
            proofState: "self_proving",
            watchOnly: false,
            activationRoot: report.activationRoot,
            servingSlot: report.active === null ? "none" : "active",
            detail: report.hook.loadability === "blocked"
                ? "current profile hook files exist, but OpenClaw will not load them until the installed hook is repaired"
                : "current profile hook is present on the selected OpenClaw home"
        };
    }
    return {
        state: "unknown",
        proofState: "activation_root_only",
        watchOnly,
        activationRoot: watchOnly ? report.activationRoot : null,
        servingSlot: report.active === null ? "none" : "active",
        detail: watchOnly
            ? "activation-root-only status can see this root serving and/or being watched, but it does not prove that the current profile is attached"
            : "activation-root-only status cannot prove whether the current profile is attached"
    };
}
function summarizeCurrentProfileBrainSummary(input) {
    const packId = input.activePackId ?? "unknown";
    if (input.activationState === "broken_install") {
        return "Brain activation is broken and needs repair before serve-path truth can be trusted.";
    }
    if (input.activationState === "stale_incomplete") {
        return "Brain activation has retained stale/incomplete state and no serving active pack.";
    }
    if (input.attachmentState === "not_attached") {
        return "Brain is not attached to the current Profile.";
    }
    if (input.hookDesynced) {
        return "Brain hook files exist, but the installed hook is not loadable yet.";
    }
    if (input.attachmentState === "unknown") {
        return input.watchOnly
            ? "This activation root is visible, but current-profile attachment is only watch/serve scoped and not self-proven."
            : "Current-profile attachment is unknown because this read is activation-root-only.";
    }
    if (input.serveState === "fail_open_static_context") {
        return "Brain is attached but would fail open to static context.";
    }
    if (input.serveState === "hard_fail") {
        return "Brain is attached but currently hard-fails learned routing.";
    }
    if (input.awaitingFirstExport) {
        return `Brain is serving seed-state pack ${packId} and awaiting the first export.`;
    }
    if (input.brainState === "pg_promoted_pack_authoritative") {
        return `Brain is serving promoted pack ${packId}; serve-visible change came from activation promotion.`;
    }
    if (input.serveState === "serving_active_pack") {
        return `Brain is serving active pack ${packId}; learned routing is live, but authority is still seed-state.`;
    }
    return "Brain is attached but has not been compile-probed yet.";
}
function summarizeCurrentProfileBrainStatusLevel(input) {
    if (input.hookDesynced || input.attachmentState === "not_attached") {
        return "fail";
    }
    if (input.serveState === "fail_open_static_context" || input.serveState === "hard_fail") {
        return "fail";
    }
    if (input.attachmentState === "unknown" || input.awaitingFirstExport || input.serveState === "unprobed") {
        return "warn";
    }
    return input.routeFreshness === "updated" ? "ok" : "warn";
}
function buildCurrentProfileTurnAttributionFromReport(report, policyMode, profileId) {
    if (report.servePath.state === "unprobed" || report.attachmentTruth.state !== "attached") {
        return null;
    }
    const brainStatus = report.servePath.state;
    const sessionId = "current-profile-status-probe";
    const channel = "operator_status";
    const createdAt = report.generatedAt;
    const sourceStream = "openclaw/operator/status";
    const packId = brainStatus === "serving_active_pack"
        ? report.servePath.activePackId ?? report.brain.activePackId ?? report.active?.packId ?? null
        : null;
    const routerIdentity = brainStatus === "serving_active_pack"
        ? report.servePath.routerIdentity ?? report.learnedRouting.routerIdentity ?? report.active?.routerIdentity ?? null
        : null;
    const usedLearnedRouteFn = brainStatus === "serving_active_pack" ? report.servePath.usedLearnedRouteFn : null;
    const selectionMode = brainStatus === "serving_active_pack" ? report.servePath.selectionMode : null;
    const selectionDigest = brainStatus === "serving_active_pack" ? report.servePath.selectionDigest : null;
    const contextAttribution = report.servePath.contextAttribution;
    const probeTurn = {
        sessionId,
        channel,
        sourceStream,
        userMessage: DEFAULT_ATTACH_STATUS_MESSAGE,
        createdAt,
        runtimeHints: [...DEFAULT_ATTACH_STATUS_RUNTIME_HINTS],
        profileSelector: "current_profile",
        profileId,
        brainAttachmentPolicy: policyMode
    };
    return {
        contract: CONTRACT_IDS.profileTurnAttribution,
        hostRuntimeOwner: "openclaw",
        profileSelector: "current_profile",
        profileId,
        brainAttachmentPolicy: policyMode,
        brainStatus,
        sessionId,
        channel,
        interactionEventId: deterministicEventId({
            source: sourceStream,
            activationRoot: report.activationRoot,
            createdAt,
            brainStatus,
            packId,
            routerIdentity,
            selectionDigest
        }),
        createdAt,
        packId,
        routerIdentity,
        usedLearnedRouteFn,
        selectionMode,
        selectionTiers: contextAttribution.selectionTiers,
        selectionDigest,
        contextFingerprint: buildRuntimeContextFingerprint({
            turn: probeTurn,
            sourceStream,
            profileSelector: "current_profile",
            brainAttachmentPolicy: policyMode,
            brainStatus,
            activePackId: packId,
            usedLearnedRouteFn,
            routerIdentity,
            selectionDigest
        }),
        selectedContextCount: contextAttribution.selectedContextCount,
        stableKernelBlockCount: contextAttribution.stableKernelBlockCount,
        brainCompiledBlockCount: contextAttribution.brainCompiledBlockCount,
        stableKernelSources: [...contextAttribution.stableKernelSources],
        brainCompiledSources: [...contextAttribution.brainCompiledSources],
        contextEvidence: contextAttribution.evidence === "unprobed" ? "stable_kernel_only" : contextAttribution.evidence,
        detail: brainStatus === "serving_active_pack"
            ? "current profile status probe compiled from the active pack with explicit route-function and block-source attribution"
            : brainStatus === "hard_fail"
                ? "current profile status probe hit a learned-route hard fail before serving context could compile"
                : "current profile status probe fail-opened to static context because no serving pack was available"
    };
}
function buildCurrentProfileBrainStatusFromReport(report, policyMode, profileId) {
    const attachmentState = report.attachmentTruth.state;
    const awaitingFirstExport = isAwaitingFirstExportSlot(report.active);
    const activationState = report.activation.state;
    const routerIdentity = report.servePath.routerIdentity ?? report.learnedRouting.routerIdentity ?? report.active?.routerIdentity ?? null;
    const routeFreshness = report.servePath.refreshStatus === "updated" || report.servePath.refreshStatus === "no_supervision"
        ? report.servePath.refreshStatus
        : "unknown";
    const activePackId = report.brain.activePackId ?? report.servePath.activePackId ?? report.active?.packId ?? null;
    const passiveLearning = summarizeCurrentProfilePassiveLearning(report, activePackId);
    const status = summarizeCurrentProfileBrainStatusLevel({
        attachmentState,
        hookDesynced: report.hook.desynced,
        serveState: report.servePath.state,
        routeFreshness,
        awaitingFirstExport
    });
    return {
        contract: CURRENT_PROFILE_BRAIN_STATUS_CONTRACT,
        generatedAt: report.generatedAt,
        host: {
            noun: "Host",
            runtimeOwner: "openclaw",
            activationRoot: report.activationRoot
        },
        profile: {
            noun: "Profile",
            selector: "current_profile",
            profileId,
            detail: attachmentState === "attached"
                ? "The Host resolves the current Profile through the selected OpenClaw home and its active Attachment boundary."
                : attachmentState === "not_attached"
                    ? "The selected OpenClaw home has no OpenClawBrain hook for the current Profile."
                    : "This read is activation-root-only, so current-profile attachment is not self-proven."
        },
        brain: {
            noun: "Brain",
            activationRoot: attachmentState === "not_attached" ? null : report.activationRoot,
            logRoot: summarizeCurrentProfileLogRoot(report.activationRoot),
            activePackId,
            initMode: report.learnedRouting.initMode,
            state: report.brain.state,
            routeFreshness,
            routerIdentity,
            routerChecksum: report.learnedRouting.routerChecksum,
            lastExportAt: report.supervision.exportedAt,
            lastLearningUpdateAt: summarizeCurrentProfileLastLearningUpdateAt(report.activationRoot, report.learning, report.teacherLoop),
            lastPromotionAt: report.promotion.lastPromotion.at,
            summary: summarizeCurrentProfileBrainSummary({
                activationState,
                attachmentState,
                watchOnly: report.attachmentTruth.watchOnly,
                hookDesynced: report.hook.desynced,
                serveState: report.servePath.state,
                brainState: report.brain.state,
                awaitingFirstExport,
                activePackId,
                activationDetail: report.activation.detail
            }),
            detail: activationState === "broken_install" || activationState === "stale_incomplete" || activationState === "detached"
                ? report.activation.detail
                : report.brain.detail
        },
        hook: {
            noun: "Hook",
            scope: report.hook.scope,
            openclawHome: report.hook.openclawHome,
            extensionDir: report.hook.extensionDir,
            hookPath: report.hook.hookPath,
            runtimeGuardPath: report.hook.runtimeGuardPath,
            manifestPath: report.hook.manifestPath,
            packageJsonPath: report.hook.packageJsonPath,
            manifestId: report.hook.manifestId,
            installId: report.hook.installId,
            packageName: report.hook.packageName,
            installLayout: report.hook.installLayout,
            additionalInstallCount: report.hook.additionalInstallCount,
            installState: report.hook.installState,
            loadability: report.hook.loadability,
            loadProof: report.hook.loadProof,
            desynced: report.hook.desynced,
            detail: report.hook.detail
        },
        attachment: {
            noun: "Attachment",
            state: attachmentState,
            activationRoot: report.attachmentTruth.activationRoot,
            servingSlot: report.attachmentTruth.servingSlot,
            policyMode,
            policy: buildCurrentProfileAttachmentPolicy(policyMode),
            proofState: report.attachmentTruth.proofState,
            watchOnly: report.attachmentTruth.watchOnly,
            detail: report.attachmentTruth.detail
        },
        brainStatus: {
            status,
            brainState: report.brain.state,
            serveState: report.servePath.state,
            activationState,
            usedLearnedRouteFn: report.servePath.usedLearnedRouteFn,
            failOpen: report.servePath.fallbackToStaticContext,
            awaitingFirstExport,
            structuralDecision: report.servePath.structuralDecision,
            timing: report.servePath.timing,
            detail: activationState === "broken_install"
                ? `current profile activation is broken: ${report.activation.detail}`
                : report.hook.desynced
                    ? `current profile hook is blocked: ${report.hook.detail}`
                    : activationState === "stale_incomplete"
                        ? `current profile activation is stale/incomplete: ${report.activation.detail}`
                        : attachmentState === "not_attached"
                            ? "current profile is not attached to this Brain activation root"
                            : attachmentState === "unknown"
                                ? "current profile attachment is not self-proven from this activation-root-only read"
                                : activationState === "detached"
                                    ? "current profile has no attached active pack at the activation boundary"
                                    : report.servePath.state === "serving_active_pack"
                                        ? awaitingFirstExport
                                            ? `current profile is serving seed-state pack ${activePackId ?? "unknown"} while awaiting the first exported turn`
                                            : report.brain.state === "pg_promoted_pack_authoritative"
                                                ? `current profile is serving promoted pack ${activePackId ?? "unknown"}; serve-visible change came from activation promotion, not hot-path mutation`
                                                : `current profile is serving active pack ${activePackId ?? "unknown"}; learned routing is active, but authority is still seed-state`
                                        : report.servePath.state === "fail_open_static_context"
                                            ? "current profile would fail open to static context because no serving pack is available"
                                            : report.servePath.state === "hard_fail"
                                                ? "current profile cannot serve because the learned-route or activation requirement hard-failed"
                                                : "current profile serve state has not been compile-probed yet"
        },
        passiveLearning,
        currentTurnAttribution: buildCurrentProfileTurnAttributionFromReport(report, policyMode, profileId)
    };
}
export function buildOperatorSurfaceReport(input) {
    const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    const updatedAt = normalizeIsoTimestamp(input.updatedAt, "updatedAt", new Date().toISOString());
    const brainAttachmentPolicy = normalizeBrainAttachmentPolicy(input.brainAttachmentPolicy);
    let inspection = null;
    let observability = null;
    let inspectionError = null;
    try {
        inspection = inspectActivationState(activationRoot, updatedAt);
    }
    catch (error) {
        inspectionError = toErrorMessage(error);
    }
    const active = inspection === null ? null : summarizeOperatorSlot(inspection.active, inspection.pointers.active?.updatedAt ?? null);
    const candidate = inspection === null ? null : summarizeOperatorSlot(inspection.candidate, inspection.pointers.candidate?.updatedAt ?? null);
    const previous = inspection === null ? null : summarizeOperatorSlot(inspection.previous, inspection.pointers.previous?.updatedAt ?? null);
    if (inspection !== null && inspection.active !== null) {
        try {
            observability = describeActivationObservability(activationRoot, "active", {
                updatedAt
            });
        }
        catch (error) {
            inspectionError ??= `activation observability failed: ${toErrorMessage(error)}`;
        }
    }
    const activation = summarizeOperatorActivationState({
        inspection,
        observability,
        active,
        candidate,
        previous,
        inspectionError
    });
    const activeObservability = inspectionError === null
        ? summarizeActivePackObservability(activationRoot, active)
        : {
            labelFlow: buildMissingLabelFlowSummary(`activation observability is unavailable: ${inspectionError}`),
            learningPath: buildMissingLearningPathSummary(`activation observability is unavailable: ${inspectionError}`)
        };
    const servePath = probeOperatorServePath(activationRoot, observability, active?.packId ?? null);
    const learnedRouting = observability === null
        ? summarizeLearnedRoutingWithoutObservability(active)
        : {
            required: observability.learnedRouteFn.required,
            available: observability.learnedRouteFn.available,
            routerIdentity: observability.learnedRouteFn.routerIdentity,
            routeFnVersion: observability.learnedRouteFn.routeFnVersion,
            trainingMethod: observability.learnedRouteFn.trainingMethod,
            routerTrainedAt: observability.learnedRouteFn.routerTrainedAt,
            objective: observability.learnedRouteFn.objective,
            pgProfile: observability.learnedRouteFn.pgProfile,
            routerChecksum: observability.learnedRouteFn.routerChecksum,
            objectiveChecksum: observability.learnedRouteFn.objectiveChecksum,
            updateMechanism: observability.learnedRouteFn.updateMechanism,
            updateVersion: observability.learnedRouteFn.updateVersion,
            updateCount: observability.learnedRouteFn.updateCount,
            supervisionCount: observability.learnedRouteFn.supervisionCount,
            collectedLabelsTotal: observability.learnedRouteFn.collectedLabels?.total ?? null,
            freshnessChecksum: observability.learnedRouteFn.freshnessChecksum,
            handoffState: observability.initHandoff.handoffState,
            initMode: observability.initHandoff.initMode,
            seedStateVisible: observability.initHandoff.seedStateVisible
        };
    const routeFn = summarizeRouteFnFreshness({
        activationRoot,
        activePackId: active?.packId ?? null,
        learnedRouting
    });
    const teacherLoop = summarizeTeacherLoop(input);
    const hook = summarizeOperatorHook({
        activationRoot,
        openclawHome: normalizeOptionalString(input.openclawHome) ?? null,
        servePath
    });
    const attachmentTruth = summarizeOperatorAttachmentTruth({
        activationRoot,
        active,
        teacherLoop,
        hook
    });
    const reportBase = {
        generatedAt: updatedAt,
        activationRoot,
        activation,
        active,
        candidate,
        previous,
        freshness: {
            activeBehindPromotionReadyCandidate: observability?.promotionFreshness.activeBehindPromotionReadyCandidate ?? false,
            candidateAheadBy: summarizeCandidateAheadBy(observability?.promotionFreshness.candidateAheadBy ?? null)
        },
        brain: observability === null ? summarizeBrainStateWithoutObservability(active, activation) : summarizeBrainState(active, observability),
        graph: observability === null
            ? summarizeGraphWithoutObservability(input, active, activation)
            : summarizeGraphObservability(input, active, observability),
        labelFlow: activeObservability.labelFlow,
        learningPath: activeObservability.learningPath,
        learnedRouting,
        servePath,
        promotion: {
            allowed: inspection?.promotion.allowed ?? false,
            findings: [...(inspection?.promotion.findings ?? [])],
            lastPromotion: inspection === null ? {
                known: false,
                at: null,
                confidence: "unknown_from_local_pointers",
                note: activation.detail
            } : summarizeLastPromotion(inspection),
            activeUpdatedAt: inspection?.pointers.active?.updatedAt ?? null,
            candidateUpdatedAt: inspection?.pointers.candidate?.updatedAt ?? null,
            previousUpdatedAt: inspection?.pointers.previous?.updatedAt ?? null
        },
        rollback: {
            allowed: inspection?.rollback.allowed ?? false,
            findings: [...(inspection?.rollback.findings ?? [])],
            previousPackId: inspection?.previous?.packId ?? inspection?.pointers.previous?.packId ?? null,
            state: inspection === null ? "unknown" : inspection.rollback.allowed ? "ready" : inspection.active === null ? "unknown" : "blocked"
        },
        supervision: summarizeSupervision(input),
        learning: summarizeAlwaysOnLearning(input, active),
        teacherLoop,
        routeFn,
        hook,
        attachmentTruth,
        principal: summarizePrincipalObservability(input, active),
        manyProfile: summarizeManyProfileSupport(brainAttachmentPolicy)
    };
    const findings = buildOperatorFindings(reportBase);
    return {
        ...reportBase,
        status: summarizeOperatorStatus(findings),
        findings
    };
}
export function describeCurrentProfileBrainStatus(input) {
    const report = buildOperatorSurfaceReport(input);
    return buildCurrentProfileBrainStatusFromReport(report, report.manyProfile.declaredAttachmentPolicy, normalizeOptionalString(input.profileId) ?? null);
}
export function formatOperatorRollbackReport(result) {
    const header = result.allowed ? (result.dryRun ? "ROLLBACK ready" : "ROLLBACK ok") : "ROLLBACK blocked";
    return [
        header,
        `preview     ${yesNo(result.dryRun)} activation=${result.activationRoot} updatedAt=${result.updatedAt}`,
        `before      active=${result.before.activePackId ?? "none"} candidate=${result.before.candidatePackId ?? "none"} previous=${result.before.previousPackId ?? "none"}`,
        `after       active=${result.after?.activePackId ?? "none"} candidate=${result.after?.candidatePackId ?? "none"} previous=${result.after?.previousPackId ?? "none"}`,
        `result      restored=${result.restoredPackId ?? "none"} parkedCandidate=${result.parkedCandidatePackId ?? "none"}`,
        `findings    ${formatList(result.findings)}`
    ].join("\n");
}
/**
 * Describes the kernel/brain boundary for a single compile response.
 *
 * Combines:
 * - Brain context summary (from the compile response diagnostics)
 * - Kernel surface validation (if a surface descriptor is supplied)
 * - A coverage advisory based on routing signals
 *
 * See `docs/kernel-brain-boundary.md` for the full decision framework.
 */
export function describeKernelBrainBoundary(compileResponse, surface) {
    const diag = compileResponse.diagnostics;
    // Collect the roles of selected blocks.
    const selectedRoles = [
        ...new Set(compileResponse.selectedContext
            .map((b) => b.source)
            .filter((s) => typeof s === "string" && s.length > 0))
    ];
    // Detect whether any block was compacted (compactedFrom set on the block).
    const compactionApplied = compileResponse.selectedContext.some((b) => Array.isArray(b.compactedFrom) && (b.compactedFrom?.length ?? 0) > 0);
    // Coverage advisory.
    // Token match evidence lives in diagnostics.notes as "selection_mode=token_match(...)"
    // or "selection_tiers=token_match_only" / "selection_tiers=token_match+priority_fallback".
    const notesStr = diag.notes.join(" ");
    const hasTokenMatches = notesStr.includes("selection_mode=token_match") ||
        notesStr.includes("selection_tiers=token_match");
    let brainCoverageAdvisory;
    if (diag.usedLearnedRouteFn && hasTokenMatches) {
        brainCoverageAdvisory = "likely_covered";
    }
    else if (hasTokenMatches || diag.usedLearnedRouteFn) {
        brainCoverageAdvisory = "partial";
    }
    else {
        brainCoverageAdvisory = "likely_gap";
    }
    const kernelValidation = surface !== undefined ? validateKernelSurface(surface) : null;
    return {
        brain: {
            packId: compileResponse.packId,
            mode: diag.modeEffective,
            selectedBlockCount: compileResponse.selectedContext.length,
            selectedRoles,
            usedLearnedRouteFn: diag.usedLearnedRouteFn,
            compactionApplied
        },
        kernelValidation,
        brainCoverageAdvisory
    };
}
// Re-export install-first helpers so fresh consumers can start from one package.
export { CONTRACT_IDS, buildNormalizedEventExport, createFeedbackEvent, createInteractionEvent, validateRuntimeCompileRequest } from "@openclawbrain/contracts";
export { describeNormalizedEventExportObservability } from "@openclawbrain/event-export";
export { describeCompileFallbackUsage } from "@openclawbrain/compiler";
export { describeActivationObservability, inspectActivationState, rollbackActivePack } from "@openclawbrain/pack-format";
export { createOpenClawLocalSessionTail, OpenClawLocalSessionTail } from "./session-tail.js";
export { discoverOpenClawMainSessionStores, discoverOpenClawSessionStores, loadOpenClawSessionIndex, readOpenClawAcpStreamFile, readOpenClawSessionFile } from "./session-store.js";
export { buildPassiveLearningSessionExportFromOpenClawSessionStore, buildPassiveLearningStoreExportFromOpenClawSessionIndex } from "./local-session-passive-learning.js";
export { DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_TIMEOUT_MS, OllamaClient, OllamaClientError, createOllamaClient } from "./ollama-client.js";
export { resolveActivationRoot } from "./resolve-activation-root.js";
export { runDaemonCommand, parseDaemonArgs } from "./daemon.js";
//# sourceMappingURL=index.js.map
