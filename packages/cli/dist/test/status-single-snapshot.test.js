import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunction({ file, startMarker, endMarker, prelude = "" }) {
    const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${file}`);
    }
    const block = source.slice(start, end).replace(/^export\s+/gmu, "");
    const match = /function\s+([A-Za-z0-9_]+)/u.exec(startMarker);
    if (match === null) {
        throw new Error(`failed to extract function name from ${startMarker}`);
    }
    return new Function(`${prelude}\n${block}\nreturn ${match[1]};`)();
}

test("current profile status with report reuses one live operator snapshot", () => {
    const describeCurrentProfileBrainStatusWithReport = loadFunction({
        file: "index.js",
        startMarker: "export function describeCurrentProfileBrainStatusWithReport",
        endMarker: "export function describeCurrentProfileBrainStatus(input)",
        prelude: `
            let buildCalls = 0;
            function buildOperatorSurfaceReport() {
                buildCalls += 1;
                return {
                    marker: buildCalls,
                    manyProfile: {
                        declaredAttachmentPolicy: "dedicated"
                    }
                };
            }
            function buildCurrentProfileBrainStatusFromReport(report, policyMode, profileId) {
                return {
                    marker: report.marker,
                    policyMode,
                    profileId
                };
            }
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
            }
            globalThis.__ocbBuildCalls = () => buildCalls;
        `
    });

    const result = describeCurrentProfileBrainStatusWithReport({
        profileId: " current_profile "
    });

    assert.equal(globalThis.__ocbBuildCalls(), 1);
    assert.equal(result.report.marker, 1);
    assert.deepEqual(result.status, {
        marker: 1,
        policyMode: "dedicated",
        profileId: "current_profile"
    });

    delete globalThis.__ocbBuildCalls;
});

test("current profile status without report uses the internal summary detail marker", () => {
    const describeCurrentProfileBrainStatus = loadFunction({
        file: "index.js",
        startMarker: "export function describeCurrentProfileBrainStatus(input)",
        endMarker: "export function formatOperatorRollbackReport",
        prelude: `
            const OPERATOR_SURFACE_DETAIL_LEVEL = Symbol.for("@openclawbrain/cli/operatorSurfaceDetailLevel");
            function withOperatorSurfaceReportDetailLevel(input, detailLevel) {
                const taggedInput = { ...input };
                Object.defineProperty(taggedInput, OPERATOR_SURFACE_DETAIL_LEVEL, {
                    value: detailLevel,
                    enumerable: false,
                    configurable: true
                });
                return taggedInput;
            }
            let seenInput = null;
            function describeCurrentProfileBrainStatusWithReport(input) {
                seenInput = input;
                return {
                    status: {
                        marker: "summary",
                        input
                    }
                };
            }
            globalThis.__ocbSummaryInput = () => seenInput;
            globalThis.__ocbSummaryDetailKey = OPERATOR_SURFACE_DETAIL_LEVEL;
        `
    });

    const input = {
        activationRoot: "/tmp/activation"
    };
    const result = describeCurrentProfileBrainStatus(input);
    const summaryInput = globalThis.__ocbSummaryInput();
    const detailKey = globalThis.__ocbSummaryDetailKey;

    assert.equal(result.marker, "summary");
    assert.equal(result.input.activationRoot, "/tmp/activation");
    assert.equal(input[detailKey], undefined);
    assert.equal(summaryInput.activationRoot, "/tmp/activation");
    assert.equal(summaryInput[detailKey], "summary");
    assert.deepEqual(Object.keys(summaryInput), ["activationRoot"]);
    assert.notEqual(summaryInput, input);

    delete globalThis.__ocbSummaryInput;
    delete globalThis.__ocbSummaryDetailKey;
});

test("summary operator status report skips detailed-only active-pack and principal surfaces", () => {
    const buildOperatorSurfaceReport = loadFunction({
        file: "index.js",
        startMarker: "export function buildOperatorSurfaceReport",
        endMarker: "export function describeCurrentProfileBrainStatusWithReport",
        prelude: `
            const OPERATOR_SURFACE_DETAIL_LEVEL = Symbol.for("@openclawbrain/cli/operatorSurfaceDetailLevel");
            const path = {
                resolve(value) {
                    return value;
                }
            };
            function normalizeNonEmptyString(value) {
                return value;
            }
            function normalizeIsoTimestamp(value, _fieldName, fallbackValue) {
                return value ?? fallbackValue;
            }
            function normalizeBrainAttachmentPolicy(value) {
                return value ?? "dedicated";
            }
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
            }
            function resolveOperatorTeacherSnapshotPath() {
                return null;
            }
            function loadTeacherSurface() {
                return null;
            }
            function loadOperatorEventExport() {
                return null;
            }
            function inspectActivationState() {
                return {
                    active: {
                        slot: "active",
                        packId: "pack-1",
                        activationReady: true,
                        routePolicy: "optional",
                        routerIdentity: "pack-1:route_fn",
                        workspaceSnapshot: null,
                        workspaceRevision: null,
                        eventRange: { count: 0, end: null },
                        eventExportDigest: null,
                        builtAt: "2026-04-04T14:00:00.000Z",
                        findings: []
                    },
                    candidate: null,
                    previous: null,
                    pointers: {
                        active: { updatedAt: "2026-04-04T14:00:00.000Z" },
                        candidate: null,
                        previous: null
                    },
                    promotion: {
                        allowed: false,
                        findings: []
                    },
                    rollback: {
                        allowed: false,
                        findings: []
                    }
                };
            }
            function summarizeOperatorSlot(slot, updatedAt) {
                return {
                    ...slot,
                    updatedAt
                };
            }
            function describeActivationObservability() {
                return {
                    promotionFreshness: {
                        candidateAheadBy: null
                    },
                    initHandoff: {
                        handoffState: "seed_state_authoritative",
                        initMode: "fast_boot_defaults",
                        seedStateVisible: true
                    },
                    learnedRouteFn: {
                        required: false,
                        available: false,
                        routerIdentity: "pack-1:route_fn",
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
                        collectedLabels: null,
                        freshnessChecksum: null
                    },
                    graphEvolutionLog: null,
                    graphDynamics: {
                        runtimePlasticitySource: "teacher_snapshot"
                    }
                };
            }
            function toErrorMessage(error) {
                return error instanceof Error ? error.message : String(error);
            }
            function summarizeOperatorActivationState() {
                return {
                    state: "healthy_seed",
                    detail: "ok"
                };
            }
            let activePackObservabilityCalls = 0;
            function buildMissingLabelFlowSummary(detail) {
                return {
                    source: "missing",
                    detail
                };
            }
            function buildMissingLearningPathSummary(detail) {
                return {
                    available: false,
                    source: "missing",
                    detail
                };
            }
            function buildSummaryOnlyPrincipalObservability(active) {
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
                        note: "summary-only principal surface"
                    },
                    servingDownstreamOfLatestCorrection: null,
                    detail: "summary-only principal surface"
                };
            }
            function summarizeActivePackObservability() {
                activePackObservabilityCalls += 1;
                return {
                    labelFlow: {
                        source: "active_pack",
                        detail: "active-pack label flow"
                    },
                    learningPath: {
                        available: true,
                        source: "active_pack",
                        detail: "active-pack learning path"
                    }
                };
            }
            function probeOperatorServePath() {
                return {
                    state: "serving_active_pack",
                    fallbackToStaticContext: false,
                    hardRequirementViolated: false,
                    activePackId: "pack-1",
                    usedLearnedRouteFn: false,
                    routerIdentity: "pack-1:route_fn",
                    selectionMode: null,
                    selectionDigest: null,
                    requestedBudgetStrategy: null,
                    resolvedBudgetStrategy: null,
                    resolvedMaxContextBlocks: null,
                    structuralBudgetSource: null,
                    structuralBudgetEvidence: null,
                    structuralBudgetPressures: null,
                    contextAttribution: {
                        selectedContextCount: 0,
                        stableKernelBlockCount: 0,
                        brainCompiledBlockCount: 0,
                        stableKernelSources: [],
                        brainCompiledSources: [],
                        selectionTiers: null,
                        evidence: "stable_kernel_only",
                        detail: "kernel only"
                    },
                    structuralDecision: {
                        origin: "static",
                        basis: "none",
                        detail: "static"
                    },
                    timing: {
                        totalMs: null,
                        routeSelectionMs: null,
                        promptAssemblyMs: null,
                        otherMs: null,
                        backgroundWorkIncluded: false
                    },
                    error: null,
                    refreshStatus: "updated",
                    freshnessChecksum: null
                };
            }
            function summarizeLearnedRoutingWithoutObservability(active) {
                return {
                    required: false,
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
                    handoffState: "seed_state_authoritative",
                    initMode: "fast_boot_defaults",
                    seedStateVisible: true
                };
            }
            function summarizeRouteFnFreshness() {
                return {
                    available: false,
                    trainedAt: null,
                    updatedAt: null,
                    usedAt: null,
                    fallbackReason: null,
                    detail: "route_fn not visible"
                };
            }
            function summarizeTeacherLoop() {
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
                    watch: null,
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
                    observationBinding: {},
                    lastObservedDelta: {
                        explanation: "none"
                    },
                    notes: [],
                    detail: "teacher snapshot unavailable"
                };
            }
            function summarizeOperatorHook() {
                return {
                    scope: "exact_openclaw_home",
                    openclawHome: null,
                    extensionDir: null,
                    hookPath: null,
                    runtimeGuardPath: null,
                    manifestPath: null,
                    packageJsonPath: null,
                    manifestId: null,
                    installId: null,
                    packageName: null,
                    packageVersion: null,
                    installLayout: null,
                    additionalInstallCount: 0,
                    installState: "installed",
                    loadability: "loadable",
                    loadProof: "status_probe_ready",
                    guardSeverity: "none",
                    guardActionability: "none",
                    guardSummary: "ok",
                    guardAction: "none",
                    desynced: false,
                    detail: "ok"
                };
            }
            function summarizeOperatorAttachmentTruth() {
                return {
                    state: "attached",
                    activationRoot: "/tmp/activation",
                    servingSlot: "active",
                    proofState: "self_proving",
                    watchOnly: false,
                    detail: "attached"
                };
            }
            function summarizeBrainState(active) {
                return {
                    state: "seed_state_authoritative",
                    initMode: "fast_boot_defaults",
                    runtimePlasticitySource: "teacher_snapshot",
                    seedStateVisible: true,
                    seedBlockCount: 1,
                    activePackId: active.packId,
                    activeWorkspaceSnapshot: null,
                    activeEventExportDigest: null,
                    detail: "brain"
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
                    activeWorkspaceSnapshot: null,
                    activeEventExportDigest: null,
                    detail: activation.detail
                };
            }
            function summarizeGraphObservability() {
                return {
                    available: true,
                    runtimePlasticitySource: "teacher_snapshot",
                    structuralOps: null,
                    connectDiagnostics: null,
                    changed: false,
                    blockCount: 7,
                    operationsApplied: [],
                    liveBlockCount: 7,
                    prunedBlockCount: 0,
                    prePruneBlockCount: 7,
                    strongestBlockId: "block-1",
                    operatorSummary: "stable",
                    latestMaterialization: {
                        known: true,
                        packId: "pack-1",
                        changed: false,
                        connectDiagnostics: null,
                        operatorSummary: "stable",
                        detail: "latest"
                    },
                    detail: "graph"
                };
            }
            function summarizeGraphWithoutObservability() {
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
                    latestMaterialization: {
                        known: false,
                        packId: null,
                        changed: null,
                        connectDiagnostics: null,
                        operatorSummary: null,
                        detail: "none"
                    },
                    detail: "graph"
                };
            }
            function summarizeLastPromotion() {
                return {
                    known: true,
                    at: "2026-04-04T14:00:00.000Z",
                    note: "promoted"
                };
            }
            function summarizeCandidateAheadBy() {
                return [];
            }
            function summarizeSupervision() {
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
                    detail: "no event export"
                };
            }
            function summarizeAlwaysOnLearning() {
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
                    principalLagToPromotion: {
                        activeEventRangeEnd: null,
                        latestPrincipalSequence: null,
                        sequenceLag: null,
                        status: "unavailable"
                    },
                    warningStates: ["teacher_snapshot_unavailable"],
                    learnedRange: null,
                    materializationCount: null,
                    lastMaterializedAt: null,
                    lastMaterializationReason: null,
                    lastMaterializationLane: null,
                    lastMaterializationPriority: null,
                    lastMaterializedPackId: null,
                    detail: "missing"
                };
            }
            let principalObservabilityCalls = 0;
            function summarizePrincipalObservability() {
                principalObservabilityCalls += 1;
                return {
                    available: true,
                    sourcePath: "/tmp/export",
                    sourceKind: "bundle_root",
                    latestFeedback: null,
                    latestCorrection: null,
                    pendingCount: 0,
                    pendingItems: [],
                    latestPromotion: {
                        known: false,
                        at: null,
                        activePackId: "pack-1",
                        activeEventRangeEnd: null,
                        includesLatestFeedback: null,
                        includesLatestCorrection: null,
                        note: "principal"
                    },
                    servingDownstreamOfLatestCorrection: null,
                    detail: "principal"
                };
            }
            function summarizeManyProfileSupport(policyMode) {
                return {
                    declaredAttachmentPolicy: policyMode
                };
            }
            function buildOperatorFindings() {
                return [];
            }
            function summarizeOperatorStatus() {
                return "ok";
            }
            globalThis.__statusSummaryCounters = {
                get() {
                    return {
                        activePackObservabilityCalls,
                        principalObservabilityCalls
                    };
                },
                reset() {
                    activePackObservabilityCalls = 0;
                    principalObservabilityCalls = 0;
                }
            };
        `
    });

    globalThis.__statusSummaryCounters.reset();
    const summaryInput = {
        activationRoot: "/tmp/activation"
    };
    Object.defineProperty(summaryInput, Symbol.for("@openclawbrain/cli/operatorSurfaceDetailLevel"), {
        value: "summary",
        enumerable: false,
        configurable: true
    });
    const summaryReport = buildOperatorSurfaceReport(summaryInput);

    assert.deepEqual(globalThis.__statusSummaryCounters.get(), {
        activePackObservabilityCalls: 0,
        principalObservabilityCalls: 0
    });
    assert.equal(summaryReport.labelFlow.source, "missing");
    assert.equal(summaryReport.learningPath.available, false);
    assert.equal(summaryReport.principal.available, false);

    globalThis.__statusSummaryCounters.reset();
    const detailedReport = buildOperatorSurfaceReport({
        activationRoot: "/tmp/activation"
    });

    assert.deepEqual(globalThis.__statusSummaryCounters.get(), {
        activePackObservabilityCalls: 1,
        principalObservabilityCalls: 1
    });
    assert.equal(detailedReport.labelFlow.source, "active_pack");
    assert.equal(detailedReport.learningPath.available, true);
    assert.equal(detailedReport.principal.available, true);

    delete globalThis.__statusSummaryCounters;
});

test("current profile status preserves hook packageVersion for hotfix-boundary reporting", () => {
    const buildCurrentProfileBrainStatusFromReport = loadFunction({
        file: "index.js",
        startMarker: "function buildCurrentProfileBrainStatusFromReport",
        endMarker: "export function buildOperatorSurfaceReport",
        prelude: `
            const CURRENT_PROFILE_BRAIN_STATUS_CONTRACT = "current_profile_brain_status.v1";
            function isAwaitingFirstExportSlot() { return false; }
            function summarizeCurrentProfilePassiveLearning() {
                return { learnerRunning: true, exportState: "latest_export_visible", backlogState: "caught_up", pendingLive: 0, pendingBackfill: 0, firstExportOccurred: true, currentServingPackId: "pack-123", lastObservedDelta: { explanation: "none" } };
            }
            function summarizeCurrentProfileBrainStatusLevel() { return "ok"; }
            function summarizeCurrentProfileLogRoot(activationRoot) { return activationRoot + "/logs"; }
            function summarizeCurrentProfileLastLearningUpdateAt() { return "2026-04-04T14:00:00.000Z"; }
            function summarizeCurrentProfileBrainSummary() { return "summary"; }
            function buildCurrentProfileAttachmentPolicy(policyMode) { return { mode: policyMode }; }
            function buildCurrentProfileTurnAttributionFromReport() { return { source: "stub" }; }
        `
    });

    const result = buildCurrentProfileBrainStatusFromReport({
        generatedAt: "2026-04-04T14:00:00.000Z",
        activationRoot: "/tmp/activation",
        activation: { state: "active", detail: "ok" },
        brain: { activePackId: "pack-123", state: "pg_promoted_pack_authoritative", detail: "ok" },
        servePath: {
            routerIdentity: "pack-123:route_fn",
            refreshStatus: "updated",
            activePackId: "pack-123",
            state: "serving_active_pack",
            usedLearnedRouteFn: true,
            fallbackToStaticContext: false,
            structuralDecision: null,
            timing: null
        },
        learnedRouting: {
            routerIdentity: "pack-123:route_fn",
            initMode: "fast_boot_defaults",
            routerChecksum: "sha256-test"
        },
        supervision: { exportedAt: "2026-04-04T14:00:00.000Z" },
        promotion: { lastPromotion: { at: "2026-04-04T14:00:00.000Z" } },
        learning: {},
        teacherLoop: { observationBinding: {} },
        hook: {
            scope: "exact_openclaw_home",
            openclawHome: "/tmp/.openclaw",
            extensionDir: "/tmp/.openclaw/extensions/openclawbrain",
            hookPath: "/tmp/.openclaw/extensions/openclawbrain/dist/extension/index.js",
            runtimeGuardPath: "/tmp/.openclaw/extensions/openclawbrain/dist/extension/runtime-guard.js",
            manifestPath: "/tmp/.openclaw/extensions/openclawbrain/openclaw.plugin.json",
            packageJsonPath: "/tmp/.openclaw/extensions/openclawbrain/package.json",
            manifestId: "openclawbrain",
            installId: "openclaw",
            packageName: "@openclawbrain/openclaw",
            packageVersion: "0.4.30",
            installLayout: "native_package_plugin",
            additionalInstallCount: 0,
            installState: "installed",
            loadability: "loadable",
            loadProof: "status_probe_ready",
            guardSeverity: "none",
            guardActionability: "none",
            guardSummary: "profile hook is installed and loadable",
            guardAction: "none",
            desynced: false,
            detail: "ok"
        },
        attachmentTruth: {
            state: "attached",
            activationRoot: "/tmp/activation",
            servingSlot: "active",
            proofState: "proven",
            watchOnly: false,
            detail: "attached"
        },
        active: { packId: "pack-123", routerIdentity: "pack-123:route_fn" }
    }, "dedicated", "current_profile");

    assert.equal(result.hook.packageName, "@openclawbrain/openclaw");
    assert.equal(result.hook.packageVersion, "0.4.30");
});
