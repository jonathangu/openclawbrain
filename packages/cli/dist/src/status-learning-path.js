function formatRawLearningPathSummary(learningPath) {
    return `source=${learningPath.source} pg=${learningPath.policyGradientVersion} method=${learningPath.policyGradientMethod ?? "none"} target=${learningPath.targetConstruction ?? "none"} connect=${learningPath.connectOpsFired ?? "none"} trajectories=${learningPath.reconstructedTrajectoryCount ?? "none"}`;
}
function isSeedAwaitingFirstPromotion(status) {
    return status?.brain?.state === "seed_state_authoritative" && status?.brainStatus?.awaitingFirstExport === true;
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
function normalizeCount(value) {
    return Number.isFinite(value) && value >= 0 ? Math.trunc(value) : 0;
}
function formatOptionalFeedbackLatest(tracedLearning) {
    const latestLabel = normalizeOptionalString(tracedLearning?.feedbackSummary?.latestLabel);
    return latestLabel === null ? "" : ` latest=${latestLabel}`;
}
function isLearningSurfaceVisible(tracedLearning) {
    return tracedLearning?.present !== false;
}
function hasKnownAttributionCoverage(coverage) {
    return coverage !== null &&
        typeof coverage === "object" &&
        !Array.isArray(coverage) &&
        (coverage.visible === true ||
            coverage.gatingVisible === true ||
            Number.isFinite(coverage.completedWithoutEvaluationCount) ||
            Number.isFinite(coverage.readyCount) ||
            Number.isFinite(coverage.delayedCount) ||
            Number.isFinite(coverage.budgetDeferredCount));
}
function readSelectedTraceCount(tracedLearning) {
    return normalizeCount(tracedLearning?.routeTraceCount);
}
function readBudgetDeferredCount(tracedLearning) {
    return hasKnownAttributionCoverage(tracedLearning?.attributionCoverage)
        ? normalizeCount(tracedLearning?.attributionCoverage?.budgetDeferredCount)
        : null;
}
function readMaterializedPackId(tracedLearning) {
    return normalizeOptionalString(tracedLearning?.materializedPackId);
}
function summarizeOperatorLearningFlow(tracedLearning) {
    const surfaceVisible = isLearningSurfaceVisible(tracedLearning);
    return {
        harvested: surfaceVisible ? normalizeCount(tracedLearning?.teacherArtifactCount) : null,
        eligible: hasKnownAttributionCoverage(tracedLearning?.attributionCoverage)
            ? normalizeCount(tracedLearning?.attributionCoverage?.readyCount)
            : null,
        selected: surfaceVisible ? readSelectedTraceCount(tracedLearning) : null,
        budgetDeferred: readBudgetDeferredCount(tracedLearning),
        supervised: surfaceVisible ? normalizeCount(tracedLearning?.supervisionCount) : null,
        updated: surfaceVisible ? normalizeCount(tracedLearning?.routerUpdateCount) : null,
        materialized: surfaceVisible ? readMaterializedPackId(tracedLearning) : null
    };
}
function formatKnownOperatorValue(value) {
    return value === null ? "unknown" : String(value);
}
function summarizeOperatorLearningState(flow) {
    if (flow.harvested === null &&
        flow.eligible === null &&
        flow.selected === null &&
        flow.budgetDeferred === null &&
        flow.supervised === null &&
        flow.updated === null &&
        flow.materialized === null) {
        return {
            state: "learning-unknown",
            detail: "learning stage truth is not visible in the current status surface"
        };
    }
    if ((flow.selected ?? 0) > 0 ||
        (flow.supervised ?? 0) > 0 ||
        (flow.updated ?? 0) > 0 ||
        flow.materialized !== null) {
        return {
            state: "progress-visible",
            detail: `selected=${flow.selected ?? 0} budgetDeferred=${flow.budgetDeferred ?? 0} supervised=${flow.supervised ?? 0} updated=${flow.updated ?? 0} materialized=${flow.materialized ?? "none"}`
        };
    }
    if (flow.harvested > 0 &&
        flow.eligible !== null &&
        flow.eligible > 0 &&
        (flow.budgetDeferred ?? 0) > 0 &&
        (flow.selected ?? 0) === 0 &&
        (flow.supervised ?? 0) === 0 &&
        (flow.updated ?? 0) === 0) {
        return {
            state: "budget-deferred-learning",
            detail: "harvested artifacts and eligible feedback are visible, but ready items are budget-deferred before any selected routes, supervision, or router updates are visible"
        };
    }
    if (flow.harvested > 0 &&
        flow.eligible !== null &&
        flow.eligible > 0 &&
        (flow.selected ?? 0) === 0 &&
        (flow.supervised ?? 0) === 0 &&
        (flow.updated ?? 0) === 0) {
        return {
            state: "stalled-learning",
            detail: "harvested artifacts and eligible feedback are visible, but no selected routes, supervision, or router updates are visible"
        };
    }
    if (flow.harvested > 0 && flow.eligible === 0) {
        return {
            state: "harvested-not-yet-eligible",
            detail: "teacher artifacts are visible, but no eligible feedback is queued yet"
        };
    }
    if (flow.harvested > 0 && flow.eligible === null) {
        return {
            state: "harvested-eligibility-unknown",
            detail: "teacher artifacts are visible, but eligible feedback truth is not surfaced yet"
        };
    }
    if (flow.harvested === 0 && flow.eligible !== null && flow.eligible > 0) {
        return {
            state: "eligible-without-harvest",
            detail: "eligible feedback is visible without harvested teacher artifacts"
        };
    }
    if (flow.harvested === 0 && flow.eligible === 0) {
        return {
            state: "idle-no-eligible-feedback",
            detail: "no harvested teacher artifacts or eligible feedback are visible"
        };
    }
    return {
        state: "learning-unknown",
        detail: "learning stage truth is incomplete"
    };
}
function summarizeOperatorDaemonState(teacher) {
    if (teacher?.enabled !== true) {
        return "daemon-disabled";
    }
    if (teacher?.healthy === true) {
        return "healthy-daemon";
    }
    if (teacher?.healthy === false) {
        return "degraded-daemon";
    }
    return "daemon-unknown";
}
function formatOperatorFeedbackSummary({ tracedLearning }) {
    const routeTraceCount = tracedLearning?.feedbackSummary?.routeTraceCount ?? tracedLearning?.routeTraceCount ?? 0;
    const supervisedTraceCount = tracedLearning?.feedbackSummary?.supervisedTraceCount ?? tracedLearning?.supervisionCount ?? 0;
    return [
        `helpful=${tracedLearning?.feedbackSummary?.helpfulCount ?? 0}`,
        `irrelevant=${tracedLearning?.feedbackSummary?.irrelevantCount ?? 0}`,
        `harmful=${tracedLearning?.feedbackSummary?.harmfulCount ?? 0}`,
        `supervisedTraceCount=${supervisedTraceCount}`,
        `routeTraceCount=${routeTraceCount}`
    ].join(" ") + formatOptionalFeedbackLatest(tracedLearning);
}
function formatOperatorAttributionCoverageSummary({ tracedLearning }) {
    return [
        `completedWithoutEvaluation=${tracedLearning?.attributionCoverage?.completedWithoutEvaluationCount ?? 0}`,
        `ready=${tracedLearning?.attributionCoverage?.readyCount ?? 0}`,
        `delayed=${tracedLearning?.attributionCoverage?.delayedCount ?? 0}`,
        `budgetDeferred=${tracedLearning?.attributionCoverage?.budgetDeferredCount ?? 0}`
    ].join(" ");
}
function formatOperatorLearningAttributionSummary({ status }) {
    const attribution = status?.learningAttribution ?? null;
    if (!attribution) {
        return "quality=unavailable source=unavailable detail=no_learning_attribution_surface";
    }
    const source = [normalizeOptionalString(attribution.source), normalizeOptionalString(attribution.snapshotKind)]
        .filter((value) => value !== null)
        .join("/");
    if (attribution.available !== true) {
        return `quality=${normalizeOptionalString(attribution.quality) ?? "unavailable"} source=${source || "unavailable"} detail=${normalizeOptionalString(attribution.detail) ?? "unavailable"}`;
    }
    const matchedByMode = attribution.matchedByMode ?? {};
    return [
        `quality=${normalizeOptionalString(attribution.quality) ?? "unavailable"}`,
        `source=${source || "latest_materialization"}`,
        `nonZero=${attribution.nonZeroObservationCount ?? 0}`,
        `exact=${attribution.exactMatchCount ?? 0}`,
        `heuristic=${attribution.heuristicMatchCount ?? 0}`,
        `unmatched=${attribution.unmatchedCount ?? 0}`,
        `ambiguous=${attribution.ambiguousCount ?? 0}`,
        `modes=decision:${matchedByMode.exactDecisionId ?? 0}|digest:${matchedByMode.exactSelectionDigest ?? 0}|compile:${matchedByMode.turnCompileEventId ?? 0}|heuristic:${matchedByMode.legacyHeuristic ?? 0}`
    ].join(" ");
}
export function formatOperatorLearningPathSummary({ status, learningPath, tracedLearning }) {
    const attribution = status?.learningAttribution ?? null;
    if (!isSeedAwaitingFirstPromotion(status)) {
        const rawSummary = formatRawLearningPathSummary(learningPath);
        const bindingQuality = normalizeOptionalString(attribution?.quality);
        return bindingQuality === null || bindingQuality === "unavailable"
            ? rawSummary
            : `${rawSummary} bindingQuality=${bindingQuality}`;
    }
    const detailParts = [
        "detail=seed_state_awaiting_first_promotion",
        `tracedPg=${normalizeOptionalString(tracedLearning?.pgVersionUsed) ?? "none"}`,
        `tracedPack=${normalizeOptionalString(tracedLearning?.materializedPackId) ?? "none"}`,
        `bindingQuality=${normalizeOptionalString(attribution?.quality) ?? "unavailable"}`
    ];
    return [
        "source=seed_state",
        "pg=seed",
        "method=not_yet_promoted",
        "target=not_yet_promoted",
        "connect=none",
        "trajectories=none",
        ...detailParts
    ].join(" ");
}
export function formatOperatorLearningFlowSummary({ tracedLearning }) {
    const flow = summarizeOperatorLearningFlow(tracedLearning);
    return [
        `harvested=${formatKnownOperatorValue(flow.harvested)}`,
        `eligible=${formatKnownOperatorValue(flow.eligible)}`,
        `selected=${formatKnownOperatorValue(flow.selected)}`,
        `budgetDeferred=${formatKnownOperatorValue(flow.budgetDeferred)}`,
        `supervised=${formatKnownOperatorValue(flow.supervised)}`,
        `updated=${formatKnownOperatorValue(flow.updated)}`,
        `materialized=${flow.materialized ?? "none"}`
    ].join(" ");
}
export function formatOperatorLearningHealthSummary({ tracedLearning, teacher }) {
    const flow = summarizeOperatorLearningFlow(tracedLearning);
    const learning = summarizeOperatorLearningState(flow);
    return `daemon=${summarizeOperatorDaemonState(teacher)} learning=${learning.state} detail=${learning.detail}`;
}
export { formatOperatorAttributionCoverageSummary, formatOperatorFeedbackSummary, formatOperatorLearningAttributionSummary };
