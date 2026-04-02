function formatRawLearningPathSummary(learningPath) {
    return `source=${learningPath.source} pg=${learningPath.policyGradientVersion} method=${learningPath.policyGradientMethod ?? "none"} target=${learningPath.targetConstruction ?? "none"} connect=${learningPath.connectOpsFired ?? "none"} trajectories=${learningPath.reconstructedTrajectoryCount ?? "none"}`;
}
function isSeedAwaitingFirstPromotion(status) {
    return status?.brain?.state === "seed_state_authoritative" && status?.brainStatus?.awaitingFirstExport === true;
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
function formatOptionalFeedbackLatest(tracedLearning) {
    const latestLabel = normalizeOptionalString(tracedLearning?.feedbackSummary?.latestLabel);
    return latestLabel === null ? "" : ` latest=${latestLabel}`;
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
export { formatOperatorAttributionCoverageSummary, formatOperatorFeedbackSummary, formatOperatorLearningAttributionSummary };
