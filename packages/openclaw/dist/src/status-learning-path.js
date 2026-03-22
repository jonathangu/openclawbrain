function formatRawLearningPathSummary(learningPath) {
    return `source=${learningPath.source} pg=${learningPath.policyGradientVersion} method=${learningPath.policyGradientMethod ?? "none"} target=${learningPath.targetConstruction ?? "none"} connect=${learningPath.connectOpsFired ?? "none"} trajectories=${learningPath.reconstructedTrajectoryCount ?? "none"}`;
}
function isSeedAwaitingFirstPromotion(status) {
    return status?.brain?.state === "seed_state_authoritative" && status?.brainStatus?.awaitingFirstExport === true;
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
export function formatOperatorLearningPathSummary({ status, learningPath, tracedLearning }) {
    if (!isSeedAwaitingFirstPromotion(status)) {
        return formatRawLearningPathSummary(learningPath);
    }
    const detailParts = [
        "detail=seed_state_awaiting_first_promotion",
        `tracedPg=${normalizeOptionalString(tracedLearning?.pgVersionUsed) ?? "none"}`,
        `tracedPack=${normalizeOptionalString(tracedLearning?.materializedPackId) ?? "none"}`
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
