function formatRawLearningPathSummary(learningPath) {
    return `source=${learningPath.source} pg=${learningPath.policyGradientVersion} method=${learningPath.policyGradientMethod ?? "none"} target=${learningPath.targetConstruction ?? "none"} connect=${learningPath.connectOpsFired ?? "none"} trajectories=${learningPath.reconstructedTrajectoryCount ?? "none"}`;
}
function isSeedAwaitingFirstPromotion(status) {
    return status?.brain?.state === "seed_state_authoritative" && status?.brainStatus?.awaitingFirstExport === true;
}
function normalizeOptionalString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value : null;
}
function formatOptionalBoolean(value) {
    return typeof value === "boolean" ? (value ? "yes" : "no") : "unknown";
}
function formatOptionalCount(value) {
    return Number.isFinite(value) && value >= 0 ? String(Math.trunc(value)) : "unknown";
}
function formatArtifactLabel(id, version) {
    const artifactId = normalizeOptionalString(id);
    const artifactVersion = normalizeOptionalString(version);
    if (artifactId === null) {
        return "none";
    }
    return artifactVersion === null ? artifactId : `${artifactId}@${artifactVersion}`;
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
export function formatOperatorRetrainLineageSummary({ tracedLearning }) {
    const lineage = tracedLearning?.retrainLineage ?? null;
    if (lineage === null || typeof lineage !== "object") {
        return "status=unknown detail=retrain_lineage_not_visible";
    }
    return [
        "status=visible",
        `prior=${formatArtifactLabel(lineage.priorBaseArtifactId, lineage.priorBaseArtifactVersion)}`,
        `seedChecksum=${normalizeOptionalString(lineage.priorBaseArtifactChecksum) ?? "none"}`,
        `candidate=${formatArtifactLabel(lineage.candidateArtifactId, lineage.candidateArtifactVersion)}`,
        `routerChecksum=${normalizeOptionalString(lineage.candidateArtifactChecksum) ?? "none"}`,
        `priorRooted=${formatOptionalBoolean(lineage.priorRooted)}`,
        `promotionValid=${formatOptionalBoolean(lineage.promotionValid)}`,
        `residualUpdates=${formatOptionalCount(lineage.residualUpdateCount)}`
    ].join(" ");
}
