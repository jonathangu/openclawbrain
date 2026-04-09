import test from "node:test";
import assert from "node:assert/strict";
import { formatOperatorAttributionCoverageSummary, formatOperatorFeedbackSummary, formatOperatorLearningAttributionSummary, formatOperatorLearningFlowSummary, formatOperatorLearningHealthSummary, formatOperatorLearningPathSummary, formatOperatorRetrainLineageSummary } from "../src/status-learning-path.js";

const baseLearningPath = {
    source: "active_pack",
    policyGradientVersion: "v1",
    policyGradientMethod: "policy_gradient_v1",
    targetConstruction: "event_block_plus_related_interaction",
    connectOpsFired: 0,
    reconstructedTrajectoryCount: 0
};

test("seed state awaiting first promotion hides mixed v1 metadata on the path line", () => {
    const summary = formatOperatorLearningPathSummary({
        status: {
            brain: { state: "seed_state_authoritative" },
            brainStatus: { awaitingFirstExport: true },
            learningAttribution: {
                quality: "exact_only"
            }
        },
        learningPath: baseLearningPath,
        tracedLearning: {
            pgVersionUsed: "v2",
            materializedPackId: "pack-63dc0caf"
        }
    });
    assert.match(summary, /source=seed_state/);
    assert.match(summary, /pg=seed/);
    assert.match(summary, /method=not_yet_promoted/);
    assert.match(summary, /detail=seed_state_awaiting_first_promotion/);
    assert.match(summary, /tracedPg=v2/);
    assert.match(summary, /bindingQuality=exact_only/);
    assert.doesNotMatch(summary, /pg=v1/);
    assert.doesNotMatch(summary, /method=policy_gradient_v1/);
});

test("non-seed states preserve the raw learning-path summary", () => {
    const summary = formatOperatorLearningPathSummary({
        status: {
            brain: { state: "promoted_pack_active" },
            brainStatus: { awaitingFirstExport: false },
            learningAttribution: {
                quality: "exact_with_unmatched"
            }
        },
        learningPath: {
            ...baseLearningPath,
            source: "materialized_candidate",
            policyGradientVersion: "v2",
            policyGradientMethod: "policy_gradient_v2",
            targetConstruction: "trajectory_reconstruction",
            connectOpsFired: 4,
            reconstructedTrajectoryCount: 12
        },
        tracedLearning: {
            pgVersionUsed: "v2",
            materializedPackId: "pack-promoted"
        }
    });
    assert.equal(summary, "source=materialized_candidate pg=v2 method=policy_gradient_v2 target=trajectory_reconstruction connect=4 trajectories=12 bindingQuality=exact_with_unmatched");
});

test("attribution summary reports exact-vs-heuristic and unresolved counters", () => {
    const summary = formatOperatorLearningAttributionSummary({
        status: {
            learningAttribution: {
                available: true,
                source: "latest_materialization",
                snapshotKind: "watch_snapshot",
                quality: "exact_with_unmatched",
                nonZeroObservationCount: 3,
                exactMatchCount: 2,
                heuristicMatchCount: 1,
                unmatchedCount: 1,
                ambiguousCount: 0,
                matchedByMode: {
                    exactDecisionId: 1,
                    exactSelectionDigest: 1,
                    turnCompileEventId: 0,
                    legacyHeuristic: 1
                }
            }
        }
    });
    assert.equal(summary, "quality=exact_with_unmatched source=latest_materialization/watch_snapshot nonZero=3 exact=2 heuristic=1 unmatched=1 ambiguous=0 modes=decision:1|digest:1|compile:0|heuristic:1");
});

test("feedback and attribution coverage summaries stay thin and conservative", () => {
    const tracedLearning = {
        routeTraceCount: 3,
        supervisionCount: 2,
        feedbackSummary: {
            helpfulCount: 1,
            irrelevantCount: 1,
            harmfulCount: 0,
            supervisedTraceCount: 2,
            routeTraceCount: 3,
            latestLabel: "main:subagent"
        },
        attributionCoverage: {
            completedWithoutEvaluationCount: 1,
            readyCount: 2,
            delayedCount: 1,
            budgetDeferredCount: 1
        }
    };
    assert.equal(formatOperatorFeedbackSummary({ tracedLearning }), "helpful=1 irrelevant=1 harmful=0 supervisedTraceCount=2 routeTraceCount=3 latest=main:subagent");
    assert.equal(formatOperatorAttributionCoverageSummary({ tracedLearning }), "completedWithoutEvaluation=1 ready=2 delayed=1 budgetDeferred=1");
});

test("retrain lineage summary surfaces prior-rooted promotion truth without flattening it into learn-flow", () => {
    const tracedLearning = {
        retrainLineage: {
            priorBaseArtifactId: "router-base-prior-v0",
            priorBaseArtifactVersion: "v0",
            priorBaseArtifactChecksum: "sha256:prior",
            candidateArtifactId: "router-artifact-periodic-retrain-v1",
            candidateArtifactVersion: "v1",
            candidateArtifactChecksum: "sha256:candidate",
            priorRooted: true,
            promotionValid: true,
            residualUpdateCount: 7
        }
    };
    assert.equal(formatOperatorRetrainLineageSummary({ tracedLearning }), "status=visible prior=router-base-prior-v0@v0 seedChecksum=sha256:prior candidate=router-artifact-periodic-retrain-v1@v1 routerChecksum=sha256:candidate priorRooted=yes promotionValid=yes residualUpdates=7");
});

test("learning flow summary separates harvested through updated stages without flattening them", () => {
    const tracedLearning = {
        present: true,
        teacherArtifactCount: 4,
        materializedPackId: "pack-status",
        routeTraceCount: 7,
        supervisionCount: 2,
        routerUpdateCount: 3,
        attributionCoverage: {
            visible: true,
            readyCount: 2,
            delayedCount: 1,
            budgetDeferredCount: 1
        }
    };
    assert.equal(formatOperatorLearningFlowSummary({ tracedLearning }), "harvested=4 eligible=2 selected=7 budgetDeferred=1 supervised=2 updated=3 materialized=pack-status");
});

test("learning health summary calls out budget-deferred learning before calling it stalled", () => {
    const tracedLearning = {
        present: true,
        teacherArtifactCount: 4,
        materializedPackId: null,
        routeTraceCount: 0,
        supervisionCount: 0,
        routerUpdateCount: 0,
        attributionCoverage: {
            visible: true,
            readyCount: 2,
            delayedCount: 1,
            budgetDeferredCount: 1
        }
    };
    const teacher = {
        enabled: true,
        healthy: true
    };
    assert.equal(formatOperatorLearningHealthSummary({ tracedLearning, teacher }), "daemon=healthy-daemon learning=budget-deferred-learning detail=harvested artifacts and eligible feedback are visible, but ready items are budget-deferred before any selected routes, supervision, or router updates are visible");
});

test("learning health summary distinguishes a healthy daemon from stalled learning", () => {
    const tracedLearning = {
        present: true,
        teacherArtifactCount: 4,
        materializedPackId: null,
        routeTraceCount: 0,
        supervisionCount: 0,
        routerUpdateCount: 0,
        attributionCoverage: {
            visible: true,
            readyCount: 2,
            budgetDeferredCount: 0
        }
    };
    const teacher = {
        enabled: true,
        healthy: true
    };
    assert.equal(formatOperatorLearningHealthSummary({ tracedLearning, teacher }), "daemon=healthy-daemon learning=stalled-learning detail=harvested artifacts and eligible feedback are visible, but no selected routes, supervision, or router updates are visible");
});
