import test from "node:test";
import assert from "node:assert/strict";
import { formatOperatorLearningPathSummary, formatOperatorRetrainLineageSummary } from "../src/status-learning-path.js";

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
            brainStatus: { awaitingFirstExport: true }
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
    assert.doesNotMatch(summary, /pg=v1/);
    assert.doesNotMatch(summary, /method=policy_gradient_v1/);
});

test("retrain lineage summary surfaces prior-rooted promotion truth", () => {
    const summary = formatOperatorRetrainLineageSummary({
        tracedLearning: {
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
        }
    });
    assert.equal(summary, "status=visible prior=router-base-prior-v0@v0 seedChecksum=sha256:prior candidate=router-artifact-periodic-retrain-v1@v1 routerChecksum=sha256:candidate priorRooted=yes promotionValid=yes residualUpdates=7");
});

test("non-seed states preserve the raw learning-path summary", () => {
    const summary = formatOperatorLearningPathSummary({
        status: {
            brain: { state: "promoted_pack_active" },
            brainStatus: { awaitingFirstExport: false }
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
    assert.equal(summary, "source=materialized_candidate pg=v2 method=policy_gradient_v2 target=trajectory_reconstruction connect=4 trajectories=12");
});
