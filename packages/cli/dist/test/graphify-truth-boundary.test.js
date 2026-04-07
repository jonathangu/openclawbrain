import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { buildGraphifyCompiledArtifactPack, writeGraphifyCompiledArtifactPack } from "../src/graphify-compiled-artifacts.js";
import { exportGraphifyImportSlice } from "../src/import-export.js";

function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-boundary-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}

test("graphify import slice keeps the truth boundary explicit and rollback-safe", (t) => {
    const tempRoot = createTempRoot(t);
    const packRoot = path.join(tempRoot, "graphify-pack");
    const outputRoot = path.join(tempRoot, "artifacts", "graphify-imports");
    const bundle = buildGraphifyCompiledArtifactPack({
        bundleId: "boundary-fixture",
        bundleStartedAt: "2026-04-06T22:40:00.000Z",
        outputDir: packRoot,
        graphifyRunId: "graphify-run-boundary-fixture",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "compiled-artifacts-target-state-scaffold",
        sourceBundleHash: "sha256:source-bundle-hash",
        graphHash: "sha256:graph-hash",
        configHash: "sha256:config-hash",
        labelsHash: "sha256:labels-hash",
    });
    writeGraphifyCompiledArtifactPack(bundle.outputDir, bundle);

    const result = exportGraphifyImportSlice({
        bundleRoot: bundle.outputDir,
        outputRoot,
        runId: "boundary-run",
    });

    assert.equal(result.ok, true);

    const slice = JSON.parse(readFileSync(result.paths.importSlice, "utf8"));
    const candidatePackInput = JSON.parse(readFileSync(result.paths.candidatePackInput, "utf8"));
    const envelope = JSON.parse(readFileSync(result.paths.proposalEnvelope, "utf8"));
    const replayGate = JSON.parse(readFileSync(result.paths.replayGate, "utf8"));

    assert.equal(slice.truthBoundary.artifactFirst, true);
    assert.equal(slice.truthBoundary.rollbackSafe, true);
    assert.equal(slice.truthBoundary.removable, true);
    assert.equal(slice.truthBoundary.liveEligible, false);
    assert.equal(candidatePackInput.seedingBoundary.removable, true);
    assert.equal(candidatePackInput.seedingBoundary.rollbackSafe, true);
    assert.equal(candidatePackInput.seedingBoundary.liveEligible, false);
    assert.equal(candidatePackInput.seedingBoundary.currentTruthWrites, false);
    assert.equal(candidatePackInput.seedingBoundary.correctionMemoryWrites, false);
    assert.equal(candidatePackInput.seedingBoundary.hotPathDependency, false);
    assert.equal(candidatePackInput.targetStateOnly, true);
    assert.deepEqual(slice.truthBoundary.blockedTrustClasses, ["INFERRED", "AMBIGUOUS"]);
    assert.deepEqual(replayGate.blockedTrustClasses, ["INFERRED", "AMBIGUOUS"]);
    assert.ok(replayGate.requirements.some((entry) => entry.id === "artifact-first"));
    assert.ok(replayGate.requirements.some((entry) => entry.id === "extracted-only"));
    assert.ok(replayGate.requirements.some((entry) => entry.id === "correction-precedence"));
    assert.ok(replayGate.requirements.some((entry) => entry.id === "rollback-safe"));
    assert.ok(replayGate.requirements.some((entry) => entry.id === "boundedness"));
    assert.ok(replayGate.blockedEffects.includes("current_truth_write"));
    assert.ok(replayGate.blockedEffects.includes("correction_like_memory"));
    assert.ok(replayGate.blockedEffects.includes("live_eligible_edge"));
    assert.ok(replayGate.blockedEffects.includes("hot_path_serve_integration"));
    assert.equal(envelope.targetStateOnly, true);
    assert.equal(envelope.reviewMode, "candidate_only");
    assert.ok(Array.isArray(envelope.strongerTruthAnchors));
    assert.ok(envelope.strongerTruthAnchors.some((anchor) => anchor.id === "graphify-bridge-artifact-first"));
    assert.ok(envelope.strongerTruthAnchors.some((anchor) => anchor.id === "graphify-bridge-extracted-only"));
    assert.ok(envelope.strongerTruthAnchors.some((anchor) => anchor.id === "graphify-bridge-rollback-discipline"));
    assert.match(JSON.stringify(slice), /EXTRACTED/);
    assert.doesNotMatch(JSON.stringify(slice), /live truth write/i);
});
