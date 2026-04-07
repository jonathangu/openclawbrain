import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { buildGraphifyCompiledArtifactPack, writeGraphifyCompiledArtifactPack } from "../src/graphify-compiled-artifacts.js";
import { exportGraphifyImportSlice } from "../src/import-export.js";
import { parseOperatorCliArgs, runOperatorCli } from "../src/cli.js";

function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-import-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}

function buildFixturePack(outputDir) {
    return buildGraphifyCompiledArtifactPack({
        bundleId: "import-slice-fixture",
        bundleStartedAt: "2026-04-06T22:30:00.000Z",
        outputDir,
        graphifyRunId: "graphify-run-import-fixture",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "compiled-artifacts-target-state-scaffold",
        sourceBundleHash: "sha256:source-bundle-hash",
        graphHash: "sha256:graph-hash",
        configHash: "sha256:config-hash",
        labelsHash: "sha256:labels-hash",
    });
}

test("graphify import slice writes a bounded EXTRACTED-only slice from a compiled-artifact pack", (t) => {
    const tempRoot = createTempRoot(t);
    const packRoot = path.join(tempRoot, "graphify-pack");
    const outputRoot = path.join(tempRoot, "artifacts", "graphify-imports");
    const bundle = buildFixturePack(packRoot);
    writeGraphifyCompiledArtifactPack(bundle.outputDir, bundle);

    const result = exportGraphifyImportSlice({
        bundleRoot: bundle.outputDir,
        outputRoot,
        runId: "import-slice-smoke",
    });

    assert.equal(result.ok, true);
    assert.equal(result.runId, "import-slice-smoke");
    assert.equal(result.outputDir, path.join(outputRoot, "import-slice-smoke"));
    assert.equal(result.fileCount, 5);
    assert.ok(result.digest.bundleHash.startsWith("sha256:"));
    assert.equal(result.counts.hubPriors, 2);
    assert.equal(result.counts.neighborhoodPriors, 1);
    assert.ok(result.counts.evidencePointers > 0);
    assert.ok(result.counts.rationalePointers > 0);

    const slice = JSON.parse(readFileSync(result.paths.importSlice, "utf8"));
    const candidatePackInput = JSON.parse(readFileSync(result.paths.candidatePackInput, "utf8"));
    const report = readFileSync(result.paths.importReport, "utf8");
    const envelope = JSON.parse(readFileSync(result.paths.proposalEnvelope, "utf8"));
    const replayGate = JSON.parse(readFileSync(result.paths.replayGate, "utf8"));

    assert.equal(slice.contract, "graphify_import_slice.v1");
    assert.equal(candidatePackInput.contract, "graphify_import_slice_candidate_pack_input.v1");
    assert.equal(candidatePackInput.targetStateOnly, true);
    assert.equal(candidatePackInput.reviewMode, "candidate_only");
    assert.equal(candidatePackInput.seedingBoundary.removable, true);
    assert.equal(candidatePackInput.seedingBoundary.rollbackSafe, true);
    assert.equal(candidatePackInput.seedingBoundary.liveEligible, false);
    assert.equal(candidatePackInput.seedingBoundary.currentTruthWrites, false);
    assert.equal(candidatePackInput.seedingBoundary.hotPathDependency, false);
    assert.equal(candidatePackInput.importedPriors.hubPriors.length, 2);
    assert.equal(candidatePackInput.importedPriors.neighborhoodPriors.length, 1);
    assert.equal(candidatePackInput.importedPriors.evidencePointers.length, result.counts.evidencePointers);
    assert.ok(candidatePackInput.importedPriors.hubPriors.every((prior) => prior.trustClass === "EXTRACTED"));
    assert.ok(candidatePackInput.importedPriors.neighborhoodPriors.every((prior) => prior.trustClass === "EXTRACTED"));
    assert.ok(candidatePackInput.importedPriors.evidencePointers.every((pointer) => pointer.trustClass === "EXTRACTED"));
    assert.equal(slice.truthBoundary.allowedTrustClasses.join(","), "EXTRACTED");
    assert.deepEqual(slice.truthBoundary.blockedTrustClasses, ["INFERRED", "AMBIGUOUS"]);
    assert.ok(slice.hubPriors.every((prior) => prior.trustClass === "EXTRACTED"));
    assert.ok(slice.neighborhoodPriors.every((prior) => prior.trustClass === "EXTRACTED"));
    assert.ok(slice.evidencePointers.every((pointer) => pointer.trustClass === "EXTRACTED"));
    assert.ok(slice.rationalePointers.every((pointer) => pointer.trustClass === "EXTRACTED"));
    assert.match(report, /artifact-first then import-second/);
    assert.match(report, /EXTRACTED only/);
    assert.match(report, /rollback-safe/);
    assert.match(report, /no current-truth-like write/);
    assert.equal(envelope.contract, "graphify_import_slice_proposal.v1");
    assert.equal(envelope.reviewMode, "candidate_only");
    assert.equal(envelope.targetStateOnly, true);
    assert.equal(envelope.replayGate.reviewMode, "candidate_only");
    assert.equal(replayGate.contract, "graphify_import_slice_replay_gate.v1");
    assert.ok(replayGate.blockedEffects.includes("current_truth_write"));
    assert.ok(replayGate.blockedEffects.includes("hot_path_serve_integration"));
});

test("graphify-import-slice parses and runs through the public CLI surface", (t) => {
    const tempRoot = createTempRoot(t);
    const packRoot = path.join(tempRoot, "graphify-pack");
    const outputRoot = path.join(tempRoot, "artifacts", "graphify-imports");
    const bundle = buildFixturePack(packRoot);
    writeGraphifyCompiledArtifactPack(bundle.outputDir, bundle);

    const argv = [
        "graphify-import-slice",
        "--bundle-root",
        bundle.outputDir,
        "--output-root",
        outputRoot,
        "--run-id",
        "cli-run",
        "--json",
    ];

    const parsed = parseOperatorCliArgs(argv);
    assert.equal(parsed.command, "graphify-import-slice");
    assert.equal(parsed.bundleRoot, path.resolve(bundle.outputDir));
    assert.equal(parsed.outputRoot, path.resolve(outputRoot));
    assert.equal(parsed.runId, "cli-run");
    assert.equal(parsed.json, true);

    const logs = [];
    const errors = [];
    const originalLog = console.log;
    const originalError = console.error;
    console.log = (...args) => {
        logs.push(args.join(" "));
    };
    console.error = (...args) => {
        errors.push(args.join(" "));
    };
    try {
        const exitCode = runOperatorCli(argv);
        assert.equal(exitCode, 0);
    }
    finally {
        console.log = originalLog;
        console.error = originalError;
    }

    assert.equal(errors.length, 0);
    const payload = JSON.parse(logs.join("\n"));
    assert.equal(payload.ok, true);
    assert.equal(payload.runId, "cli-run");
    assert.equal(payload.counts.hubPriors, 2);
    assert.equal(payload.counts.neighborhoodPriors, 1);
    assert.equal(payload.candidatePackInput.contract, "graphify_import_slice_candidate_pack_input.v1");
    assert.ok(payload.paths.candidatePackInput.endsWith(path.join("cli-run", "candidate-pack-input.json")));
    assert.ok(payload.paths.importSlice.endsWith(path.join("cli-run", "import-slice.json")));
    assert.ok(payload.paths.replayGate.endsWith(path.join("cli-run", "replay-gate.json")));
});
