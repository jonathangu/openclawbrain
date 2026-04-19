import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildGraphifyCompiledArtifactPack, resolveGraphifyCompiledArtifactPackOutputDir, writeGraphifyCompiledArtifactPack, GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1 } from "../src/graphify-compiled-artifacts.js";
import { exportGraphifyCompiledArtifactsPack } from "../src/import-export.js";
function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-pack-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}
function buildFixtureBundle(outputDir) {
    return buildGraphifyCompiledArtifactPack({
        bundleId: "bridge-fixture",
        bundleStartedAt: "2026-04-06T20:30:00.000Z",
        outputDir,
        graphifyRunId: "graphify-run-fixture",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "compiled-artifacts-target-state-scaffold",
        sourceBundleHash: "sha256:source-bundle-hash",
        graphHash: "sha256:graph-hash",
        configHash: "sha256:config-hash",
        labelsHash: "sha256:labels-hash",
    });
}
test("graphify compiled-artifact pack bridge writes the scaffold pack with stable hashes and four artifact kinds", (t) => {
    const tempRoot = createTempRoot(t);
    const outputDir = path.join(tempRoot, "bridge-output");
    const bundle = buildFixtureBundle(outputDir);
    assert.deepEqual(bundle.artifactEntries.map((entry) => entry.kind), GRAPHIFY_COMPILED_ARTIFACT_KIND_ORDER_V1);
    assert.equal(bundle.packManifest.artifacts.length, 4);
    assert.equal(bundle.surfaceMap.counts.shippedSurfaceCount, 5);
    assert.equal(bundle.surfaceMap.counts.targetSurfaceCount, 9);
    assert.equal(bundle.surfaceMap.counts.totalSurfaceCount, 14);
    assert.equal(bundle.validation.ok, true);
    assert.equal(bundle.verdict.verdict, "reviewable");
    assert.match(bundle.proposalReport.recommendations.join("\n"), /Keep Graphify-derived surfaces off the serve path/);
    assert.equal(bundle.digest.fileCount, Object.keys(bundle.files).length);
    const secondBundle = buildFixtureBundle(path.join(tempRoot, "bridge-output-2"));
    assert.equal(bundle.digest.bundleHash, secondBundle.digest.bundleHash);
    const writeResult = writeGraphifyCompiledArtifactPack(bundle.outputDir, bundle);
    assert.equal(writeResult.fileCount, Object.keys(bundle.files).length);
    assert.ok(existsSync(path.join(bundle.outputDir, "pack.manifest.json")));
    assert.ok(existsSync(path.join(bundle.outputDir, "proposals", "compiler-proposal.json")));
    assert.ok(existsSync(path.join(bundle.outputDir, "surface-map.json")));
    assert.ok(existsSync(path.join(bundle.outputDir, "proposal-report.json")));
    assert.ok(existsSync(path.join(bundle.outputDir, "verdict.json")));
    assert.ok(existsSync(path.join(bundle.outputDir, "artifacts", "ca_graphify_concept_page_01", "artifact.md")));
    assert.ok(existsSync(path.join(bundle.outputDir, "artifacts", "ca_graphify_neighborhood_summary_01", "artifact.meta.json")));
    const manifest = JSON.parse(readFileSync(path.join(bundle.outputDir, "pack.manifest.json"), "utf8"));
    const verdict = JSON.parse(readFileSync(path.join(bundle.outputDir, "verdict.json"), "utf8"));
    const conceptMarkdown = readFileSync(path.join(bundle.outputDir, "artifacts", "ca_graphify_concept_page_01", "artifact.md"), "utf8");
    assert.equal(manifest.contract, "graphify_compiled_artifact_pack.v1");
    assert.equal(manifest.artifacts[0].artifactId, "ca_graphify_map_of_territory_01");
    assert.equal(verdict.reviewMode, "promotable");
    assert.match(conceptMarkdown, /## Stronger-truth anchors/);
    assert.match(conceptMarkdown, /The sidecar JSON should be the authoritative metadata source/);
});

test("graphify compiled-artifact pack CLI help and export wrapper keep the bridge path explicit", (t) => {
    const tempRoot = createTempRoot(t);
    const cliSource = readFileSync(fileURLToPath(new URL("../src/cli.js", import.meta.url)), "utf8");
    assert.match(cliSource, /graphify-compiled-artifacts/);
    assert.match(cliSource, /--bundle-id <id>/);
    assert.match(cliSource, /graphify-compiled-artifacts derive a Graphify-shaped compiled-artifact pack/);
    assert.match(cliSource, /GRAPHIFY COMPILED ARTIFACTS ok/);
    const resolvedOutputDir = resolveGraphifyCompiledArtifactPackOutputDir({
        bundleId: "bridge-fixture",
        bundleStartedAt: "2026-04-06T20:30:00.000Z",
    });
    assert.match(resolvedOutputDir, /artifacts\/teacher-v3-proof\/bridge-fixture\/compiled-artifacts$/);
    const exportResult = exportGraphifyCompiledArtifactsPack({
        bundleId: "bridge-fixture",
        outputDir: path.join(tempRoot, "export-output"),
        bundleStartedAt: "2026-04-06T20:30:00.000Z",
        graphifyRunId: "graphify-run-fixture",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "compiled-artifacts-target-state-scaffold",
        sourceBundleHash: "sha256:source-bundle-hash",
        graphHash: "sha256:graph-hash",
        configHash: "sha256:config-hash",
        labelsHash: "sha256:labels-hash",
    });
    assert.equal(exportResult.ok, true);
    assert.equal(exportResult.packId, "pack_graphify_compiled_artifacts_bridge-fixture");
    assert.ok(existsSync(path.join(exportResult.outputDir, "pack.manifest.json")));
    assert.ok(existsSync(path.join(exportResult.outputDir, "verdict.json")));
    assert.equal(exportResult.validation.errors.length, 0);
    assert.ok(exportResult.digest.bundleHash.startsWith("sha256:"));
});
