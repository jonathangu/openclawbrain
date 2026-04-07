import test from "node:test";
import assert from "node:assert/strict";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { buildGraphifyCompiledArtifactPack, writeGraphifyCompiledArtifactPack } from "../src/graphify-compiled-artifacts.js";
import { exportGraphifyImportSlice } from "../src/import-export.js";
import { runManagedGraphifyRunner } from "../src/graphify-runner.js";
import { buildGraphifyMaintenanceDiffBundle, writeGraphifyMaintenanceDiffBundle } from "../src/graphify-maintenance-diff.js";
import { parseOperatorCliArgs, runOperatorCli } from "../src/cli.js";

function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-maintenance-diff-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}

function writeNewCurrentArtifact(compiledOutputDir) {
    const artifactDir = path.join(compiledOutputDir, "artifacts", "ca_graphify_new_current_hub_01");
    mkdirSync(artifactDir, { recursive: true });
    writeFileSync(path.join(artifactDir, "artifact.md"), [
        "---",
        "artifact_id: ca_graphify_new_current_hub_01",
        "kind: map_of_territory",
        "status: proposed",
        "title: Graphify new current hub",
        "proposal_id: prop_graphify_new_current_hub_01",
        "pack_id: pack_graphify_compiled_artifacts_smoke",
        "subject_ids:",
        "  - topic:graphify-maintenance",
        "  - topic:new-hub",
        "confidence: 0.88",
        "created_at: 2026-04-06T20:30:00.000Z",
        "updated_at: 2026-04-06T20:30:00.000Z",
        "---",
        "",
        "## Summary",
        "",
        "New current hub for diff smoke.",
        "",
        "## Evidence",
        "",
        "- `ev-missing` — docs/architecture/graphify-bridge.md",
    ].join("\n"));
    writeFileSync(path.join(artifactDir, "artifact.meta.json"), JSON.stringify({
        schemaVersion: 1,
        artifactId: "ca_graphify_new_current_hub_01",
        kind: "map_of_territory",
        title: "Graphify new current hub",
        status: "proposed",
        packId: "pack_graphify_compiled_artifacts_smoke",
        proposalId: "prop_graphify_new_current_hub_01",
        proposalLane: "compiler",
        subjectIds: ["topic:graphify-maintenance", "topic:new-hub"],
        evidence: [
            {
                evidenceId: "ev-missing",
                sourceKind: "file",
                sourceId: "docs/architecture/graphify-bridge.md",
                authority: "raw_source",
                derivation: "teacher_compilation",
                excerpt: "Graphify outputs are review-only.",
                sourceHash: null,
            },
        ],
        counterevidence: [],
        provenance: {
            producer: "graphify",
            producerVersion: "test",
            scope: "graphify/compiled-artifacts",
            idempotencyKey: "smoke",
            sourceRoots: ["docs/architecture"],
            transformChain: ["extract"],
            sourceBundleId: "source-bundle",
            graphHash: "sha256:graph",
            graphifyRunId: "run-smoke",
        },
        contentHash: "sha256:new-current",
        markdownPath: "artifacts/ca_graphify_new_current_hub_01/artifact.md",
        metaPath: "artifacts/ca_graphify_new_current_hub_01/artifact.meta.json",
        createdAt: "2026-04-06T20:30:00.000Z",
        updatedAt: "2026-04-06T20:30:00.000Z",
        confidence: 0.88,
        claims: [
            {
                claimId: "claim-new-current",
                text: "New current hub.",
                confidence: 0.88,
                status: "supported",
                evidenceIds: [],
            },
        ],
        promotion: {
            replaySuites: ["smoke"],
            rollbackKey: "rollback:smoke:new-current",
        },
    }, null, 2) + "\n");

    const manifestPath = path.join(compiledOutputDir, "pack.manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    manifest.artifacts.push({
        artifactId: "ca_graphify_new_current_hub_01",
        kind: "map_of_territory",
        title: "Graphify new current hub",
        markdownPath: "artifacts/ca_graphify_new_current_hub_01/artifact.md",
        metaPath: "artifacts/ca_graphify_new_current_hub_01/artifact.meta.json",
        contentHash: "sha256:new-current",
        summary: "New current hub for diff smoke.",
    });
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
}

function buildFixtureRoots(t) {
    const root = createTempRoot(t);
    const currentRoot = path.join(root, "current");
    const ocbRoot = path.join(root, "ocb");
    mkdirSync(currentRoot, { recursive: true });
    mkdirSync(ocbRoot, { recursive: true });

    const compiledDir = path.join(currentRoot, "compiled");
    const compiled = buildGraphifyCompiledArtifactPack({
        bundleId: "smoke",
        bundleStartedAt: "2026-04-06T20:30:00.000Z",
        outputDir: compiledDir,
        graphifyRunId: "run-smoke",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "source-bundle",
        sourceBundleHash: "sha256:source",
        graphHash: "sha256:graph",
        configHash: "sha256:config",
        labelsHash: "sha256:labels",
    });
    writeGraphifyCompiledArtifactPack(compiled.outputDir, compiled);

    exportGraphifyImportSlice({ bundleRoot: compiled.outputDir, outputRoot: currentRoot, runId: "import" });
    runManagedGraphifyRunner({
        sourceBundlePath: compiled.outputDir,
        outputRoot: currentRoot,
        runId: "run",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyMode: "managed-off-path",
        graphifyCommand: null,
        graphifyArgs: [],
        graphifyFlags: [],
        graphifyConfig: {},
        labels: [],
    });

    writeNewCurrentArtifact(compiled.outputDir);
    cpSync(compiled.outputDir, path.join(ocbRoot, "candidate"), { recursive: true });
    cpSync(compiled.outputDir, path.join(ocbRoot, "compiled"), { recursive: true });
    cpSync(compiled.outputDir, path.join(ocbRoot, "promoted"), { recursive: true });
    const promotedMeta = path.join(ocbRoot, "promoted", "artifacts", "ca_graphify_concept_page_01", "artifact.meta.json");
    const promotedMetaData = JSON.parse(readFileSync(promotedMeta, "utf8"));
    promotedMetaData.contentHash = "sha256:deadbeef";
    writeFileSync(promotedMeta, JSON.stringify(promotedMetaData, null, 2) + "\n");

    const importPath = path.join(currentRoot, "import", "import-slice.json");
    const importSlice = JSON.parse(readFileSync(importPath, "utf8"));
    const hub0 = importSlice.hubPriors[0];
    importSlice.hubPriors.push({ ...hub0, priorId: "hub-prior-duplicate", label: `${hub0.label} duplicate`, title: `${hub0.title} duplicate` });
    importSlice.counts.hubPriors += 1;
    writeFileSync(importPath, JSON.stringify(importSlice, null, 2) + "\n");

    return { root, currentRoot, ocbRoot };
}

test("graphify maintenance diff emits bounded operator diagnostics for current-vs-OCB surface drift", (t) => {
    const { currentRoot, ocbRoot } = buildFixtureRoots(t);
    const outputRoot = path.join(createTempRoot(t), "out");
    const result = buildGraphifyMaintenanceDiffBundle({
        graphifyRoot: currentRoot,
        ocbRoot,
        outputRoot,
        runId: "maintenance-diff-smoke",
    });
    writeGraphifyMaintenanceDiffBundle(result.outputDir, result);

    assert.equal(result.ok, true);
    assert.equal(result.runId, "maintenance-diff-smoke");
    assert.equal(result.verdict.verdict, "needs_review");
    assert.equal(result.digest.fileCount, 4);
    assert.ok(result.report.counts.missing_from_ocb > 0);
    assert.ok(result.report.counts.stale_in_ocb > 0);
    assert.ok(result.report.counts.candidate_only_edges_without_source_support > 0);
    assert.ok(result.report.counts.new_current_source_hubs > 0);
    assert.ok(result.report.counts.provenance_gap_candidates > 0);
    assert.ok(result.report.counts.possible_merge_split_review_hints > 0);
    assert.ok(existsSync(result.paths.maintenanceDiff));
    assert.ok(existsSync(result.paths.proposalSuggestion));
    assert.ok(existsSync(result.paths.verdict));
    assert.match(result.summary, /Graphify × OCB maintenance diff/);
    assert.match(result.summary, /missing_from_ocb/);
    assert.match(readFileSync(result.paths.summary, "utf8"), /possible merge\/split review hints/);
});

test("graphify-maintenance-diff parses and runs through the public CLI surface", (t) => {
    const { currentRoot, ocbRoot } = buildFixtureRoots(t);
    const outputRoot = path.join(createTempRoot(t), "cli-out");
    const argv = [
        "graphify-maintenance-diff",
        "--graphify-root",
        currentRoot,
        "--ocb-root",
        ocbRoot,
        "--output-root",
        outputRoot,
        "--run-id",
        "cli-run",
        "--json",
    ];

    const parsed = parseOperatorCliArgs(argv);
    assert.equal(parsed.command, "graphify-maintenance-diff");
    assert.equal(parsed.graphifyRoot, path.resolve(currentRoot));
    assert.equal(parsed.ocbRoot, path.resolve(ocbRoot));
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
    assert.equal(payload.report.diffId, "graphify-maintenance-diff-cli-run");
    assert.equal(payload.verdict.verdict, "needs_review");
    assert.ok(payload.report.counts.stale_in_ocb > 0);
    assert.ok(payload.report.counts.candidate_only_edges_without_source_support > 0);
    assert.ok(payload.paths.maintenanceDiff.endsWith(path.join("cli-run", "maintenance-diff.json")));
    assert.ok(payload.paths.verdict.endsWith(path.join("cli-run", "verdict.json")));
});
