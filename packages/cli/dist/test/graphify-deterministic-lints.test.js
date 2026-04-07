import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildGraphifyDeterministicLintBundle, parseGraphifyDeterministicLintCliArgs, runGraphifyDeterministicLints } from "../src/graphify-lints.js";

function createTempRoot(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-lints-"));
  t.after(() => {
    rmSync(root, { recursive: true, force: true });
  });
  return root;
}

function sha256(text) {
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

test("graphify deterministic pre-lint writes bounded proposal surfaces for a real proof bundle", (t) => {
  const tempRoot = createTempRoot(t);
  const repoRoot = path.resolve(process.cwd());
  const bundleRoot = path.join(repoRoot, "artifacts", "teacher-v3-promotable-examples", "lint", "proof-bundle");
  assert.equal(existsSync(bundleRoot), true, "fixture bundle should exist");

  const result = buildGraphifyDeterministicLintBundle({
    bundleRoot,
    repoRoot,
    outputRoot: path.join(tempRoot, "out"),
    runId: "real-bundle",
  });

  assert.equal(result.ok, false);
  assert.equal(result.report.ok, false);
  assert.ok(result.report.findings.length >= 2);
  assert.equal(result.verdict.verdict, "rejected");
  assert.equal(existsSync(result.paths.deterministicLints), true);
  assert.equal(existsSync(result.paths.summary), true);
  assert.equal(existsSync(result.paths.proposalEnvelope), true);
  assert.equal(existsSync(result.paths.verdict), true);

  const envelope = readJson(result.paths.proposalEnvelope);
  const verdict = readJson(result.paths.verdict);
  const deterministic = readJson(result.paths.deterministicLints);
  assert.equal(envelope.status, "rejected");
  assert.equal(verdict.verdict, "rejected");
  assert.equal(deterministic.bundleId, "teacher-v3-lint-worked-example");
  const findingCodes = new Set(result.report.findings.map((finding) => finding.code));
  assert.ok(findingCodes.has("release_docs_version_drift"));
  assert.ok(findingCodes.has("broken_bundle_joins"));
  assert.ok(!findingCodes.has("missing_source_files"));
  assert.match(result.summary, /Graphify deterministic pre-lint/);
});

test("graphify deterministic pre-lint detects missing files, hash drift, trust promotion, evidence gaps, joins, and release drift", (t) => {
  const tempRoot = createTempRoot(t);
  const repoRoot = path.resolve(process.cwd());
  const bundleRoot = path.join(tempRoot, "broken-bundle");
  const artifactDir = path.join(bundleRoot, "artifacts", "example");
  mkdirSync(artifactDir, { recursive: true });

  const artifactBody = "artifact body v1\n";
  const staleBodyHash = sha256("artifact body v0\n");
  writeFileSync(path.join(artifactDir, "artifact.md"), artifactBody, "utf8");
  writeFileSync(path.join(artifactDir, "artifact.meta.json"), JSON.stringify({
    schemaVersion: 1,
    contentHash: staleBodyHash,
    markdownPath: "artifacts/example/artifact.md",
    metaPath: "artifacts/example/artifact.meta.json",
  }, null, 2), "utf8");

  writeFileSync(path.join(bundleRoot, "summary.md"), "# broken bundle\n", "utf8");
  writeFileSync(path.join(bundleRoot, "status.json"), JSON.stringify({
    contract: "graphify_bundle_status.v1",
    bundleId: "bundle-a",
    proposalId: "proposal-1",
    proposalClass: "mutation",
    proposalLane: "lint",
    proposalStatus: "promotable",
    reviewMode: "promotable",
  }, null, 2), "utf8");
  writeFileSync(path.join(bundleRoot, "surface-map.json"), JSON.stringify({
    contract: "graphify_surface_map.v1",
    bundleId: "bundle-b",
    observedSurfaces: [
      { id: "runtime-truth", state: "shipped", kind: "runtime_truth", source: "docs/architecture/teacher-v3-lints.md" },
      { id: "release-notes", state: "shipped", kind: "docs_truth", source: "docs/release-notes-0.4.29.md" },
      { id: "missing-source", state: "target", kind: "docs_truth", source: "docs/missing-source.md" },
    ],
    bundleArtifacts: [
      { id: "artifact-1", state: "target", kind: "proposal_truth", source: "artifacts/example/artifact.md" },
    ],
    counts: {
      observedSurfaceCount: 2,
      shippedSurfaceCount: 2,
      targetSurfaceCount: 1,
      totalSurfaceCount: 3,
    },
  }, null, 2), "utf8");
  writeFileSync(path.join(bundleRoot, "proposal-report.json"), JSON.stringify({
    contract: "graphify_proposal_report.v1",
    bundleId: "bundle-c",
    proposal: {
      proposalId: "proposal-1",
      proposalLane: "lint",
      proposalClass: "mutation",
      status: "promotable",
      reviewMode: "promotable",
      evidence: [
        {
          evidenceId: "ev-1",
          sourceKind: "file",
          sourceId: "docs/architecture/teacher-v3-lints.md",
          authority: "raw_source",
          derivation: "teacher_lint",
        },
      ],
      claims: [
        {
          claimId: "claim-1",
          text: "This claim is missing evidence ids.",
          status: "supported",
          evidenceIds: [],
        },
      ],
    },
    docsTruth: {
      path: "docs/release-notes-0.4.29.md",
      version: "0.4.29",
      title: "release notes",
      state: "shipped",
    },
    publicationSafeArtifacts: [
      { artifactId: "summary", kind: "summary", path: "openclawbrain/artifacts/graphify-lints/broken/summary.md", containsRawLogs: false },
      { artifactId: "verdict", kind: "verdict", path: "openclawbrain/artifacts/graphify-lints/broken/verdict.json", containsRawLogs: false },
    ],
  }, null, 2), "utf8");
  writeFileSync(path.join(bundleRoot, "verdict.json"), JSON.stringify({
    contract: "graphify_bundle_verdict.v1",
    bundleId: "bundle-d",
    proposalId: "proposal-1",
    verdict: "reviewable",
    severity: "info",
    reviewMode: "promotable",
    proposalClass: "mutation",
  }, null, 2), "utf8");
  writeFileSync(path.join(bundleRoot, "pack.manifest.json"), JSON.stringify({
    contract: "graphify_pack_manifest.v1",
    packId: "pack-broken-1",
    bundleId: "bundle-a",
    sourceDocs: [
      "docs/architecture/teacher-v3-lints.md",
      "docs/release-notes-0.4.29.md",
      "docs/missing-source.md",
    ],
    artifacts: [
      {
        artifactId: "artifact-1",
        kind: "concept_page",
        markdownPath: "artifacts/example/artifact.md",
        metaPath: "artifacts/example/artifact.meta.json",
        contentHash: staleBodyHash,
      },
    ],
  }, null, 2), "utf8");

  const outputRoot = path.join(tempRoot, "out");
  const result = runGraphifyDeterministicLints({
    bundleRoot,
    repoRoot,
    workspaceRoot: path.resolve(repoRoot, ".."),
    outputRoot,
    runId: "broken-bundle",
  });

  assert.equal(result.ok, false);
  assert.equal(result.report.ok, false);
  assert.equal(result.verdict.verdict, "rejected");
  assert.equal(existsSync(result.paths.deterministicLints), true);

  const findingCodes = new Set(result.report.findings.map((finding) => finding.code));
  assert.ok(findingCodes.has("missing_source_files"));
  assert.ok(findingCodes.has("manifest_hash_drift"));
  assert.ok(findingCodes.has("illegal_trust_class_promotion"));
  assert.ok(findingCodes.has("missing_evidence_refs"));
  assert.ok(findingCodes.has("broken_bundle_joins"));
  assert.ok(findingCodes.has("release_docs_version_drift"));

  const deterministic = readJson(result.paths.deterministicLints);
  assert.equal(deterministic.severity, "error");
  assert.equal(deterministic.ok, false);
  assert.ok(deterministic.findings.length >= 6);
  assert.match(result.summary, /findings:/);
});

test("graphify deterministic lint CLI parsing accepts the pre-lint bundle flags", () => {
  const parsed = parseGraphifyDeterministicLintCliArgs([
    "--bundle-root",
    "/tmp/bundle",
    "--repo-root",
    "/tmp/repo",
    "--workspace-root",
    "/tmp",
    "--output-root",
    "/tmp/out",
    "--run-id",
    "run-123",
    "--json",
  ]);

  assert.equal(parsed.command, "graphify-lints");
  assert.equal(parsed.bundleRoot, "/tmp/bundle");
  assert.equal(parsed.repoRoot, path.resolve("/tmp/repo"));
  assert.equal(parsed.workspaceRoot, path.resolve("/tmp"));
  assert.equal(parsed.outputRoot, path.resolve("/tmp/out"));
  assert.equal(parsed.runId, "run-123");
  assert.equal(parsed.json, true);
});
