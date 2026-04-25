import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { runGraphifyBridgeCut } from "../src/graphify-bridge.js";
import { parseGraphifyBridgeCliArgs } from "../../../../scripts/graphify-bridge.mjs";

function writeFile(filePath, content) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${content.replace(/\n?$/u, "")}\n`, "utf8");
}

function createBridgeFixture(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-bridge-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));

  const openclawHome = path.join(root, ".openclaw");
  const activationRoot = path.join(root, ".openclawbrain", "activation");
  const workspaceRoot = path.join(root, "workspace");
  const repoRoot = path.join(root, "repo");
  const outputRoot = path.join(root, "artifacts", "graphify-bridge");
  const sessionsDir = path.join(openclawHome, "agents", "main", "sessions");
  const sessionFile = path.join(sessionsDir, "bridge-session.jsonl");
  const sessionProjectionSource = path.join(workspaceRoot, "session-source.md");
  const proofSummarySource = path.join(workspaceRoot, "proof-summary.md");

  mkdirSync(sessionsDir, { recursive: true });
  mkdirSync(path.join(activationRoot, "attachment-truth"), { recursive: true });
  writeFileSync(path.join(activationRoot, "activation-pointers.json"), JSON.stringify({ activePack: "pack-fixture" }, null, 2), "utf8");
  writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify({ profile: "bridge-fixture" }, null, 2), "utf8");
  writeFileSync(
    sessionFile,
    [
      JSON.stringify({
        type: "session",
        version: 1,
        id: "session-graphify-bridge",
        timestamp: "2026-04-25T12:00:00.000Z",
        cwd: repoRoot,
      }),
      JSON.stringify({
        type: "message",
        id: "msg-bridge-1",
        parentId: null,
        timestamp: "2026-04-25T12:00:01.000Z",
        message: {
          role: "user",
          content: "Build the first safe Graphify bridge cut.",
          timestamp: 1777118401000,
        },
      }),
      JSON.stringify({
        type: "message",
        id: "msg-bridge-2",
        parentId: "msg-bridge-1",
        timestamp: "2026-04-25T12:00:02.000Z",
        message: {
          role: "assistant",
          content: "Graphify stays artifact-first and off the live runtime path.",
          timestamp: 1777118402000,
        },
      }),
    ].join("\n") + "\n",
    "utf8",
  );
  writeFileSync(
    path.join(sessionsDir, "sessions.json"),
    JSON.stringify({ bridge: { sessionId: "session-graphify-bridge", sessionFile, updatedAt: 1 } }, null, 2),
    "utf8",
  );

  writeFile(path.join(workspaceRoot, "MEMORY.md"), "# MEMORY\n- Graphify projection is not current truth.");
  writeFile(path.join(workspaceRoot, "TASKS.md"), "# TASKS\n- Keep bridge output artifact-only.");
  writeFile(sessionProjectionSource, "# Session source\n- Dual source export fixture.");
  writeFile(proofSummarySource, "# Proof source\n- Canonical machine export wins over projection.");
  writeFile(path.join(repoRoot, "docs", "architecture", "graphify-bridge.md"), "# Graphify bridge\nArtifact-first.");
  writeFile(path.join(repoRoot, "docs", "architecture", "compiled-artifacts.md"), "# Compiled artifacts\nDerived only.");
  writeFile(path.join(repoRoot, "packages", "cli", "dist", "src", "import-export.js"), "export const fixture = true;");
  writeFile(path.join(repoRoot, "packages", "cli", "dist", "src", "graphify-runner.js"), "export const runner = true;");

  return {
    root,
    openclawHome,
    activationRoot,
    workspaceRoot,
    repoRoot,
    outputRoot,
    sessionProjectionSource,
    proofSummarySource,
  };
}

test("runGraphifyBridgeCut writes dual source, managed run, compiled pack, and explicit off-path status", (t) => {
  const fixture = createBridgeFixture(t);
  const result = runGraphifyBridgeCut({
    openclawHome: fixture.openclawHome,
    activationRoot: fixture.activationRoot,
    workspaceRoot: fixture.workspaceRoot,
    repoRoot: fixture.repoRoot,
    outputRoot: fixture.outputRoot,
    runId: "safe-cut-fixture",
    generatedAt: "2026-04-25T12:00:00.000Z",
    sessionKey: "bridge",
    sessionSourcePath: fixture.sessionProjectionSource,
    proofSummarySourcePath: fixture.proofSummarySource,
    homeDir: fixture.root,
  });

  assert.equal(result.ok, true);
  assert.equal(result.runId, "safe-cut-fixture");
  assert.ok(existsSync(path.join(result.sourceBundleRoot, "canonical", "corpus-manifest.json")));
  assert.ok(existsSync(path.join(result.sourceBundleRoot, "projection", "corpus-manifest.json")));
  assert.ok(existsSync(result.sourceBundleManifestPath));
  assert.ok(existsSync(path.join(result.graphifyRun.runDir, "graphify-summary.json")));
  assert.ok(existsSync(path.join(result.compiledPack.outputDir, "pack.manifest.json")));
  assert.ok(existsSync(path.join(result.compiledPack.outputDir, "surface-map.json")));
  assert.ok(existsSync(path.join(result.compiledPack.outputDir, "proposal-report.json")));
  assert.ok(existsSync(path.join(result.compiledPack.outputDir, "verdict.json")));
  assert.ok(existsSync(result.statusPath));
  assert.ok(existsSync(result.summaryPath));

  const dualManifest = JSON.parse(readFileSync(result.sourceBundleManifestPath, "utf8"));
  const graphifySummary = JSON.parse(readFileSync(path.join(result.graphifyRun.runDir, "graphify-summary.json"), "utf8"));
  const compiledManifest = JSON.parse(readFileSync(path.join(result.compiledPack.outputDir, "pack.manifest.json"), "utf8"));
  const status = JSON.parse(readFileSync(result.statusPath, "utf8"));
  const summary = readFileSync(result.summaryPath, "utf8");

  assert.equal(dualManifest.contract, "graphify_dual_source_bundle_manifest.v1");
  assert.equal(dualManifest.authoritativeLane, "canonical");
  assert.equal(dualManifest.projectionTruth, "projection_only");
  assert.equal(dualManifest.truthBoundary.beforePromptBuildEligible, false);
  assert.equal(dualManifest.truthBoundary.liveRuntimeEligible, false);
  assert.equal(graphifySummary.contract, "graphify_run_summary.v1");
  assert.equal(graphifySummary.sourceBundleHash, result.graphifyRun.sourceBundleHash);
  assert.equal(graphifySummary.execution.state, "synthesized");
  assert.deepEqual(
    compiledManifest.artifacts.map((artifact) => artifact.kind),
    ["map_of_territory", "concept_page", "neighborhood_summary", "provenance_gap_report"],
  );
  assert.equal(status.contract, "graphify_bridge_cut.v1");
  assert.equal(status.offPath, true);
  assert.equal(status.liveRuntimeTouched, false);
  assert.equal(status.beforePromptBuildTouched, false);
  assert.equal(status.truthBoundary.importOrPromotionPerformed, false);
  assert.match(summary, /before_prompt_build touched: no/);
  assert.match(summary, /compiled artifact pack/);
});

test("graphify bridge CLI parser keeps the cut explicit and synthesized by default", () => {
  const parsed = parseGraphifyBridgeCliArgs([
    "--openclaw-home", "/tmp/openclaw",
    "--activation-root", "/tmp/activation",
    "--output-root", "/tmp/out",
    "--run-id", "cli-cut",
    "--generated-at", "2026-04-25T12:00:00.000Z",
    "--label", "lane-c",
    "--graphify-arg", "--fixture",
    "--json",
  ]);
  assert.equal(parsed.runId, "cli-cut");
  assert.equal(parsed.generatedAt, "2026-04-25T12:00:00.000Z");
  assert.equal(parsed.json, true);
  assert.deepEqual(parsed.labels, ["lane-c"]);
  assert.deepEqual(parsed.graphifyArgs, ["--fixture"]);
  assert.equal(parsed.graphifyCommand, undefined);
});
