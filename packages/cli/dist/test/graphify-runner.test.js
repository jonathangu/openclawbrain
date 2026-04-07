import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { runManagedGraphifyRunner } from "../src/graphify-runner.js";

function createSourceBundle(t, label) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), `${label}-source-bundle-`));
  mkdirSync(path.join(rootDir, "docs"), { recursive: true });
  mkdirSync(path.join(rootDir, "code"), { recursive: true });
  writeFileSync(
    path.join(rootDir, "corpus-manifest.json"),
    JSON.stringify(
      {
        contract: "graphify_source_bundle.v1",
        labels: ["alpha", "beta"],
        provenance: {
          source: "test-fixture",
        },
      },
      null,
      2,
    ),
    "utf8",
  );
  writeFileSync(path.join(rootDir, "docs", "intro.md"), "# Intro\nGraphify source bundle test fixture.\n", "utf8");
  writeFileSync(path.join(rootDir, "code", "app.ts"), "export const answer = 42;\n", "utf8");
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return rootDir;
}

function createOutputRoot(t, label) {
  const outputRoot = mkdtempSync(path.join(os.tmpdir(), `${label}-graphify-out-`));
  t.after(() => {
    rmSync(outputRoot, { recursive: true, force: true });
  });
  return outputRoot;
}

test("managed graphify runner emits the full run bundle and records managed command metadata", (t) => {
  const sourceBundle = createSourceBundle(t, "graphify-managed-command");
  const outputRoot = createOutputRoot(t, "graphify-managed-command");

  const result = runManagedGraphifyRunner({
    sourceBundlePath: sourceBundle,
    outputRoot,
    graphifyVersion: "graphify 9.9.9",
    graphifyMode: "compiler",
    graphifyConfig: {
      layout: "force",
      seed: 7,
    },
    graphifyFlags: ["--static", "--reproducible"],
    graphifyCommand: process.execPath,
    graphifyArgs: ["-e", "process.exit(0)"],
    labels: ["wave1"],
  });

  assert.equal(result.ok, true);
  assert.equal(result.graphifyVersion, "graphify 9.9.9");
  assert.equal(result.graphifyVersionSource, "provided");
  assert.equal(result.graphifyMode, "compiler");
  assert.equal(result.execution.state, "executed");
  assert.equal(result.execution.command, process.execPath);
  assert.ok(result.sourceBundleHash.length === 64);

  const command = JSON.parse(readFileSync(result.outputs.command, "utf8"));
  const summary = JSON.parse(readFileSync(result.outputs.summary, "utf8"));
  const graph = JSON.parse(readFileSync(result.outputs.graph, "utf8"));
  const labels = JSON.parse(readFileSync(result.outputs.labels, "utf8"));
  const benchmark = JSON.parse(readFileSync(result.outputs.benchmark, "utf8"));
  const report = readFileSync(result.outputs.report, "utf8");

  assert.equal(command.contract, "graphify_command_record.v1");
  assert.equal(command.runId, result.runId);
  assert.equal(command.sourceBundle.hash, result.sourceBundleHash);
  assert.equal(command.graphify.version, "graphify 9.9.9");
  assert.equal(command.graphify.versionSource, "provided");
  assert.equal(command.execution.state, "executed");
  assert.equal(command.execution.command, process.execPath);
  assert.equal(command.outputs.graph, result.outputs.graph);

  assert.equal(summary.contract, "graphify_run_summary.v1");
  assert.equal(summary.runId, result.runId);
  assert.equal(summary.sourceBundleHash, result.sourceBundleHash);
  assert.equal(summary.graphifyVersion, "graphify 9.9.9");
  assert.equal(summary.graph.nodeCount, result.graph.nodeCount);
  assert.equal(summary.graph.edgeCount, result.graph.edgeCount);
  assert.equal(summary.execution.state, "executed");
  assert.ok(Array.isArray(summary.labels));
  assert.ok(summary.labels.includes("wave1"));

  assert.equal(graph.contract, "graphify_graph.v1");
  assert.equal(graph.runId, result.runId);
  assert.equal(graph.sourceBundleHash, result.sourceBundleHash);
  assert.ok(graph.nodes.some((node) => node.id === "source-bundle"));
  assert.ok(graph.nodes.some((node) => node.kind === "file" && node.path === "docs/intro.md"));
  assert.ok(graph.edges.length >= 2);

  assert.equal(labels.contract, "graphify_labels.v1");
  assert.ok(labels.labels.includes("alpha"));
  assert.ok(labels.labels.includes("beta"));
  assert.ok(labels.labels.includes("wave1"));

  assert.equal(benchmark.contract, "graphify_benchmark.v1");
  assert.equal(benchmark.runId, result.runId);
  assert.ok(benchmark.totalMs >= 0);
  assert.equal(typeof report, "string");
  assert.ok(report.includes(`# Graphify run ${result.runId}`));
});
