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

test("managed graphify runner produces stable run-bundle outputs for the same source bundle and config", (t) => {
  const sourceBundle = createSourceBundle(t, "graphify-stable-bundle");
  const firstOutputRoot = createOutputRoot(t, "graphify-stable-bundle-first");
  const secondOutputRoot = createOutputRoot(t, "graphify-stable-bundle-second");

  const sharedOptions = {
    sourceBundlePath: sourceBundle,
    graphifyVersion: "graphify 9.9.9",
    graphifyMode: "compiler",
    graphifyConfig: {
      layout: "force",
      seed: 7,
    },
    graphifyFlags: ["--static", "--reproducible"],
    labels: ["wave1"],
  };

  const first = runManagedGraphifyRunner({
    ...sharedOptions,
    outputRoot: firstOutputRoot,
  });
  const second = runManagedGraphifyRunner({
    ...sharedOptions,
    outputRoot: secondOutputRoot,
  });

  assert.equal(first.ok, true);
  assert.equal(second.ok, true);
  assert.equal(first.runId, second.runId);
  assert.equal(first.sourceBundleHash, second.sourceBundleHash);
  assert.equal(readFileSync(first.outputs.summary, "utf8"), readFileSync(second.outputs.summary, "utf8"));
  assert.equal(readFileSync(first.outputs.graph, "utf8"), readFileSync(second.outputs.graph, "utf8"));
  assert.equal(readFileSync(first.outputs.html, "utf8"), readFileSync(second.outputs.html, "utf8"));
  assert.equal(readFileSync(first.outputs.report, "utf8"), readFileSync(second.outputs.report, "utf8"));
  assert.equal(readFileSync(first.outputs.labels, "utf8"), readFileSync(second.outputs.labels, "utf8"));
});
