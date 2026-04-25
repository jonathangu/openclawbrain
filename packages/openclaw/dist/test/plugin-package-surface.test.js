import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageJson = JSON.parse(readFileSync(path.join(__dirname, "..", "..", "package.json"), "utf8"));
const indexSource = readFileSync(path.join(__dirname, "..", "src", "index.js"), "utf8");

test("plugin package only exposes runtime surface", () => {
  assert.equal(packageJson.name, "@openclawbrain/openclaw");
  assert.equal("bin" in packageJson, false);
  assert.match(indexSource, /compileRuntimeContext/);
  assert.match(indexSource, /recordOpenClawProfileRuntimeLoadProof/);
  assert.doesNotMatch(indexSource, /bootstrapRuntimeAttach/);
  assert.doesNotMatch(indexSource, /describeCurrentProfileBrainStatus/);
  assert.doesNotMatch(indexSource, /runDaemonCommand/);
});

const indexDtsSource = readFileSync(path.join(__dirname, "..", "src", "index.d.ts"), "utf8");

test("type surface exports bounded serving interruption truth", () => {
  assert.match(indexDtsSource, /export interface InterruptionAccounting/);
  assert.match(indexDtsSource, /droppedFrontierNodeIds/);
  assert.match(indexDtsSource, /droppedProposalNodeIds/);
  assert.match(indexDtsSource, /budgetUtilization/);
  assert.match(indexDtsSource, /droppedProposalReasons/);
});

test("type surface exports context feedback truth", () => {
  assert.match(indexDtsSource, /export type ContextFeedbackVerdict/);
  assert.match(indexDtsSource, /export interface ContextFeedbackSummary/);
  assert.match(indexDtsSource, /export interface ContextFeedbackCoverageSummary/);
  assert.match(indexDtsSource, /export interface ContextFeedbackAgentCoverageSummary/);
  assert.match(indexDtsSource, /verdictCounts/);
  assert.match(indexDtsSource, /supervisionCoverage/);
});
