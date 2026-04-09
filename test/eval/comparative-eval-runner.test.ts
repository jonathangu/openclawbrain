import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { checksumJsonPayload } from "@openclawbrain/contracts";
import { afterEach, describe, expect, it } from "vitest";
import {
  CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT,
  COMPARATIVE_EVAL_SCORECARD_CONTRACT,
  COMPARATIVE_EVAL_RUNNER_REPORT_CONTRACT,
  FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT,
  runComparativeEval,
  type CanonicalRecordedSessionTraceSetManifestV1,
  type FrozenRecordedSessionEvalManifestV1,
} from "../../src/eval/comparative-eval-runner.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");

const tempRoots: string[] = [];

afterEach(() => {
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

function createTempRoot(label: string): string {
  const root = mkdtempSync(path.join(os.tmpdir(), `${label}-`));
  tempRoots.push(root);
  return root;
}

function writeJson(filePath: string, payload: unknown): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function createWorkspace(rootDir: string, label: string, readmeText: string) {
  const workspaceRoot = path.join(rootDir, `${label}-workspace`);
  mkdirSync(workspaceRoot, { recursive: true });
  writeFileSync(path.join(workspaceRoot, "README.md"), readmeText, "utf8");
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-04-01T00:00:00.000Z",
    rootDir: workspaceRoot,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["test"],
  };
}

function buildComparativeTrace(rootDir: string, label: string) {
  return {
    contract: "recorded_session_trace.v1",
    traceId: `trace-comparative-${label}`,
    source: "sanitized_recorded_session",
    recordedAt: "2026-04-01T00:00:00.000Z",
    bundleBuiltAt: "2026-04-01T00:10:00.000Z",
    agentId: "agent",
    sessionId: `session-comparative-${label}`,
    channel: "cli",
    sourceStream: "recorded/session",
    privacy: {
      sanitized: true,
      notes: ["test fixture"],
    },
    workspace: createWorkspace(rootDir, label, "# Comparative workspace\nThe routing guide lives here.\n"),
    seedBuiltAt: "2026-04-01T00:01:00.000Z",
    seedActivatedAt: "2026-04-01T00:02:00.000Z",
    seedCues: [
      {
        cueId: `cue-routing-guide-${label}`,
        createdAt: "2026-04-01T00:00:30.000Z",
        content: "The routing guide lives here.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: `${label}-turn-1`,
        createdAt: "2026-04-01T00:03:00.000Z",
        deliveredAt: "2026-04-01T00:03:30.000Z",
        userMessage: "show the routing guide",
        runtimeHints: ["routing", "guide"],
        feedback: [
          {
            createdAt: "2026-04-01T00:03:45.000Z",
            content: "Keep the routing guide easy to find.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["routing guide"],
      },
      {
        turnId: `${label}-turn-2`,
        createdAt: "2026-04-01T00:05:00.000Z",
        deliveredAt: "2026-04-01T00:05:30.000Z",
        userMessage: "show the routing guide again",
        runtimeHints: ["routing", "guide", "again"],
        expectedContextPhrases: ["routing guide"],
      },
    ],
  };
}

function buildTrainFreezeTrace(rootDir: string, label: string) {
  return {
    contract: "recorded_session_trace.v1",
    traceId: `trace-train-freeze-${label}`,
    source: "sanitized_recorded_session",
    recordedAt: "2026-04-01T10:00:00.000Z",
    bundleBuiltAt: "2026-04-01T10:30:00.000Z",
    sessionId: `session-train-freeze-${label}`,
    channel: "chat",
    sourceStream: "recorded/session",
    privacy: { sanitized: true, notes: ["test"] },
    workspace: createWorkspace(rootDir, label, "# Train freeze workspace\nAlways read README before editing code.\n"),
    seedBuiltAt: "2026-04-01T09:56:00.000Z",
    seedActivatedAt: "2026-04-01T09:57:00.000Z",
    seedCues: [
      {
        cueId: `cue-readme-${label}`,
        createdAt: "2026-04-01T09:50:00.000Z",
        content: "Always read README before editing code.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: `${label}-turn-1`,
        createdAt: "2026-04-01T10:01:00.000Z",
        deliveredAt: "2026-04-01T10:01:30.000Z",
        userMessage: "What should I read before editing?",
        feedback: [
          {
            createdAt: "2026-04-01T10:01:45.000Z",
            content: "Correct: read README before editing code.",
            kind: "approval",
          },
        ],
        expectedContextPhrases: ["readme before editing"],
      },
      {
        turnId: `${label}-turn-2`,
        createdAt: "2026-04-01T10:05:00.000Z",
        deliveredAt: "2026-04-01T10:05:30.000Z",
        userMessage: "Before changing files, what is the rule?",
        expectedContextPhrases: ["readme before editing"],
      },
    ],
  };
}

function buildUnsanitizedTrace(rootDir: string, label: string) {
  const trace = buildComparativeTrace(rootDir, label);
  return {
    ...trace,
    privacy: {
      ...trace.privacy,
      sanitized: false,
    },
  };
}

function writeTrace(rootDir: string, trace: Record<string, unknown>) {
  const tracePath = path.join(rootDir, `${trace.traceId as string}.json`);
  writeJson(tracePath, trace);
  return {
    traceId: trace.traceId as string,
    tracePath,
    traceHash: checksumJsonPayload(trace),
  };
}

function writeCanonicalManifest(
  rootDir: string,
  manifestId: string,
  traces: Array<{ tracePath: string }>,
): string {
  const manifestPath = path.join(rootDir, `${manifestId}.canonical-manifest.json`);
  const manifest: CanonicalRecordedSessionTraceSetManifestV1 = {
    contract: CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT,
    setId: manifestId,
    traceCount: traces.length,
    realTraceCoverage: {
      summary: "equivalent-only recorded replay corpus",
    },
    redactionPolicy: {
      summary: "sanitized traces only",
    },
    entries: traces.map((trace, index) => ({
      slotId: `slot-${index + 1}`,
      path: path.relative(rootDir, trace.tracePath),
    })),
  };
  writeJson(manifestPath, manifest);
  return manifestPath;
}

function writeFrozenManifest(
  rootDir: string,
  manifestId: string,
  traces: Array<{ traceId: string; tracePath: string; traceHash: string }>,
): string {
  const manifestPath = path.join(rootDir, `${manifestId}.frozen-manifest.json`);
  const manifest: FrozenRecordedSessionEvalManifestV1 = {
    contract: FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT,
    manifestId,
    expectedTraceCount: traces.length,
    traces: traces.map((trace) => ({
      traceId: trace.traceId,
      tracePath: path.relative(rootDir, trace.tracePath),
      traceHash: trace.traceHash,
    })),
  };
  writeJson(manifestPath, manifest);
  return manifestPath;
}

describe("comparative eval runner", () => {
  it("runs the canonical manifest contract and writes a deterministic scorecard with a passing policy verdict", () => {
    const rootDir = createTempRoot("comparative-eval-runner");
    const outputDir = path.join(rootDir, "output");
    const comparativeTrace = writeTrace(rootDir, buildComparativeTrace(rootDir, "canon-a"));
    const trainFreezeTrace = writeTrace(rootDir, buildTrainFreezeTrace(rootDir, "canon-b"));
    const manifestPath = writeCanonicalManifest(rootDir, "canonical-eval", [comparativeTrace, trainFreezeTrace]);

    const descriptor = runComparativeEval({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
      workedTraceLimit: 1,
      policy: {
        maxCandidateTiePromotionDeltaVsBaseline: 2,
      },
    });

    expect(descriptor.report.contract).toBe(COMPARATIVE_EVAL_RUNNER_REPORT_CONTRACT);
    expect(descriptor.report.status).toBe("ok");
    expect(descriptor.scorecard.contract).toBe(COMPARATIVE_EVAL_SCORECARD_CONTRACT);
    expect(descriptor.scorecard.requestedTraceCount).toBe(2);
    expect(descriptor.scorecard.successfulTraceCount).toBe(2);
    expect(descriptor.scorecard.failedTraceCount).toBe(0);
    expect(descriptor.report.gateStatus).toBe("pass");
    expect(descriptor.report.gateDecisive).toBe(true);
    expect(descriptor.scorecard.modeOrder).toEqual([
      "no_brain",
      "vector_only",
      "graph_prior_only",
      "learned_route",
    ]);
    expect(descriptor.scorecard.scorecardHash).toMatch(/^sha256-/);
    expect(descriptor.report.notes.some((note) => note.includes("truth boundary"))).toBe(true);
    expect(descriptor.scorecard.policy.status).toBe("pass");
    expect(descriptor.scorecard.policy.checks.find((check) => check.id === "candidate_trace_tie_or_better_vs_baseline")?.status).toBe("pass");
    expect(descriptor.scorecard.policy.checks.find((check) => check.id === "candidate_tie_promotion_delta_vs_baseline")?.status).toBe("pass");
    expect(
      descriptor.scorecard.pairwise.find(
        (row) => row.leftMode === "graph_prior_only" && row.rightMode === "learned_route",
      )?.traceTieOrBetter.rightRate,
    ).toBe(1);

    const learnedRouteRow = descriptor.scorecard.modes.find((row) => row.mode === "learned_route");
    expect(learnedRouteRow?.estimatedPromptTokens).toBeGreaterThan(0);
    expect(learnedRouteRow?.estimatedPromptCostUsd).not.toBeNull();

    expect(existsSync(descriptor.sourceManifestPath!)).toBe(true);
    expect(existsSync(descriptor.reportPath)).toBe(true);
    expect(existsSync(descriptor.scorecardPath)).toBe(true);
    expect(existsSync(descriptor.summaryPath)).toBe(true);
    expect(existsSync(path.join(outputDir, "traces", "_lane", "index.json"))).toBe(true);
    expect(existsSync(path.join(outputDir, "traces", "_lane", "worked-traces.md"))).toBe(true);
  });

  it("marks the gate as fail when the explicit policy thresholds are not met", () => {
    const rootDir = createTempRoot("comparative-eval-runner-fail");
    const outputDir = path.join(rootDir, "output");
    const comparativeTrace = writeTrace(rootDir, buildComparativeTrace(rootDir, "fail-a"));
    const trainFreezeTrace = writeTrace(rootDir, buildTrainFreezeTrace(rootDir, "fail-b"));
    const manifestPath = writeCanonicalManifest(rootDir, "fail-eval", [comparativeTrace, trainFreezeTrace]);

    const descriptor = runComparativeEval({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
      policy: {
        minBaselineMeanQualityGainVsFloor: 101,
      },
    });

    expect(descriptor.report.status).toBe("ok");
    expect(descriptor.report.gateStatus).toBe("fail");
    expect(descriptor.report.gateDecisive).toBe(true);
    expect(descriptor.report.gateFailedCheckIds).toContain("baseline_mean_quality_gain_vs_floor");
    expect(descriptor.scorecard.policy.status).toBe("fail");
    expect(descriptor.scorecard.policy.reasons).toContain(
      "baseline_mean_quality_gain_vs_floor: baseline does not clear the floor anchor by the configured mean quality margin",
    );
  });

  it("fails the gate when learned routing adds promotion churn on tie traces", () => {
    const rootDir = createTempRoot("comparative-eval-runner-tie-promotion");
    const outputDir = path.join(rootDir, "output");
    const comparativeTrace = writeTrace(rootDir, buildComparativeTrace(rootDir, "tie-a"));
    const trainFreezeTrace = writeTrace(rootDir, buildTrainFreezeTrace(rootDir, "tie-b"));
    const manifestPath = writeCanonicalManifest(rootDir, "tie-promotion-eval", [comparativeTrace, trainFreezeTrace]);

    const descriptor = runComparativeEval({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
      policy: {
        maxCandidateTiePromotionDeltaVsBaseline: -1_000_000,
      },
    });

    expect(descriptor.report.status).toBe("ok");
    expect(descriptor.report.gateStatus).toBe("fail");
    expect(descriptor.report.gateDecisive).toBe(true);
    expect(descriptor.report.gateFailedCheckIds).toContain("candidate_tie_promotion_delta_vs_baseline");
    expect(descriptor.scorecard.policy.status).toBe("fail");
    expect(descriptor.scorecard.policy.reasons.some((reason) => reason.includes("candidate_tie_promotion_delta_vs_baseline"))).toBe(true);
  });

  it("marks the gate as partial when only a subset of traces validate", () => {
    const rootDir = createTempRoot("comparative-eval-runner-partial");
    const outputDir = path.join(rootDir, "output");
    const validTrace = writeTrace(rootDir, buildComparativeTrace(rootDir, "partial-a"));
    const invalidTrace = writeTrace(rootDir, buildUnsanitizedTrace(rootDir, "partial-b"));
    const manifestPath = writeFrozenManifest(rootDir, "partial-eval", [validTrace, invalidTrace]);

    const descriptor = runComparativeEval({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
    });

    expect(descriptor.report.status).toBe("partial");
    expect(descriptor.report.gateStatus).toBe("partial");
    expect(descriptor.report.gateDecisive).toBe(false);
    expect(descriptor.scorecard.successfulTraceCount).toBe(1);
    expect(descriptor.scorecard.failedTraceCount).toBe(1);
    expect(descriptor.scorecard.policy.status).toBe("partial");
    expect(descriptor.scorecard.policy.checks.find((check) => check.id === "trace_coverage_complete")?.status).toBe("fail");
    expect(descriptor.scorecard.policy.reasons[0]).toContain("only 1/2 traces validated");
    expect(descriptor.report.issues.some((issue) => issue.includes("recorded session trace must be explicitly sanitized"))).toBe(true);
  });

  it("accepts the frozen manifest contract and blocks on an invalid manifest contract", () => {
    const validRoot = createTempRoot("comparative-eval-runner-frozen");
    const validTrace = writeTrace(validRoot, buildComparativeTrace(validRoot, "frozen-a"));
    const validManifestPath = writeFrozenManifest(validRoot, "frozen-eval", [validTrace]);
    const validDescriptor = runComparativeEval({
      manifestPath: validManifestPath,
      outputDir: path.join(validRoot, "output"),
      scratchRootDir: validRoot,
      policy: {
        maxCandidateTiePromotionDeltaVsBaseline: 2,
      },
    });

    expect(validDescriptor.report.status).toBe("ok");
    expect(validDescriptor.report.gateStatus).toBe("pass");
    expect(validDescriptor.report.manifestContract).toBe(FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT);
    expect(validDescriptor.scorecard.successfulTraceCount).toBe(1);

    const invalidRoot = createTempRoot("comparative-eval-runner-invalid");
    const invalidManifestPath = path.join(invalidRoot, "invalid-manifest.json");
    writeJson(invalidManifestPath, {
      contract: "unknown_manifest_contract.v1",
      traces: [],
    });

    const blockedDescriptor = runComparativeEval({
      manifestPath: invalidManifestPath,
      outputDir: path.join(invalidRoot, "output"),
    });

    expect(blockedDescriptor.report.status).toBe("blocked");
    expect(blockedDescriptor.report.gateStatus).toBe("blocked");
    expect(blockedDescriptor.scorecard.policy.status).toBe("blocked");
    expect(blockedDescriptor.report.files.laneDir).toBeNull();
    expect(blockedDescriptor.report.issues[0]).toContain("manifest contract must be");
  });

  it("exposes a CLI smoke path", () => {
    const rootDir = createTempRoot("comparative-eval-runner-cli");
    const trace = writeTrace(rootDir, buildComparativeTrace(rootDir, "cli-a"));
    const manifestPath = writeFrozenManifest(rootDir, "cli-eval", [trace]);
    const outputDir = path.join(rootDir, "output");

    const stdout = execFileSync(
      "node",
      [
        "--experimental-transform-types",
        "scripts/eval/run-comparative-eval.ts",
        "--manifest",
        manifestPath,
        "--output-dir",
        outputDir,
        "--scratch-root-dir",
        rootDir,
        "--max-candidate-tie-promotion-delta",
        "2",
      ],
      {
        cwd: repoRoot,
        encoding: "utf8",
      },
    );

    expect(stdout).toContain("Comparative eval runner: ok");
    expect(stdout).toContain("Comparative eval gate: pass");
    expect(stdout).toContain("candidate_trace_tie_or_better_vs_baseline: pass");
    expect(stdout).toContain("candidate_tie_promotion_delta_vs_baseline: pass");
    expect(stdout).toContain(`outputDir: ${outputDir}`);
    expect(existsSync(path.join(outputDir, "report.json"))).toBe(true);

    const scorecard = JSON.parse(readFileSync(path.join(outputDir, "scorecard.json"), "utf8")) as {
      contract: string;
      successfulTraceCount: number;
    };
    expect(scorecard.contract).toBe(COMPARATIVE_EVAL_SCORECARD_CONTRACT);
    expect(scorecard.successfulTraceCount).toBe(1);
  });
});
